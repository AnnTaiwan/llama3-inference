# Raw Block API 完整参考文档

本文档整合了在 raw block 设备上实现的所有权重流式传输 API，按功能分为四大类。

---

## 1. Staging Buffer APIs (暂存缓冲区)

### 1.1 `alloc_pinned_aligned(nbytes, block_size=4096)`
**位置**: [weights_io_ssd_dram.py:175-188](llama3/weights_io_ssd_dram.py#L175-L188)

**功能**: 分配一个 pinned + block_size 对齐的 uint8 张量

**参数**:
- `nbytes` (int): 要分配的字节数，必须是 `block_size` 的倍数
- `block_size` (int, 默认4096): 对齐边界，通常为 512B 或 4096B

**返回**: `torch.Tensor` (dtype=uint8, pinned, 地址对齐)

**使用场景**:
- 作为 Direct I/O 读写的中转缓冲区
- 确保 DMA 传输的三对齐要求 (offset/nbytes/buffer_ptr)

**示例**:
```python
# 分配 64MB 的对齐 staging buffer
staging = alloc_pinned_aligned(64 * 1024 * 1024, block_size=4096)
assert staging.data_ptr() % 4096 == 0  # 地址对齐检查
```

**技术细节**:
- PyTorch pinned 内存通常页对齐 (4K)，但不保证 block_size 对齐
- 通过多次尝试分配 (最多4次) 来找到对齐的地址
- 如果失败，建议使用 C helper 配合 `posix_memalign`

---

## 2. Block-Level Read/Write APIs (块级读写)

### 2.1 `DirectIOFile` 类
**位置**: [weights_io_ssd_dram.py:37-173](llama3/weights_io_ssd_dram.py#L37-L173)

以 `O_DIRECT` 方式打开块设备/文件，提供零拷贝、绕过 page cache 的高性能 I/O。

#### 2.1.1 `__init__(path, mode='r', block_size=None)`
**功能**: 初始化 DirectIO 文件句柄

**参数**:
- `path` (str): 设备路径，如 `/dev/nvme0n1p3`
- `mode` (str): `'r'` (只读) / `'w'` (只写) / `'rw'` (读写)
- `block_size` (int, 可选): 手动指定块大小，默认自动检测

**行为**:
- 使用 `O_DIRECT | O_LARGEFILE` 标志打开设备
- 通过 `ioctl(BLKSSZGET)` 自动检测逻辑块大小
- 默认回退到 4096 字节

**示例**:
```python
dio = DirectIOFile("/dev/nvme0n1p3", mode="r", block_size=4096)
print(f"Block size: {dio.block_size}")  # 4096
```

---

#### 2.1.2 `pread_into_tensor(t, nbytes, offset)`
**位置**: [weights_io_ssd_dram.py:87-119](llama3/weights_io_ssd_dram.py#L87-L119)

**功能**: 从块设备读取数据，直接填充到 PyTorch pinned tensor

**参数**:
- `t` (torch.Tensor): 目标缓冲区 (必须 pinned, CPU, contiguous, uint8)
- `nbytes` (int): 读取字节数 (必须是 `block_size` 倍数)
- `offset` (int): 设备偏移量 (必须是 `block_size` 倍数)

**返回**: `int` - 实际读取的字节数

**约束条件** (三对齐):
1. `offset % block_size == 0`
2. `nbytes % block_size == 0`
3. `t.data_ptr() % block_size == 0`

**示例**:
```python
# 从 offset=4MB 处读取 128MB 权重
offset = 4 * 1024 * 1024
nbytes = 128 * 1024 * 1024
nbytes_aligned = ((nbytes + 4095) // 4096) * 4096

staging = alloc_pinned_aligned(nbytes_aligned, 4096)
dio.pread_into_tensor(staging, nbytes_aligned, offset)
```

**错误处理**:
- `ValueError`: 如果违反任何对齐要求或 tensor 属性检查
- `OSError`: 如果底层 `pread()` 系统调用失败

---

#### 2.1.3 `pwrite_from_tensor(t, nbytes, offset)`
**位置**: [weights_io_ssd_dram.py:142-162](llama3/weights_io_ssd_dram.py#L142-L162)

**功能**: 从 PyTorch pinned tensor 写入数据到块设备

**参数**: 同 `pread_into_tensor`

**返回**: `int` - 实际写入的字节数

**使用场景**:
- 打包 checkpoint 到 raw 设备
- 权重预写入 (warmup)

**示例**:
```python
# 将权重写入 raw 设备
weight_bytes = weight.view(torch.uint8).cpu().numpy().tobytes()
nbytes_aligned = ((len(weight_bytes) + 4095) // 4096) * 4096
buf = alloc_pinned_aligned(nbytes_aligned, 4096)
buf[:len(weight_bytes)] = torch.frombuffer(weight_bytes, dtype=torch.uint8)

dio.pwrite_from_tensor(buf, nbytes_aligned, offset)
```

---

#### 2.1.4 `fdatasync()`
**位置**: [weights_io_ssd_dram.py:121-124](llama3/weights_io_ssd_dram.py#L121-L124)

**功能**: 确保已写数据持久化到物理设备

**使用场景**:
- 打包完成后强制刷盘
- 关键数据写入后确保不丢失

**示例**:
```python
dio.pwrite_from_tensor(buf, nbytes, offset)
dio.fdatasync()  # 等待数据落盘
```

---

#### 2.1.5 `fadvise_dontneed(offset, length)`
**位置**: [weights_io_ssd_dram.py:126-140](llama3/weights_io_ssd_dram.py#L126-L140)

**功能**: 提示内核丢弃页缓存 (非强制)

**参数**:
- `offset` (int): 起始偏移
- `length` (int): 长度

**使用场景**:
- warmup 阶段用 buffered I/O，推理前清除缓存
- 释放不再需要的缓存页

**示例**:
```python
# Warmup 后清理缓存
dio.fadvise_dontneed(offset=0, length=total_size)
```

---

#### 2.1.6 `close()`
**功能**: 关闭文件描述符

**示例**:
```python
dio.close()
```

---

### 2.2 辅助函数

#### 2.2.1 `get_logical_block_size(fd)`
**位置**: [weights_io_ssd_dram.py:31-35](llama3/weights_io_ssd_dram.py#L31-L35)

**功能**: 通过 `ioctl(BLKSSZGET)` 获取设备的逻辑块大小

**参数**: `fd` (int) - 文件描述符

**返回**: `int` - 块大小 (字节)

**示例**:
```python
fd = os.open("/dev/nvme0n1p3", os.O_RDONLY)
bsz = get_logical_block_size(fd)  # 4096
os.close(fd)
```

---

#### 2.2.2 `round_up(x, a)`
**位置**: [weights_io_ssd_dram.py:25](llama3/weights_io_ssd_dram.py#L25)

**功能**: 将 `x` 向上对齐到 `a` 的倍数

**示例**:
```python
stride = round_up(nbytes=134217728, a=4096)  # 134221824
```

---

## 3. Layer Table APIs (分层索引表)

### 3.1 权重分类规则

#### 3.1.1 `classify_policy(name)`
**位置**: [weights_io_ssd_dram.py:205-212](llama3/weights_io_ssd_dram.py#L205-L212)

**功能**: 根据参数名称判断是常驻 (resident) 还是流式 (stream)

**参数**: `name` (str) - 参数全名，如 `"layers.0.attention.wq.weight"`

**返回**: `str` - `"resident"` 或 `"stream"`

**分类规则**:
```python
# Resident (常驻 GPU):
- embed_tokens.*
- norm.*
- output.*
- layers.*.attention_norm.* / ffn_norm.*
- *.bias

# Stream (按需流式):
- layers.*.attention.w[qkvo].weight  # Q/K/V/O
- layers.*.feed_forward.w[123].weight  # FFN W1/W2/W3
- layers.*.feed_forward.(gate|up|down).*weight
```

**示例**:
```python
classify_policy("embed_tokens.weight")  # "resident"
classify_policy("layers.0.attention.wq.weight")  # "stream"
```

---

### 3.2 Layer 元数据构建

#### 3.2.1 `build_runtime_manifest(shapes_meta_path, manifest_out_path)`
**位置**: [weights_io_ssd_dram.py:337-387](llama3/weights_io_ssd_dram.py#L337-L387)

**功能**: 每次启动时从 `shapes_meta.json` 生成运行时 manifest (含 offset/stride)

**参数**:
- `shapes_meta_path` (str): 打包时生成的形状元数据路径
- `manifest_out_path` (str): 输出的运行时 manifest 路径

**返回**: `str` - manifest 路径

**生成内容**:
```json
{
  "version": 1,
  "raw_device": "/dev/nvme0n1p3",
  "block_size": 4096,
  "header_reserve": 4194304,
  "params": [
    {
      "name": "layers.0.attention.wq.weight",
      "layer": 0,
      "dtype": "bfloat16",
      "shape": [8192, 8192],
      "offset": 4194304,
      "nbytes": 134217728,
      "stride": 134221824,
      "policy": "stream"
    }
  ]
}
```

**关键逻辑**:
1. 读取 `shapes_meta.json`
2. 打开 raw 设备查询当前 `block_size`
3. **严格按原始顺序** (不排序) 线性推导 offset/stride
4. 从 `header_reserve` 开始累计偏移

**示例**:
```python
manifest_path = build_runtime_manifest(
    "model.shapes_meta.json",
    "/dev/shm/runtime_manifest.json"
)
```

---

#### 3.2.2 `streamable_entries_for_layer(manifest, layer_id)`
**位置**: [weights_io_ssd_dram.py:543-545](llama3/weights_io_ssd_dram.py#L543-L545)

**功能**: 查询某层的所有流式权重条目

**参数**:
- `manifest` (dict): 运行时 manifest
- `layer_id` (int): 层编号

**返回**: `List[Dict]` - 该层的流式参数列表 (Q/K/V/O, W1/W2/W3)

**示例**:
```python
manifest = json.loads(Path("/dev/shm/runtime_manifest.json").read_text())
layer_0_params = streamable_entries_for_layer(manifest, layer_id=0)

for p in layer_0_params:
    print(f"{p['name']}: offset={p['offset']}, nbytes={p['nbytes']}")
```

**输出**:
```
layers.0.attention.wq.weight: offset=4194304, nbytes=134217728
layers.0.attention.wk.weight: offset=138412032, nbytes=134217728
...
```

---

### 3.3 常驻权重加载

#### 3.3.1 `load_resident_to_gpu(model, manifest, device='cuda:0', staging_bytes=16*1024*1024)`
**位置**: [weights_io_ssd_dram.py:400-540](llama3/weights_io_ssd_dram.py#L400-L540)

**功能**: 启动时一次性加载所有常驻权重到 GPU

**参数**:
- `model` (nn.Module): PyTorch 模型
- `manifest` (dict): 运行时 manifest
- `device` (str): 目标设备，如 `"cuda:0"`
- `staging_bytes` (int): staging buffer 大小

**行为**:
1. 遍历 manifest 中所有 `policy="resident"` 的参数
2. 从 raw 设备读取到 staging buffer
3. 反序列化为正确的 dtype 和 shape
4. 复制到模型参数的 GPU 内存

**形状验证**:
- 自动检测 vocab-parallel 分片权重
- 提供详细错误提示 (embed/output 层的常见问题)

**名称映射**:
```python
# 自动处理别名
"tok_embeddings.weight" -> "embed_tokens.weight"
```

**示例**:
```python
manifest = json.loads(Path("/dev/shm/runtime_manifest.json").read_text())
load_resident_to_gpu(model, manifest, device="cuda:0")
# [RESIDENT] all resident params loaded to GPU
```

---

## 4. On-Disk Layout APIs (磁盘布局)

### 4.1 打包 (一次性)

#### 4.1.1 `pack_any_to_raw(ckpt_path_or_dir, raw_dev, shapes_meta_out=None, header_reserve_bytes=4*1024*1024)`
**位置**: [weights_io_ssd_dram.py:254-332](llama3/weights_io_ssd_dram.py#L254-L332)

**功能**: 将 PyTorch checkpoint 打包到 raw 块设备

**参数**:
- `ckpt_path_or_dir` (str): checkpoint 路径 (`.pth` 文件或包含 `consolidated*.pth` 的目录)
- `raw_dev` (str): raw 设备路径，如 `/dev/nvme0n1p3`
- `shapes_meta_out` (str, 可选): 输出元数据路径，默认自动生成
- `header_reserve_bytes` (int, 默认4MB): 预留头部大小 (必须块对齐)

**返回**: `str` - `shapes_meta.json` 路径

**磁盘布局**:
```
┌────────────────────────────────────┐
│ Header Reserve (4MB, 预留)          │ <- offset=0
├────────────────────────────────────┤
│ Param 0: embed_tokens.weight       │ <- offset=4194304
│   - nbytes 实际数据                  │
│   - padding (补零到块边界)           │
├────────────────────────────────────┤
│ Param 1: layers.0.attention.wq     │
│   - nbytes 实际数据                  │
│   - padding                        │
├────────────────────────────────────┤
│ ...                                │
└────────────────────────────────────┘
```

**生成的 `shapes_meta.json`**:
```json
{
  "version": 1,
  "raw_device": "/dev/nvme0n1p3",
  "header_reserve": 4194304,
  "params": [
    {
      "name": "embed_tokens.weight",
      "layer": -1,
      "dtype": "bfloat16",
      "shape": [128256, 8192],
      "nbytes": 2101215232,
      "policy": "resident"
    }
  ]
}
```

**注意**:
- **不包含 offset/stride** (运行时推导)
- 参数按名称排序写入 (保证顺序稳定)
- 自动分类 `policy` (resident/stream)

**示例**:
```python
# 从单一 .pth 文件打包
shapes_meta = pack_any_to_raw(
    "model.pth",
    "/dev/nvme0n1p3",
    shapes_meta_out="model.shapes_meta.json"
)

# 从 consolidated.*.pth 目录打包
shapes_meta = pack_any_to_raw(
    "/data/llama3-70b/",
    "/dev/nvme0n1p3"
)
```

---

### 4.2 辅助函数

#### 4.2.1 `_layer_idx_of(name)`
**位置**: [weights_io_ssd_dram.py:217-223](llama3/weights_io_ssd_dram.py#L217-L223)

**功能**: 从参数名提取层编号

**示例**:
```python
_layer_idx_of("layers.42.attention.wq.weight")  # 42
_layer_idx_of("embed_tokens.weight")           # -1
```

---

#### 4.2.2 `_iter_tensors_from_pth(ckpt_path)` / `_iter_tensors_from_dir(ckpt_dir)`
**位置**: [weights_io_ssd_dram.py:233-252](llama3/weights_io_ssd_dram.py#L233-L252)

**功能**: 从 checkpoint 迭代所有 tensor

**返回**: `Iterable[Tuple[str, torch.Tensor]]` - (参数名, tensor)

**兼容性**:
- 自动处理 `{'state_dict': ...}` 或 `{'model': ...}` 包装
- 支持多分片 `consolidated.*.pth` 文件

---

### 4.3 性能测试

#### 4.3.1 `bench_raw_read(manifest_path, rounds=8, chunk_bytes=64*1024*1024)`
**位置**: [weights_io_ssd_dram.py:550-575](llama3/weights_io_ssd_dram.py#L550-L575)

**功能**: 测试 raw 设备读取吞吐量

**参数**:
- `manifest_path` (str): 运行时 manifest 路径
- `rounds` (int): 测试轮数
- `chunk_bytes` (int): 每次读取块大小

**返回**: `float` - 吞吐量 (MB/s)

**示例**:
```python
throughput = bench_raw_read(
    "/dev/shm/runtime_manifest.json",
    rounds=10,
    chunk_bytes=128*1024*1024
)
# [BENCH] raw_read_MBps=3420.5 (chunk=128 MiB, rounds=10)
```

---

## 5. CLI 工具

### 5.1 打包命令
```bash
python -m llama3.weights_io_ssd_dram pack \
    /data/llama3-70b/ \
    /dev/nvme0n1p3 \
    --meta-out model.shapes_meta.json \
    --header-reserve 4194304
```

### 5.2 生成 manifest
```bash
python -m llama3.weights_io_ssd_dram manifest \
    model.shapes_meta.json \
    --out /dev/shm/runtime_manifest.json
```

### 5.3 性能测试
```bash
python -m llama3.weights_io_ssd_dram bench-read \
    /dev/shm/runtime_manifest.json \
    --rounds 10 \
    --chunk 134217728
```

---

## 6. 完整工作流程

### 6.1 初次设置 (一次性)
```python
# 1. 打包 checkpoint 到 raw 设备
from llama3.weights_io_ssd_dram import pack_any_to_raw

shapes_meta = pack_any_to_raw(
    ckpt_path_or_dir="/data/llama3-70b/",
    raw_dev="/dev/nvme0n1p3",
    shapes_meta_out="llama3_70b.shapes_meta.json",
    header_reserve_bytes=4*1024*1024
)
```

### 6.2 每次启动
```python
# 2. 构建运行时 manifest
from llama3.weights_io_ssd_dram import build_runtime_manifest
import json

manifest_path = build_runtime_manifest(
    "llama3_70b.shapes_meta.json",
    "/dev/shm/runtime_manifest.json"
)

# 3. 加载常驻权重
from llama3.weights_io_ssd_dram import load_resident_to_gpu

manifest = json.loads(Path(manifest_path).read_text())
load_resident_to_gpu(model, manifest, device="cuda:0")
```

### 6.3 推理时流式加载
```python
# 4. 按需加载层权重
from llama3.weights_io_ssd_dram import (
    DirectIOFile, streamable_entries_for_layer,
    alloc_pinned_aligned, DTYPE_MAP
)

dio = DirectIOFile(manifest["raw_device"], mode="r", block_size=manifest["block_size"])
staging = alloc_pinned_aligned(256*1024*1024, manifest["block_size"])

for layer_id in range(num_layers):
    entries = streamable_entries_for_layer(manifest, layer_id)

    for p in entries:
        # SSD -> CPU (pinned)
        dio.pread_into_tensor(staging, p["stride"], p["offset"])

        # 反序列化
        weight = torch.empty(p["shape"], dtype=DTYPE_MAP[p["dtype"]], pin_memory=True)
        weight.view(-1).view(torch.uint8)[:p["nbytes"]].copy_(staging[:p["nbytes"]])

        # CPU -> GPU (异步)
        weight_gpu = weight.to("cuda:0", non_blocking=True)

        # 使用 weight_gpu 进行前向计算...

dio.close()
```

---

## 7. 关键设计原则

### 7.1 三对齐约束
所有 Direct I/O 操作必须满足:
1. **offset 对齐**: `offset % block_size == 0`
2. **长度对齐**: `nbytes % block_size == 0`
3. **缓冲区对齐**: `buffer_ptr % block_size == 0`

### 7.2 顺序保证
- **打包阶段**: 按参数名排序写入
- **manifest 构建**: 严格按 `shapes_meta.json` 顺序推导 offset
- **不允许**: 任何重排序或插入操作

### 7.3 零拷贝路径
```
SSD (O_DIRECT) -> Pinned CPU Memory -> GPU (async)
     ^                  ^                 ^
     |                  |                 |
  pread_into      no page cache     cudaMemcpyAsync
```

### 7.4 策略分类
- **Resident**: 小权重 (embed, norm, bias) 启动时加载到 GPU
- **Stream**: 大权重 (Q/K/V/O, FFN) 推理时按层流式加载

---

## 8. 数据类型映射

```python
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}
```

---

## 9. 系统常量

```python
# Linux 系统调用标志
O_DIRECT = 0o40000      # 绕过 page cache
O_LARGEFILE = 0         # 支持大文件

# ioctl 命令
BLKSSZGET = 0x1268      # 获取逻辑块大小

# 默认配置
DEFAULT_BLOCK_SIZE = 4096           # 4KB
DEFAULT_HEADER_RESERVE = 4194304    # 4MB
DEFAULT_STAGING_BYTES = 16777216    # 16MB
```

---

## 10. 错误处理

### 10.1 常见错误

**ValueError: offset not block-aligned**
```python
# 错误: offset 未对齐
dio.pread_into_tensor(buf, 4096, offset=1000)  # ❌

# 正确: offset 对齐到 4096
dio.pread_into_tensor(buf, 4096, offset=4096)  # ✅
```

**ValueError: Tensor is not pinned**
```python
# 错误: 普通 CPU tensor
buf = torch.empty(4096, dtype=torch.uint8)  # ❌

# 正确: pinned tensor
buf = torch.empty(4096, dtype=torch.uint8, pin_memory=True)  # ✅
```

**RuntimeError: 形状不匹配**
- 检查 `params.json` 中的 `vocab_size` 和 `dim`
- 确认是否使用了 vocab-parallel 分片权重
- 参考 [weights_io_ssd_dram.py:458-527](llama3/weights_io_ssd_dram.py#L458-L527) 的详细诊断

---

## 11. 性能优化建议

### 11.1 Staging Buffer 大小
- **太小**: 频繁扩容，增加延迟
- **太大**: 浪费 pinned 内存资源
- **推荐**: 单层最大权重 × 1.5 (如 256MB)

### 11.2 预取策略
- 在计算 layer N 时，异步预取 layer N+1
- 使用专用 CUDA stream 避免阻塞计算

### 11.3 内存对齐
- 使用 4096 字节对齐 (大多数 NVMe SSD)
- 某些企业级 SSD 可能需要 8192 字节

### 11.4 并发加载
- 常驻权重可使用线程池并发加载
- 注意避免 pinned 内存过度分配

---

## 12. 调试工具

### 12.1 检查块设备属性
```bash
sudo blockdev --getbsz /dev/nvme0n1p3   # 逻辑块大小
sudo blockdev --getpbsz /dev/nvme0n1p3  # 物理块大小
```

### 12.2 验证对齐
```python
import torch
buf = alloc_pinned_aligned(4096, 4096)
print(f"Aligned: {buf.data_ptr() % 4096 == 0}")  # True
```

### 12.3 吞吐量测试
```bash
# 测试 raw 读取吞吐
python -m llama3.weights_io_ssd_dram bench-read \
    /dev/shm/runtime_manifest.json --rounds 20
```

---

## 13. 参考资料

- **O_DIRECT**: `man 2 open` - Linux Direct I/O 文档
- **pread/pwrite**: `man 2 pread` - POSIX 系统调用
- **ioctl**: `man 2 ioctl_list` - 设备控制命令
- **PyTorch Pinned Memory**: [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html#memory-pinning)

---

**版本**: 1.0
**最后更新**: 2026-01-05
**作者**: Roger (llama3-inference project)
