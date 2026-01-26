# Layer Table 实现例子 - 来自 Llama3-70B 推理系统

本文档提供三个具体的Layer Table数据结构例子，对应论文中的三个核心概念。

---

## 数据结构定义

### FileSegment (文件段)

```python
@dataclass
class FileSegment:
    """
    基本映射单元：定义从 SSD 到 GPU tensor 的单次传输
    """
    file_offset: int       # Raw device 上的字节偏移量（源地址）
    nbytes: int           # 要读取和传输的字节数
    dtype_from: str       # 磁盘上的数据类型 (例如 "fp16", "bf16", "int8")
    dst_offset: int       # 目标 tensor 内的字节偏移量（目标地址）
    dtype_to: str         # GPU HBM 上参数的数据类型
```

**字段说明**:
- `file_offset`: SSD/raw block device上的物理字节偏移量，这是Raw Block Device Manager需要的地址
- `nbytes`: 本次传输的字节长度
- `dtype_from`: 磁盘上存储的dtype（可能是量化后的，如int8）
- `dst_offset`: GPU tensor内的字节偏移量（相对tensor起始的偏移）
- `dtype_to`: GPU上运行时的dtype（可能需要反量化）

---

### ParamCopyPlan (参数复制计划)

```python
@dataclass
class ParamCopyPlan:
    """
    单个参数的完整复制计划，包含一个或多个 FileSegments
    """
    layer_id: int                               # 层索引
    group: str                                  # "mha" / "ffn" / "other"
    name: str                                   # 完整参数名 (例如 "layers.5.attention.wq.weight")
    dtype_to: str                               # 目标 dtype
    total_bytes: int                            # 总字节数
    segments: List[FileSegment]                 # 文件段列表

    def coalesce(self) -> None:
        """合并相邻的、dtype相同且地址连续的 segments"""

    def iter_dst_slices(self, param_tensor: torch.Tensor) -> Iterable[Tuple[FileSegment, torch.Tensor]]:
        """
        迭代所有 segments，返回 (segment, dst_u8_slice) 元组
        dst_u8_slice 是 param.data.view(torch.uint8)[dst_offset:dst_offset+nbytes]
        """
```

**字段说明**:
- `name`: 参数的完整名称，对应到 `dst_tensor_name`
- `segments`: 包含多个FileSegment，每个定义一次SSD→GPU的传输
- `total_bytes`: 该参数的总字节数（所有segments的nbytes之和）
- `coalesce()`: 优化方法，合并相邻segments来减少传输次数

---

### LayerCopyPlan (层复制计划)

```python
@dataclass
class LayerCopyPlan:
    """
    Layer group 级别的复制计划，聚合多个 ParamCopyPlan
    """
    layer_id: int                               # 层索引
    group: str                                  # "mha" / "ffn"
    params: Dict[str, ParamCopyPlan]           # 参数名 → ParamCopyPlan 映射

    def total_bytes(self) -> int:
        """返回该 layer group 的总字节数"""

    def iter_copy_entries(self) -> Iterable[Tuple[str, FileSegment, torch.Tensor]]:
        """
        迭代所有 segments，返回 (param_name, FileSegment, dst_u8_slice) 元组
        供 Stream & Event Manager 实际执行拷贝
        """
```

**字段说明**:
- `params`: 字典，key是参数名（如"layers.5.attention.wq.weight"），value是对应的ParamCopyPlan
- `iter_copy_entries()`: 展平所有ParamCopyPlan的segments，返回可直接用于拷贝的迭代器

---

### GroupBlockDescriptor (组块描述符)

```python
@dataclass
class GroupBlockDescriptor:
    """
    组级块描述符：描述一个 group（attn/ffn）在 raw device 上的布局和 I/O 策略
    """
    layer_id: int                               # 层索引
    group: str                                  # "attn" / "ffn"
    mode: str                                   # I/O 模式: "block" (紧凑) 或 "scatter" (分散)

    # Block 模式相关字段
    start_offset: int = 0                       # Raw device 上的起始字节偏移
    total_span: int = 0                         # 总跨度（从第一个参数到最后一个参数，包含间隙）
    useful_bytes: int = 0                       # 实际有效载荷字节数
    fragmentation: float = 0.0                  # 碎片率 [0, 1]: (span - useful) / span

    # 参数元信息（按 SSD offset 排序）
    params: List[dict] = field(default_factory=list)  # 每个 dict 包含 {name, offset, nbytes, stride, shape, dtype}

    def can_merge(self, frag_threshold: float = 0.15) -> bool:
        """判断是否可以用单次 block-aligned IO 读取"""
```

**字段说明**:
- `mode`: I/O策略决策结果
  - `"block"`: 碎片率低，使用单次大的顺序读取
  - `"scatter"`: 碎片率高，回退到每个参数单独读取
- `start_offset`: 该group第一个参数的SSD偏移量
- `total_span`: 从第一个参数到最后一个参数结束的总字节跨度（包含中间的gap）
- `useful_bytes`: 所有参数的实际载荷总和（不含gap）
- `fragmentation`: 碎片率，计算公式 `(total_span - useful_bytes) / total_span`
- `params`: 该group的所有参数信息，按SSD offset排序

**关键计算**:
```python
span_to_payload_ratio = total_span / useful_bytes
fragmentation = (total_span - useful_bytes) / total_span = 1 - 1/span_to_payload_ratio

if fragmentation <= 0.15:  # 相当于 span_to_payload_ratio <= 1.176
    mode = "block"         # 单次读取，最多浪费15%带宽
else:
    mode = "scatter"       # 多次读取，避免浪费过多带宽
```

---

### LayerBlockTable (层块表)

```python
@dataclass
class LayerBlockTable:
    """
    整层的块表：包含 attn 和 ffn 两个组的块描述符
    """
    layer_id: int                               # 层索引
    attn_block: GroupBlockDescriptor           # Attention 组的块描述符
    ffn_block: GroupBlockDescriptor            # FFN 组的块描述符

    def total_ios_current(self) -> int:
        """Baseline 方式的 IO 次数（每个参数一次）"""

    def total_ios_optimized(self) -> int:
        """LBT 优化后的 IO 次数（每个 block-mode group 一次）"""
```

**字段说明**:
- `attn_block`: Attention组（wq, wk, wv, wo等参数）的块描述符
- `ffn_block`: FFN组（w1, w2, w3等参数）的块描述符
- 每层通常有这两个group，分别管理不同的参数集合

---

## 字段映射总结表

### FileSegment 字段对照

| 字段名 | 类型 | 含义 | 论文中的概念 |
|--------|------|------|--------------|
| `file_offset` | int | SSD/raw device上的字节偏移量 | source byte range on raw partition (起始) |
| `nbytes` | int | 传输的字节数 | length / size |
| `dtype_from` | str | 磁盘上的数据类型 | source dtype |
| `dst_offset` | int | GPU tensor内的字节偏移量 | destination byte range (起始) |
| `dtype_to` | str | GPU上的数据类型 | target dtype |

**映射关系**:
```
SSD [file_offset, file_offset + nbytes)
  → GPU tensor byte[dst_offset, dst_offset + nbytes)
```

### ParamCopyPlan 字段对照

| 字段名 | 类型 | 含义 | 论文中的概念 |
|--------|------|------|--------------|
| `layer_id` | int | 层索引 | layer id |
| `group` | str | 参数组类别 | group (MHA/FFN) |
| `name` | str | 参数完整名称 | dst_tensor_name / parameter tensor name |
| `dtype_to` | str | 目标dtype | target dtype |
| `total_bytes` | int | 总字节数 | total size |
| `segments` | List[FileSegment] | 文件段列表 | copy segments / byte-to-tensor mapping |

**作用**:
- 定义单个参数的"per-tensor ParamCopyPlan as a list of copy segments"

### LayerCopyPlan 字段对照

| 字段名 | 类型 | 含义 | 论文中的概念 |
|--------|------|------|--------------|
| `layer_id` | int | 层索引 | layer id |
| `group` | str | 组类别 | layer group (MHA/FFN) |
| `params` | Dict[str, ParamCopyPlan] | 参数名→复制计划映射 | aggregated ParamCopyPlans |

**作用**:
- "aggregates all ParamCopyPlans that belong to the same layer group into a LayerCopyPlan"

### GroupBlockDescriptor 字段对照

| 字段名 | 类型 | 含义 | 论文中的概念 |
|--------|------|------|--------------|
| `layer_id` | int | 层索引 | layer id |
| `group` | str | 组类别 | layer group |
| `mode` | str | I/O模式 | I/O mode (compact/scattered) |
| `start_offset` | int | 起始SSD偏移量 | span boundary (起始) |
| `total_span` | int | 总地址跨度 | span bytes |
| `useful_bytes` | int | 有效载荷字节数 | payload bytes |
| `fragmentation` | float | 碎片率 | 1 - (payload/span) |
| `params` | List[dict] | 参数元信息列表 | group's segments |

**作用**:
- 计算"span-to-payload ratio"来选择I/O模式
- 记录"chosen mode, span boundaries, and expected read count"

### LayerBlockTable 字段对照

| 字段名 | 类型 | 含义 | 论文中的概念 |
|--------|------|------|--------------|
| `layer_id` | int | 层索引 | layer id |
| `attn_block` | GroupBlockDescriptor | Attention组描述符 | MHA group block descriptor |
| `ffn_block` | GroupBlockDescriptor | FFN组描述符 | FFN group block descriptor |

**作用**:
- 整合整层的"Layer Block Table"，为每个group选择I/O模式

---

## 例子1: FileSegment - SSD到GPU的基本映射单元

**来源**: `layers.5.attention.wq.weight` 参数

```python
FileSegment {
    file_offset:  12,763,447,296,    # SSD上的物理字节偏移 (11.887 GB)
    nbytes:       134,217,728,        # 要传输的字节数 (128 MB)
    dtype_from:   "bfloat16",         # 磁盘上的数据类型
    dst_offset:   0,                  # GPU tensor中的目标字节偏移
    dtype_to:     "bfloat16"          # GPU上的数据类型
}
```

**说明**:
- `file_offset` 是Raw Block Device Manager需要的物理偏移量
- `dst_offset` 是GPU tensor内的字节偏移量
- 这个结构实现了论文中所说的 "maps a source byte range on the raw partition to a destination byte range in the parameter tensor"
- 传输映射: SSD `[12,763,447,296, 12,897,665,024)` → GPU tensor `byte[0, 134,217,728)`

**对应论文段落**:
> "Each segment maps a source byte range on the raw partition to a destination byte range in the parameter tensor, together with any required dtype interpretation."

---

## 例子2: ParamCopyPlan 和 LayerCopyPlan - 参数和层级的复制计划

### ParamCopyPlan 结构

**来源**: `layers.5.attention.wq.weight` 的完整复制计划

```python
ParamCopyPlan {
    layer_id:     5,
    group:        "mha",
    name:         "layers.5.attention.wq.weight",
    dtype_to:     "bfloat16",
    total_bytes:  134,217,728,        # 128 MB
    segments:     [
        FileSegment(file_offset=12763447296, nbytes=134217728,
                   dtype_from="bfloat16", dst_offset=0, dtype_to="bfloat16")
    ]
}
```

### LayerCopyPlan 结构

**来源**: Layer 5 MHA组的所有参数

```python
LayerCopyPlan {
    layer_id:     5,
    group:        "mha",
    total_bytes:  301,989,888,        # 288 MB (4个attention参数总和)
    params:       {
        "layers.5.attention.wq.weight": ParamCopyPlan(...),  # 128 MB
        "layers.5.attention.wk.weight": ParamCopyPlan(...),  # 16 MB
        "layers.5.attention.wv.weight": ParamCopyPlan(...),  # 16 MB
        "layers.5.attention.wo.weight": ParamCopyPlan(...)   # 128 MB
    }
}
```

**说明**:
- ParamCopyPlan 定义了单个参数的所有copy segments
- `coalesce()` 方法会合并相邻的segments来减少传输次数
- LayerCopyPlan 聚合了同一layer group的所有ParamCopyPlan
- Stream & Event Manager 使用LayerCopyPlan来:
  - 在transfer stream上enqueue pinned-DRAM→GPU拷贝
  - 用events gate compute stream，确保数据ready后才launch kernel

**对应论文段落**:
> "The Layer Table builds a per-tensor ParamCopyPlan as a list of copy segments... then aggregates all ParamCopyPlans that belong to the same layer group into a LayerCopyPlan."

---

## 例子3: Layer Block Table - Span-to-Payload比率计算和I/O模式选择

**来源**: Layer 5 的完整Layer Block Table (Llama3-70B)

### Attention Group Block Descriptor

```python
GroupBlockDescriptor {
    layer_id:       5,
    group:          "attn",
    mode:           "block",              # ← I/O策略: 单次大读取

    // Span和Payload指标
    start_offset:   12,763,447,296,       # 11.887 GB (SSD起始位置)
    total_span:     301,989,888,          # 288 MB (覆盖的地址范围)
    useful_bytes:   301,989,888,          # 288 MB (实际有效载荷)

    // 碎片率计算
    fragmentation:  0.0000,               # (span - payload) / span
                                          # span/payload ratio = 1.0000

    // 包含的参数 (按SSD offset排序)
    params: [
        {name: "layers.5.attention.wq.weight", offset: 12,763,447,296, nbytes: 134,217,728},
        {name: "layers.5.attention.wk.weight", offset: 12,897,665,024, nbytes:  16,777,216},
        {name: "layers.5.attention.wv.weight", offset: 12,914,442,240, nbytes:  16,777,216},
        {name: "layers.5.attention.wo.weight", offset: 12,931,219,456, nbytes: 134,217,728}
    ]
}
```

**计算过程**:
```
span-to-payload ratio = total_span / useful_bytes = 301,989,888 / 301,989,888 = 1.0000
fragmentation = (total_span - useful_bytes) / total_span = 0 / 301,989,888 = 0.0000

由于 fragmentation (0.00%) < threshold (15%)
→ 选择 mode = "block" (紧凑模式，单次大IO)
```

### FFN Group Block Descriptor

```python
GroupBlockDescriptor {
    layer_id:       5,
    group:          "ffn",
    mode:           "block",              # ← I/O策略: 单次大读取

    start_offset:   13,065,437,184,       # 12.168 GB
    total_span:     1,409,286,144,        # 1344 MB
    useful_bytes:   1,409,286,144,        # 1344 MB
    fragmentation:  0.0000,               # 0.00% (完全紧凑)

    params: [
        {name: "layers.5.feed_forward.w1.weight", offset: 13,065,437,184, nbytes: 469,762,048},
        {name: "layers.5.feed_forward.w3.weight", offset: 13,535,199,232, nbytes: 469,762,048},
        {name: "layers.5.feed_forward.w2.weight", offset: 14,004,961,280, nbytes: 469,762,048}
    ]
}
```

### I/O优化效果对比

```
Baseline方式 (scattered reads):
  - Attention组: 4次小I/O (每个参数单独读取)
  - FFN组:       3次小I/O
  - 总计:        7次I/O

LBT方式 (block reads):
  - Attention组: 1次大I/O (读取288 MB span，浪费0 MB)
  - FFN组:       1次大I/O (读取1344 MB span，浪费0 MB)
  - 总计:        2次I/O

优化效果:
  - I/O次数减少: 71.4% (从7次降到2次)
  - 浪费空间:    0% (这个模型参数在SSD上完全紧凑排列)
```

**说明**:
- Layer Block Table记录了每个group的I/O模式选择
- `fragmentation` 指标 = `1 - (useful_bytes / total_span)` 等价于论文中的span-to-payload ratio
- 当fragmentation ≤ 15%时，选择"block"模式（一次大读取+unpack）
- 当fragmentation > 15%时，选择"scatter"模式（每个参数单独读取）
- 在这个例子中，两个group都完全紧凑（0% fragmentation），因此都使用block模式

**对应论文段落**:
> "It computes a span-to-payload ratio by sorting the group's segments by SSD offset, measuring the covered address span, and dividing span bytes by payload bytes. If the ratio is below a threshold, the group is considered compact on disk and is fetched via one block-aligned sequential read into a pinned staging buffer, followed by unpacking into tensors using the LayerCopyPlan offsets. Otherwise, the backend falls back to scattered reads."

---

## 高碎片率例子 (假设场景)

假设某个layer的参数在SSD上分散存储：

```python
GroupBlockDescriptor {
    layer_id:       15,
    group:          "attn",
    mode:           "scatter",            # ← 碎片率过高，回退到scattered reads

    start_offset:   20,000,000,000,       # 起始
    total_span:     500,000,000,          # 476 MB span
    useful_bytes:   300,000,000,          # 286 MB payload
    fragmentation:  0.4000,               # 40% 碎片率！

    params: [
        {offset: 20,000,000,000, nbytes: 100,000,000},  # 参数1
        // [gap: 150 MB]                                 ← 大量浪费空间
        {offset: 20,250,000,000, nbytes: 100,000,000},  # 参数2
        // [gap: 150 MB]
        {offset: 20,500,000,000, nbytes: 100,000,000}   # 参数3
    ]
}
```

**计算**:
```
span-to-payload ratio = 500,000,000 / 300,000,000 = 1.67
fragmentation = (500,000,000 - 300,000,000) / 500,000,000 = 0.40 (40%)

由于 fragmentation (40%) > threshold (15%)
→ 选择 mode = "scatter" (分散模式，每个参数单独读取)
→ 避免读取300 MB额外的gap数据
```

**决策逻辑**:
- 如果用block模式: 1次I/O读取476 MB，但其中190 MB是gap (浪费40%)
- 使用scatter模式: 3次I/O各读取95 MB，总共286 MB (无浪费)
- 虽然增加了系统调用次数，但减少了带宽浪费和DRAM staging buffer压力

---

## 总结

三个table的层次关系：

```
LayerBlockTable (Layer 5)
├── GroupBlockDescriptor (attn)
│   ├── mode: "block"
│   ├── fragmentation: 0.00%
│   └── params: [wq, wk, wv, wo] → 决定I/O策略
│
└── GroupBlockDescriptor (ffn)
    ├── mode: "block"
    ├── fragmentation: 0.00%
    └── params: [w1, w3, w2] → 决定I/O策略

LayerCopyPlan (Layer 5, MHA)
├── ParamCopyPlan (wq)
│   └── segments: [FileSegment(...)]  → SSD offset → GPU tensor byte slice
├── ParamCopyPlan (wk)
│   └── segments: [FileSegment(...)]
└── ...

FileSegment: 基本映射单元
  SSD [file_offset, file_offset+nbytes) → GPU tensor [dst_offset, dst_offset+nbytes)
```

**设计要点**:
1. **FileSegment**: 实现物理层(SSD offset)到逻辑层(GPU tensor offset)的映射
2. **ParamCopyPlan/LayerCopyPlan**: 从runtime manifest构建，提供正确的byte-to-tensor映射
3. **Layer Block Table**: 基于碎片率分析，选择compact或scattered I/O模式，减少SSD-side fragmentation

这三个结构共同实现了论文中描述的"bridges these views by translating the runtime manifest into group-scoped copy plans"的功能。

---

## 实际使用流程示例

### 阶段1: 初始化时构建表结构

```python
# 1. 加载 runtime manifest
manifest = json.load(open("runtime_manifest.json"))

# 2. 构建 LayerBlockTable (Layer Block Table)
layer_5_table = build_layer_block_table(manifest, layer_id=5)

# 输出:
LayerBlockTable {
    layer_id: 5,
    attn_block: GroupBlockDescriptor {
        mode: "block",
        start_offset: 12,763,447,296,
        total_span: 301,989,888,
        useful_bytes: 301,989,888,
        fragmentation: 0.0,
        params: [wq, wk, wv, wo]  # 4个参数
    },
    ffn_block: GroupBlockDescriptor {
        mode: "block",
        start_offset: 13,065,437,184,
        total_span: 1,409,286,144,
        useful_bytes: 1,409,286,144,
        fragmentation: 0.0,
        params: [w1, w3, w2]  # 3个参数
    }
}

# 3. 构建 LayerCopyPlan (从 ParamStore)
store = ParamStore("runtime_manifest.json")
param_tensors = {
    "layers.5.attention.wq.weight": torch.empty([8192, 8192], dtype=torch.bfloat16, device='cuda:0'),
    "layers.5.attention.wk.weight": torch.empty([1024, 8192], dtype=torch.bfloat16, device='cuda:0'),
    # ... 其他参数
}

layer_copy_plan = build_layer_copy_plan(store, param_tensors, layer_id=5, group="mha")

# 输出:
LayerCopyPlan {
    layer_id: 5,
    group: "mha",
    params: {
        "layers.5.attention.wq.weight": ParamCopyPlan {
            segments: [
                FileSegment(file_offset=12763447296, nbytes=134217728,
                           dst_offset=0, dtype_from="bfloat16", dtype_to="bfloat16")
            ]
        },
        # ... 其他参数
    }
}
```

### 阶段2: 运行时使用 (两个不同的路径)

#### 路径A: 使用 LayerBlockTable 进行批量I/O (Raw Device → CPU Pinned DRAM)

```python
# Raw Block Device Manager 使用 GroupBlockDescriptor 来决定 I/O 策略

if layer_5_table.attn_block.mode == "block":
    # Compact模式: 单次大I/O
    offset = layer_5_table.attn_block.start_offset      # 12,763,447,296
    size = layer_5_table.attn_block.total_span          # 301,989,888 bytes

    # 一次 pread() 读取整个 span 到 staging buffer
    dio_file.pread_into_tensor(staging_buffer, size, offset)

    # 从 staging buffer 解包到各个参数
    for param_info in layer_5_table.attn_block.params:
        param_offset_in_block = param_info["offset"] - offset
        param_nbytes = param_info["nbytes"]

        # 创建 pinned tensor 并复制有效字节
        param_tensor = torch.empty(param_info["shape"], dtype=..., pin_memory=True)
        param_tensor.view(torch.uint8)[:param_nbytes].copy_(
            staging_buffer[param_offset_in_block : param_offset_in_block + param_nbytes]
        )

        weights[param_info["name"]] = param_tensor

else:
    # Scatter模式: 逐参数I/O
    for param_info in layer_5_table.attn_block.params:
        offset = param_info["offset"]
        stride = param_info["stride"]

        # 每个参数单独读取
        dio_file.pread_into_tensor(staging_buffer, stride, offset)

        param_tensor = torch.empty(param_info["shape"], dtype=..., pin_memory=True)
        param_tensor.view(torch.uint8)[:param_info["nbytes"]].copy_(
            staging_buffer[:param_info["nbytes"]]
        )

        weights[param_info["name"]] = param_tensor
```

#### 路径B: 使用 LayerCopyPlan 进行数据传输 (CPU Pinned DRAM → GPU HBM)

```python
# Stream & Event Manager 使用 LayerCopyPlan 来调度 GPU 拷贝

# 假设已经通过路径A将数据加载到 CPU pinned memory
# 现在需要传输到 GPU

for param_name, file_seg, dst_u8_slice in layer_copy_plan.iter_copy_entries():
    # file_seg: FileSegment 包含源和目标信息
    # dst_u8_slice: GPU tensor 的字节切片视图

    # 从 pinned memory 读取对应的字节 (已由 Raw Device Manager 准备好)
    pinned_src = pinned_weights[param_name].view(torch.uint8)[
        file_seg.dst_offset : file_seg.dst_offset + file_seg.nbytes
    ]

    # 在 transfer stream 上异步拷贝到 GPU
    with torch.cuda.stream(transfer_stream):
        dst_u8_slice.copy_(pinned_src, non_blocking=True)

    # 记录 event
    transfer_event = torch.cuda.Event()
    transfer_event.record(transfer_stream)

    # Compute stream 等待数据 ready
    compute_stream.wait_event(transfer_event)

# 现在 compute stream 可以安全地 launch kernel
with torch.cuda.stream(compute_stream):
    output = attention_forward(...)  # 所有参数已就绪
```

### 数据流总结

```
初始化阶段:
  Runtime Manifest
    ↓ build_layer_block_table()
  LayerBlockTable (决定 I/O 策略: block vs scatter)
    ↓
  GroupBlockDescriptor (记录 span, fragmentation)

  Runtime Manifest + GPU Param Tensors
    ↓ build_layer_copy_plan()
  LayerCopyPlan
    ↓
  ParamCopyPlan → FileSegments (映射 SSD offset → GPU tensor offset)

运行时阶段:
  SSD (Raw Block Device)
    ↓ Raw Block Device Manager (使用 GroupBlockDescriptor)
    ↓ pread() - block mode: 1次大IO | scatter mode: N次小IO
  CPU Pinned DRAM (Staging Buffer)
    ↓ Unpack (使用 GroupBlockDescriptor.params)
  CPU Pinned DRAM (Per-Param Tensors)
    ↓ Stream & Event Manager (使用 LayerCopyPlan)
    ↓ cudaMemcpyAsync() - 使用 FileSegment 的映射信息
  GPU HBM (Parameter Tensors)
    ↓ Event gating (使用 LayerCopyPlan 的 event)
  Compute Stream (Kernel Launch)
```

### 关键设计决策对照

| 组件 | 决策点 | 输入 | 输出 |
|------|--------|------|------|
| **Layer Block Table** | I/O模式选择 | fragmentation ratio | mode: "block" / "scatter" |
| **GroupBlockDescriptor** | 单次I/O大小 | start_offset, total_span | pread(offset, size) |
| **LayerCopyPlan** | GPU拷贝调度 | FileSegments | iter_copy_entries() |
| **FileSegment** | 字节映射 | file_offset, dst_offset | SSD→GPU映射 |

这个流程展示了论文中描述的完整数据路径：
1. **Layer Block Table** 减少 SSD-side fragmentation (通过选择合适的I/O模式)
2. **LayerCopyPlan** 提供正确的 byte-to-tensor mapping (通过FileSegments)
3. **Stream & Event Manager** gate compute stream on data readiness (通过events)
