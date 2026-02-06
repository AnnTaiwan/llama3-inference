## 執行inference直接使用 `inferencellama3-1-70B_overlap_metrics.py`
## llama3
###  `__init__.py`
* 初始化
### `config.py`
* 默認參數設定
* 主要是還KV args以及model args兩個部分
* kv 其實是kv pool的設定也就是kv 的最大容量和block部分的配置
* model 主要是從json中获取这个model的一些metadata
### `generator.py`
* 初始化model, 也就是如何在cpu中將checkpoint進行build
* 如果要做將weight kv這些東西下放到SSD也就是FABLE這樣的事, 這邊比較重要
* 因為你要在這邊控制初始化在CPU上, 並且只能初始化, 不能使用pytorch的api直接來做checkpoint
* 這個部分要自己來做, 不然會在build的時候將所有的weight一次加載到CPU的導致OOM, 當然小於70B的model那臺主機都能跑
* `build`這個function還有一個就是順便將小weight都一次性常駐GPU中, 也就是embedding, norm之類的
* 還有就是`text_completion`這個function也挺重要的如果是和我一樣使用chunk prefilling的方法, 記得在這邊將prefilling的logits關了, 就是prefill不會生成token
### `gloable_state_tracker.py`
* 基本可以pass, 當初debug和用來測一些時間和數據, 好像後面也沒怎麼用到
### `gpu_utils.py`
* 一樣是debug用的, 防止一些oom的情況, 但是好像就算用了也沒有辦法, 因為checkpoint還是會oom XD
### `hbm_slab.py`
* 沒用, 應該都使用`memory_manager.py`, 這個應該是測layer granularity的時候的東西
### `kv_offload.py`
* 老實說我寫的太噁心了, 自己都有點看不懂
* 其實對於kv block的部分可以直接去看vllm的部分, 我的是一個很簡單的版本, 還有topk也是一樣
* 並且我的版本是完全使用raw block device這個東西來從SSD讀取的, 我好像沒有留一般FS的api
* `push`相對來說比較簡單
* `fetch()`, 同步, 返回 KV , 使用場景是立即需要数据
* `prefetch_async()`,	异步, 不返回數據,	prefetch data to gpu
* `prefetch_blocks_async()`	异步, 不返回數據,	prefetch_async 的wrapper
* `prefetch_for_next_layer()`, 异步, 不返回數據, pipeline的部分
* `_gather_from_gpu_cache_strict()`, 同步, 返回 KV,	fetch()一些優化
### `layer.py`
* 如果不是和我一樣使用cuda stream和cuda event進行inference控制的話不要看我這個, 只需要直接看nanovllm的部分, 將mha和ffn這些接起來就好
* `make_stub_linear` 感興趣可搜一下meta骨架那種東西來做model的初始化部分, 但是我這個有太多bug了, 我後面直接使用chunk prefill, 可能裡面還有很多meta骨架的痕跡如果發現可以刪掉
* `RMSNorm`基本可以照搬到你們自己東西, 但是記得看我的精度是BF16, 這個是50-level gpu才能用的精度我記得
* `apply_rotary_embeddings` 和 `precompute_theta_pos_frequencies`這兩個在我印象中好像不是全部model都會用這個部分, LLaMA是可以, 我不確定別的是否可以用, 還有後面的MHA在LLaMA中是使用GQA技術, 感興趣自己可以搜一下, 這個也不一定所有model都支持, 這個要小心一點
* `SelfAttention` 和 `FeedForward` 兩個都是會被cuda stream和event包出的對於forward部分, 這個要小心界限, 就是確認是否完成這個計算再將event送出去
* `EncoderBlock` 是我為了fine grained專門做的部分, 為了將MHA和FFN兩個部分分開, 在這邊進行順序計算的擺放以及資料的prefetch順序
* 還有一個重點就是對於資料和計算的順序我是嚴格按照先MHA再FFN, 其實這個有可能可以進行優化
* 有一個小tip 就是你們後面實做的過程中一定會遇到prefetch沒有成功的例子, 可能會報錯說資料不在cuda上無法進行計算, 要使用一個lock確認資料已經完整的在gpu memory中才能開始進行計算, 不然會很經常報錯
* 忘記說了每一個如果使用cuda stream的話記得在計算開始的時候綁定, 再注意一點就是cuda stream我是用pytorch api做的, 我不確認c / cpp能不能做到這個事, 可能要自己確認
### `memory_manager.py`
* `GlobalMemoryManager` 配置gpu memory上限大小, 超過閥值就報錯之類的部分
* `HostPinnedExtentPool` pin memory pool
### `model.py`
* 用來跑`layer.py`的部分, 主要使用`forward pipeline`來跑這個是asyn的, 另一個是同步的, 注意的一個點是好像`forward pipeline`在計算的過程中因為是asyn所以有可能會出現資料不在schedule中, 導致報錯的部分, 因為沒有每次都進行確認資料是否在device中, 也就是gpu memory中
### `raw_param_store.py`
* raw block device的wrapper, 這個部分很簡單因為主要是weight的read部分, 用法在裡面都有, 想搞懂流程可以看我的論文
### `registered_pool.py`
* staging buffer的創建和使用部分
### `SSDBacked.py`
* raw block device的wrapper, 但是這個部分是kv cache的read和write部分, 和上面那個wrapper, 如果可以使用cpp應該會更快, 我裏面每個function也都有詳細的說明
### `stream_mnt.py`
* cuda stream的創建和優先級設定, 以及這也是stream的wrapper, 舉例stream的操作都在這邊
### `weight_lbt.py`
* 這個部分寫的不好, 有興趣可以按照我的論文中的那種方法進行修改, 這邊太多table了, 反而會拖慢效率, 這個改了說不定會更快
### `weight_streaming_manager.py`
* 這個是sliding window的核心, 如果要復現FABLE就需要看懂這個就好, 但是裡面肯定有很多是用不到的function, 所以其實沒有那麼多東西, 有些是同步的部分我好像沒有刪掉
* 注意一個點就是sliding window是不行從尾巴變成第一個, 也就是layer 79 不會自己指向 layer 0, 所以我有一個ring window用來做這個事
* 注意一個點就是mark mha或者ffn這個部分是用來lock前後關係的
* 裡面可能還有殘留的LRU和OPT策略, 看到記得刪除, 只需要使用sliding window來管理
### `weights_io_ssd_dram.py`
* 這個是把整個direct IO包起來的wrapper
* 見誰整個runtime manifest的部分也在這邊, 也就是初始化raw block device的部分

## `inferencellama3-1-70B_overlap_metrics.py
* 絕大多數都是測試時間和數據的patch, 也就是`class InferenceProfiler`
* 主要的參數在開頭部分有 
* `CHUNK_SIZE = int(os.environ.setdefault("PREFILL_T_CHUNK", "512"))# prefill 分块大小`
* `MIRCO_BATCH_SIZE  = os.environ.setdefault("MIRCO_BATCH_SIZE", "8")# micro batch size`
* `ATTN_MICRO_B = os.environ.setdefault("ATTN_MICRO_B", "8")# attention micro batch size`
* 要看inference的流程直接從 main開始看就好了
* 參數配置就不多說了
* 主要是對prompt的處理部分, 裁剪和batch的部分可能需要仔細看一下
```
GPU_AHEAD_LAYERS = 8 #prefetch distance
GPU_MAX_GROUPS   = 12 #gpu最大能放多少個
GPU_WARMUP_LAYERS = 10 #warmup的時候放多少個layer到gpu中
CPU_CACHE_LAYERS = 47# #dram最多多少layer
DEFAULT_BATCH_SIZE  = int(os.getenv("PROMPT_BATCH", "64"))   # batch size
DEFAULT_MAX_GEN_LEN = int(os.getenv("GEN_TOKENS", "32"))    # 生成 token 数
```
* 如果要復現實驗就最主要通過修改這個來inference
## `generate_manifest.py`
* 這個是生產model使用raw block device的初始化腳本
* 具體用法在code中都有, 只需要按照步驟就會生成了
```
def _hf_to_internal_name(name: str) -> str:
    n = name
    if n.startswith("model."): n = n[len("model."):]
    n = n.replace(".self_attn.q_proj.", ".attention.wq.")
    n = n.replace(".self_attn.k_proj.", ".attention.wk.")
    n = n.replace(".self_attn.v_proj.", ".attention.wv.")
    n = n.replace(".self_attn.o_proj.", ".attention.wo.")
    n = n.replace(".input_layernorm.", ".attention_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".feed_forward.w1.")
    n = n.replace(".mlp.up_proj.",   ".feed_forward.w3.")
    n = n.replace(".mlp.down_proj.", ".feed_forward.w2.")
    if n == "model.embed_tokens.weight": n = "embed_tokens.weight"
    if n == "lm_head.weight": n = "output.weight"
    return n
```
不同model的這個部分肯定不一樣, 這個要看那個model的結構要自己改

