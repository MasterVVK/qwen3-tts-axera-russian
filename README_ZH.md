# Qwen3-TTS 在 CM3588 上运行 — RTF 2.0x (llama.cpp + GGML)

高性能 C++/Python 管道，用于在 FriendlyElec CM3588 (RK3588) 上运行 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)。

**实现 RTF 2.0x** — 边缘设备上最快的 Qwen3-TTS（基线 RKLLM 为 RTF 5.5x）。

## 架构

```
                   ┌─────────────────────┐
文本 ──────────────┤  分词器 + 嵌入      ├──→ 嵌入向量
                   └────────┬────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │    llamacpp_talker_server.py         │
         │    llama.cpp GGUF Q4_K_M (A76)      │
         │    ~30-35 tok/s, ~30ms/步            │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    code_pred_server (GGML Q4_0)     │
         │    比 ONNX 快 4.8 倍                │
         │    ~55ms/步                          │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    vocoder_server.py                 │
         │    RKNN FP16 (NPU, 3 核)            │
         │    有状态流式，无接缝                │
         └──────────────────────────────────────┘
```

三个独立服务器通过 Unix socket 通信，可独立重启。

## 性能

在 FriendlyElec CM3588 (RK3588 4xA76+4xA55, 32GB RAM) 上测试:

| 组件 | 耗时 | 备注 |
|---|---|---|
| Talker (llama.cpp A76) | ~30ms/步 | GGUF Q4_K_M, 469 MB |
| Code Predictor (GGML Q4_0) | ~55ms/步 | **比 ONNX 快 4.8 倍** |
| 声码器 (ONNX FP32) | ~5.0s/64-tok | Overlap-crossfade |
| **总 RTF** | | **2.0x** |

在纯 RK3588 板卡上实测（4×A76 绑核，无 Axera NPU，俄语文本）：原样是 RTF 3.9-4.4x。
当时 code predictor 每个 15 步请求耗时 199 ms；删掉
`qwen3-tts.cpp/src/tts_transformer.cpp` 里的 `ggml_cast(ffn_down, F32)` 节点后（ggml CPU 后端的
`mul_mat` 可直接吃 F16/Q8_0 权重；cast 节点每层每步反量化 12.6 MB，每个 token 重复 75 次）
降到 64 ms，生成速度 4.5 → 11.5 tok/s，**RTF 2.6x**。把声码器以 **RKNN FP16** 移到 RK3588 NPU
后（无损，SNR 56.8 dB，见下文），流式模式达到 **RTF 1.8x**（生成与声码并行 8.9 tok/s，首个分块约 3 s）。

### 声码器：CPU 上的 FP32 ONNX vs NPU 上的 RKNN

声码器使用 **Snake 激活函数** (`x + sin²(α·x)/α`)，与*量化*根本不兼容:
- **RKNN Q8**: 快 1.75 倍但有可听噪声（随机标定 SNR 9.5 dB，真实录音标定 13.4 dB）
- **ONNX INT8**: 慢 3 倍且质量差（SNR 4.2 dB）
- **AX650N NPU**: 不支持 Conv1D dilation=9
- **MNN GPU**: SineGen 形状广播错误

**RKNN FP16 (`w16a16`) 不是量化，实测无损**：相对 FP32 参考 SNR 56.8 dB、corr 1.0000，速度是
4 核 CPU 的 1.34 倍——而且跑在 NPU 上，不再和 code predictor 抢 A76 核心，所以流式模式现在更快。
部署：`python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx`（不加 `--quantize`），
`VOCODER_MODEL` 指向结果，并设置 `PYTHONPATH=dual_npu/rknnlite_shim`（板载 Python 比 lite2 wheel
更新时，由该 shim 通过 `librknnrt.so` 提供 `RKNNLite` API）。

### 声码器分块接缝

WavTokenizer 解码器是**因果的**：同一个 token 在零填充的块首和在序列中间解码出的波形*不同*
（实测 corr 0.445，SNR 0.3 dB）。原来的 16-token 零填充 overlap-crossfade 混合了两条不对齐的
波形——每个边界都有可听的"双声"。现在的分块策略分层实现，每层都与同一批 codes 的 256-token
单次 ONNX 运行对比测量：

- **真实上下文代替填充**：每块前拼接上一块的尾部 token（112 槽模型用 48；短的尾块用上下文填满
  所有空槽），其音频裁掉。边界 SNR：16 ctx 时 4.1 dB → 32 时 9.9 → 48 时 19.1 → 60 时 19.8
  （饱和；感受野远大于此）。
- **接缝移入停顿**：相邻段不能 crossfade（19 dB 去相关波形相加 = 梳状滤波爆音），删语音又可听
  见——所以接缝*移动*：块末 3 s 内最长的连续静音段（≥150 ms，20 ms-RMS < 窗口 RMS 的 25%）成为
  块尾，`stitch()` 在那里只丢静音（cos/sin 等能量窗）。句中接缝退化为接缝附近最静点处 20 ms
  等能量微混合。
- **Warmup-truncate**（CosyVoice 式重叠）：每块多解码接缝之后 2 个 token，下一块从接缝开始，
  共享区（含下一块块首无上下文卷积瞬态）在停顿中丢弃或在响接缝处混合。接缝 max|diff| 相对
  单次参考：97–108 m → 0.1–9.5 m。

端到端（RKNN FP16，112 槽）：对单次运行的全局 SNR 26–32 dB，接缝局部 25–57 dB；残余是接缝后
的 FP16 重解码松弛。彻底的解法是下面的有状态导出——它根本没有接缝；分块方案保留为无 NPU 部署的
回退路径。

### 声码器：有状态流式（无分块）

WavTokenizer v2 解码器**完全因果**，所以它的全部历史可以放进模型自身的 I/O，而不必每个分块重新
喂入。`scripts/export_vocoder_stateful.py` 导出一个流式步：所有因果卷积的尾部（`(k-1)*d` 个输入，
转置卷积为 `ceil((k-1)/s)`）以及 pre-transformer 的滑动窗口 KV（71 个 key × 8 层 × 2 × 16 头 × 64）
打包进**一个扁平 float32 缓存张量——1303424 个数，5.2 MB——它既是输入也是输出**：

```
audio_codes [1,16,64] int64 ─┐
freqs_in    [64,32]   fp32  ─┤→ 一步 →  audio_values [1,122880]  （64 token = 5.12 秒 24 kHz）
cache_valid [1]       fp32  ─┤          cache_out   [1303424] fp32
cache_in    [1303424] fp32  ─┘
```

各处形状都是固定的，所以转换方式与任何声码器模型相同：

```bash
python3 scripts/export_vocoder_stateful.py --codes-length 64 \
    --output models/vocoder_stateful_64.onnx
python3 scripts/convert_vocoder_rknn.py --model models/vocoder_stateful_64.onnx --stateful \
    --output models/vocoder_stateful_64_fp16.rknn
# 板端：VOCODER_MODEL=/root/tts-rknn/vocoder/vocoder_stateful_64_fp16.rknn（文件名必须含
#       'stateful'——服务器和客户端就是靠它选择流式路径）
```

**可选，降低 TTFB：** 再导出一个小 B 的步，用它发声每句话的*第一*组 token。一步的固定开销是
那 5.2 MB 缓存往返，与 B 无关，所以 16 个 token 几乎和 64 个一样便宜——而这是 1.28 秒音频，
听者早 7 秒听到：

```bash
python3 scripts/export_vocoder_stateful.py --codes-length 16 \
    --output models/vocoder_stateful_16.onnx
python3 scripts/convert_vocoder_rknn.py --model models/vocoder_stateful_16.onnx --stateful \
    --output models/vocoder_stateful_16_fp16.rknn
# 板端：VOCODER_MODEL=.../vocoder_stateful_64b_fp16.rknn \
#       VOCODER_MODEL_FIRST=.../vocoder_stateful_16_fp16.rknn
```

`cache_numel` 与 B 无关（卷积尾部 + 71 键的窗口），所以 B=16 那步的 `cache_out` 可以直接喂给
B=64 那步：混合 `[16,64,64,64]` 对单次完整解码测得 **110.4 dB**，最大差 1e-05。不设
`VOCODER_MODEL_FIRST` 时服务器保持单模型行为。

协议（同一 socket，无新操作码）：服务器**缓冲**请求里的 token，只在凑满 64 个一组时前进，因此
请求边界落在组中间时绝不做零填充（零填充会把静音写进缓存——仅此一项端到端就损失 25 dB）。
`n_tokens=0` 的请求就是 **flush**：把剩余的不足 64 个 token 发声并返回其音频，然后清零状态。客户端
在一句话开始时发一次该 reset（空 flush），最后一个分块之后再发一次，然后把各段直接拼接——不
stitch、不 overlap、无接缝。

NPU 的 fp16 构建强制了四个不显然的选择，每个都经过测量：

- **RoPE 角度作为宿主输入，cos/sin 在图内计算。** 把 cos/sin 作为输入喂入会让 fp16 流水线爆炸
  （q 的 rms 0.68 → 120）。原始角度 `pos*inv_freq` 可达 ~1e4，那里 fp16 ulp 是 0.5，所以宿主先对
  2π 取模（值 ≤ 6.28，ulp 0.004）。
- **attention mask 是图内常量，取值 −100**，建立在*相对*网格（`qp-kp`）上，因此与步序号无关。
  −3e4 会穿透 NPU 的 softmax（注意力变均匀）；而把绝对 mask 作为*输入*并与图内 cos/sin 同时使用，
  会让板端运行时 SEGFAULT。
- **缓存有效性是标量输入**——而且是一个**计数**，`cache_valid = min(pos, 71)`，不是布尔标志。
  用标志时，第一步之后的每一步都会把 `Lc - B` 个从未写入的窗口槽当成真实 key：零值 key 在 softmax
  里得分是 0 而不是 −inf，于是从真实 key 那里抢走质量（逐步 SNR：B=64 时 48.8 dB，B=16 时
  10.2 dB）。图里用 Sub/Clip/Mul 把该计数变成逐 slot 的加性偏置。改从缓存数据推导有效性
  （对 1.3 MB 缓存做 Abs/ReduceSum/Clip 偏置链）会损失 30 dB。
- **KV 历史通过常量选择矩阵乘接入**（`A@cache_k + P@k`），而不是 Concat：RKNN 的 Concat 路径会
  破坏滑动窗口注意力（9.5 dB 对 56.8 dB）。

板端实测，同一段 200 token 俄语文本，`--streaming`，与同一批 codes 的 256-token ONNX FP32 单次
运行对比：

| 声码器 | 全局 SNR | 每 64 token 分区 | 生成速度 | 声码等待 | RTF |
|---|---|---|---|---|---|
| 分块接缝（112 槽） | 26.4 dB | 57.5 / 21.1 / 29.7 | 6.5 tok/s | 6.0 s | 2.6x |
| **有状态流式** | **48.0 dB** | 56.2 / 44.5 / 51.1 / 53.4 | 7.5 tok/s | 0.0 s | **1.9x** |

有状态方案在整句话中任何位置都不低于 44 dB（分块方案在第一个接缝之后就掉 30 dB），并且它不再从
code predictor 手里抢 A76 核，所以生成本身也变快了。（有状态这一行的全局 SNR 是在缓存有效性修复
之前的模型上测的；把同一个模型文件重新对同一批 144 个录制 codes 的 torch-fp32 完整解码来测，得到
全局 **25.1 dB**，分区 55.6 / 21.3 / 23.8——衰减正好从第二步开始，也就是幻影 key 开始出现的地方。
修复后的导出测得 **54.3 dB**（分区 55.6 / 53.4 / 51.7，平坦），出厂的 B=16 + B=64 路径 **53.9 dB**
（55.0 / 53.5 / 50.7）：小的首组值 0.4 dB，等于不花钱。三者都是 float32 进程内、无 int16 往返；
3 步各 64 token 用 10.8 秒，3 步（16+64+64）用 8.1 秒。）导出的 fp32 自检（逐步 vs 完整
解码，`--check-only`）在 B=64 时为 111.4 dB，B=16 时为 103.7 dB（用 onnxruntime 重跑导出的图：
111.1 / 103.1）——残差是 KV 选择矩阵乘的 fp32 累加噪声，不是语义错误。（修复前同一检查是 48.8 dB，
所以它的阈值是 40 dB：导出会在 `torch.onnx.export` *之前*中止，从而给转换器留下一个过期的 ONNX。）

### 优化历程

| 版本 | Code Predictor | 声码器 | RTF |
|---|---|---|---|
| 基线 (RKLLM talker) | ONNX 单线程 (374ms) | RKNN Q8 | 5.5x |
| + GGML Q4_0 CP | **GGML (55ms)** | RKNN Q8 | 2.2x |
| + ONNX FP32 声码器 | GGML (55ms) | **ONNX FP32** | **2.0x** |
| + 修复 ffn_down cast | GGML Q8_0 (64ms/15步) | ONNX FP32 | 2.6x（流式 3.0x） |
| + RKNN FP16 声码器上 NPU | GGML Q8_0 (64ms/15步) | **RKNN FP16（112 token 分块）** | 1.8x |
| + 有状态流式声码器 | GGML Q8_0 (64ms/15步) | **RKNN FP16 有状态（无分块）** | **1.9x** |
| + 首组 16 token | GGML Q8_0 (64ms/15步) | **RKNN FP16 有状态（B=16 + B=64）** | **1.7x** |

最后四行是在本板卡实测（俄语文本，4×A76；修复 cast 那行是非流式，其余三行 `--streaming`）；
上面各行的 CP 耗时是上游报告的数值。1.8x、1.9x 与 1.7x 三行文本长度不同（137 / 200 / 443
token）——有状态路径与分块接缝声码器在同一 200 token 文本上的 A/B 是 2.6x → 1.9x。

有状态路径的流式 TTFB：单个 B=64 导出时 **≈9.3 秒**——首个 64-token 分块在 +5.7 秒提交，有状态
声码器一步 3.6 秒（FP32-CPU 声码器时代是 19–23 秒）；加上 16-token 首组模型后是 **2.4 秒**（实测：
首个音频在 +2.4 秒，其中 1.28 秒音频，948 字符俄语文本，4×A76，两个模型都是 NPU 上的 FP16）。
code predictor 占每个 token 的 63% CPU 时间，也是唯一
NPU *帮不上忙*的组件：有状态单步图（`scripts/export_code_predictor_stateful.py` +
`scripts/convert_code_predictor_rknn.py`，自检 15/15 码本、首步 logit 相对 fp64 参考差 2.8e-05）
在三个 NPU 核上是 **23.2 ms/步，而 4×A76 + GGML Q8_0 是 4.3 ms/步——慢 5.4 倍**。157 MB fp16
权重 23 ms 读完只有 6.8 GB/s，比板卡实测上限低 4 倍（4×A76 流式读 4 KB 块 27.3 GB/s，
4×A55 是 17.2 GB/s），所以代价在单算子开销：每次调用约 75 个 S=1 的微型 GEMV，外加
1.3 MB KV 缓存跨越运行时边界。NPU 在真正的 GEMM 上才划算（声码器 64-token 分块），而不是
16 个一次一个 token 的串行步。`.rkllm` 路线被同一个不等式关闭：CPU 路径每步真正读取的
~72 MB Q8_0 权重用 4.24 ms 读完 = 17 GB/s，即实测上限的 62%，而 RKLLM 在本板实测的有效
权重读取速度约 15 GB/s。

长句不再被截断。talker 上下文是 `N_CTX`（默认 2048；`N_BATCH` 独立且保持小值——ggml compute
buffer 由 n_batch 决定大小：2048 时 1243 MiB，512 时 311 MiB，而它占的正是 code predictor 用的
那几个核），每请求预算是 `min(MAX_TOKENS, N_CTX − prefix − 1)`，所以长文本由模型自己的 EOS 收尾。
板端实测，同一个 948 字符请求（784–790 token，62–63 秒音频）：非流式 9.6 tok/s，wall 123 秒，
**RTF 2.0x**；`--streaming` 6.9 tok/s，声码器等待 0.0 秒，wall 117 秒，**RTF 1.9x**。流式买到的是
TTFB 而不是吞吐：有状态声码器的 CPU 侧工作（每个 64-token 分块有 10.6 MB 缓存与音频进出
RKNN 运行时）与 code predictor 共享四个 A76 核，代价约其速率的 28%。把声码器搬到 A55 集群
（`VOC_CORES=0-3`）实测是亏损：5.3 tok/s，RTF 3.9x——A55 扛不住这份 marshalling
（17.2 对 27.3 GB/s）。

客户端不再按自己的环境变量猜声码器协议，而是在每句话开头用一个 0-token 请求去问
（有状态服务器会应答，分块接缝服务器直接关连接）。旧的推断方式会让有状态服务器静默走
分块路径：前置 48 个上下文 token，然后对服务器根本没分块的音频做裁剪与接缝拼接——
整句静默丢掉 26%（795 token 只出 47.04 秒音频，应为 63.6 秒），且没有任何报错。

## 快速开始

```bash
# 单次合成
bash dual_npu/launch_qwen3_tts.sh "你好，今天天气怎么样？"

# 守护进程模式
bash dual_npu/launch_qwen3_tts.sh --daemon

# 客户端
python3 dual_npu/tts_client.py "要合成的文本" --language chinese

# 流式声码器（与生成并行）
python3 dual_npu/tts_client.py "长文本..." --language chinese --streaming
```

## Code Predictor 变体

管道支持三种后端（自动选择）:

| 后端 | 速度 | 备注 |
|---|---|---|
| **GGML Q4_0**（默认）| 55ms/步 | 需要 [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp) |
| C++ ONNX Runtime | 240ms/步 | 内置, `dual_npu/code_predictor_cpp/` |
| Python ONNX Runtime | 248ms/步 | 回退方案，无需编译 |

GGML 后端需要的量化模型仓库里没有现成生产者（`llama-quantize` 不认识 `qwen3-tts` 架构）：
用 `python3 scripts/quantize_code_predictor_gguf.py --in <f16.gguf> --out <cp-q8_0.gguf>`
生成，只保留 code predictor 读取的张量并把 matmul 权重量化成 Q8_0。RK3588 实测：
1745 MB → 183 MB，CP 服务进程 RSS 1764 → 217 MB，速度不变。

## 与 CosyVoice3 对比

| | Qwen3-TTS（本项目）| [CosyVoice3](https://github.com/nickovchinnikov/cosyvoice3-axera-russian) |
|---|---|---|
| RTF | 2.0x | **1.26x** |
| 架构 | Talker → CP → Vocoder | LLM → Flow DiT → HiFT |
| NPU (AX650N) | 未使用 | Flow DiT 运行在 NPU |

CosyVoice3 更快，因为 Flow DiT（MatMul+Attention）运行在 AX650N NPU 上。Qwen3-TTS 没有*适合量化*的 NPU 组件——声码器的 Snake 激活让 Q8/INT8 不可用——但未量化的 RKNN FP16 路径是无损的，所以声码器确实跑在 RKNN NPU 上（见上文）。

## 支持的语言

中文、英文、德文、俄文、法文、日文、韩文

## 许可证

Apache-2.0。模型权重遵循 [Qwen3-TTS 许可证](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)。
