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
         │    ONNX FP32 (A76, 4线程)           │
         │    分块 overlap-crossfade            │
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

### 优化历程

| 版本 | Code Predictor | 声码器 | RTF |
|---|---|---|---|
| 基线 (RKLLM talker) | ONNX 单线程 (374ms) | RKNN Q8 | 5.5x |
| + GGML Q4_0 CP | **GGML (55ms)** | RKNN Q8 | 2.2x |
| + ONNX FP32 声码器 | GGML (55ms) | **ONNX FP32** | **2.0x** |

### 为什么使用 ONNX FP32 声码器？

声码器使用 **Snake 激活函数** (`x + sin²(α·x)/α`)，与量化不兼容:
- **RKNN Q8**: 快 1.75 倍但有可听噪声
- **ONNX INT8**: 慢 3 倍且质量差
- **AX650N NPU**: 不支持 Conv1D dilation=9
- **MNN GPU**: SineGen 形状广播错误

所有量化路径已穷尽 — FP32 是唯一可接受的方案。

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

## 与 CosyVoice3 对比

| | Qwen3-TTS（本项目）| [CosyVoice3](https://github.com/nickovchinnikov/cosyvoice3-axera-russian) |
|---|---|---|
| RTF | 2.0x | **1.26x** |
| 架构 | Talker → CP → Vocoder | LLM → Flow DiT → HiFT |
| NPU (AX650N) | 未使用 | Flow DiT 运行在 NPU |

CosyVoice3 更快，因为 Flow DiT（MatMul+Attention）运行在 AX650N NPU 上。Qwen3-TTS 没有适合 NPU 的组件。

## 支持的语言

中文、英文、德文、俄文、法文、日文、韩文

## 许可证

Apache-2.0。模型权重遵循 [Qwen3-TTS 许可证](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)。
