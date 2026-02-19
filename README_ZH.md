# Qwen3-TTS 在 CM3588 上运行 (llama.cpp + AX650N)

混合 C++/Python 管道，用于在 CM3588 上运行 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)，使用 llama.cpp 运行 talker LLM，可选 AX650N NPU 加速。

## 架构

```
文本 → 分词器 → 嵌入 → llamacpp_talker (A76, GGUF)
     → hidden + code_0 → code_predictor (ONNX) → codes 1-15
     → vocoder (RKNN/AX650N) → 24kHz 音频
```

## 快速开始

```bash
bash dual_npu/launch_qwen3_tts.sh "你好，今天天气怎么样？"
```

## 许可证

Apache-2.0。模型权重遵循 [Qwen3-TTS 许可证](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)。
