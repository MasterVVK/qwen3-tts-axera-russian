# Qwen3-TTS на CM3588 (llama.cpp + AX650N)

Гибридный C++/Python пайплайн для [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) на CM3588 с llama.cpp для LLM и опциональным ускорением на AX650N NPU.

## Архитектура

```
Текст → Токенизатор → Эмбеддинги → llamacpp_talker (A76, GGUF)
     → hidden + code_0 → code_predictor (ONNX) → codes 1-15
     → vocoder (RKNN/AX650N) → аудио 24kHz
```

## Преимущество llama.cpp

Talker — 28-слойный Qwen3 transformer. llama.cpp даёт:
- **~30-35 ток/с** на A76 (vs ~17 ток/с RKLLM)
- Q4_K_M квантизация (469 МБ vs 887 МБ)
- Persistence KV-cache для повторных спикеров

## Ожидаемая производительность

| Компонент | Время/шаг | RTF |
|---|---|---|
| RKLLM + ONNX-1t + RKNN | 434мс | 6.2x |
| **llama.cpp + ONNX-1t + RKNN** | **~404мс** | **~5.6x** |
| llama.cpp + ONNX-4t + RKNN | ~230мс | ~3.2x |

## Быстрый старт

```bash
# Синтез
bash dual_npu/launch_qwen3_tts.sh "Привет, как дела?"

# Режим демона
bash dual_npu/launch_qwen3_tts.sh --daemon
python3 dual_npu/tts_client.py "Текст"
```

## Лицензия

Apache-2.0. Модели: [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
