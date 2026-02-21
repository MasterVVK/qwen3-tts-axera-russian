# Qwen3-TTS на CM3588 — RTF 2.0x (llama.cpp + GGML)

Высокопроизводительный C++/Python пайплайн для [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) на FriendlyElec CM3588 (RK3588).

**Достигнут RTF 2.0x** — самый быстрый Qwen3-TTS на edge-железе (vs RTF 5.5x на RKLLM).

## Архитектура

```
                   ┌─────────────────────┐
Текст ─────────────┤  Токенизатор + Embed├──→ эмбеддинги
                   └────────┬────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │    llamacpp_talker_server.py         │
         │    llama.cpp GGUF Q4_K_M (A76)      │
         │    ~30-35 ток/с, ~30мс/шаг           │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    code_pred_server (GGML Q4_0)     │
         │    в 4.8x быстрее ONNX              │
         │    ~55мс/шаг                         │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    vocoder_server.py                 │
         │    ONNX FP32 (A76, 4 потока)        │
         │    overlap-crossfade для чанков      │
         └──────────────────────────────────────┘
```

Три независимых сервера общаются через Unix-сокеты. Каждый можно перезапустить отдельно.

## Производительность

Тестировано на FriendlyElec CM3588 (RK3588 4xA76+4xA55, 32GB RAM):

| Компонент | Время | Примечание |
|---|---|---|
| Talker (llama.cpp A76) | ~30мс/шаг | GGUF Q4_K_M, 469 МБ |
| Code Predictor (GGML Q4_0) | ~55мс/шаг | **в 4.8x быстрее ONNX** |
| Вокодер (ONNX FP32) | ~5.0с/64-tok | Overlap-crossfade |
| **Итого RTF** | | **2.0x** |

### Путь оптимизации

| Версия | Code Predictor | Вокодер | RTF |
|---|---|---|---|
| Базовая (RKLLM talker) | ONNX 1-поток (374мс) | RKNN Q8 | 5.5x |
| + llama.cpp talker | ONNX 1-поток (374мс) | RKNN Q8 | 5.5x |
| + 2-поточный ONNX CP | ONNX 2-потока (265мс) | RKNN Q8 | 4.4x |
| + GGML Q4_0 CP | **GGML (55мс)** | RKNN Q8 | 2.2x |
| + ONNX FP32 вокодер | GGML (55мс) | **ONNX FP32** | **2.0x** |

### Почему ONNX FP32 вокодер, а не RKNN Q8?

Вокодер использует **Snake activation** (`x + sin²(α·x)/α`), которая несовместима с квантизацией:
- **RKNN Q8**: быстрее в 1.75x, но слышен шум (SNR 9.5 дБ)
- **ONNX INT8**: в 3x *медленнее* + плохое качество (SNR 4.2 дБ)
- **AX650N NPU**: Conv1D dilation=9 не поддерживается
- **MNN GPU**: ошибка broadcast в SineGen

Все пути квантизации исчерпаны — FP32 единственный вариант с приемлемым качеством.

## Быстрый старт

```bash
# Одиночный синтез
bash dual_npu/launch_qwen3_tts.sh "Привет, как дела?"

# Режим демона (постоянные серверы)
bash dual_npu/launch_qwen3_tts.sh --daemon

# Клиент (при работающем демоне)
python3 dual_npu/tts_client.py "Текст для синтеза" --language russian

# Со стриминг-вокодером (параллельно с генерацией)
python3 dual_npu/tts_client.py "Длинный текст..." --language russian --streaming
```

## Варианты Code Predictor

Пайплайн поддерживает три бэкенда (авто-выбор):

| Бэкенд | Скорость | Качество | Примечание |
|---|---|---|---|
| **GGML Q4_0** (по умолч.) | 55мс/шаг | Лучшее | Требует [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp) |
| C++ ONNX Runtime | 240мс/шаг | Хорошее | Встроен, `dual_npu/code_predictor_cpp/` |
| Python ONNX Runtime | 248мс/шаг | Хорошее | Fallback, без компиляции |

## Подготовка моделей

1. Извлечь talker как Qwen3: `python3 scripts/extract_talker_as_qwen3.py`
2. Конвертировать в GGUF: `python3 scripts/convert_talker_gguf.py`
3. Извлечь эмбеддинги: `python3 scripts/extract_embeddings.py`
4. Экспортировать code predictor: `python3 scripts/export_code_predictor_weights.py`
5. Экспортировать вокодер: `python3 scripts/export_vocoder_traced.py`
6. (Опционально) Конвертировать GGML code predictor: [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp)

См. [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md).

## Сравнение с CosyVoice3

| | Qwen3-TTS (этот проект) | [CosyVoice3](https://github.com/nickovchinnikov/cosyvoice3-axera-russian) |
|---|---|---|
| RTF | 2.0x | **1.26x** |
| Архитектура | Talker → CP → Vocoder | LLM → Flow DiT → HiFT |
| NPU (AX650N) | Не используется | Flow DiT на NPU |
| Качество | Хорошее | Хорошее |

CosyVoice3 быстрее, потому что Flow DiT (MatMul+Attention) работает на AX650N NPU, освобождая CPU. В Qwen3-TTS нет компонентов, подходящих для NPU — вокодер использует Snake activation, несовместимую с квантизацией.

## Поддерживаемые языки

Китайский, английский, немецкий, русский, французский, японский, корейский

## Лицензия

Apache-2.0. Модели: [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
