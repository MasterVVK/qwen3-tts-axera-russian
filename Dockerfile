# syntax=docker/dockerfile:1
# qwen3-tts-axera-russian — optimized Qwen3-TTS board pipeline (aarch64): llama.cpp GGUF
# talker + GGML code predictor servers, vocoder RKNN INT8 (or Axera .axmodel).
#
# The NPU runtime libs are built out-of-tree on the board (see README). Copy them into
# third_party/ before building:
#   llama_wrapper.so     # dual_npu/llama_wrapper.c in this repo
#   libllama.so libggml.so libggml-base.so libggml-cpu.so   # llama.cpp aarch64 build
#   librknnrt.so         # rknn-toolkit2: rknpu2/runtime/Linux/librknn_api/aarch64/
#
#   docker buildx build --platform linux/arm64 -t qwen3-tts-axera .
#   docker run --rm -it -v /root/tts-rknn:/models \
#     --device /dev/rknpu0 --device /dev/rknpu1 --device /dev/rknpu2 \
#     qwen3-tts-axera bash   # start the servers per README, then: python3 dual_npu/tts_client.py "Text"

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir numpy onnxruntime

COPY . .

# Board runtime libs (see header for the source of each).
COPY third_party/*.so /usr/local/lib/
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib
CMD ["bash"]
