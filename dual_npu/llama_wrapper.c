/*
 * llama_wrapper.c -- Thin C wrapper around llama.cpp for Python ctypes.
 *
 * Avoids returning structs by value (which causes segfaults with ctypes on aarch64).
 * All functions take/return simple types (int, float, pointers).
 *
 * Used for Qwen3-TTS Talker LLM (28-layer Qwen3, embedding mode).
 * Returns hidden states (embeddings from last layer) instead of logits.
 *
 * Build on CM3588:
 *   gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
 *       -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
 *       -L/root/llama.cpp/build/bin \
 *       -lllama -lggml -lggml-base -lggml-cpu \
 *       -Wl,-rpath,/root/llama.cpp/build/bin
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "llama.h"

/* ---- Model loading ---- */

void wrapper_backend_init(void) {
    llama_backend_init();
}

void wrapper_backend_free(void) {
    llama_backend_free();
}

struct llama_model* wrapper_load_model(const char* path, int n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, params);
}

void wrapper_free_model(struct llama_model* model) {
    if (model) llama_model_free(model);
}

int wrapper_model_n_embd(const struct llama_model* model) {
    return llama_model_n_embd(model);
}

/* ---- Context ---- */

struct llama_context* wrapper_create_context(
    struct llama_model* model,
    int n_ctx,
    int n_batch,
    int n_threads,
    int embeddings
) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = (uint32_t)n_ctx;
    params.n_batch = (uint32_t)n_batch;
    params.n_ubatch = (uint32_t)n_batch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.embeddings = embeddings ? true : false;

    struct llama_context* ctx = llama_init_from_model(model, params);
    if (ctx && embeddings) {
        llama_set_embeddings(ctx, true);
    }
    return ctx;
}

void wrapper_free_context(struct llama_context* ctx) {
    if (ctx) llama_free(ctx);
}

/* ---- KV cache ---- */

void wrapper_kv_clear(struct llama_context* ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, true);
}

/* ---- KV state persistence ---- */

size_t wrapper_state_get_size(struct llama_context* ctx) {
    return llama_state_get_size(ctx);
}

int wrapper_state_save_file(struct llama_context* ctx, const char* path) {
    bool ok = llama_state_save_file(ctx, path, NULL, 0);
    if (!ok) {
        fprintf(stderr, "wrapper_state_save_file: failed to save to %s\n", path);
        return -1;
    }
    return 0;
}

int wrapper_state_load_file(struct llama_context* ctx, const char* path) {
    /* Must provide valid token buffer even if we don't use tokens.
     * Passing NULL causes segfault! */
    llama_token dummy_tokens[1];
    size_t n_token_count = 0;
    bool ok = llama_state_load_file(ctx, path, dummy_tokens, 0, &n_token_count);
    if (!ok) {
        fprintf(stderr, "wrapper_state_load_file: failed to load from %s\n", path);
        return -1;
    }
    fprintf(stderr, "wrapper_state_load_file: loaded from %s, n_tokens=%zu\n", path, n_token_count);
    return 0;
}

/* ---- Core: decode with custom embeddings, return hidden state ---- */

/*
 * wrapper_decode_embd:
 *   Feed custom embeddings into the model and extract hidden state of last token.
 *
 *   embd_data:   float array [n_tokens * n_embd], row-major
 *   n_tokens:    number of tokens
 *   n_embd:      embedding dimension (must match model)
 *   pos_start:   starting position in KV cache
 *   out_hidden:  float array [n_embd] to write result into (caller-allocated)
 *
 *   Returns: 0 on success, negative on error
 */
int wrapper_decode_embd(
    struct llama_context* ctx,
    const float* embd_data,
    int n_tokens,
    int n_embd,
    int pos_start,
    float* out_hidden
) {
    struct llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;

    memcpy(batch.embd, embd_data, (size_t)n_tokens * n_embd * sizeof(float));

    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i] = pos_start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "wrapper_decode_embd: llama_decode failed with %d\n", ret);
        llama_batch_free(batch);
        return -1;
    }

    float* emb = llama_get_embeddings_ith(ctx, n_tokens - 1);
    if (!emb) {
        fprintf(stderr, "wrapper_decode_embd: llama_get_embeddings_ith returned NULL\n");
        llama_batch_free(batch);
        return -2;
    }

    memcpy(out_hidden, emb, (size_t)n_embd * sizeof(float));

    llama_batch_free(batch);
    return 0;
}
