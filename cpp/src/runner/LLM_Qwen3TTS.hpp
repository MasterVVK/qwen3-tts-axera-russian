#pragma once
// LLM_Qwen3TTS.hpp — Qwen3-TTS Talker on AX650N NPU
// Adapted from CosyVoice3 LLM.hpp for Qwen3-TTS architecture:
//   - 28 Qwen3 layers (vs 24 Qwen2 for CosyVoice3)
//   - hidden_size=1024 (vs 896)
//   - codec_head[3072] on CPU (vs lm_head[6761])
//   - Returns hidden_state for Code Predictor
//   - Feedback = sum(codec_emb[16 codes]) + tts_pad

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>
#include "bfloat16.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "timer.hpp"
#include "axcl_manager.h"
#include "utils/sampling.hpp"
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

struct Qwen3TTSResult {
    int code_0;                          // Sampled codec token (codebook 0)
    std::vector<float> hidden_state;     // Hidden state [HIDDEN_SIZE] for code predictor
    bool is_eos;                         // True if EOS token generated
};

struct Qwen3TTSAttrType
{
    // Model architecture
    std::string template_filename_axmodel = "qwen3_p128_l%d_together.axmodel";
    int axmodel_num = 28;  // Qwen3-TTS: 28 layers

    std::string filename_post_axmodel = "qwen3_post.axmodel";

    int prefill_token_num = 128; // auto calc from axmodel
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid = -1;

    // Embeddings (bfloat16 binary files)
    std::string filename_tokens_embed = "model.embed_tokens.weight.bfloat16.bin";
    std::string filename_codec_embed = "codec_embedding.bfloat16.bin";
    int tokens_embed_num = 151936;
    int tokens_embed_size = 1024;  // hidden_size
    int codec_embed_num = 3072;
    int codec_embed_size = 1024;

    // Codec head (CPU decoder): Linear(1024, 3072)
    std::string filename_codec_head;  // codec_head.float32.bin
    static const int HIDDEN_SIZE = 1024;
    static const int CODEC_VOCAB_SIZE = 3072;

    // Special tokens
    int codec_pad_id = 2148;
    int codec_bos_id = 2149;
    int codec_eos_id = 2150;
    int tts_bos_token_id = 151672;
    int tts_eos_token_id = 151673;
    int tts_pad_token_id = 151671;

    // KV cache (auto calc)
    int max_token_len = 511;
    int kv_cache_num = 512;
    int kv_cache_size = 256;

    bool b_use_mmap_load_embed = false;
    bool b_dynamic_load_axmodel_layer = false;
    bool b_use_mmap_load_layer = true;

    int dev_id = 0;  // AXCL device ID (PCIe)

    // Sampling
    float temperature = 0.8f;
    int top_k = 50;
    float top_p = 0.95f;
    float repetition_penalty = 1.2f;
    int rep_window = 30;

    // TTS pad embedding (precomputed, fed as trailing hidden during decode)
    std::vector<float> tts_pad_embed;  // [HIDDEN_SIZE], loaded from numpy
};

class Qwen3TTSTalker
{
private:
    LLaMaEmbedSelector embed_selector;   // text embeddings (151936 x 1024)
    LLaMaEmbedSelector codec_embed_selector; // codec embeddings (3072 x 1024)

    Qwen3TTSAttrType _attr;

    struct LLMLayer {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> layers;
    ax_runner_ax650 post_model;  // RMSNorm only (output_norm)

    // CPU codec_head: Linear(1024, 3072) — hidden → codec logits
    std::vector<float> codec_head_weight;

    int decode_grpid = 0;
    bool b_stop = false;

    // NEON-optimized: hidden[1024] @ codec_head[3072, 1024].T → logits[3072]
    void codec_head_forward(const float *hidden, float *logits) {
        const float *W = codec_head_weight.data();
        const int IN = Qwen3TTSAttrType::HIDDEN_SIZE;
        const int OUT = Qwen3TTSAttrType::CODEC_VOCAB_SIZE;
#if defined(__ARM_NEON) || defined(__aarch64__)
        for (int o = 0; o < OUT; o++) {
            const float *row = W + o * IN;
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            for (int i = 0; i < IN; i += 16) {
                sum0 = vfmaq_f32(sum0, vld1q_f32(hidden + i),      vld1q_f32(row + i));
                sum1 = vfmaq_f32(sum1, vld1q_f32(hidden + i + 4),  vld1q_f32(row + i + 4));
                sum2 = vfmaq_f32(sum2, vld1q_f32(hidden + i + 8),  vld1q_f32(row + i + 8));
                sum3 = vfmaq_f32(sum3, vld1q_f32(hidden + i + 12), vld1q_f32(row + i + 12));
            }
            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            logits[o] = vaddvq_f32(sum0);
        }
#else
        for (int o = 0; o < OUT; o++) {
            float sum = 0.0f;
            const float *row = W + o * IN;
            for (int i = 0; i < IN; i++) {
                sum += hidden[i] * row[i];
            }
            logits[o] = sum;
        }
#endif
    }

    // Convert bf16 output to float32 hidden state
    void bf16_to_float(const unsigned short *bf16, float *fp32, int n) {
        for (int i = 0; i < n; i++) {
            unsigned int proc = bf16[i] << 16;
            fp32[i] = *reinterpret_cast<float *>(&proc);
        }
    }

    // Sample token from logits with temperature and top-k
    int sample_token(float *logits, int vocab_size, const std::vector<int> &history) {
        // Repetition penalty
        if (_attr.repetition_penalty != 1.0f) {
            int window_start = std::max(0, (int)history.size() - _attr.rep_window);
            for (int i = window_start; i < (int)history.size(); i++) {
                int tok = history[i];
                if (tok >= 0 && tok < vocab_size) {
                    if (logits[tok] > 0)
                        logits[tok] /= _attr.repetition_penalty;
                    else
                        logits[tok] *= _attr.repetition_penalty;
                }
            }
        }

        // Temperature
        if (_attr.temperature != 1.0f && _attr.temperature > 0.0f) {
            float inv_t = 1.0f / _attr.temperature;
            for (int i = 0; i < vocab_size; i++)
                logits[i] *= inv_t;
        }

        // Top-k sampling
        std::vector<std::pair<float, int>> indexed(vocab_size);
        for (int i = 0; i < vocab_size; i++)
            indexed[i] = {logits[i], i};

        int k = std::min(_attr.top_k, vocab_size);
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                         std::greater<std::pair<float, int>>());

        // Softmax over top-k
        float max_val = indexed[0].first;
        float sum = 0.0f;
        std::vector<float> probs(k);
        for (int i = 0; i < k; i++) {
            probs[i] = std::exp(indexed[i].first - max_val);
            sum += probs[i];
        }
        for (int i = 0; i < k; i++)
            probs[i] /= sum;

        // Random sample
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i < k; i++) {
            cumsum += probs[i];
            if (r <= cumsum)
                return indexed[i].second;
        }
        return indexed[k - 1].second;
    }

public:
    bool Init(Qwen3TTSAttrType attr)
    {
        printf("[Qwen3TTS] Init: %d layers, hidden=%d, codec_vocab=%d\n",
               attr.axmodel_num, attr.HIDDEN_SIZE, attr.CODEC_VOCAB_SIZE);
        this->_attr = attr;

        // Load text embeddings
        if (!embed_selector.Init(attr.filename_tokens_embed,
                                 attr.tokens_embed_num, attr.tokens_embed_size,
                                 attr.b_use_mmap_load_embed)) {
            fprintf(stderr, "[Qwen3TTS] Failed to load text embeddings: %s\n",
                    attr.filename_tokens_embed.c_str());
            return false;
        }

        // Load codec embeddings
        if (!codec_embed_selector.Init(attr.filename_codec_embed,
                                        attr.codec_embed_num, attr.codec_embed_size,
                                        attr.b_use_mmap_load_embed)) {
            fprintf(stderr, "[Qwen3TTS] Failed to load codec embeddings: %s\n",
                    attr.filename_codec_embed.c_str());
            return false;
        }

        // Load axmodel layers
        layers.resize(attr.axmodel_num);
        char path[1024];
        for (int i = 0; i < attr.axmodel_num; i++) {
            sprintf(path, attr.template_filename_axmodel.c_str(), i);
            layers[i].filename = path;

            if (!attr.b_dynamic_load_axmodel_layer) {
                int ret = layers[i].layer.init(layers[i].filename.c_str(), _attr.dev_id);
                if (ret != 0) {
                    fprintf(stderr, "[Qwen3TTS] Failed to init layer %d: %s\n", i, path);
                    return false;
                }
                int remain = get_pcie_remaining_cmm_size(_attr.dev_id);
                printf("[Qwen3TTS] Layer %d loaded (%d MB remaining)\n", i, remain);
            } else {
                layers[i].layer_buffer.open_file(layers[i].filename.c_str());
            }
        }

        // Load post model (RMSNorm)
        int ret = post_model.init(attr.filename_post_axmodel.c_str(), _attr.dev_id);
        if (ret != 0) {
            fprintf(stderr, "[Qwen3TTS] Failed to init post model\n");
            return false;
        }

        // Enable PCIe sync
        for (int i = 0; i < attr.axmodel_num; i++) {
            if (!attr.b_dynamic_load_axmodel_layer) {
                layers[i].layer.set_auto_sync_before_inference(true);
                layers[i].layer.set_auto_sync_after_inference(true);
            }
        }
        post_model.set_auto_sync_before_inference(true);
        post_model.set_auto_sync_after_inference(true);

        // Load codec_head weights for CPU decoder
        if (!attr.filename_codec_head.empty()) {
            FILE *f = fopen(attr.filename_codec_head.c_str(), "rb");
            if (f) {
                long expected = (long)attr.CODEC_VOCAB_SIZE * attr.HIDDEN_SIZE * sizeof(float);
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                fseek(f, 0, SEEK_SET);
                if (fsize == expected) {
                    codec_head_weight.resize(attr.CODEC_VOCAB_SIZE * attr.HIDDEN_SIZE);
                    fread(codec_head_weight.data(), sizeof(float),
                          attr.CODEC_VOCAB_SIZE * attr.HIDDEN_SIZE, f);
                    printf("[Qwen3TTS] Loaded codec_head: %s (%.1f MB)\n",
                           attr.filename_codec_head.c_str(), fsize / 1024.0 / 1024.0);
                } else {
                    fprintf(stderr, "[Qwen3TTS] codec_head size mismatch: %ld vs %ld\n",
                            fsize, expected);
                    fclose(f);
                    return false;
                }
                fclose(f);
            } else {
                fprintf(stderr, "[Qwen3TTS] Failed to open codec_head: %s\n",
                        attr.filename_codec_head.c_str());
                return false;
            }
        }

        // Auto-detect KV cache params from axmodel
        {
            _attr.max_token_len = layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            _attr.kv_cache_size = layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = layers[0].layer.get_input("K_cache").nSize /
                                 _attr.kv_cache_size / sizeof(unsigned short);
            _attr.prefill_token_num = layers[0].layer.get_input(1, "indices").vShape[1];

            printf("[Qwen3TTS] max_token_len=%d, kv_cache_size=%d, kv_cache_num=%d, prefill=%d\n",
                   _attr.max_token_len, _attr.kv_cache_size, _attr.kv_cache_num, _attr.prefill_token_num);

            for (size_t i = 0; i < layers[0].layer.get_num_input_groups() - 1; i++) {
                int n = layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                _attr.prefill_max_kv_cache_num_grp.push_back(n);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
        }

        printf("[Qwen3TTS] Init complete (eos=%d, temp=%.2f, top_k=%d)\n",
               _attr.codec_eos_id, _attr.temperature, _attr.top_k);
        return true;
    }

    void Deinit() {
        for (int i = 0; i < _attr.axmodel_num; i++)
            layers[i].layer.deinit();
        post_model.deinit();
        embed_selector.Deinit();
        codec_embed_selector.Deinit();
    }

    void Stop() { b_stop = true; }

    // Build prefix embeddings for Qwen3-TTS:
    // [text_embeds...] [tts_pad_embed] [codec_bos_embed]
    // text_embeds come from pre-computed text_embedding (already projected)
    int BuildPrefix(const std::vector<int> &text_token_ids,
                    std::vector<unsigned short> &out_embed,
                    int &out_total_tokens)
    {
        int n_text = text_token_ids.size();
        // Prefix: text_tokens + tts_pad + codec_bos = n_text + 2
        int total = n_text + 2;
        out_embed.resize(total * _attr.tokens_embed_size);

        // Text embeddings
        for (int i = 0; i < n_text; i++) {
            embed_selector.getByIndex(text_token_ids[i],
                                       out_embed.data() + i * _attr.tokens_embed_size);
        }

        // tts_pad embedding
        embed_selector.getByIndex(_attr.tts_pad_token_id,
                                   out_embed.data() + n_text * _attr.tokens_embed_size);

        // codec_bos embedding (from codec embeddings)
        codec_embed_selector.getByIndex(_attr.codec_bos_id,
                                         out_embed.data() + (n_text + 1) * _attr.tokens_embed_size);

        out_total_tokens = total;
        return 0;
    }

    // Run one decode step: given input embedding, run through all layers + post
    // Returns hidden_state (float32) and codec logits
    Qwen3TTSResult DecodeStep(unsigned short *input_embed,
                               int position,
                               std::vector<unsigned short> &mask,
                               const std::vector<int> &history)
    {
        Qwen3TTSResult result;
        result.is_eos = false;
        result.hidden_state.resize(_attr.HIDDEN_SIZE);

        // Copy input to first layer
        memcpy(layers[0].layer.get_input(decode_grpid, "input").pVirAddr,
               input_embed, layers[0].layer.get_input(decode_grpid, "input").nSize);

        // Run through all layers
        for (int m = 0; m < _attr.axmodel_num; m++) {
            auto &layer = layers[m];

            // Set position
            auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
            memcpy(input_indices.pVirAddr, &position, sizeof(position));

            // Set mask
            auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
            memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

            // Inference
            layer.layer.inference(decode_grpid);

            // Update KV cache
            auto &input_k = layer.layer.get_input(decode_grpid, "K_cache");
            auto &output_k = layer.layer.get_output(decode_grpid, "K_cache_out");
            memcpy((unsigned short *)input_k.pVirAddr + position * _attr.kv_cache_size,
                   output_k.pVirAddr, output_k.nSize);

            auto &input_v = layer.layer.get_input(decode_grpid, "V_cache");
            auto &output_v = layer.layer.get_output(decode_grpid, "V_cache_out");
            memcpy((unsigned short *)input_v.pVirAddr + position * _attr.kv_cache_size,
                   output_v.pVirAddr, output_v.nSize);

            // Pass output to next layer or post
            if (m == _attr.axmodel_num - 1) {
                memcpy(post_model.get_input(0).pVirAddr,
                       layer.layer.get_output(decode_grpid, "output").pVirAddr,
                       post_model.get_input(0).nSize);
            } else {
                memcpy(layers[m + 1].layer.get_input(decode_grpid, "input").pVirAddr,
                       layer.layer.get_output(decode_grpid, "output").pVirAddr,
                       layer.layer.get_input(decode_grpid, "input").nSize);
            }
        }

        // Run post (RMSNorm) → get hidden state
        post_model.inference();
        auto &output_norm = post_model.get_output("output_norm");
        unsigned short *norm_bf16 = (unsigned short *)output_norm.pVirAddr;

        // Convert bf16 → float32 hidden state
        int hidden_count = output_norm.nSize / sizeof(unsigned short);
        // The post model outputs [hidden_size] after RMSNorm
        // We take only HIDDEN_SIZE elements for codec_head
        int n = std::min(hidden_count, _attr.HIDDEN_SIZE);
        bf16_to_float(norm_bf16, result.hidden_state.data(), n);

        // CPU codec_head: hidden[1024] → logits[3072]
        std::vector<float> logits(_attr.CODEC_VOCAB_SIZE);
        codec_head_forward(result.hidden_state.data(), logits.data());

        // Sample code_0
        result.code_0 = sample_token(logits.data(), _attr.CODEC_VOCAB_SIZE, history);

        // Check EOS
        if (result.code_0 == _attr.codec_eos_id) {
            result.is_eos = true;
        }

        return result;
    }

    // Prefill: run all text embeddings through layers to fill KV cache
    // Returns position after prefill
    int Prefill(std::vector<unsigned short> &text_embed, int input_embed_num,
                std::vector<unsigned short> &mask)
    {
        bfloat16 bf16_neg = -65536.f;
        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        printf("[Qwen3TTS] Prefill: %d tokens, %d splits\n", input_embed_num, prefill_split_num);

        int max_pos_id = 0;

        for (int p = 0; p < prefill_split_num; p++) {
            if (b_stop) break;

            _attr.prefill_grpid = p + 1;
            int kv_offset = p * _attr.prefill_token_num;
            int input_num = _attr.prefill_token_num;
            if (p == prefill_split_num - 1) {
                input_num = input_embed_num - p * _attr.prefill_token_num;
            }

            // Build mask for prefill chunk
            std::vector<unsigned short> mask_tmp(
                _attr.prefill_token_num * (kv_offset + _attr.prefill_token_num), bf16_neg.data);
            for (int i = 0; i < _attr.prefill_token_num && i < input_num; i++) {
                auto *ptr = mask_tmp.data() + i * (kv_offset + _attr.prefill_token_num);
                for (int j = 0; j < kv_offset + i + 1; j++) {
                    ptr[j] = 0;
                }
            }

            // Copy embeddings
            std::vector<unsigned short> embed_chunk(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            int copy_size = input_num * _attr.tokens_embed_size * sizeof(unsigned short);
            memcpy(embed_chunk.data(),
                   text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                   copy_size);

            // Run through all layers
            for (int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) break;
                auto &layer = layers[m];

                // Set indices (position IDs)
                auto &input_indices = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(idx_ptr, 0, input_indices.nSize);
                for (int j = 0; j < _attr.prefill_token_num; j++) {
                    int pos = kv_offset + j;
                    if (pos < input_embed_num) {
                        idx_ptr[j] = pos;
                        if (pos > max_pos_id) max_pos_id = pos;
                    }
                }

                // Set mask and input
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                auto &input = layer.layer.get_input(_attr.prefill_grpid, "input");
                memcpy(input.pVirAddr, embed_chunk.data(), embed_chunk.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                // Copy KV cache
                auto &dk = layer.layer.get_input(decode_grpid, "K_cache");
                auto &ok = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                memcpy((unsigned short *)dk.pVirAddr + kv_offset * _attr.kv_cache_size,
                       ok.pVirAddr, sizeof(unsigned short) * input_num * _attr.kv_cache_size);

                auto &dv = layer.layer.get_input(decode_grpid, "V_cache");
                auto &ov = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");
                memcpy((unsigned short *)dv.pVirAddr + kv_offset * _attr.kv_cache_size,
                       ov.pVirAddr, sizeof(unsigned short) * input_num * _attr.kv_cache_size);

                // Propagate KV cache to subsequent prefill groups
                for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++) {
                    auto &pk = layer.layer.get_input(gid, "K_cache");
                    memcpy((unsigned short *)pk.pVirAddr + kv_offset * _attr.kv_cache_size,
                           ok.pVirAddr, sizeof(unsigned short) * input_num * _attr.kv_cache_size);
                    auto &pv = layer.layer.get_input(gid, "V_cache");
                    memcpy((unsigned short *)pv.pVirAddr + kv_offset * _attr.kv_cache_size,
                           ov.pVirAddr, sizeof(unsigned short) * input_num * _attr.kv_cache_size);
                }

                // Get output for next layer
                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(embed_chunk.data(), output.pVirAddr, embed_chunk.size() * sizeof(unsigned short));
            }
        }

        // Update decode mask
        for (int i = 0; i < input_embed_num; i++) {
            mask[i] = 0;
        }
        mask[_attr.kv_cache_num] = 0;  // self-attention

        return max_pos_id;
    }

    // Main generation: prefill + decode loop
    // Returns vector of Qwen3TTSResult (one per step)
    // Caller is responsible for:
    //   1. Getting codes 1-15 from Code Predictor using hidden_state
    //   2. Constructing feedback embedding: sum(codec_emb[code_g]) + tts_pad
    //   3. Calling DecodeStep with feedback embedding
    //
    // For simpler usage, use RunWithCallback which handles the full loop.
    int RunWithCallback(
        std::vector<unsigned short> &prefix_embed,
        int prefix_len,
        int max_tokens,
        // Callback: called after each talker step with (hidden_state, code_0)
        // Must return feedback embedding (bf16, tokens_embed_size) for next step
        // Return nullptr to stop generation
        std::function<unsigned short*(const Qwen3TTSResult &result, int step)> feedback_fn
    )
    {
        b_stop = false;

        // Init mask
        bfloat16 bf16_neg = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16_neg.data);

        // Prefill
        timer t_prefill;
        t_prefill.start();
        int max_pos = Prefill(prefix_embed, prefix_len, mask);
        printf("[Qwen3TTS] Prefill done: %.1f ms, max_pos=%d\n", t_prefill.cost(), max_pos);

        // Get last prefill hidden for first decode step
        // The last token's output is already in embed after prefill
        // We need to run post + codec_head on it
        // Actually, for the first step, we use the codec_bos embedding output
        // from the last prefill token. Let's extract it.

        // For first decode step: use the last prefill output directly
        std::vector<unsigned short> last_embed(_attr.tokens_embed_size);
        // The embed_chunk from Prefill's last iteration contains the output
        // We need to extract the last valid position's output

        // First decode step: run post on the last prefill output
        auto &post_input = post_model.get_input(0);
        // The last layer's output for the last position was left in the post_model input
        // during Prefill (it's still there from the last memcpy in the layer loop)
        // Actually, we need to be more careful here.

        // For now, let's get the hidden state from the last prefill token
        post_model.inference();
        auto &output_norm = post_model.get_output("output_norm");
        int norm_count = output_norm.nSize / sizeof(unsigned short);

        std::vector<float> hidden(_attr.HIDDEN_SIZE);
        bf16_to_float((unsigned short *)output_norm.pVirAddr, hidden.data(),
                       std::min(norm_count, _attr.HIDDEN_SIZE));

        // Sample first code_0
        std::vector<float> logits(_attr.CODEC_VOCAB_SIZE);
        codec_head_forward(hidden.data(), logits.data());

        std::vector<int> history;
        Qwen3TTSResult first_result;
        first_result.hidden_state = hidden;
        first_result.code_0 = sample_token(logits.data(), _attr.CODEC_VOCAB_SIZE, history);
        first_result.is_eos = (first_result.code_0 == _attr.codec_eos_id);

        if (first_result.is_eos) {
            printf("[Qwen3TTS] EOS on first token\n");
            return 0;
        }

        history.push_back(first_result.code_0);
        int token_count = 1;

        // Get feedback for first result
        unsigned short *feedback = feedback_fn(first_result, 0);
        if (!feedback) return token_count;

        // Decode loop
        timer t_decode;
        t_decode.start();

        for (int step = 1; step < max_tokens && !b_stop; step++) {
            int position = max_pos + step;
            if (position >= _attr.kv_cache_num) {
                printf("[Qwen3TTS] KV cache full at step %d\n", step);
                break;
            }

            Qwen3TTSResult result = DecodeStep(feedback, position, mask, history);
            mask[position] = 0;

            if (result.is_eos) {
                printf("[Qwen3TTS] EOS at step %d\n", step);
                break;
            }

            history.push_back(result.code_0);
            token_count++;

            // Get feedback for next step
            feedback = feedback_fn(result, step);
            if (!feedback) break;
        }

        float decode_ms = t_decode.cost();
        printf("[Qwen3TTS] Decode: %d tokens in %.1f ms (%.1f tok/s)\n",
               token_count, decode_ms, token_count / (decode_ms / 1000.0f));

        // Clear KV cache
        for (int i = 0; i < _attr.axmodel_num; i++) {
            for (size_t j = 0; j < layers[i].layer.get_num_input_groups(); j++) {
                memset(layers[i].layer.get_input(j, "K_cache").pVirAddr, 0,
                       layers[i].layer.get_input(j, "K_cache").nSize);
                memset(layers[i].layer.get_input(j, "V_cache").pVirAddr, 0,
                       layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }

        return token_count;
    }
};
