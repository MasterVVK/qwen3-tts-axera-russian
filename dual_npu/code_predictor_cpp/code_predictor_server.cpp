/**
 * C++ Code Predictor Server for Qwen3-TTS
 *
 * Drop-in replacement for Python code_predictor_server.py
 * Same Unix socket protocol: recv 4096+4 bytes -> send 60 bytes
 *
 * Optimizations over Python version:
 * - Zero Python interpreter overhead
 * - Pre-allocated I/O tensors (no numpy allocation per step)
 * - NEON SIMD for lm_head projection (15 × [1024] @ [2048, 1024])
 * - Move semantics for KV cache (no copies)
 */

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <csignal>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "npy_reader.h"

// ============================================================
// Constants
// ============================================================
constexpr int HIDDEN_SIZE = 1024;
constexpr int NUM_LAYERS = 5;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int NUM_GROUPS = 15;

// ============================================================
// Globals
// ============================================================
static volatile sig_atomic_t g_running = 1;
static void signal_handler(int) { g_running = 0; }

// ============================================================
// NEON-optimized matmul: hidden[H] @ weight[vocab, H] -> logits[vocab]
// ============================================================
static void matmul_neon(const float* hidden, const float* weight,
                        float* logits, int vocab, int hdim) {
#ifdef __aarch64__
    for (int o = 0; o < vocab; o++) {
        const float* row = weight + (size_t)o * hdim;
        float32x4_t s0 = vdupq_n_f32(0.0f);
        float32x4_t s1 = vdupq_n_f32(0.0f);
        float32x4_t s2 = vdupq_n_f32(0.0f);
        float32x4_t s3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 16 <= hdim; i += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(hidden + i),      vld1q_f32(row + i));
            s1 = vfmaq_f32(s1, vld1q_f32(hidden + i + 4),  vld1q_f32(row + i + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(hidden + i + 8),  vld1q_f32(row + i + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(hidden + i + 12), vld1q_f32(row + i + 12));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
        for (; i < hdim; i++) sum += hidden[i] * row[i];
        logits[o] = sum;
    }
#else
    for (int o = 0; o < vocab; o++) {
        const float* row = weight + (size_t)o * hdim;
        float sum = 0;
        for (int i = 0; i < hdim; i++) sum += hidden[i] * row[i];
        logits[o] = sum;
    }
#endif
}

// ============================================================
// Socket helpers
// ============================================================
static bool recv_exact(int fd, void* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, (char*)buf + got, n - got, 0);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

static bool send_exact(int fd, const void* buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = send(fd, (const char*)buf + sent, n - sent, 0);
        if (r <= 0) return false;
        sent += r;
    }
    return true;
}

// ============================================================
// Code Predictor
// ============================================================
struct CodePredictor {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "cp"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    // Model I/O names
    std::vector<std::string> input_name_strs;
    std::vector<const char*> input_names;
    std::vector<std::string> output_name_strs;
    std::vector<const char*> output_names;
    size_t num_outputs = 0;

    // Embeddings and projection heads
    std::vector<float> codec_embedding;    // [talker_vocab, 1024]
    int talker_vocab = 0;
    std::vector<std::vector<float>> codec_embeddings; // 15 × [code_vocab, 1024]
    std::vector<std::vector<float>> lm_heads;         // 15 × [code_vocab, 1024]
    int code_vocab = 0;

    // Sampling
    float temperature = 0.1f;
    int top_k = 50;
    std::mt19937 rng{42};

    // Pre-allocated buffers
    std::vector<float> logits_buf;

    void load(const std::string& model_path,
              const std::string& weights_dir,
              const std::string& codec_emb_path,
              int num_threads) {
        // Load ONNX model
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(num_threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        printf("Loading ONNX model: %s (threads=%d)\n", model_path.c_str(), num_threads);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);

        // Get I/O names
        Ort::AllocatorWithDefaultOptions alloc;
        size_t num_inputs = session->GetInputCount();
        num_outputs = session->GetOutputCount();

        // Collect names first, then build pointer arrays
        // (can't take c_str() while vector is still growing — realloc invalidates pointers)
        input_name_strs.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session->GetInputNameAllocated(i, alloc);
            input_name_strs.push_back(name.get());
        }
        for (auto& s : input_name_strs) input_names.push_back(s.c_str());

        output_name_strs.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session->GetOutputNameAllocated(i, alloc);
            output_name_strs.push_back(name.get());
        }
        for (auto& s : output_name_strs) output_names.push_back(s.c_str());

        printf("  Inputs (%zu):", num_inputs);
        for (auto& n : input_name_strs) printf(" %s", n.c_str());
        printf("\n  Outputs (%zu):", num_outputs);
        for (auto& n : output_name_strs) printf(" %s", n.c_str());
        printf("\n");

        // Load talker codec embedding
        printf("Loading codec embedding: %s\n", codec_emb_path.c_str());
        auto ce = load_npy(codec_emb_path.c_str());
        talker_vocab = ce.shape[0];
        codec_embedding = std::move(ce.data);
        printf("  codec_embedding: [%lld, %lld]\n", (long long)ce.shape[0], (long long)ce.shape[1]);

        // Load CP embeddings and lm_heads from extracted weights
        codec_embeddings.resize(NUM_GROUPS);
        lm_heads.resize(NUM_GROUPS);

        for (int g = 0; g < NUM_GROUPS; g++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/codec_emb_%d.npy", weights_dir.c_str(), g);
            auto emb = load_npy(path);
            if (g == 0) {
                code_vocab = emb.shape[0];
                logits_buf.resize(code_vocab);
            }
            codec_embeddings[g] = std::move(emb.data);

            snprintf(path, sizeof(path), "%s/lm_head_%d.npy", weights_dir.c_str(), g);
            auto lm = load_npy(path);
            lm_heads[g] = std::move(lm.data);

            if (g == 0) printf("  codec_emb/lm_head shape: [%lld, %lld]\n",
                               (long long)emb.shape[0], (long long)emb.shape[1]);
        }
        printf("  Loaded %d groups, code_vocab=%d\n", NUM_GROUPS, code_vocab);
    }

    int sample(const float* logits, int vocab_size) {
        // Top-K selection via partial sort
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        if (top_k < vocab_size) {
            std::nth_element(indices.begin(), indices.end() - top_k, indices.end(),
                [logits](int a, int b) { return logits[a] < logits[b]; });
        }

        int start = (top_k < vocab_size) ? vocab_size - top_k : 0;
        int count = vocab_size - start;

        // Find max for numerical stability
        float max_val = -1e30f;
        for (int i = start; i < vocab_size; i++) {
            float v = logits[indices[i]];
            if (v > max_val) max_val = v;
        }

        // Softmax with temperature
        float inv_temp = 1.0f / std::max(temperature, 1e-6f);
        float sum = 0;
        std::vector<float> probs(count);
        for (int i = 0; i < count; i++) {
            probs[i] = std::exp((logits[indices[start + i]] - max_val) * inv_temp);
            sum += probs[i];
        }
        float inv_sum = 1.0f / sum;
        for (auto& p : probs) p *= inv_sum;

        // Weighted random sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumsum = 0;
        for (int i = 0; i < count; i++) {
            cumsum += probs[i];
            if (r < cumsum) return indices[start + i];
        }
        return indices[vocab_size - 1];
    }

    Ort::Value make_tensor(float* data, size_t count, const int64_t* shape, size_t ndim) {
        return Ort::Value::CreateTensor<float>(mem_info, data, count, shape, ndim);
    }

    bool batch_prefill = true; // Batch 2 prefill tokens in 1 call (saves ~15ms, cos_sim~0.98)

    std::vector<int32_t> predict(const float* hidden_state, int code_0) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // Prepare code_0 embedding from talker table
        std::vector<float> code0_emb(HIDDEN_SIZE);
        if (code_0 >= 0 && code_0 < talker_vocab) {
            memcpy(code0_emb.data(), &codec_embedding[code_0 * HIDDEN_SIZE],
                   HIDDEN_SIZE * sizeof(float));
        }

        // Initialize empty KV caches
        std::vector<Ort::Value> kv_values;
        float dummy_kv = 0;
        for (int i = 0; i < NUM_LAYERS; i++) {
            int64_t empty_shape[] = {1, NUM_KV_HEADS, 0, HEAD_DIM};
            kv_values.push_back(Ort::Value::CreateTensor<float>(
                mem_info, &dummy_kv, 0, empty_shape, 4));
            kv_values.push_back(Ort::Value::CreateTensor<float>(
                mem_info, &dummy_kv, 0, empty_shape, 4));
        }

        std::vector<float> h0(HIDDEN_SIZE);

        if (batch_prefill) {
            // --- Batch prefill: hidden_state + code_0 in one call ---
            std::vector<float> h_batch(2 * HIDDEN_SIZE);
            memcpy(h_batch.data(), hidden_state, HIDDEN_SIZE * sizeof(float));
            memcpy(h_batch.data() + HIDDEN_SIZE, code0_emb.data(), HIDDEN_SIZE * sizeof(float));

            int64_t h_shape[] = {1, 2, HIDDEN_SIZE};
            int64_t p_shape[] = {2};
            int64_t pos_vals[] = {0, 1};

            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info, h_batch.data(), 2 * HIDDEN_SIZE, h_shape, 3));
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, pos_vals, 2, p_shape, 1));
            for (auto& kv : kv_values) inputs.push_back(std::move(kv));

            auto outputs = session->Run(Ort::RunOptions{nullptr},
                input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), num_outputs);

            // Extract last token's hidden output
            float* h_out = outputs[0].GetTensorMutableData<float>();
            memcpy(h0.data(), h_out + HIDDEN_SIZE, HIDDEN_SIZE * sizeof(float));

            kv_values.clear();
            for (int i = 0; i < NUM_LAYERS * 2; i++) {
                kv_values.push_back(std::move(outputs[1 + i]));
            }
        } else {
            // --- Sequential prefill ---
            memcpy(h0.data(), hidden_state, HIDDEN_SIZE * sizeof(float));
            {
                int64_t h_shape[] = {1, 1, HIDDEN_SIZE};
                int64_t p_shape[] = {1};
                int64_t pos_val = 0;

                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, h0.data(), HIDDEN_SIZE, h_shape, 3));
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem_info, &pos_val, 1, p_shape, 1));
                for (auto& kv : kv_values) inputs.push_back(std::move(kv));

                auto outputs = session->Run(Ort::RunOptions{nullptr},
                    input_names.data(), inputs.data(), inputs.size(),
                    output_names.data(), num_outputs);

                float* h_out = outputs[0].GetTensorMutableData<float>();
                memcpy(h0.data(), h_out, HIDDEN_SIZE * sizeof(float));

                kv_values.clear();
                for (int i = 0; i < NUM_LAYERS * 2; i++) {
                    kv_values.push_back(std::move(outputs[1 + i]));
                }
            }
            {
                int64_t h_shape[] = {1, 1, HIDDEN_SIZE};
                int64_t p_shape[] = {1};
                int64_t pos_val = 1;

                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, code0_emb.data(), HIDDEN_SIZE, h_shape, 3));
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem_info, &pos_val, 1, p_shape, 1));
                for (auto& kv : kv_values) inputs.push_back(std::move(kv));

                auto outputs = session->Run(Ort::RunOptions{nullptr},
                    input_names.data(), inputs.data(), inputs.size(),
                    output_names.data(), num_outputs);

                float* h_out = outputs[0].GetTensorMutableData<float>();
                memcpy(h0.data(), h_out, HIDDEN_SIZE * sizeof(float));

                kv_values.clear();
                for (int i = 0; i < NUM_LAYERS * 2; i++) {
                    kv_values.push_back(std::move(outputs[1 + i]));
                }
            }
        }

        // --- Project and sample group 0 ---
        std::vector<int32_t> predicted(NUM_GROUPS);
        matmul_neon(h0.data(), lm_heads[0].data(), logits_buf.data(), code_vocab, HIDDEN_SIZE);
        predicted[0] = sample(logits_buf.data(), code_vocab);

        // --- Decode groups 1-14 ---
        std::vector<float> embed(HIDDEN_SIZE);
        for (int step = 1; step < NUM_GROUPS; step++) {
            // Look up embedding of previous predicted token (CP's table, not talker's)
            int prev_token = predicted[step - 1];
            if (prev_token >= 0 && prev_token < code_vocab) {
                memcpy(embed.data(),
                       &codec_embeddings[step - 1][(size_t)prev_token * HIDDEN_SIZE],
                       HIDDEN_SIZE * sizeof(float));
            } else {
                memset(embed.data(), 0, HIDDEN_SIZE * sizeof(float));
            }

            int64_t h_shape[] = {1, 1, HIDDEN_SIZE};
            int64_t p_shape[] = {1};
            int64_t pos_val = step + 1;

            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info, embed.data(), HIDDEN_SIZE, h_shape, 3));
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, &pos_val, 1, p_shape, 1));
            for (auto& kv : kv_values) inputs.push_back(std::move(kv));

            auto outputs = session->Run(Ort::RunOptions{nullptr},
                input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), num_outputs);

            float* h_out = outputs[0].GetTensorMutableData<float>();

            // NEON lm_head projection
            matmul_neon(h_out, lm_heads[step].data(), logits_buf.data(), code_vocab, HIDDEN_SIZE);
            predicted[step] = sample(logits_buf.data(), code_vocab);

            kv_values.clear();
            for (int i = 0; i < NUM_LAYERS * 2; i++) {
                kv_values.push_back(std::move(outputs[1 + i]));
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        int ort_calls = batch_prefill ? (1 + NUM_GROUPS - 1) : (2 + NUM_GROUPS - 1);
        printf("  predict: %.1fms (%d ORT calls, %.1fms/call, batch=%d)\n",
               ms, ort_calls, ms / ort_calls, batch_prefill ? 1 : 0);

        return predicted;
    }
};

// ============================================================
// Server
// ============================================================
static int create_server_socket(const char* path) {
    unlink(path);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return -1;
    }
    chmod(path, 0666);

    if (listen(fd, 4) < 0) {
        perror("listen"); close(fd); return -1;
    }
    return fd;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --model PATH       ONNX model (default: /root/tts-rknn/code_predictor/code_predictor_decode_step.onnx)\n"
           "  --weights DIR      Extracted weights dir (default: /root/tts-rknn/code_predictor/weights_npy)\n"
           "  --codec_emb PATH   Talker codec embedding (default: /root/tts-rknn/embeddings/codec_embedding.npy)\n"
           "  --socket PATH      Unix socket path (default: /tmp/qwen3_cp.sock)\n"
           "  --threads N        ONNX threads (default: 3)\n"
           "  --temperature F    Sampling temperature (default: 0.1)\n"
           "  --top_k N          Top-K sampling (default: 50)\n",
           prog);
}

int main(int argc, char** argv) {
    // Disable stdout buffering for SSH/pipe
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string model_path = "/root/tts-rknn/code_predictor/code_predictor_decode_step.onnx";
    std::string weights_dir = "/root/tts-rknn/code_predictor/weights_npy";
    std::string codec_emb_path = "/root/tts-rknn/embeddings/codec_embedding.npy";
    std::string socket_path = "/tmp/qwen3_cp.sock";
    int num_threads = 3;
    float temperature = 0.1f;
    int top_k = 50;
    bool use_batch_prefill = true;

    static struct option long_opts[] = {
        {"model",     required_argument, nullptr, 'm'},
        {"weights",   required_argument, nullptr, 'w'},
        {"codec_emb", required_argument, nullptr, 'c'},
        {"socket",    required_argument, nullptr, 's'},
        {"threads",   required_argument, nullptr, 't'},
        {"temperature", required_argument, nullptr, 'T'},
        {"top_k",     required_argument, nullptr, 'k'},
        {"no_batch_prefill", no_argument, nullptr, 'B'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:w:c:s:t:T:k:Bh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'w': weights_dir = optarg; break;
            case 'c': codec_emb_path = optarg; break;
            case 's': socket_path = optarg; break;
            case 't': num_threads = atoi(optarg); break;
            case 'T': temperature = atof(optarg); break;
            case 'k': top_k = atoi(optarg); break;
            case 'B': use_batch_prefill = false; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    // Load model and weights
    CodePredictor cp;
    cp.temperature = temperature;
    cp.top_k = top_k;
    cp.batch_prefill = use_batch_prefill;
    cp.load(model_path, weights_dir, codec_emb_path, num_threads);
    printf("  batch_prefill: %s\n", use_batch_prefill ? "ON" : "OFF");

    // Warmup
    printf("\nWarmup inference...\n");
    {
        std::vector<float> dummy_h(HIDDEN_SIZE, 0.1f);
        auto codes = cp.predict(dummy_h.data(), 100);
        printf("  warmup result: [");
        for (int i = 0; i < NUM_GROUPS; i++) printf("%d%s", codes[i], i < NUM_GROUPS-1 ? "," : "");
        printf("]\n");
    }

    // Create server socket
    int server_fd = create_server_socket(socket_path.c_str());
    if (server_fd < 0) return 1;
    printf("\nListening on %s\n", socket_path.c_str());

    // Serve loop
    while (g_running) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (g_running) perror("accept");
            break;
        }

        // Read: 4096 bytes (hidden_state) + 4 bytes (code_0)
        std::vector<float> hidden(HIDDEN_SIZE);
        int32_t code_0;

        if (!recv_exact(client_fd, hidden.data(), HIDDEN_SIZE * sizeof(float))) {
            close(client_fd);
            continue;
        }
        if (!recv_exact(client_fd, &code_0, sizeof(code_0))) {
            close(client_fd);
            continue;
        }

        // Run prediction
        auto codes = cp.predict(hidden.data(), code_0);

        // Send: 60 bytes (15 × int32)
        send_exact(client_fd, codes.data(), NUM_GROUPS * sizeof(int32_t));
        close(client_fd);
    }

    close(server_fd);
    unlink(socket_path.c_str());
    printf("Server stopped.\n");
    return 0;
}
