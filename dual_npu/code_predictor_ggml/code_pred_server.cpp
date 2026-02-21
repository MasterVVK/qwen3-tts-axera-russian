/**
 * code_pred_ggml_server.cpp — GGML-based Code Predictor Server for Qwen3-TTS
 *
 * Drop-in replacement for ONNX code_predictor_server (same socket protocol).
 * Uses qwen3-tts.cpp TTSTransformer for Q4_0/Q8_0 quantized inference.
 *
 * Protocol (Unix socket):
 *   Client -> Server: [4096 bytes: float32[1024] hidden_state] [4 bytes: int32 code_0]
 *   Server -> Client: [60 bytes: int32[15] codes for codebooks 1-15]
 *
 * Build (inside qwen3-tts.cpp/build):
 *   cmake .. -DCMAKE_BUILD_TYPE=Release -DQWEN3_TTS_TIMING=ON && make code_pred_server
 *
 * Usage:
 *   taskset -c 4-7 ./code_pred_server --model ../models/qwen3-tts-0.6b-q4_0.gguf \
 *       --socket /tmp/qwen3_cp.sock --temperature 0.1 --top_k 50
 */

#include "tts_transformer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <vector>
#include <string>
#include <chrono>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>

// ============================================================
// Globals
// ============================================================
static volatile sig_atomic_t g_running = 1;
static void signal_handler(int) { g_running = 0; }

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

// ============================================================
// Main
// ============================================================
static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --model PATH       GGUF model (required, e.g. qwen3-tts-0.6b-q4_0.gguf)\n"
           "  --socket PATH      Unix socket path (default: /tmp/qwen3_cp.sock)\n"
           "  --temperature F    Sampling temperature (default: 0.1)\n"
           "  --top_k N          Top-K sampling (default: 50)\n"
           "  --help             Show this help\n",
           prog);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string model_path;
    std::string socket_path = "/tmp/qwen3_cp.sock";
    float temperature = 0.1f;
    int top_k = 50;

    static struct option long_opts[] = {
        {"model",       required_argument, nullptr, 'm'},
        {"socket",      required_argument, nullptr, 's'},
        {"temperature", required_argument, nullptr, 'T'},
        {"top_k",       required_argument, nullptr, 'k'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:T:k:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 's': socket_path = optarg; break;
            case 'T': temperature = atof(optarg); break;
            case 'k': top_k = atoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    // --- Load model ---
    printf("=== GGML Code Predictor Server ===\n");
    printf("Model:    %s\n", model_path.c_str());
    printf("Temp:     %.2f\n", temperature);
    printf("Top-K:    %d\n", top_k);

    qwen3_tts::TTSTransformer transformer;

    printf("\nLoading model...\n");
    auto t_load_start = std::chrono::high_resolution_clock::now();

    if (!transformer.load_model(model_path)) {
        fprintf(stderr, "ERROR: Failed to load model: %s\n", model_path.c_str());
        fprintf(stderr, "  %s\n", transformer.get_error().c_str());
        return 1;
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    auto config = transformer.get_config();
    printf("Model loaded in %.0f ms\n", load_ms);
    printf("  hidden_size=%d, cp_layers=%d, codebooks=%d, cp_vocab=%d\n",
           config.hidden_size, config.code_pred_layers,
           config.n_codebooks, config.code_pred_vocab_size);

    int hidden_size = config.hidden_size;
    int num_output_codes = config.n_codebooks - 1; // codebooks 1..15

    // --- Warmup ---
    printf("\nWarmup...\n");
    for (int w = 0; w < 3; w++) {
        std::vector<float> dummy_h(hidden_size, 0.1f * (w + 1));
        std::vector<int32_t> output_codes;
        auto t0 = std::chrono::high_resolution_clock::now();
        transformer.predict_codes_autoregressive(
            dummy_h.data(), 100 + w, output_codes, temperature, top_k);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  warmup %d: %.1fms, %zu codes\n", w + 1, ms, output_codes.size());
    }

    // --- Create server socket ---
    int server_fd = create_server_socket(socket_path.c_str());
    if (server_fd < 0) return 1;
    printf("\nListening on %s\n", socket_path.c_str());
    printf("Protocol: recv %d + 4 bytes -> send %d bytes\n",
           hidden_size * (int)sizeof(float),
           num_output_codes * (int)sizeof(int32_t));

    // --- Serve loop ---
    long request_count = 0;
    double total_ms = 0;

    while (g_running) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (g_running) perror("accept");
            break;
        }

        // Read: hidden_state + code_0
        std::vector<float> hidden(hidden_size);
        int32_t code_0;

        if (!recv_exact(client_fd, hidden.data(), hidden_size * sizeof(float))) {
            close(client_fd);
            continue;
        }
        if (!recv_exact(client_fd, &code_0, sizeof(code_0))) {
            close(client_fd);
            continue;
        }

        // Run prediction
        std::vector<int32_t> codes;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = transformer.predict_codes_autoregressive(
            hidden.data(), code_0, codes, temperature, top_k);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!ok) {
            fprintf(stderr, "  [%ld] ERROR: %s\n", request_count + 1,
                    transformer.get_error().c_str());
            close(client_fd);
            continue;
        }

        request_count++;
        total_ms += ms;

        printf("  [%ld] code_0=%d -> %.1fms (avg %.1fms)\n",
               request_count, code_0, ms, total_ms / request_count);

        // Pad/truncate to expected size
        codes.resize(num_output_codes, 0);

        // Send: 15 x int32
        send_exact(client_fd, codes.data(), num_output_codes * sizeof(int32_t));
        close(client_fd);
    }

    close(server_fd);
    unlink(socket_path.c_str());
    printf("\nServer stopped. %ld requests, avg %.1fms/req\n",
           request_count, request_count > 0 ? total_ms / request_count : 0);
    return 0;
}
