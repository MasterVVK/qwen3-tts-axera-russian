#pragma once
// Simple .npy file reader (float32 only, C order)
// Supports NPY format v1.0 and v2.0

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;
    size_t total_elements() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

static inline NpyArray load_npy(const char* path) {
    NpyArray result;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }

    // Read magic + version
    uint8_t magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); fprintf(stderr, "ERROR: bad npy header\n"); exit(1); }
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fclose(f); fprintf(stderr, "ERROR: not a .npy file: %s\n", path); exit(1);
    }

    uint8_t major = magic[6], minor = magic[7];
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        fread(&hl, 2, 1, f);
        header_len = hl;
    } else if (major == 2) {
        fread(&header_len, 4, 1, f);
    } else {
        fclose(f); fprintf(stderr, "ERROR: unsupported npy version %d.%d\n", major, minor); exit(1);
    }

    // Read header string
    std::string header(header_len, '\0');
    fread(&header[0], 1, header_len, f);

    // Parse shape from header: ...'shape': (N, M), ...
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\"");
    if (shape_pos == std::string::npos) {
        fclose(f); fprintf(stderr, "ERROR: no shape in npy header\n"); exit(1);
    }
    auto paren_start = header.find('(', shape_pos);
    auto paren_end = header.find(')', paren_start);
    std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

    // Parse comma-separated integers
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        int64_t val = strtoll(&shape_str[pos], nullptr, 10);
        result.shape.push_back(val);
        while (pos < shape_str.size() && shape_str[pos] != ',') pos++;
    }

    // Verify dtype is float32
    auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) descr_pos = header.find("\"descr\"");
    if (descr_pos != std::string::npos) {
        bool is_f4 = header.find("f4", descr_pos) != std::string::npos ||
                     header.find("float32", descr_pos) != std::string::npos;
        if (!is_f4) {
            // Check for f8 (float64) - we'll convert
            bool is_f8 = header.find("f8", descr_pos) != std::string::npos;
            if (is_f8) {
                size_t n = result.total_elements();
                std::vector<double> tmp(n);
                fread(tmp.data(), sizeof(double), n, f);
                fclose(f);
                result.data.resize(n);
                for (size_t i = 0; i < n; i++) result.data[i] = (float)tmp[i];
                return result;
            }
            fprintf(stderr, "WARNING: npy dtype may not be float32 in %s\n", path);
        }
    }

    // Read raw float32 data
    size_t n = result.total_elements();
    result.data.resize(n);
    size_t read = fread(result.data.data(), sizeof(float), n, f);
    fclose(f);

    if (read != n) {
        fprintf(stderr, "ERROR: expected %zu floats, got %zu in %s\n", n, read, path);
        exit(1);
    }

    return result;
}
