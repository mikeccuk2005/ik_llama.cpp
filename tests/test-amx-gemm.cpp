// Reference check for the AMX bf16 tile path.
//
// C[M x N] = A[M x K] * W[N x K]^T, computed two ways: the packed bf16 AMX
// kernel and a plain scalar loop over the same bf16 inputs. If the tile stride
// or the VNNI pack order is wrong, the numbers diverge and there is no point
// measuring throughput.
//
// Build:
//   g++ -O2 -std=c++17 test-amx-gemm.cpp -I ggml/include -I ggml/src \
//       -L build-amx/ggml/src -lggml -o /tmp/test-amx-gemm -Wl,-rpath,$PWD/build-amx/ggml/src
#include "iqk/iqk_amx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Throughput probe: time the int8 tile gemm at realistic shapes so we know the
// kernel's own ceiling before blaming the call structure.
static int bench_int8(int M, int N, int K, int reps) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d8(-127, 127);
    std::vector<int8_t>  Ai((size_t)M * K);
    std::vector<int8_t>  Wi((size_t)N * K);
    std::vector<int8_t>  Wp((size_t)N * K);
    for (auto & v : Ai) v = (int8_t)d8(rng);
    for (auto & v : Wi) v = (int8_t)d8(rng);
    if (!ggml_amx_pack_int8(Wi.data(), N, K, Wp.data())) { printf("pack failed\n"); return 1; }
    std::vector<int32_t> C((size_t)M * N, 0);

    ggml_amx_gemm_int8(M, N, K, Ai.data(), (size_t)K, Wp.data(), C.data(), (size_t)N); // warm
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        ggml_amx_gemm_int8(M, N, K, Ai.data(), (size_t)K, Wp.data(), C.data(), (size_t)N);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gmac = (double)M * N * K * reps / 1e9;
    printf("int8 bench M=%d N=%d K=%d reps=%d  %.2f GMAC in %.3f s  = %.1f GMAC/s (1 thread)\n",
           M, N, K, reps, gmac, s, gmac / s);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench") {
        const int N = argc > 2 ? atoi(argv[2]) : 17408;
        const int K = argc > 3 ? atoi(argv[3]) : 5120;
        return bench_int8(16, N, K, argc > 4 ? atoi(argv[4]) : 20);
    }
    const int M = argc > 1 ? atoi(argv[1]) : 16;
    const int N = argc > 2 ? atoi(argv[2]) : 64;
    const int K = argc > 3 ? atoi(argv[3]) : 128;

    printf("amx bf16 support: %d\n", (int)ggml_amx_bf16_supported());

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // A is M x K row-major bf16, already in tile-friendly layout.
    std::vector<uint16_t> A((size_t)M * K);
    for (auto & v : A) v = f2bf(dist(rng));

    // W is N x K row-major bf16 as the pack helper expects.
    std::vector<uint16_t> W((size_t)N * K);
    for (auto & v : W) v = f2bf(dist(rng));

    std::vector<uint16_t> Wp((size_t)N * K);
    if (!ggml_amx_pack_bf16(W.data(), N, K, Wp.data())) {
        printf("FAIL: pack rejected N=%d K=%d\n", N, K);
        return 1;
    }

    std::vector<float> C((size_t)M * N, -12345.0f);
    ggml_amx_gemm_bf16(M, N, K, A.data(), (size_t)K, Wp.data(), C.data(), (size_t)N);

    // scalar reference over the same bf16 values
    double worst = 0.0;
    double scale = 0.0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += (double)bf2f(A[(size_t)m * K + k]) * (double)bf2f(W[(size_t)n * K + k]);
            }
            const double got = C[(size_t)m * N + n];
            const double err = std::fabs(got - acc);
            if (err > worst) worst = err;
            if (std::fabs(acc) > scale) scale = std::fabs(acc);
        }
    }
    const double rel = scale > 0 ? worst / scale : worst;
    printf("M=%d N=%d K=%d  max_abs_err=%.6g  max_rel_err=%.3g  %s\n",
           M, N, K, worst, rel, rel < 1e-3 ? "OK" : "MISMATCH");
    if (rel >= 1e-3) return 1;

    // ---- int8 path: the 48-column block gemm the hook actually calls ----
    // N is a multiple of 48 by construction (the shadow pads to 48), so every
    // block is full and there is no tail case to check.
    for (int Nt : { 48, 96, 192 }) {
        std::vector<int8_t> Ai((size_t)M * K);
        std::vector<int8_t> Wi((size_t)Nt * K);
        std::uniform_int_distribution<int> d8(-127, 127);
        for (auto & v : Ai) v = (int8_t)d8(rng);
        for (auto & v : Wi) v = (int8_t)d8(rng);

        std::vector<int8_t> Wip((size_t)Nt * K);
        if (!ggml_amx_pack_int8_tiles(Wi.data(), Nt, K, Wip.data())) {
            printf("FAIL: int8 pack rejected N=%d K=%d\n", Nt, K);
            return 1;
        }

        std::vector<int32_t> Ci((size_t)M * Nt, 0);
        for (int n0 = 0; n0 < Nt; n0 += 48) {
            for (int m0 = 0; m0 < M; m0 += 16) {
                alignas(64) int32_t tile[16 * 48];
                ggml_amx_int8_block48_tiled(Ai.data() + (size_t)m0 * K, (size_t)K,
                                            Wip.data(), n0 / 16, K, tile, (size_t)48 * 4);
                const int mrows = std::min(16, M - m0);
                for (int r = 0; r < mrows; r++) {
                    memcpy(Ci.data() + (size_t)(m0 + r) * Nt + n0,
                           tile + (size_t)r * 48, sizeof(int32_t) * 48);
                }
            }
        }

        long long worst_i = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < Nt; n++) {
                long long acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += (long long)Ai[(size_t)m * K + k] * (long long)Wi[(size_t)n * K + k];
                }
                const long long err = std::llabs((long long)Ci[(size_t)m * Nt + n] - acc);
                if (err > worst_i) worst_i = err;
            }
        }
        printf("int8 block48 M=%d N=%d K=%d  max_abs_err=%lld  %s\n",
               M, Nt, K, worst_i, worst_i == 0 ? "OK" : "MISMATCH");
        if (worst_i != 0) return 1;
    }
    return 0;
}