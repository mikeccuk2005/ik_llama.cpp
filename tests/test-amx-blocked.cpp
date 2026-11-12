// Correctness and throughput harness for the blocked int8 AMX GEMM.
//
//   test-amx-blocked                 -> correctness sweep
//   test-amx-blocked bench NY N K [reps] [kb]
//
// Correctness compares against a scalar int32 reference. Integer arithmetic
// means any difference at all is a bug, so the test requires exact equality.
#include "iqk_amx.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> run_blocked(int ny, int nx, int K, int kb, bool verbose) {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> d8(-127, 127);

    std::vector<int8_t> A((size_t)ny * K);
    std::vector<int8_t> W((size_t)nx * K);
    for (auto & v : A) v = (int8_t)d8(rng);
    for (auto & v : W) v = (int8_t)d8(rng);

    int npad = 0;
    std::vector<int8_t> Wp((size_t)nx * K + 96 * K);
    if (!ggml_amx_pack_int8_panels(W.data(), nx, K, Wp.data(), &npad)) {
        printf("FAIL: pack rejected nx=%d K=%d\n", nx, K);
        exit(1);
    }

    const int bands = (ny + 15) / 16;
    std::vector<int32_t> acc((size_t)bands * 16 * 96, 0);
    std::vector<float> C((size_t)ny * nx, 0.0f);

    ggml_amx_int8_gemm_blocked(ny, K, A.data(), (size_t)K, Wp.data(), npad,
                               acc.data(), C.data(), (size_t)nx,
                               nullptr, nullptr, kb, -1);

    if (verbose) {
        long long worst = 0;
        for (int m = 0; m < ny; m++) {
            for (int n = 0; n < nx; n++) {
                long long ref = 0;
                const int8_t * a = A.data() + (size_t)m * K;
                const int8_t * w = W.data() + (size_t)n * K;
                for (int k = 0; k < K; k++) ref += (long long)a[k] * (long long)w[k];
                const long long got = (long long)C[(size_t)m * nx + n];
                const long long err = llabs(got - ref);
                if (err > worst) worst = err;
            }
        }
        printf("blocked M=%-4d N=%-5d K=%-5d kb=%-4d npad=%-5d max_abs_err=%lld  %s\n",
               ny, nx, K, kb, npad, worst, worst == 0 ? "OK" : "MISMATCH");
        fflush(stdout);
        if (worst != 0) exit(1);
    }
    return C;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench") {
        const int ny   = argc > 2 ? atoi(argv[2]) : 2048;
        const int nx   = argc > 3 ? atoi(argv[3]) : 17408;
        const int K    = argc > 4 ? atoi(argv[4]) : 5120;
        const int reps = argc > 5 ? atoi(argv[5]) : 3;
        const int kb   = argc > 6 ? atoi(argv[6]) : 1024;

        std::mt19937 rng(5);
        std::uniform_int_distribution<int> d8(-127, 127);
        std::vector<int8_t> A((size_t)ny * K);
        std::vector<int8_t> W((size_t)nx * K);
        for (auto & v : A) v = (int8_t)d8(rng);
        for (auto & v : W) v = (int8_t)d8(rng);
        int npad = 0;
        std::vector<int8_t> Wp((size_t)nx * K + 96 * K);
        if (!ggml_amx_pack_int8_panels(W.data(), nx, K, Wp.data(), &npad)) return 1;
        const int bands = (ny + 15) / 16;
        std::vector<int32_t> acc((size_t)bands * 16 * 96, 0);
        std::vector<float> C((size_t)ny * nx, 0.0f);

        ggml_amx_int8_gemm_blocked(ny, K, A.data(), (size_t)K, Wp.data(), npad,
                                   acc.data(), C.data(), (size_t)nx,
                                   nullptr, nullptr, kb, -1);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++)
            ggml_amx_int8_gemm_blocked(ny, K, A.data(), (size_t)K, Wp.data(), npad,
                                       acc.data(), C.data(), (size_t)nx,
                                       nullptr, nullptr, kb, -1);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double gmac = (double)ny * nx * K * reps / 1e9;
        printf("blocked bench ny=%d N=%d K=%d kb=%d reps=%d  %.1f GMAC in %.3f s = %.1f GMAC/s (1 thread)\n",
               ny, nx, K, kb, reps, gmac, s, gmac / s);
        return 0;
    }

    // Correctness: odd row counts, panel tails, K not a multiple of the K block.
    run_blocked(16, 96, 128, 1024, true);
    run_blocked(16, 192, 256, 1024, true);
    run_blocked(37, 96, 128, 64, true);
    run_blocked(37, 100, 192, 128, true);
    run_blocked(64, 288, 256, 128, true);
    run_blocked(48, 96, 320, 192, true);
    run_blocked(5, 96, 128, 64, true);
    return 0;
}