// Self-test for ggml_amx_gemm_int8 / ggml_amx_gemm_bf16 against a scalar reference.
#include "iqk_amx.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

int main() {
    printf("amx int8 supported: %d, bf16 supported: %d\n",
           (int)ggml_amx_supported(), (int)ggml_amx_bf16_supported());
    if (!ggml_amx_supported()) { printf("SKIP: no AMX on this CPU\n"); return 0; }

    const int M = 16, N = 32, K = 128;
    int8_t  A[M*K] __attribute__((aligned(64)));
    int8_t  W[N*K] __attribute__((aligned(64)));
    int8_t  Wp[N*K] __attribute__((aligned(64)));
    int32_t C[M*N] __attribute__((aligned(64)));

    srand(7);
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < N*K; i++) W[i] = (int8_t)(rand() % 255 - 127);

    if (!ggml_amx_pack_int8(W, N, K, Wp)) { printf("pack int8 failed\n"); return 1; }
    memset(C, 0, sizeof(C));
    ggml_amx_gemm_int8(M, N, K, A, K, Wp, C, N);

    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)W[n*K+k];
            if (s != C[m*N+n]) bad++;
        }
    printf("int8 %dx%dx%d: %s\n", M, N, K, bad ? "MISMATCH" : "OK");

    if (ggml_amx_bf16_supported()) {
        uint16_t Ab[M*K] __attribute__((aligned(64)));
        uint16_t Wb[N*K] __attribute__((aligned(64)));
        uint16_t Wpb[N*K] __attribute__((aligned(64)));
        float    Cb[M*N] __attribute__((aligned(64)));
        for (int i = 0; i < M*K; i++) Ab[i] = f2bf((float)(rand()%200-100)/64.f);
        for (int i = 0; i < N*K; i++) Wb[i] = f2bf((float)(rand()%200-100)/64.f);
        if (!ggml_amx_pack_bf16(Wb, N, K, Wpb)) { printf("pack bf16 failed\n"); return 1; }
        memset(Cb, 0, sizeof(Cb));
        ggml_amx_gemm_bf16(M, N, K, Ab, K, Wpb, Cb, N);
        int badb = 0; double maxrel = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int k = 0; k < K; k++) s += bf2f(Ab[m*K+k]) * bf2f(Wb[n*K+k]);
                const double rel = fabs(s - Cb[m*N+n]) / (fabs(s) + 1.f);
                if (rel > maxrel) maxrel = rel;
                if (rel > 0.02) badb++;
            }
        printf("bf16 %dx%dx%d: %s (max rel err %.4f%%)\n", M, N, K, badb ? "MISMATCH" : "OK", maxrel*100);
        bad += badb;
    }
    return bad ? 2 : 0;
}