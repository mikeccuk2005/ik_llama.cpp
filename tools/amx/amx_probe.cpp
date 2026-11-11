// AMX tile microbenchmark for ik_llama: correctness + throughput of 16x16 blocks.
// int8  : dpbssd, A=16x64 int8, B=16x64 int8 in VNNI (K-major, groups of 4), C=16x16 int32
// bf16  : dpbf16ps, A=16x32 bf16, B=16x32 bf16 in VNNI (groups of 2), C=16x16 f32
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

struct alignas(64) TileCfg {
    uint8_t  palette_id = 1;
    uint8_t  start_row  = 0;
    uint8_t  reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t  rows[16]  = {};
};

static inline bool amx_permission() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

static inline float bf16_to_f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f32_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

// 16x16 int8 block: C += A * B, both K=64, B in VNNI layout.
static void amx_int8_16x16x64(const int8_t *A, const int8_t *Bvnni, int32_t *C) {
    TileCfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    _tile_zero(2);
    _tile_loadd(0, A, 64);
    _tile_loadd(1, Bvnni, 64);
    _tile_dpbssd(2, 0, 1);
    _tile_stored(2, C, 64);
    _tile_release();
}

// 16x16 bf16 block: C += A * B, K=32, B in VNNI layout (pairs).
static void amx_bf16_16x16x32(const uint16_t *A, const uint16_t *Bvnni, float *C) {
    TileCfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    _tile_zero(2);
    _tile_loadd(0, A, 64);
    _tile_loadd(1, Bvnni, 64);
    _tile_dpbf16ps(2, 0, 1);
    _tile_stored(2, C, 64);
    _tile_release();
}

// inner-loop variants: configure once, loop, store once (how a real GEMM does it)
static void amx_int8_loop(const int8_t *A, const int8_t *Bvnni, int32_t *C, long iters) {
    TileCfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    _tile_zero(2);
    for (long i = 0; i < iters; i++) {
        _tile_loadd(0, A, 64);
        _tile_loadd(1, Bvnni, 64);
        _tile_dpbssd(2, 0, 1);
    }
    _tile_stored(2, C, 64);
    _tile_release();
}

static void amx_bf16_loop(const uint16_t *A, const uint16_t *Bvnni, float *C, long iters) {
    TileCfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    _tile_zero(2);
    for (long i = 0; i < iters; i++) {
        _tile_loadd(0, A, 64);
        _tile_loadd(1, Bvnni, 64);
        _tile_dpbf16ps(2, 0, 1);
    }
    _tile_stored(2, C, 64);
    _tile_release();
}

// AVX-512 VNNI baseline: same 16x16 output from K=64 int8 inputs, the path ggml uses today.
static void avx512_vnni_16x16x64(const int8_t *A, const int8_t *Bvnni, int32_t *C) {
    __m512i acc[16];
    for (int m = 0; m < 16; m++) acc[m] = _mm512_setzero_si512();
    // B tile: 16 rows (k4) x 64 bytes (16 n x 4)
    for (int m = 0; m < 16; m++)
        for (int k4 = 0; k4 < 16; k4++) {
            __m512i a = _mm512_set1_epi32(*(const int32_t *)(A + m*64 + k4*4));
            __m512i b = _mm512_loadu_si512((const void *)(Bvnni + k4*64));
            acc[m] = _mm512_dpbusd_epi32(acc[m], b, a);
        }
    for (int m = 0; m < 16; m++) _mm512_storeu_si512((void *)(C + m*16), acc[m]);
}

static void avx512_loop(const int8_t *A, const int8_t *Bvnni, int32_t *C, long iters) {
    for (long i = 0; i < iters; i++) avx512_vnni_16x16x64(A, Bvnni, C);
}

// Reference: C[m][n] = sum_k A[m][k] * W[n][k]  (W row-major N x K, like a quant row)
static void ref_int8(const int8_t *A, const int8_t *W, int M, int N, int K, int32_t *C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)W[n*K+k];
            C[m*N+n] = s;
        }
}

static void pack_vnni_int8(const int8_t *W, int N, int K, int8_t *out) {
    // out[k/4 * N*4 + n*4 + i] = W[n][4k + i]
    for (int k4 = 0; k4 < K/4; k4++)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < 4; i++)
                out[k4*N*4 + n*4 + i] = W[n*K + k4*4 + i];
}

static void pack_vnni_bf16(const uint16_t *W, int N, int K, uint16_t *out) {
    for (int k2 = 0; k2 < K/2; k2++)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < 2; i++)
                out[k2*N*2 + n*2 + i] = W[n*K + k2*2 + i];
}

int main() {
    if (!amx_permission()) { printf("AMX permission denied\n"); return 1; }
    printf("AMX permission granted\n");

    int8_t  A8[16*64] __attribute__((aligned(64)));
    int8_t  W8[16*64] __attribute__((aligned(64)));
    int8_t  B8[16*64] __attribute__((aligned(64)));
    int32_t C8t[16*16] __attribute__((aligned(64)));
    int32_t C8r[16*16];

    srand(42);
    for (int i = 0; i < 16*64; i++) { A8[i] = (int8_t)(rand() % 255 - 127); W8[i] = (int8_t)(rand() % 255 - 127); }
    pack_vnni_int8(W8, 16, 64, B8);

    amx_int8_16x16x64(A8, B8, C8t);
    ref_int8(A8, W8, 16, 16, 64, C8r);

    int bad = 0;
    for (int i = 0; i < 256; i++) if (C8t[i] != C8r[i]) bad++;
    printf("int8 16x16x64: %s  (first: tile=%d ref=%d)\n", bad ? "MISMATCH" : "OK", C8t[0], C8r[0]);

    uint16_t Ab[16*32] __attribute__((aligned(64)));
    uint16_t Wb[16*32] __attribute__((aligned(64)));
    uint16_t Bb[16*32] __attribute__((aligned(64)));
    float    Cbt[16*16] __attribute__((aligned(64)));
    for (int i = 0; i < 16*32; i++) { Ab[i] = f32_to_bf16((float)(rand()%200)/100.f); Wb[i] = f32_to_bf16((float)(rand()%200)/100.f); }
    pack_vnni_bf16(Wb, 16, 32, Bb);

    amx_bf16_16x16x32(Ab, Bb, Cbt);
    int badb = 0;
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 16; n++) {
            float s = 0;
            for (int k = 0; k < 32; k++) s += bf16_to_f32(Ab[m*32+k]) * bf16_to_f32(Wb[n*32+k]);
            if (fabsf(s - Cbt[m*16+n]) > 0.05f * (fabsf(s) + 1.f)) badb++;
        }
    printf("bf16 16x16x32: %s  (first: tile=%.4f)\n", badb ? "MISMATCH" : "OK", Cbt[0]);

    // throughput per tile op (16x16x64 int8 = 16,384 MACs; 16x16x32 bf16 = 8,192 MACs)
    const long iters = 200000;
    printf("\n-- per-call (config+ops+release each time) --\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < iters; i++) amx_int8_16x16x64(A8, B8, C8t);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("int8 : %.1f GMAC/s\n", iters*16384.0/sec/1e9);

    printf("\n-- inner loop (config once, ops in a loop) --\n");
    t0 = std::chrono::high_resolution_clock::now();
    amx_int8_loop(A8, B8, C8t, iters);
    t1 = std::chrono::high_resolution_clock::now();
    sec = std::chrono::duration<double>(t1 - t0).count();
    printf("int8 : %.2f G-tile/s -> %.1f GMAC/s (%.1f GOPS)\n", iters/sec/1e9, iters*16384.0/sec/1e9, iters*16384.0*2/sec/1e9);

    t0 = std::chrono::high_resolution_clock::now();
    amx_bf16_loop(Ab, Bb, Cbt, iters);
    t1 = std::chrono::high_resolution_clock::now();
    sec = std::chrono::duration<double>(t1 - t0).count();
    printf("bf16 : %.2f G-tile/s -> %.1f GMAC/s (%.1f GOPS)\n", iters/sec/1e9, iters*16*16*32.0/sec/1e9, iters*16*16*32.0*2/sec/1e9);
    t0 = std::chrono::high_resolution_clock::now();
    avx512_loop(A8, B8, C8t, iters);
    t1 = std::chrono::high_resolution_clock::now();
    sec = std::chrono::duration<double>(t1 - t0).count();
    printf("avx512 vnni int8 16x16x64: %.1f GMAC/s (%.1f GOPS)\n", iters*16384.0/sec/1e9, iters*16384.0*2/sec/1e9);
    return bad || badb ? 2 : 0;
}