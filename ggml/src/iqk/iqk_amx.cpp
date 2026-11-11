// AMX kernels for ik_llama: int8 (dpbssd) and bf16 (dpbf16ps).
//
// Each 16x16 output block is produced by one accumulator tile; K is walked in
// 64-byte chunks (64 int8 or 32 bf16 elements) with the tile config loaded once
// per call. Loading the tile config per multiply instead of per call costs ~15x,
// which is why the config lives outside the K loop here.
#include "iqk_amx.h"

#include <string.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AMX_TILE__) && (defined(__GNUC__) || defined(__clang__))
#define IQK_HAVE_AMX 1
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#define IQK_HAVE_AMX 0
#endif

#if IQK_HAVE_AMX
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

struct alignas(64) iqk_tile_cfg {
    uint8_t  palette_id = 1;
    uint8_t  start_row  = 0;
    uint8_t  reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t  rows[16]  = {};
};

// Ask the kernel for permission to touch tile data; without it any AMX op traps.
static bool iqk_amx_request(void) {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}
#endif

static int  s_checked = 0;
static bool s_int8_ok = false;
static bool s_bf16_ok = false;

// Probe CPUID leaf 7 for amx_tile(22)/amx_int8(25)/amx_bf16(23) once, then cache.
static void iqk_amx_probe(void) {
    if (s_checked) return;
    s_checked = 1;
#if IQK_HAVE_AMX
    if (!iqk_amx_request()) return;
    unsigned int eax = 7, ebx = 0, ecx = 0, edx = 0;
    __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "c"(0));
    const bool tile = (edx >> 22) & 1;
    const bool int8 = (edx >> 25) & 1;
    const bool bf16 = (edx >> 23) & 1;
    s_int8_ok = tile && int8;
    s_bf16_ok = tile && bf16;
#endif
}

bool ggml_amx_supported(void)      { iqk_amx_probe(); return s_int8_ok; }
bool ggml_amx_bf16_supported(void) { iqk_amx_probe(); return s_bf16_ok; }

// out[k4*N*4 + n*4 + i] = W[n][4*k4 + i]
bool ggml_amx_pack_int8(const int8_t * W, int N, int K, int8_t * out) {
    if (N % 16 != 0 || K % 64 != 0) return false;
    for (int k4 = 0; k4 < K/4; k4++)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < 4; i++)
                out[k4*N*4 + n*4 + i] = W[n*K + k4*4 + i];
    return true;
}

// out[k2*N*2 + n*2 + i] = W[n][2*k2 + i]
bool ggml_amx_pack_bf16(const uint16_t * W, int N, int K, uint16_t * out) {
    if (N % 16 != 0 || K % 32 != 0) return false;
    for (int k2 = 0; k2 < K/2; k2++)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < 2; i++)
                out[k2*N*2 + n*2 + i] = W[n*K + k2*2 + i];
    return true;
}

#if IQK_HAVE_AMX
// One 16-row band of A times all N columns of the packed weights.
static void iqk_amx_int8_band(const int8_t * A, const int8_t * Wp, int32_t * C,
                              int n, int k) {
    iqk_tile_cfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);

    for (int n0 = 0; n0 < n; n0 += 16) {
        _tile_zero(2);
        for (int k0 = 0; k0 < k; k0 += 64) {
            _tile_loadd(0, A + k0, (size_t)k);
            _tile_loadd(1, Wp + (size_t)(k0/4)*n*4 + (size_t)n0*4, (size_t)n*4);
            _tile_dpbssd(2, 0, 1);
        }
        _tile_stored(2, C + n0, (size_t)n*4);
    }
    _tile_release();
}

static void iqk_amx_bf16_band(const uint16_t * A, const uint16_t * Wp, float * C,
                              int n, int k) {
    iqk_tile_cfg cfg;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);

    for (int n0 = 0; n0 < n; n0 += 16) {
        _tile_zero(2);
        for (int k0 = 0; k0 < k; k0 += 32) {
            _tile_loadd(0, A + k0, (size_t)k*2);
            // row is n*2 bf16 values, so the stride in bytes is n*4, not n*2
            _tile_loadd(1, Wp + (size_t)(k0/2)*n*2 + (size_t)n0*2, (size_t)n*4);
            _tile_dpbf16ps(2, 0, 1);
        }
        _tile_stored(2, C + n0, (size_t)n*4);
    }
    _tile_release();
}
#endif

// C[M x N] int32 = A[M x K] int8 x Wpacked^T, 16-row bands of A.
void ggml_amx_gemm_int8(int M, int N, int K,
                        const int8_t * A, size_t lda,
                        const int8_t * Wpacked,
                        int32_t * C, size_t ldc) {
#if IQK_HAVE_AMX
    if (!ggml_amx_supported() || N % 16 != 0 || K % 64 != 0) return;
    // full K per row: the tiles are loaded for every 64-byte chunk of K
    int8_t * abuf = (int8_t *) aligned_alloc(64, (size_t)16 * K);
    if (!abuf) return;
    for (int m0 = 0; m0 < M; m0 += 16) {
        // pad the tail rows by repeating the last one
        const int mrows = (M - m0 < 16) ? (M - m0) : 16;
        memset(abuf, 0, (size_t)16 * K);
        for (int r = 0; r < 16; r++) {
            const int src = m0 + (r < mrows ? r : mrows - 1);
            memcpy(abuf + (size_t)r*K, A + (size_t)src*lda, (size_t)K);
        }
        int32_t * dst = C + (size_t)m0*ldc;
        if (mrows == 16) {
            iqk_amx_int8_band(abuf, Wpacked, dst, N, K);
        } else {
            // the tile always produces 16 rows; only mrows of them are ours
            int32_t * tmp = (int32_t *) aligned_alloc(64, (size_t)16*N*sizeof(int32_t));
            if (tmp) {
                iqk_amx_int8_band(abuf, Wpacked, tmp, N, K);
                memcpy(dst, tmp, (size_t)mrows*N*sizeof(int32_t));
                free(tmp);
            }
        }
    }
    free(abuf);
#else
    (void)M; (void)N; (void)K; (void)A; (void)lda; (void)Wpacked; (void)C; (void)ldc;
#endif
}

// C[M x N] f32 = A[M x K] bf16 x Wpacked^T.
void ggml_amx_gemm_bf16(int M, int N, int K,
                        const uint16_t * A, size_t lda,
                        const uint16_t * Wpacked,
                        float * C, size_t ldc) {
#if IQK_HAVE_AMX
    if (!ggml_amx_bf16_supported() || N % 16 != 0 || K % 32 != 0) return;
    // fast path: a single 16-row band can be fed to the tiles straight from A
    if (M == 16 && (size_t)lda == (size_t)K) {
        iqk_amx_bf16_band(A, Wpacked, C, N, K);
        return;
    }
    uint16_t * abuf = (uint16_t *) aligned_alloc(64, (size_t)16 * K * 2);
    if (!abuf) return;
    for (int m0 = 0; m0 < M; m0 += 16) {
        const int mrows = (M - m0 < 16) ? (M - m0) : 16;
        memset(abuf, 0, (size_t)16 * K * 2);
        for (int r = 0; r < 16; r++) {
            const int src = m0 + (r < mrows ? r : mrows - 1);
            memcpy(abuf + (size_t)r*K, A + (size_t)src*lda, (size_t)K*2);
        }
        float * dst = C + (size_t)m0*ldc;
        if (mrows == 16) {
            iqk_amx_bf16_band(abuf, Wpacked, dst, N, K);
        } else {
            float * tmp = (float *) aligned_alloc(64, (size_t)16*N*sizeof(float));
            if (tmp) {
                iqk_amx_bf16_band(abuf, Wpacked, tmp, N, K);
                memcpy(dst, tmp, (size_t)mrows*N*sizeof(float));
                free(tmp);
            }
        }
    }
    free(abuf);
#else
    (void)M; (void)N; (void)K; (void)A; (void)lda; (void)Wpacked; (void)C; (void)ldc;
#endif
}