#pragma once
// AMX (Intel Advanced Matrix Extensions) helpers for ik_llama.
// Two kernels: int8 (dpbssd) and bf16 (dpbf16ps). Both consume weights that were
// pre-packed into the VNNI order AMX expects, so packing happens once at load,
// never in the hot loop.
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 when the CPU exposes amx_tile/amx_int8/amx_bf16 and the kernel has permission
// to use tile data (arch_prctl XFEATURE_XTILEDATA). Cached after the first call.
bool ggml_amx_supported(void);
bool ggml_amx_bf16_supported(void);

// Pack weights W (N rows x K cols, row-major) into the layout AMX wants.
//   int8 : out[k/4 * N*4 + n*4 + i] = W[n][4k+i]   -> N*K bytes, N must be a multiple of 16
//   bf16 : out[k/2 * N*2 + n*2 + i] = W[n][2k+i]   -> N*K*2 bytes, N multiple of 16
// Returns false when N % 16 != 0 or K % 64 != 0 (int8) / K % 32 != 0 (bf16).
bool ggml_amx_pack_int8(const int8_t * W, int N, int K, int8_t * out);
bool ggml_amx_pack_bf16(const uint16_t * W, int N, int K, uint16_t * out);

// C[M x N] (int32, row-major, ldc stride) = A[M x K] (int8) * Wpacked^T
// A stride lda bytes, K must be a multiple of 64, M padded to 16 internally.
void ggml_amx_gemm_int8(int M, int N, int K,
                        const int8_t * A, size_t lda,
                        const int8_t * Wpacked,
                        int32_t * C, size_t ldc);

// Blocked int8 GEMM (panel-major packing, K-blocked accumulation).
// Weights are streamed once per matmul instead of once per 16-row band.
bool ggml_amx_pack_int8_panels(const int8_t * W, int nx, int K, int8_t * out, int * npad_out);
void ggml_amx_int8_tile_begin(void);
void ggml_amx_int8_tile_end(void);
void ggml_amx_int8_panel_acc(const int8_t * A, size_t lda,
                             const int8_t * Wp, int panel, int K,
                             int k0, int kcount,
                             int32_t * acc, size_t ldc);
void ggml_amx_int8_gemm_blocked(int ny, int K, const int8_t * A, size_t lda,
                                const int8_t * Wp, int npad,
                                int32_t * accbuf,
                                float * C, size_t ldc_f,
                                const float * row_scale, const float * a_scale,
                                int kb_size, int only_panel);

// Tile-major int8 pack (each 16-column tile contiguous, 64 bytes per row).
bool ggml_amx_pack_int8_tiles(const int8_t * W, int N, int K, int8_t * out);

// 48 columns from tile index t0, 16 rows of A, tile-major weights.
void ggml_amx_int8_block48_tiled(const int8_t * A, size_t lda,
                                 const int8_t * Wp, int t0, int K,
                                 int32_t * C, size_t ldc);

// One 48-column block of the int8 gemm: C[16 x 48] = A[16 x K] x W[n0..n0+48]^T.
// Weights are read once per call, so the caller should loop n0 outside rows.
void ggml_amx_int8_block48(const int8_t * A, size_t lda,
                           const int8_t * Wpacked, int n, int n0,
                           int32_t * C, size_t ldc, int K);

// Prebuild the tile shadow for one weight tensor (bf16, or int8 for Q8_0).
// Safe to call from any thread when no compute region is active; call it at
// model load so the first prefill does not pay the dequant.
// C linkage: llama-model-loader.cpp includes this header inside an extern "C"
// block, so the definition must not be C++-mangled or the link fails with
// "undefined reference to iqk_amx_prewarm".
#ifdef __cplusplus
extern "C"
#endif
void iqk_amx_prewarm(int type, const void * A, size_t strideA, int nx, int k);

// C[M x N] (f32) = A[M x K] (bf16) * Wpacked^T. K multiple of 32.
void ggml_amx_gemm_bf16(int M, int N, int K,
                        const uint16_t * A, size_t lda,
                        const uint16_t * Wpacked,
                        float * C, size_t ldc);

#ifdef __cplusplus
}
#endif