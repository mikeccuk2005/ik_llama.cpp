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

// C[M x N] (f32) = A[M x K] (bf16) * Wpacked^T. K multiple of 32.
void ggml_amx_gemm_bf16(int M, int N, int K,
                        const uint16_t * A, size_t lda,
                        const uint16_t * Wpacked,
                        float * C, size_t ldc);

#ifdef __cplusplus
}
#endif