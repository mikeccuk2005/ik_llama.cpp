// AMX-backed mul_mat for the CPU path.
//
// Weights arrive quantised. AMX wants bf16 (or int8) tiles, so on first use each
// weight tensor gets a bf16 shadow, packed into the VNNI order the tiles read.
// The shadow costs 2 bytes per weight instead of ~0.5, which is the price of
// not doing per-block scale fixups inside the tile loop.
//
// Enable with GGML_AMX_MUL_MAT=1. Falls back (returns false) for any shape or
// type the kernel cannot take.
#include "iqk_amx.h"

#include "../ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <thread>

namespace {

struct Shadow {
    uint16_t * packed = nullptr;  // n_pad x K, VNNI order
    int nx_pad = 0;
    int k = 0;
};

std::mutex g_mu;
std::unordered_map<const void *, Shadow> g_shadows;

// One bf16 value, rounded to nearest.
inline uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

// Dequantise one row of weights into bf16. Returns false for unsupported types.
bool dequant_row(int type, const void * src, float * tmp, int k) {
    switch (type) {
        case GGML_TYPE_Q4_0:   dequantize_row_q4_0((const block_q4_0 *)   src, tmp, k); return true;
        case GGML_TYPE_Q8_0:   dequantize_row_q8_0((const block_q8_0 *)   src, tmp, k); return true;
        case GGML_TYPE_Q4_K:   dequantize_row_q4_K((const block_q4_K *)   src, tmp, k); return true;
        case GGML_TYPE_Q5_K:   dequantize_row_q5_K((const block_q5_K *)   src, tmp, k); return true;
        case GGML_TYPE_Q6_K:   dequantize_row_q6_K((const block_q6_K *)   src, tmp, k); return true;
        case GGML_TYPE_IQ4_XS: dequantize_row_iq4_xs((const block_iq4_xs *) src, tmp, k); return true;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs *) src, tmp, k); return true;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) src, tmp, k); return true;
        default: return false;
    }
}

bool type_ok(int type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ3_XXS: return true;
        default: return false;
    }
}

// Build (or fetch) the packed bf16 shadow for this weight tensor.
const Shadow * get_shadow(int type, const void * A, size_t strideA, int nx, int k) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_shadows.find(A);
    if (it != g_shadows.end()) return &it->second;

    const int nx_pad = ((nx + 15) / 16) * 16;
    const size_t bytes = (size_t)nx_pad * k * sizeof(uint16_t);
    uint16_t * packed = (uint16_t *) aligned_alloc(64, bytes);
    if (!packed) return nullptr;
    memset(packed, 0, bytes);

    std::vector<float> row(k);
    uint16_t * dense = (uint16_t *) aligned_alloc(64, (size_t)nx_pad * k * 2);
    if (!dense) { free(packed); return nullptr; }
    memset(dense, 0, (size_t)nx_pad * k * 2);

    for (int n = 0; n < nx; n++) {
        const void * src = (const char *) A + (size_t)n * strideA;
        if (!dequant_row(type, src, row.data(), k)) { free(packed); free(dense); return nullptr; }
        for (int j = 0; j < k; j++) dense[(size_t)n * k + j] = f2bf(row[j]);
    }
    if (!ggml_amx_pack_bf16(dense, nx_pad, k, packed)) { free(packed); free(dense); return nullptr; }
    free(dense);

    Shadow sh{packed, nx_pad, k};
    g_shadows[A] = sh;
    return &g_shadows[A];
}

} // namespace

// True when the AMX mul_mat path is compiled in, the CPU can run it, and the
// user asked for it.
bool iqk_amx_mul_mat_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("GGML_AMX_MUL_MAT");
        cached = (e && e[0] == '1' && ggml_amx_bf16_supported()) ? 1 : 0;
    }
    return cached == 1;
}

// C[nx x ny] = W[nx x k] (quantised) x B[ny x k]^T (f32 rows), 16 activation rows
// at a time. Parallel over activation bands via ith/nth.
bool iqk_amx_mul_mat(int nx, int ny, int k,
                     int typeA, const void * A, size_t strideA,
                     const float * B, size_t strideB,
                     float * C, size_t strideC, int ith, int nth) {
    if (!iqk_amx_mul_mat_enabled()) return false;
    if (k % 32 != 0 || !type_ok(typeA)) return false;

    const Shadow * sh = get_shadow(typeA, A, strideA, nx, k);
    if (!sh) return false;

    // bf16 copy of 16 activation rows at a time
    const int bands = (ny + 15) / 16;
    std::vector<uint16_t> abuf((size_t)16 * k);
    for (int band = ith; band < bands; band += nth) {
        const int m0 = band * 16;
        const int mrows = (ny - m0 < 16) ? (ny - m0) : 16;
        memset(abuf.data(), 0, (size_t)16 * k * 2);
        for (int r = 0; r < mrows; r++) {
            const float * b = (const float *)((const char *) B + (size_t)(m0 + r) * strideB);
            for (int j = 0; j < k; j++) abuf[(size_t)r * k + j] = f2bf(b[j]);
        }
        // strideC is in floats: C[weight_row + act_row*strideC]
        ggml_amx_gemm_bf16(mrows, sh->nx_pad, k, abuf.data(), k, sh->packed,
                           C + (size_t)m0 * strideC, strideC);
    }
    return true;
}