// Blocked int8 GEMM for AMX tiles.
//
// The previous kernel was one 16-row activation band against all N columns at
// once. That reads the whole weight matrix again for every band: at 2048
// tokens, 128 bands times 89 MB is 11.4 GB of weight traffic for a single
// matmul. This kernel fixes the loop order instead of the tile layout:
//
//   for each panel of 96 output columns       (6 accumulator tiles)
//       for each K block of kb_size
//           for each 16-row activation band
//               accumulate 16x96 into an int32 buffer that stays in cache
//       scale the int32 buffer into the f32 output
//
// Weights are then streamed exactly once per matmul, and the 96-column weight
// panel (96 x K bytes, 491 KB at K=5120) is reused by every band while it is
// still hot.
//
// Tile budget (8 tiles, all 16 rows x 64 bytes):
//   tmm0  A      16 x 64 int8
//   tmm1-6 accumulators, 16 x 16 int32 each
//   tmm7  B      16 x 64 int8, reloaded per accumulator
#include "iqk_amx.h"

#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <chrono>
#include <string.h>
#include <stdlib.h>

#if defined(__AMX_TILE__) && (defined(__GNUC__) || defined(__clang__))
#define IQK_BLK_AMX 1
#include <immintrin.h>
#else
#define IQK_BLK_AMX 0
#endif

// Instrumentation: how much wall time the microkernel itself spends.
std::atomic<long long> g_panel_calls{0};
std::atomic<long long> g_panel_ns{0};
std::atomic<long long> g_gemm_calls{0};
std::atomic<long long> g_total_ns{0};
// The epilogue (int32 -> f32 with row and column scales) is a separate pass over
// the whole output, so it needs its own number. Without it panel_s looks small
// while the dequant pass goes unmeasured.
std::atomic<long long> g_epi_ns{0};
std::atomic<long long> g_epi_elems{0};
std::atomic<long long> g_memset_ns{0};

namespace {

constexpr int PANEL_TILES = 6;                 // accumulators per panel
constexpr int PANEL_COLS  = PANEL_TILES * 16;  // 96 output columns
constexpr int TILE_BYTES  = 64;                // one tile row, int8

#if IQK_BLK_AMX
struct alignas(64) blk_cfg {
    uint8_t  palette_id = 1;
    uint8_t  start_row  = 0;
    uint8_t  reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t  rows[16]  = {};
};

// Every tile in this kernel is 16 rows of 64 bytes, which is also the int32
// layout (16 x 16 int32) the accumulators need, so one config serves all eight.
blk_cfg make_cfg() {
    blk_cfg c;
    for (int i = 0; i < 8; i++) { c.rows[i] = 16; c.colsb[i] = 64; }
    return c;
}
#endif

} // namespace

// Panel-major, K-blocked packing. Output layout for a panel p of 6 tiles:
//   out[((p*6 + j) * (K/4) + k4) * 64 + n*4 + i] = W[(p*96 + j*16 + n)*K + k4*4 + i]
// so one 64-element K step of one tile is a contiguous 1 KB read.
bool ggml_amx_pack_int8_panels(const int8_t * W, int nx, int K, int8_t * out, int * npad_out) {
    if (!W || !out || K % 64 != 0 || nx <= 0) return false;
    const int npad = ((nx + PANEL_COLS - 1) / PANEL_COLS) * PANEL_COLS;
    const int k4 = K / 4;
    memset(out, 0, (size_t)npad * K);
    for (int p = 0; p < npad / PANEL_COLS; p++)
        for (int j = 0; j < PANEL_TILES; j++) {
            int8_t * base = out + (size_t)(p * PANEL_TILES + j) * k4 * 64;
            for (int n = 0; n < 16; n++) {
                const int col = p * PANEL_COLS + j * 16 + n;
                if (col >= nx) continue;
                const int8_t * src = W + (size_t)col * K;
                for (int q = 0; q < k4; q++)
                    memcpy(base + (size_t)q * 64 + n * 4, src + q * 4, 4);
            }
        }
    if (npad_out) *npad_out = npad;
    return true;
}

// Accumulate one 16-row activation band against one weight panel over a K slice.
// acc is 16 x PANEL_COLS int32, row stride ldc (in elements), and is read back
// and rewritten, so successive K blocks add up without a separate pass.
// Load the tile config once for a whole matmul. Reloading it per panel cost
// 15x in the old kernel's measurements, and this kernel calls the microkernel
// once per panel per band.
void ggml_amx_int8_tile_begin(void) {
#if IQK_BLK_AMX
    if (!ggml_amx_supported()) return;
    const blk_cfg cfg = make_cfg();
    _tile_loadconfig(&cfg);
#endif
}

void ggml_amx_int8_tile_end(void) {
#if IQK_BLK_AMX
    if (!ggml_amx_supported()) return;
    _tile_release();
#endif
}

void ggml_amx_int8_panel_acc(const int8_t * A, size_t lda,
                             const int8_t * Wp, int panel, int K,
                             int k0, int kcount,
                             int32_t * acc, size_t ldc) {
#if IQK_BLK_AMX
    const auto t_start = std::chrono::steady_clock::now();
    if (!ggml_amx_supported() || K % 64 != 0 || kcount <= 0) return;
    const int k4 = K / 4;

    const size_t accstride = ldc * 4;
    // Tile numbers have to be immediates, so the six accumulators are unrolled.
    _tile_loadd(1, acc + 0  * 16, accstride);
    _tile_loadd(2, acc + 1  * 16, accstride);
    _tile_loadd(3, acc + 2  * 16, accstride);
    _tile_loadd(4, acc + 3  * 16, accstride);
    _tile_loadd(5, acc + 4  * 16, accstride);
    _tile_loadd(6, acc + 5  * 16, accstride);

    const int8_t * wpan = Wp + (size_t)panel * PANEL_TILES * k4 * 64;
    const size_t tstep = (size_t)k4 * 64;
    for (int k = k0; k < k0 + kcount; k += 64) {
        _tile_loadd(0, A + k, lda);
        const int8_t * wcol = wpan + (size_t)(k / 4) * 64;
        _tile_loadd(7, wcol,            TILE_BYTES); _tile_dpbssd(1, 0, 7);
        _tile_loadd(7, wcol + tstep,    TILE_BYTES); _tile_dpbssd(2, 0, 7);
        _tile_loadd(7, wcol + 2*tstep,  TILE_BYTES); _tile_dpbssd(3, 0, 7);
        _tile_loadd(7, wcol + 3*tstep,  TILE_BYTES); _tile_dpbssd(4, 0, 7);
        _tile_loadd(7, wcol + 4*tstep,  TILE_BYTES); _tile_dpbssd(5, 0, 7);
        _tile_loadd(7, wcol + 5*tstep,  TILE_BYTES); _tile_dpbssd(6, 0, 7);
    }

    _tile_stored(1, acc + 0 * 16, accstride);
    _tile_stored(2, acc + 1 * 16, accstride);
    _tile_stored(3, acc + 2 * 16, accstride);
    _tile_stored(4, acc + 3 * 16, accstride);
    _tile_stored(5, acc + 4 * 16, accstride);
    _tile_stored(6, acc + 5 * 16, accstride);

    g_panel_calls++;
    g_panel_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t_start).count();
#else
    (void)A; (void)lda; (void)Wp; (void)panel; (void)K; (void)k0; (void)kcount;
    (void)acc; (void)ldc;
#endif
}

// Full matmul: C[ny x nx] f32 = (A[ny x K] int8 * scales) x W[nx x K]^T.
// npad must be the value reported by the packing call; accbuf needs
// bands * 16 * npad int32 (786 KB at 2048 tokens and N=17408).
void ggml_amx_int8_gemm_blocked(int ny, int K, const int8_t * A, size_t lda,
                                const int8_t * Wp, int npad,
                                int32_t * accbuf,
                                float * C, size_t ldc_f,
                                const float * row_scale, const float * a_scale,
                                int kb_size, int only_panel) {
#if IQK_BLK_AMX
    if (!ggml_amx_supported() || K % 64 != 0) return;
    if (kb_size <= 0 || kb_size % 64) kb_size = K;
    ggml_amx_int8_tile_begin();
    const int bands = (ny + 15) / 16;
    long long epi_elems = 0;
    const float * rscale = row_scale;
    float * out = C;

    for (int p = 0; p < npad / PANEL_COLS; p++) {
        if (only_panel >= 0 && p != only_panel) continue;
        // The accumulator only needs to hold one panel: 16 rows x 96 columns per
        // band. Striding it by npad made this memset 142 MB per panel, which is
        // 25 GB of cache-destroying traffic per matmul.
        const auto t_ms0 = std::chrono::steady_clock::now();
        memset(accbuf, 0, (size_t)bands * 16 * PANEL_COLS * sizeof(int32_t));
        g_memset_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_ms0).count();

        for (int kb = 0; kb < K; kb += kb_size) {
            const int kcount = (K - kb < kb_size) ? (K - kb) : kb_size;
            for (int band = 0; band < bands; band++) {
                int32_t * acc = accbuf + (size_t)band * 16 * PANEL_COLS;
                ggml_amx_int8_panel_acc(A + (size_t)band * 16 * lda, lda, Wp, p, K,
                                        kb, kcount, acc, PANEL_COLS);
            }
        }

        const auto t_epi0 = std::chrono::steady_clock::now();
        for (int band = 0; band < bands; band++) {
            const int m0 = band * 16;
            const int mrows = (ny - m0 < 16) ? (ny - m0) : 16;
            const int32_t * acc = accbuf + (size_t)band * 16 * PANEL_COLS;
            for (int r = 0; r < mrows; r++) {
                const float as = a_scale ? a_scale[m0 + r] : 1.0f;
                float * dst = out + (size_t)(m0 + r) * ldc_f;
                for (int j = 0; j < PANEL_COLS; j++) {
                    const int col = p * PANEL_COLS + j;
                    if (col >= (int)ldc_f) break;
                    dst[col] = as * (rscale ? rscale[col] : 1.0f) * (float)acc[r * PANEL_COLS + j];
                    ++epi_elems;
                }
            }
        }
        g_epi_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_epi0).count();
    }
    g_epi_elems += epi_elems;
    ggml_amx_int8_tile_end();
#else
    (void)ny; (void)K; (void)A; (void)lda; (void)Wp; (void)npad; (void)accbuf;
    (void)C; (void)ldc_f; (void)row_scale; (void)a_scale; (void)kb_size; (void)only_panel;
#endif
}
// Reported only with GGML_AMX_STATS so a normal run stays quiet.
__attribute__((constructor)) static void iqk_blk_stats_hook(void) {
    static bool installed = false;
    if (installed) return;
    installed = true;
    if (!getenv("GGML_AMX_STATS")) return;
    atexit([] {
        fprintf(stderr, "amx_blk_stats: panel_calls=%lld panel_s=%.1f epi_s=%.1f epi_elems=%.1fM memset_s=%.1f\n",
                g_panel_calls.load(), g_panel_ns.load() / 1e9,
                g_epi_ns.load() / 1e9, g_epi_elems.load() / 1e6,
                g_memset_ns.load() / 1e9);
    });
}
