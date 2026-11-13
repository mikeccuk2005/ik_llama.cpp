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
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include <functional>
#include <condition_variable>
#include <omp.h>
#include <sched.h>
#include <immintrin.h>
#include <sys/mman.h>

namespace {

struct Shadow {
    std::shared_ptr<std::vector<uint16_t>> packed;  // n_pad x K, VNNI order
    int nx_pad = 0;
    int k = 0;
};

// Values are held behind shared_ptr: the map keeps growing while compute
// threads read, so handing out &it->second would dangle the moment a rehash
// moved the node, and the OpenMP team would then hang at a barrier.
std::mutex g_mu;

// Keyed by pointer AND shape: a merged QKV tensor shares its base pointer with
// the separate Q/K/V tensors but has a different nx, so pointer alone would hand
// back a shadow of the wrong width.
struct ShadowKey {
    const void * p;
    int nx;
    int k;
    bool operator==(const ShadowKey & o) const { return p == o.p && nx == o.nx && k == o.k; }
};
struct ShadowKeyHash {
    size_t operator()(const ShadowKey & s) const {
        size_t h = std::hash<const void *>()(s.p);
        h ^= std::hash<int>()(s.nx) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(s.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
std::unordered_map<ShadowKey, std::shared_ptr<Shadow>, ShadowKeyHash> g_shadows;

// Engagement counters, printed at exit when GGML_AMX_STATS=1. Without these we
// cannot tell "the kernel fired and did not help" from "the kernel never fired".
std::atomic<long long> g_calls{0};
std::atomic<long long> g_gmac{0};
std::atomic<long long> g_builds{0};
std::atomic<long long> g_fast{0};
std::atomic<long long> g_path_ns{0};
std::atomic<long long> g_quant_ns{0};
std::atomic<long long> g_lookup_ns{0};
std::atomic<long long> g_quantloop_ns{0};
std::atomic<long long> g_wide_calls{0};
std::atomic<long long> g_wide_ns{0};
std::atomic<long long> g_ny_max{0};
std::atomic<long long> g_path_calls{0};
std::atomic<long long> g_ith_seen[128] = {};
std::atomic<long long> g_ith_calls[128] = {};   // whole-matrix path (nth == 1)
std::atomic<long long> g_band{0};   // caller already split (nth > 1)
std::atomic<long long> g_build_ns{0};  // wall time spent building shadows

bool quant_row_i8(const float * src, int8_t * dst, int k, float * out_scale);

__attribute__((constructor)) void amx_stats_register() {
    if (getenv("GGML_AMX_STATS")) {
        std::atexit([] {
            {
            int distinct = 0;
            for (int i = 0; i < 128; i++) if (g_ith_seen[i].load()) distinct++;
            fprintf(stderr, "amx_ith: distinct_ith=%d", distinct);
            for (int i = 0; i < 128; i++)
                if (g_ith_seen[i].load() && i < 70) fprintf(stderr, " %d:%lld", i, g_ith_seen[i].load());
            fprintf(stderr, "\n");
        }
        {
#if defined(__AVX512F__)
            const char * qp = "avx512";
#else
            const char * qp = "scalar";
#endif
            const int kq = 5120, rowsq = 16;
            std::vector<float> sbuf((size_t)kq * rowsq, 0.5f);
            std::vector<int8_t> dbuf((size_t)kq * rowsq);
            float sc = 1.0f;
            const int reps = 200;
            auto q0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++)
                for (int row = 0; row < rowsq; row++)
                    quant_row_i8(sbuf.data() + (size_t)row * kq, dbuf.data() + (size_t)row * kq, kq, &sc);
            const double qs = std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
            const double mb = (double)sizeof(float) * kq * rowsq * reps / 1e6;
            fprintf(stderr, "wide: calls=%lld ns_s=%.1f ny_max=%lld\n",
                    g_wide_calls.load(), g_wide_ns.load() / 1e9, g_ny_max.load());
            fprintf(stderr, "split: lookup_s=%.1f quantloop_s=%.1f\n",
                    g_lookup_ns.load() / 1e9, g_quantloop_ns.load() / 1e9);
            fprintf(stderr, "quant_probe: path=%s %.1f MB/s per call %.3f ms\n",
                    qp, mb / qs, 1000.0 * qs / reps);
        }
        fprintf(stderr, "amx_stats: calls=%lld gmac=%.1f shadows=%lld build_s=%.1f fast=%lld band=%lld path_s=%.1f quant_s=%.1f path_calls=%lld\n",
                    g_calls.load(), g_gmac.load() / 1e9, g_builds.load(),
                    g_build_ns.load() / 1e9, g_fast.load(), g_band.load(),
                    g_path_ns.load() / 1e9, g_quant_ns.load() / 1e9, g_path_calls.load());
        });
    }
}

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
//
// The dequant pass runs on worker threads: the caller is inside an OpenMP
// region where 127 threads are parked at a barrier, so building 7.2B weights on
// one core costs about a minute of dead time per process. Splitting the rows
// across threads turns that into a couple of seconds.
std::shared_ptr<const Shadow> get_shadow(int type, const void * A, size_t strideA, int nx, int k) {
    const ShadowKey key{ A, nx, k };
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_shadows.find(key);
        if (it != g_shadows.end()) return it->second;
    }
    // Build without the lock. Holding it across a build serialises every caller
    // (and the load-time prewarm loop) into one core's worth of work.

    const int nx_pad = ((nx + 47) / 48) * 48;
    const size_t bytes = (size_t)nx_pad * k * sizeof(uint16_t);
    auto owned = std::make_shared<std::vector<uint16_t>>((bytes + 1) / 2, 0);
    uint16_t * packed = owned->data();
    if (!packed) return nullptr;

    const auto t0 = std::chrono::steady_clock::now();
    uint16_t * dense = (uint16_t *) aligned_alloc(64, (size_t)nx_pad * k * 2);
    if (!dense) return nullptr;
    memset(dense, 0, (size_t)nx_pad * k * 2);

    // Serial by design: spawning threads from inside the OpenMP region deadlocks
    // the team (workers block while the region's threads sit at a barrier).
    std::vector<float> row(k);
    for (int n = 0; n < nx; n++) {
        const void * src = (const char *) A + (size_t)n * strideA;
        if (!dequant_row(type, src, row.data(), k)) { free(dense); return nullptr; }
        uint16_t * dst = dense + (size_t)n * k;
        for (int j = 0; j < k; j++) dst[j] = f2bf(row[j]);
    }

    if (!ggml_amx_pack_bf16(dense, nx_pad, k, packed)) { free(dense); return nullptr; }
    free(dense);

    g_build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();

    auto sh = std::make_shared<Shadow>();
    sh->packed = std::move(owned);
    sh->nx_pad = nx_pad;
    sh->k = k;
    g_builds++;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_shadows.find(key);
        if (it != g_shadows.end()) return it->second;  // someone else won the race
        g_shadows[key] = sh;
    }
    return sh;
}

} // namespace

// ---- int8 (W8A8) path, for weights that are already int8 ----------------
//
// Q8_0 stores int8 quants with one scale per 32 values. Keeping the per-32
// scales would force a float fixup over the whole 16 x N tile for every block,
// which costs more traffic than the weights themselves. So the shadow collapses
// the block scales into one per row (per output column for weights, per token
// row for activations) and the tile gemm runs once over the full K with a single
// scale pass at the end. One byte per weight, no widening, int8 tiles at 2x the
// bf16 tile rate.

namespace {

// Shadow buffers run to gigabytes. THP is madvise-only on this box, so a plain
// std::vector leaves them on 4 KB pages and every tile load pays TLB misses
// that the AVX-512 path never incurs.
struct AmxBuf {
    int8_t * p = nullptr;
    size_t   n = 0;

    bool alloc(size_t bytes) {
        void * q = nullptr;
        if (posix_memalign(&q, 64, bytes) != 0 || !q) return false;
        p = (int8_t *) q;
        n = bytes;
        memset(p, 0, bytes);
        madvise(p, bytes, MADV_HUGEPAGE);
        return true;
    }
    ~AmxBuf() { free(p); }
    int8_t * data() const { return p; }
    size_t   size() const { return n; }
};

struct ShadowI8 {
    AmxBuf packed;                // VNNI order, nx_pad x k
    std::vector<float>  row_scale; // per output column
    int nx_pad = 0;
    int k = 0;
};

std::unordered_map<ShadowKey, std::shared_ptr<ShadowI8>, ShadowKeyHash> g_shadows_i8;

// Single-flight per tensor. Without this, every thread of the ggml team that
// calls the hook at the same time misses the cache and builds its own copy of
// the same 7 MB shadow: 2832 builds instead of 498 in one measured run.
std::unordered_map<ShadowKey, std::shared_ptr<std::mutex>, ShadowKeyHash> g_build_mu;

// Shared quantised activations for one matmul. Splitting panels across the
// team means every thread needs every activation row, and letting each thread
// quantise its own copy made that pass 90% of the runtime (1268 s of 1402 s).
// One thread converts its slice, they meet on a condition variable, and the
// slot is fingerprinted so a recycled activation buffer is never mistaken for
// the previous layer's data.
struct AqShared {
    // Park, do not spin. Measured at 128 threads: a spin barrier cost 5%
    // (230.6 against 244.2) because waiters fight the workers for cores. The
    // futex wakeup was never the ceiling. done and next_panel still get their
    // own cache lines: both are written by every thread.
    std::mutex m;
    std::condition_variable cv;
    alignas(64) std::atomic<int> done{0};
    alignas(64) std::atomic<int> next_panel{0};   // per matmul, one reset
    std::atomic<bool> ready{false};
    std::vector<int8_t> q;
    std::vector<float>  s;
};

std::mutex g_aq_mu;
std::shared_ptr<AqShared> g_aq;
const void * g_aq_key = nullptr;
int g_aq_ny = 0, g_aq_k = 0;
float g_aq_fp[4] = {0, 0, 0, 0};

// Sample a few values so we can tell a reused buffer from fresh data.
static void aq_fingerprint(const void * B, int k, float out[4]) {
    const float * f = (const float *) B;
    out[0] = f[0];
    out[1] = f[k / 3];
    out[2] = f[(2 * k) / 3];
    out[3] = f[k - 1];
}

std::shared_ptr<AqShared> aq_slot(const void * B, int ny, int k, size_t need) {
    float fp[4];
    aq_fingerprint(B, k, fp);
    std::lock_guard<std::mutex> lk(g_aq_mu);
    if (g_aq_key != B || g_aq_ny != ny || g_aq_k != k ||
        g_aq_fp[0] != fp[0] || g_aq_fp[1] != fp[1] ||
        g_aq_fp[2] != fp[2] || g_aq_fp[3] != fp[3]) {
        g_aq = std::make_shared<AqShared>();
        g_aq->q.resize(need);
        g_aq->s.resize(need / (size_t)k);
        g_aq_key = B; g_aq_ny = ny; g_aq_k = k;
        for (int i = 0; i < 4; i++) g_aq_fp[i] = fp[i];
    }
    return g_aq;
}

std::shared_ptr<std::mutex> build_mutex_for(const ShadowKey & key) {
    auto it = g_build_mu.find(key);
    if (it != g_build_mu.end()) return it->second;
    auto m = std::make_shared<std::mutex>();
    g_build_mu.emplace(key, m);
    return m;
}

// Quantise one dequantised row to int8 with a single scale for the row.
// Scalar reference: one pass for amax, one for the conversion. Kept for the
// no-AVX-512 build and as the thing to check the vector path against.
static bool quant_row_i8_scalar(const float * src, int8_t * dst, int k, float * out_scale) {
    float amax = 0.0f;
    for (int j = 0; j < k; j++) {
        const float a = std::fabs(src[j]);
        if (a > amax) amax = a;
    }
    if (amax <= 0.0f) {
        memset(dst, 0, (size_t)k);
        *out_scale = 1.0f;
        return true;
    }
    const float s = amax / 127.0f;
    const float inv = 1.0f / s;
    for (int j = 0; j < k; j++) {
        int v = (int)lrintf(src[j] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        dst[j] = (int8_t)v;
    }
    *out_scale = s;
    return true;
}

// Convert one f32 row to int8 with a symmetric row scale. Sixteen lanes at a
// time: abs+max in the first pass, then multiply, convert, clamp and pack. The
// scalar version ran at tens of MB/s and was 85% of the AMX path's time.
bool quant_row_i8(const float * src, int8_t * dst, int k, float * out_scale) {
#if defined(__AVX512F__)
    int j = 0;
    __m512 vmax = _mm512_setzero_ps();
    for (; j + 16 <= k; j += 16) {
        const __m512 v = _mm512_loadu_ps(src + j);
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v));
    }
    float amax = _mm512_reduce_max_ps(vmax);
    for (; j < k; j++) {
        const float a = std::fabs(src[j]);
        if (a > amax) amax = a;
    }
    if (amax <= 0.0f) {
        memset(dst, 0, (size_t)k);
        *out_scale = 1.0f;
        return true;
    }
    const float s = amax / 127.0f;
    const float inv = 1.0f / s;
    const __m512 vinv  = _mm512_set1_ps(inv);
    const __m512i vhi  = _mm512_set1_epi32(127);
    const __m512i vlo  = _mm512_set1_epi32(-127);
    j = 0;
    for (; j + 16 <= k; j += 16) {
        __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(src + j), vinv));
        q = _mm512_max_epi32(q, vlo);
        q = _mm512_min_epi32(q, vhi);
        _mm_storeu_si128((__m128i *)(dst + j), _mm512_cvtepi32_epi8(q));
    }
    for (; j < k; j++) {
        int v = (int)lrintf(src[j] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        dst[j] = (int8_t)v;
    }
    *out_scale = s;
    return true;
#else
    return quant_row_i8_scalar(src, dst, k, out_scale);
#endif
}

// Direct-mapped, lock-free lookup published after the shadow is built. The
// owning map never erases, so a raw pointer stays valid for the process and the
// hot path never touches a mutex or a refcount. Readers key on the tensor
// pointer, and a slot reused by another tensor simply reads as a miss.
namespace {
constexpr int kShadowSlots = 4096;
struct ShadowSlot {
    std::atomic<const void *>    a{nullptr};
    std::atomic<int>             nx{0};
    std::atomic<int>             k{0};
    std::atomic<const ShadowI8 *> sh{nullptr};
};
ShadowSlot g_slots[kShadowSlots];

void shadow_publish(const void * A, int nx, int k, const ShadowI8 * sh) {
    const size_t h = (((uintptr_t)A >> 6) * 0x9E3779B97F4A7C15ULL
                      ^ (size_t)(nx * 131 + k)) & (kShadowSlots - 1);
    g_slots[h].sh.store(sh, std::memory_order_release);
    g_slots[h].nx.store(nx, std::memory_order_relaxed);
    g_slots[h].k.store(k, std::memory_order_relaxed);
    g_slots[h].a.store(A, std::memory_order_release);
}
} // namespace

// Build (or fetch) the int8 shadow of a Q8_0 weight tensor.
const ShadowI8 * get_shadow_i8(const void * A, size_t strideA, int nx, int k) {
    const ShadowKey key{ A, nx, k };
    const size_t h = (((uintptr_t)A >> 6) * 0x9E3779B97F4A7C15ULL
                      ^ (size_t)(nx * 131 + k)) & (kShadowSlots - 1);
    if (g_slots[h].a.load(std::memory_order_acquire) == A &&
        g_slots[h].nx.load(std::memory_order_relaxed) == nx &&
        g_slots[h].k.load(std::memory_order_relaxed) == k) {
        if (const ShadowI8 * hit = g_slots[h].sh.load(std::memory_order_acquire)) return hit;
    }

    std::shared_ptr<std::mutex> kmu;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_shadows_i8.find(key);
        if (it != g_shadows_i8.end()) {
            shadow_publish(A, nx, k, it->second.get());
            return it->second.get();
        }
        kmu = build_mutex_for(key);
    }
    std::lock_guard<std::mutex> keylock(*kmu);   // one builder per tensor
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_shadows_i8.find(key);
        if (it != g_shadows_i8.end()) {
            shadow_publish(A, nx, k, it->second.get());
            return it->second.get();
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    // Panels are 6 tiles = 96 columns, so nx is padded to a whole number of panels.
    const int nx_pad = ((nx + 95) / 96) * 96;

    auto sh = std::make_shared<ShadowI8>();
    if (!sh->packed.alloc((size_t)nx_pad * k)) return nullptr;
    sh->row_scale.assign((size_t)nx_pad, 1.0f);
    sh->nx_pad = nx_pad;
    sh->k = k;

    std::vector<int8_t> dense((size_t)nx_pad * k, 0);
    std::vector<float> row(k);

    for (int n = 0; n < nx; n++) {
        const void * src = (const char *) A + (size_t)n * strideA;
        if (!dequant_row(GGML_TYPE_Q8_0, src, row.data(), k)) return nullptr;
        if (!quant_row_i8(row.data(), dense.data() + (size_t)n * k, k, &sh->row_scale[n])) return nullptr;
    }
    int npad_out = 0;
    if (!ggml_amx_pack_int8_panels(dense.data(), nx_pad, k, sh->packed.data(), &npad_out)) return nullptr;
    if (npad_out != nx_pad) return nullptr;

    g_build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    g_builds++;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_shadows_i8.find(key);
        if (it != g_shadows_i8.end()) {
            shadow_publish(A, nx, k, it->second.get());
            return it->second.get();
        }
        g_shadows_i8[key] = sh;
    }
    shadow_publish(A, nx, k, sh.get());
    return sh.get();
}

// Persistent work-sharing pool. One thread per core lives for the whole process
// and spins between calls. Creating threads per matmul cost us a 2.4x swings
// between identical runs (97 vs 237 t/s), because the hook is called hundreds of
// times and each call paid 64 thread creations.
//
// Not OpenMP: this runs inside the region the ggml team owns, where nested
// pragmas either serialise or park everyone at a barrier.
class AmxWorkPool {
public:
    static AmxWorkPool & get() {
        static AmxWorkPool pool;
        return pool;
    }

    // Run fn(i) for i in [0, n). The caller works too, so no core idles.
    void run(int n, const std::function<void(int)> & fn) {
        if (n <= 0) return;
        if (n == 1) { fn(0); return; }
        // The hook is called by every thread of the ggml team, so several
        // submitters can arrive at once. One job slot means they queue.
        std::lock_guard<std::mutex> submit(run_m_);
        if (busy_) { for (int i = 0; i < n; i++) fn(i); return; }

        const int want = std::min(n, usable_cpus()) - 1;
        ensure(want);
        if (want <= 0) { for (int i = 0; i < n; i++) fn(i); return; }

        {
            std::lock_guard<std::mutex> lk(m_);
            job_      = &fn;
            n_        = n;
            next_     = 0;
            arrived_  = 0;
            batch_   += 1;
            busy_    = true;
        }
        cv_.notify_all();

        for (;;) {
            const int i = next_.fetch_add(1);
            if (i >= n) break;
            fn(i);
        }
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return arrived_.load() >= workers_; });
        job_ = nullptr;
        busy_ = false;
    }

private:
    // CPUs this process may actually run on. hardware_concurrency() reports the
    // whole box, so a taskset/numactl run would get 127 workers on 32 cores.
    static int usable_cpus() {
        static int cached = [] {
            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) == 0) {
                const int n = CPU_COUNT(&set);
                if (n > 0) return n;
            }
            const unsigned hw = std::thread::hardware_concurrency();
            return hw ? (int)hw : 1;
        }();
        return cached;
    }

    // Workers park on cv_ forever, so the process must tell them to stop and
    // join them. Without this a run that actually used the pool (any call with
    // nth == 1) hangs on exit while -t 4 and above, which never start a worker,
    // exit cleanly.
    ~AmxWorkPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto & th : threads_) if (th.joinable()) th.join();
    }

    void ensure(int want) {
        std::lock_guard<std::mutex> lk(m_);
        while (workers_ < want) {
            threads_.emplace_back([this] { loop(); });
            workers_++;
        }
    }

    void loop() {
        unsigned long long seen = 0;
        for (;;) {
            const std::function<void(int)> * fn = nullptr;
            int n = 0;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return batch_ != seen || quit_; });
                if (quit_) return;
                seen = batch_;
                fn = job_;
                n  = n_;
            }
            for (;;) {
                const int i = next_.fetch_add(1);
                if (i >= n) break;
                (*fn)(i);
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                arrived_.fetch_add(1);
            }
            done_cv_.notify_all();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex run_m_;
    bool busy_ = false;
    std::mutex m_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    unsigned long long batch_ = 0;
    int workers_ = 0;
    int n_ = 0;
    std::atomic<int> next_{0};
    std::atomic<int> arrived_{0};
    std::atomic<bool> quit_{false};
    const std::function<void(int)> * job_ = nullptr;
};

void amx_parallel_for(int n, const std::function<void(int)> & fn) {
    AmxWorkPool::get().run(n, fn);
}

bool amx_int8_available(void) {
    static int cached = -1;
    if (cached < 0) cached = ggml_amx_supported() ? 1 : 0;
    return cached == 1;
}

// C[nx x ny] = W[nx x k] (Q8_0) x B[ny x k]^T, int8 tiles, 16 rows at a time.
bool amx_mul_mat_i8(int nx, int ny, int k, const void * A, size_t strideA,
                    const float * B, size_t strideB,
                    float * C, size_t strideC, int ith, int nth) {
    const auto t_enter = std::chrono::steady_clock::now();
    const ShadowI8 * sh = get_shadow_i8(A, strideA, nx, k);
    if (!sh) return false;
    const auto t_call = std::chrono::steady_clock::now();
    if (getenv("GGML_AMX_STATS"))
        g_lookup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_call - t_enter).count();
    if (ith >= 0 && ith < 128) g_ith_seen[ith].fetch_add(1);
    const int npad = sh->nx_pad;              // a multiple of 96
    const int bands = (ny + 15) / 16;
    // ggml hands every thread the whole matrix plus (ith, nth), expecting the
    // callee to take its slice. We slice by column panel, not by row: each
    // thread then owns a few 96-column weight panels, walks every activation
    // band against them, and the team reads each weight exactly once. Slicing
    // by row instead makes every thread stream the entire weight matrix.
    const int npanels = npad / 96;

    // Quantise every activation row once, padded to whole bands.
    // thread_local variables are not captured by [&]: they have static storage
    // duration, so a lambda body refers to whatever thread executes it. Reading
    // them through ordinary references pins the calling thread's buffers, which
    // is what the pool workers must write into.
    static thread_local std::vector<int8_t> aq_tl;
    static thread_local std::vector<float>  asc_tl;
    std::vector<int8_t> & aq  = aq_tl;
    std::vector<float>  & asc = asc_tl;
    const int rows_pad = bands * 16;
    const size_t need = (size_t)rows_pad * k;
    if (aq.size() < need) aq.resize(need);
    if (asc.size() < (size_t)rows_pad) asc.resize(rows_pad);

    auto quant_row = [&](int r) {
        const float * b = (const float *)((const char *) B + (size_t)r * strideB);
        quant_row_i8(b, aq.data() + (size_t)r * k, k, &asc[r]);
    };
    // Quantised activations: shared across the team for this matmul. Each
    // thread converts its own slice and they meet on a condition variable; the
    // wait has a timeout and falls back to converting the rest itself, so a
    // thread that never shows up can cost time but can never hang the graph.
    const int8_t * aqp = nullptr;
    const float *  ascp = nullptr;
    std::shared_ptr<AqShared> slot;
    // A call carrying one band is 16 activation rows, 80 KB at k=5120. Sharing
    // that through a 128-thread barrier costs more than converting it again, so
    // every thread converts its own copy and nobody waits. Only wider calls are
    // worth synchronising.
    if (ny > g_ny_max.load()) g_ny_max.store(ny);
    const auto t_wide0 = std::chrono::steady_clock::now();
    if (nth > 1 && ny > 16) {
        g_wide_calls.fetch_add(1);
        slot = aq_slot(B, ny, k, need);
        const int chunk = (ny + nth - 1) / nth;
        const int lo = ith * chunk;
        const int hi = (lo + chunk < ny) ? (lo + chunk) : ny;
        for (int r = lo; r < hi; r++) quant_row(r);
        {
            // Team join for the wide calls, measured at 3.9 ms each. A spin
            // loop on an atomic counter was tried and measured worse (231-236
            // t/s against 245): 128 threads spinning steal cycles from the
            // threads still converting. The parked wait stays.
            std::unique_lock<std::mutex> lk(slot->m);
            slot->done.fetch_add(1);
            slot->cv.notify_all();
            if (!slot->cv.wait_for(lk, std::chrono::seconds(2),
                                   [&] { return slot->done.load() >= nth; })) {
                lk.unlock();
                for (int r = 0; r < ny; r++) quant_row(r);
            }
        }
        aqp = slot->q.data();
        ascp = slot->s.data();
        if (getenv("GGML_AMX_STATS"))
            g_wide_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_wide0).count();
    } else {
        // We are already one of the caller's threads and this call owns at most
        // a band of rows. Submitting that to the shared pool made 128 callers
        // queue work on the same workers and wait for it, which measured as 85%
        // of the path's time. Just convert here.
        const auto t_q0 = std::chrono::steady_clock::now();
        for (int r = 0; r < ny; r++) quant_row(r);
        if (getenv("GGML_AMX_STATS"))
            g_quantloop_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_q0).count();
        aqp = aq.data();
        ascp = asc.data();
    }
    if (rows_pad > ny) {
        memset(aq.data() + (size_t)ny * k, 0, (size_t)(rows_pad - ny) * k);
        for (int r = ny; r < rows_pad; r++) asc[r] = 1.0f;
    }

    // Parallel over column panels and serial over bands inside each one: a
    // thread walks one 96-column weight panel, which is 491 KB at K=5120 and
    // stays in its L2 while every band reuses it. The alternative, bands
    // outside panels, streams the whole weight matrix once per 16 rows.
    auto work_panel = [&](int p) {
        // The tile config is per-thread, so each pool worker loads it once for
        // the life of the process instead of once per panel. Both tile formats
        // use 16 rows x 64 bytes, so a bf16 call clobbering it is harmless.
        static thread_local bool cfg_ok = false;
        if (!cfg_ok) { ggml_amx_int8_tile_begin(); cfg_ok = true; }
        alignas(64) int32_t acc[16 * 96];
        for (int band = 0; band < bands; band++) {
            const int m0 = band * 16;
            const int mrows = (ny - m0 < 16) ? (ny - m0) : 16;
            memset(acc, 0, sizeof(acc));
            ggml_amx_int8_panel_acc(aqp + (size_t)m0 * k, (size_t)k,
                                    sh->packed.data(), p, k, 0, k, acc, 96);
            for (int r = 0; r < mrows; r++) {
                const float as = ascp[m0 + r];
                float * dst = C + (size_t)(m0 + r) * strideC;
                const int32_t * row = acc + (size_t)r * 96;
                const int base = p * 96;
                const int lim = (base + 96 <= nx) ? 96 : (nx - base);
                for (int j = 0; j < lim; j++) {
                    dst[base + j] = as * sh->row_scale[base + j] * (float)row[j];
                }
            }
        }
    };

    const auto t_panels = std::chrono::steady_clock::now();
    if (nth > 1 && slot) {
        // Wider calls: panels outnumber nothing in particular, so a shared
        // counter balances the tail.
        for (int p = slot->next_panel.fetch_add(1); p < npanels;
             p = slot->next_panel.fetch_add(1))
            work_panel(p);
    } else if (nth > 1) {
        // One band per call: stride by thread index, touching no shared state.
        // A stride, not a single index, because nx=17408 gives 182 panels which
        // is more than the thread count.
        for (int p = ith; p < npanels; p += nth) work_panel(p);
    } else {
        amx_parallel_for(npanels, work_panel);
    }
    if (getenv("GGML_AMX_STATS")) {
        g_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_call).count();
        g_quant_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_panels - t_call).count();
        g_path_calls++;
        for (int i = 0; i < 128; i++)
            if (g_ith_seen[i].load()) g_ith_calls[i].fetch_add(1);
    }
    return true;
}

} // namespace

// True when the AMX mul_mat path is compiled in and the CPU can run it.
// The build option (-DGGML_AMX=ON) is the only switch; there is no runtime env
// gate, because a silent "compiled in but never used" mode is exactly how we
// ended up measuring noise and calling it a kernel.
bool iqk_amx_mul_mat_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_amx_bf16_supported() ? 1 : 0;
    }
    return cached == 1;
}

// Build the shadow for one weight tensor ahead of compute. Call this at model
// load, outside any OpenMP region, so all cores can be used; building lazily
// from inside a compute region serialises on the cache mutex and costs minutes.
void iqk_amx_prewarm(int type, const void * A, size_t strideA, int nx, int k) {
    if (!iqk_amx_mul_mat_enabled()) return;
    // The band loop fans out with its own parallel for inside the caller's
    // region, so nesting has to be allowed or those bands run serially.
    static bool nested_ok = [] { omp_set_max_active_levels(2); return true; }();
    (void)nested_ok;
    if (k % 32 != 0 || nx < 16) return;
    if (type == GGML_TYPE_Q8_0 && k % 64 == 0 && amx_int8_available() && !getenv("GGML_AMX_NO_I8")) {
        (void)get_shadow_i8(A, strideA, nx, k);
        return;
    }
    if (type_ok(type)) (void)get_shadow(type, A, strideA, nx, k);
}

// C[nx x ny] = W[nx x k] (quantised) x B[ny x k]^T (f32 rows), 16 activation rows
// at a time. Parallel over activation bands via ith/nth.
bool iqk_amx_mul_mat(int nx, int ny, int k,
                     int typeA, const void * A, size_t strideA,
                     const float * B, size_t strideB,
                     float * C, size_t strideC, int ith, int nth) {
    if (!iqk_amx_mul_mat_enabled()) return false;
    if (k % 32 != 0 || !type_ok(typeA)) return false;

    // Q8_0 weights are already int8, so they take the W8A8 path: byte-identical
    // shadow, no widening, 2x the tile throughput of bf16.
    // GGML_AMX_NO_I8=1 forces Q8_0 through the bf16 path, so the two tile
    // formats can be compared on identical weights from one binary.
    static const bool no_i8 = getenv("GGML_AMX_NO_I8") != nullptr;
    if (!no_i8 && typeA == GGML_TYPE_Q8_0 && k % 64 == 0 && amx_int8_available()) {
        g_calls++;
        g_gmac += (long long)nx * ny * k;
        if (amx_mul_mat_i8(nx, ny, k, A, strideA, B, strideB, C, strideC, ith, nth)) return true;
    }

    std::shared_ptr<const Shadow> sh = get_shadow(typeA, A, strideA, nx, k);
    if (!sh) return false;

    g_calls++;
    g_gmac += (long long)nx * ny * k;

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
        ggml_amx_gemm_bf16(mrows, sh->nx_pad, k, abuf.data(), k, sh->packed->data(),
                           C + (size_t)m0 * strideC, strideC);
    }
    return true;
}