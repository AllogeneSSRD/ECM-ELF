#include "../../kernels/cuda/stage2_ntt_twist.cuh"
#include "ecm/cuda/ntt/runtime.h"
#include "ecm/platform/program_version.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

using Word = unsigned long long;
namespace twist = ecm::stage2::ntt_twist;

static unsigned ceil_log2(unsigned n)
{
    unsigned k = 0;
    for (--n; n; n >>= 1) ++k;
    return k;
}

static unsigned read_uint(std::string_view arg)
{
    unsigned value;
    const auto end = arg.data() + arg.size();
    const auto result = std::from_chars(arg.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        std::fprintf(stderr, "Invalid unsigned argument: %s\n", arg.data());
        std::exit(2);
    }
    return value;
}

struct Transform {
    FuseCtx ctx{};
    Word root, inverse_root, scale;

    Transform(Word n, int k, Word omega)
        : root(omega), inverse_root(gl_pow_host(omega, GL_P - 2)),
          scale(gl_pow_host(n, GL_P - 2))
    {
        fuse_init(ctx, n, k, root, inverse_root);
        ntt_fuse_cache_tables(ctx, root, inverse_root);
    }

    ~Transform() { fuse_release(ctx); }

    void multiply(Word* a, Word* b, unsigned batch, cudaStream_t stream) const
    {
        ntt_forward_fused(a, ctx, root, batch, stream);
        ntt_forward_fused(b, ctx, root, batch, stream);
        ntt_inverse_fused(a, b, ctx, inverse_root, scale, batch, stream);
    }
};

static void print_outer(const char* label, const FuseCtx& ctx)
{
    std::printf(" %s=", label);
    if (!ctx.nms) std::printf("-");
    for (int i = 0; i < ctx.nms; ++i) std::printf("%s%d", i ? "," : "", ctx.ms[i]);
}

static void fill_packed(std::vector<Word>& input, const NttCfg& cfg, unsigned bits,
                        unsigned slots, unsigned batch, std::mt19937_64& rng)
{
    const Word half = cfg.nwords / 2;
    const Word digits = (bits + cfg.bpw - 1) / cfg.bpw;
    for (unsigned s = 0; s < batch; ++s) {
        for (unsigned j = 0; j < slots; ++j) {
            for (Word d = 0; d < digits; ++d) {
                const auto width = std::min<Word>(cfg.bpw, bits - d * cfg.bpw);
                const Word mask = (Word{1} << width) - 1;
                const Word value = j == 0 || j == slots - 1 ? mask : (rng() & mask);
                input[s * half + j * cfg.slot_words + d] = value;
            }
        }
    }
}

static void check_word(const Word* a, const Word* b, Word used, Word i, Word got)
{
    const Word first = i < used ? 0 : i - used + 1;
    const Word last = std::min(i, used - 1);
    Word expected = 0;
    for (Word j = first; j <= last; ++j) expected += a[j] * b[i - j];
    if (got != expected) {
        std::fprintf(stderr, "Integer oracle mismatch at %llu: %llu != %llu\n", i, got, expected);
        std::exit(1);
    }
}

int main(int argc, char** argv)
{
    const std::string_view submission = argc == 7 ? argv[6] : "graph";
    if ((argc != 6 && argc != 7) || (submission != "graph" && submission != "direct")) {
        std::fprintf(stderr, "Usage: %s DEVICE COEFFICIENT_BITS SLOTS BATCH REPEATS [graph|direct]\n", argv[0]);
        return 2;
    }
    const unsigned device = read_uint(argv[1]);
    const unsigned bits = read_uint(argv[2]);
    const unsigned slots = read_uint(argv[3]);
    const unsigned batch = read_uint(argv[4]);
    const unsigned repeats = read_uint(argv[5]);
    if (!bits || bits > 32768 || !slots || slots > 65536 ||
        !batch || batch > 32 || !repeats || repeats > 100) return 2;
    int devices = 0;
    CK(cudaGetDeviceCount(&devices));
    if (device >= static_cast<unsigned>(devices)) return 2;
    CK(cudaSetDevice(device));
    ntt_gl_reduce_configure();
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    char bus[32]{};
    CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));

    const auto slot_bits = 2 * bits + std::max(1u, ceil_log2(slots));
    const auto cfg = choose_cfg(0, slot_bits, slots);
    if (!cfg.ok || cfg.nwords * batch > (1ull << 25)) {
        std::fprintf(stderr, "Unsupported probe shape\n");
        return 2;
    }
    const Word n = cfg.nwords;
    const Word half = n / 2;
    const Word psi = gl_pow_host(7, (GL_P - 1) / n);
    Transform full(n, cfg.k, psi);
    Transform split(half, cfg.k - 1, gl_mul_host(psi, psi));
    // Initialization sets the CUDA function's shared-memory limit globally.
    // Keep the larger limit if both graphs share the same tile kernel.
    const auto& larger_tile = full.ctx.t >= split.ctx.t ? full.ctx : split.ctx;
    ntt_tile_attributes(larger_tile.warp_tail && larger_tile.t >= 5,
                        static_cast<int>(sizeof(Word) << larger_tile.t));
    std::printf("case version=%s device=%u bus=%s gpu=\"%s\" bits=%u slots=%u batch=%u "
                "bpw=%d slot_words=%llu length=%llu full_tile=%d half_tile=%d submission=%s",
                ecm_program::version, device, bus, prop.name, bits, slots, batch, cfg.bpw,
                cfg.slot_words, n, full.ctx.t, split.ctx.t, submission.data());
    print_outer("full_outer", full.ctx);
    print_outer("half_outer", split.ctx);
    std::puts("");

    const size_t input_words = half * batch;
    const size_t output_words = n * batch;
    Word *a, *b, *input_a, *input_b, *weights, *inverse_weights;
    CK(cudaMalloc(&a, output_words * sizeof(Word)));
    CK(cudaMalloc(&b, output_words * sizeof(Word)));
    CK(cudaMalloc(&input_a, input_words * sizeof(Word)));
    CK(cudaMalloc(&input_b, input_words * sizeof(Word)));
    CK(cudaMalloc(&weights, half * sizeof(Word)));
    CK(cudaMalloc(&inverse_weights, half * sizeof(Word)));

    std::mt19937_64 rng(73963);
    std::vector<Word> host_a(input_words), host_b(input_words);
    fill_packed(host_a, cfg, bits, slots, batch, rng);
    fill_packed(host_b, cfg, bits, slots, batch, rng);
    CK(cudaMemcpy(input_a, host_a.data(), input_words * sizeof(Word), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(input_b, host_b.data(), input_words * sizeof(Word), cudaMemcpyHostToDevice));
    {
        std::vector<Word> powers(half);
        for (const auto& root_table : {std::make_pair(psi, weights),
                                       std::make_pair(full.inverse_root, inverse_weights)}) {
            powers[0] = 1;
            for (Word i = 1; i < half; ++i) powers[i] = gl_mul_host(powers[i - 1], root_table.first);
            CK(cudaMemcpy(root_table.second, powers.data(), half * sizeof(Word), cudaMemcpyHostToDevice));
        }
    }

    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraphExec_t graphs[2]{};
    const dim3 grid(static_cast<unsigned>((half + 255) / 256), batch);
    const auto enqueue = [&](unsigned mode) {
        if (mode == 0) {
            twist::pack<false><<<grid, 256, 0, stream>>>(a, b, input_a, input_b, nullptr, half);
            full.multiply(a, b, batch, stream);
        } else {
            twist::pack<true><<<grid, 256, 0, stream>>>(a, b, input_a, input_b, weights, half);
            split.multiply(a, b, 2 * batch, stream);
            twist::merge<<<grid, 256, 0, stream>>>(a, inverse_weights, half);
        }
        CK(cudaGetLastError());
    };
    if (submission == "graph") {
        for (unsigned mode = 0; mode < 2; ++mode) {
            cudaGraph_t graph;
            CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            enqueue(mode);
            CK(cudaStreamEndCapture(stream, &graph));
            CK(cudaGraphInstantiate(&graphs[mode], graph, 0));
            CK(cudaGraphDestroy(graph));
        }
    }
    const auto launch = [&](unsigned mode) {
        if (graphs[mode]) CK(cudaGraphLaunch(graphs[mode], stream));
        else enqueue(mode);
    };

    size_t oracle_words = 0;
    {
        std::vector<Word> baseline(output_words), weighted(output_words);
        for (unsigned mode = 0; mode < 2; ++mode) {
            launch(mode);
            CK(cudaStreamSynchronize(stream));
            auto& target = mode == 0 ? baseline : weighted;
            CK(cudaMemcpy(target.data(), a, output_words * sizeof(Word), cudaMemcpyDeviceToHost));
        }
        const auto diff = std::mismatch(baseline.begin(), baseline.end(), weighted.begin());
        if (diff.first != baseline.end()) {
            std::fprintf(stderr, "Raw convolution mismatch at %zu: %llu != %llu\n",
                         static_cast<size_t>(diff.first - baseline.begin()), *diff.first, *diff.second);
            return 1;
        }
        for (unsigned s = 0; s < batch; ++s) {
            const auto verify = [&](Word i) {
                check_word(host_a.data() + s * half, host_b.data() + s * half, cfg.L, i,
                           weighted[s * n + i]);
                ++oracle_words;
            };
            if (n <= 4096) {
                for (Word i = 0; i < n; ++i) verify(i);
            } else {
                for (Word i : {Word{0}, Word{1}, cfg.slot_words - 1, cfg.slot_words, cfg.L - 1,
                               half - 1, half, half + 1, 2 * cfg.L - 2, 2 * cfg.L - 1, n - 1}) verify(i);
                for (int i = 0; i < 8; ++i) verify(rng() % n);
            }
        }
    }
    std::printf("verified words=%zu integer_oracle_words=%zu\n", output_words, oracle_words);

    cudaEvent_t start, end;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&end));
    for (int i = 0; i < 3; ++i) {
        launch(0);
        launch(1);
    }
    CK(cudaStreamSynchronize(stream));
    const unsigned launches = std::clamp<unsigned>(32768 / (n * batch), 1, 32);
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        for (unsigned step = 0; step < 2; ++step) {
            const unsigned mode = (repeat + step) % 2;
            CK(cudaEventRecord(start, stream));
            for (unsigned i = 0; i < launches; ++i) launch(mode);
            CK(cudaEventRecord(end, stream));
            CK(cudaEventSynchronize(end));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, start, end));
            std::printf("sample mode=%s repeat=%u launches=%u us=%.6f\n",
                        mode == 0 ? "linear" : "twisted", repeat + 1, launches, 1000.0 * ms / launches);
        }
    }
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(end));
    for (const auto graph : graphs) {
        if (graph) CK(cudaGraphExecDestroy(graph));
    }
    CK(cudaStreamDestroy(stream));
    for (auto p : {a, b, input_a, input_b, weights, inverse_weights}) CK(cudaFree(p));
}
