// Host-only, opt-in partial-product timing. Never publishes a Stage1 result.
#include <cmath>
#include <cstdlib>

struct PracWindow {
    std::string position;
    uint64_t first = 0, count = 0, chunk = 0;
    uint32_t warmup = 2;
    double seconds = 6;
    bool dump = false;
    bool enabled() const { return !position.empty(); }
};
static uint64_t prac_window_integer(const char *key, uint64_t fallback) {
    const char *value = getenv(key);
    if (!value || !*value) return fallback;
    uint64_t result = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > (UINT64_MAX - uint64_t(*p - '0')) / 10)
            throw std::runtime_error(std::string("invalid ") + key);
        result = result * 10 + uint64_t(*p - '0');
    }
    return result;
}
static PracWindow prac_window_settings(bool prac, uint64_t records) {
    PracWindow w;
    const char *position = getenv("ECM_PRAC_WINDOW");
    if (!position || !*position) return w;
    if (!prac) throw std::runtime_error("ECM_PRAC_WINDOW requires PRAC");
    w.position = position;
    if (w.position != "prefix" && w.position != "middle" && w.position != "tail")
        throw std::runtime_error("ECM_PRAC_WINDOW must be prefix, middle or tail");
    uint64_t count = prac_window_integer("ECM_PRAC_WINDOW_COUNT", 16);
    if (!count || count > 32 || !records) throw std::runtime_error("PRAC window count must be 1..32");
    w.count = std::min(count, records);
    w.chunk = prac_window_integer("ECM_PRAC_WINDOW_CHUNK", 0);
    if (!w.chunk) w.chunk = w.count;
    if (w.chunk > w.count) throw std::runtime_error("PRAC window chunk must not exceed selected count");
    if (w.position == "middle") w.first = (records - w.count) / 2;
    else if (w.position == "tail") w.first = records - w.count;
    uint64_t warmup = prac_window_integer("ECM_PRAC_WINDOW_WARMUP", 2);
    if (warmup > 32) throw std::runtime_error("PRAC window warmup must be 0..32");
    w.warmup = uint32_t(warmup);
    if (const char *seconds = getenv("ECM_GPU_STAGE1_SAMPLE_SECONDS")) {
        char *end = nullptr;
        w.seconds = std::strtod(seconds, &end);
        if (end == seconds || *end || !std::isfinite(w.seconds) || w.seconds <= 0 || w.seconds > 600)
            throw std::runtime_error("PRAC window sample seconds must be finite and in (0,600]");
    }
    uint64_t dump = prac_window_integer("ECM_PRAC_WINDOW_DUMP", 0);
    if (dump > 1) throw std::runtime_error("PRAC window dump must be 0 or 1");
    w.dump = dump != 0;
    return w;
}

static int prac_window_run(const PracWindow &w, const EcmPracPlan &plan,
    ResidentResources &gpu, cgbn_stage1_kernel_fn kernel, uint64_t scalar_bits,
    uint32_t bits, uint32_t tpi, uint32_t requested_tpi, uint32_t curves,
    uint64_t sigma, uint64_t b1, uint32_t np0, uint32_t blocks,
    std::vector<uint32_t> &data, float *gputime, uint32_t boundary_bn_count) {
    const size_t bytes = data.size() * sizeof(uint32_t);
    uint64_t work = 0;
    for (uint64_t i = w.first; i < w.first + w.count; ++i)
        work += uint64_t(plan.primes[size_t(i)].work) * plan.primes[size_t(i)].repetitions;
    if (!work) throw std::runtime_error("empty PRAC window work");
    uint32_t *seed_raw = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&seed_raw), bytes));
    auto release = [](uint32_t *p) { if (p) cudaFree(p); };
    std::unique_ptr<uint32_t, decltype(release)> seed(seed_raw, release);
    CUDA_CHECK(cudaMemcpy(seed.get(), gpu.data, bytes, cudaMemcpyDeviceToDevice));
    outputf(OUTPUT_ALWAYS, "GPU: PRAC_WINDOW position=%s first=%llu count=%llu p_first=%u p_last=%u work=%llu full_work=%llu warmup=%u; partial product, no Stage1 save or checkpoint\n",
        w.position.c_str(), (unsigned long long)w.first, (unsigned long long)w.count,
        plan.primes[size_t(w.first)].p, plan.primes[size_t(w.first + w.count - 1)].p,
        (unsigned long long)work, (unsigned long long)plan.work, w.warmup);
    const uint64_t launches_per_round = (w.count + w.chunk - 1) / w.chunk;
    outputf(OUTPUT_ALWAYS, "GPU: PRAC_WINDOW_SLICING chunk=%llu launches_per_round=%llu; same contiguous subproduct, restore once per round\n",
        (unsigned long long)w.chunk, (unsigned long long)launches_per_round);
    auto wall_begin = std::chrono::steady_clock::now();
    uint64_t rounds = 0, measured = 0;
    double kernel_ms = 0, measured_wall_ms = 0, next_print = 0;
    double wall = 0;
    do {
        auto round_begin = std::chrono::steady_clock::now();
        // Restore exactly the same non-degenerate input point each round. This
        // copy is outside event timing and cannot become repeated scalar growth.
        CUDA_CHECK(cudaMemcpy(gpu.data, seed.get(), bytes, cudaMemcpyDeviceToDevice));
        double ms = 0;
        for (uint64_t offset = 0; offset < w.count; offset += w.chunk) {
            uint64_t length = std::min(w.chunk, w.count - offset);
            CUDA_CHECK(cudaEventRecord(gpu.begin));
            kernel<<<blocks, TPB_DEFAULT>>>(gpu.report, scalar_bits, w.first + offset, length,
                gpu.control, gpu.data, curves, 0, np0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(gpu.end)); CUDA_CHECK(cudaEventSynchronize(gpu.end));
            CGBN_CHECK(gpu.report);
            float slice_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&slice_ms, gpu.begin, gpu.end));
            ms += slice_ms;
        }
        ++rounds;
        if (rounds > w.warmup) {
            kernel_ms += ms; ++measured;
            measured_wall_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - round_begin).count();
        }
        wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_begin).count();
        if (measured && kernel_ms > 0 && wall >= next_print) {
            double projected = double(plan.work) * kernel_ms / (double(work) * measured) / 1000 / curves;
            outputf(OUTPUT_NORMAL, "GPU: PRAC_WINDOW_SAMPLE round=%llu kernel_ms=%.6f projected=%.6f s/curve\n",
                (unsigned long long)rounds, double(ms), projected);
            next_print = wall + 0.2;
        }
    } while (rounds < uint64_t(w.warmup) + 3 || wall < w.seconds);
    if (kernel_ms <= 0) throw std::runtime_error("PRAC window event timer returned zero");
    double projected = double(plan.work) * kernel_ms / (double(work) * measured) / 1000 / curves;
    outputf(OUTPUT_ALWAYS, "GPU: PRAC_WINDOW_DONE rounds=%llu measured=%llu kernel_ms=%.6f work=%llu projected=%.6f s/curve wall=%.6f seed_bytes=%zu restore_bytes=%llu\n",
        (unsigned long long)rounds, (unsigned long long)measured, kernel_ms,
        (unsigned long long)work, projected, wall, bytes, (unsigned long long)(bytes * rounds));
    outputf(OUTPUT_ALWAYS, "GPU: PRAC_WINDOW_COST measured_wall_ms=%.6f measured_launches=%llu boundary_logical_bytes_per_round=%llu\n",
        measured_wall_ms, (unsigned long long)(launches_per_round * measured),
        (unsigned long long)(uint64_t(boundary_bn_count) * curves * (bits / 8) * launches_per_round));
    if (w.dump) {
        uint32_t dummy;
        auto export_kernel = cgbn_stage1_domain_dispatch(bits, &dummy, ECM_DOMAIN_EXPORT, requested_tpi);
        export_kernel<<<blocks, TPB_DEFAULT>>>(gpu.report, scalar_bits, 0, 0, gpu.control, gpu.data, curves, 0, np0);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); CGBN_CHECK(gpu.report);
        CUDA_CHECK(cudaMemcpy(data.data(), gpu.data, bytes, cudaMemcpyDeviceToHost));
        FILE *dump = fopen("window_q.csv", "wb");
        if (!dump) throw std::runtime_error("cannot write PRAC window diagnostic CSV");
        bool failed = fprintf(dump, "kind,b1,first,count,work,sigma,x,z\n") < 0;
        size_t limbs = bits / 32;
        for (uint32_t i = 0; i < curves; ++i) {
            failed |= fprintf(dump, "partial_product,%llu,%llu,%llu,%llu,%llu",
                (unsigned long long)b1, (unsigned long long)w.first, (unsigned long long)w.count,
                (unsigned long long)work, (unsigned long long)(sigma + i)) < 0;
            for (size_t word : {size_t(3), size_t(4)}) {
                failed |= fputs(",0x", dump) < 0;
                const uint32_t *field = data.data() + (7 * size_t(i) + word) * limbs;
                for (size_t j = limbs; j > 0; --j) failed |= fprintf(dump, "%08x", field[j - 1]) < 0;
            }
            failed |= fputc('\n', dump) == EOF;
        }
        failed |= fclose(dump) != 0;
        if (failed) throw std::runtime_error("PRAC window diagnostic CSV write failed");
    }
    *gputime = float(wall * 1000);
    (void)tpi;
    return ECM_ERROR; // Partial products are never complete Stage1 outputs.
}
