// Included by the Stage1 host TU after curve construction/result/checkpoint helpers.
#include "cgbn_stage1_prac_kernel.cuh"
#include "ecm_stage1_exp_cache.h"
#include <chrono>
#include <memory>
#include <fstream>
#include <string>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct ResidentCheckpoint {
    uint64_t magic, b1, scalar, modulus, identity, next, total, sigma, bytes, checksum;
    uint32_t version, algorithm, domain, bits, tpi, curves, torsion, reserved;
};
struct ResidentResources {
    uint32_t *data = nullptr, *control = nullptr;
    cgbn_error_report_t *report = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    ~ResidentResources() {
        if (data) cudaFree(data);
        if (control) cudaFree(control);
        if (report) cgbn_error_report_free(report);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
static bool resident_checkpoint_write(const std::string &path, ResidentCheckpoint h,
                                      const std::vector<uint32_t> &data) {
    h.checksum = ecm_prac_hash(data.data(), data.size() * sizeof(uint32_t));
    std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(data.data()), h.bytes);
    out.close();
    if (!out) return false;
#ifdef _WIN32
    return MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(temporary.c_str(), path.c_str()) == 0;
#endif
}
#include "cgbn_stage1_prac_window.cuh"
static int cgbn_stage1_resident(mpz_t *factors, int *found, mpz_srcptr N, mpz_srcptr s,
    uint32_t curves, uint64_t *sigma, unsigned long checkpoint_ms, float *gputime,
    uint64_t B1, uint32_t torsion, bool prac) {
    try {
        if (!curves || !sigma || *sigma > UINT64_MAX - curves || B1 < 2 || B1 > UINT32_MAX ||
            (torsion != 1 && torsion != 12)) throw std::runtime_error("invalid resident Stage1 parameters");
        uint32_t bits = 0, tpi = 0;
        int mode = prac ? ECM_DOMAIN_PRAC : ECM_DOMAIN_LADDER;
        if (prac) if (const char *cap = getenv("ECM_PRAC_REG_TARGET")) {
            if (strcmp(cap, "255") == 0) mode = ECM_DOMAIN_PRAC_NATURAL;
            else if (strcmp(cap, "168") == 0) mode = ECM_DOMAIN_PRAC_168;
            else if (*cap && strcmp(cap, "0") != 0) throw std::runtime_error("ECM_PRAC_REG_TARGET must be 0, 168 or 255");
        }
        bool compact = false;
        if (const char *variant = getenv("ECM_PRAC_VARIANT")) {
            if (strcmp(variant, "compact") == 0) compact = true;
            else if (*variant && strcmp(variant, "baseline") != 0)
                throw std::runtime_error("ECM_PRAC_VARIANT must be baseline or compact");
        }
        if (compact) {
            if (!prac) throw std::runtime_error("compact variant requires PRAC");
            if (mode == ECM_DOMAIN_PRAC_NATURAL) mode = ECM_DOMAIN_PRAC_COMPACT;
            else if (mode == ECM_DOMAIN_PRAC_168) mode = ECM_DOMAIN_PRAC_COMPACT_168;
            else throw std::runtime_error("compact variant requires register policy 255 or 168");
        }
        uint32_t requested_tpi = 0;
        if (const char *env = getenv("ECM_STAGE1_TPI")) {
            if (!*env || strcmp(env, "0") == 0) requested_tpi = 0;
            else if (strcmp(env, "16") == 0) requested_tpi = 16;
            else if (strcmp(env, "32") == 0) requested_tpi = 32;
            else throw std::runtime_error("ECM_STAGE1_TPI must be 0, 16 or 32");
        }
        std::vector<uint32_t> tiers; ecm_build_tier_list_param0(tiers);
        const size_t nbits = mpz_sizeinbase(N, 2);
        cgbn_stage1_kernel_fn kernel = nullptr;
        for (uint32_t tier : tiers) if (tier >= nbits + CARRY_BITS) {
            // Choose the normal smallest tier first; a policy cannot silently pad N
            // into a larger tier just because its requested kernel is unavailable.
            kernel = cgbn_stage1_domain_dispatch(tier, &tpi, prac ? ECM_DOMAIN_PRAC : ECM_DOMAIN_LADDER);
            if (kernel) { bits = tier; break; }
        }
        if (!kernel) throw std::runtime_error("no resident/PRAC kernel covers N in this build");
        kernel = cgbn_stage1_domain_dispatch(bits, &tpi, mode, requested_tpi);
        if (!kernel) throw std::runtime_error("requested Stage1 TPI/register policy is unavailable for the selected tier");
        if (verify_size_of_n(N, bits) != ECM_NO_FACTOR_FOUND) return ECM_ERROR;
        EcmPracPlan plan;
        if (prac) {
            outputf(OUTPUT_ALWAYS, "GPU: preparing Prime95 PRAC plan (B1=%llu, t=%u, search=7)\n",
                    (unsigned long long)B1, torsion);
            std::string cache = ecm_exp_cache_get_dir();
            if (const char *env = getenv("ECM_PRAC_PLAN_CACHE")) cache = env;
            plan = ecm_prac_build(uint32_t(B1), torsion, s, cache);
            outputf(OUTPUT_ALWAYS, "GPU: PRAC plan %s; work=%llu\n", plan.status.c_str(),
                    (unsigned long long)plan.work);
        }
        uint64_t s_bits = 0;
        std::unique_ptr<uint32_t, decltype(&free)> exponent(nullptr, &free);
        if (prac) s_bits = mpz_sizeinbase(s, 2);
        else exponent.reset(allocate_and_set_s_bits(s, &s_bits));
        const uint64_t total = prac ? plan.primes.size() : s_bits;
        const uint64_t total_work = prac ? plan.work : s_bits - 1;
        const PracWindow window = prac_window_settings(prac, total);
        const size_t bytes = size_t(7) * curves * (bits / 8);
        ResidentCheckpoint h{0x31524d444d434545ull, B1, ecm_prac_hash_mpz(s), ecm_prac_hash_mpz(N),
            prac ? plan.identity : ecm_prac_hash_mpz(s), prac ? 0ull : 1ull, total, *sigma,
            bytes, 0, 1, prac ? 3u : 2u, 1, bits, tpi, curves, torsion, 0};
        std::string path = std::string(get_checkpoint_filename(N)) + (prac ? ".prac-v1" : ".resident-v1");
        std::vector<uint32_t> data(bytes / sizeof(uint32_t));
        bool resumed = false;
        if (!window.enabled()) {
            std::ifstream in(path, std::ios::binary); ResidentCheckpoint old{};
            if (in.read(reinterpret_cast<char *>(&old), sizeof(old))) {
                bool compatible = old.magic == h.magic && old.version == h.version && old.b1 == h.b1 &&
                    old.scalar == h.scalar && old.modulus == h.modulus && old.identity == h.identity &&
                    old.total == total && old.bytes == bytes && old.algorithm == h.algorithm && old.domain == 1 &&
                    old.bits == bits && old.tpi == tpi && old.curves == curves && old.torsion == torsion &&
                    old.next <= total && (prac || old.next >= 1) && old.sigma <= UINT64_MAX - curves;
                if (compatible && in.read(reinterpret_cast<char *>(data.data()), bytes) &&
                    in.peek() == std::char_traits<char>::eof() && ecm_prac_hash(data.data(), bytes) == old.checksum) {
                    h = old; *sigma = h.sigma; resumed = true;
                    outputf(OUTPUT_ALWAYS, "GPU: resident checkpoint resumed (next=%llu/%llu, sigma=%llu)\n",
                        (unsigned long long)h.next, (unsigned long long)total, (unsigned long long)*sigma);
                } else outputf(OUTPUT_ALWAYS, "GPU: resident checkpoint mismatch/corruption; starting fresh\n");
            }
        }
        if (!resumed) {
            size_t initial_bytes = 0;
            std::unique_ptr<uint32_t, decltype(&free)> initial(set_p_2p_suyama(N, curves, *sigma, bits, &initial_bytes), &free);
            if (!initial) throw std::bad_alloc();
            if (initial_bytes != bytes) throw std::runtime_error("resident curve buffer stride mismatch");
            memcpy(data.data(), initial.get(), bytes);
        }
        ResidentResources gpu;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.data), bytes));
        CUDA_CHECK(cudaMemcpy(gpu.data, data.data(), bytes, cudaMemcpyHostToDevice));
        const size_t control_bytes = prac ? plan.primes.size() * sizeof(EcmPracPrime) : size_t((s_bits + 31) / 32) * 4;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.control), control_bytes));
        CUDA_CHECK(cudaMemcpy(gpu.control, prac ? static_cast<const void *>(plan.primes.data()) : exponent.get(),
                              control_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cgbn_error_report_alloc(&gpu.report));
        CUDA_CHECK(cudaEventCreate(&gpu.begin)); CUDA_CHECK(cudaEventCreate(&gpu.end));
        const uint32_t np0 = find_np0(N), blocks = (curves + TPB_DEFAULT / tpi - 1) / (TPB_DEFAULT / tpi);
        auto launch = [&](cgbn_stage1_kernel_fn fn, uint64_t first, uint64_t length) {
            fn<<<blocks, TPB_DEFAULT>>>(gpu.report, s_bits, first, length, gpu.control, gpu.data, curves, 0, np0);
            CUDA_CHECK(cudaGetLastError());
        };
        if (!resumed) {
            uint32_t dummy;
            launch(cgbn_stage1_domain_dispatch(bits, &dummy, ECM_DOMAIN_INIT, requested_tpi), 0, 0);
            CUDA_CHECK(cudaDeviceSynchronize()); CGBN_CHECK(gpu.report);
        }
        int occupied = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupied, kernel, TPB_DEFAULT, 0));
        outputf(OUTPUT_ALWAYS, "GPU: parametrization = Suyama param0; algorithm=%s; resident Montgomery domain\n",
                prac ? "prac" : "resident-ladder");
        if (prac) {
            outputf(OUTPUT_ALWAYS, "GPU: PRAC register policy=%s\n",
                (mode == ECM_DOMAIN_PRAC_NATURAL || mode == ECM_DOMAIN_PRAC_COMPACT) ? "natural (255)" :
                (mode == ECM_DOMAIN_PRAC_168 || mode == ECM_DOMAIN_PRAC_COMPACT_168) ? "168 (4608/TPI16)" : "per-tier");
            outputf(OUTPUT_ALWAYS, "GPU: PRAC variant=%s\n", compact ? "compact (2-temporary DBL)" : "baseline");
        }
        outputf(OUTPUT_ALWAYS, "GPU: sigma=%llu, CGBN<%u,%u>, curves=%u, blocks=%u, blocks/SM=%d, control=%zu bytes\n",
                (unsigned long long)*sigma, tpi, bits, curves, blocks, occupied, control_bytes);
        if (window.enabled()) return prac_window_run(window, plan, gpu, kernel, s_bits,
            bits, tpi, requested_tpi, curves, *sigma, B1, np0, blocks, data, gputime);
        double target_ms = 100;
        if (prac) {
            if (const char *value = getenv("ECM_PRAC_TARGET_MS")) {
                char *end = nullptr;
                target_ms = std::strtod(value, &end);
                if (end == value || *end || !std::isfinite(target_ms) || target_ms < 10 || target_ms > 500)
                    throw std::runtime_error("ECM_PRAC_TARGET_MS must be finite and in [10,500]");
            }
            outputf(OUTPUT_ALWAYS, "GPU: PRAC slice target=%.3f ms, adaptive bounds=%.3f..%.3f ms\n",
                target_ms, target_ms * 0.8, target_ms * 1.2);
        }
        uint64_t completed_work = 0;
        if (prac) for (uint64_t i = 0; i < h.next; ++i) completed_work += uint64_t(plan.primes[size_t(i)].work) * plan.primes[size_t(i)].repetitions;
        else completed_work = h.next - 1;
        auto start_wall = std::chrono::steady_clock::now(), last_ckpt = start_wall;
        uint64_t chunk = prac ? 16 : 200;
        double ring[50]{}; size_t ring_used = 0, ring_next = 0; double ring_sum = 0;
        double kernel_seconds = 0, last_print = -1;
        double sample_limit = 0;
        if (const char *env = getenv("ECM_GPU_STAGE1_SAMPLE_SECONDS")) {
            sample_limit = atof(env);
            if (!(sample_limit >= 0 && sample_limit < 1e9)) throw std::runtime_error("invalid sample limit");
        }
        while (h.next < total) {
            uint64_t length = std::min(chunk, total - h.next), work = 0;
            if (prac) for (uint64_t i = h.next; i < h.next + length; ++i) work += uint64_t(plan.primes[size_t(i)].work) * plan.primes[size_t(i)].repetitions;
            else work = length;
            CUDA_CHECK(cudaEventRecord(gpu.begin)); launch(kernel, h.next, length);
            CUDA_CHECK(cudaEventRecord(gpu.end)); CUDA_CHECK(cudaEventSynchronize(gpu.end));
            CGBN_CHECK(gpu.report);
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, gpu.begin, gpu.end));
            kernel_seconds += ms / 1000.0; h.next += length; completed_work += work;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_wall).count();
            if (ms > 0) {
                double rate = work / double(ms);
                if (ring_used == 50) ring_sum -= ring[ring_next]; else ++ring_used;
                ring[ring_next] = rate; ring_sum += rate; ring_next = (ring_next + 1) % 50;
            }
            double speed = ring_used ? ring_sum / ring_used : 0;
            double per_curve = speed > 0 ? total_work / speed / 1000.0 / curves : 0;
            double remaining = speed > 0 ? (total_work - completed_work) / speed / 1000.0 : 0;
            bool sampled = sample_limit > 0 && elapsed >= sample_limit && h.next < total;
            if (elapsed - last_print >= 0.2 || h.next == total || sampled) {
                // Weighted work, not prime count: prime sizes/chains vary over the run.
                print_progress(100.0 * completed_work / total_work, completed_work, work, per_curve, elapsed * 1000, remaining, true,
                               prac ? "M-equiv" : "bits");
                outputf(OUTPUT_NORMAL, "GPU: %s slice next=%llu/%llu, kernel=%.6f s, projected=%.6f s/curve, length=%llu, slice-ms=%.3f\n",
                        prac ? "PRAC" : "resident", (unsigned long long)h.next, (unsigned long long)total, kernel_seconds, per_curve,
                        (unsigned long long)length, double(ms));
                last_print = elapsed;
            }
            auto now = std::chrono::steady_clock::now();
            if (sampled || (checkpoint_ms && std::chrono::duration<double, std::milli>(now - last_ckpt).count() >= checkpoint_ms)) {
                CUDA_CHECK(cudaMemcpy(data.data(), gpu.data, bytes, cudaMemcpyDeviceToHost));
                if (!resident_checkpoint_write(path, h, data)) throw std::runtime_error("resident checkpoint write failed");
                outputf(OUTPUT_ALWAYS, "GPU: checkpoint saved (Montgomery domain, next=%llu/%llu)\n",
                        (unsigned long long)h.next, (unsigned long long)total);
                last_ckpt = std::chrono::steady_clock::now();
            }
            if (sampled) {
                *gputime = float(elapsed * 1000);
                outputf(OUTPUT_ALWAYS, "GPU: sample limit reached; incomplete Stage1 saved only as checkpoint\n");
                return ECM_ERROR; // Driver must not publish an incomplete final Stage1 save.
            }
            if (ms < target_ms * 0.8) chunk = std::max<uint64_t>(chunk + 1, chunk * 11 / 10);
            else if (ms > target_ms * 1.2) chunk = std::max<uint64_t>(1, chunk * 9 / 10);
        }
        uint32_t dummy;
        launch(cgbn_stage1_domain_dispatch(bits, &dummy, ECM_DOMAIN_EXPORT, requested_tpi), 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize()); CGBN_CHECK(gpu.report);
        CUDA_CHECK(cudaMemcpy(data.data(), gpu.data, bytes, cudaMemcpyDeviceToHost));
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_wall).count();
        *gputime = float(elapsed * 1000);
        outputf(OUTPUT_ALWAYS, "GPU: Stage1 %s completed; kernel=%.6f s, execution-wall=%.6f s\n", prac ? "PRAC" : "resident", kernel_seconds, elapsed);
        int result = process_results(factors, found, N, data.data(), bits, curves, *sigma, 7, 3, 5, 0);
        if (result != ECM_ERROR) std::remove(path.c_str());
        return result;
    } catch (const std::exception &e) {
        outputf(OUTPUT_ERROR, "GPU: resident/PRAC Stage1 failed: %s\n", e.what());
        return ECM_ERROR;
    }
}
