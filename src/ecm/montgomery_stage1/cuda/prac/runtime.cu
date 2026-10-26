#include "../../../../io/stage1_checkpoint.h"
#include "runtime.h"
#include "../registry.cuh"
#include "../resources.cuh"
#include "../checkpoint.h"
#include "../host_policy.h"
#include "../output.h"
#include "../../../cuda/runtime/cuda_ecm_shim.h"
#include <algorithm>
#include "ecm/stage1/progress.h"
#include <cstring>
using namespace ecm_cuda_mont_host;
#define CUDA_CHECK(action) ecm_cuda_stage1::check_cuda(action,#action)
#define CGBN_CHECK(report) ecm_cuda_stage1::check_cgbn(report,__FILE__,__LINE__)
#include "launch.h"
#include "ecm_prac_plan.h"
#include "constants.h"
#include "../../../../io/exponent_cache.h"
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct ResidentResources {
    uint32_t *data = nullptr, *control = nullptr;
    cgbn_error_report_t *report = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    ResidentResources()=default;
    ResidentResources(const ResidentResources &)=delete;
    ResidentResources &operator=(const ResidentResources &)=delete;
    ~ResidentResources() {
        if (data) cudaFree(data);
        if (control) cudaFree(control);
        if (report) cgbn_error_report_free(report);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
#include "prac_window.inl"
int ecm_cuda_stage1::run_mont_resident(ecm_stage1_checkpoint::Context& checkpoint, mpz_t *factors, int *found, mpz_srcptr N, mpz_srcptr s,
    uint32_t curves, uint64_t *sigma, unsigned long checkpoint_ms, float *gputime,
    uint64_t B1, uint32_t torsion, bool prac,ecm_cuda_stage1::RunResult &outcome,
    const ConstructionResults &construction,ecm_cuda_stage1::SigmaGenerator generate_sigma,const MontRequestedSpec &requested) {
        using namespace ecm_cuda_stage1;
        if (!curves || !sigma || *sigma > UINT64_MAX - curves || B1 < 2 || B1 > UINT32_MAX ||
            (torsion != 1 && torsion != 12)) throw std::runtime_error("invalid resident Stage1 parameters");
        uint32_t bits = 0, tpi = 0;
        const auto policy=read_prac_policy(requested,prac);
        int mode=policy.mode;
        const bool compact=policy.compact,outlined=policy.outlined,single_add=policy.single_add,single_compact=policy.single_compact;
        const uint32_t requested_tpi=policy.requested_tpi;
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
        const char *constants=prac_constants_policy(requested,prac,mode,bits,requested_tpi,N,nbits);
        kernel = cgbn_stage1_domain_dispatch(bits, &tpi, mode, requested_tpi);
        if (!kernel) throw std::runtime_error("requested Stage1 TPI/register policy is unavailable for the selected tier");
        if (verify_size_of_n(N,bits,CARRY_BITS) != ECM_NO_FACTOR_FOUND) return ECM_ERROR;
        EcmPracPlan plan;
        if (prac) {
            outputf(OUTPUT_ALWAYS, "GPU: preparing Prime95 PRAC plan (B1=%llu, t=%u, search=7)\n",
                    (unsigned long long)B1, torsion);
            const auto cache=prac_plan_cache(requested);
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
        const PracWindow window = prac_window_settings(requested,prac, total);
        const size_t bytes=checked_size({7,size_t(curves),size_t(bits/8)});
        const MontResolvedKernel resolved{kernel,bits,tpi,TPB_DEFAULT,
            (size_t(curves)+TPB_DEFAULT/tpi-1)/(TPB_DEFAULT/tpi),bytes};
        ResidentCheckpoint h{0x31524d444d434545ull, B1, ecm_prac_hash_mpz(s), ecm_prac_hash_mpz(N),
            prac ? plan.identity : ecm_prac_hash_mpz(s), prac ? 0ull : 1ull, total, *sigma,
            bytes, 0, 1, prac ? 3u : 2u, 1, bits, tpi, curves, torsion, 0};
        const auto path=checkpoint_record(checkpoint);
        std::vector<uint32_t> data(bytes / sizeof(uint32_t));
        bool resumed=false;
        checkpoint.validate_pending([&](const std::vector<std::string> &records,std::vector<uint64_t> &sigmas) {
            if(window.enabled())throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
                "window sample cannot resume a production checkpoint");
            if(records.size()!=curves || sigmas.size()!=curves)return false;
            for(size_t i=1;i<records.size();++i)if(!records[i].empty())return false;
            if(records[0].empty())return true;
            const auto decoded=decode_checkpoint(records[0],N,h,data,prac);
            if(decoded.state==ecm_stage1_checkpoint::RestoreState::Incompatible)
                throw ecm_stage1_checkpoint::Error(decoded.state,decoded.error);
            if(!decoded.restored())return false;
            for(uint32_t i=0;i<curves;++i)if(sigmas[i] && sigmas[i]!=h.sigma+i)return false;
            for(uint32_t i=0;i<curves;++i)sigmas[i]=h.sigma+i;
            resumed=true;return true;
        });
        if(!resumed && !window.enabled() && !path.empty()) {
            std::string saved;
            const auto read=checkpoint.read_checkpoint_blob(path,saved);
            if(read==ecm_stage1_checkpoint::BlobRead::IoFailure)
                throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::IoFailure,"cannot read resident checkpoint");
            if(read==ecm_stage1_checkpoint::BlobRead::Read) {
                const auto decoded=decode_checkpoint(saved,N,h,data,prac);
                if(!decoded.restored())throw ecm_stage1_checkpoint::Error(decoded.state,decoded.error);
                resumed=true;
            }
        }
        if(resumed) {
            *sigma=h.sigma;
            outputf(OUTPUT_ALWAYS,"GPU: resident checkpoint resumed (next=%llu/%llu, sigma=%llu)\n",
                (unsigned long long)h.next,(unsigned long long)h.total,(unsigned long long)*sigma);
        } else {
            uint64_t saved=0;
            if(checkpoint.saved_sigma(0,saved))*sigma=saved;
            else if(generate_sigma)*sigma=generate_sigma(curves,0);
            h.sigma=*sigma;
        }
        if(*sigma>UINT64_MAX-(curves-1ull))throw RunError(Failure::Preparation,"resident sigma range overflow");
        for(uint32_t i=0;i<curves;++i)
            if(checkpoint.sigma(i,*sigma+i)!=*sigma+i)
                throw RunError(Failure::Preparation,"nonconsecutive resident checkpoint sigmas");
        {
            size_t initial_bytes = 0;
            std::unique_ptr<uint32_t, decltype(&free)> initial(set_p_2p_suyama(N, curves, *sigma, bits, &initial_bytes,&construction), &free);
            if (!initial) throw std::bad_alloc();
            if (initial_bytes != bytes) throw std::runtime_error("resident curve buffer stride mismatch");
            if(!resumed)memcpy(data.data(), initial.get(), bytes);
        }
        outcome.failure=Failure::Execution;
        ResidentResources gpu;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.data),resolved.state_bytes));
        CUDA_CHECK(cudaMemcpy(gpu.data, data.data(), bytes, cudaMemcpyHostToDevice));
        const size_t control_bytes = prac ? checked_size({plan.primes.size(),sizeof(EcmPracPrime)}) : checked_size({size_t(s_bits/32+(s_bits%32!=0)),4});
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.control), control_bytes));
        CUDA_CHECK(cudaMemcpy(gpu.control, prac ? static_cast<const void *>(plan.primes.data()) : exponent.get(),
                              control_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cgbn_error_report_alloc(&gpu.report));
        CUDA_CHECK(cudaEventCreate(&gpu.begin)); CUDA_CHECK(cudaEventCreate(&gpu.end));
        const uint32_t np0 = find_np0(N), blocks=uint32_t(resolved.blocks);
        auto launch = [&](cgbn_stage1_kernel_fn fn, uint64_t first, uint64_t length) {
            cgbn_stage1_launch(fn,blocks,TPB_DEFAULT,gpu.report, s_bits, first, length, gpu.control, gpu.data, curves, 0, np0);
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
                (mode == ECM_DOMAIN_PRAC_NATURAL || mode == ECM_DOMAIN_PRAC_COMPACT || mode == ECM_DOMAIN_PRAC_OUTLINE_ADD || mode == ECM_DOMAIN_PRAC_SINGLE_ADD || mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT) ? "natural (255)" :
                (mode == ECM_DOMAIN_PRAC_168 || mode == ECM_DOMAIN_PRAC_COMPACT_168 || mode == ECM_DOMAIN_PRAC_OUTLINE_ADD_168 || mode == ECM_DOMAIN_PRAC_SINGLE_ADD_168 || mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168 ||
                 mode == ECM_DOMAIN_PRAC_CONSTANT_RUNTIME || mode == ECM_DOMAIN_PRAC_CONSTANT_NP0 || mode == ECM_DOMAIN_PRAC_CONSTANT_M4423) ? "168 (4608 candidate)" :
                mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_128 ? "128 (4608/TPI16 single-compact)" : "per-tier");
            outputf(OUTPUT_ALWAYS, "GPU: PRAC variant=%s\n", single_compact ? "single-compact (one inline xADD site, 2-temporary DBL)" : single_add ? "single-add (one inline xADD site)" : outlined ? "outline-add (shared xADD)" : compact ? "compact (2-temporary DBL)" : "baseline");
            outputf(OUTPUT_ALWAYS, "GPU: PRAC constants policy=%s; np0=%u\n", constants, np0);
        }
        outputf(OUTPUT_ALWAYS, "GPU: sigma=%llu, CGBN<%u,%u>, curves=%u, blocks=%u, blocks/SM=%d, control=%zu bytes\n",
                (unsigned long long)*sigma, tpi, bits, curves, blocks, occupied, control_bytes);
        if (window.enabled()) return prac_window_run(window, plan, gpu, resolved.entry, s_bits,
            bits, tpi, requested_tpi, curves, *sigma, B1, np0, blocks, data, gputime,
            mode == ECM_DOMAIN_PRAC_CONSTANT_M4423 ? 5 : 6);
        const double target_ms=prac_target_ms(requested,prac);
        uint64_t completed_work = 0;
        if (prac) for (uint64_t i = 0; i < h.next; ++i) completed_work += uint64_t(plan.primes[size_t(i)].work) * plan.primes[size_t(i)].repetitions;
        else completed_work = h.next - 1;
        auto start_wall = std::chrono::steady_clock::now(), last_ckpt = start_wall;
        outcome.total=total;outcome.cursor=h.next;outcome.snapshot_cursor=h.next;outcome.durable_cursor=resumed?h.next:0;
        uint64_t chunk = prac ? 16 : 200;
        ecm_stage1_gpu_progress::SpeedWindow speed;
        double kernel_seconds = 0, last_print = -1;
        const double sample_limit=prac_sample_limit(requested);
        while (h.next < total) {
            uint64_t length = std::min(chunk, total - h.next), work = 0;
            if (prac) for (uint64_t i = h.next; i < h.next + length; ++i) work += uint64_t(plan.primes[size_t(i)].work) * plan.primes[size_t(i)].repetitions;
            else work = length;
            CUDA_CHECK(cudaEventRecord(gpu.begin)); launch(resolved.entry, h.next, length);
            CUDA_CHECK(cudaEventRecord(gpu.end)); CUDA_CHECK(cudaEventSynchronize(gpu.end));
            CGBN_CHECK(gpu.report);
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, gpu.begin, gpu.end));
            kernel_seconds += ms / 1000.0; h.next += length; completed_work += work;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_wall).count();
            outcome.cursor=h.next;
            speed.add(work,ms);
            const double per_curve=speed.per_curve(total_work,curves);
            const double remaining=speed.remaining(total_work,completed_work);
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
            const bool stopping=ecm_stage1_stop_requested();
            if (sampled || stopping || h.next>=total || (checkpoint_ms && std::chrono::duration<double, std::milli>(now - last_ckpt).count() >= checkpoint_ms)) {
                CUDA_CHECK(cudaMemcpy(data.data(), gpu.data, bytes, cudaMemcpyDeviceToHost));
                outcome.snapshot_cursor=h.next;
                if(!path.empty()) {
                    if(!save_checkpoint(checkpoint, path,h,data))throw RunError(Failure::Persistence,"resident checkpoint write failed");
                    outcome.durable_cursor=h.next;
                }
                outputf(OUTPUT_ALWAYS,path.empty()?"GPU: no checkpoint configured (next=%llu/%llu)\n":"GPU: checkpoint saved (Montgomery domain, next=%llu/%llu)\n",
                        (unsigned long long)h.next, (unsigned long long)total);
                last_ckpt = std::chrono::steady_clock::now();
            }
            if(stopping && h.next<total){outcome.termination=Termination::Stopped;outputf(OUTPUT_ALWAYS,path.empty()?"Stage1 stopped: progress is not recoverable\n":"Stage1 stopped: checkpoint saved\n");return ECM_ERROR;}
            if (sampled) {
                outcome.termination=Termination::SampleLimit;
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
        int result = process_results(factors, found, N, data.data(), bits, curves, *sigma, 7, 3, 5, 0,true,&outcome.degenerate_curve);
        outcome.cursor=h.next; // Task finalization removes the durable native snapshot.
        return result;
}
