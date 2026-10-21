#include "ecm/backend/opencl/runtime/cgbn_opencl.h"
#include "opencl_ecm_addsub_manifest.h"
#include "ecm/application/configuration/runtime.h"

#include <CL/cl.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint32_t MAX_BENCH_BITS = 16384;
constexpr uint32_t MAX_BENCH_WORDS = MAX_BENCH_BITS / 32;

void fill_from_gmp(const mpz_t v, uint32_t *out_words, size_t words) {
    mpz_t tmp, mod;
    mpz_init(tmp);
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2, (unsigned long)(words * 32));
    mpz_mod(tmp, v, mod);
    size_t count = 0;
    mpz_export(out_words, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out_words[i] = 0u;
    mpz_clear(tmp);
    mpz_clear(mod);
}

void fill_to_gmp(const uint32_t *in_words, size_t words, mpz_t out) {
    mpz_import(out, words, -1, sizeof(uint32_t), 0, 0, in_words);
}

bool run_kernel(cl_command_queue q, cl_kernel k, size_t global, int total_enqueues, double &ms) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < total_enqueues; ++i) {
        cl_int err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Enqueue failed: " << err << std::endl;
            return false;
        }
    }
    clFinish(q);
    auto t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

void query_kernel_resources(cl_kernel k, cl_device_id dev, size_t &private_bytes,
                            size_t &local_bytes, size_t &pref_wg_multiple,
                            size_t &workgroup_size) {
    private_bytes = local_bytes = pref_wg_multiple = workgroup_size = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(size_t), &private_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), &local_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
                             &pref_wg_multiple, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, nullptr);
}

} // namespace

bool runOpenClEcmAddSubBenchmark(int bits, int kernel_iterations, int instances, int launch_repeats,
                                 bool bench_unroll_only, bool verbose, bool no_overflow) {
    if (bits <= 0 || (bits % 32) != 0 || (uint32_t)bits > MAX_BENCH_BITS) {
        std::cerr << "bits must be a positive multiple of 32 and <= " << MAX_BENCH_BITS
                  << std::endl;
        return false;
    }
    const uint32_t BITS = (uint32_t)bits;
    const uint32_t WORDS = BITS / 32;

    std::cout << "ECM add/sub microbench: " << BITS
              << "-bit, kernel_iterations=" << kernel_iterations
              << ", instances=" << instances
              << ", launch_repeats=" << launch_repeats
              << ", unroll_only=" << (bench_unroll_only ? "1" : "0") << std::endl;

    // Two test cases per bit-width:
    //   0 = a+b >= N (overflow — fused speculative subtract triggers)
    //   1 = a+b <  N (no-overflow — sum stays below N)
    // Random data seeded by bits for reproducibility.
    struct BenchCase {
        const char *label;
        std::vector<uint32_t> a_words, b_words, n_words;
        mpz_t a_gmp, b_gmp, n_gmp;  // GMP references for verification
    };
    BenchCase cases[2] = {};
    {
        gmp_randstate_t rng;
        gmp_randinit_default(rng);
        for (int ic = 0; ic < 2; ++ic) {
            gmp_randseed_ui(rng, (unsigned long)(BITS * 31337u + (unsigned)ic * 0x9e3779b9u));
            cases[ic].label = (ic == 0) ? "overflow (a+b>=N)" : "no-overflow (a+b<N)";
            mpz_init(cases[ic].a_gmp); mpz_init(cases[ic].b_gmp); mpz_init(cases[ic].n_gmp);
            mpz_t &N = cases[ic].n_gmp, &a = cases[ic].a_gmp, &b = cases[ic].b_gmp;
            mpz_urandomb(N, rng, BITS); mpz_setbit(N, BITS-1); mpz_setbit(N, 0);
            if (ic == 0) {  // overflow: a in [N/2, N), force a+b >= N
                mpz_t half; mpz_init(half); mpz_tdiv_q_ui(half, N, 2u);
                mpz_urandomm(a, rng, half); mpz_add(a, a, half);
                mpz_urandomm(b, rng, N);
                mpz_t sum; mpz_init(sum); mpz_add(sum, a, b);
                if (mpz_cmp(sum, N) < 0) { mpz_sub(b, N, a); mpz_sub_ui(b, b, 1u); }
                mpz_clear(sum); mpz_clear(half);
            } else {  // no-overflow: both < N/4
                mpz_t quar; mpz_init(quar); mpz_tdiv_q_ui(quar, N, 4u);
                mpz_urandomm(a, rng, quar); mpz_urandomm(b, rng, quar);
                mpz_clear(quar);
            }
            cases[ic].a_words.resize(WORDS); cases[ic].b_words.resize(WORDS); cases[ic].n_words.resize(WORDS);
            fill_from_gmp(a, cases[ic].a_words.data(), WORDS);
            fill_from_gmp(b, cases[ic].b_words.data(), WORDS);
            fill_from_gmp(N, cases[ic].n_words.data(), WORDS);
        }
        gmp_randclear(rng);
    }

    // Host buffers (reused across both cases)
    std::vector<uint32_t> host_a((size_t)instances * WORDS);
    std::vector<uint32_t> host_b((size_t)instances * WORDS);
    std::vector<uint32_t> host_n((size_t)instances * WORDS);
    std::vector<uint32_t> host_out((size_t)instances * WORDS);

    // Helper: load one case into host buffers
    auto upload_case = [&](int ic) {
        for (int i = 0; i < instances; ++i) {
            uint32_t base = (uint32_t)i * WORDS;
            for (uint32_t j = 0; j < WORDS; ++j) {
                host_a[base + j] = cases[ic].a_words[j];
                host_b[base + j] = cases[ic].b_words[j];
                host_n[base + j] = cases[ic].n_words[j];
            }
        }
    };

    const int case_idx = no_overflow ? 1 : 0;
    upload_case(case_idx);
    std::cout << "  [" << cases[case_idx].label << "]\n" << std::endl;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    std::string bench_src = cgbn::opencl::load_kernel_file("bench/ecm_addsub_bench.cl");
    if (bench_src.empty()) {
        std::cerr << "Failed to load ecm_addsub_bench.cl" << std::endl;
        return false;
    }
    bool asm_enabled = false;
    bool asm_b64_enabled = false;
    if (ecm_runtime_config().addsub_asm_disable) {
        std::cout << "--no-asm: skipping AMD asm kernels\n";
    } else if (WORDS == 8u || WORDS == 16u || WORDS == 128u) {
        asm_b64_enabled = ecm_runtime_config().addsub_asm_b64;
        asm_enabled = true;
    }
    EcmAddSubBuildManifest build =
        opencl_ecm_addsub_build_manifest(WORDS, asm_enabled, asm_b64_enabled);
    std::string src = bench_src;
    for (const std::string &rel : build.source_paths) {
        if (rel.find("ecm_addsub_bench.cl") != std::string::npos) {
            continue;
        }
        std::string part = cgbn::opencl::load_kernel_file(rel.c_str());
        if (part.empty()) {
            std::cerr << "Warning: missing " << rel << " (run python tools/mp_addsub/gen_all.py)\n";
            continue;
        }
        src += "\n" + part;
    }
    // ── Hot-loop wrappers for manifest / asm bench kernels ──────────
    cl_int buildErr = CL_SUCCESS;
    int fused_unroll = 2;
    {
        fused_unroll = ecm_runtime_config().add_mod_fused_unroll;
        if (fused_unroll != 1 && fused_unroll != 2) {
            std::cerr << "Warning: invalid --fused-unroll=" << fused_unroll
                      << ", fallback to 2\n";
            fused_unroll = 2;
        }
    }
    char build_opts[256];
    if (asm_enabled) {
        if (asm_b64_enabled) {
            snprintf(build_opts, sizeof(build_opts),
                     "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d -DMP_ADDMOD_ASM_ENABLE=1 "
                     "-DMP_ADDMOD_ASM_B64=1",
                     WORDS, fused_unroll);
        } else {
            snprintf(build_opts, sizeof(build_opts),
                     "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d -DMP_ADDMOD_ASM_ENABLE=1", WORDS,
                     fused_unroll);
        }
    } else {
        snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d", WORDS,
                 fused_unroll);
    }
    std::cout << "addsub build: fused_unroll=" << fused_unroll
              << " asm=" << (asm_enabled ? "1" : "0")
              << " asm_b64=" << (asm_b64_enabled ? "1" : "0")
              << " src_kib=" << (src.size() / 1024u) << std::endl;
    std::cout << "OpenCL compiling"
              << (asm_enabled && !asm_b64_enabled
                      ? " (b64 asm off; set ECM_ADDSUB_ASM_B64=1 to enable)"
                      : "")
              << "..."
              << std::flush;
    const auto compile_t0 = std::chrono::steady_clock::now();
    cl_program program = cgbn::opencl::build_program_from_source(
        ctx, src.c_str(), build_opts, buildErr);
    const auto compile_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - compile_t0)
                                .count();
    std::cout << " done in " << compile_ms << " ms" << std::endl;
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build ecm_addsub_bench.cl: " << buildErr << std::endl;
        return false;
    }

    size_t totalWords = (size_t)instances * WORDS;
    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_a.data(), &err);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_b.data(), &err);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_n.data(), &err);
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * totalWords, nullptr, &err);

    cl_uint limbs = WORDS;
    cl_uint vi = 1u;   // inner_iters=1 for verification / hot-loop single-shot kernels
    size_t global = (size_t)instances;

    const std::string &csv_path = ecm_runtime_config().bench_csv;
    bool csv_enabled = !csv_path.empty();
    std::ofstream csv;
    if (csv_enabled) {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv.is_open()) {
            csv << "kernel,ms,ops_per_s,private_mem_bytes,local_mem_bytes,preferred_wg_multiple,max_wg_size\n";
        } else {
            std::cerr << "Warning: failed to open ECM_BENCH_CSV path: " << csv_path << std::endl;
            csv_enabled = false;
        }
    }

    auto run_pure = [&](const char *kname, bool needsN, bool hot, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (needsN) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        } else if (std::string(kname) == "ecm_mp_sub_n") {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &limbs);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &limbs);
        }
        cl_uint inner_iters = (cl_uint)kernel_iterations;
        const int total_enqueues = hot ? launch_repeats : (launch_repeats * kernel_iterations);
        if (hot) {
            // hot-loop kernels: inner loop inside kernel, extra arg 5
            clSetKernelArg(k, needsN ? 5 : 4, sizeof(cl_uint), &inner_iters);
        }
        bool ok = run_kernel(ctx.queue, k, global, total_enqueues, ms_out);
        if (verbose) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg);
            double op_count = (double)instances * (double)total_enqueues;
            double ops_s = op_count / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b
                      << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref
                      << " max_wg=" << wg << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b
                    << "," << pref << "," << wg << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    auto run_pure_wg = [&](const char *kname, bool needsN, bool hot, size_t local_size, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
        clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
        clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        const int total_enqueues_wg = hot ? launch_repeats : (launch_repeats * kernel_iterations);
        if (hot) {
            cl_uint wg_iters = (cl_uint)kernel_iterations;
            clSetKernelArg(k, 5, sizeof(cl_uint), &wg_iters);
        }
        size_t global_wg = (size_t)instances * local_size;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total_enqueues_wg; ++i) {
            cl_int err2 =
                clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global_wg, &local_size, 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) {
                std::cerr << "Enqueue " << kname << " failed: " << err2 << std::endl;
                clReleaseKernel(k);
                return false;
            }
        }
        clFinish(ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
        size_t priv_b = 0, loc_b = 0, pref = 0, wg = 0;
        query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg);
        double op_count_wg = (double)instances * (double)total_enqueues_wg;
        double ops_s = op_count_wg / (ms_out / 1000.0);
        std::cout << "  [" << kname << "] wg=" << local_size << " private_mem=" << priv_b
                  << "B local_mem=" << loc_b << "B pref_wg=" << pref << " max_wg=" << wg << std::endl;
        if (csv_enabled) {
            csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << "," << pref
                << "," << wg << "\n";
        }
        clReleaseKernel(k);
        return true;
    };

    auto verify_add_mod_kernels = [&]() -> bool {
        auto run_once = [&](const char *kname) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &vi);
            size_t g = 1u;
            cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            return err2 == CL_SUCCESS;
        };

        bool ok_legacy = true;
        bool ok_mask = true;
        cl_int err2 = CL_SUCCESS;
        std::vector<uint32_t> out_legacy(WORDS);
        std::vector<uint32_t> out_mask(WORDS);
        if (!bench_unroll_only) {
            if (!run_once("ecm_mp_add_mod_legacy")) return false;
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_legacy.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;

            if (!run_once("ecm_mp_add_mod_mask")) return false;
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_mask.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
        }

        if (!run_once("ecm_mp_add_mod_fused")) return false;
        std::vector<uint32_t> out_fused(WORDS);
        err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, out_fused.data(),
                                  0, nullptr, nullptr);
        if (err2 != CL_SUCCESS) return false;

        bool ok_unroll = true;
        bool have_unroll = false;
        bool have_unroll_stage1 = false;
        {
            cl_int kerr = CL_SUCCESS;
            cl_kernel ku = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll = true;
                clSetKernelArg(ku, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ku, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ku, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ku, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ku, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(ku, 5, sizeof(cl_uint), &vi);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ku, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ku);
                if (err2 != CL_SUCCESS) return false;
            }
            cl_kernel ks = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll_auto", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll_stage1 = true;
                clSetKernelArg(ks, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ks, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ks, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ks, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ks, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(ks, 5, sizeof(cl_uint), &vi);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ks, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ks);
                if (err2 != CL_SUCCESS) return false;
            }
        }
        mpz_t expect, got_fused, got_unroll;
        mpz_t got_unroll_stage1;
        mpz_t got_legacy, got_mask;
        mpz_init(expect);
        mpz_init(got_fused);
        mpz_init(got_unroll);
        mpz_init(got_unroll_stage1);
        if (!bench_unroll_only) {
            mpz_init(got_legacy);
            mpz_init(got_mask);
        }
        mpz_add(expect, cases[case_idx].a_gmp, cases[case_idx].b_gmp);
        mpz_mod(expect, expect, cases[case_idx].n_gmp);

        auto verify_wg = [&](const char *kname, size_t local) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) return true;
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            size_t g = local;
            err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, &local, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            if (err2 != CL_SUCCESS) return false;
            std::vector<uint32_t> out_w(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, out_w.data(),
                                       0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            mpz_t got;
            mpz_init(got);
            fill_to_gmp(out_w.data(), WORDS, got);
            bool ok = (mpz_cmp(expect, got) == 0);
            if (!ok) {
                std::cerr << "add_mod verify: " << kname << " FAIL" << std::endl;
            }
            mpz_clear(got);
            return ok;
        };
        bool ok_lpt_all = true;
        bool ok_asm = true;
        if (asm_enabled) {
            std::vector<const char *> asm_kernels = {
                "ecm_mp_add_mod_fused_unroll_asm",
                "ecm_mp_add_mod_fused_unroll_asm_b16",
                "ecm_mp_add_mod_fused_unroll_asm_asmfix",
                "ecm_mp_add_mod_fused_unroll_asm_soft",
                "ecm_mp_add_mod_fused_unroll_asm_soft_b16",
            };
            if (WORDS == 128u) {
                asm_kernels.push_back("ecm_mp_add_mod_fused_unroll_asm_b32");
                if (asm_b64_enabled) {
                    asm_kernels.push_back("ecm_mp_add_mod_fused_unroll_asm_b64");
                }
            }
            if (!bench_unroll_only) {
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm_b16");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm_b16_vccsoft");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8_asmfix");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8_vccsoft");
            }
            for (const char *ak : asm_kernels) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel kt = clCreateKernel(program, ak, &kerr);
                if (kerr != CL_SUCCESS) continue;
                clReleaseKernel(kt);
                if (!run_once(ak)) continue;
                std::vector<uint32_t> out_asm(WORDS);
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_asm.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
                mpz_t got_asm;
                mpz_init(got_asm);
                fill_to_gmp(out_asm.data(), WORDS, got_asm);
                if (mpz_cmp(expect, got_asm) != 0) {
                    std::cerr << "add_mod verify: " << ak << " FAIL" << std::endl;
                    ok_asm = false;
                }
                mpz_clear(got_asm);
            }
        }
        const int lpt_chunks[] = {16, 32, 48, 64};
        if (have_unroll) {
            for (int chunk : lpt_chunks) {
                if (WORDS % (uint32_t)chunk != 0u) continue;
                if (WORDS / (uint32_t)chunk <= 1u) continue;
                char kname[64];
                std::snprintf(kname, sizeof(kname), "ecm_mp_add_mod_fused_lpt%d", chunk);
                if (!verify_wg(kname, (size_t)(WORDS / (uint32_t)chunk))) {
                    ok_lpt_all = false;
                }
            }
        }

        fill_to_gmp(out_fused.data(), WORDS, got_fused);
        if (!bench_unroll_only) {
            fill_to_gmp(out_legacy.data(), WORDS, got_legacy);
            fill_to_gmp(out_mask.data(), WORDS, got_mask);
            ok_legacy = (mpz_cmp(expect, got_legacy) == 0);
            ok_mask = (mpz_cmp(expect, got_mask) == 0);
        }
        bool ok_unroll_stage1 = true;
        if (have_unroll) {
            if (!run_once("ecm_mp_add_mod_fused_unroll")) return false;
            std::vector<uint32_t> out_unroll(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_unroll.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_unroll.data(), WORDS, got_unroll);
            ok_unroll = (mpz_cmp(expect, got_unroll) == 0);
        }
        if (have_unroll_stage1) {
            if (!run_once("ecm_mp_add_mod_fused_unroll_auto")) return false;
            std::vector<uint32_t> out_s(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_s.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_s.data(), WORDS, got_unroll_stage1);
            ok_unroll_stage1 = (mpz_cmp(expect, got_unroll_stage1) == 0);
        }
        bool ok_fused = (mpz_cmp(expect, got_fused) == 0);
        if ((!bench_unroll_only && (!ok_legacy || !ok_mask)) || !ok_fused || !ok_unroll ||
            !ok_unroll_stage1 || !ok_lpt_all || !ok_asm) {
            std::cerr << "add_mod verify:";
            if (!bench_unroll_only) {
                std::cerr << " legacy=" << (ok_legacy ? "PASS" : "FAIL")
                          << " mask=" << (ok_mask ? "PASS" : "FAIL");
            }
            std::cerr << " fused=" << (ok_fused ? "PASS" : "FAIL")
                      << " unroll=" << (ok_unroll ? "PASS" : "FAIL")
                      << " unroll_stage1=" << (ok_unroll_stage1 ? "PASS" : "FAIL")
                      << " lpt=" << (ok_lpt_all ? "PASS" : "FAIL")
                      << " asm=" << (ok_asm ? "PASS" : "FAIL") << std::endl;
            mpz_clears(expect, got_unroll_stage1, nullptr);
            if (!bench_unroll_only) {
                mpz_clears(got_legacy, got_mask, got_fused, got_unroll, nullptr);
            } else {
                mpz_clears(got_fused, got_unroll, nullptr);
            }
            return false;
        }
        std::cout << "  [ecm_mp_add_mod] GMP verify: PASS (";
        if (!bench_unroll_only) {
            std::cout << "legacy, mask, ";
        }
        std::cout << "fused";
        if (have_unroll) {
            std::cout << ", fused_unroll";
        }
        if (have_unroll_stage1) {
            std::cout << ", fused_unroll_auto";
        }
        if (have_unroll) {
            std::cout << ", lpt{16,32,48,64}";
        }
        if (asm_enabled) {
            std::cout << ", unroll_asm";
            if (WORDS == 128u) {
                std::cout << "+b32";
                if (asm_b64_enabled) {
                    std::cout << "+b64";
                }
            }
        }
        std::cout << ")" << std::endl;
        mpz_clears(expect, got_unroll_stage1, nullptr);
        if (!bench_unroll_only) {
            mpz_clears(got_legacy, got_mask, got_fused, got_unroll, nullptr);
        } else {
            mpz_clears(got_fused, got_unroll, nullptr);
        }
        return true;
    };

    auto verify_sub_mod_kernels = [&]() -> bool {
        auto run_once = [&](const char *kname) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &vi);
            size_t g = 1u;
            cl_int err2 =
                clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            return err2 == CL_SUCCESS;
        };

        if (!run_once("ecm_mp_sub_mod")) return false;
        std::vector<uint32_t> out_base(WORDS);
        cl_int err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                          out_base.data(), 0, nullptr, nullptr);
        if (err2 != CL_SUCCESS) return false;

        bool ok_unroll = true;
        bool have_unroll = false;
        bool have_unroll_stage1 = false;
        {
            cl_int kerr = CL_SUCCESS;
            cl_kernel ku = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll = true;
                clSetKernelArg(ku, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ku, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ku, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ku, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ku, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(ku, 5, sizeof(cl_uint), &vi);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ku, 1, nullptr, &g, nullptr, 0, nullptr,
                                              nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ku);
                if (err2 != CL_SUCCESS) return false;
            }
            cl_kernel ks = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll_auto", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll_stage1 = true;
                clSetKernelArg(ks, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ks, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ks, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ks, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ks, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(ks, 5, sizeof(cl_uint), &vi);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ks, 1, nullptr, &g, nullptr, 0, nullptr,
                                              nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ks);
                if (err2 != CL_SUCCESS) return false;
            }
        }

        mpz_t expect, got_base, got_unroll;
        mpz_t got_unroll_stage1;
        mpz_init(expect);
        mpz_init(got_base);
        mpz_init(got_unroll);
        mpz_init(got_unroll_stage1);
        mpz_sub(expect, cases[case_idx].a_gmp, cases[case_idx].b_gmp);
        mpz_mod(expect, expect, cases[case_idx].n_gmp);

        fill_to_gmp(out_base.data(), WORDS, got_base);
        bool ok_base = (mpz_cmp(expect, got_base) == 0);
        bool ok_unroll_stage1 = true;
        if (have_unroll) {
            if (!run_once("ecm_mp_sub_mod_fused_unroll")) return false;
            std::vector<uint32_t> out_unroll(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_unroll.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_unroll.data(), WORDS, got_unroll);
            ok_unroll = (mpz_cmp(expect, got_unroll) == 0);
        }
        if (have_unroll_stage1) {
            if (!run_once("ecm_mp_sub_mod_fused_unroll_auto")) return false;
            std::vector<uint32_t> out_s(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_s.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_s.data(), WORDS, got_unroll_stage1);
            ok_unroll_stage1 = (mpz_cmp(expect, got_unroll_stage1) == 0);
        }
        bool ok_asm = true;
        if (asm_enabled) {
            std::vector<const char *> asm_kernels;
            if (WORDS == 128u) {
                asm_kernels = {
                    "ecm_mp_sub_mod_fused_unroll_asm_b32",
                };
                if (asm_b64_enabled) {
                    asm_kernels.push_back("ecm_mp_sub_mod_fused_unroll_asm_b64");
                }
            }
            for (const char *ak : asm_kernels) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel kt = clCreateKernel(program, ak, &kerr);
                if (kerr != CL_SUCCESS) continue;
                clReleaseKernel(kt);
                if (!run_once(ak)) continue;
                std::vector<uint32_t> out_asm(WORDS);
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_asm.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
                mpz_t got_asm;
                mpz_init(got_asm);
                fill_to_gmp(out_asm.data(), WORDS, got_asm);
                if (mpz_cmp(expect, got_asm) != 0) {
                    std::cerr << "sub_mod verify: " << ak << " FAIL" << std::endl;
                    ok_asm = false;
                }
                mpz_clear(got_asm);
            }
        }

        if (!ok_base || !ok_unroll || !ok_unroll_stage1 || !ok_asm) {
            std::cerr << "sub_mod verify: base=" << (ok_base ? "PASS" : "FAIL")
                      << " unroll=" << (ok_unroll ? "PASS" : "FAIL")
                      << " unroll_stage1=" << (ok_unroll_stage1 ? "PASS" : "FAIL")
                      << " asm=" << (ok_asm ? "PASS" : "FAIL") << std::endl;
            mpz_clears(expect, got_base, got_unroll, got_unroll_stage1, nullptr);
            return false;
        }
        std::cout << "  [ecm_mp_sub_mod] GMP verify: PASS (base";
        if (have_unroll) {
            std::cout << ", fused_unroll";
        }
        if (have_unroll_stage1) {
            std::cout << ", fused_unroll_auto";
        }
        if (asm_enabled && WORDS == 128u) {
            std::cout << ", unroll_asm_b32";
            if (asm_b64_enabled) {
                std::cout << "+b64";
            }
        }
        std::cout << ")" << std::endl;
        mpz_clears(expect, got_base, got_unroll, got_unroll_stage1, nullptr);
        return true;
    };

    double t_add_n = 0.0, t_sub_n = 0.0, t_add_mod = 0.0, t_add_mod_legacy = 0.0, t_add_mod_mask = 0.0,
           t_add_mod_unroll = 0.0, t_add_mod_unroll_priv = 0.0, t_add_mod_unroll_auto = 0.0,
           t_add_mod_unroll_asm = 0.0,
           t_add_mod_unroll_asm_b16 = 0.0, t_add_mod_unroll_asm_asmfix = 0.0,
           t_add_mod_unroll_asm_soft = 0.0, t_add_mod_unroll_asm_soft_b16 = 0.0,
           t_add_mod_unroll_asm_b32 = 0.0, t_add_mod_unroll_asm_b64 = 0.0,
           t_add_mod_asm_b16 = 0.0, t_add_mod_asm_b16_vccsoft = 0.0, t_add_mod_asm8 = 0.0,
           t_add_mod_asm8_asmfix = 0.0, t_add_mod_asm8_vccsoft = 0.0,
           t_sub_mod = 0.0, t_sub_mod_unroll = 0.0, t_sub_mod_unroll_priv = 0.0,
           t_sub_mod_unroll_auto = 0.0,
           t_sub_mod_unroll_asm_b32 = 0.0, t_sub_mod_unroll_asm_b64 = 0.0;
    std::map<int, double> t_lpt_ms;
    if (!bench_unroll_only) {
        if (!run_pure("ecm_mp_add_n", false, true, t_add_n)) return false;
        if (!run_pure("ecm_mp_sub_n", false, true, t_sub_n)) return false;
    }
    if (!verify_add_mod_kernels()) return false;
    if (!verify_sub_mod_kernels()) return false;
    if (!bench_unroll_only) {
        if (!run_pure("ecm_mp_add_mod_legacy", true, true, t_add_mod_legacy)) return false;
        if (!run_pure("ecm_mp_add_mod_mask", true, true, t_add_mod_mask)) return false;
    }
    if (!run_pure("ecm_mp_add_mod_fused", true, true, t_add_mod)) return false;
    {
        cl_int kerr = CL_SUCCESS;
        cl_kernel ku = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(ku);
            if (!run_pure("ecm_mp_add_mod_fused_unroll", true, false, t_add_mod_unroll)) return false;
        } else {
            std::cout << "mp_add_mod_fused_unroll: (no kernel for MAX_LIMBS=" << WORDS << ")" << std::endl;
        }
        cl_kernel kp = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll_priv", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(kp);
            if (!run_pure("ecm_mp_add_mod_fused_unroll_priv", true, false, t_add_mod_unroll_priv)) return false;
        }
        {
            auto try_bench_stage1 = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, false, t_out);
            };
            try_bench_stage1("ecm_mp_add_mod_fused_unroll_auto", t_add_mod_unroll_auto);
        }
        if (asm_enabled) {
            auto try_bench_asm = [&](const char *kname, double &t_out) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, false, t_out);
            };
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm", t_add_mod_unroll_asm);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b16", t_add_mod_unroll_asm_b16);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_asmfix", t_add_mod_unroll_asm_asmfix);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_soft", t_add_mod_unroll_asm_soft);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_soft_b16", t_add_mod_unroll_asm_soft_b16);
            if (WORDS == 128u) {
                try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b32", t_add_mod_unroll_asm_b32);
                if (asm_b64_enabled) {
                    try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b64", t_add_mod_unroll_asm_b64);
                }
            }
            if (!bench_unroll_only) {
                try_bench_asm("ecm_mp_add_mod_fused_asm_b16", t_add_mod_asm_b16);
                try_bench_asm("ecm_mp_add_mod_fused_asm_b16_vccsoft", t_add_mod_asm_b16_vccsoft);
                try_bench_asm("ecm_mp_add_mod_fused_asm8", t_add_mod_asm8);
                try_bench_asm("ecm_mp_add_mod_fused_asm8_asmfix", t_add_mod_asm8_asmfix);
                try_bench_asm("ecm_mp_add_mod_fused_asm8_vccsoft", t_add_mod_asm8_vccsoft);
            }
        }
        const int lpt_chunks[] = {16, 32, 48, 64};
        for (int chunk : lpt_chunks) {
            if (WORDS % (uint32_t)chunk != 0u) continue;
            uint32_t threads = WORDS / (uint32_t)chunk;
            if (threads <= 1u) continue;
            char kname[64];
            std::snprintf(kname, sizeof(kname), "ecm_mp_add_mod_fused_lpt%d", chunk);
            cl_kernel kl = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) continue;
            clReleaseKernel(kl);
            double t_lpt = 0.0;
            if (!run_pure_wg(kname, true, false, (size_t)threads, t_lpt)) return false;
            t_lpt_ms[chunk] = t_lpt;
        }
    }
    if (!run_pure("ecm_mp_sub_mod", true, true, t_sub_mod)) return false;
    {
        cl_int kerr = CL_SUCCESS;
        cl_kernel ku = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(ku);
            if (!run_pure("ecm_mp_sub_mod_fused_unroll", true, false, t_sub_mod_unroll)) return false;
        } else {
            std::cout << "mp_sub_mod_fused_unroll: (no kernel for MAX_LIMBS=" << WORDS << ")"
                      << std::endl;
        }
        cl_kernel kp = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll_priv", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(kp);
            if (!run_pure("ecm_mp_sub_mod_fused_unroll_priv", true, false, t_sub_mod_unroll_priv))
                return false;
        }
        {
            auto try_bench_sub_stage1 = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, false, t_out);
            };
            try_bench_sub_stage1("ecm_mp_sub_mod_fused_unroll_auto", t_sub_mod_unroll_auto);
        }
        if (asm_enabled && WORDS == 128u) {
            auto try_bench_sub_asm = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, false, t_out);
            };
            try_bench_sub_asm("ecm_mp_sub_mod_fused_unroll_asm_b32", t_sub_mod_unroll_asm_b32);
            if (asm_b64_enabled) {
                try_bench_sub_asm("ecm_mp_sub_mod_fused_unroll_asm_b64", t_sub_mod_unroll_asm_b64);
            }
        }
    }

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Read back failed: " << err << std::endl;
        return false;
    }

    double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
    {
        struct Row {
            std::string path;
            double ms;
        };
        std::vector<Row> rows;
        auto push = [&](const char *path, double ms) {
            if (ms > 0.0) {
                rows.push_back({path, ms});
            }
        };
        auto push_lpt = [&](int chunk, double ms) {
            if (ms > 0.0) {
                rows.push_back({"fused_lpt" + std::to_string(chunk), ms});
            }
        };
        if (asm_enabled) {
            push("fused_unroll_asm_b64", t_add_mod_unroll_asm_b64);
            push("fused_unroll_asm_b32", t_add_mod_unroll_asm_b32);
            push("fused_unroll_asm_soft_b16", t_add_mod_unroll_asm_soft_b16);
            push("fused_unroll_asm_b16", t_add_mod_unroll_asm_b16);
            push("fused_unroll_asm_soft_b8", t_add_mod_unroll_asm_soft);
            push("fused_unroll_asm_asmfix_b8", t_add_mod_unroll_asm_asmfix);
            push("fused_unroll_asm_b8", t_add_mod_unroll_asm);
            if (!bench_unroll_only) {
                push("fused_asm_b16_vccsoft", t_add_mod_asm_b16_vccsoft);
                push("fused_asm_b16", t_add_mod_asm_b16);
            }
        }
        const int lpt_chunks[] = {64, 48, 32, 16};
        for (int chunk : lpt_chunks) {
            auto it = t_lpt_ms.find(chunk);
            if (it != t_lpt_ms.end()) {
                push_lpt(chunk, it->second);
            }
        }
        push("fused_unroll", t_add_mod_unroll);
        push("fused_unroll_priv", t_add_mod_unroll_priv);
        push("fused_unroll_auto", t_add_mod_unroll_auto);
        if (!bench_unroll_only) {
            push("fused", t_add_mod);
            push("mask", t_add_mod_mask);
            push("legacy", t_add_mod_legacy);
        }
        std::cout << "\n--- mp_add_mod (priority high -> low) ---\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const double ops = op_count / (rows[i].ms / 1000.0);
            std::cout << "  [" << (i + 1) << "] " << rows[i].path << ": " << rows[i].ms << " ms, "
                      << ops << " ops/s";
            if (i + 1 < rows.size()) {
                std::cout << " (" << (rows[i + 1].ms / rows[i].ms) << "x vs next tier)";
            } else if (t_add_mod > 0.0 && rows[i].path != "fused") {
                std::cout << " (" << (t_add_mod / rows[i].ms) << "x vs fused)";
            }
            std::cout << std::endl;
        }
        auto print_sub = [&](const char *path, double ms) {
            if (ms > 0.0) {
                std::cout << "  " << path << ": " << ms << " ms, "
                          << (op_count / (ms / 1000.0)) << " ops/s\n";
            }
        };
        std::cout << "--- mp_sub_mod (priority high -> low) ---\n";
        if (asm_enabled && WORDS == 128u) {
            print_sub("fused_unroll_asm_b64", t_sub_mod_unroll_asm_b64);
            print_sub("fused_unroll_asm_b32", t_sub_mod_unroll_asm_b32);
        }
        print_sub("fused_unroll", t_sub_mod_unroll);
        print_sub("fused_unroll_priv", t_sub_mod_unroll_priv);
        print_sub("fused_unroll_auto", t_sub_mod_unroll_auto);
        print_sub("fused_loop", t_sub_mod);
        std::cout << std::endl;
    }
    if (!bench_unroll_only) {
        std::cout << "mp_add_n:   " << t_add_n << " ms, " << (op_count / (t_add_n / 1000.0)) << " ops/s"
                  << std::endl;
        std::cout << "mp_sub_n:   " << t_sub_n << " ms, " << (op_count / (t_sub_n / 1000.0)) << " ops/s"
                  << std::endl;
        std::cout << "mp_add_mod_legacy: " << t_add_mod_legacy << " ms, "
                  << (op_count / (t_add_mod_legacy / 1000.0)) << " ops/s" << std::endl;
        std::cout << "mp_add_mod_mask:   " << t_add_mod_mask << " ms, "
                  << (op_count / (t_add_mod_mask / 1000.0)) << " ops/s"
                  << " (vs legacy: " << (t_add_mod_legacy / t_add_mod_mask) << "x)" << std::endl;
    }
    std::cout << "mp_add_mod_fused:  " << t_add_mod << " ms, " << (op_count / (t_add_mod / 1000.0))
              << " ops/s";
    if (!bench_unroll_only) {
        std::cout << " (vs legacy: " << (t_add_mod_legacy / t_add_mod) << "x)";
    }
    std::cout << std::endl;
    if (t_add_mod_unroll > 0.0) {
        std::cout << "mp_add_mod_fused_unroll:               " << t_add_mod_unroll << " ms, "
                  << (op_count / (t_add_mod_unroll / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll) << "x";
        if (!bench_unroll_only) {
            std::cout << ", vs legacy: " << (t_add_mod_legacy / t_add_mod_unroll) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_add_mod_unroll_priv > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_priv:          " << t_add_mod_unroll_priv << " ms, "
                  << (op_count / (t_add_mod_unroll_priv / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll_priv) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_priv) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_auto > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_auto:        " << t_add_mod_unroll_auto << " ms, "
                  << (op_count / (t_add_mod_unroll_auto / 1000.0)) << " ops/s (ECM fused_unroll, "
                  << "vs fused: " << (t_add_mod / t_add_mod_unroll_auto) << "x";
        if (t_add_mod_unroll > 0.0) {
            std::cout << ", vs scalar unroll: " << (t_add_mod_unroll / t_add_mod_unroll_auto) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_add_mod_unroll_asm > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b8:        " << t_add_mod_unroll_asm << " ms, "
                  << (op_count / (t_add_mod_unroll_asm / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll_asm) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_asmfix > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_fix_b8:    "
                  << t_add_mod_unroll_asm_asmfix << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_asmfix / 1000.0)) << " ops/s (vs c fix: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_asmfix) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_asmfix) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_soft > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_soft_b8:   " << t_add_mod_unroll_asm_soft
                  << " ms, " << (op_count / (t_add_mod_unroll_asm_soft / 1000.0)) << " ops/s (vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_soft) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_soft) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b16:       " << t_add_mod_unroll_asm_b16
                  << " ms, " << (op_count / (t_add_mod_unroll_asm_b16 / 1000.0)) << " ops/s (vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_b16) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b16) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_soft_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_soft_b16:  "
                  << t_add_mod_unroll_asm_soft_b16 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_soft_b16 / 1000.0)) << " ops/s (vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_soft_b16) << "x, vs soft b8: "
                  << (t_add_mod_unroll_asm_soft / t_add_mod_unroll_asm_soft_b16) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_soft_b16) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b32 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b32:       "
                  << t_add_mod_unroll_asm_b32 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_b32 / 1000.0)) << " ops/s (vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_b32) << "x, vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_b32) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b32) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b64 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b64:       "
                  << t_add_mod_unroll_asm_b64 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_b64 / 1000.0)) << " ops/s (vs b32: "
                  << (t_add_mod_unroll_asm_b32 / t_add_mod_unroll_asm_b64) << "x, vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_b64) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b64) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_asm_b16 (512b block16 chain): " << t_add_mod_asm_b16 << " ms, "
                  << (op_count / (t_add_mod_asm_b16 / 1000.0)) << " ops/s" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm_b16_vccsoft > 0.0) {
        std::cout << "mp_add_mod_fused_asm_b16_vccsoft (512b VCC switch): " << t_add_mod_asm_b16_vccsoft
                  << " ms, " << (op_count / (t_add_mod_asm_b16_vccsoft / 1000.0)) << " ops/s (vs b16 chain: "
                  << (t_add_mod_asm_b16 / t_add_mod_asm_b16_vccsoft) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8 > 0.0) {
        std::cout << "mp_add_mod_fused_asm8 (fix=c ulong): " << t_add_mod_asm8 << " ms, "
                  << (op_count / (t_add_mod_asm8 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_asm8) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8_asmfix > 0.0) {
        std::cout << "mp_add_mod_fused_asm8_asmfix (fix=v_add_co asm): " << t_add_mod_asm8_asmfix
                  << " ms, " << (op_count / (t_add_mod_asm8_asmfix / 1000.0)) << " ops/s (vs c fix: "
                  << (t_add_mod_asm8 / t_add_mod_asm8_asmfix) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8_vccsoft > 0.0) {
        std::cout << "mp_add_mod_fused_asm8_vccsoft (VCC switch): " << t_add_mod_asm8_vccsoft << " ms, "
                  << (op_count / (t_add_mod_asm8_vccsoft / 1000.0)) << " ops/s (vs asm8: "
                  << (t_add_mod_asm8 / t_add_mod_asm8_vccsoft) << "x)" << std::endl;
    }
    if (!t_lpt_ms.empty()) {
        std::cout << "mp_add_mod_fused_lpt (limbs/thread):" << std::endl;
        for (int chunk : {16, 32, 48, 64}) {
            auto it = t_lpt_ms.find(chunk);
            if (it == t_lpt_ms.end()) {
                if (WORDS % (uint32_t)chunk != 0u) {
                    std::cout << "  lpt" << chunk << ": (n/a, " << WORDS << " limbs not divisible by " << chunk
                              << ")" << std::endl;
                }
                continue;
            }
            double ms = it->second;
            uint32_t threads = WORDS / (uint32_t)chunk;
            double ops = op_count / (ms / 1000.0);
            std::cout << "  lpt" << chunk << " (" << threads << " thr): " << ms << " ms, " << ops
                      << " ops/s (vs fused: " << (t_add_mod / ms) << "x";
            if (t_add_mod_unroll > 0.0) {
                std::cout << ", vs unroll: " << (t_add_mod_unroll / ms) << "x";
            }
            std::cout << ")" << std::endl;
        }
    }
    if (!bench_unroll_only) {
        std::cout << "mp_sub_mod: " << t_sub_mod << " ms, " << (op_count / (t_sub_mod / 1000.0))
                  << " ops/s" << std::endl;
    }
    if (t_sub_mod_unroll > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll:               " << t_sub_mod_unroll << " ms, "
                  << (op_count / (t_sub_mod_unroll / 1000.0)) << " ops/s (vs base: "
                  << (t_sub_mod / t_sub_mod_unroll) << "x)" << std::endl;
    }
    if (t_sub_mod_unroll_priv > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_priv:          " << t_sub_mod_unroll_priv << " ms, "
                  << (op_count / (t_sub_mod_unroll_priv / 1000.0)) << " ops/s (vs base: "
                  << (t_sub_mod / t_sub_mod_unroll_priv) << "x, vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_priv) << "x)" << std::endl;
    }
    if (t_sub_mod_unroll_auto > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_auto:        " << t_sub_mod_unroll_auto << " ms, "
                  << (op_count / (t_sub_mod_unroll_auto / 1000.0)) << " ops/s (ECM fused_unroll, "
                  << "vs base: " << (t_sub_mod / t_sub_mod_unroll_auto) << "x";
        if (t_sub_mod_unroll > 0.0) {
            std::cout << ", vs scalar unroll: " << (t_sub_mod_unroll / t_sub_mod_unroll_auto) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_sub_mod_unroll_asm_b32 > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_asm_b32:       " << t_sub_mod_unroll_asm_b32 << " ms, "
                  << (op_count / (t_sub_mod_unroll_asm_b32 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_asm_b32) << "x";
        if (t_sub_mod_unroll_asm_b64 > 0.0) {
            std::cout << ", vs b64: " << (t_sub_mod_unroll_asm_b64 / t_sub_mod_unroll_asm_b32) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_sub_mod_unroll_asm_b64 > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_asm_b64:       " << t_sub_mod_unroll_asm_b64 << " ms, "
                  << (op_count / (t_sub_mod_unroll_asm_b64 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_asm_b64) << "x";
        if (t_sub_mod_unroll_asm_b32 > 0.0) {
            std::cout << ", vs b32: " << (t_sub_mod_unroll_asm_b32 / t_sub_mod_unroll_asm_b64) << "x";
        }
        std::cout << ")" << std::endl;
    }
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    if (csv_enabled && csv.is_open()) csv.close();
    cgbn::opencl::destroy_context(ctx);
    for (int ic = 0; ic < 2; ++ic) {
        mpz_clear(cases[ic].a_gmp);
        mpz_clear(cases[ic].b_gmp);
        mpz_clear(cases[ic].n_gmp);
    }
    return true;
}

// ── Width-specific Stage-1 operator bench ────────────────────────────
static bool run_width_specific_addsub_bench(int bits, int kernel_iterations,
                                            int instances, int launch_repeats,
                                            bool verbose, bool no_overflow) {
    if (bits <= 0 || bits % 32 != 0) return false;

    static const int kWidths[] = {128,192,256,384,512,768,1024,1536,2048,2560,3072,3584,4096};
    int bench_width = 0;
    for (int w : kWidths) { if (bits == w) { bench_width = w; break; } }
    if (bench_width == 0) return true;

    const std::string W  = std::to_string(bench_width);
    const uint32_t WORDS = (uint32_t)bits / 32u;
    const std::string UL = std::to_string(WORDS);

    std::string add_asm_src = cgbn::opencl::load_kernel_file(("add_mod/add_mod_asm_" + W + "b.cl").c_str());
    std::string sub_asm_src = cgbn::opencl::load_kernel_file(("sub_mod/sub_mod_asm_" + W + "b.cl").c_str());
    std::string add_unroll_src = cgbn::opencl::load_kernel_file(("add_mod/add_mod_unroll_" + W + "b.cl").c_str());
    std::string sub_unroll_src = cgbn::opencl::load_kernel_file(("sub_mod/sub_mod_unroll_" + W + "b.cl").c_str());
    std::string asm_common_src = cgbn::opencl::load_kernel_file("common/asm_common.h.cl");

    bool have_asm = !add_asm_src.empty() && !sub_asm_src.empty() && !asm_common_src.empty();
    bool have_unroll = !add_unroll_src.empty() && !sub_unroll_src.empty();
    if (!have_asm && !have_unroll) return true;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) { std::cerr << "width bench: context failed " << err << "\n"; return false; }

    // Detect GPU vendor: ASM paths only valid on AMD (asm_common.h.cl macros)
    char vendor_str[256] = {};
    clGetDeviceInfo(ctx.device, CL_DEVICE_VENDOR, sizeof(vendor_str), vendor_str, nullptr);
    const bool is_amd = (std::strstr(vendor_str, "AMD") != nullptr ||
                         std::strstr(vendor_str, "Advanced Micro Devices") != nullptr);

    // ASM program: include asm_common.h.cl + asm wrappers, AMD only, honour --no-asm
    bool asm_disabled = ecm_runtime_config().addsub_asm_disable;
    cl_program prog_asm = nullptr;
    if (have_asm && is_amd && !asm_disabled) {
        const std::string kiters = std::to_string(kernel_iterations);
        std::string src_asm = asm_common_src + "\n" + add_asm_src + "\n" + sub_asm_src;
        src_asm +=
            "\n__kernel void ecm_add_asm_" + W + "b_bench(__global uint *a,__global uint *b,"
            "__global uint *n,__global uint *out,uint limbs){\n"
            " uint gid=get_global_id(0);uint base=gid*limbs;\n"
            " uint la["+UL+"],lb["+UL+"],ln["+UL+"],lout["+UL+"];\n"
            " for(uint i=0;i<limbs;++i){la[i]=a[base+i];lb[i]=b[base+i];ln[i]=n[i];}\n"
            " for(uint it=0;it<"+kiters+";++it){\n"
            "  if(it==0)add_mod_asm_"+W+"b(lout,la,lb,ln,limbs);\n"
            "  else add_mod_asm_"+W+"b(lout,lout,lb,ln,limbs);\n"
            " }\n"
            " for(uint i=0;i<limbs;++i)out[base+i]=lout[i];\n}\n"
            "\n__kernel void ecm_sub_asm_" + W + "b_bench(__global uint *a,__global uint *b,"
            "__global uint *n,__global uint *out,uint limbs){\n"
            " uint gid=get_global_id(0);uint base=gid*limbs;\n"
            " uint la["+UL+"],lb["+UL+"],ln["+UL+"],lout["+UL+"];\n"
            " for(uint i=0;i<limbs;++i){la[i]=a[base+i];lb[i]=b[base+i];ln[i]=n[i];}\n"
            " for(uint it=0;it<"+kiters+";++it){\n"
            "  if(it==0)sub_mod_asm_"+W+"b(lout,la,lb,ln,limbs);\n"
            "  else sub_mod_asm_"+W+"b(lout,lout,lb,ln,limbs);\n"
            " }\n"
            " for(uint i=0;i<limbs;++i)out[base+i]=lout[i];\n}\n";
        cl_int berr = CL_SUCCESS;
        prog_asm = cgbn::opencl::build_program_from_source(ctx, src_asm.c_str(),
            ("-DMAX_LIMBS=" + std::to_string(WORDS)).c_str(), berr);
    }

    // Unroll program — inline-loop uses kernel_iterations to amortise global↔private copy
    cl_program prog_unroll = nullptr;
    if (have_unroll) {
        const std::string kiters = std::to_string(kernel_iterations);
        std::string src_unroll = add_unroll_src + "\n" + sub_unroll_src;
        src_unroll +=
            "\n__kernel void ecm_add_unroll_" + W + "b_bench(__global uint *a,__global uint *b,"
            "__global uint *n,__global uint *out,uint limbs){\n"
            " uint gid=get_global_id(0);uint base=gid*limbs;\n"
            " uint la["+UL+"],lb["+UL+"],ln["+UL+"],lout["+UL+"];\n"
            " for(uint i=0;i<limbs;++i){la[i]=a[base+i];lb[i]=b[base+i];ln[i]=n[i];}\n"
            " for(uint it=0;it<"+kiters+";++it){\n"
            "  if(it==0)add_mod_unroll_"+W+"b(lout,la,lb,ln,limbs);\n"
            "  else add_mod_unroll_"+W+"b(lout,lout,lb,ln,limbs);\n"
            " }\n"
            " for(uint i=0;i<limbs;++i)out[base+i]=lout[i];\n}\n"
            "\n__kernel void ecm_sub_unroll_" + W + "b_bench(__global uint *a,__global uint *b,"
            "__global uint *n,__global uint *out,uint limbs){\n"
            " uint gid=get_global_id(0);uint base=gid*limbs;\n"
            " uint la["+UL+"],lb["+UL+"],ln["+UL+"],lout["+UL+"];\n"
            " for(uint i=0;i<limbs;++i){la[i]=a[base+i];lb[i]=b[base+i];ln[i]=n[i];}\n"
            " for(uint it=0;it<"+kiters+";++it){\n"
            "  if(it==0)sub_mod_unroll_"+W+"b(lout,la,lb,ln,limbs);\n"
            "  else sub_mod_unroll_"+W+"b(lout,lout,lb,ln,limbs);\n"
            " }\n"
            " for(uint i=0;i<limbs;++i)out[base+i]=lout[i];\n}\n";
        cl_int berr = CL_SUCCESS;
        prog_unroll = cgbn::opencl::build_program_from_source(ctx, src_unroll.c_str(),
            ("-DMAX_LIMBS=" + std::to_string(WORDS)).c_str(), berr);
    }

    size_t totalWords = (size_t)instances * WORDS;
    std::vector<uint32_t> host_a(totalWords), host_b(totalWords), host_n(WORDS);

    // Fill with random GMP data (same seed scheme as main bench, case-sensitive)
    {
        gmp_randstate_t rng;
        gmp_randinit_default(rng);
        gmp_randseed_ui(rng, (unsigned long)((uint32_t)bench_width * 31337u + (no_overflow ? 1u : 0u) * 0x9e3779b9u));
        mpz_t N, a, b;
        mpz_init(N); mpz_init(a); mpz_init(b);
        mpz_urandomb(N, rng, bench_width); mpz_setbit(N, bench_width-1); mpz_setbit(N, 0);
        if (no_overflow) {
            mpz_t quar; mpz_init(quar); mpz_tdiv_q_ui(quar, N, 4u);
            mpz_urandomm(a, rng, quar); mpz_urandomm(b, rng, quar);
            mpz_clear(quar);
        } else {
            mpz_t half; mpz_init(half); mpz_tdiv_q_ui(half, N, 2u);
            mpz_urandomm(a, rng, half); mpz_add(a, a, half);
            mpz_urandomm(b, rng, N);
            mpz_t sum; mpz_init(sum); mpz_add(sum, a, b);
            if (mpz_cmp(sum, N) < 0) { mpz_sub(b, N, a); mpz_sub_ui(b, b, 1u); }
            mpz_clear(sum); mpz_clear(half);
        }
        std::vector<uint32_t> aw(WORDS), bw(WORDS), nw(WORDS);
        fill_from_gmp(a, aw.data(), WORDS);
        fill_from_gmp(b, bw.data(), WORDS);
        fill_from_gmp(N, nw.data(), WORDS);
        for (size_t i = 0; i < (size_t)instances; ++i) {
            uint32_t base = (uint32_t)i * WORDS;
            for (uint32_t j = 0; j < WORDS; ++j) {
                host_a[base + j] = aw[j];
                host_b[base + j] = bw[j];
                host_n[j] = nw[j];
            }
        }
        mpz_clear(a); mpz_clear(b); mpz_clear(N);
        gmp_randclear(rng);
    }

    cl_int cerr;
    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(uint32_t)*totalWords,host_a.data(),&cerr);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(uint32_t)*totalWords,host_b.data(),&cerr);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(uint32_t)*WORDS,     host_n.data(),&cerr);
    cl_mem bufO = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE,                          sizeof(uint32_t)*totalWords,nullptr,&cerr);
    cl_uint limbs = WORDS;
    size_t global = (size_t)instances;

    auto wrun = [&](cl_program prog, const char *kname, bool is_add, double &ms) -> bool {
        if (prog == nullptr) return false;
        cl_int kerr; cl_kernel k = clCreateKernel(prog, kname, &kerr);
        if (kerr != CL_SUCCESS) return false;
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (is_add) {clSetKernelArg(k,1,sizeof(cl_mem),&bufB);clSetKernelArg(k,2,sizeof(cl_mem),&bufN);clSetKernelArg(k,3,sizeof(cl_mem),&bufO);clSetKernelArg(k,4,sizeof(cl_uint),&limbs);}
        else        {clSetKernelArg(k,1,sizeof(cl_mem),&bufB);clSetKernelArg(k,2,sizeof(cl_mem),&bufN);clSetKernelArg(k,3,sizeof(cl_mem),&bufO);clSetKernelArg(k,4,sizeof(cl_uint),&limbs);}
        int total = launch_repeats;  // kernel_iterations now inside the bench wrapper loop
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total; ++i)
            if (clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr) != CL_SUCCESS)
                { clReleaseKernel(k); return false; }
        clFinish(ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (verbose) {
            size_t pb=0,lb=0,pr=0,wg=0;
            query_kernel_resources(k, ctx.device, pb, lb, pr, wg);
            std::cout << "  [" << kname << "] priv=" << pb << "B loc=" << lb << "B pref=" << pr << " wg=" << wg << "\n";
        }
        clReleaseKernel(k);
        return true;
    };

    std::cout << "\n-- Stage-1 fixed-width -- " << bench_width << "b --\n";
    double ta=0,tu=0,tsa=0,tsu=0;
    if (wrun(prog_asm, ("ecm_add_asm_"+W+"b_bench").c_str(), true, ta)) {
        double ops = (double)instances*kernel_iterations*launch_repeats/(ta/1000.0);
        std::cout << "  add_mod_asm_" << W << "b:     " << ta << " ms, " << ops << " ops/s\n";
    }
    if (wrun(prog_unroll, ("ecm_add_unroll_"+W+"b_bench").c_str(), true, tu)) {
        double ops = (double)instances*kernel_iterations*launch_repeats/(tu/1000.0);
        std::cout << "  add_mod_unroll_" << W << "b:  " << tu << " ms, " << ops << " ops/s\n";
    }
    if (wrun(prog_asm, ("ecm_sub_asm_"+W+"b_bench").c_str(), false, tsa)) {
        double ops = (double)instances*kernel_iterations*launch_repeats/(tsa/1000.0);
        std::cout << "  sub_mod_asm_" << W << "b:     " << tsa << " ms, " << ops << " ops/s\n";
    }
    if (wrun(prog_unroll, ("ecm_sub_unroll_"+W+"b_bench").c_str(), false, tsu)) {
        double ops = (double)instances*kernel_iterations*launch_repeats/(tsu/1000.0);
        std::cout << "  sub_mod_unroll_" << W << "b:  " << tsu << " ms, " << ops << " ops/s\n";
    }

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufN); clReleaseMemObject(bufO);
    if (prog_asm) clReleaseProgram(prog_asm);
    if (prog_unroll) clReleaseProgram(prog_unroll);
    cgbn::opencl::destroy_context(ctx);
    return true;
}

#ifdef BUILD_OPENCL_ECM_ADDSUB_MAIN
#include <cstdlib>
int main(int argc, char **argv) {
    int bits = 1024;
    int kernel_iterations = 1000;
    int instances = 128;
    int launch_repeats = 10;
    bool bench_unroll_only = false;
    bool verbose = false;
    bool fixed_width_only = false;
    bool no_overflow = false;
    int device_index = -1;
    auto print_usage = [&]() {
        std::cout
            << "Usage: opencl_ecm_addsub [options] [bits] [kernel_iterations] [instances] [launch_repeats]\n"
            << "  Positional args:\n"
            << "    bits                    Benchmark bit width (multiple of 32, <= 8192, default: 1024)\n"
            << "    kernel_iterations       Kernel inner-loop count; supports 1e6 notation (default: 1000)\n"
            << "    instances               Batched instances (default: 128)\n"
            << "    launch_repeats          Measurement repeats (default: 10)\n"
            << "  Options:\n"
            << "  --bits <bits>            Alias for 1st positional (multiple of 32, <= 8192)\n"
            << "  --unroll                 Only benchmark fused_unroll / asm unroll / lpt paths\n"
            << "  --fixed                  Only benchmark Stage-1 fixed-width operators\n"
            << "  --no-overflow            Use a+b < N test data (default: a+b >= N)\n"
            << "  -v, --verbose            Verbose: print kernel resource details\n"
            << "  -d, --device <index>     OpenCL device index\n"
            << "  --no-asm                 Skip AMD asm kernels (was ECM_ADDSUB_ASM_DISABLE)\n"
            << "  --asm-b64                Enable b64 asm kernels (was ECM_ADDSUB_ASM_B64)\n"
            << "  --addsub-fused-unroll <1|2>  add/sub fused-unroll mode\n"
            << "  --csv <file>             Write results CSV (was ECM_BENCH_CSV)\n"
            << "  -h, --help               Show this help message\n"
            << "\nExamples:\n"
            << "  opencl_ecm_addsub -d 1 512 1e4 16 1 --fixed\n"
            << "  opencl_ecm_addsub -d 1 512 5000 6144 1\n";
    };
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage();
            return EXIT_SUCCESS;
        }
        if (a == "--bits" && i + 1 < argc) {
            bits = std::stoi(std::string(argv[++i]));
            continue;
        }
        if (a == "--unroll") {
            bench_unroll_only = true;
            continue;
        }
        if (a == "-v" || a == "--verbose") {
            verbose = true;
            continue;
        }
        if (a == "--fixed") {
            fixed_width_only = true;
            continue;
        }
        if (a == "--no-overflow") {
            no_overflow = true;
            continue;
        }
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            device_index = std::stoi(std::string(argv[++i]));
            continue;
        }
        if (a == "--no-asm") { ecm_runtime_config().addsub_asm_disable = true; continue; }
        if (a == "--asm-b64") { ecm_runtime_config().addsub_asm_b64 = true; continue; }
        if (a == "--addsub-fused-unroll" && i + 1 < argc) { ecm_runtime_config().add_mod_fused_unroll = std::stoi(std::string(argv[++i])); continue; }
        if (a == "--csv" && i + 1 < argc) { ecm_runtime_config().bench_csv = argv[++i]; continue; }
        pos.push_back(a);
    }
    if (device_index >= 0) { ecm_runtime_config().device_index = device_index; }
    auto parse_count = [&](const std::string &s, const char *label, int &out) {
        try { double d = std::stod(s); out = (int)(d + 0.5); } catch (...) {
            std::cerr << "Invalid " << label << ": " << s << std::endl; return false; }
        return true;
    };
    if (pos.size() >= 1 && !parse_count(pos[0], "bits", bits)) return EXIT_FAILURE;
    if (pos.size() >= 2 && !parse_count(pos[1], "kernel_iterations", kernel_iterations)) return EXIT_FAILURE;
    if (pos.size() >= 3 && !parse_count(pos[2], "instances", instances)) return EXIT_FAILURE;
    if (pos.size() >= 4 && !parse_count(pos[3], "launch_repeats", launch_repeats)) return EXIT_FAILURE;

    bool ok = true;
    if (!fixed_width_only) {
        ok = runOpenClEcmAddSubBenchmark(bits, kernel_iterations, instances, launch_repeats,
                                         bench_unroll_only, verbose, no_overflow);
    }
    if (ok || fixed_width_only) {
        run_width_specific_addsub_bench(bits, kernel_iterations, instances, launch_repeats, verbose, no_overflow);
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
