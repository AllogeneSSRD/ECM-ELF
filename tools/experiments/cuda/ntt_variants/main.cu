#include "engine.cuh"
int main(int argc, char **argv)
{
    /* Unbuffered stdout: this probe allocates GBs and is run under scripts, and a run that
       dies (or is killed) must not lose the lines that say how far it got.  Same lesson as
       the stage2 tree reference (docs section 15.4). */
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (gl_selftest() != 0) return 2;

    const std::string mode = (argc > 1) ? argv[1] : "poly";
    /*
     * nttcheck [log2 N] [device] -- the transform's own selftest, using the O(n^2) direct
     * DFT kernel as the oracle (it shares no index algebra with the butterfly stages, so
     * the two cannot agree by accident):
     *   (1) forward NTT of a fixed pseudo-random vector == the direct DFT sum, accounting
     *       for the bit-reversed output order;
     *   (2) inverse(forward(x)) == x, bit-exact;
     *   (3) the twiddle powers gl_twiddle(omega, i) == omega^i.
     * The earlier butterfly stages failed (1) while passing (3), which is how the bug in the
     * per-stage index decomposition was localised.
     */
    if (mode == "nttcheck") {
        const int logn = (argc > 2) ? std::atoi(argv[2]) : 8;
        const int dev = (argc > 3) ? std::atoi(argv[3]) : 0;
        CK(cudaSetDevice(dev));
        const unsigned long long n = 1ull << logn;
        const unsigned long long om = gl_pow_host(7ull, (GL_P - 1ull) / n);
        const unsigned long long omi = gl_pow_host(om, GL_P - 2ull);
        const unsigned long long nsc = gl_pow_host(n % GL_P, GL_P - 2ull);
        /* the FUSED transform (section 12) is what is checked here: same stages, same order,
           same twiddles -- the pass structure is the only thing that changed */
        FuseCtx fc;
        fuse_init(fc, n, logn, om, omi);
        std::vector<unsigned long long> x(n, 0);
        unsigned long long s = 0x1234567ull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x[(size_t)i] = s % GL_P;
        }
        unsigned long long *dx = nullptr, *dsp = nullptr, *dref = nullptr, *dtw = nullptr,
                           *dback = nullptr, *dsave = nullptr;
        CK(cudaMalloc(&dx, n * 8));
        CK(cudaMalloc(&dsp, n * 8));
        CK(cudaMalloc(&dref, n * 8));
        CK(cudaMalloc(&dtw, n * 8));
        CK(cudaMalloc(&dback, n * 8));
        CK(cudaMalloc(&dsave, n * 8));
        CK(cudaMemcpy(dx, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsp, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsave, x.data(), n * 8, cudaMemcpyHostToDevice));
        const unsigned int threads = 256;
        const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
        /* bit-reversal kernel test: dst[i] must equal src[bitrev(i, logn)] */
        {
            std::vector<unsigned long long> bi(n, 0);
            for (unsigned long long i = 0; i < n; ++i) bi[(size_t)i] = i;
            CK(cudaMemcpy(dref, bi.data(), n * 8, cudaMemcpyHostToDevice));
            bitrev_kernel<<<blocks, threads>>>(dref, dsave, (unsigned long long)logn, n);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<unsigned long long> bo(n, 0), want(n, 0);
            CK(cudaMemcpy(bo.data(), dsave, n * 8, cudaMemcpyDeviceToHost));
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                want[(size_t)i] = r;
            }
            int bb = 0;
            for (unsigned long long i = 0; i < n; ++i) if (bo[(size_t)i] != want[(size_t)i]) ++bb;
            std::printf("nttcheck: bitrev kernel test bad=%d/%llu  (got", bb,
                        (unsigned long long)n);
            for (int i = 0; i < (int)n && i < 8; ++i) std::printf(" %llu", bo[(size_t)i]);
            std::printf("  want");
            for (int i = 0; i < (int)n && i < 8; ++i) std::printf(" %llu", want[(size_t)i]);
            std::printf(")\n");
            CK(cudaMemcpy(dsave, x.data(), n * 8, cudaMemcpyHostToDevice));
        }
        ntt_forward_fused(dsp, fc, om);
        dft_direct_kernel<<<(unsigned int)n, 1>>>(dsave, dref, n, om);
        gl_twiddle_test_kernel<<<blocks, threads>>>(dtw, n, om);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> sp(n, 0), ref(n, 0), tw(n, 0);
        CK(cudaMemcpy(sp.data(), dsp, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ref.data(), dref, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tw.data(), dtw, n * 8, cudaMemcpyDeviceToHost));
        /* host model of the EXACT steps the device performs (bitrev, DIF desc, bitrev) */
        {
            auto bitrev_idx = [&](unsigned long long i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                return r;
            };
            std::vector<unsigned long long> h1(n, 0), h2(n, 0);
            for (unsigned long long i = 0; i < n; ++i) h1[(size_t)i] = x[(size_t)bitrev_idx(i)];
            h2 = h1;
            for (int s = logn - 1; s >= 0; --s) {
                const unsigned long long half = 1ull << s;
                const unsigned long long stride = 2 * half;
                for (unsigned long long g = 0; g < n / stride; ++g) {
                    for (unsigned long long j = 0; j < half; ++j) {
                        const unsigned long long i0 = g * stride + j, i1 = i0 + half;
                        const unsigned long long twv =
                            gl_pow_host(om, (j * (n / stride)) % n);
                        const unsigned long long u = h2[(size_t)i0], v2 = h2[(size_t)i1];
                        h2[(size_t)i0] = gl_add_host(u, v2);
                        h2[(size_t)i1] = gl_mul_host(gl_sub_host(u, v2), twv);
                    }
                }
            }
            std::vector<unsigned long long> h3(n, 0);
            for (unsigned long long i = 0; i < n; ++i) h3[(size_t)i] = h2[(size_t)bitrev_idx(i)];
            int hb = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                if (sp[(size_t)i] != h3[(size_t)i]) {
                    if (hb < 3) {
                        std::printf("  model mismatch i=%llu: device=%llu model=%llu\n",
                                    (unsigned long long)i, sp[(size_t)i], h3[(size_t)i]);
                    }
                    ++hb;
                }
            }
            std::printf("nttcheck: device vs host-model(bitrev,DIF,bitrev): bad=%d/%llu\n", hb,
                        (unsigned long long)n);
            {
                std::vector<unsigned long long> sq(n, 0);
                CK(cudaMemcpy(sq.data(), dx, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg input buffer after fwd: %llu %llu   out buffer: %llu %llu %llu\n",
                            sq[0], sq[1], sp[0], sp[1], sp[2]);
                std::vector<unsigned long long> sd(n, 0);
                CK(cudaMemcpy(sd.data(), dsp, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg dsp direct: %llu %llu %llu\n", sd[0], sd[1], sd[2]);
                std::vector<unsigned long long> sm(n, 0);
                CK(cudaMemcpy(sm.data(), dsave, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg dsave (input copy): %llu %llu\n", sm[0], sm[1]);
            }
            int hd = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long acc = 0;
                for (unsigned long long j = 0; j < n; ++j) {
                    acc = gl_add_host(acc, gl_mul_host(x[(size_t)j], gl_pow_host(om, (j * i) % n)));
                }
                if (ref[(size_t)i] != acc) ++hd;
            }
            std::printf("nttcheck: oracle check (again) bad=%d\n", hd);
        }
        int bad1 = 0, bad3 = 0;
        /* FIRST: is the on-device DFT oracle itself right?  Compare ref[k] against a host
           sum computed with the (GMP-verified) host arithmetic. */
        {
            int oracle_bad = 0;
            for (unsigned long long kk = 0; kk < n; ++kk) {
                unsigned long long acc = 0;
                for (unsigned long long j = 0; j < n; ++j) {
                    acc = gl_add_host(acc, gl_mul_host(x[(size_t)j],
                                                       gl_pow_host(om, (j * kk) % n)));
                }
                if (acc != ref[(size_t)kk]) {
                    if (oracle_bad < 3) {
                        std::printf("  ORACLE mismatch k=%llu: device=%llu host=%llu\n",
                                    (unsigned long long)kk, ref[(size_t)kk], acc);
                    }
                    ++oracle_bad;
                }
            }
            std::printf("nttcheck: direct-DFT oracle vs host sums: bad=%d/%llu\n", oracle_bad,
                        (unsigned long long)n);
        }
        /* which convention did the forward actually produce?  Count matches under both
           hypotheses: sp[i] == X_bitrev(i) (bit-reversed output) and sp[i] == X_i (natural). */
        {
            int as_bitrev = 0, as_natural = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                if (sp[(size_t)i] == ref[(size_t)r]) ++as_bitrev;
                if (sp[(size_t)i] == ref[(size_t)i]) ++as_natural;
            }
            std::printf("  dbg convention: matches-as-bitrev=%d/%llu matches-as-natural=%d/%llu\n",
                        as_bitrev, (unsigned long long)n, as_natural, (unsigned long long)n);
            if (n <= 16) {
                std::printf("  dbg sp:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", sp[(size_t)i]);
                std::printf("\n  dbg ref:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", ref[(size_t)i]);
                std::printf("\n  dbg x:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", x[(size_t)i]);
                std::printf("\n");
            }
        }
        for (unsigned long long i = 0; i < n; ++i) {
            unsigned long long r = 0, v = i;
            for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
            if (sp[(size_t)i] != ref[(size_t)r]) {
                if (bad1 < 3) {
                    std::printf("  fwd mismatch i=%llu (point %llu): got %llu want %llu\n",
                                (unsigned long long)i, r, sp[(size_t)i], ref[(size_t)r]);
                }
                ++bad1;
            }
            if (tw[(size_t)i] != gl_pow_host(om, i)) ++bad3;
        }
        /* (2) inverse(forward(x)) == x, up to the 1/N factor which the pointwise kernel
           applies (omega^(i*nsc mod N) = omega^i / N) */
        /* stage-by-stage host mirror: find the first stage where the device diverges */
        {
            std::vector<unsigned long long> hs(n, 0);
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                hs[(size_t)i] = x[(size_t)r];
            }
            for (int st = 0; st < logn; ++st) {
                const unsigned long long half = 1ull << st;
                const unsigned long long stride = 2 * half;
                const unsigned long long exp_step = n / (2 * half);
                for (unsigned long long g = 0; g < n / stride; ++g) {
                    for (unsigned long long j = 0; j < half; ++j) {
                        const unsigned long long i0 = g * stride + j;
                        const unsigned long long i1 = i0 + half;
                        const unsigned long long tw = gl_pow_host(om, (j * exp_step) % n);
                        const unsigned long long u = hs[(size_t)i0], v2 = hs[(size_t)i1];
                        hs[(size_t)i0] = gl_add_host(u, v2);
                        hs[(size_t)i1] = gl_mul_host(gl_sub_host(u, v2), tw);
                    }
                }
            }
            int mirror_bad = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                if (sp[(size_t)i] != hs[(size_t)i]) {
                    if (mirror_bad < 3) {
                        std::printf("  mirror mismatch i=%llu: device=%llu hostmirror=%llu\n",
                                    (unsigned long long)i, sp[(size_t)i], hs[(size_t)i]);
                    }
                    ++mirror_bad;
                }
            }
            std::printf("nttcheck: host DIT mirror mismatches=%d/%llu\n", mirror_bad,
                        (unsigned long long)n);
        }
        CK(cudaMemcpy(dx, sp.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dback, sp.data(), n * 8, cudaMemcpyHostToDevice));
        {
            std::vector<unsigned long long> ones(n, 1ull);
            unsigned long long *dones = nullptr;
            CK(cudaMalloc(&dones, n * 8));
            CK(cudaMemcpy(dones, ones.data(), n * 8, cudaMemcpyHostToDevice));
            ntt_inverse_fused(dback, dones, fc, omi, nsc);   /* fused (pointwise by ones) */
            (void)dones;
            CK(cudaGetLastError());
            cudaFree(dones);
        }
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> back(n, 0);
        CK(cudaMemcpy(back.data(), dback, n * 8, cudaMemcpyDeviceToHost));
        int bad2 = 0;
        for (unsigned long long i = 0; i < n; ++i) {
            if (back[(size_t)i] != x[(size_t)i]) {
                if (bad2 < 3) {
                    std::printf("  roundtrip mismatch i=%llu: got %llu want %llu\n",
                                (unsigned long long)i, back[(size_t)i], x[(size_t)i]);
                }
                ++bad2;
            }
        }
        std::printf("nttcheck: logn=%d  forward-vs-directDFT bad=%d/%llu  roundtrip bad=%d/%llu"
                    "  twiddle bad=%d/%llu\n", logn, bad1, (unsigned long long)n, bad2,
                    (unsigned long long)n, bad3, (unsigned long long)n);
        fuse_release(fc);
        cudaFree(dx); cudaFree(dsp); cudaFree(dref); cudaFree(dtw); cudaFree(dback);
        cudaFree(dsave);
        return (bad1 || bad2 || bad3) ? 1 : 0;
    }

    /*
     * fusecheck <log2 N> [device] -- does the STAGE-FUSED transform produce exactly the same
     * numbers as the per-stage (one kernel per stage) implementation it replaced?  This is
     * the cheapest way to localise a fusion bug: the transform's meaning (DIF descending, DIT
     * ascending, bit-reversed spectrum in between, omega vs omega^-1) is unchanged, so any
     * difference is an index/twiddle error in one of the fused passes.
     *   fwd: per-stage forward  vs fused forward                (must be bit-identical)
     *   inv: per-stage inverse  vs fused inverse with B = all ones, which reproduces the
     *        pointwise product and the 1/N scale the fused inverse absorbs  (bit-identical)
     */
    if (mode == "fusecheck") {
        const int logn = (argc > 2) ? std::atoi(argv[2]) : 12;
        const int dev = (argc > 3) ? std::atoi(argv[3]) : 1;
        CK(cudaSetDevice(dev));
        const unsigned long long n = 1ull << logn;
        const unsigned long long om = gl_pow_host(7ull, (GL_P - 1ull) / n);
        const unsigned long long omi = gl_pow_host(om, GL_P - 2ull);
        const unsigned long long nsc = gl_pow_host(n % GL_P, GL_P - 2ull);
        std::vector<unsigned long long> x(n, 0);
        unsigned long long s = 0xC0FFEEull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x[(size_t)i] = s % GL_P;
        }
        unsigned long long *dOld = nullptr, *dNew = nullptr, *dOnes = nullptr, *dScr = nullptr;
        CK(cudaMalloc(&dOld, n * 8));
        CK(cudaMalloc(&dNew, n * 8));
        CK(cudaMalloc(&dOnes, n * 8));
        CK(cudaMalloc(&dScr, n * 8));
        std::vector<unsigned long long> ones(n, 1ull);
        CK(cudaMemcpy(dOnes, ones.data(), n * 8, cudaMemcpyHostToDevice));
        FuseCtx fc;
        fuse_init(fc, n, logn, om, omi);
        std::printf("fusecheck: logn=%d n=%llu tile_t=%d outer=", logn,
                    (unsigned long long)n, fc.t);
        for (int p = 0; p < fc.nms; ++p) std::printf(" radix-%d", 1 << fc.ms[p]);
        std::printf(" -> %d passes per transform, %d total\n", fc.passes_fwd,
                    3 * fc.passes_fwd + 1);
        std::vector<unsigned long long> a(n, 0), b(n, 0);
        int badF = 0, badI = 0;
        unsigned long long firstF = 0, firstI = 0;
        CK(cudaMemcpy(dOld, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dNew, x.data(), n * 8, cudaMemcpyHostToDevice));
        ntt_forward(dOld, n, logn, om, dScr);          /* per-stage reference */
        ntt_forward_fused(dNew, fc, om);               /* fused */
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), dOld, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dNew, n * 8, cudaMemcpyDeviceToHost));
        for (unsigned long long i = 0; i < n; ++i) {
            if (a[(size_t)i] != b[(size_t)i]) { if (!badF) firstF = i; ++badF; }
        }
        /* inverse: write the forward result back into BOTH buffers, then compare */
        CK(cudaMemcpy(dOld, b.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dNew, b.data(), n * 8, cudaMemcpyHostToDevice));
        ntt_inverse(dOld, n, logn, omi, nsc, dScr);    /* per-stage reference */
        ntt_inverse_fused(dNew, dOnes, fc, omi, nsc);  /* fused, pointwise by ones */
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), dOld, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dNew, n * 8, cudaMemcpyDeviceToHost));
        for (unsigned long long i = 0; i < n; ++i) {
            if (a[(size_t)i] != b[(size_t)i]) { if (!badI) firstI = i; ++badI; }
        }
        std::printf("fusecheck: forward per-stage-vs-fused bad=%d/%llu (first=%llu)  "
                    "inverse bad=%d/%llu (first=%llu)\n", badF, (unsigned long long)n,
                    firstF, badI, (unsigned long long)n, firstI);
        fuse_release(fc);
        cudaFree(dOld); cudaFree(dNew); cudaFree(dOnes); cudaFree(dScr);
        return (badF || badI) ? 1 : 0;
    }

    if ((mode == "poly" || mode == "pipedump") && argc > 2) {
        const unsigned long long P = std::strtoull(argv[2], nullptr, 10);
        const int S = (argc > 3) ? std::atoi(argv[3]) : 5153;
        const int dev = (argc > 4) ? std::atoi(argv[4]) : 0;
        const bool verify = (argc > 5) ? (std::atoi(argv[5]) != 0) : true;
        return run_poly(P, S, dev, verify, (mode == "pipedump") ? 1 : 0);
    }
    if (mode == "bench" && argc > 2) {
        const unsigned long long bits = std::strtoull(argv[2], nullptr, 10);
        /* bench: an optional bpw override, then the device (same CLI shape as the cuFFT
           probe's `bench <bits> [chunk_bits] [device]`) */
        int bpw = (argc > 3) ? std::atoi(argv[3]) : 0;
        int dev = (argc > 4) ? std::atoi(argv[4]) : 0;
        return run_bench(bits, bpw, dev);
    }
    std::fprintf(stderr,
                 "usage: ntt_poly_probe poly <P> <S> [device] [verify]\n"
                 "       ntt_poly_probe pipedump <P> <S> [device] [verify]  (digit dump)\n"
                 "       ntt_poly_probe nttcheck <log2N> [device]           (DFT oracle)\n"
                 "       ntt_poly_probe fusecheck <log2N> [device]          (fused vs per-stage)\n"
                 "       ntt_poly_probe bench <bits> [bpw] [device]\n");
    return 2;
}
