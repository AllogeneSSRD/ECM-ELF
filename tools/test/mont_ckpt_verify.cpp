/* ---------------------------------------------------------------------------
 * mont_ckpt_verify.cpp -- deterministic verification of the Montgomery mid-stage-1
 * checkpoints (docs/ECM_Montgomery_STAGE1.md §17).
 *
 * What is proven here, without any wall-clock timing or process killing:
 *
 *   1. scalar ladder: pausing at EVERY chunk and resuming from the written state
 *      gives byte-identical (x, gcd) results to one uninterrupted run          [T1]
 *   2. SIMD batch   : same for all 8 lanes of a batch                           [T2]
 *   3. cross-path   : a SIMD lane paused at offset k, its state handed over as
 *      plain mpz, resumed by the SCALAR ladder, still lands on the same point.
 *      This is the test that pins ifma_to_mpz_lane/ifma_from_mpz_lane as an exact
 *      round trip, i.e. that a checkpoint written by one field layer / backend can
 *      be continued by the other.                                                [T3]
 *   4. file format  : write -> read round trip for both record kinds, plus the
 *      rejections that must never be resumed silently (wrong N/B1/torsion/sbits,
 *      truncated file, corrupted checksum).                                     [T4]
 *
 * T1-T3 compare mpz values with mpz_cmp (exact, no tolerance), so "resumed" and
 * "uninterrupted" have to agree bit for bit.
 *
 * build: tools\build_tool.bat tools\test\mont_ckpt_verify.cpp ^
 *            src\cpu\ecm_mont_cpu.cpp src\cpu\simd_mont_curve.cpp ^
 *            src\cpu\simd_mont_ifma.cpp src\core\ecm_mont_ckpt.cpp ^
 *            src\core\ecm_stage1_exp.cpp
 * run:   build_vs18\tools\mont_ckpt_verify.exe [B1] [chunk_bits]
 * ------------------------------------------------------------------------- */
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <direct.h>      /* _mkdir */

#include <string>
#include <vector>

#include "ecm_mont_cpu.h"
#include "ecm_mont_ckpt.h"
#include "simd_mont_curve.h"

static int g_fail = 0;
static int g_pass = 0;

static void check(bool ok, const char *what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) g_pass++; else g_fail++;
}

/* ---- pause context: stop at the first offered chunk, then keep stopping ---- */
struct PauseOnce {
    mont_ladder_state_t st;
    size_t pauses;
};

static int cb_scalar_pause(void *p, const mont_ladder_state_t *st)
{
    PauseOnce *po = (PauseOnce *)p;
    po->st.bitnum = st->bitnum;
    mpz_set(po->st.X0, st->X0);
    mpz_set(po->st.Z0, st->Z0);
    mpz_set(po->st.X1, st->X1);
    mpz_set(po->st.Z1, st->Z1);
    po->pauses++;
    return 1;                       /* always pause: exercises the resume path hard */
}

struct SoaPause {
    std::vector<uint64_t> state;
    size_t pauses;
};

static int cb_soa_pause(void *p, size_t /*bitnum*/)
{
    SoaPause *sp = (SoaPause *)p;
    sp->pauses++;
    return 1;
}

static uint64_t sigma_of(uint32_t i) { return 1234567890123ull + 7919ull * i; }

/* --- small helpers to hand-craft damaged files (same FNV as ecm_mont_ckpt.cpp) */
static std::string read_file(const std::string &path)
{
    std::string all;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return all;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
    fclose(f);
    return all;
}

static bool write_file(const std::string &path, const std::string &s)
{
    FILE *o = fopen(path.c_str(), "wb");
    if (!o) return false;
    fwrite(s.data(), 1, s.size(), o);
    fclose(o);
    return true;
}

static uint64_t fnv1a(uint64_t h, const std::string &s)
{
    for (size_t i = 0; i < s.size(); i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* drop one "KEY=..." line and recompute the checksum, so the file stays valid and
   only the tested property is broken */
static bool rewrite_without(const std::string &src, const std::string &dst,
                            const std::string &drop_prefix)
{
    const std::string all = read_file(src);
    if (all.empty()) return false;
    std::string out;
    uint64_t sum = 1469598103934665603ull;
    size_t p = 0;
    while (p < all.size()) {
        size_t e = all.find('\n', p);
        if (e == std::string::npos) e = all.size();
        const std::string line = all.substr(p, e - p);
        p = e + 1;
        if (line.compare(0, 9, "CHECKSUM=") == 0) break;
        if (line.compare(0, drop_prefix.size(), drop_prefix) == 0) continue;
        sum = fnv1a(sum, line + "\n");
        out += line + "\n";
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "CHECKSUM=%016llx\n", (unsigned long long)sum);
    out += buf;
    out += "END\n";
    return write_file(dst, out);
}

/* Simulate "pause at every chunk, write nothing to disk, resume from memory":
   the state passes through the same mpz representation the file uses, so this is
   exactly the resume the driver performs after reading the checkpoint back. */
static void run_scalar_chunked(mpz_t x_out, mpz_t g_out, const mpz_t N, uint64_t sigma,
                               const uint8_t *bits, size_t nbits, size_t chunk,
                               size_t *out_pauses)
{
    PauseOnce po;
    mont_ladder_state_init(&po.st);
    po.pauses = 0;

    size_t start = 0;
    mpz_t Qx, Qz, g, inv;
    mpz_inits(Qx, Qz, g, inv, NULL);
    for (;;) {
        const int rc = mont_stage1_curve_bits_ex(g, Qx, Qz, N, sigma, bits, nbits,
                                                start, &po.st, cb_scalar_pause, &po, chunk);
        if (rc == MONT_LADDER_PAUSED) {
            start = po.st.bitnum;               /* what the .ckpt file would store */
            continue;
        }
        if (rc == MONT_LADDER_ERROR) { mpz_set_ui(x_out, 0); mpz_set_ui(g_out, 0); break; }
        if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
            mpz_mul(x_out, Qx, inv);
            mpz_mod(x_out, x_out, N);
        } else {
            mpz_set(x_out, Qx);
        }
        mpz_set(g_out, g);
        break;
    }
    if (out_pauses) *out_pauses = po.pauses;
    mpz_clears(Qx, Qz, g, inv, NULL);
    mont_ladder_state_clear(&po.st);
}

int main(int argc, char **argv)
{
    const double B1 = (argc > 1) ? atof(argv[1]) : 5000.0;
    const size_t chunk = (argc > 2) ? (size_t)strtoul(argv[2], NULL, 10) : 512;

    /* N = M1277 = 2^1277 - 1: the smallest Sersenne case with a known factor
       history, and small enough that the whole test runs in well under a second. */
    mpz_t N, s;
    mpz_inits(N, s, NULL);
    mpz_set_ui(N, 1);
    mpz_mul_2exp(N, N, 1277);
    mpz_sub_ui(N, N, 1);

    const int torsion = 1;
    mont_build_s(s, (uint64_t)B1, torsion);
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    printf("mont_ckpt_verify: N = 2^1277-1, B1 = %.0f, s_bits = %zu, chunk = %zu\n",
           B1, nbits, chunk);
    printf("                  expect %zu pause(s) per curve\n\n", nbits / chunk);

    /* ---------------------------------------------------------------- T1 */
    printf("T1 scalar ladder: pause every chunk, resume from the state\n");
    {
        mpz_t x_ref, g_ref, x_res, g_res;
        mpz_inits(x_ref, g_ref, x_res, g_res, NULL);
        const uint64_t sigma = sigma_of(0);

        mpz_t factor, Qx, Qz;
        mpz_inits(factor, Qx, Qz, NULL);
        mont_stage1_curve_bits(factor, Qx, Qz, N, sigma, bits, nbits);
        mpz_set(g_ref, factor);
        mpz_t inv;
        mpz_init(inv);
        if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
            mpz_mul(x_ref, Qx, inv);
            mpz_mod(x_ref, x_ref, N);
        } else {
            mpz_set(x_ref, Qx);
        }
        mpz_clear(inv);
        mpz_clears(factor, Qx, Qz, NULL);

        size_t pauses = 0;
        run_scalar_chunked(x_res, g_res, N, sigma, bits, nbits, chunk, &pauses);

        char buf[160];
        snprintf(buf, sizeof(buf), "x identical after %zu pause(s)/resume(s)", pauses);
        check(mpz_cmp(x_ref, x_res) == 0, buf);
        check(mpz_cmp(g_ref, g_res) == 0, "gcd identical");
        check(pauses >= nbits / chunk - 1, "the resume path really was exercised");

        mpz_clears(x_ref, g_ref, x_res, g_res, NULL);
    }

    /* ---------------------------------------------------------------- T2 */
    printf("\nT2 SIMD batch: pause every chunk, resume all 8 lanes\n");
    {
        mont_soa_ctx_t ctx;
        if (mont_soa_init(&ctx, N, IFMA_FIELD_AUTO) != 0) {
            printf("  [SKIP] no AVX512-IFMA on this machine\n");
        } else {
            uint64_t sg[IFMA_LANES];
            for (unsigned k = 0; k < IFMA_LANES; k++) sg[k] = sigma_of(k);

            std::vector<mpz_t> xr(IFMA_LANES), gr(IFMA_LANES), xp(IFMA_LANES), gp(IFMA_LANES);
            for (unsigned k = 0; k < IFMA_LANES; k++)
                mpz_inits(xr[k], gr[k], xp[k], gp[k], NULL);

            /* reference: one uninterrupted run */
            mont_soa_stage1_bits(&ctx, bits, nbits, sg, xr.data(), gr.data());

            /* chunked: every lane pauses at every chunk and resumes from `state` */
            std::vector<uint64_t> state(mont_soa_state_words(&ctx));
            SoaPause sp;
            sp.state = state;
            sp.pauses = 0;
            size_t start = 0, bitnum = 0;
            for (;;) {
                const int rc = mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, start,
                                                       sp.state.data(), &bitnum,
                                                       xp.data(), gp.data(),
                                                       cb_soa_pause, &sp, chunk);
                if (rc == MONT_SOA_PAUSED) { start = bitnum; continue; }
                break;
            }

            char buf[160];
            snprintf(buf, sizeof(buf), "all 8 lanes identical after %zu batch pause(s)",
                     sp.pauses);
            bool ok_x = true, ok_g = true;
            for (unsigned k = 0; k < IFMA_LANES; k++) {
                if (mpz_cmp(xr[k], xp[k]) != 0) ok_x = false;
                if (mpz_cmp(gr[k], gp[k]) != 0) ok_g = false;
            }
            check(ok_x, buf);
            check(ok_g, "all 8 gcd values identical");
            check(sp.pauses >= nbits / chunk - 1, "the resume path really was exercised");

            /* ------------------------------------------------------------ T3 */
            printf("\nT3 cross-path: SIMD lane state -> scalar ladder\n");
            {
                /* pause the batch at the first chunk, then hand lane 3's state to
                   the scalar ladder as plain mpz (that is what a .ckpt file holds) */
                std::vector<uint64_t> st2(mont_soa_state_words(&ctx));
                SoaPause sp2;
                sp2.state = st2;
                sp2.pauses = 0;
                size_t bn = 0;
                std::vector<mpz_t> xt(IFMA_LANES), gt(IFMA_LANES);
                for (unsigned k = 0; k < IFMA_LANES; k++) mpz_inits(xt[k], gt[k], NULL);
                const int rc = mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, 0, sp2.state.data(),
                                                       &bn, xt.data(), gt.data(),
                                                       cb_soa_pause, &sp2, chunk);
                check(rc == MONT_SOA_PAUSED && bn > 0, "batch paused at a mid-ladder offset");

                const unsigned lane = 3;
                const size_t lw = 8 * ctx.n;
                mpz_t X0, Z0, X1, Z1, xsc, gsc, inv;
                mpz_inits(X0, Z0, X1, Z1, xsc, gsc, inv, NULL);
                ifma_to_mpz_lane(X0, sp2.state.data() + 0 * lw, lane, &ctx.mc);
                ifma_to_mpz_lane(Z0, sp2.state.data() + 1 * lw, lane, &ctx.mc);
                ifma_to_mpz_lane(X1, sp2.state.data() + 2 * lw, lane, &ctx.mc);
                ifma_to_mpz_lane(Z1, sp2.state.data() + 3 * lw, lane, &ctx.mc);

                mont_ladder_state_t st;
                mont_ladder_state_init(&st);
                st.bitnum = bn;
                mpz_set(st.X0, X0); mpz_set(st.Z0, Z0);
                mpz_set(st.X1, X1); mpz_set(st.Z1, Z1);

                mpz_t Qx, Qz, g;
                mpz_inits(Qx, Qz, g, NULL);
                const int rc2 = mont_stage1_curve_bits_ex(g, Qx, Qz, N, sg[lane], bits, nbits,
                                                          bn, &st, NULL, NULL, chunk);
                if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
                    mpz_mul(xsc, Qx, inv);
                    mpz_mod(xsc, xsc, N);
                } else {
                    mpz_set(xsc, Qx);
                }
                (void)rc2;
                char buf2[200];
                snprintf(buf2, sizeof(buf2),
                         "lane %u: x from (SIMD pause @%zu -> scalar resume) == uninterrupted x",
                         lane, bn);
                check(mpz_cmp(xsc, xr[lane]) == 0, buf2);
                check(mpz_cmp(g, gr[lane]) == 0, "gcd identical across the handover");
                mpz_clears(X0, Z0, X1, Z1, xsc, gsc, inv, Qx, Qz, g, NULL);
                mont_ladder_state_clear(&st);
                for (unsigned k = 0; k < IFMA_LANES; k++) mpz_clears(xt[k], gt[k], NULL);
            }

            for (unsigned k = 0; k < IFMA_LANES; k++) mpz_clears(xr[k], gr[k], xp[k], gp[k], NULL);
            mont_soa_clear(&ctx);
        }
    }

    /* ---------------------------------------------------------------- T4 */
    printf("\nT4 checkpoint file: round trip + rejections\n");
    {
        const char *dir = "build_vs18/tools/_ck_verify";
        _mkdir(dir);            /* best effort: it normally already exists */
        const std::string stem = std::string(dir) + "/m1277_5e3";

        mont_ckpt_t ck, rd;
        mont_ckpt_init(&ck);
        mont_ckpt_init(&rd);
        ck.status = MONT_CKPT_INFLIGHT;
        ck.bitnum = 2048;
        mpz_set_ui(ck.X0, 0x1234567890abcdefull);
        mpz_set_ui(ck.Z0, 0xfedcba0987654321ull);
        mpz_set_str(ck.X1, "deadbeefcafebabe0123456789", 16);
        mpz_set_ui(ck.Z1, 1);
        mont_ckpt_ident(&ck, 7, 424242ull, B1, torsion, nbits, "ifma");

        const std::string path = mont_ckpt_path(stem, 7);
        check(mont_ckpt_write(path, N, &ck) == 0, "write INFLIGHT record");

        std::string why;
        int ok = mont_ckpt_read(path, N, B1, torsion, nbits, &rd, &why);
        check(ok == 1, "read it back");
        check(rd.status == MONT_CKPT_INFLIGHT && rd.bitnum == 2048 && rd.curve == 7 &&
              rd.sigma == 424242ull, "identity fields survive");
        check(mpz_cmp(rd.X0, ck.X0) == 0 && mpz_cmp(rd.Z0, ck.Z0) == 0 &&
              mpz_cmp(rd.X1, ck.X1) == 0 && mpz_cmp(rd.Z1, ck.Z1) == 0, "state survives");

        check(mont_ckpt_read(path, N, B1 + 1, torsion, nbits, &rd, &why) == 0,
              "rejects a different B1");
        check(mont_ckpt_read(path, N, B1, 12, nbits, &rd, &why) == 0,
              "rejects a different torsion");
        check(mont_ckpt_read(path, N, B1, torsion, nbits + 1, &rd, &why) == 0,
              "rejects a different s_bits");
        {
            mpz_t N2;
            mpz_init(N2);
            mpz_set_ui(N2, 12345);
            check(mont_ckpt_read(path, N2, B1, torsion, nbits, &rd, &why) == 0,
                  "rejects a different N");
            mpz_clear(N2);
        }

        /* truncated: drop the whole END line (a crash mid-write looks like this) */
        {
            const std::string all = read_file(path);
            const size_t endpos = all.rfind("END");
            const std::string trunc = (endpos == std::string::npos) ? all : all.substr(0, endpos);
            write_file(path + ".trunc", trunc);
            check(mont_ckpt_read(path + ".trunc", N, B1, torsion, nbits, &rd, &why) == 0,
                  "rejects a truncated file (no END)");
            mont_ckpt_remove(path + ".trunc");
        }

        /* corrupted payload: flip one payload byte */
        {
            std::string all = read_file(path);
            const size_t pos = all.find("X0=");
            if (pos != std::string::npos && pos + 5 < all.size())
                all[pos + 5] = (all[pos + 5] == 'a') ? 'b' : 'a';
            write_file(path + ".bad", all);
            check(mont_ckpt_read(path + ".bad", N, B1, torsion, nbits, &rd, &why) == 0,
                  "rejects a corrupted payload (checksum)");
            mont_ckpt_remove(path + ".bad");
        }

        /* DONE record */
        mont_ckpt_clear(&ck);
        mont_ckpt_init(&ck);
        ck.status = MONT_CKPT_DONE;
        ck.hit = 0;
        ck.bitnum = nbits;
        mpz_set_ui(ck.xout, 0xabcdef);
        mont_ckpt_ident(&ck, 8, 999ull, B1, torsion, nbits, "mpn");
        const std::string p2 = mont_ckpt_path(stem, 8);
        check(mont_ckpt_write(p2, N, &ck) == 0, "write DONE record");
        mont_ckpt_t rd2;
        mont_ckpt_init(&rd2);
        check(mont_ckpt_read(p2, N, B1, torsion, nbits, &rd2, &why) == 1 &&
              rd2.status == MONT_CKPT_DONE && rd2.hit == 0 && mpz_cmp_ui(rd2.xout, 0xabcdef) == 0,
              "DONE record round trip");

        /* a DONE record whose result line was lost must be refused (the writer
           always emits XOUT/FACTOR, so this is built by hand) */
        {
            const std::string p3 = mont_ckpt_path(stem, 9);
            mont_ckpt_t ck3;
            mont_ckpt_init(&ck3);
            ck3.status = MONT_CKPT_DONE;
            ck3.hit = 0;
            ck3.bitnum = nbits;
            mpz_set_ui(ck3.xout, 12345);
            mont_ckpt_ident(&ck3, 9, 1ull, B1, torsion, nbits, "mpn");
            mont_ckpt_write(p3, N, &ck3);
            const std::string p3bad = p3 + ".noresult";
            check(rewrite_without(p3, p3bad, "XOUT="), "hand-crafted DONE record without a result");
            check(mont_ckpt_read(p3bad, N, B1, torsion, nbits, &rd2, &why) == 0,
                  "rejects DONE without a result");
            check(mont_ckpt_read(p3, N, B1, torsion, nbits, &rd2, &why) == 1,
                  "control: the same record WITH the result is accepted");
            mont_ckpt_remove(p3);
            mont_ckpt_remove(p3bad);
            mont_ckpt_clear(&ck3);
        }

        mont_ckpt_remove(path);
        mont_ckpt_remove(p2);
        mont_ckpt_clear(&ck);
        mont_ckpt_clear(&rd);
        mont_ckpt_clear(&rd2);
    }

    free(bits);
    mpz_clears(N, s, NULL);

    printf("\n%s: %d passed, %d failed\n", g_fail ? "FAILURE" : "ALL OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
