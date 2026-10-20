#pragma once

#include "points.h"

#include "tree.h"

#include "remainder.h"

#include "factors.h"

struct BatchedRun {
    ecm_stage2::timing::Boundaries phase_boundaries;
    DeviceGLeafStats device_leaf;
    FoldFlatStats fold_flat;
    FoldDeviceStats fold_device;
    Stage2Tail tail;
    unsigned long long giant_points = 0, num_poly_g = 0, loops = 0, P = 0;
    unsigned long long giant_first=1,giant_last=0;
    unsigned long long descent_divmods = 0, leaf_values = 0, apply_blocks = 0, block_per = 0;
    unsigned long long small_primes = 0;
    unsigned long long hit_blocks = 0, named_searches = 0, candidates_tested = 0;
    /* the naming loop's own accounting: a HIT LEAF is a leaf whose value shares a factor with N
       (there can be many of them -- the batched engine evaluates H at the BABY points, and one
       stage-2 prime can be covered by several baby points, so `hit_leaves` is NOT comparable
       with the CPU reference's `hits`, whose leaves are the GIANT points).  `unnamed` counts hit
       leaves whose scan found no confirming prime: their factor is recorded from the leaf gcd
       itself, exactly as the reference's name_culprit does. */
    unsigned long long hit_leaves = 0, unnamed = 0;
    double t_scan = 0.0, t_ladder = 0.0;
    /* the giant-point chain accounting: how many chunks went through the chain and how many
       LADDER points the seeds cost (a chain needs two seeds per block, so this is the part of the
       phase that is not one-xADD-per-point -- section 31) */
    unsigned long long giant_chain_chunks = 0, giant_seed_points = 0;
    /* objective 3: giant points whose Z shares a factor with N (the hit made explicit) */
    unsigned long long giant_degenerate = 0;
    unsigned long long unnamed_hits = 0, cand_lists = 0;
    /* device-side bookkeeping, so the breakdown line reports measured work, not a guess */
    unsigned long long ladder_calls = 0, ladder_points = 0, prod_launches = 0;
    unsigned long long arena_fuse_builds = 0, arena_fuse_reuse = 0, arena_buf_builds = 0,
                       arena_buf_reuse = 0, arena_overflow = 0;
    double arena_mb = 0.0;
    unsigned long long ntt_calls = 0;
    /* slice S5: the device descent's own accounting (its leaf values live on the device) */
    unsigned long long s5_divmods = 0, s5_generic = 0, s5_linear = 0, s5_copies = 0, s5_zeros = 0,
                       s5_ntt_launches = 0, s5_forest_bytes = 0, s5_frontier_bytes = 0;
    double s5_t_ntt = 0.0, s5_t_pack = 0.0, s5_t_generic = 0.0, s5_t_copy = 0.0;
    bool s5_readback = false;
    double t_giant = 0.0, t_gtrees = 0.0, t_fold = 0.0, t_descent = 0.0, t_inv = 0.0,
           t_accum = 0.0, t_name = 0.0;
    /* ---- THE BOOKS MUST BALANCE (section 41) -------------------------------------------
       `batched_progress: t=` is the SUM of the three loop timers, not a wall clock, so a run
       can look fully accounted while the loop is really much longer: at the production shape
       those three add up to 308.6 s and the curve's wall is 555.3 s, and the missing ~186 s
       was invisible for several rounds because nothing timed the loop ITSELF.  These four
       numbers close the loop: pre + loop_wall + post must equal the caller's `elapsed`, and
       `gleaves` is the one part of the loop body that no phase timer owned (the host GMP
       affine conversion of every giant point, section 31.4/38). */
    double t_pre_loop = 0.0, t_loop_wall = 0.0, t_post_loop = 0.0, t_gleaves = 0.0;
    /* the one pass over H that removes the projective scale (section 42) */
    double t_gscale = 0.0, t_post_prepare = 0.0;
    double t_post_end = 0.0, t_return_finalize = 0.0;
    PlainScaleStats gscale;
    /* ---- THE PROJECTIVE BOOKKEEPING MUST BE EXACTLY SELF-CONSISTENT ----------------------
       The factor set CANNOT validate Gamma: any invertible Gamma gives the same gcds, so a
       doubled or missing factor would still "pass".  The invariant that can be checked is
       combinatorial and exact: every projective leaf must be covered by exactly one accumulated
       segment product, i.e. gamma_points == proj_points at the end of the run.  A missed segment
       makes it too small, a double count makes it too large, and either way it is a hard error
       rather than a silently different scale. */
    unsigned long long proj_points = 0, proj_gamma_points = 0, proj_segments = 0, proj_fallbacks = 0;
    /* ... and the four parts of `gleaves`, because "the host affine conversion" is 36% of the
       whole curve and each part has a different fix (section 41) */
    double t_gin = 0.0, t_ginv = 0.0, t_gmul = 0.0, t_gout = 0.0;
    bool dbg_progress = false;              /* one phase line per G-tree batch (long shapes) */
};

BatchedRun run_batched(PolyLayer &L, const LadderCtx &C, const Stage2Params &SP,
                              const std::vector<std::vector<unsigned long long>> &Ft,
                              const std::vector<size_t> &Fdeg, size_t Fpad,
                              const SmallPrimeBabyCache *baby_cache=nullptr,
                              const ecm_stage2::GiantRange *requested_range=nullptr);

int run_real(const char *n_str, bool n_is_hex, unsigned long long sigma,
                    unsigned long long B1, unsigned long long B2, unsigned long long D_in,
                    bool choose_d,
                    const char *saved_qx_hex = nullptr, Stage2Tail *saved_result = nullptr,
                    unsigned carrier_exponent = 0,
                    const char *saved_a24_hex = nullptr);
