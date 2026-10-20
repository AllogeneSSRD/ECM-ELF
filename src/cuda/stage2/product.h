#pragma once

#include "context.h"

#include "s4.h"

struct MulCost {
    long S = 0;
    unsigned long long muls[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long bits[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long coeff[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long hcnt[64] = {}, hbits[64] = {};
    unsigned long long max_m1 = 0, max_m2 = 0, max_bits = 0;
    unsigned long long cat_mul_max[BC_NCAT] = {0, 0, 0, 0, 0};   /* widest operand per category */

    void add(int cat, size_t m1, size_t m2);
    unsigned long long tot_muls() const;
    unsigned long long tot_bits() const;
};

struct S4Ctx {
    S4Reduce *red = nullptr;
    unsigned long long *d_out = nullptr;
    size_t d_out_cap = 0;
    unsigned long long launches = 0, muls = 0, groups = 0, level_calls = 0;
    unsigned long long chunk_plan_calls=0, chunk_plan_changed=0;
    unsigned long long chunk_plan_chunks=0, chunk_plan_legacy_chunks=0;
    unsigned long long chunk_plan_request_peak=0, chunk_plan_subset_peak=0;
    unsigned long long chunk_plan_over_budget=0;
    ecm_stage2::TreeWorkspacePlan phase_requests[BC_NCAT];
    ecm_stage2::RequestSignature request_signatures[BC_NCAT],request_signature;
    ecm_stage2::RequestSignature s4_program_signature;
    unsigned long long phase_big_live_peak[BC_NCAT]{}, phase_subset_peak[BC_NCAT]{};
    unsigned long long phase_ntt_peak[BC_NCAT]{};
    long long sample_limit = 4096;      /* a full GMP check below this many coefficients */
    bool selftested = false;
    /* OBJECTIVE 4 (section 33): the DEVICE-side operand packing.  Before this, every batched
       multiply packed its operands in a host loop (36.8 us per polynomial multiplication,
       measured) and then uploaded the PACKED operands (20.2 us) -- 33% of the whole NTT budget.
       Packing on the device needs the raw coefficients on the device (P*W words per slice, which
       is ~9x SMALLER than the packed form at S=5261) and a digit buffer per operand. */
    unsigned long long *d_rawA = nullptr, *d_rawB = nullptr, *d_packA = nullptr, *d_packB = nullptr;
    size_t d_rawA_cap = 0, d_rawB_cap = 0, d_pack_cap = 0;
    size_t raw_capacity(const unsigned long long *p) const;
    void raw_reserve(size_t a,size_t b);
    size_t raw_release();
    size_t output_release();
    double t_h2d_raw = 0.0, t_packdev = 0.0;
    unsigned long long raw_words = 0, pack_launches = 0;
    /* All S4 device-packed calls, including F-tree/inverse before the main-loop timers. */
    unsigned long long input_direct = 0, input_copied = 0, input_d2d_bytes = 0;
    unsigned long long input_avoided_bytes = 0, packed_peak_bytes = 0, temp_peak_bytes = 0;
    double t_input_copy_host = 0.0;
    ~S4Ctx();
};

struct PolyLayer : ecm_stage2::ModulusContext {
    int device = 1;
    /* slice S3: the persistent multiply arena (buffers + fusion plan + per-pass twiddle
       tables built once per SHAPE instead of once per call).  nullptr = the original
       per-call path, unchanged. */
    NttArena *arena = nullptr;
    /* slice S4: the batched + device-reduced multiply.  nullptr = the per-call path. */
    S4Ctx *s4 = nullptr;
    /* slice S3: the batched structure's multiplication accounting.  `cat` is the CURRENT
       category: every multiply that does not name one explicitly (the ones inside
       cp_divmod/cp_inv_series, whose caller knows what phase it is in) is charged to it.
       -1 = do not account (the S1/S2 paths, which keep their own counters). */
    MulCost cost;
    int cat = -1;
    /* statistics */
    unsigned long long ntt_calls = 0, muls = 0, slot_checks = 0;
    ecm_stage2::NttLengthCounts ntt_lengths;
    unsigned long long short_batch_changed=0;
    void count_ntt(unsigned long long n,unsigned long long pairs,unsigned long long chunks);
    void print_ntt_lengths()const;
    /* slice S4: batched multiplies are ONE launch for `nbatch` poly multiplies, so the launch
       count is reported separately from the multiply count (which keeps the S3 numbers
       comparable) */
    unsigned long long ntt_launches = 0;
    unsigned long long max_ntt_words = 0, max_ntt_coeffs = 0, max_slot_bits = 0;
    /* the BINDING exactness shape: the call whose L*(2^bpw-1)^2 is the largest, i.e. the one
       that comes closest to p -- that is the shape the bound has to be re-proved for */
    unsigned long long bind_P = 0, bind_L = 0, bind_slot_bits = 0, bind_slot_words = 0;
    int bind_bpw = 0;
    double bind_bound_bits = 0.0;
    double ntt_seconds = 0.0;
    /* slice S3: where the time inside the multiply goes.  t_fwd/t_inv/t_slot are the NTT
       implementation's own measurements (two forward transforms, the inverse pass with the
       pointwise product and the 1/N scale, then carry + slot assembly + readback); whatever
       is left of ntt_seconds is the host side of the call (packing into bpw digits, the
       memcpys, the exact-coefficient extraction and the two-extraction cross-check). */
    double t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* slice S4: the per-call fixed cost, split (see NttMulStats).  host_side = everything in
       ntt_seconds that is not t_fwd/t_inv/t_slot; these four are the measured parts of it. */
    double t_setup = 0.0, t_pack = 0.0, t_maxc = 0.0, t_h2d = 0.0, t_d2h = 0.0, t_ext = 0.0,
           t_xchk = 0.0;
    /* OBJECTIVE 4 (section 32): the copy of the reduced PRODUCT back to the host that
       poly_mul_batch_modN does once per chunk -- the volume and the time of it, measured here
       because no timer in the probe covers it (the probe's host-side timers belong to the host
       implementation, which the batched path does not use). */
    double t_d2h_coeff = 0.0;
    unsigned long long d2h_coeff_words = 0;
    /* OBJECTIVE 4: the three phases of the BATCHED entry point that its timers never covered:
       the host-side packing of the operands, the max-coefficient scan over the packed batch, and
       the upload of the packed operands (section 32). */
    double t_hpack = 0.0, t_scan = 0.0, t_h2d_batch = 0.0;
    /* the carry-convergence assert's own pass (section 34) */
    double t_check = 0.0;
    /* the three parts of t_check (section 41): the reset launch, the kernel's own GPU time and
       the blocking 16-byte readback that drains the pipeline */
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    double t_plan = 0.0, t_opcopy = 0.0;
    /* the slot assembly + its D2H, skipped when a reduction hook is installed (section 35) */
    double t_hout = 0.0;

    PolyLayer() = default;
    ~PolyLayer() = default;
    PolyLayer(const PolyLayer &) = delete;
    PolyLayer &operator=(const PolyLayer &) = delete;
};

struct OutputWindowStats {
    unsigned long long calls=0, source_coeffs=0, reduced_coeffs=0, returned_coeffs=0;
    unsigned long long skipped_coeffs=0, d2h_words=0, device_peak_bytes=0, pinned_peak_bytes=0;
};

struct FinalReadbackStats {
    unsigned long long calls=0, copied_words=0, avoided_words=0, host_peak_bytes=0;
    double t_copy=0.0;
};

struct ChunkOutputStats {
    unsigned long long calls=0, reused_calls=0, whole_calls=0, multi_chunk_reused=0;
    unsigned long long reused_chunks=0, legacy_calls=0, grows=0;
    unsigned long long request_peak_bytes=0, whole_peak_bytes=0, retained_peak_bytes=0;
};

struct S4ResidentOwner {
    unsigned long long *data[2]={nullptr,nullptr};
    size_t words[2]={0,0};
    S4ResidentOwner()=default;
    S4ResidentOwner(const S4ResidentOwner&)=delete;
    S4ResidentOwner& operator=(const S4ResidentOwner&)=delete;
    ~S4ResidentOwner();
    void release();
    bool owns(const unsigned long long *p,size_t n)const;
};

struct S4DeviceBatch {
    const unsigned long long *src=nullptr,*meta=nullptr;
    unsigned long long *dst=nullptr;
    const unsigned long long *host_meta=nullptr;
    size_t src_words=0,dst_words=0,nbatch=0;
    const S4ResidentOwner *owner=nullptr; // nullptr = the original exclusive raw frontier lease
    bool reverse_b=false; // scaled sibling input stays in ordinary host order
};

struct S4GatherPack {
    const S4DeviceBatch *batch=nullptr;S4Ctx *C=nullptr;size_t s0=0,ma=0,mb=0;
    unsigned long long P=0,N=0,sw=0;int bits=0,bpw=0,W=0;
};

struct S4InputPack {
    S4Ctx *C = nullptr;
    unsigned long long P = 0, N = 0, slot_words = 0;
    int S = 0, bpw = 0, W = 0;
};

struct FlatInputStats {
    unsigned long long calls = 0, borrowed = 0, padded = 0, alias_clones = 0;
    unsigned long long copy_bytes = 0, zero_bytes = 0, avoided_copy_bytes = 0, avoided_zero_bytes = 0;
    unsigned long long temp_peak_bytes = 0, control_peak_bytes = 0;
    double t_prepare = 0.0;
};

bool poly_is_one(const unsigned long long *p, size_t W);

std::vector<unsigned long long> poly_mul_modN(PolyLayer &L,
                                                     const std::vector<unsigned long long> &a,
                                                     size_t da,
                                                     const std::vector<unsigned long long> &b,
                                                     size_t db,
                                                     int cat = -1);

void poly_mul_batch_modN(PolyLayer &L,
                                const unsigned long long *wa, const unsigned long long *wb,
                                size_t ma, size_t mb, size_t nbatch,
                                std::vector<unsigned long long> &out, int cat = -1,
                                NttMulStats *st_out = nullptr,
                                size_t first = 0, size_t count = (size_t)-1,
                                const S4DeviceBatch *resident = nullptr);

void flat_mul_batch(PolyLayer &L, const std::vector<unsigned long long> &A, size_t ma,
                           const std::vector<unsigned long long> &B, size_t mb, size_t nbatch,
                           std::vector<unsigned long long> &out, int cat,
                           size_t first = 0, size_t count = (size_t)-1);

extern FlatInputStats g_flat_input;

extern ChunkOutputStats g_chunk_output;

extern FinalReadbackStats g_final_readback;

extern OutputWindowStats g_output_window;

int s4_gather_final_input(void *ctx,const NttShape &sh,unsigned long long m,
    unsigned long long *a,unsigned long long *b);

int s4_pack_final_input(void *ctx, const NttShape &sh, unsigned long long nbatch,
                              unsigned long long *dA, unsigned long long *dB);

struct GDeviceStats {
    unsigned long long trees=0, fallbacks=0, levels=0, groups=0, pairs=0, copies=0;
    unsigned long long leaf_words=0, root_words=0, resident_words=0, trace_words=0;
    unsigned long long metadata_words=0, metadata_peak_bytes=0, raw_peak_bytes=0;
    unsigned long long logical_frontier_peak_bytes=0, host_staging_peak_bytes=0;
    unsigned long long checked_nodes=0, checked_words=0;
};

extern GDeviceStats g_gdevice;
