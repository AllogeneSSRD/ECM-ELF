#pragma once

#include "context.h"

struct S4Reduce {
    /* the modulus (shared by every shape of the run) */
    int nw = 0;                          /* words of N */
    int mersenne_bits = 0;               /* verified N=2^S-1, owned by this modulus */
    unsigned long long ninv = 0;         /* -N^-1 mod 2^64 */
    size_t w = 0;                        /* words per coefficient in the tree (W = ceil(S/64)) */
    mpz_t N{};                           /* the actual modulus, for the GMP oracle */
    std::vector<unsigned long long> hn;
    unsigned long long *dn = nullptr;
    unsigned long long *dbad = nullptr;      /* the reduction's slot-canonical counter (device) */
    unsigned long long dbad_host = 0;
    /* the per-SHAPE parameters: L, the limb count and the domain-correction constant all
       depend on slot_bits, i.e. on P, so one entry per (P) the run multiplies at */
    struct Shape {
        unsigned long long slot_bits = 0, slot_stride = 0, slot_words = 0;
        int bpw = 0, L = 0, nlimb = 0;
        double bound_bits = 0.0;
        unsigned long long P = 0;
        std::vector<unsigned long long> hy;
        unsigned long long *dy = nullptr;
        unsigned long long calls = 0, coeffs = 0, canon_bad = 0;
        /* 1 once the window canonicality of this shape has been checked (section 49) */
        unsigned long long canon_checked = 0;
        unsigned long long checked = 0, check_bad = 0, check_first = 0, full_checks = 0;
        double t_hookd2h = 0.0, t_hooksample = 0.0;
        unsigned long long selftest_cases = 0, selftest_bad = 0;
        long long selftest_first = -1;
        double t_reduce = 0.0;
        // Deferred GPU timing, bounded independently for each shape. A full
        // ring recycles only its oldest mark; newer submissions remain live.
        static const int DTRING = 8;
        cudaEvent_t dt_ev[DTRING][2] = {};
        bool dt_used[DTRING] = {};
        unsigned long long dt_sequence[DTRING] = {}, dt_next_sequence = 1;
        double t_reduce_host = 0.0;
        unsigned long long dt_blocks = 0;
        ~Shape();
        void dt_destroy();
        /* resolve every mark the device has already passed; NEVER blocks */
        void dt_reap();
        void dt_resolve(int slot);
        int dt_oldest() const;
        /* take a slot for a new mark: reuse a resolved one, else wait for what is already
           queued.  The wait happens only when the host is a whole ring ahead of the device,
           which is the one place the deferred timing can still cost anything. */
        int dt_slot();
        /* resolve everything, waiting if necessary -- called once, when the run is over and the
           stream has drained anyway, so it never costs anything */
        void dt_flush();
        Shape() = default;
        Shape(const Shape &) = delete;
        Shape &operator=(const Shape &) = delete;
    };
    std::vector<Shape *> shapes;
    unsigned long long reduce_calls = 0, coeffs_total = 0;
    /* the asynchronous `dbad` readback (section 43): one copy in flight at a time, compared on
       the following call, and resolved for good by s4_dbad_resolve() */
    unsigned long long *h_dbad = nullptr;
    cudaEvent_t ev_dbad = nullptr;
    bool dbad_inflight = false;
    Shape *dbad_shape = nullptr;
    unsigned long long dbad_expected = 0;

    S4Reduce();
    ~S4Reduce();
    /* the key must be the WHOLE shape, not slot_bits: slot_bits = 2S + ceil(log2 P) is the same
       for P = 5, 6, 7 and 8, while bpw/slot_words/slot_stride differ between them (found the
       hard way: the lookup returned the P=5 entry for the P=8 multiply) */
    Shape *find(unsigned long long slot_bits, unsigned long long slot_words, int bpw) const;
};

struct S4OracleSlot {
    unsigned long long *digits = nullptr, *reduced = nullptr;
    size_t digcap = 0, redcap = 0;
    cudaEvent_t ready = nullptr;
    S4Reduce::Shape *shape = nullptr;
    unsigned long long slice = 0, k0 = 0, count = 0, out_slots = 0;
};

struct S4OracleStats {
    unsigned long long selected = 0, queued = 0, compared = 0, samples = 0;
    unsigned long long head = 0, tail = 0, ring_waits = 0, fallbacks = 0;
    unsigned long long signature = 1469598103934665603ull, pinned_peak = 0;
    double t_wait = 0.0, t_copy = 0.0, t_gmp = 0.0, t_alloc = 0.0;
    double t_capture = 0.0, t_reap = 0.0, t_drain = 0.0;
    double t_num = 0.0, t_mod = 0.0;
};

unsigned long long *pin_words(unsigned long long **slot, size_t *cap, size_t words);

int s4_reduce_init(S4Reduce &R, const mpz_t N, size_t W,
                          const std::vector<unsigned long long> &hn, unsigned long long ninv);

S4Reduce::Shape *s4_shape_init(S4Reduce &R, unsigned long long P,
                                      unsigned long long slot_bits,
                                      unsigned long long slot_stride,
                                      unsigned long long slot_words, int bpw);

void s4_reduce_hook(void *ctx, const unsigned long long *digits, unsigned long long n,
                           int bpw, unsigned long long slot_words, unsigned long long slot_bits,
                           unsigned long long out_slots, unsigned long long nbatch,
                           unsigned long long *out, unsigned long long w,
                           unsigned long long first, unsigned long long count);

extern S4OracleStats g_oracle;

extern S4Reduce *g_oracle_owner;

extern S4OracleSlot g_oracle_slot[8];
