#pragma once

#include "s4.h"

void s4_oracle_reset(S4Reduce &R);

void s4_oracle_report();

void s4_dbad_resolve(S4Reduce &R, bool wait = false);

void s4_mersenne_check();

void s4_div_check(const std::vector<unsigned long long> &actual);

void s4_gmp_assemble(mpz_t v, const unsigned long long *digits,
                           unsigned long long nslots, int bpw);

void s4_gmp_assemble_slow(mpz_t v, const unsigned long long *digits,
                                unsigned long long nslots, int bpw);

void s4_gmp_pack_check();

void s4_gmp_reduce(mpz_t out, const unsigned long long *digits, unsigned long long nslots,
                          int bpw, const mpz_t N, bool timed = false);

int s4_reduce_selftest(S4Reduce &R, S4Reduce::Shape *S);

void s4_compare_snapshot(S4Reduce &R, S4Reduce::Shape *S,
    unsigned long long s, unsigned long long k0, unsigned long long cnt,
    unsigned long long out_slots, const unsigned long long *digbuf,
    const unsigned long long *redbuf);

bool s4_oracle_one(S4Reduce &R, bool wait);

void s4_oracle_reap(S4Reduce &R, bool wait);

void s4_oracle_drain(S4Reduce &R);

void s4_oracle_release(S4Reduce &R);

void s4_oracle_block_ready();

void s4_check_reduced(S4Reduce &R, S4Reduce::Shape *S, const unsigned long long *ddig,
                             unsigned long long n, unsigned long long out_slots,
                             unsigned long long nbatch, const unsigned long long *d_out,
                             long long sample_limit, unsigned long long first,
                             unsigned long long count);

int mont_selftest(const std::vector<unsigned long long> &hn, size_t nw,
                         unsigned long long ninv, const mpz_t N, const mpz_t R);

int xadd_selftest(const std::vector<unsigned long long> &hn,size_t nw,
                          unsigned long long ninv,const mpz_t N,const mpz_t R);
