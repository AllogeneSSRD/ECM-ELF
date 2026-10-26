#pragma once

#include "context.h"

struct S3Workspace;

struct Stage2Tail {
    EcmStage2Metrics metrics;
    bool nonunit=false;
    unsigned long long hits = 0, bad_factors = 0;
    /* hits whose prime was counted but not named (the naming budget stopped the scan): hits and
       bad_factors stay EXACTLY comparable with the CPU reference either way, only the
       hit_primes list is short, and that is reported rather than hidden */
    unsigned long long unnamed_hits = 0;
    std::vector<std::string> factors;
    std::vector<unsigned long long> hit_primes;
};

void dev_block_products(S3Workspace &W, int nvals, int per_block,
                               std::vector<std::vector<unsigned long long>> &out);

void s3_record(Stage2Tail &out, const mpz_t f, unsigned long long prime, const mpz_t N,
                      bool count_hit = true);
