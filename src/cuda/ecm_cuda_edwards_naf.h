#pragma once
#include "ecm_cuda_edwards.h"

struct EcmEdwardsNafMetrics {
    uint32_t columns=0,additions=0,container_bits=0,tpi=0,launches=0;
    uint64_t dictionary_bytes=0;
    float dictionary_gpu_ms=0,loop_gpu_ms=0;
};
// done counts MSB-first NAF columns (possibly one more than binary scalar
// length). Rebuilding the dictionary never changes the supplied accumulator.
// Windows 2..12; doubling 4 (square-based) or 5 (2XY). Both publish full XYZT.
int ecm_cuda_edwards_stage1_naf(const mpz_t n,const mpz_t scalar,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk_bits,int device,
        EcmEdwardsProgress progress,void *context,std::string &error,
        unsigned window=6,int doubling=4,EcmEdwardsNafMetrics *metrics=nullptr);
