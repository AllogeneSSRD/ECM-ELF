#pragma once
#include <cstdint>
#include <cstddef>
#include <gmp.h>
#include "runtime.h"
namespace ecm_cuda_mont_host {
struct ConstructionResults {
    mpz_t *factors;
    int *found;
    ecm_cuda_stage1::FactorSink publish=nullptr;
    void *context=nullptr;
    uint32_t *degenerate_curve=nullptr;
};
uint32_t find_np0(mpz_srcptr);
int verify_size_of_n(mpz_srcptr,size_t container_bits,size_t carry_bits);
uint32_t *allocate_and_set_s_bits(mpz_srcptr,uint64_t *);
uint32_t *set_p_2p(mpz_srcptr,uint32_t,uint32_t,uint32_t,size_t *);
uint32_t *set_p_2p_suyama(mpz_srcptr,uint32_t,uint64_t,uint32_t,size_t *,const ConstructionResults *results=nullptr);
uint32_t *set_p_2p_param2(mpz_srcptr,uint32_t,uint32_t,uint32_t,size_t *,const ConstructionResults *results=nullptr);
int process_results(mpz_t *,int *,mpz_srcptr,const uint32_t *,uint32_t,int,uint64_t,uint32_t,int,int,int,bool construction_results=false,uint32_t *degenerate_curve=nullptr);
}
