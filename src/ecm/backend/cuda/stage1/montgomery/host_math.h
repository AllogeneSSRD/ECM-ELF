#pragma once
#include <cstdint>
#include <cstddef>
#include <gmp.h>
namespace ecm_cuda_mont_host {
uint32_t find_np0(mpz_srcptr);
uint32_t *allocate_and_set_s_bits(mpz_srcptr,uint64_t *);
uint32_t *set_p_2p(mpz_srcptr,uint32_t,uint32_t,uint32_t,size_t *);
uint32_t *set_p_2p_suyama(mpz_srcptr,uint32_t,uint64_t,uint32_t,size_t *);
uint32_t *set_p_2p_param2(mpz_srcptr,uint32_t,uint32_t,uint32_t,size_t *);
int process_results(mpz_t *,int *,mpz_srcptr,const uint32_t *,uint32_t,int,uint64_t,uint32_t,int,int,int);
}
