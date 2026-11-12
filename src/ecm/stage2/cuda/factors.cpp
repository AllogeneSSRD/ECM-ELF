// Block-product readback and target-ring proper-divisor recording.
#include "factors.h"
#include "points.h"
#include "options.h"
#include "point_launch.h"
#include "../../platform/compute_log.h"

void dev_block_products(
    S3Workspace &W,
    int nvals,
    int per_block,
    std::vector<std::vector<unsigned long long>> &out,
    const unsigned long long *resident_leaves,
    unsigned long long *resident_output)
{
    const LadderCtx &C = *W.C;
    const size_t nw = C.nw;
    const int nb = (nvals + per_block - 1) / per_block;
    out.assign((size_t)nb, std::vector<unsigned long long>(nw, 0ull));
    if (nvals == 0) return;
    if(!resident_output)W.need_products(s3_compact_products_requested()?(size_t)nb:(size_t)nvals);
    auto *result=resident_output?resident_output:W.dprod;
    const auto start=now_s();
    S2G_DISPATCH((int)nw, s2g_launch_block_prod, (int)nw, nvals, per_block,
                 resident_leaves?resident_leaves:W.dvals,W.dn,C.ninv,result);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const auto ready=now_s();
    std::vector<unsigned long long> flat((size_t)nb * nw, 0ull);
    CK(cudaMemcpy(flat.data(),result,flat.size()*8,cudaMemcpyDeviceToHost));
    stage2_log::print(stage2_log::debug,"stage2_block_product_gpu: resident=%d kernel_seconds=%.6f read_seconds=%.6f bytes=%llu\n",
        int(resident_leaves!=nullptr),ready-start,now_s()-ready,(unsigned long long)flat.size()*8);
    for (int b = 0; b < nb; ++b)
        for (size_t i = 0; i < nw; ++i) out[(size_t)b][i] = flat[(size_t)b * nw + i];
    ++W.prod_launches;
}

void s3_record(Stage2Tail &out,const mpz_t f,unsigned long long prime,const mpz_t N,bool count_hit)
{
    if(mpz_cmp_ui(f,1)>0)out.nonunit=true;
    if (mpz_cmp_ui(f, 1) <= 0 || mpz_cmp(f, N) == 0) return;
    mpz_t tq;
    mpz_init(tq);
    mpz_mod(tq, N, f);
    if (mpz_cmp_ui(tq, 0) != 0) ++out.bad_factors;
    mpz_clear(tq);
    char *s = mpz_get_str(nullptr, 10, f);
    bool seen = false;
    for (const std::string &v : out.factors)
        if (v == s) seen = true;
    if (!seen) out.factors.push_back(s);
    void (*ff)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &ff);
    ff(s, std::strlen(s) + 1);
    if (prime) out.hit_primes.push_back(prime);
    if (count_hit) ++out.hits;
}
