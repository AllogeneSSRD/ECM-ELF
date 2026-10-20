// Block-product readback and target-ring proper-divisor recording.
#include "factors.h"
#include "points.h"
#include "options.h"
#include "point_launch.h"

void dev_block_products(
    S3Workspace &W,
    int nvals,
    int per_block,
    std::vector<std::vector<unsigned long long>> &out)
{
    const LadderCtx &C = *W.C;
    const size_t nw = C.nw;
    const int nb = (nvals + per_block - 1) / per_block;
    out.assign((size_t)nb, std::vector<unsigned long long>(nw, 0ull));
    if (nvals == 0) return;
    W.need_products(s3_compact_products_requested()?(size_t)nb:(size_t)nvals);
    S2G_DISPATCH((int)nw, s2g_launch_block_prod, (int)nw, nvals, per_block, W.dvals, W.dn,
                 C.ninv, W.dprod);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> flat((size_t)nb * nw, 0ull);
    CK(cudaMemcpy(flat.data(), W.dprod, flat.size() * 8, cudaMemcpyDeviceToHost));
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
