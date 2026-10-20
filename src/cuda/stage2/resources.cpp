// Allocation, leases, events and destruction of the Stage2 owners. No curve orchestration.
#include "product.h"
#include "points.h"
#include "remainder.h"
#include "checks.h"
#include "options.h"
#include "point_launch.h"

GfinvBatch::~GfinvBatch()
{
    if(enabled) {
        for(size_t j=0;j<=GROUP;++j)mpz_clear(prefix[j]);
        for(size_t j=0;j<GROUP;++j)mpz_clear(inverse[j]);
        mpz_clears(acc,value,nullptr);
    }
}

size_t S4Ctx::raw_capacity(const unsigned long long *p) const
{
    return p==d_rawA ? d_rawA_cap : p==d_rawB ? d_rawB_cap : 0;
}

void S4Ctx::raw_reserve(size_t a,size_t b)
{
    // Each operand/frontier keeps its own capacity; a larger A never implies a larger B.
    if(a>d_rawA_cap) {
        if(d_rawA)CK(cudaFree(d_rawA));d_rawA=nullptr;d_rawA_cap=0;
        CK(cudaMalloc(&d_rawA,a*8));d_rawA_cap=a;
    }
    if(b>d_rawB_cap) {
        if(d_rawB)CK(cudaFree(d_rawB));d_rawB=nullptr;d_rawB_cap=0;
        CK(cudaMalloc(&d_rawB,b*8));d_rawB_cap=b;
    }
}

size_t S4Ctx::raw_release()
{
    // At the G/fold -> resident-descent boundary, all G-tree roots have
    // been copied into the independent fold owner. cudaFree completes
    // queued default-stream readers before invalidating either pointer.
    const size_t bytes=8*(d_rawA_cap+d_rawB_cap);
    if(d_rawA)CK(cudaFree(d_rawA));
    if(d_rawB)CK(cudaFree(d_rawB));
    d_rawA=d_rawB=nullptr;d_rawA_cap=d_rawB_cap=0;
    return bytes;
}

size_t S4Ctx::output_release()
{
    // Only at phase boundaries after the result has been copied to host
    // or to an independent owner. cudaFree completes queued default-stream
    // scatter/readback/oracle snapshots; delayed CPU checks own host copies.
    const size_t bytes=8*d_out_cap;
    if(d_out)CK(cudaFree(d_out));
    d_out=nullptr;d_out_cap=0;
    return bytes;
}

S4Ctx::~S4Ctx()
{
    if (d_out) cudaFree(d_out);
    if (d_rawA) cudaFree(d_rawA);
    if (d_rawB) cudaFree(d_rawB);
    if (d_packA) cudaFree(d_packA);
    if (d_packB) cudaFree(d_packB);
}

S4Reduce::Shape::~Shape()
{ if (dy) cudaFree(dy); dt_destroy(); }

S4Reduce::~S4Reduce()
{
    s4_oracle_release(*this);       /* snapshots reference shapes and N: drain FIRST */
    for (Shape *s : shapes) delete s;
    if (dn) cudaFree(dn);
    if (dbad) cudaFree(dbad);
    if (h_dbad) cudaFreeHost(h_dbad);
    if (ev_dbad) cudaEventDestroy(ev_dbad);
    mpz_clear(N);
}

S4ResidentOwner::~S4ResidentOwner()
{release();}

void S4ResidentOwner::release()
{for(int i=0;i<2;++i)if(data[i]){CK(cudaFree(data[i]));data[i]=nullptr;words[i]=0;}}

bool S4ResidentOwner::owns(const unsigned long long *p,size_t n)const
{
    return (p==data[0] && n<=words[0]) || (p==data[1] && n<=words[1]);
}

FoldDeviceState::~FoldDeviceState()
{release();}

void FoldDeviceState::release()
{
    memory.release();
    if(map){CK(cudaFree(map));map=nullptr;}
    if(length){CK(cudaFree(length));length=nullptr;}
    if(modulus){CK(cudaFree(modulus));modulus=nullptr;}
    if(digest){CK(cudaFree(digest));digest=nullptr;}
    active=false;
}

S3Workspace::~S3Workspace()
{
    if (dn) cudaFree(dn);
    if (dqx) cudaFree(dqx);
    if (dqz) cudaFree(dqz);
    if (da24) cudaFree(da24);
    if (dmone) cudaFree(dmone);
    if (djs) cudaFree(djs);
    if (dx) cudaFree(dx);
    if (dz) cudaFree(dz);
    if (giant_base) cudaFree(giant_base);
    if (dvals) cudaFree(dvals);
    if (dprod) cudaFree(dprod);
    if (dsegfix) cudaFree(dsegfix);
}

void S3Workspace::init(const LadderCtx &c)
{
    C = &c;
    const size_t nw = c.nw;
    CK(cudaMalloc(&dn, nw * 8));
    CK(cudaMalloc(&dqx, nw * 8));
    CK(cudaMalloc(&dqz, nw * 8));
    CK(cudaMalloc(&da24, nw * 8));
    CK(cudaMalloc(&dmone, nw * 8));
    CK(cudaMemcpy(dn, c.hn.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqx, c.hqx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqz, c.hqz.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da24, c.ha24.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmone, c.hmone.data(), nw * 8, cudaMemcpyHostToDevice));
    bytes = 5 * nw * 8;
}

void S3Workspace::need_segfix(size_t seg)
{
    if(seg<=segfix_cap)return;
    const size_t nw=C->nw;
    std::vector<unsigned long long> powers((seg+1)*nw,0),word(nw);
    mpz_t N,R,p;mpz_inits(N,R,p,nullptr);
    words_to_mpz(N,C->hn.data(),nw);words_to_mpz(R,C->hmone.data(),nw);
    mpz_set_ui(p,1);
    for(size_t m=0;m<=seg;++m) {
        mpz_to_words(word,nw,p);std::copy(word.begin(),word.end(),powers.begin()+m*nw);
        mpz_mul(p,p,R);mpz_mod(p,p,N);
    }
    mpz_clears(N,R,p,nullptr);
    if(dsegfix){CK(cudaFree(dsegfix));bytes-=(segfix_cap+1)*nw*8;}
    dsegfix=nullptr;CK(cudaMalloc(&dsegfix,powers.size()*8));
    CK(cudaMemcpy(dsegfix,powers.data(),powers.size()*8,cudaMemcpyHostToDevice));
    segfix_cap=seg;bytes+=powers.size()*8;
}

void S3Workspace::need_pts(size_t n)
{
    if (n <= pt_cap) return;
    if (djs) cudaFree(djs);
    if (dx) cudaFree(dx);
    if (dz) cudaFree(dz);
    djs = nullptr; dx = nullptr; dz = nullptr;
    const size_t nw = C->nw;
    CK(cudaMalloc(&djs, n * 8));
    CK(cudaMalloc(&dx, n * nw * 8));
    CK(cudaMalloc(&dz, n * nw * 8));
    bytes -= pt_cap * 8 + 2 * pt_cap * nw * 8;
    bytes += n * 8 + 2 * n * nw * 8;
    pt_cap = n;
}

void S3Workspace::need_vals(size_t n)
{
    if (n <= val_cap) return;
    if(s3_compact_products_requested()) {
        if(dvals)CK(cudaFree(dvals));dvals=nullptr;
        CK(cudaMalloc(&dvals,n*C->nw*8));
        bytes-=(size_t)val_cap*C->nw*8;bytes+=n*C->nw*8;val_cap=n;
        return; // Products are unused during descent; reserve at their consumer.
    }
    if (dvals) cudaFree(dvals);
    if (dprod) cudaFree(dprod);
    dvals = nullptr; dprod = nullptr;
    const size_t nw = C->nw;
    CK(cudaMalloc(&dvals, n * nw * 8));
    CK(cudaMalloc(&dprod, n * nw * 8));
    bytes -= (val_cap + prod_cap) * nw * 8;
    bytes += 2 * n * nw * 8;
    val_cap = n;
    prod_cap = n;
}

void S3Workspace::need_products(size_t n)
{
    if(n<=prod_cap)return;
    if(dprod)CK(cudaFree(dprod));dprod=nullptr;
    CK(cudaMalloc(&dprod,n*C->nw*8));
    bytes-=prod_cap*C->nw*8;bytes+=n*C->nw*8;prod_cap=n;
}

ResidentGiant::~ResidentGiant()
{
    if(own){if(x)CK(cudaFree(x));if(z)CK(cudaFree(z));}
    if(segments)CK(cudaFree(segments));if(groups_device)CK(cudaFree(groups_device));
    if(group_fix)CK(cudaFree(group_fix));
}

BabyBuffer::~BabyBuffer()
{if(ptr)CK(cudaFree(ptr));}

bool BabyBuffer::allocate(size_t bytes)
{
    const auto code=cudaMalloc(&ptr,bytes);
    if(code==cudaErrorMemoryAllocation){cudaGetLastError();ptr=nullptr;return false;}
    CK(code);return true;
}

unsigned long long *BabyBuffer::words()
{return static_cast<unsigned long long*>(ptr);}

ScaledFrontierDevice::~ScaledFrontierDevice()
{release();}

void ScaledFrontierDevice::release()
{if(metadata){CK(cudaFree(metadata));metadata=nullptr;}}
