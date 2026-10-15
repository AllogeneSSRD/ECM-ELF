// Production input pack/zero study. GMP oracle does not use device indexing.
#include "../../src/core/ecm_stage2_logging.h"
#include "../../src/cuda/stage2/ntt_runtime.cuh"
#include "../../src/cuda/stage2/s4_input_pack.cuh"
#include <cuda_profiler_api.h>
using Word=unsigned long long;
struct Buffer {
    Word *p=nullptr;size_t n;
    explicit Buffer(size_t words):n(words){CK(cudaMalloc(&p,8*n));}
    ~Buffer(){CK(cudaFree(p));}
};
__global__ void poison(Word *p){if(!threadIdx.x && !blockIdx.x)p[0]^=1;}
__global__ void compare(const Word *a,const Word *b,Word n,Word *bad) {
    const Word j=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n && a[j]!=b[j])atomicAdd(bad,1ull);
}
__global__ void division_check(const Word *values,Word n,Word *bad) {
    const Word j=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const unsigned x=(unsigned)values[4*j],sw=(unsigned)values[4*j+1];unsigned r=0;
    const unsigned q=stage2_pack::quotient(x,sw,sw==1 ? 0 : (unsigned)((1ull<<32)/sw),r);
    if(q!=values[4*j+2] || r!=values[4*j+3])atomicAdd(bad,1ull);
}
static Word random_word(Word &seed){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
struct Fixture {
    int bits,bpw,W;Word nc,m,sw,N;std::vector<Word> source,offsets;
    Fixture(int bits_,int bpw_,Word count,Word batch,Word extra):bits(bits_),bpw(bpw_),W((bits+63)/64),
        nc(count),m(batch),sw((2ull*bits+bpw-1)/bpw+extra),N(2),offsets(m+4,0) {
        while(N<2*nc*sw+1)N*=2;
        Word seed=26;
        const Word stride=nc*W+7;
        source.resize((m+4)*stride+17);
        for(auto &v:source)v=random_word(seed);
        for(Word s=0;s<m+4;++s) {
            offsets[s]=s*stride+3;
            for(Word i=0;i<nc;++i)for(int w=0;w<W;++w) {
                Word &v=source[offsets[s]+i*W+w];
                if((i+s)%3==0)v=~0ull;
                if((i+s)%3==1)v=0;
                if(w==W-1 && (bits%64))v&=(1ull<<(bits%64))-1;
            }
        }
    }
    std::vector<Word> contiguous()const {
        std::vector<Word> out(std::max(1ull,m*nc*W),0);
        for(Word s=0;s<m;++s)std::copy_n(source.data()+offsets[s+2],nc*W,out.data()+s*nc*W);
        return out;
    }
    std::vector<Word> oracle(bool reverse)const {
        std::vector<Word> out(m*N+32,0xa5a5a5a5a5a5a5a5ull);
        std::fill(out.begin()+16,out.end()-16,0);
        mpz_t x,q,d;mpz_inits(x,q,d,nullptr);
        for(Word s=0;s<m;++s)for(Word i=0;i<nc;++i) {
            const auto *c=source.data()+offsets[s+2]+(reverse ? nc-1-i : i)*W;
            mpz_import(x,W,-1,8,0,0,c);
            for(int k=0;k<(bits+bpw-1)/bpw;++k) {
                mpz_fdiv_q_2exp(q,x,(mp_bitcnt_t)k*bpw);mpz_fdiv_r_2exp(d,q,bpw);
                Word value=0;size_t count=0;mpz_export(&value,&count,-1,8,0,0,d);
                out[16+s*N+i*sw+k]=value;
            }
        }
        mpz_clears(x,q,d,nullptr);return out;
    }
};
static void launch(bool fused,int path,const Fixture &f,Buffer &source,Buffer &meta,Word *dst) {
    if(fused) {
        if(path==0)stage2_pack::launch<false>(source.p,nullptr,f.m,f.nc,f.W,f.bits,f.bpw,f.sw,f.N,dst);
        else if(path==1)stage2_pack::launch<true>(source.p,meta.p+2,f.m,f.nc,f.W,f.bits,f.bpw,f.sw,f.N,dst);
        else stage2_pack::launch<true,true>(source.p,meta.p+2,f.m,f.nc,f.W,f.bits,f.bpw,f.sw,f.N,dst);
    } else {
        CK(cudaMemsetAsync(dst,0,8*f.m*f.N));
        if(f.nc) {
            if(path==0)s4_launch_pack_batch(source.p,f.bits,f.bpw,f.sw,f.N,f.m,f.nc,f.W,dst);
            else if(path==1)s4_pack_gather_kernel<<<(unsigned)((f.m*f.nc+255)/256),256>>>(
                source.p,meta.p+2,f.m,f.nc,f.W,f.bits,f.bpw,f.sw,f.N,dst);
            else s4_pack_gather_reverse_kernel<<<(unsigned)((f.m*f.nc+255)/256),256>>>(
                source.p,meta.p+2,f.m,f.nc,f.W,f.bits,f.bpw,f.sw,f.N,dst);
            CK(cudaGetLastError());
        }
    }
}
static int gate(bool fault) {
    Word cases=0,words=0,bad=0,seed=27;bool injected=false;
    std::vector<Word> div;
    for(unsigned sw=1;sw<=65536;++sw)for(unsigned t=0;t<16;++t) {
        const unsigned j=t==0 ? 0 : t==1 ? sw-1 : t==2 ? sw : t==3 ? 0xffffffffu : (unsigned)random_word(seed);
        div.insert(div.end(),{j,sw,j/sw,j%sw});
    }
    Buffer dv(div.size()),db(1);CK(cudaMemcpy(dv.p,div.data(),8*div.size(),cudaMemcpyHostToDevice));CK(cudaMemset(db.p,0,8));
    division_check<<<(unsigned)((div.size()/4+255)/256),256>>>(dv.p,div.size()/4,db.p);
    CK(cudaMemcpy(&bad,db.p,8,cudaMemcpyDeviceToHost));
    std::printf("pack_zero_division: cases=%llu bad=%llu\n",(Word)div.size()/4,bad);
    for(int bits:{1,63,64,65,129,318,2203,4423,7995,12011,16384})for(int bpw:{1,7,19,31,62})
    for(Word extra:{0ull,3ull})for(int path:{0,1,2}) {
        Fixture f(bits,bpw,cases%7==0 ? 0 : cases%5==0 ? 1 : 5,3,extra);
        if(!stage2_pack::supported(f.N,f.sw,f.nc,bits,bpw,f.W))return 3;
        const auto input=path==0 ? f.contiguous() : f.source;
        Buffer ds(input.size()),dm(f.offsets.size()),out(f.m*f.N+32);
        CK(cudaMemcpy(ds.p,input.data(),8*input.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dm.p,f.offsets.data(),8*f.offsets.size(),cudaMemcpyHostToDevice));
        const auto want=f.oracle(path==2);
        for(int fused:{0,1}) {
            CK(cudaMemset(out.p,0xa5,out.n*8));launch(fused,path,f,ds,dm,out.p+16);
            if(fault && fused && !injected && path==2) {
                poison<<<1,1>>>(out.p+16+f.N-1);injected=true;
            }
            std::vector<Word> got(out.n);CK(cudaMemcpy(got.data(),out.p,out.n*8,cudaMemcpyDeviceToHost));
            for(size_t j=0;j<got.size();++j)bad+=got[j]!=want[j];
            words+=got.size();++cases;
        }
    }
    // Batch launch seam: independently exercise slice65534/65535/65536 plus guards.
    for(int path:{0,1,2}) {
        Fixture f(1,19,1,65537,0);const auto input=path==0 ? f.contiguous() : f.source;
        Buffer ds(input.size()),dm(f.offsets.size()),out(f.m*f.N+32);
        CK(cudaMemcpy(ds.p,input.data(),8*input.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dm.p,f.offsets.data(),8*f.offsets.size(),cudaMemcpyHostToDevice));
        CK(cudaMemset(out.p,0xa5,out.n*8));launch(true,path,f,ds,dm,out.p+16);
        const auto want=f.oracle(path==2);std::vector<Word> got(out.n);
        CK(cudaMemcpy(got.data(),out.p,out.n*8,cudaMemcpyDeviceToHost));
        for(size_t j=0;j<got.size();++j)bad+=got[j]!=want[j];words+=got.size();++cases;
    }
    std::printf("pack_zero_gate: cases=%llu words=%llu bad=%llu fault=%d injected=%d\n",cases,words,bad,(int)fault,(int)injected);
    return bad ? 3 : 0;
}
static int bench(int bits,Word nc,Word m,int path,int repeats,int iterations) {
    NttShape sh;FuseCtx fc;FuseCallGuard guard(fc);int dev=0;CK(cudaGetDevice(&dev));
    if(ntt_shape_plan(nc,bits,dev,nullptr,fc,sh))return 2;
    Fixture f(bits,sh.bpw,nc,m,0);f.sw=sh.slot_words;f.N=sh.N;
    if(f.m*f.N>(1ull<<26))return 2;
    const auto input=path==0 ? f.contiguous() : f.source;
    Buffer ds(input.size()),dm(f.offsets.size()),a(m*f.N+32),b(m*f.N+32),bad(1);
    CK(cudaMemcpy(ds.p,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm.p,f.offsets.data(),f.offsets.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemset(a.p,0xa5,a.n*8));CK(cudaMemset(b.p,0xa5,b.n*8));
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int turn=0;turn<repeats;++turn)for(int pos=0;pos<2;++pos) {
        const int fused=turn%2 ? 1-pos : pos;Word *dst=fused ? b.p+16 : a.p+16;
        CK(cudaEventRecord(start));for(int i=0;i<iterations;++i)launch(fused,path,f,ds,dm,dst);
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
        std::printf("pack_zero_sample: bits=%d P=%llu batch=%llu N=%llu sw=%llu bpw=%d path=%d turn=%d fused=%d seconds=%.9f\n",
            bits,nc,m,f.N,f.sw,f.bpw,path,turn,fused,ms/(1000.0*iterations));
    }
    CK(cudaMemset(bad.p,0,8));compare<<<(unsigned)((a.n+255)/256),256>>>(a.p,b.p,a.n,bad.p);
    Word wrong=0;CK(cudaMemcpy(&wrong,bad.p,8,cudaMemcpyDeviceToHost));
    std::printf("pack_zero_bench_done: words=%llu bad=%llu\n",(Word)a.n,wrong);
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return wrong ? 3 : 0;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;std::setvbuf(stdout,nullptr,_IONBF,0);CK(cudaSetDevice(std::atoi(argv[1])));
    if(!std::strcmp(argv[2],"--check"))return gate(false);
    if(!std::strcmp(argv[2],"--fault"))return gate(true);
    if(!std::strcmp(argv[2],"--bench") && argc==9) {
        const int bits=std::atoi(argv[3]),path=std::atoi(argv[6]),r=std::atoi(argv[7]),i=std::atoi(argv[8]);
        const Word P=std::strtoull(argv[4],nullptr,10),m=std::strtoull(argv[5],nullptr,10);
        if(bits>0 && bits<=32768 && P>0 && P<=155521 && m>0 && m<=65537 && path>=0 && path<=2 && r>0 && r<=31 && i>0 && i<=128)
            return bench(bits,P,m,path,r,i);
    }
    return 2;
}
