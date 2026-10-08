#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
#include "../bench/ntt_goldilocks_addsub.cuh"
using Word=unsigned long long;

template<int OP,bool PTX>
__global__ void addsub_primitive(const Word*a,const Word*b,Word*out,size_t n)
{
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        if constexpr(OP==0)out[i]=PTX ? gl_add_canonical_ptx(a[i],b[i]) : gl_add_dev(a[i],b[i]);
        else out[i]=PTX ? gl_sub_canonical_ptx(a[i],b[i]) : gl_sub_dev(a[i],b[i]);
    }
}
template<int MASK>
__global__ void addsub_chain(const Word*a,const Word*b,Word*out,size_t n)
{
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        Word v=a[i],w=b[i];
        for(int j=0;j<64;++j) {
            v=(MASK&2) ? gl_add_canonical_ptx(v,w) : gl_add_dev(v,w);
            w=(MASK&1) ? gl_sub_canonical_ptx(v,w) : gl_sub_dev(v,w);
        }
        out[i]=v;
    }
}
static void put(mpz_t z,Word v){mpz_import(z,1,-1,8,0,0,&v);}
static Word get(mpz_t z){Word v=0;size_t n=0;mpz_export(&v,&n,-1,8,0,0,z);return v;}
int main(int argc,char**argv)
{
    CK(cudaSetDevice(1));bool fault=argc>1 && !std::strcmp(argv[1],"--fault");
    std::vector<Word>a,b;Word seed=0x173817381ull;
    const Word edges[]={0,1,2,GL_P-1,GL_P-2,GL_P/2,0xffffffffull,0x100000000ull,
        0xffffffff00000000ull,0xfffffffe00000000ull,1ull<<63,(1ull<<63)-1};
    auto append=[&](Word x,Word y){if(x<GL_P && y<GL_P){a.push_back(x);b.push_back(y);}};
    for(auto x:edges)for(auto y:edges)append(x,y);
    for(auto x:edges)for(int delta:{-1,0,1}) {
        append(x,x+delta);append(x,GL_P-x+delta);
    }
    for(int i=0;i<1000000;++i) {
        seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;Word x=seed%GL_P;
        seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;append(x,seed%GL_P);
    }
    Word *da=nullptr,*db=nullptr,*out=nullptr;size_t bytes=a.size()*8;
    CK(cudaMalloc(&da,bytes));CK(cudaMalloc(&db,bytes));CK(cudaMalloc(&out,bytes));
    CK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    std::vector<Word>want(a.size()),got(a.size());Word bad=0;
    mpz_t p,x,y,z;mpz_inits(p,x,y,z,nullptr);put(p,GL_P);
    for(int op:{0,1}) {
        for(size_t i=0;i<a.size();++i) {
            put(x,a[i]);put(y,b[i]);if(op==0)mpz_add(z,x,y);else mpz_sub(z,x,y);
            mpz_mod(z,z,p);want[i]=get(z);
        }
        for(int mode:{0,1}) {
            const auto grid=(unsigned)((a.size()+255)/256);
            if(op==0 && mode==0)addsub_primitive<0,false><<<grid,256>>>(da,db,out,a.size());
            if(op==0 && mode==1)addsub_primitive<0,true><<<grid,256>>>(da,db,out,a.size());
            if(op==1 && mode==0)addsub_primitive<1,false><<<grid,256>>>(da,db,out,a.size());
            if(op==1 && mode==1)addsub_primitive<1,true><<<grid,256>>>(da,db,out,a.size());
            CK(cudaMemcpy(got.data(),out,bytes,cudaMemcpyDeviceToHost));
            if(fault && mode==1 && op==1)got[0]^=1;
            Word wrong=0;for(size_t i=0;i<a.size();++i)wrong+=got[i]!=want[i] || got[i]>=GL_P;
            bad+=wrong;std::printf("addsub_gate: op=%d ptx=%d words=%zu bad=%llu\n",op,mode,a.size(),wrong);
        }
    }
    for(size_t i=0;i<256;++i) {
        put(x,a[i]);put(y,b[i]);
        for(int j=0;j<64;++j){mpz_add(z,x,y);mpz_mod(x,z,p);mpz_sub(z,x,y);mpz_mod(y,z,p);}
        want[i]=get(x);
    }
    for(int mask:{0,1,2,3}) {
        if(mask==0)addsub_chain<0><<<1,256>>>(da,db,out,256);
        if(mask==1)addsub_chain<1><<<1,256>>>(da,db,out,256);
        if(mask==2)addsub_chain<2><<<1,256>>>(da,db,out,256);
        if(mask==3)addsub_chain<3><<<1,256>>>(da,db,out,256);
        CK(cudaMemcpy(got.data(),out,256*8,cudaMemcpyDeviceToHost));Word wrong=0;
        for(int i=0;i<256;++i)wrong+=got[i]!=want[i] || got[i]>=GL_P;
        bad+=wrong;std::printf("addsub_chain_gate: mask=%d words=256 iterations=64 bad=%llu\n",mask,wrong);
    }
    mpz_clears(p,x,y,z,nullptr);CK(cudaFree(da));CK(cudaFree(db));CK(cudaFree(out));
    std::printf("addsub_done: fault=%d bad=%llu payload_peak=%zu live=0\n",(int)fault,bad,3*bytes);
    return bad ? 3 : 0;
}
