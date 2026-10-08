// Compile against a frozen isolated arithmetic source closure; no production edits.
#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
using BatchWord=unsigned long long;
static constexpr BatchWord BATCH_PAD=0x12345678ull;

__global__ void batch_input(BatchWord*a,BatchWord*b,size_t n,size_t stride,size_t count)
{
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<stride*count) {
        size_t j=i%stride,s=i/stride,t=s%3;
        a[i]=j>=n ? BATCH_PAD : j==0 ? GL_P-1-t : j==1 ? (1ull<<63)+t : j==2 ? GL_P-2-2*t : 0;
        b[i]=j>=n ? BATCH_PAD : j==0 ? 3+t : j==1 ? 5+2*t : j==2 ? 7+t : 0;
    }
}
__global__ void batch_compare(const BatchWord*a,const BatchWord*ref,size_t n,size_t stride,
    size_t count,int kind,BatchWord*bad)
{
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<stride*count) {
        size_t j=i%stride,t=(i/stride)%3;
        BatchWord want=j>=n ? BATCH_PAD : kind==2 ? (j<5 ? ref[t*5+j] : 0) : ref[(t*2+kind)*n+j];
        if(a[i]!=want || (j<n && a[i]>=GL_P))atomicAdd(bad,1ull);
    }
}
__global__ void batch_poison(BatchWord*a,size_t index){a[index]^=1;}

static BatchWord batch_export(mpz_t z)
{BatchWord v=0;size_t count=0;mpz_export(&v,&count,-1,8,0,0,z);return v;}
static void batch_import(mpz_t z,BatchWord v){mpz_import(z,1,-1,8,0,0,&v);}
static void batch_reference(int k,std::vector<BatchWord>&spectra,std::vector<BatchWord>&product,
    BatchWord&om,BatchWord&omi,BatchWord&scale)
{
    size_t n=1ull<<k;spectra.resize(6*n);product.resize(15);
    mpz_t p,g,w,step,x,y,z;mpz_inits(p,g,w,step,x,y,z,nullptr);batch_import(p,GL_P);mpz_set_ui(g,7);
    // Windows unsigned long is 32-bit; the root exponent needs an mpz operand.
    batch_import(z,(GL_P-1)/n);mpz_powm(w,g,z,p);om=batch_export(w);mpz_invert(z,w,p);omi=batch_export(z);
    mpz_set_ui(z,(unsigned long)n);mpz_invert(z,z,p);scale=batch_export(z);
    for(size_t t=0;t<3;++t) {
        BatchWord aa[3]={GL_P-1-t,(1ull<<63)+t,GL_P-2-2*t},bb[3]={3+t,5+2*t,7+t};
        mpz_set_ui(step,1);
        for(size_t f=0;f<n;++f) {
            size_t rev=0,v=f;for(int bit=0;bit<k;++bit){rev=(rev<<1)|(v&1);v>>=1;}
            for(int kind=0;kind<2;++kind) {
                mpz_set_ui(z,0);mpz_set_ui(g,1);
                for(int c=0;c<3;++c) {
                    batch_import(x,kind ? bb[c] : aa[c]);mpz_addmul(z,x,g);
                    mpz_mul(g,g,step);mpz_mod(g,g,p);
                }
                mpz_mod(z,z,p);spectra[(t*2+kind)*n+rev]=batch_export(z);
            }
            mpz_mul(step,step,w);mpz_mod(step,step,p);
        }
        for(int c=0;c<5;++c) {
            mpz_set_ui(z,0);
            for(int i=0;i<3;++i)if(c-i>=0 && c-i<3) {
                batch_import(x,aa[i]);batch_import(y,bb[c-i]);mpz_addmul(z,x,y);
            }
            mpz_mod(z,z,p);product[t*5+c]=batch_export(z);
        }
    }
    mpz_clears(p,g,w,step,x,y,z,nullptr);
}

static BatchWord batch_case(size_t nb,size_t padding,int repeats,bool fault)
{
    const int k=11;const size_t n=1ull<<k,stride=n+padding,total=stride*nb,bytes=8*total;
    BatchWord om,omi,scale;std::vector<BatchWord>spectra,product;
    batch_reference(k,spectra,product,om,omi,scale);
    BatchWord *a=nullptr,*b=nullptr,*sr=nullptr,*pr=nullptr,*bad=nullptr;
    CK(cudaMalloc(&a,bytes));CK(cudaMalloc(&b,bytes));CK(cudaMalloc(&sr,8*spectra.size()));
    CK(cudaMalloc(&pr,8*product.size()));CK(cudaMalloc(&bad,8));
    CK(cudaMemcpy(sr,spectra.data(),8*spectra.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pr,product.data(),8*product.size(),cudaMemcpyHostToDevice));
    fuse_fixture_env("NTT_FUSE_T","11");fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");
    fuse_fixture_env("NTT_FUSE_COOP_OUTER","2");fuse_fixture_env("NTT_OUTER_NARROW","0");
    NttArena arena;arena.device=1;FuseCtx fc;ntt_arena_fuse(&arena,n,k,om,omi,fc);
    if(fc.t!=11 || fc.nms || !fc.warp_tail)std::exit(3);
    dim3 grid(1,(unsigned)nb);const int smem=8*n;
    for(int inv=0;inv<2;++inv) {
        const void*fn=inv ? (const void*)tile_kernel<true,true> : (const void*)tile_kernel<false,true>;
        cudaFuncAttributes attr{};int blocks=0;CK(cudaFuncGetAttributes(&attr,fn));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,FUSE_TILE_THREADS,smem));
        std::printf("ntt_batch_resource: mask=%d inverse=%d regs=%d local=%zu shared=%zu dynamic=%d blocks=%d\n",
            NTT_GL_ADD_SUB_MASK,inv,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,smem,blocks);
    }
    auto forward=[&](BatchWord*d){tile_kernel<false,true><<<grid,FUSE_TILE_THREADS,smem>>>(d,nullptr,n,k,fc.t,fc.tblF,0,stride);CK(cudaGetLastError());};
    auto inverse=[&](){tile_kernel<true,true><<<grid,FUSE_TILE_THREADS,smem>>>(a,b,n,k,fc.t,fc.tblI,scale,stride);CK(cudaGetLastError());};
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));BatchWord wrong=0;
    for(int phase=0;phase<3;++phase)for(int repeat=0;repeat<repeats;++repeat) {
        batch_input<<<(unsigned)((total+255)/256),256>>>(a,b,n,stride,nb);CK(cudaGetLastError());
        if(phase==1){forward(a);forward(b);}
        CK(cudaEventRecord(start));
        if(phase==0)forward(a);
        if(phase==1)inverse();
        if(phase==2){forward(a);forward(b);inverse();}
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
        if(fault && phase==2)batch_poison<<<1,1>>>(a,(nb-1)*stride+4);
        CK(cudaMemset(bad,0,8));
        batch_compare<<<(unsigned)((total+255)/256),256>>>(a,phase==0?sr:pr,n,stride,nb,phase==0?0:2,bad);
        CK(cudaGetLastError());BatchWord errors=0;CK(cudaMemcpy(&errors,bad,8,cudaMemcpyDeviceToHost));wrong+=errors;
        std::printf("ntt_batch_sample: mask=%d phase=%d repeat=%d N=%zu batch=%zu stride=%zu words=%zu seconds=%.9f bad=%llu\n",
            NTT_GL_ADD_SUB_MASK,phase,repeat,n,nb,stride,total,ms/1000.0,errors);
    }
    arena.release();CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(sr));CK(cudaFree(pr));CK(cudaFree(bad));
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    std::printf("ntt_batch_case: mask=%d batch=%zu padding=%zu repeats=%d payload=%zu bad=%llu live=%llu\n",
        NTT_GL_ADD_SUB_MASK,nb,padding,repeats,2*bytes+8*spectra.size()+8*product.size()+8,wrong,g_fuse_base.live_bytes);
    if(g_fuse_base.live_bytes)std::exit(3);return wrong;
}
int main(int argc,char**argv)
{
    CK(cudaSetDevice(1));std::printf("ntt_addsub_mask: value=%d\n",NTT_GL_ADD_SUB_MASK);
    if(argc!=2)return 2;BatchWord bad=0;
    if(!std::strcmp(argv[1],"--gate"))for(size_t nb:{1,3,990})for(size_t pad:{0,17})bad+=batch_case(nb,pad,1,false);
    else if(!std::strcmp(argv[1],"--fault"))bad=batch_case(990,17,1,true);
    else if(!std::strcmp(argv[1],"--timing"))bad=batch_case(990,0,25,false);
    else return 2;
    std::printf("ntt_batch_done: mask=%d bad=%llu live=%llu\n",NTT_GL_ADD_SUB_MASK,bad,g_fuse_base.live_bytes);
    return bad ? 3 : 0;
}
