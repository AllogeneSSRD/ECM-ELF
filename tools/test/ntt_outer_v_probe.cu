// Exercise the actual development launcher, with both V variants in one binary.
#define main original_coop_probe_main
#include "ntt_coop_outer_probe.cu"
#undef main

template<int M,bool INV,int V> static void v_resources()
{
    cudaFuncAttributes a{};int blocks=0;
    CK(cudaFuncGetAttributes(&a,outer_coop_kernel<M,INV,V>));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,outer_coop_kernel<M,INV,V>,256,0));
    std::printf("ntt_v_resources: M=%d inverse=%d V=%d regs=%d local=%zu shared=%zu max_blocks=%d\n",
        M,(int)INV,V,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,blocks);
}

// Independent dense DIF oracle: GMP modular multiply, natural input,
// bit-reversed spectrum. No device arithmetic or GPU-generated reference.
static void dense_gmp(std::vector<unsigned long long>& a,int k,unsigned long long om)
{
    mpz_t p,x,y,z;mpz_inits(p,x,y,z,nullptr);const auto prime=GL_P;
    mpz_import(p,1,-1,8,0,0,&prime);
    auto mul=[&](unsigned long long u,unsigned long long v) {
        mpz_import(x,1,-1,8,0,0,&u);mpz_import(y,1,-1,8,0,0,&v);
        mpz_mul(z,x,y);mpz_mod(z,z,p);unsigned long long w=0;size_t count=0;
        mpz_export(&w,&count,-1,8,0,0,z);return w;
    };
    const size_t n=1ull<<k;
    // Root powers also come from GMP, rather than the device reduction helper.
    std::vector<unsigned long long> powers(n/2);powers[0]=1;
    for(size_t j=1;j<powers.size();++j)powers[j]=mul(powers[j-1],om);
    for(size_t len=n;len>=2;len>>=1)for(size_t base=0;base<n;base+=len)
        for(size_t j=0;j<len/2;++j) {
            const auto u=a[base+j],v=a[base+j+len/2];
            mpz_import(x,1,-1,8,0,0,&u);mpz_import(y,1,-1,8,0,0,&v);
            mpz_add(z,x,y);mpz_mod(z,z,p);size_t count=0;unsigned long long sum=0;
            mpz_export(&sum,&count,-1,8,0,0,z);
            mpz_sub(z,x,y);mpz_mod(z,z,p);unsigned long long diff=0;
            mpz_export(&diff,&count,-1,8,0,0,z);
            a[base+j]=sum;a[base+j+len/2]=mul(diff,powers[j*(n/len)]);
        }
    mpz_clears(p,x,y,z,nullptr);
}

static void dense_gate()
{
    unsigned long long bad=0,words=0,cases=0,seed=0x981723541ull;
    for(int k:{13,15}) {
        const size_t n=1ull<<k,stride=n+17,nbatch=3;
        const auto om=gl_pow_host(7,(GL_P-1)/n),omi=gl_pow_host(om,GL_P-2),scale=gl_pow_host(n,GL_P-2);
        std::vector<unsigned long long> input(stride*nbatch,0x12345678ull),spectrum=input,ones(stride*nbatch,1),got(input.size());
        for(size_t s=0;s<nbatch;++s) {
            std::vector<unsigned long long> slice(n);
            for(size_t j=0;j<n;++j) {
                seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
                slice[j]=j%17==0 ? GL_P-1 : j%19==0 ? 0 : seed%GL_P;
                input[s*stride+j]=slice[j];
            }
            dense_gmp(slice,k,om);
            for(size_t j=0;j<n;++j)spectrum[s*stride+j]=slice[j];
        }
        unsigned long long *a=nullptr,*b=nullptr;CK(cudaMalloc(&a,input.size()*8));CK(cudaMalloc(&b,input.size()*8));
        CK(cudaMemcpy(b,ones.data(),ones.size()*8,cudaMemcpyHostToDevice));
        fuse_fixture_env("NTT_FUSE_T","5");fuse_fixture_env("NTT_FUSE_COOP_OUTER","1");
        fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");
        NttArena arena;
        for(int m:{7,8}) {
            fuse_fixture_env("NTT_FUSE_T",std::to_string(k-m).c_str());
            fuse_fixture_env("NTT_FUSE_COOP_M",std::to_string(m).c_str());
            // One shared cached plan, repeated switches exercise runtime dispatch.
            for(int mask:{0,3,1,2,0,3}) {
                fuse_fixture_env("NTT_OUTER_NARROW",std::to_string(mask).c_str());
                FuseCtx fc;ntt_arena_fuse(&arena,n,k,om,omi,fc);
                if(!fc.tables_cached || fc.nms!=1 || fc.ms[0]!=m)std::exit(3);
                CK(cudaMemcpy(a,input.data(),input.size()*8,cudaMemcpyHostToDevice));
                launch_outer_coop<false>(m,a,n,0,fc.passF[0],fc.radF[0],nbatch,stride);
                dim3 grid((unsigned)(n>>fc.t),(unsigned)nbatch);
                const int shared=(1<<fc.t)*8;
                tile_kernel<false,true><<<grid,FUSE_TILE_THREADS,shared>>>(a,nullptr,n,k,fc.t,fc.tblF,0,stride);
                CK(cudaMemcpy(got.data(),a,got.size()*8,cudaMemcpyDeviceToHost));
                if(std::getenv("NTT_V_DENSE_BAD") && mask==3)got[0]^=1;
                for(size_t j=0;j<got.size();++j)bad+=got[j]!=spectrum[j];words+=got.size();
                tile_kernel<true,true><<<grid,FUSE_TILE_THREADS,shared>>>(a,b,n,k,fc.t,fc.tblI,scale,stride);
                launch_outer_coop<true>(m,a,n,fc.t,fc.passI[0],fc.radI[0],nbatch,stride);
                CK(cudaMemcpy(got.data(),a,got.size()*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<got.size();++j)bad+=got[j]!=input[j];words+=got.size();++cases;
            }
        }
        arena.release();CK(cudaFree(a));CK(cudaFree(b));
    }
    std::printf("ntt_v_dense: cases=%llu words=%llu bad=%llu live=%llu\n",cases,words,bad,g_fuse_base.live_bytes);
    if(bad || g_fuse_base.live_bytes)std::exit(3);
}

static void v_bench(int device,int k)
{
    if(k<23 || k>27)std::exit(2);
    const auto n=1ull<<k,om=gl_pow_host(7,(GL_P-1)/n),omi=gl_pow_host(om,GL_P-2),scale=gl_pow_host(n,GL_P-2);
    unsigned long long *a=nullptr,*b=nullptr,*want=nullptr,*bad=nullptr;
    CK(cudaMalloc(&a,8*n));CK(cudaMalloc(&b,8*n));CK(cudaMalloc(&want,40));CK(cudaMalloc(&bad,8));
    const unsigned long long aa[3]={GL_P-1,1ull<<63,GL_P-2},bb[3]={3,5,7};unsigned long long expected[5]={};
    mpz_t p,x,y,z;mpz_inits(p,x,y,z,nullptr);const auto q=GL_P;mpz_import(p,1,-1,8,0,0,&q);
    for(int c=0;c<5;++c) {
        mpz_set_ui(z,0);for(int i=0;i<3;++i)if(c-i>=0 && c-i<3) {
            mpz_import(x,1,-1,8,0,0,&aa[i]);mpz_import(y,1,-1,8,0,0,&bb[c-i]);mpz_addmul(z,x,y);
        }
        mpz_mod(z,z,p);size_t count=0;mpz_export(expected+c,&count,-1,8,0,0,z);
    }
    mpz_clears(p,x,y,z,nullptr);CK(cudaMemcpy(want,expected,40,cudaMemcpyHostToDevice));
    fuse_fixture_env("NTT_FUSE_T","12");fuse_fixture_env("NTT_FUSE_M","4");
    fuse_fixture_env("NTT_FUSE_COOP_OUTER","2");fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");
    NttArena arena;arena.device=device;cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    int run=0;
    // Two balanced Latin blocks; every mode occupies every order position once.
    for(int mask:{0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1}) {
        fuse_fixture_env("NTT_OUTER_NARROW",std::to_string(mask).c_str());
        FuseCtx fc;ntt_arena_fuse(&arena,n,k,om,omi,fc);
        for(int repeat=0;repeat<4;++repeat) {
            sparse_input<<<(unsigned)((n+255)/256),256>>>(a,b,n);
            CK(cudaEventRecord(start));ntt_forward_fused(a,fc,om);ntt_forward_fused(b,fc,om);ntt_inverse_fused(a,b,fc,omi,scale);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
            CK(cudaMemset(bad,0,8));sparse_result<<<(unsigned)((n+255)/256),256>>>(a,n,want,bad);
            unsigned long long wrong=0;CK(cudaMemcpy(&wrong,bad,8,cudaMemcpyDeviceToHost));
            std::printf("ntt_v_bench: run=%d repeat=%d mask=%d k=%d N=%llu batch=1 passes=%d selected_M=%d coop=%d seconds=%.9f bad=%llu\n",
                run,repeat,mask,k,n,fc.passes_fwd,fc.m_max,(int)fc.coop_outer,ms/1000.0,wrong);
            if(wrong)std::exit(3);
        }
        ++run;
    }
    arena.release();CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(want));CK(cudaFree(bad));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    std::printf("ntt_v_done: live=%llu\n",g_fuse_base.live_bytes);
    if(g_fuse_base.live_bytes)std::exit(3);
}

int main(int argc,char**argv)
{
    int device=argc>1 ? std::atoi(argv[1]) : 1;CK(cudaSetDevice(device));
    if(argc>2 && !std::strcmp(argv[2],"--dense")){dense_gate();return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--vbench")){v_bench(device,argc>3 ? std::atoi(argv[3]) : 27);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--resources")) {
        v_resources<7,false,32>();v_resources<7,true,32>();v_resources<7,false,16>();v_resources<7,true,16>();
        v_resources<8,false,16>();v_resources<8,true,16>();v_resources<8,false,8>();v_resources<8,true,8>();return 0;
    }
    return original_coop_probe_main(argc,argv);
}
