// Exact GPU-resident integer products using CGBN or linked production NTT.
// Packing, carry and unpacking are timed; tables/graphs/allocations are separate.
#include <gmp.h>
#include "../../third_party/cgbn/include/cgbn/cgbn.h"
#include "../../src/ecm/cuda/ntt/runtime.h"
#include <memory>
#include <stdexcept>

#ifndef ECM_CROSS_BITS
#define ECM_CROSS_BITS 8192
#endif
#ifndef ECM_CROSS_TPI
#define ECM_CROSS_TPI 16
#endif
using U=unsigned long long;
constexpr unsigned Bits=ECM_CROSS_BITS,Tpi=ECM_CROSS_TPI,Tpb=128,Words=Bits/32;
struct Params {static constexpr unsigned TPB=Tpb,MAX_ROTATION=1,SHM_LIMIT=0;static constexpr bool CONSTANT_TIME=false;};
using Context=cgbn_context_t<Tpi,Params>;using Env=cgbn_env_t<Context,Bits>;
using Mem=cgbn_mem_t<Bits>;

template<class T> struct Buffer {
    T *owner=nullptr,*data=nullptr;size_t words=0;
    explicit Buffer(size_t count):words(count) {
        CK(cudaMalloc(&owner,(words+64)*sizeof(T)));data=owner+32;
        CK(cudaMemset(owner,0xa5,(words+64)*sizeof(T)));
    }
    ~Buffer(){if(owner)cudaFree(owner);}
    void check()const {
        unsigned char bytes[32*sizeof(T)];
        for(auto ptr:{owner,data+words}) {
            CK(cudaMemcpy(bytes,ptr,sizeof(bytes),cudaMemcpyDeviceToHost));
            for(auto b:bytes)if(b!=0xa5)throw std::runtime_error("buffer canary changed");
        }
    }
    Buffer(const Buffer &)=delete;
};

template<bool Modular>
__global__ void cgbn_chain(uint32_t *state,const uint32_t *input_b,const uint32_t *modulus,
    uint32_t *output,unsigned count,unsigned iterations,uint32_t np0) {
    const unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/Tpi;if(i>=count)return;
    Context ctx(cgbn_no_checks);Env env(ctx);typename Env::cgbn_t a,b,n,r;
    typename Env::cgbn_wide_t wide;
    cgbn_load(env,a,(Mem *)state+i);cgbn_load(env,b,(Mem *)input_b+i);
    if constexpr(Modular)cgbn_load(env,n,(Mem *)modulus);
    for(unsigned step=0;step<iterations;++step) {
        if constexpr(Modular) {
            cgbn_mont_mul(env,r,a,b,n,np0);
            if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
            cgbn_set(env,a,r);
        } else {
            cgbn_mul_wide(env,wide,a,b);
            cgbn_bitwise_xor(env,a,wide._low,wide._high);
        }
    }
    if constexpr(Modular) {
        cgbn_store(env,(Mem *)output+2*i,a);cgbn_set_ui32(env,r,0);cgbn_store(env,(Mem *)output+2*i+1,r);
    } else {
        cgbn_store(env,(Mem *)output+2*i,wide._low);cgbn_store(env,(Mem *)output+2*i+1,wide._high);
    }
    cgbn_store(env,(Mem *)state+i,a);
}

__global__ void pack_integer(const uint32_t *a,const uint32_t *b,U *x,U *y,
    U n,unsigned digits,int bpw) {
    const U j=blockIdx.x*(U)blockDim.x+threadIdx.x;if(j>=n)return;
    const unsigned slice=blockIdx.y;U va=0,vb=0;
    if(j<digits) {
        const U bit=j*bpw;const unsigned w=(unsigned)(bit/32),offset=bit%32;
        const size_t base=(size_t)slice*Words;
        va=a[base+w];vb=b[base+w];
        if(w+1<Words){va|=(U)a[base+w+1]<<32;vb|=(U)b[base+w+1]<<32;}
        const U mask=(1ull<<bpw)-1;va=(va>>offset)&mask;vb=(vb>>offset)&mask;
    }
    x[slice*n+j]=va;y[slice*n+j]=vb;
}

__device__ __forceinline__ uint32_t read_digit_word(const U *digits,U n,int bpw,unsigned word) {
    const U bit=32ull*word,index=bit/bpw;const unsigned off=bit%bpw;
    U value=digits[index]>>off;
    if(off+32>(unsigned)bpw && index+1<n)value|=digits[index+1]<<(bpw-off);
    if(off+32>2u*bpw && index+2<n)value|=digits[index+2]<<(2*bpw-off);
    return (uint32_t)value;
}
__global__ void unpack_integer(const U *digits,U n,int bpw,uint32_t *output,uint32_t *state,bool modular) {
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=2*Words)return;
    const unsigned slice=blockIdx.y;digits+=(U)slice*n;
    output[(size_t)slice*2*Words+j]=read_digit_word(digits,n,bpw,j);
    if(!modular && j<Words)state[(size_t)slice*Words+j]=
        read_digit_word(digits,n,bpw,j)^read_digit_word(digits,n,bpw,j+Words);
}
__global__ void reduce_integer(uint32_t *state,uint32_t *output,const uint32_t *modulus,
    unsigned count,uint32_t np0) {
    const unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/Tpi;if(i>=count)return;
    Context ctx(cgbn_no_checks);Env env(ctx);typename Env::cgbn_wide_t product;
    typename Env::cgbn_t n,r;
    cgbn_load(env,product._low,(Mem *)output+2*i);cgbn_load(env,product._high,(Mem *)output+2*i+1);
    cgbn_load(env,n,(Mem *)modulus);cgbn_mont_reduce_wide(env,r,product,n,np0);
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
    cgbn_store(env,(Mem *)state+i,r);cgbn_store(env,(Mem *)output+2*i,r);
    cgbn_set_ui32(env,r,0);cgbn_store(env,(Mem *)output+2*i+1,r);
}
__global__ void corrupt_high(uint32_t *output){if(!threadIdx.x && !blockIdx.x)output[Words]^=1;}
__global__ void check_integer_tail(const U *digits,U n,int bpw,U *bad) {
    const U i=blockIdx.x*(U)blockDim.x+threadIdx.x;if(i>=n)return;
    const U v=digits[blockIdx.y*n+i],bit=i*bpw;
    bool invalid=v>=(1ull<<bpw);
    if(bit>=2*Bits)invalid=invalid || v!=0;
    else if(bit+bpw>2*Bits)invalid=invalid || (v>>(2*Bits-bit))!=0;
    if(invalid)atomicAdd(bad,1ull);
}

static void export_words(uint32_t *out,const mpz_t value) {
    std::memset(out,0,Words*4);size_t count=0;mpz_export(out,&count,-1,4,0,0,value);
    if(count>Words)throw std::runtime_error("oracle word count");
}

int main(int argc,char **argv) {
    if(argc!=10){std::fprintf(stderr,"device count iterations backend[0=CGBN|1=NTT|2=NTTgraph] op[0=wide|1=Mont] tile samples fault kind[0=random|1=edges]\n");return 2;}
    try {
        const int device=std::atoi(argv[1]),count=std::atoi(argv[2]),iterations=std::atoi(argv[3]);
        const int backend=std::atoi(argv[4]),op=std::atoi(argv[5]),tile=std::atoi(argv[6]);
        const int samples=std::atoi(argv[7]),fault=std::atoi(argv[8]),kind=std::atoi(argv[9]);
        if(count<1 || count>4096 || iterations<1 || iterations>128 || backend<0 || backend>2 || op<0 || op>1 ||
            tile<5 || tile>12 || samples<1 || samples>9 || fault<0 || fault>1 || kind<0 || kind>1)return 2;
        std::setvbuf(stdout,nullptr,_IONBF,0);stage2_log::level=stage2_log::quiet;
        CK(cudaSetDevice(device));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
        _putenv_s("NTT_FUSE_T",std::to_string(tile).c_str());_putenv_s("NTT_FUSE_WARP_TAIL","1");
        _putenv_s("NTT_FUSE_M","4");_putenv_s("NTT_FUSE_COOP_OUTER","0");_putenv_s("NTT_FUSE_COMPACT_SCRATCH","1");
        const auto oracle_begin=std::chrono::steady_clock::now();
        const size_t words=(size_t)count*Words;
        std::vector<uint32_t> ha(words),hb(words),hn(Words),expected(2*words),expected_state(words);
        mpz_t a,b,n,t,rinv,lo,hi;mpz_inits(a,b,n,t,rinv,lo,hi,nullptr);
        mpz_set_ui(n,1);mpz_mul_2exp(n,n,Bits-6);mpz_sub_ui(n,n,189);export_words(hn.data(),n);
        mpz_set_ui(rinv,1);mpz_mul_2exp(rinv,rinv,Env::UNPADDED_BITS);if(!mpz_invert(rinv,rinv,n))return 2;
        gmp_randstate_t random;gmp_randinit_mt(random);gmp_randseed_ui(random,0x72ab+Bits);
        for(int i=0;i<count;++i) {
            mpz_urandomb(a,random,Bits);mpz_urandomb(b,random,Bits);
            if(op){mpz_mod(a,a,n);mpz_mod(b,b,n);}
            if(kind && i<8) {
                if(i==0)mpz_set_ui(a,0);
                if(i==1)mpz_set_ui(a,1);
                if(i==2 || i==3) {mpz_set_ui(a,1);mpz_mul_2exp(a,a,Bits);mpz_sub_ui(a,a,1);if(op)mpz_sub_ui(a,n,1);if(i==2)mpz_set(b,a);}
                if(i==4){mpz_set_ui(a,1);mpz_mul_2exp(a,a,Bits/2);if(op)mpz_mod(a,a,n);}
                if(i==5) {mpz_set_ui(a,0);for(unsigned j=0;j<Bits;j+=2)mpz_setbit(a,j);if(op)mpz_mod(a,a,n);}
            }
            export_words(ha.data()+(size_t)i*Words,a);export_words(hb.data()+(size_t)i*Words,b);
            for(int step=0;step<iterations;++step) {
                mpz_mul(t,a,b);
                if(op) {mpz_mul(t,t,rinv);mpz_mod(a,t,n);}
                else {mpz_fdiv_r_2exp(lo,t,Bits);mpz_fdiv_q_2exp(hi,t,Bits);mpz_xor(a,lo,hi);}
            }
            if(op)export_words(expected.data()+(size_t)i*2*Words,a);
            else {export_words(expected.data()+(size_t)i*2*Words,lo);export_words(expected.data()+((size_t)i*2+1)*Words,hi);}
            export_words(expected_state.data()+(size_t)i*Words,a);
        }
        uint32_t inverse=1;for(int j=0;j<5;++j)inverse*=2-hn[0]*inverse;
        const uint32_t np0=0-inverse;
        U input_hash=14695981039346656037ull;
        for(const auto *values:{&ha,&hb,&hn})for(auto value:*values){input_hash^=value;input_hash*=1099511628211ull;}
        const double oracle_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-oracle_begin).count();
        const auto setup_begin=std::chrono::steady_clock::now();
        Buffer<uint32_t> initial(words),state(words),input_b(words),modulus(Words),output(2*words);
        CK(cudaMemcpy(initial.data,ha.data(),words*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(input_b.data,hb.data(),words*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(modulus.data,hn.data(),Words*4,cudaMemcpyHostToDevice));
        int bpw=0,k=0;U digits=0,length=0,omega=0,omega_inv=0,n_scale=0;FuseCtx context;
        std::unique_ptr<Buffer<U>> x,y,carried,diagnostic;
        const unsigned cgblocks=(count*Tpi+Tpb-1)/Tpb;
        if(backend) {
            for(int candidate=31;candidate>=16;--candidate) {
                const U m=(Bits+candidate-1)/candidate;
                if(exact_ok_terms(m,candidate)){bpw=candidate;digits=m;break;}
            }
            if(!bpw)throw std::runtime_error("no exact digit width");
            length=1;while(length<2*digits+1){length*=2;++k;}
            omega=gl_pow_host(7,(GL_P-1)/length);omega_inv=gl_pow_host(omega,GL_P-2);n_scale=gl_pow_host(length,GL_P-2);
            fuse_init(context,length,k,omega,omega_inv,true);ntt_fuse_cache_tables(context,omega,omega_inv);
            x=std::make_unique<Buffer<U>>(length*count);y=std::make_unique<Buffer<U>>(length*count);carried=std::make_unique<Buffer<U>>(length*count);
            diagnostic=std::make_unique<Buffer<U>>(1);
        }
        auto ntt_step=[&](cudaStream_t stream) {
            const auto grid=dim3((unsigned)((length+255)/256),(unsigned)count);
            pack_integer<<<grid,256,0,stream>>>(state.data,input_b.data,x->data,y->data,length,(unsigned)digits,bpw);
            ntt_forward_fused(x->data,context,omega,count,stream);
            ntt_forward_fused(y->data,context,omega,count,stream);
            ntt_inverse_fused(x->data,y->data,context,omega_inv,n_scale,count,stream);
            ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(grid,256,0,stream),x->data,carried->data,length,bpw,length);
            unpack_integer<<<dim3((2*Words+255)/256,(unsigned)count),256,0,stream>>>(carried->data,length,bpw,output.data,state.data,op!=0);
            if(op)reduce_integer<<<cgblocks,Tpb,0,stream>>>(state.data,output.data,modulus.data,count,np0);
            CK(cudaGetLastError());
        };
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        CK(cudaDeviceSynchronize());
        const auto graph_begin=std::chrono::steady_clock::now();
        if(backend==2) {
            cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int j=0;j<iterations;++j)ntt_step(stream);
            CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&executable,graph,0));
            CK(cudaStreamDestroy(stream));
        }
        // Graph capture is implemented on an explicit stream in the launcher.
        // No calculation is run as a warmup or discarded timing sample.
        auto launch=[&]() {
            if(!backend) {
                if(op)cgbn_chain<true><<<cgblocks,Tpb>>>(state.data,input_b.data,modulus.data,output.data,count,iterations,np0);
                else cgbn_chain<false><<<cgblocks,Tpb>>>(state.data,input_b.data,modulus.data,output.data,count,iterations,np0);
            } else if(backend==2)CK(cudaGraphLaunch(executable,0));
            else for(int j=0;j<iterations;++j)ntt_step(0);
        };
        const double graph_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-graph_begin).count();
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_begin).count();
        cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));
        std::vector<float> times;std::vector<uint32_t> got(2*words),got_state(words);
        for(int sample=0;sample<samples;++sample) {
            CK(cudaMemcpy(state.data,initial.data,words*4,cudaMemcpyDeviceToDevice));
            CK(cudaEventRecord(start));launch();if(fault)corrupt_high<<<1,32>>>(output.data);
            CK(cudaGetLastError());CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));
            float milliseconds=0;CK(cudaEventElapsedTime(&milliseconds,start,stop));times.push_back(milliseconds);
            if(backend) {
                CK(cudaMemset(diagnostic->data,0,sizeof(U)));
                check_integer_tail<<<dim3((unsigned)((length+255)/256),(unsigned)count),256>>>(carried->data,length,bpw,diagnostic->data);
                U invalid=0;CK(cudaMemcpy(&invalid,diagnostic->data,sizeof(U),cudaMemcpyDeviceToHost));
                if(invalid)throw std::runtime_error("noncanonical digits or nonzero product tail");
            }
            CK(cudaMemcpy(got.data(),output.data,2*words*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(got_state.data(),state.data,words*4,cudaMemcpyDeviceToHost));
            if(got!=expected || got_state!=expected_state){std::printf("crossover_bad: backend=%d sample=%d\n",backend,sample);return 3;}
        }
        for(auto pair:{std::make_pair(initial.data,&ha),std::make_pair(input_b.data,&hb),std::make_pair(modulus.data,&hn)}) {
            std::vector<uint32_t> copy(pair.second->size());CK(cudaMemcpy(copy.data(),pair.first,copy.size()*4,cudaMemcpyDeviceToHost));
            if(copy!=*pair.second)throw std::runtime_error("immutable input changed");
        }
        initial.check();state.check();input_b.check();modulus.check();output.check();
        if(backend){x->check();y->check();carried->check();diagnostic->check();}
        cudaFuncAttributes attr{};
        if(op)CK(cudaFuncGetAttributes(&attr,cgbn_chain<true>));else CK(cudaFuncGetAttributes(&attr,cgbn_chain<false>));
        std::printf("crossover: bits=%u radix_bits=%u tpi=%u count=%d iterations=%d backend=%d op=%d tile=%d actual_tile=%d length=%llu bpw=%d digits=%llu passes=%d cgbn_regs=%d cgbn_local_bytes=%zu sm_count=%d input_hash=%016llx bad=0 guards=1 setup_seconds=%.9f graph_seconds=%.9f oracle_seconds=%.9f samples_ms=",
            Bits,Env::UNPADDED_BITS,Tpi,count,iterations,backend,op,tile,context.t,length,bpw,digits,context.passes_fwd,attr.numRegs,attr.localSizeBytes,prop.multiProcessorCount,input_hash,setup_seconds,graph_seconds,oracle_seconds);
        for(size_t j=0;j<times.size();++j)std::printf("%s%.6f",j?",":"",times[j]);puts("");
        CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));if(executable)CK(cudaGraphExecDestroy(executable));if(graph)CK(cudaGraphDestroy(graph));
        if(backend)fuse_release(context);gmp_randclear(random);mpz_clears(a,b,n,t,rinv,lo,hi,nullptr);return 0;
    } catch(const std::exception &e){std::fprintf(stderr,"crossover_error: %s\n",e.what());return 2;}
}
