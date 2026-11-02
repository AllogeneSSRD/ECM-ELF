// Test the shared square alongside isolated experimental alternatives. Compare
// the exact wide product and the existing Montgomery contract against GMP.
#include <gmp.h>
#include "../../src/ecm/cuda/arithmetic/cgbn/square.cuh"
#include "../experiments/cuda/cgbn_square/column.cuh"
#include "../experiments/cuda/cgbn_square/karatsuba.cuh"
#include "../../src/ecm/stage2/cuda/device/point_coop_field.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef ECM_SQUARE_BITS
#define ECM_SQUARE_BITS 8192
#endif
#ifndef ECM_SQUARE_TPI
#define ECM_SQUARE_TPI 16
#endif
#ifndef ECM_SQUARE_MODE_MASK
#define ECM_SQUARE_MODE_MASK 32767
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int BITS=ECM_SQUARE_BITS,TPI=ECM_SQUARE_TPI,TPB=128;
struct ProbeParameters : cgbn_default_parameters_t {
    static constexpr unsigned TPB=128,MAX_ROTATION=1,SHM_LIMIT=0;
};
using Context=cgbn_context_t<TPI,ProbeParameters>;
using Env=cgbn_env_t<Context,BITS>;
using Bn=typename Env::cgbn_t;
using Mem=cgbn_mem_t<BITS>;
template<int MODE> constexpr bool is_mont=MODE<3 || MODE==5 || MODE==7 || MODE==9 || MODE==11 || MODE==13 || MODE==14;
template<int MODE> __global__ void square_probe(Mem *input,Mem *modulus,Mem *output,
    int count,int repeats,bool alias,uint32_t np0,int fault,int mersenne_bits) {
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/TPI;if(i>=count)return;
    Context ctx(cgbn_no_checks);Env e(ctx);Bn x,n,v;typename Env::cgbn_wide_t wide;
    cgbn_load(e,x,input+i);cgbn_load(e,n,modulus);
    extern __shared__ uint32_t workspace[];
    auto *scratch=workspace+(threadIdx.x/TPI)*ecm_cgbn::square_scratch_words<Env>();
    for(int k=0;k<repeats;++k) {
        if constexpr(MODE==0) {
            if(alias)cgbn_mont_sqr(e,x,x,n,np0);else {cgbn_mont_sqr(e,v,x,n,np0);cgbn_set(e,x,v);}
            if(cgbn_compare(e,x,n)>=0)cgbn_sub(e,x,x,n);
        } else if constexpr(MODE==1 || MODE==2) {
            if constexpr(MODE==1) {
                cgbn_sqr_wide(e,wide,x);cgbn_mont_reduce_wide(e,v,wide,n,np0);
                if(cgbn_compare(e,v,n)>=0)cgbn_sub(e,v,v,n);cgbn_set(e,x,v);
            } else {
                if(alias)ecm_cgbn::mont_square(e,x,x,n,np0,scratch);
                else {ecm_cgbn::mont_square(e,v,x,n,np0,scratch);cgbn_set(e,x,v);}
            }
        } else if constexpr(MODE==5 || MODE==7) {
            constexpr unsigned depth=MODE==5?1:2;
            if(alias)ecm_cgbn::mont_square_split<depth>(e,x,x,n,np0);
            else {ecm_cgbn::mont_square_split<depth>(e,v,x,n,np0);cgbn_set(e,x,v);}
        } else if constexpr(MODE==9 || MODE==11) {
            constexpr unsigned depth=MODE==9?1:2;
            if(alias)ecm_cgbn::mont_square_karatsuba<depth>(e,x,x,n,np0);
            else {ecm_cgbn::mont_square_karatsuba<depth>(e,v,x,n,np0);cgbn_set(e,x,v);}
        } else if constexpr(MODE==13 || MODE==14) {
            const Stage2CoopField<Env> field{e,n,np0,mersenne_bits};
            if constexpr(MODE==13) {
                if(alias)field.square(x,x);else {field.square(v,x);cgbn_set(e,x,v);}
            } else {
                ecm_cgbn::square_wide_split<2>(e,wide,x);
                if(alias)field.reduce_mersenne(x,wide);else {field.reduce_mersenne(v,wide);cgbn_set(e,x,v);}
            }
        } else {
            if(alias) {
                cgbn_set(e,wide._low,x);
                if constexpr(MODE==3)cgbn_sqr_wide(e,wide,wide._low);
                else if constexpr(MODE==4)ecm_cgbn::square_wide(e,wide,wide._low,scratch);
                else if constexpr(MODE==6 || MODE==8){constexpr unsigned depth=MODE==6?1:2;ecm_cgbn::square_wide_split<depth>(e,wide,wide._low);}
                else {constexpr unsigned depth=MODE==10?1:2;ecm_cgbn::square_wide_karatsuba<depth>(e,wide,wide._low);}
            } else {
                if constexpr(MODE==3)cgbn_sqr_wide(e,wide,x);
                else if constexpr(MODE==4)ecm_cgbn::square_wide(e,wide,x,scratch);
                else if constexpr(MODE==6 || MODE==8){constexpr unsigned depth=MODE==6?1:2;ecm_cgbn::square_wide_split<depth>(e,wide,x);}
                else {constexpr unsigned depth=MODE==10?1:2;ecm_cgbn::square_wide_karatsuba<depth>(e,wide,x);}
            }
            if(fault==2 && k==0 && i==0) {
                cgbn_set_ui32(e,v,1);cgbn_bitwise_xor(e,wide._high,wide._high,v);
            }
            // Both halves feed the next input. A low-only recurrence would
            // allow unused intermediate high products to be optimized away.
            if(k+1<repeats)cgbn_bitwise_xor(e,x,wide._low,wide._high);
        }
    }
    if constexpr(is_mont<MODE>){cgbn_store(e,output+2*i,x);cgbn_set_ui32(e,v,0);cgbn_store(e,output+2*i+1,v);}
    else {cgbn_store(e,output+2*i,wide._low);cgbn_store(e,output+2*i+1,wide._high);}
    if(fault==1 && i==0) {
        constexpr unsigned bits=0xffffffffu>>(32-TPI);
        __syncwarp(bits<<((threadIdx.x&31)&~(TPI-1)));
        if(threadIdx.x%TPI==0)output[0]._limbs[0]^=1;
    }
}
void pack(uint32_t *p,mpz_srcptr x){std::fill(p,p+BITS/32,0);size_t n=0;mpz_export(p,&n,-1,4,0,0,x);}
template<int MODE> int run(int device,int count,int repeats,int alias,int kind,int samples,int fault) {
    CK(cudaSetDevice(device));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    mpz_t n,a,t,rinv,limit,feedback;mpz_inits(n,a,t,rinv,limit,feedback,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x5112026+BITS+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,BITS);mpz_sub_ui(n,n,kind==1?1:189);
    if(kind==2){mpz_urandomb(n,rng,BITS-6);mpz_setbit(n,BITS-7);mpz_setbit(n,0);}
    if(kind==3){mpz_set_ui(n,1);mpz_mul_2exp(n,n,BITS-6);mpz_sub_ui(n,n,1);}
    const int mersenne_bits=kind==1?BITS:kind==3?BITS-6:0;
    // CGBN deliberately uses the padded physical radix, not the public BITS.
    mpz_set_ui(t,1);mpz_mul_2exp(t,t,Env::UNPADDED_BITS);
    if(!mpz_invert(rinv,t,n))return 2;
    mpz_set_ui(limit,1);mpz_mul_2exp(limit,limit,BITS);
    const size_t words=size_t(count)*(BITS/32);
    constexpr size_t nw=BITS/32;
    std::vector<uint32_t> input(words),hn(nw),expected(2*words),got(2*words+2*nw);
    for(int i=0;i<count;++i) {
        mpz_urandomb(a,rng,BITS);
        if(count==1)mpz_setbit(a,0); // Latency uses a nontrivial operand, not the zero edge case.
        else if(i<2)mpz_set_ui(a,i);
        else if(i<4)mpz_sub_ui(a,is_mont<MODE>?n:limit,i-1);
        else if(i<12){mpz_set_ui(a,1);mpz_mul_2exp(a,a,(BITS-1)-(i-4));mpz_sub_ui(a,a,i&1);}
        else if(i<16){mpz_set_ui(a,0);for(int b=i&1;b<BITS;b+=2)mpz_setbit(a,b);}
        else if(i<32){mpz_set_ui(a,1);mpz_mul_2exp(a,a,std::min(BITS-1,32*(i-16)+31));mpz_add_ui(a,a,1);}
        if constexpr(is_mont<MODE>)mpz_mod(a,a,n);
        pack(input.data()+size_t(i)*(BITS/32),a);
        for(int k=0;k<repeats;++k) {
            mpz_mul(t,a,a);
            if constexpr(is_mont<MODE>){mpz_mul(t,t,rinv);mpz_mod(a,t,n);}
            else {if(k+1<repeats){mpz_fdiv_r_2exp(a,t,BITS);mpz_fdiv_q_2exp(feedback,t,BITS);mpz_xor(a,a,feedback);}}
        }
        if constexpr(is_mont<MODE>)pack(expected.data()+size_t(2*i)*(BITS/32),a);
        else {
            mpz_fdiv_r_2exp(a,t,BITS);pack(expected.data()+size_t(2*i)*(BITS/32),a);
            mpz_fdiv_q_2exp(a,t,BITS);pack(expected.data()+size_t(2*i+1)*(BITS/32),a);
        }
    }
    pack(hn.data(),n);uint32_t inv=1;for(int k=0;k<5;++k)inv*=2-hn[0]*inv;
    Mem *di_owner=nullptr,*dn_owner=nullptr,*out_owner=nullptr;
    CK(cudaMalloc(&di_owner,(words+2*nw)*4));CK(cudaMalloc(&dn_owner,3*nw*4));CK(cudaMalloc(&out_owner,(2*words+2*nw)*4));
    CK(cudaMemset(di_owner,0xa5,(words+2*nw)*4));CK(cudaMemset(dn_owner,0xa5,3*nw*4));CK(cudaMemset(out_owner,0xa5,(2*words+2*nw)*4));
    Mem *di=di_owner+1,*dn=dn_owner+1,*out=out_owner+1;
    CK(cudaMemcpy(di,input.data(),words*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dn,hn.data(),BITS/8,cudaMemcpyHostToDevice));
    constexpr size_t shared=(MODE==2 || MODE==4)?4*(TPB/TPI)*ecm_cgbn::square_scratch_words<Env>():0;
    cudaFuncAttributes attr{};CK(cudaFuncGetAttributes(&attr,square_probe<MODE>));int blocks=0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,square_probe<MODE>,TPB,shared));
    cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));std::vector<float> times;
    for(int k=0;k<samples;++k) {
        CK(cudaEventRecord(start));square_probe<MODE><<<(count*TPI+TPB-1)/TPB,TPB,shared>>>(di,dn,out,count,repeats,alias,0-inv,fault,mersenne_bits);
        CK(cudaGetLastError());CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));float ms=0;
        CK(cudaEventElapsedTime(&ms,start,stop));times.push_back(ms);
        CK(cudaMemcpy(got.data(),out_owner,got.size()*4,cudaMemcpyDeviceToHost));
        size_t bad=0;for(size_t i=0;i<2*words;++i)bad+=got[nw+i]!=expected[i];
        for(size_t i=0;i<nw;++i){bad+=got[i]!=0xa5a5a5a5u;bad+=got[nw+2*words+i]!=0xa5a5a5a5u;}
        if(bad){printf("cgbn_square_bad: mode=%d sample=%d words=%zu\n",MODE,k,bad);return 3;}
    }
    // Input/modulus are immutable, including their padding and tail regions.
    std::vector<uint32_t> read_input(words+2*nw),read_n(3*nw);
    CK(cudaMemcpy(read_input.data(),di_owner,read_input.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(read_n.data(),dn_owner,read_n.size()*4,cudaMemcpyDeviceToHost));
    size_t bad=0;for(size_t i=0;i<words;++i)bad+=read_input[nw+i]!=input[i];
    for(size_t i=0;i<nw;++i){bad+=read_n[nw+i]!=hn[i];bad+=read_input[i]!=0xa5a5a5a5u;bad+=read_input[nw+words+i]!=0xa5a5a5a5u;bad+=read_n[i]!=0xa5a5a5a5u;bad+=read_n[2*nw+i]!=0xa5a5a5a5u;}
    if(bad){printf("cgbn_square_bad: mode=%d immutable_or_guard_words=%zu\n",MODE,bad);return 3;}
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    unsigned depth=0;
    if constexpr(((MODE>=5 && MODE<=8)||MODE==14) && ecm_cgbn::square_split_eligible<Env>())depth=std::min(unsigned(MODE<7?1:2),unsigned(TPI==8?1:TPI==16?2:3));
    if constexpr(MODE>=9 && MODE<=12 && ecm_cgbn::square_karatsuba_eligible<Env>()) {
        unsigned levels=0,l=Env::LIMBS;while(l && !(l&1)){++levels;l/=2;}
        depth=std::min(unsigned(MODE<11?1:2),levels);
    }
    printf("cgbn_square: bits=%d radix_bits=%u tpi=%d mode=%d count=%d repeats=%d alias=%d kind=%d mersenne_bits=%d feedback=%s bad=0 guards=1 max_rotation=1 effective_depth=%u regs=%d local=%zu shared=%zu blocks_sm=%d sm_count=%d median_ms=%.6f samples_ms=",
        BITS,Env::UNPADDED_BITS,TPI,MODE,count,repeats,alias,kind,mersenne_bits,is_mont<MODE>?"modular":"xor_halves",depth,attr.numRegs,attr.localSizeBytes,shared,blocks,prop.multiProcessorCount,sorted[sorted.size()/2]);
    for(size_t i=0;i<times.size();++i)printf("%s%.6f",i?",":"",times[i]);puts("");
    CK(cudaFree(di_owner));CK(cudaFree(dn_owner));CK(cudaFree(out_owner));CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));
    gmp_randclear(rng);mpz_clears(n,a,t,rinv,limit,feedback,nullptr);return 0;
}
int main(int argc,char **argv) {
    if(argc!=9){fprintf(stderr,"device count repeats mode alias kind samples fault\n");return 2;}
    const int device=atoi(argv[1]),count=atoi(argv[2]),repeats=atoi(argv[3]),mode=atoi(argv[4]),alias=atoi(argv[5]),kind=atoi(argv[6]),samples=atoi(argv[7]),fault=atoi(argv[8]);
    if(count<1||count>16384||repeats<1||repeats>128||mode<0||mode>14||alias<0||alias>1||kind<0||kind>3||samples<1||samples>9||fault<0||fault>2)return 2;
    if(fault==2 && (repeats<2 || (mode!=3 && mode!=4 && mode!=6 && mode!=8 && mode!=10 && mode!=12)))return 2;
    if((mode==13 || mode==14) && ((kind!=1 && kind!=3) || Env::BITS!=Env::UNPADDED_BITS))return 2;
#define CASE(M) case M:if constexpr(ECM_SQUARE_MODE_MASK&(1<<M))return run<M>(device,count,repeats,alias,kind,samples,fault);else return 2
    switch(mode){CASE(0);CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);CASE(9);CASE(10);CASE(11);CASE(12);CASE(13);CASE(14);}return 2;
}
