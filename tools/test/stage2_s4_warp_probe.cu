#include <gmp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "../../src/ecm/stage2/cuda/device/point.cuh"
#include "../../src/ecm/stage2/cuda/device/s4_reduce.cuh"
// The probe uses the exact production S4 implementation.
static int probe_policy=0,probe_kernel=1;
int ecm_stage2_s4_coop_mode(){return probe_policy;}
int ecm_stage2_s4_coop_threads(){return 0;} // The probe supplies its exact block.
int ecm_stage2_s4_coop_kernel(){return probe_kernel;}
using Word=unsigned long long;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return 2;}}while(0)
static void pack(Word *p,int w,mpz_srcptr v){std::fill(p,p+w,0);size_t n=0;mpz_export(p,&n,-1,8,0,0,v);}
int main(int argc,char **argv) {
    if(argc<9 || argc>12){fprintf(stderr,"bits coeffs kind bpw backend repeats threads device [modulus_hex_file|-] [fault] [batches]\n");return 2;}
    const int bits=atoi(argv[1]),coeffs=atoi(argv[2]),kind=atoi(argv[3]),bpw=atoi(argv[4]),mode=atoi(argv[5]),
        repeats=atoi(argv[6]),threads=atoi(argv[7]),device=atoi(argv[8]),nw=(bits+63)/64;
    if(bits<65 || bits>16384 || coeffs<2 || coeffs>32768 || kind<0 || kind>3 || bpw<1 || bpw>64 ||
       (mode!=0 && mode!=32 && mode!=33 && mode!=34 && mode!=4096 && mode!=8192 && mode!=16384) || repeats<1 || repeats>5 || threads<32 || threads>256 || threads%32)return 2;
    const int fault=argc>=11?atoi(argv[10]):0,batches=argc==12?atoi(argv[11]):1;
    if(fault<0 || fault>2 || batches<1 || batches>4)return 2;
    probe_policy=(mode>=32 && mode<=34)?32:0;probe_kernel=mode==34?3:mode==33?2:1;
    CK(cudaSetDevice(device));
    mpz_t n,v,normalized,reciprocal,radix,expected;mpz_inits(n,v,normalized,reciprocal,radix,expected,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x571300+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,189);
    if(kind==1){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    if(kind==2){mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits-1);mpz_add_ui(n,n,1);}
    if(kind==3){mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,1);}
    if(argc>=10 && std::string(argv[9])!="-"){std::ifstream file(argv[9]);std::string hex;file>>hex;if(!file || mpz_set_str(n,hex.c_str(),16))return 2;}
    if(mpz_sizeinbase(n,2)!=(unsigned)bits || !mpz_odd_p(n))return 2;
    const int shift=nw*64-bits;mpz_mul_2exp(normalized,n,shift);
    S2GDeviceConstants constants{};pack(constants.divisor,nw,normalized);
    const Word top=constants.divisor[nw-1];
    mpz_set_ui(v,1);mpz_mul_2exp(v,v,128);mpz_sub_ui(v,v,1);mpz_import(radix,1,-1,8,0,0,&top);
    mpz_fdiv_q(reciprocal,v,radix);mpz_set_ui(radix,1);mpz_mul_2exp(radix,radix,64);mpz_sub(reciprocal,reciprocal,radix);
    Word recip=0;mpz_export(&recip,nullptr,-1,8,0,0,reciprocal);
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    const Word slots=(2ull*bits+17+bpw-1)/bpw,first=1,count=coeffs-2,mask=bpw==64?~0ull:(1ull<<bpw)-1;
    if((size_t)coeffs*batches*slots>(1ull<<27))return 2;
    std::vector<Word> digits((size_t)coeffs*batches*slots),output((size_t)coeffs*batches*nw,0xdeadbeefdeadbeefull),want(output);
    Word seed=0x123456789abcdefull;
    for(int i=0;i<coeffs*batches;++i)for(Word j=0;j<slots;++j) {
        seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
        digits[(size_t)i*slots+j]=i%7==0?0:i%7==1?mask:i%7==2?(j%2?mask:0):i%7==3?(j+1==slots?1:0):seed&mask;
    }
    // Production has zero padding above slot_bits. Keep the original kernel's
    // bound assertion active; rounded digit capacity is not a larger input.
    const int top_bits=int(2ull*bits+17-(slots-1)*bpw);
    for(int i=0;i<coeffs*batches;++i)digits[(size_t)(i+1)*slots-1]&=(1ull<<top_bits)-1;
    if(fault==1)digits[slots-1]|=1ull<<top_bits; // Omitted input must still trigger the bound guard.
    for(Word i=0;i<count*batches;++i) {
        mpz_set_ui(v,0);
        for(Word j=slots;j--;) {mpz_mul_2exp(v,v,bpw);const Word digit=digits[((size_t)(i/count)*coeffs+i%count+first)*slots+j];
            mpz_import(expected,1,-1,8,0,0,&digit);mpz_add(v,v,expected);}
        mpz_mod(expected,v,n);pack(want.data()+i*nw,nw,expected);
    }
    Word *di,*doo,*dbad;CK(cudaMalloc(&di,digits.size()*8));CK(cudaMalloc(&doo,output.size()*8));CK(cudaMalloc(&dbad,8));
    CK(cudaMemcpy(di,digits.data(),digits.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(doo,output.data(),output.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemset(dbad,0,8));
    const size_t shared=(nw+(threads/32)*(2*nw+4))*8;
    cudaFuncAttributes attr{};
    if(mode==34)CK(cudaFuncGetAttributes(&attr,s4_warp_reduce_kernel<256,true,true>));else if(mode==33)CK(cudaFuncGetAttributes(&attr,s4_warp_reduce_kernel<256,true>));else if(mode==32)CK(cudaFuncGetAttributes(&attr,s4_warp_reduce_kernel<256>));else CK(cudaFuncGetAttributes(&attr,s4_reduce_kernel<256,false>));
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));std::vector<float> times;
    for(int repeat=0;repeat<repeats;++repeat) {
        CK(cudaEventRecord(start));
        if(mode>=32 && mode<=34)ecm_launch_s4_reduce_kernel<256,false>(EcmCudaLaunch{dim3(unsigned((coeffs*batches+threads-1)/threads)),dim3(unsigned(threads))},di,slots*coeffs,bpw,slots,coeffs,batches,nullptr,0,nw,nw,nullptr,nw,doo,2ull*bits+17,dbad,nullptr,0,shift,recip,first,count,0);
        else if(!mode)s4_reduce_kernel<256,false><<<(coeffs*batches+threads-1)/threads,threads>>>(di,slots*coeffs,bpw,slots,coeffs,batches,nullptr,0,nw,nw,nullptr,nw,doo,2ull*bits+17,dbad,nullptr,0,shift,recip,first,count,0);
        else for(int batch=0;batch<batches;++batch)for(Word offset=0;offset<count;) {
            const Word take=std::min(count-offset,mode?Word(mode):count);
            s4_reduce_kernel<256,false><<<(coeffs+threads-1)/threads,threads>>>(di+batch*slots*coeffs,slots*coeffs,bpw,slots,coeffs,1,nullptr,0,nw,nw,nullptr,nw,doo+(batch*count+offset)*nw,2ull*bits+17,dbad,nullptr,0,shift,recip,first+offset,take,0);
            offset+=take;
        }
        CK(cudaGetLastError());CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));times.push_back(ms);
    }
    Word gpu_bad=0;CK(cudaMemcpy(output.data(),doo,output.size()*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&gpu_bad,dbad,8,cudaMemcpyDeviceToHost));
    if(fault==2)output[0]^=1;
    size_t bad=0;Word hash=1469598103934665603ull;
    for(size_t i=0;i<output.size();++i){bad+=output[i]!=want[i];hash=(hash^output[i])*1099511628211ull;}
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    printf("s4_warp_probe: bits=%d nw=%d kind=%d bpw=%d coeffs=%d batches=%d active=%llu backend=%d threads=%d shared=%zu bad=%zu gpu_bad=%llu checked_words=%zu regs=%d local=%zu median_ms=%.6f hash=%016llx samples_ms=",
        bits,nw,kind,bpw,coeffs,batches,count*batches,mode,threads,(mode>=32 && mode<=34)?shared:0,bad,gpu_bad,output.size(),attr.numRegs,attr.localSizeBytes,sorted[sorted.size()/2],hash);
    for(size_t i=0;i<times.size();++i)printf("%s%.6f",i?",":"",times[i]);puts("");
    CK(cudaFree(di));CK(cudaFree(doo));CK(cudaFree(dbad));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    mpz_clears(n,v,normalized,reciprocal,radix,expected,nullptr);gmp_randclear(rng);return bad || gpu_bad?1:0;
}
