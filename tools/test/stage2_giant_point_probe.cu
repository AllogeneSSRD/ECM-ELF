#include <cuda_runtime.h>
#include <gmp.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../src/ecm/stage2/cuda/launch.h"
#include "../experiments/cuda/stage2_tree/giant_launch.h"
using Word=unsigned long long;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
void pack(Word*p,int w,mpz_srcptr v){std::fill(p,p+w,0);size_t z;mpz_export(p,&z,-1,8,0,0,v);}
void unpack(mpz_ptr v,const Word*p,int w){mpz_import(v,w,-1,8,0,0,p);}
template<int NW>void launch(int rotate,int threads,int nw,int npts,int block,Word ninv,Word*in,Word*out){
    int count=(npts+block-1)/block;auto *n=in,*dx=in+nw,*dz=in+2*nw,*sx=in+3*nw,*sz=sx+count*nw;
    auto *ex=sz+count*nw,*ez=ex+count*nw;
    EcmCudaLaunch cfg{(unsigned)((count+threads-1)/threads),(unsigned)threads};
    if constexpr(NW==64 || NW==128 || NW==256) {
        if(!rotate) {
            ecm_launch_s2g_chain_kernel<NW,true>(cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,out,out+(size_t)npts*nw,1);
            return;
        }
    }
    ecm_experiment_giant_launch(NW,rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,out,out+(size_t)npts*nw,1);
}
int main(int argc,char**argv){
    if(argc!=8){fprintf(stderr,"nw template_width points chain_block threads rotate device\n");return 2;}
    int nw=atoi(argv[1]),width=atoi(argv[2]),npts=atoi(argv[3]),block=atoi(argv[4]),threads=atoi(argv[5]),rotate=atoi(argv[6]),device=atoi(argv[7]);
    if(nw<1||nw>256||width<nw||npts<1||npts>200000||block<2||threads<32||threads>256||rotate<0||rotate>1)return 2;
    CK(cudaSetDevice(device));S2GDeviceConstants constants{};CK(ecm_stage2_copy_constants(&constants,sizeof(constants)));int count=(npts+block-1)/block;
    std::vector<Word> in((3+4*count)*nw),out((size_t)2*npts*nw);
    mpz_t n,a,r,rinv,x[6],y[2],t,u;mpz_inits(n,a,r,rinv,t,u,nullptr);for(auto &v:x)mpz_init(v);for(auto &v:y)mpz_init(v);
    // An odd composite just below the radix exercises both reduction and carry.
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,64*nw);mpz_sub_ui(n,n,189);pack(in.data(),nw,n);
    mpz_set_ui(r,1);mpz_mul_2exp(r,r,64*nw);mpz_invert(rinv,r,n);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,20261018);
    for(size_t i=nw;i<in.size();i+=nw){mpz_urandomm(a,rng,n);pack(in.data()+i,nw,a);}
    Word inv=1;for(int i=0;i<6;++i)inv*=2-in[0]*inv;Word ninv=0-inv;
    Word *di,*dout;CK(cudaMalloc(&di,in.size()*8));CK(cudaMalloc(&dout,out.size()*8));CK(cudaMemcpy(di,in.data(),in.size()*8,cudaMemcpyHostToDevice));
    auto run=[&](){
        switch(width){
        case 64:launch<64>(rotate,threads,nw,npts,block,ninv,di,dout);break;
        case 128:launch<128>(rotate,threads,nw,npts,block,ninv,di,dout);break;
        case 160:launch<160>(rotate,threads,nw,npts,block,ninv,di,dout);break;
        case 192:launch<192>(rotate,threads,nw,npts,block,ninv,di,dout);break;
        case 256:launch<256>(rotate,threads,nw,npts,block,ninv,di,dout);break;
        default:exit(2);}
        CK(cudaGetLastError());
    };
    run();CK(cudaDeviceSynchronize());cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    float ms=0;CK(cudaEventRecord(begin));for(int i=0;i<3;++i)run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));
    CK(cudaMemcpy(out.data(),dout,out.size()*8,cudaMemcpyDeviceToHost));
    size_t bad=0,checked=0;
    // Independent old eight-product polynomial coordinates, including every point
    // in the first/last block and one point from all intervening blocks.
    auto mul=[&](mpz_ptr dst,mpz_srcptr b,mpz_srcptr c){mpz_mul(dst,b,c);mpz_mul(dst,dst,rinv);mpz_mod(dst,dst,n);};
    for(int b=0;b<count;++b){
        unpack(x[0],in.data()+(3+b)*nw,nw);unpack(x[1],in.data()+(3+count+b)*nw,nw);
        unpack(x[2],in.data()+(3+2*count+b)*nw,nw);unpack(x[3],in.data()+(3+3*count+b)*nw,nw);
        unpack(x[4],in.data()+nw,nw);unpack(x[5],in.data()+2*nw,nw);
        int len=std::min(block,npts-b*block);
        for(int k=0;k<len;++k){
            if(k>=2){
                mul(t,x[2],x[4]);mul(u,x[3],x[5]);mpz_sub(t,t,u);mul(t,t,t);mul(y[0],x[1],t);
                mul(t,x[2],x[5]);mul(u,x[3],x[4]);mpz_sub(t,t,u);mul(t,t,t);mul(y[1],x[0],t);
                mpz_set(x[0],x[2]);mpz_set(x[1],x[3]);mpz_set(x[2],y[0]);mpz_set(x[3],y[1]);
            }
            if(b!=0 && b!=count-1 && k!=len-1)continue;
            for(int z=0;z<2;++z){unpack(a,out.data()+((size_t)z*npts+b*block+k)*nw,nw);bad+=mpz_cmp(a,x[(k==0?0:2)+z])!=0;++checked;}
        }
    }
    printf("giant_probe: nw=%d width=%d npts=%d chain=%d threads=%d rotate=%d ms=%.6f checked=%zu bad=%zu\n",nw,width,npts,block,threads,rotate,ms/3,checked,bad);
    CK(cudaFree(di));CK(cudaFree(dout));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));gmp_randclear(rng);mpz_clears(n,a,r,rinv,t,u,nullptr);for(auto &v:x)mpz_clear(v);for(auto &v:y)mpz_clear(v);return bad?1:0;
}
