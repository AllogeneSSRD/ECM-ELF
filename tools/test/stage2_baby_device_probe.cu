#include <cuda_runtime.h>
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "stage2_baby_arithmetic_fixture.cuh"
#include "../bench/stage2_baby_device.cuh"
#define CK(c) do{auto e=(c);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(3);}}while(0)
#define DISP(w,fn,...) do{if(w<=4)fn<4>(__VA_ARGS__);else if(w<=8)fn<8>(__VA_ARGS__);else if(w<=16)fn<16>(__VA_ARGS__);else if(w<=32)fn<32>(__VA_ARGS__);else if(w<=64)fn<64>(__VA_ARGS__);else fn<128>(__VA_ARGS__);}while(0)
static void store(unsigned long long *out,size_t w,const mpz_t v){std::fill(out,out+w,0);size_t c=0;mpz_export(out,&c,-1,8,0,0,v);}
static void load(mpz_t v,const unsigned long long *in,size_t w){mpz_import(v,w,-1,8,0,0,in);}

static size_t run(size_t w,size_t count,bool mersenne,bool fault)
{
    mpz_t N,R,ri,x,z,v,p,inv,want,tmp;mpz_inits(N,R,ri,x,z,v,p,inv,want,tmp,nullptr);
    mpz_set_ui(N,1);mpz_mul_2exp(N,N,mersenne?4423:64*w-4);mpz_sub_ui(N,N,1);
    if(!mersenne)mpz_mul_ui(N,N,3);
    mpz_set_ui(R,1);mpz_mul_2exp(R,R,64*w);mpz_mod(R,R,N);mpz_invert(ri,R,N);
    std::vector<unsigned long long> hn(w),xs(count*w),zs(count*w),out(count*w),seed;
    store(hn.data(),w,N);unsigned long long ni=1;for(int i=0;i<6;++i)ni*=2-hn[0]*ni;ni=0-ni;
    std::vector<std::vector<unsigned long long>> ordinary_x(count),ordinary_z(count);
    unsigned long long random=17;
    for(size_t i=0;i<count;++i) {
        std::vector<unsigned long long> a(w),b(w);
        for(size_t j=0;j<w;++j){random=random*6364136223846793005ull+1;a[j]=random;random=random*6364136223846793005ull+1;b[j]=random;}
        load(x,a.data(),w);mpz_mod(x,x,N);load(z,b.data(),w);mpz_mod(z,z,N);
        mpz_gcd(v,z,N);while(mpz_cmp_ui(v,1)){mpz_add_ui(z,z,1);mpz_mod(z,z,N);mpz_gcd(v,z,N);}
        if(i%41==0)mpz_set_ui(x,0);else if(i%41==1)mpz_sub_ui(x,N,1);
        if(i==256)mpz_set_ui(z,0);if(i==258 && !mersenne)mpz_set_ui(z,3);
        if(i==512 && !mersenne)mpz_set_ui(z,3);
        ordinary_x[i].resize(w);ordinary_z[i].resize(w);store(ordinary_x[i].data(),w,x);store(ordinary_z[i].data(),w,z);
        mpz_mul(x,x,R);mpz_mod(x,x,N);mpz_mul(z,z,R);mpz_mod(z,z,N);
        store(xs.data()+i*w,w,x);store(zs.data()+i*w,w,z);
    }
    size_t counts[9]={count},offset[9]={},words=0;
    for(int l=1;l<=8;++l){counts[l]=(counts[l-1]+1)/2;offset[l]=words;words+=counts[l]*w;}
    const size_t groups=counts[8];seed.resize(groups*w);std::vector<unsigned char> good(groups);
    unsigned long long *dn,*dx,*dz,*dt,*dout;unsigned char *dg;
    CK(cudaMalloc(&dn,w*8));CK(cudaMalloc(&dx,xs.size()*8));CK(cudaMalloc(&dz,zs.size()*8));
    CK(cudaMalloc(&dt,words*8));CK(cudaMalloc(&dout,out.size()*8));CK(cudaMalloc(&dg,groups));
    CK(cudaMemcpy(dn,hn.data(),w*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,xs.data(),xs.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dz,zs.data(),zs.size()*8,cudaMemcpyHostToDevice));
    for(int l=1;l<=8;++l){DISP(w,s2g_launch_baby_product,(int)w,counts[l-1],l==1?dz:dt+offset[l-1],dt+offset[l],dn,ni);CK(cudaGetLastError());}
    CK(cudaMemcpy(seed.data(),dt+offset[8],seed.size()*8,cudaMemcpyDeviceToHost));
    size_t bad=0;
    for(size_t k=0;k<groups;++k){mpz_set_ui(p,1);
        for(size_t i=k*256;i<std::min(count,(k+1)*256);++i){load(z,ordinary_z[i].data(),w);mpz_mul(p,p,z);mpz_mod(p,p,N);}
        mpz_mul(v,p,R);mpz_mod(v,v,N);load(tmp,seed.data()+k*w,w);
        if(mpz_cmp(v,tmp)){std::fprintf(stderr,"root GMP mismatch\n");std::exit(3);}
        if(mpz_invert(inv,p,N)){good[k]=1;store(seed.data()+k*w,w,inv);}else{++bad;std::fill(seed.begin()+k*w,seed.begin()+(k+1)*w,0);}
    }
    CK(cudaMemcpy(dt+offset[8],seed.data(),seed.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dg,good.data(),groups,cudaMemcpyHostToDevice));
    for(int l=8;l>=2;--l){DISP(w,s2g_launch_baby_inverse,(int)w,counts[l-1],dt+offset[l],dt+offset[l-1],dg,size_t(256)>>l,dn,ni);CK(cudaGetLastError());}
    DISP(w,s2g_launch_baby_leaf,(int)w,count,dt+offset[1],dx,dz,dout,dg,dn,ni);CK(cudaGetLastError());
    CK(cudaMemcpy(out.data(),dout,out.size()*8,cudaMemcpyDeviceToHost));if(fault)out[0]^=1;
    for(size_t i=0;i<count;++i){mpz_set_ui(want,0);
        if(good[i/256]){load(x,ordinary_x[i].data(),w);load(z,ordinary_z[i].data(),w);mpz_invert(inv,z,N);mpz_mul(want,x,inv);mpz_neg(want,want);mpz_mod(want,want,N);}
        load(v,out.data()+i*w,w);if(mpz_cmp(v,want)){std::fprintf(stderr,"FATAL: baby probe affine GMP mismatch w=%llu count=%llu i=%llu\n",(unsigned long long)w,(unsigned long long)count,(unsigned long long)i);std::exit(3);}
    }
    for(auto d:{dn,dx,dz,dt,dout})CK(cudaFree(d));CK(cudaFree(dg));mpz_clears(N,R,ri,x,z,v,p,inv,want,tmp,nullptr);
    std::printf("PASS baby_device w=%llu count=%llu groups=%llu bad_groups=%llu words=%llu\n",(unsigned long long)w,(unsigned long long)count,(unsigned long long)groups,(unsigned long long)bad,(unsigned long long)(count*w));
    return count*w;
}
int main(int argc,char **argv)
{
    if(argc<2)return 2;CK(cudaSetDevice(std::atoi(argv[1])));size_t words=0,cases=0;
    for(size_t w:{size_t(1),size_t(2),size_t(3),size_t(4),size_t(8),size_t(16),size_t(32),size_t(64),size_t(70),size_t(83),size_t(128)})
        for(size_t n:{size_t(1),size_t(257),size_t(513)}){words+=run(w,n,false,argc>2);++cases;}
    for(size_t n:{size_t(2),size_t(3),size_t(127),size_t(255),size_t(256),size_t(511),size_t(768),size_t(1025)}){words+=run(3,n,false,false);++cases;}
    words+=run(70,769,true,false);++cases;
    std::printf("TOTAL cases=%llu words=%llu bad=0\n",(unsigned long long)cases,(unsigned long long)words);return 0;
}
