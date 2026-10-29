// Component experiment only. Production normalization kernels are included
// directly; the bounded binary GCD below is an experimental CUDA baseline.
// Neither path changes production dispatch or factor recovery.
#include "ecm/stage2/cuda/device/point.cuh"
#include "ecm/stage2/cuda/device/baby_giant.cuh"
#include "opencl_stage2_oracle.h"
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

namespace {
using namespace stage2_oracle;
using U = unsigned long long;
void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    std::vector<void*> memory;
    ~Device(){cudaDeviceSynchronize();for(auto p:memory)cudaFree(p);}
    template<class T>T* make(size_t count,const T*src=nullptr){T*p=nullptr;ck(cudaMalloc(&p,count*sizeof(T)));memory.push_back(p);if(src)ck(cudaMemcpy(p,src,count*sizeof(T),cudaMemcpyHostToDevice));return p;}
};
struct Events {
    cudaEvent_t a,b;
    Events(){ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));}
    ~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}
    void start(){ck(cudaEventRecord(a));}
    double stop(){ck(cudaGetLastError());ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));return ms*.001;}
};
std::vector<U> words(const Big&v,size_t nw){std::vector<U> w(nw);size_t n=0;mpz_export(w.data(),&n,-1,8,0,0,v.v);return w;}
void put(U*out,const Big&v,size_t nw){std::fill(out,out+nw,0);size_t n=0;mpz_export(out,&n,-1,8,0,0,v.v);}
void get(Big&v,const U*in,size_t nw){mpz_import(v.v,nw,-1,8,0,0,in);}
Big modulus(int bits,bool composite){static std::map<std::pair<int,bool>,Big> cache;auto key=std::make_pair(bits,composite);auto found=cache.find(key);if(found!=cache.end())return found->second;Big n;mpz_setbit(n.v,bits-1);mpz_add_ui(n.v,n.v,1337);if(composite)mpz_cdiv_q_ui(n.v,n.v,257);mpz_nextprime(n.v,n.v);if(composite)mpz_mul_ui(n.v,n.v,257);cache.emplace(key,n);return n;}
std::vector<U> random_values(int count,int nw,const Big&n,unsigned seed,bool nonzero){std::mt19937_64 r(seed);std::vector<U> out(size_t(count)*nw);Big x;for(int k=0;k<count;++k){for(int j=0;j<nw;++j)out[size_t(k)*nw+j]=r();get(x,out.data()+size_t(k)*nw,nw);mpz_mod(x.v,x.v,n.v);if(nonzero&&!mpz_sgn(x.v))mpz_set_ui(x.v,1);put(out.data()+size_t(k)*nw,x,nw);}return out;}
void require_equal(const std::vector<U>&a,const std::vector<U>&b){if(a!=b)throw std::runtime_error("GMP/CUDA output mismatch");}

// Batch-256 Montgomery trick on the host. X and Z share the same radix,
// so X_M/Z_M equals X/Z without two explicit REDC conversions per point.
void cpu_normalize(const std::vector<U>&x,const std::vector<U>&z,int nw,const Big&n,std::vector<U>&out){
    Poly prefix(257),den(256);Big inv,t,v;const size_t count=x.size()/nw;
    for(size_t lo=0;lo<count;lo+=256){const size_t m=std::min(size_t(256),count-lo);mpz_set_ui(prefix[0].v,1);
        for(size_t j=0;j<m;++j){get(den[j],z.data()+(lo+j)*nw,nw);mpz_mul(prefix[j+1].v,prefix[j].v,den[j].v);mpz_mod(prefix[j+1].v,prefix[j+1].v,n.v);}
        if(!mpz_invert(inv.v,prefix[m].v,n.v))throw std::runtime_error("normalization fixture has nonunit Z");
        for(size_t j=m;j-->0;){mpz_mul(t.v,inv.v,prefix[j].v);mpz_mod(t.v,t.v,n.v);mpz_mul(inv.v,inv.v,den[j].v);mpz_mod(inv.v,inv.v,n.v);get(v,x.data()+(lo+j)*nw,nw);mpz_mul(v.v,v.v,t.v);mpz_neg(v.v,v.v);mpz_mod(v.v,v.v,n.v);put(out.data()+(lo+j)*nw,v,nw);}
    }
}
template<int NW>__global__ void binary_gcd(const U*in,const U*n,int nw,int count,U*out,unsigned char*bad){
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
    U a[NW],b[NW];for(int j=0;j<nw;++j){a[j]=in[size_t(k)*nw+j];b[j]=n[j];}
    bool done=false;
    // N is odd, so removing powers of two from a preserves gcd(a,N).
    // Each subtraction/shift reduces one operand; the guard bounds failures.
    for(int step=0;step<8*64*nw+16;++step){bool zero=true;for(int j=0;j<nw;++j)zero=zero&&a[j]==0;if(zero){done=true;break;}
        if(!(a[0]&1)){for(int j=0;j<nw;++j)a[j]=(a[j]>>1)|(j+1<nw?a[j+1]<<63:0);continue;}
        if(s2g_ge_n(a,b,nw)){s2g_sub_n(a,b,nw);}else{for(int j=0;j<nw;++j){const U t=a[j];a[j]=b[j];b[j]=t;}s2g_sub_n(a,b,nw);}
    }
    bad[k]=done?0:1;for(int j=0;j<nw;++j)out[size_t(k)*nw+j]=b[j];
}
struct Sample {double wall=0,kernel=0,inverse=0;};
template<int NW>void norm_case(int bits,int count,int repeats){
    const int nw=(bits+63)/64;const auto n=modulus(bits,false);auto hn=words(n,nw);Big radix;mpz_setbit(radix.v,64*nw);mpz_mod(radix.v,radix.v,n.v);
    auto hx=random_values(count,nw,n,17,true),hz=random_values(count,nw,n,29,true);Big v;
    for(auto*w:{&hx,&hz})for(int k=0;k<count;++k){get(v,w->data()+size_t(k)*nw,nw);mpz_mul(v.v,v.v,radix.v);mpz_mod(v.v,v.v,n.v);put(w->data()+size_t(k)*nw,v,nw);}
    U ninv=1;for(int i=0;i<6;++i)ninv*=2-hn[0]*ninv;ninv=0-ninv;
    Device d;auto dn=d.make<U>(nw,hn.data()),dx=d.make<U>(hx.size(),hx.data()),dz=d.make<U>(hz.size(),hz.data()),out=d.make<U>(hx.size());
    size_t counts[9],offset[9]={},total=0;counts[0]=count;for(int l=1;l<=8;++l){counts[l]=(counts[l-1]+1)/2;offset[l]=total;total+=counts[l]*nw;}
    auto tree=d.make<U>(total);const auto groups=counts[8];auto mask=d.make<unsigned char>(groups);std::vector<unsigned char> good(groups,1);std::vector<U> roots(groups*nw),cx(hx.size()),cz(hz.size()),result(hx.size()),expected(hx.size());
    // Independent pointwise GMP oracle outside timing; no batch inversion reuse.
    Big x,z,t;for(int k=0;k<count;++k){get(x,hx.data()+size_t(k)*nw,nw);get(z,hz.data()+size_t(k)*nw,nw);if(!mpz_invert(t.v,z.v,n.v))throw std::runtime_error("oracle nonunit");mpz_mul(x.v,x.v,t.v);mpz_neg(x.v,x.v);mpz_mod(x.v,x.v,n.v);put(expected.data()+size_t(k)*nw,x,nw);}
    Events ev;auto run=[&](bool gpu){Sample s;const double begin=now();if(!gpu){ck(cudaMemcpy(cx.data(),dx,cx.size()*8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(cz.data(),dz,cz.size()*8,cudaMemcpyDeviceToHost));cpu_normalize(cx,cz,nw,n,result);}else{
        ev.start();for(int l=1;l<=8;++l){const U*in=l==1?dz:tree+offset[l-1];ecm_launch_s2g_baby_product_kernel<NW>(EcmCudaLaunch{unsigned((counts[l-1]+127)/128),64},in,counts[l-1],tree+offset[l],dn,ninv,nw);}s.kernel+=ev.stop();
        ck(cudaMemcpy(roots.data(),tree+offset[8],roots.size()*8,cudaMemcpyDeviceToHost));const double ti=now();
        for(size_t k=0;k<groups;++k){get(t,roots.data()+k*nw,nw);if(!mpz_invert(t.v,t.v,n.v))throw std::runtime_error("CUDA root nonunit");mpz_mul(t.v,t.v,radix.v);mpz_mod(t.v,t.v,n.v);put(roots.data()+k*nw,t,nw);}s.inverse=now()-ti;
        ck(cudaMemcpy(tree+offset[8],roots.data(),roots.size()*8,cudaMemcpyHostToDevice));ck(cudaMemcpy(mask,good.data(),groups,cudaMemcpyHostToDevice));ev.start();
        for(int l=8;l>=2;--l)ecm_launch_s2g_baby_inverse_kernel<NW>(EcmCudaLaunch{unsigned((counts[l-1]+127)/128),64},tree+offset[l],tree+offset[l-1],counts[l-1],mask,size_t(256)>>l,dn,ninv,nw);
        ecm_launch_s2g_baby_leaf_kernel<NW>(EcmCudaLaunch{unsigned((count+127)/128),64},tree+offset[1],dx,dz,count,out,mask,dn,ninv,nw);s.kernel+=ev.stop();ck(cudaMemcpy(result.data(),out,result.size()*8,cudaMemcpyDeviceToHost));
    }s.wall=now()-begin;require_equal(result,expected);return s;};
    for(int rep=-1;rep<repeats;++rep)for(int side=0;side<2;++side){const bool gpu=(side+(rep&1))%2;const auto s=run(gpu);if(rep>=0)std::cout<<"sample,normalize,"<<bits<<','<<count<<','<<(gpu?"cuda_hybrid":"cpu_batch256")<<','<<rep<<','<<s.wall<<','<<s.kernel<<','<<s.inverse<<','<<count/256+(count%256!=0)<<','<<size_t(count)*nw<<'\n';}
}
template<int NW>void gcd_case(int bits,int count,int repeats){
    const int nw=(bits+63)/64;const auto n=modulus(bits,true);auto hn=words(n,nw),values=random_values(count,nw,n,41,false);Big q;mpz_divexact_ui(q.v,n.v,257);
    // Include zero, saturation and both nontrivial factors, in addition to units.
    for(int k=0;k<count;++k)if(count>=16&&k%16<4){Big v;if(k%16==1)mpz_set_ui(v.v,257);if(k%16==2)mpz_set(v.v,q.v);if(k%16==3)mpz_set(v.v,n.v);put(values.data()+size_t(k)*nw,v,nw);}
    Device d;auto dn=d.make<U>(nw,hn.data()),dv=d.make<U>(values.size(),values.data()),dg=d.make<U>(values.size());auto db=d.make<unsigned char>(count);std::vector<unsigned char> bad(count);std::vector<U> host(values.size()),result(values.size()),expected(values.size());Big v,g;
    for(int k=0;k<count;++k){get(v,values.data()+size_t(k)*nw,nw);mpz_gcd(g.v,v.v,n.v);put(expected.data()+size_t(k)*nw,g,nw);}Events ev;
    auto run=[&](bool gpu){Sample s;const double begin=now();if(gpu){ev.start();binary_gcd<NW><<<(count+63)/64,64>>>(dv,dn,nw,count,dg,db);s.kernel=ev.stop();ck(cudaMemcpy(result.data(),dg,result.size()*8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bad.data(),db,count,cudaMemcpyDeviceToHost));if(std::find(bad.begin(),bad.end(),1)!=bad.end())throw std::runtime_error("CUDA GCD step guard");}else{ck(cudaMemcpy(host.data(),dv,host.size()*8,cudaMemcpyDeviceToHost));for(int k=0;k<count;++k){get(v,host.data()+size_t(k)*nw,nw);mpz_gcd(g.v,v.v,n.v);put(result.data()+size_t(k)*nw,g,nw);}}s.wall=now()-begin;require_equal(result,expected);return s;};
    for(int rep=-1;rep<repeats;++rep)for(int side=0;side<2;++side){const bool gpu=(side+(rep&1))%2;const auto s=run(gpu);if(rep>=0)std::cout<<"sample,gcd,"<<bits<<','<<count<<','<<(gpu?"cuda_binary_experimental":"cpu_gmp")<<','<<rep<<','<<s.wall<<','<<s.kernel<<",0,0,"<<size_t(count)*nw<<'\n';}
}
template<int NW>void accum_case(int bits,int count,int repeats){
    const int nw=(bits+63)/64,blocks=(count+63)/64;const auto n=modulus(bits,true);auto hn=words(n,nw),values=random_values(count,nw,n,73,true);
    U ninv=1;for(int i=0;i<6;++i)ninv*=2-hn[0]*ninv;ninv=0-ninv;
    // One nonunit block verifies that dropping Montgomery scale preserves GCD.
    Big q;mpz_set_ui(q.v,257);put(values.data(),q,nw);
    Device d;auto dn=d.make<U>(nw,hn.data()),dv=d.make<U>(values.size()),dp=d.make<U>(size_t(blocks)*nw);std::vector<U> products(size_t(blocks)*nw),result(products.size()),expected(products.size());Big v,p,g;
    auto cpu_blocks=[&](std::vector<U>&out){for(int b=0;b<blocks;++b){mpz_set_ui(p.v,1);for(int k=b*64;k<std::min(count,(b+1)*64);++k){get(v,values.data()+size_t(k)*nw,nw);mpz_mul(p.v,p.v,v.v);mpz_mod(p.v,p.v,n.v);}mpz_gcd(g.v,p.v,n.v);put(out.data()+size_t(b)*nw,g,nw);}};
    cpu_blocks(expected);Events ev;auto run=[&](bool gpu){Sample s;const double begin=now();if(!gpu)cpu_blocks(result);else{
        // Production descent currently returns host leaves, then uploads them
        // for block products. Preserve that boundary and include both copies.
        ck(cudaMemcpy(dv,values.data(),values.size()*8,cudaMemcpyHostToDevice));ev.start();ecm_launch_s2g_block_prod_kernel<NW>(EcmCudaLaunch{unsigned((blocks+63)/64),64},dv,count,nw,64,dn,ninv,dp);s.kernel=ev.stop();
        ck(cudaMemcpy(products.data(),dp,products.size()*8,cudaMemcpyDeviceToHost));for(int b=0;b<blocks;++b){get(p,products.data()+size_t(b)*nw,nw);mpz_gcd(g.v,p.v,n.v);put(result.data()+size_t(b)*nw,g,nw);}}
        s.wall=now()-begin;require_equal(result,expected);return s;};
    for(int rep=-1;rep<repeats;++rep)for(int side=0;side<2;++side){const bool gpu=(side+(rep&1))%2;const auto s=run(gpu);if(rep>=0)std::cout<<"sample,accumulate,"<<bits<<','<<count<<','<<(gpu?"cuda_products_cpu_gcd":"cpu_products_gcd")<<','<<rep<<','<<s.wall<<','<<s.kernel<<",0,"<<blocks<<','<<size_t(blocks)*nw<<'\n';}
}
template<int NW>void suite(int bits,int repeats){for(int count:{1024,8192}){norm_case<NW>(bits,count,repeats);accum_case<NW>(bits,count,repeats);}for(int count:{1,16,128,1024})gcd_case<NW>(bits,count,repeats);}
}
int main(int argc,char**argv){try{int device=argc>1?std::atoi(argv[1]):1,repeats=argc>2?std::atoi(argv[2]):3;if(repeats<1)throw std::runtime_error("repeats must be positive");ck(cudaSetDevice(device));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,device));int driver,runtime;ck(cudaDriverGetVersion(&driver));ck(cudaRuntimeGetVersion(&runtime));std::cout<<std::setprecision(12)<<"# device="<<device<<" name="<<prop.name<<" driver_api="<<driver<<" runtime="<<runtime<<" repeats="<<repeats<<" priming=1 input=resident output=host allocation=excluded normalizer=production_batch256 gcd=experimental_binary\n";
    std::cout<<"record,operation,bits,count,implementation,repeat,wall_seconds,kernel_seconds,cpu_root_inverse_seconds,groups,verified_words\n";
    suite<8>(512,repeats);suite<64>(2203,repeats);suite<128>(4423,repeats);suite<128>(8191,repeats);
    // Odd/partial groups and limb boundaries; correctness only, not performance.
    norm_case<8>(127,259,1);norm_case<8>(512,1,1);gcd_case<8>(65,33,1);
    std::cout<<"# status=passed\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
