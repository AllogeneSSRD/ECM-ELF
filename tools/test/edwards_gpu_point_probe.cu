// GPU Edwards Stage1 scalar path versus the existing CPU Atkin-Morain path.
// Build with tools/build/test/build_edwards_gpu_point_probe.ps1.
#include <gmp.h>  // CGBN host headers require GMP first.
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.h"
#include "../../src/ecm/stage1/edwards/cuda/point.cuh"

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"CUDA %s at line %d\n",cudaGetErrorString(e),__LINE__); exit(2); \
} } while(0)

template<uint32_t TPI_, uint32_t BITS_>
struct probe_params {
    static const uint32_t TPI=TPI_, BITS=BITS_, TPB=128, MAX_ROTATION=1, SHM_LIMIT=0;
    static const bool CONSTANT_TIME=false;
};

template<class Params>
__global__ void edwards_scalar_kernel(cgbn_error_report_t *report,
        cgbn_mem_t<Params::BITS> *data, const uint32_t *scalar,
        uint32_t scalar_bits, uint32_t count, uint32_t np0) {
    using context=cgbn_context_t<Params::TPI,Params>;
    using env=cgbn_env_t<context,Params::BITS>;
    using bn=typename env::cgbn_t;
    const uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/Params::TPI;
    if(i>=count)return;
    context ctx(cgbn_report_monitor,report,i);
    env e(ctx);
    auto *slot=&data[6*i];
    bn n,d,one;
    cgbn_edwards_point<env> p,r;
    cgbn_load(e,n,slot+0);
    cgbn_load(e,d,slot+1);
    cgbn_load(e,p.x,slot+2);
    cgbn_load(e,p.y,slot+3);
    cgbn_load(e,p.z,slot+4);
    cgbn_load(e,p.t,slot+5);
    uint32_t actual_np0=cgbn_bn2mont(e,d,d,n);
    if(actual_np0!=np0)ctx.report_error((cgbn_error_t)17);
    cgbn_bn2mont(e,p.x,p.x,n);
    cgbn_bn2mont(e,p.y,p.y,n);
    cgbn_bn2mont(e,p.z,p.z,n);
    cgbn_bn2mont(e,p.t,p.t,n);
    cgbn_set_ui32(e,one,1);
    cgbn_bn2mont(e,one,one,n);
    cgbn_set_ui32(e,r.x,0);
    cgbn_set(e,r.y,one);
    cgbn_set(e,r.z,one);
    cgbn_set_ui32(e,r.t,0);
    cgbn_edwards_field<env> field{e,n,np0};
    for(uint32_t done=0;done<scalar_bits;++done) {
        field.dbl(r,r);
        const uint32_t bit=scalar_bits-1-done;
        if((scalar[bit/32]>>(bit%32))&1u)field.add_point(r,r,p,d);
    }
    cgbn_mont2bn(e,r.x,r.x,n,np0);
    cgbn_mont2bn(e,r.y,r.y,n,np0);
    cgbn_mont2bn(e,r.z,r.z,n,np0);
    cgbn_mont2bn(e,r.t,r.t,n,np0);
    cgbn_store(e,slot+2,r.x);
    cgbn_store(e,slot+3,r.y);
    cgbn_store(e,slot+4,r.z);
    cgbn_store(e,slot+5,r.t);
}

static uint32_t np0_for(uint32_t n0) {
    uint32_t inv=1;
    for(int i=0;i<5;++i)inv*=2u-n0*inv;
    return 0u-inv;
}
template<class Mem>
static void put(Mem &out,const mpz_t x) {
    if(mpz_sgn(x)<0 || mpz_sizeinbase(x,2)>sizeof(out._limbs)*8)abort();
    memset(&out,0,sizeof(out));
    size_t used=0;
    mpz_export(out._limbs,&used,-1,4,0,0,x);
    if(used>sizeof(out._limbs)/4)abort();
}
template<class Mem>
static void get(mpz_t x,const Mem &in) {
    mpz_import(x,sizeof(in._limbs)/4,-1,4,0,0,in._limbs);
}

template<class Params>
static int run_case(unsigned modulus_bits,unsigned b1,unsigned count,uint64_t first_sigma,
                    int device,unsigned cofactor_bits=0) {
    using mem=cgbn_mem_t<Params::BITS>;
    mpz_t n,s,d,px,py,t,fac,cpu_x,cpu_z,gx,gy,gz,gt,lhs,rhs,qx,qz;
    mpz_inits(n,s,d,px,py,t,fac,cpu_x,cpu_z,gx,gy,gz,gt,lhs,rhs,qx,qz,nullptr);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,modulus_bits);mpz_sub_ui(n,n,1);
    if(cofactor_bits) {
        mpz_set_ui(t,1);mpz_mul_2exp(t,t,cofactor_bits);mpz_sub_ui(t,t,1);
        mpz_mul(n,n,t);
    }
    const unsigned target_bits=(unsigned)mpz_sizeinbase(n,2);
    mpz_set_ui(s,1);
    for(unsigned i=2;i<=b1;++i)mpz_lcm_ui(s,s,i);
    mpz_mul_ui(s,s,48);
    const uint32_t sbits=(uint32_t)mpz_sizeinbase(s,2);
    std::vector<uint32_t> bits((sbits+31)/32,0);
    size_t words=0;mpz_export(bits.data(),&words,-1,4,0,0,s);
    const unsigned ipb=Params::TPB/Params::TPI;
    const unsigned total=((count+ipb-1)/ipb)*ipb;
    std::vector<mem> host(6*total),initial(6*total);
    std::vector<uint64_t> sigmas(count);
    for(unsigned i=0;i<count;++i) {
        sigmas[i]=first_sigma+i*19;
        edwards_atkin_morain(d,px,py,sigmas[i],n);
        put(host[6*i+0],n);put(host[6*i+1],d);
        put(host[6*i+2],px);put(host[6*i+3],py);
        mpz_set_ui(t,1);put(host[6*i+4],t);
        mpz_mul(t,px,py);mpz_mod(t,t,n);put(host[6*i+5],t);
    }
    initial=host;
    CU(cudaSetDevice(device));
    mem *gpu=nullptr;uint32_t *gpu_bits=nullptr;cgbn_error_report_t *report=nullptr;
    CU(cudaMalloc(&gpu,host.size()*sizeof(mem)));
    CU(cudaMalloc(&gpu_bits,bits.size()*sizeof(uint32_t)));
    CU(cudaMemcpy(gpu,host.data(),host.size()*sizeof(mem),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(gpu_bits,bits.data(),bits.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CU(cgbn_error_report_alloc(&report));
    edwards_scalar_kernel<Params><<<(total/ipb),Params::TPB>>>(report,gpu,gpu_bits,sbits,count,np0_for(host[0]._limbs[0]));
    CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    if(cgbn_error_report_check(report)) {fprintf(stderr,"CGBN arithmetic error\n");return 2;}
    CU(cudaMemcpy(host.data(),gpu,host.size()*sizeof(mem),cudaMemcpyDeviceToHost));
    unsigned bad=0;
    for(unsigned i=0;i<count;++i) {
        get(gx,host[6*i+2]);get(gy,host[6*i+3]);
        get(gz,host[6*i+4]);get(gt,host[6*i+5]);
        // Extended-coordinate and curve invariants.
        mpz_mul(lhs,gx,gy);mpz_mod(lhs,lhs,n);
        mpz_mul(rhs,gz,gt);mpz_mod(rhs,rhs,n);
        bool valid=mpz_cmp(lhs,rhs)==0;
        get(d,initial[6*i+1]);
        mpz_mul(lhs,gx,gx);mpz_mul(t,gy,gy);mpz_add(lhs,lhs,t);mpz_mod(lhs,lhs,n);
        mpz_mul(rhs,gz,gz);mpz_mul(t,gt,gt);mpz_mul(t,t,d);mpz_add(rhs,rhs,t);mpz_mod(rhs,rhs,n);
        valid=valid && mpz_cmp(lhs,rhs)==0;
        mpz_add(qx,gz,gy);mpz_mod(qx,qx,n);
        mpz_sub(qz,gz,gy);mpz_mod(qz,qz,n);
        int rc=edwards_stage1_curve(fac,cpu_x,cpu_z,n,sigmas[i],s);
        if(rc<0)valid=false;
        valid=valid && mpz_cmp_ui(qz,0)!=0 && mpz_cmp_ui(cpu_z,0)!=0;
        mpz_mul(lhs,qx,cpu_z);mpz_mod(lhs,lhs,n);
        mpz_mul(rhs,cpu_x,qz);mpz_mod(rhs,rhs,n);
        valid=valid && mpz_cmp(lhs,rhs)==0;
        if(!valid) {fprintf(stderr,"n%u B1=%u sigma=%llu mismatch\n",target_bits,b1,(unsigned long long)sigmas[i]);++bad;}
    }
    printf("edwards_gpu: device=%d bits=%u container=%u B1=%u scalar_bits=%u curves=%u bad=%u\n",
           device,target_bits,Params::BITS,b1,sbits,count,bad);
    cgbn_error_report_free(report);CU(cudaFree(gpu));CU(cudaFree(gpu_bits));
    mpz_clears(n,s,d,px,py,t,fac,cpu_x,cpu_z,gx,gy,gz,gt,lhs,rhs,qx,qz,nullptr);
    return bad?1:0;
}

int main(int argc,char **argv) {
    const int device=argc>1?atoi(argv[1]):1;
    if(argc>2) {fprintf(stderr,"usage: edwards_gpu_point_probe [device]\n");return 2;}
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
    printf("Edwards GPU point probe: %s\n",prop.name);
    int bad=0;
    bad+=run_case<probe_params<4,128>>(127,20,4,17,device);
    bad+=run_case<probe_params<4,256>>(127,20,4,17,device,89);
    bad+=run_case<probe_params<4,384>>(347,1000,4,20260922,device);
    bad+=run_case<probe_params<8,768>>(521,1000,4,26,device);
    bad+=run_case<probe_params<8,768>>(677,1000,4,20260922,device);
    return bad?1:0;
}
