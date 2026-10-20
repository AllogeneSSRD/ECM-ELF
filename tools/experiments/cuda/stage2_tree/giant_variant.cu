#include "giant_launch.h"
#include <cstdlib>
// This TU owns a private, generic-modulus point constant. It never reads or
// uploads the production arithmetic TU's constant.
namespace ecm_stage2_giant_experiment {
#include "giant_point.cuh"
#include "giant_chain_variant.cuh"
template<int NW> static void launch(int rotate,EcmCudaLaunch cfg,
    const unsigned long long *n,unsigned long long ninv,int nw,
    const unsigned long long *dx,const unsigned long long *dz,
    const unsigned long long *sx,const unsigned long long *sz,
    const unsigned long long *ex,const unsigned long long *ez,
    unsigned long long npts,unsigned long long block,unsigned long long count,
    unsigned long long *ox,unsigned long long *oz,size_t stride) {
    if constexpr(NW==160 || NW==192) {
        if(!rotate) {
            s2g_chain_kernel<NW,true><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);
            return;
        }
    }
    s2g_chain_rotate_kernel<NW,true,true><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);
}
}
void ecm_experiment_giant_launch(int width,int rotate,EcmCudaLaunch cfg,
    const unsigned long long *n,unsigned long long ninv,int nw,
    const unsigned long long *dx,const unsigned long long *dz,
    const unsigned long long *sx,const unsigned long long *sz,
    const unsigned long long *ex,const unsigned long long *ez,
    unsigned long long npts,unsigned long long block,unsigned long long count,
    unsigned long long *ox,unsigned long long *oz,size_t stride) {
    switch(width) {
    case 64:ecm_stage2_giant_experiment::launch<64>(rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);return;
    case 128:ecm_stage2_giant_experiment::launch<128>(rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);return;
    case 160:ecm_stage2_giant_experiment::launch<160>(rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);return;
    case 192:ecm_stage2_giant_experiment::launch<192>(rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);return;
    case 256:ecm_stage2_giant_experiment::launch<256>(rotate,cfg,n,ninv,nw,dx,dz,sx,sz,ex,ez,npts,block,count,ox,oz,stride);return;
    default:std::abort();
    }
}
