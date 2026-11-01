#include <gmp.h>
#include <ecm/cuda/cgbn.h>
#include "../point_coop_ladder_launch.h"
#include "../point_coop_plan.h"
#include "point_coop_ladder.cuh"

void ecm_launch_s2g_coop_ladder_kernel(EcmCudaLaunch cfg,const unsigned long long *n,
    unsigned long long ninv,int nw,const unsigned long long *qx,const unsigned long long *qz,
    const unsigned long long *a24,const unsigned long long *mone,const unsigned long long *scalars,
    int points,int mersenne_bits,bool normal_output,unsigned long long *out_x,unsigned long long *out_z)
{
#define S2G_LADDER_RUN(W) stage2_coop_ladder<W,32><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>( \
    n,ninv,nw,qx,qz,a24,mone,scalars,points,mersenne_bits,normal_output,out_x,out_z)
    switch(s2g_coop_capacity_words(nw,32)) {
        case 160: S2G_LADDER_RUN(160);break;
        case 192: S2G_LADDER_RUN(192);break;
        case 224: S2G_LADDER_RUN(224);break;
        default: S2G_LADDER_RUN(256);break;
    }
#undef S2G_LADDER_RUN
}
