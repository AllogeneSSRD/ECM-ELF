#pragma once
#include "launch.h"

// Fixed TPI32, capacity selected by the same finite plan as cooperative chains.
void ecm_launch_s2g_coop_ladder_kernel(EcmCudaLaunch cfg,const unsigned long long *n,
    unsigned long long ninv,int nw,const unsigned long long *qx,const unsigned long long *qz,
    const unsigned long long *a24,const unsigned long long *mone,const unsigned long long *scalars,
    int points,int mersenne_bits,bool normal_output,unsigned long long *out_x,unsigned long long *out_z);
