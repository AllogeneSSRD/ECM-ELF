#pragma once
#include "../../../../src/ecm/backend/cuda/stage2/launch.h"
// Experimental rotation and unregistered 160/192-word controls only.
void ecm_experiment_giant_launch(int width,int rotate,EcmCudaLaunch cfg,
    const unsigned long long *n,unsigned long long ninv,int nw,
    const unsigned long long *dx,const unsigned long long *dz,
    const unsigned long long *sx,const unsigned long long *sz,
    const unsigned long long *ex,const unsigned long long *ez,
    unsigned long long npts,unsigned long long block,unsigned long long count,
    unsigned long long *ox,unsigned long long *oz,size_t stride);
