#pragma once
#include <cuda_runtime.h>
namespace stage2_pack {
__device__ __forceinline__ unsigned quotient(unsigned j,unsigned sw,unsigned reciprocal,unsigned &digit) {
    if(sw==1){digit=0;return j;}
    unsigned q=__umulhi(j,reciprocal),r=j-q*sw;
    const unsigned correction=r>=sw;
    digit=r-correction*sw;return q+correction;
}
}
