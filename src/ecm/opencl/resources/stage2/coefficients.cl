// Canonical ordinary residues; one work-item owns a whole coefficient.
// All layouts use explicit limb offsets/strides. No device pointers in metadata.
__kernel void ecm_poly_view(__global const ulong *src,ulong offset,ulong stride,ulong count,
    __global ulong *out,ulong out_count,ulong first,uint reverse) {
    const ulong k=get_global_id(0),s=get_global_id(1);if(k>=out_count)return;
    const ulong at=reverse?(k<=first?first-k:count):(first+k);
    for(int w=0;w<ECM_COEFF_NW;++w)
        out[(s*out_count+k)*ECM_COEFF_NW+w]=at<count?src[offset+s*stride+at*ECM_COEFF_NW+w]:0;
}
__kernel void ecm_poly_sub(__global const ulong *a,ulong ao,ulong ast,ulong na,
    __global const ulong *b,ulong bo,ulong bst,ulong nb,__global const ulong *modulus,
    __global ulong *out,ulong count) {
    const ulong k=get_global_id(0),s=get_global_id(1);if(k>=count)return;
    ulong borrow=0;
    for(int w=0;w<ECM_COEFF_NW;++w) {
        const ulong x=k<na?a[ao+s*ast+k*ECM_COEFF_NW+w]:0;
        const ulong y=k<nb?b[bo+s*bst+k*ECM_COEFF_NW+w]:0;
        const ulong t=x-y,v=t-borrow;borrow=(x<y)||(t<borrow);out[(s*count+k)*ECM_COEFF_NW+w]=v;
    }
    if(borrow) {
        ulong carry=0;for(int w=0;w<ECM_COEFF_NW;++w) {
            const ulong i=(s*count+k)*ECM_COEFF_NW+w,x=out[i],t=x+modulus[w],v=t+carry;
            carry=(t<x)||(v<t);out[i]=v;
        }
    }
}
__kernel void ecm_poly_linear(__global const ulong *roots,ulong offset,ulong count,
    __global const ulong *modulus,__global ulong *out) {
    const ulong k=get_global_id(0);if(k>=count)return;
    ulong any=0;for(int w=0;w<ECM_COEFF_NW;++w)any|=roots[offset+k*ECM_COEFF_NW+w];
    ulong borrow=0;
    for(int w=0;w<ECM_COEFF_NW;++w) {
        const ulong x=modulus[w],y=roots[offset+k*ECM_COEFF_NW+w],t=x-y,v=t-borrow;
        borrow=(x<y)||(t<borrow);out[(2*k)*ECM_COEFF_NW+w]=any?v:0;
        out[(2*k+1)*ECM_COEFF_NW+w]=w==0?1:0;
    }
}
