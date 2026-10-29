// Native OpenCL C 1.2. No CUDA translation, subgroup assumptions or 64-bit atomics.
// ECM_S4_NW, ECM_S4_BITS, ECM_S4_MERSENNE and ECM_CARRY_GROUP are plan constants.

__kernel void ecm_poly_pack(__global const ulong *input,ulong offset,ulong stride,
    ulong count,__global ulong *digits,ulong n,ulong slot_words,uint bpw) {
    const ulong i=get_global_id(0),s=get_global_id(1);if(i>=n)return;
    const ulong c=i/slot_words,bit=(i%slot_words)*bpw;
    ulong v=0;
    if(c<count && bit<ECM_S4_BITS) {
        const ulong j=bit/64;const uint shift=(uint)(bit%64);
        const ulong base=offset+s*stride+c*ECM_S4_NW;
        v=input[base+j]>>shift;
        if(shift && j+1<ECM_S4_NW)v|=input[base+j+1]<<(64-shift);
        v&=(((ulong)1<<bpw)-1);
    }
    digits[s*n+i]=v;
}
__kernel void ecm_poly_inputs(__global const ulong *a,ulong ao,ulong ast,ulong na,
    __global const ulong *b,ulong bo,ulong bst,ulong nb,__global ulong *flags,ulong fs) {
    const ulong i=get_global_id(0),s=get_global_id(1);if(i>=na+nb)return;
    const ulong top=i<na?a[ao+s*ast+(i+1)*ECM_S4_NW-1]:b[bo+s*bst+(i-na+1)*ECM_S4_NW-1];
#if ECM_S4_BITS % 64
    flags[s*fs+i]=(top>>(ECM_S4_BITS%64))!=0;
#else
    flags[s*fs+i]=0;
#endif
}

// R=ceil(64/bpw) cone rounds reduce any 64-bit coefficient to x in [0,base].
// x==base generates, x==base-1 propagates, smaller x kills the final binary carry.
inline ulong ecm_carry_cone(__global const ulong *c,ulong i,ulong n,uint bpw) {
    ulong v[65];const uint rounds=(64+bpw-1)/bpw;const ulong mask=((ulong)1<<bpw)-1;
    for(uint j=0;j<=rounds;++j)v[j]=i>=j && i-j<n?c[i-j]:0;
    for(uint r=1;r<=rounds;++r)for(uint j=0;j+r<=rounds;++j)v[j]=(v[j]&mask)+(v[j+1]>>bpw);
    return v[0];
}
__kernel void ecm_carry_local(__global const ulong *c,__global ulong *scratch,
    __global ulong *blocks,ulong n,ulong nb,uint bpw) {
    const ulong i=get_global_id(0),s=get_global_id(1),g=get_group_id(0);
    const uint lane=get_local_id(0);const ulong mask=((ulong)1<<bpw)-1;
    const ulong x=ecm_carry_cone(c+s*n,i,n,bpw);
    __local uint scan[ECM_CARRY_GROUP];
    scan[lane]=x==mask?1:(x>mask?2:0);barrier(CLK_LOCAL_MEM_FENCE);
    for(uint d=1;d<ECM_CARRY_GROUP;d*=2) {
        const uint prev=lane>=d?scan[lane-d]:1,own=scan[lane];barrier(CLK_LOCAL_MEM_FENCE);
        if(lane>=d)scan[lane]=own==1?prev:own;barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint exclusive=lane?scan[lane-1]:1;
    scratch[s*n+i]=x|((ulong)exclusive<<33);
    if(lane==ECM_CARRY_GROUP-1)blocks[s*3*nb+g]=scan[lane];
}
__kernel void ecm_carry_blocks(__global ulong *blocks,ulong nb,
    __global const ulong *c,ulong n,uint bpw) {
    const ulong s=get_global_id(0);ulong incoming=0;
    for(ulong g=0;g<nb;++g) {
        const ulong state=blocks[s*3*nb+g];blocks[s*3*nb+nb+g]=incoming;
        if(state!=1)incoming=state==2;
    }
    // Cone rounds can move high bits beyond the retained interval. Check its
    // zero extension, including the remaining binary carry from the last digit.
    const ulong mask=((ulong)1<<bpw)-1;ulong carry=incoming,bad=0;
    for(uint j=0;j<=(64+bpw-1)/bpw;++j) {
        const ulong v=ecm_carry_cone(c+s*n,n+j,n,bpw)+carry;bad|=(v&mask)!=0;carry=v>>bpw;
    }
    blocks[s*3*nb+3*nb-1]=bad || carry;
}
__kernel void ecm_carry_finish(__global const ulong *scratch,__global ulong *digits,
    __global ulong *blocks,ulong n,ulong nb,ulong active,uint bpw) {
    const ulong i=get_global_id(0),s=get_global_id(1),g=get_group_id(0);
    const uint lane=get_local_id(0);const ulong mask=((ulong)1<<bpw)-1;
    const ulong packed=scratch[s*n+i],prefix=packed>>33;
    const ulong incoming=prefix==1?blocks[s*3*nb+nb+g]:(prefix==2);
    const ulong v=(packed&0x1ffffffffUL)+incoming,digit=v&mask;
    digits[s*n+i]=digit;
    __local uint bad[ECM_CARRY_GROUP];
    bad[lane]=(i==n-1 && blocks[s*3*nb+3*nb-1]!=0)||(i>=active && digit!=0);barrier(CLK_LOCAL_MEM_FENCE);
    for(uint d=ECM_CARRY_GROUP/2;d;d/=2) {
        if(lane<d)bad[lane]|=bad[lane+d];barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(!lane)blocks[s*3*nb+2*nb+g]=bad[0];
}

inline ulong ecm_s4_quotient(ulong u1,ulong u0,ulong d,ulong reciprocal) {
    ulong q=ecm_gl_product_hi(reciprocal,u1)+u1;
    for(int guard=0;guard<8;++guard) {
        const ulong lo=q*d,hi=ecm_gl_product_hi(q,d),rlo=u0-lo,borrow=u0<lo;
        const ulong rhi=u1-hi-borrow;
        if(u1<hi || (u1==hi && borrow)){--q;continue;}
        if(rhi || rlo>=d){++q;continue;}return q;
    }
    return q;
}
inline int ecm_s4_div(__private ulong *t,int limbs,__global const ulong *ns,
    int shift,ulong reciprocal,__private ulong *out) {
    for(int i=0;i<ECM_S4_NW;++i)out[i]=0;
    if(shift) {
        ulong carry=0;for(int i=0;i<limbs;++i){const ulong v=t[i];t[i]=(v<<shift)|carry;carry=v>>(64-shift);}t[limbs++]=carry;
    }
    while(limbs>0 && !t[limbs-1])--limbs;t[limbs]=0;
    const ulong top=ns[ECM_S4_NW-1];int repairs=0;
    for(int j=limbs-ECM_S4_NW;j>=0;--j) {
        const ulong u1=t[j+ECM_S4_NW],u0=t[j+ECM_S4_NW-1];if(u1>top)return -1;
        const ulong q=u1==top?~(ulong)0:ecm_s4_quotient(u1,u0,top,reciprocal);
        ulong carry=0;
        for(int i=0;i<ECM_S4_NW;++i) {
            const ulong lo=q*ns[i];ulong hi=ecm_gl_product_hi(q,ns[i]);
            const ulong sub=lo+carry;hi+=(sub<lo);const ulong v=t[j+i];t[j+i]=v-sub;carry=hi+(v<sub);
        }
        int under=t[j+ECM_S4_NW]<carry;t[j+ECM_S4_NW]-=carry;
        for(int correction=0;under && correction<2;++correction) {
            ulong c=0;for(int i=0;i<ECM_S4_NW;++i){const ulong v=t[j+i],sum=v+ns[i],next=sum+c;c=(sum<v)||(next<sum);t[j+i]=next;}
            const ulong v=t[j+ECM_S4_NW];t[j+ECM_S4_NW]=v+c;under=t[j+ECM_S4_NW]>=v;++repairs;
        }
        if(under)return -1;
    }
    for(int i=0;i<ECM_S4_NW;++i)out[i]=shift?((t[i]>>shift)|(i+1<ECM_S4_NW?t[i+1]<<(64-shift):0)):t[i];
    return repairs;
}
inline void ecm_s4_mersenne(__private ulong *t,int live,__private ulong *out) {
    const int q=ECM_S4_BITS/64,b=ECM_S4_BITS%64;
    const ulong mask=b?(((ulong)1<<b)-1):~(ulong)0;
    while(live>0 && !t[live-1])--live;
    while(live>ECM_S4_NW
#if ECM_S4_BITS % 64
        || (live==ECM_S4_NW && (t[ECM_S4_NW-1]>>b))
#endif
    ) {
        const int len=max(live-q,ECM_S4_NW);ulong carry=0;
        for(int i=0;i<len;++i) {
            const int src=i+q;const ulong h0=src<live?t[src]:0,h1=(b && src+1<live)?t[src+1]:0;
            const ulong hi=b?((h0>>b)|(h1<<((64-b)&63))):h0;
            const ulong lo=i<ECM_S4_NW?(i==ECM_S4_NW-1?t[i]&mask:t[i]):0;
            const ulong sum=lo+hi,next=sum+carry;carry=(sum<lo)||(next<sum);t[i]=next;
        }
        live=len;if(carry)t[live++]=carry;while(live>0 && !t[live-1])--live;
    }
    int equal=1;for(int i=0;i<ECM_S4_NW;++i){out[i]=i<live?t[i]:0;if(out[i]!=(i==ECM_S4_NW-1?mask:~(ulong)0))equal=0;}
    if(equal)for(int i=0;i<ECM_S4_NW;++i)out[i]=0;
}
__kernel void ecm_poly_reduce(__global const ulong *digits,ulong n,uint bpw,
    ulong slot_words,ulong slot_bits,ulong count,__global const ulong *modulus,
    __global const ulong *normalized,int shift,ulong reciprocal,
    __global ulong *out,ulong offset,ulong stride,__global ulong *flags,ulong fs,ulong flag_offset) {
    const ulong k=get_global_id(0),s=get_global_id(1);if(k>=count)return;
    __global const ulong *d=digits+s*n+k*slot_words;
    const uint top=(uint)(slot_bits-(slot_words-1)*bpw);ulong bad=(d[slot_words-1]>>top)!=0;
    ulong t[2*ECM_S4_NW+4],u[ECM_S4_NW];for(int i=0;i<2*ECM_S4_NW+4;++i)t[i]=0;
    ulong acc=0;uint bits=0;int limb=0;
    for(ulong j=0;j<slot_words;++j) {
        const ulong v=d[j];acc|=v<<bits;
        if(bits+bpw>=64){t[limb++]=acc;acc=bits?v>>(64-bits):0;bits=bits+bpw-64;}else bits+=bpw;
    }
    if(bits)t[limb++]=acc;
#if ECM_S4_MERSENNE
    ecm_s4_mersenne(t,limb,u);
#else
    if(ecm_s4_div(t,limb,normalized,shift,reciprocal,u)<0)bad|=2;
#endif
    int cmp=0;for(int i=ECM_S4_NW-1;i>=0;--i)if(!cmp && u[i]!=modulus[i])cmp=u[i]<modulus[i]?-1:1;
    if(cmp>=0)bad|=4;
    for(int i=0;i<ECM_S4_NW;++i)out[offset+s*stride+k*ECM_S4_NW+i]=u[i];
    flags[s*fs+flag_offset+k]=bad;
}
