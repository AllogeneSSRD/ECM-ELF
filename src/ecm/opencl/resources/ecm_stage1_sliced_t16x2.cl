// ECM Stage 1 — sliced cooperative kernel T16×2 (AMD-only).
// NVIDIA: stub only.
#if defined(__AMDGCN__)
__kernel __attribute__((reqd_work_group_size(16u,1u,1u)))
void kernel_double_add_sliced_t16x2(__global const uint*sb,ulong sn,ulong st,ulong si,__global uint*d,uint cnt,uint sig,uint NP0,uint lim){
    if(lim!=32u)return;
    const uint lid=get_local_id(0), ins=get_group_id(0);
    if(ins>=cnt)return;
    uint bs=ins*160u;
    __local uint Ld[192];
    uint off0=lid*2u,off1=lid*2u+1u;
    uint N0=d[bs+off0],N1=d[bs+off1];
    uint X0=d[bs+32u+off0],X1=d[bs+32u+off1];
    uint Z0=d[bs+64u+off0],Z1=d[bs+64u+off1];
    uint W0=d[bs+96u+off0],W1=d[bs+96u+off1];
    uint V0=d[bs+128u+off0],V1=d[bs+128u+off1];
    Ld[off0]=N0;Ld[off1]=N1;Ld[32u+off0]=X0;Ld[32u+off1]=X1;
    Ld[64u+off0]=Z0;Ld[64u+off1]=Z1;Ld[96u+off0]=W0;Ld[96u+off1]=W1;
    Ld[128u+off0]=V0;Ld[128u+off1]=V1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid!=0u)return;
    uint N_full[32],X[32],Z[32],W[32],V[32];
    for(uint i=0;i<32;i++){N_full[i]=Ld[i];X[i]=Ld[32+i];Z[i]=Ld[64+i];W[i]=Ld[96+i];V[i]=Ld[128+i];}
    uint dv=sig+ins;int sw=0;ulong se=st+si;if(se>sn)se=sn;
    for(ulong b=st;b<se;b++){
        ulong nth=sn-1u-b;uint li=(uint)(nth>>5),bi=(uint)(nth&31);int bit=(int)((sb[li]>>bi)&1u);
        if(bit!=sw){sw=!sw;for(uint i=0;i<32;i++){uint t=X[i];X[i]=W[i];W[i]=t;t=Z[i];Z[i]=V[i];V[i]=t;}}
        uint t[32],CB[32],DA[32],AA[32],BB[32],K[32],dK[32],qq[32],uu[32],ww[32],vv[32];
        for(uint i=0;i<32;i++){qq[i]=X[i];uu[i]=Z[i];ww[i]=W[i];vv[i]=V[i];}
        add_mod_asm_1024b(t,vv,ww,N_full,32u);(void)sub_mod_asm_1024b(vv,vv,ww,N_full,32u);
        add_mod_asm_1024b(ww,uu,qq,N_full,32u);(void)sub_mod_asm_1024b(uu,uu,qq,N_full,32u);
        mont_mul_unroll_1024b(CB,t,uu,N_full,1u,32u);mont_mul_unroll_1024b(DA,vv,ww,N_full,1u,32u);
        mont_mul_unroll_1024b(AA,ww,ww,N_full,1u,32u);mont_mul_unroll_1024b(BB,uu,uu,N_full,1u,32u);
        mont_mul_unroll_1024b(qq,AA,BB,N_full,1u,32u);
        (void)sub_mod_asm_1024b(K,AA,BB,N_full,32u);for(uint i=0;i<32;i++)dK[i]=K[i];
        special_mult_ui32_unroll_1024b(dK,dv,N_full,1u,32u);
        add_mod_asm_1024b(uu,BB,dK,N_full,32u);mont_mul_unroll_1024b(uu,K,uu,N_full,1u,32u);
        add_mod_asm_1024b(ww,DA,CB,N_full,32u);(void)sub_mod_asm_1024b(vv,DA,CB,N_full,32u);
        mont_mul_unroll_1024b(ww,ww,ww,N_full,1u,32u);mont_mul_unroll_1024b(vv,vv,vv,N_full,1u,32u);
        mp_shift_left_1_mod(vv,vv,N_full,32u);
        for(uint i=0;i<32;i++){X[i]=qq[i];Z[i]=uu[i];W[i]=ww[i];V[i]=vv[i];}
    }
    if(sw){for(uint i=0;i<32;i++){uint t=X[i];X[i]=W[i];W[i]=t;t=Z[i];Z[i]=V[i];V[i]=t;}}
    for(uint i=0;i<32;i++){Ld[32+i]=X[i];Ld[64+i]=Z[i];Ld[96+i]=W[i];Ld[128+i]=V[i];}
    barrier(CLK_LOCAL_MEM_FENCE);
    d[bs+32u+off0]=Ld[32u+off0];d[bs+32u+off1]=Ld[32u+off1];
    d[bs+64u+off0]=Ld[64u+off0];d[bs+64u+off1]=Ld[64u+off1];
    d[bs+96u+off0]=Ld[96u+off0];d[bs+96u+off1]=Ld[96u+off1];
    d[bs+128u+off0]=Ld[128u+off0];d[bs+128u+off1]=Ld[128u+off1];
}
#else
__kernel __attribute__((reqd_work_group_size(16u,1u,1u)))
void kernel_double_add_sliced_t16x2(__global const uint*sb,ulong sn,ulong st,ulong si,__global uint*d,uint cnt,uint sig,uint NP0,uint lim){
    (void)sb;(void)sn;(void)st;(void)si;(void)d;(void)cnt;(void)sig;(void)NP0;(void)lim;
}
#endif
