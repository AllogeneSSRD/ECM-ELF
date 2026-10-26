// ECM Stage 1 — sliced cooperative kernel (AMD-only, ds_bpermute CIOS).
// NVIDIA: empty stub.
#if defined(__AMDGCN__)

__kernel __attribute__((reqd_work_group_size(32u,1u,1u)))
void kernel_double_add_sliced(__global const uint*sb,ulong sn,ulong st,ulong si,__global uint*d,uint cnt,uint sig,uint NP0,uint lim){
    if(lim!=32u)return;uint lid=get_local_id(0),ins=get_group_id(0);if(ins>=cnt)return;
    uint bs=ins*160u;
    __local uint L[160];
    uint N_my=d[bs+lid],aX_my=d[bs+32+lid],aZ_my=d[bs+64+lid],bX_my=d[bs+96+lid],bZ_my=d[bs+128+lid];
    L[lid]=N_my;L[32+lid]=aX_my;L[64+lid]=aZ_my;L[96+lid]=bX_my;L[128+lid]=bZ_my;
    barrier(CLK_LOCAL_MEM_FENCE);
    uint N_full[32],aX_full[32],aZ_full[32],bX_full[32],bZ_full[32];
    if(lid==0u)for(uint i=0;i<32;i++){N_full[i]=L[i];aX_full[i]=L[32+i];aZ_full[i]=L[64+i];bX_full[i]=L[96+i];bZ_full[i]=L[128+i];}

    uint dv=sig+ins;int sw=0;ulong se=st+si;if(se>sn)se=sn;
    for(ulong b=st;b<se;b++){
        ulong nth=sn-1u-b;uint li=(uint)(nth>>5),bi=(uint)(nth&31);int bit=(int)((sb[li]>>bi)&1u);
        if(bit!=sw){sw=!sw;if(lid==0u)for(uint i=0;i<32;i++){uint t=aX_full[i];aX_full[i]=bX_full[i];bX_full[i]=t;t=aZ_full[i];aZ_full[i]=bZ_full[i];bZ_full[i]=t;}}

        uint tt[32],CB[32],DA[32],AA[32],BB[32],K[32],dK[32],qq[32],uu[32],ww[32],vv[32];
        if(lid==0u){for(uint i=0;i<32;i++){qq[i]=aX_full[i];uu[i]=aZ_full[i];ww[i]=bX_full[i];vv[i]=bZ_full[i];}
            add_mod_asm_1024b(tt,vv,ww,N_full,32u);(void)sub_mod_asm_1024b(vv,vv,ww,N_full,32u);
            add_mod_asm_1024b(ww,uu,qq,N_full,32u);(void)sub_mod_asm_1024b(uu,uu,qq,N_full,32u);
        }

        #define CIOS(name) do{\
            barrier(CLK_LOCAL_MEM_FENCE);\
            uint my_T=0u,my_A=L[lid],my_B=L[32+lid],my_N=L[64+lid],t32=0u,t33=0u;\
            for(uint i=0u;i<32u;i++){uint A_i=L[i];uint carry=0u,Cp=0u;\
            for(uint k=0u;k<32u;k++){if(lid==k){ulong uv=(ulong)my_T+(ulong)A_i*(ulong)my_B+carry;my_T=(uint)uv;carry=(uint)(uv>>32);}if(lid==31u&&k==31u)Cp=carry;carry=__builtin_amdgcn_ds_bpermute(((lid-1u)&31u)*4u,carry);}\
            {ulong t_=(ulong)t32+(ulong)Cp;t32=(uint)t_;t33+=(uint)(t_>>32);}\
            uint m=0u;if(lid==0u)m=(uint)((ulong)my_T*(ulong)1u);m=__builtin_amdgcn_readfirstlane(m);\
            carry=0u;uint Cr=0u;\
            for(uint k=0u;k<32u;k++){if(lid==k){ulong uv=(ulong)my_T+(ulong)m*(ulong)my_N+carry;my_T=(uint)uv;carry=(uint)(uv>>32);}if(lid==31u&&k==31u)Cr=carry;carry=__builtin_amdgcn_ds_bpermute(((lid-1u)&31u)*4u,carry);}\
            {ulong t_=(ulong)t32+(ulong)Cr;t32=(uint)t_;t33+=(uint)(t_>>32);}\
            L[96+lid]=my_T;if(lid==31u){L[128]=t32;L[129]=t33;}barrier(CLK_LOCAL_MEM_FENCE);\
            my_T=L[96+lid+1u];t32=L[129];t33=0u;}\
            {uint D=0u,borrow=0u;for(uint k=0u;k<32u;k++){if(lid==k){ulong tv=(ulong)my_T,nv=(ulong)my_N;D=(uint)(tv-nv-borrow);borrow=(tv<nv+borrow)?1u:0u;}borrow=__builtin_amdgcn_ds_bpermute(((lid-1u)&31u)*4u,borrow);}uint ns=(borrow==0u)?1u:0u;ns=__builtin_amdgcn_readfirstlane(ns);uint mk=0u-ns;L[96+lid]=(D&mk)|(my_T&~mk);}\
            barrier(CLK_LOCAL_MEM_FENCE);if(lid==0u)for(uint i=0;i<32;i++)name[i]=L[96+i];\
        }while(0)

        if(lid==0u)for(uint i=0;i<32;i++){L[i]=tt[i];L[32+i]=uu[i];L[64+i]=N_full[i];} CIOS(CB);
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=vv[i];L[32+i]=ww[i];L[64+i]=N_full[i];} CIOS(DA);
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=ww[i];L[32+i]=ww[i];L[64+i]=N_full[i];} CIOS(AA);
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=uu[i];L[32+i]=uu[i];L[64+i]=N_full[i];} CIOS(BB);
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=AA[i];L[32+i]=BB[i];L[64+i]=N_full[i];} CIOS(qq);

        if(lid==0u){
            (void)sub_mod_asm_1024b(K,AA,BB,N_full,32u);for(uint i=0;i<32;i++)dK[i]=K[i];
            special_mult_ui32_unroll_1024b(dK,dv,N_full,1u,32u);
            add_mod_asm_1024b(uu,BB,dK,N_full,32u);
        }
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=K[i];L[32+i]=uu[i];L[64+i]=N_full[i];} CIOS(uu);

        if(lid==0u){add_mod_asm_1024b(ww,DA,CB,N_full,32u);(void)sub_mod_asm_1024b(vv,DA,CB,N_full,32u);}
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=ww[i];L[32+i]=ww[i];L[64+i]=N_full[i];} CIOS(ww);
        if(lid==0u)for(uint i=0;i<32;i++){L[i]=vv[i];L[32+i]=vv[i];L[64+i]=N_full[i];} CIOS(vv);

        if(lid==0u){mp_shift_left_1_mod(vv,vv,N_full,32u);
            for(uint i=0;i<32;i++){aX_full[i]=qq[i];aZ_full[i]=uu[i];bX_full[i]=ww[i];bZ_full[i]=vv[i];}
        }
        #undef CIOS
    }
    if(lid==0u){if(sw)for(uint i=0;i<32;i++){uint t=aX_full[i];aX_full[i]=bX_full[i];bX_full[i]=t;t=aZ_full[i];aZ_full[i]=bZ_full[i];bZ_full[i]=t;}
        for(uint i=0;i<32;i++){L[32+i]=aX_full[i];L[64+i]=aZ_full[i];L[96+i]=bX_full[i];L[128+i]=bZ_full[i];}}
    barrier(CLK_LOCAL_MEM_FENCE);
    d[bs+32+lid]=L[32+lid];d[bs+64+lid]=L[64+lid];d[bs+96+lid]=L[96+lid];d[bs+128+lid]=L[128+lid];
}

#else
__kernel __attribute__((reqd_work_group_size(32u,1u,1u)))
void kernel_double_add_sliced(__global const uint*sb,ulong sn,ulong st,ulong si,__global uint*d,uint cnt,uint sig,uint NP0,uint lim){
    (void)sb;(void)sn;(void)st;(void)si;(void)d;(void)cnt;(void)sig;(void)NP0;(void)lim;
}
#endif
