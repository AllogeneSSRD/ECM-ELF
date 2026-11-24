#pragma once

// One cooperative group per output coefficient. The factors and the output
// use ordinary canonical residues, including non-monic projective leaves.
template<class Env>
__device__ __forceinline__ void groot_coefficient(const Stage2CoopField<Env> &field,
    const unsigned long long *a,int da,const unsigned long long *b,int db,
    int k,int words,unsigned long long *out)
{
    using Bn=typename Env::cgbn_t;
    const auto &e=field.e;
    Bn value;
    cgbn_set_ui32(e,value,0);
    for(int i=max(0,k-db);i<=min(k,da);++i) {
        Bn x,y,term;
        stage2_coop_load<Env>(x,a+i*words,words);
        stage2_coop_load<Env>(y,b+(k-i)*words,words);
        typename Env::cgbn_wide_t wide;
        cgbn_mul_wide(e,wide,x,y);
        field.fold_mersenne(term,wide);
        const int carry=cgbn_add(e,value,value,term);
        if(carry || cgbn_compare(e,value,field.n)>=0)cgbn_sub(e,value,value,field.n);
    }
    stage2_coop_store<Env>(out+k*words,value,words);
}

template<int Words,int Roots>
__global__ void groot_subtree(const unsigned long long *leaves,unsigned long long *out,
    const unsigned long long *modulus,int words,int bits,size_t count)
{
    constexpr int tpi=16;
    const int group=threadIdx.x/tpi;
    const size_t first=size_t(blockIdx.x)*Roots;
    const int n=int(min(size_t(Roots),count-first));
    leaves+=2*first*words;
    out+=size_t(blockIdx.x)*(Roots+1)*words;
    if(n==1) {
        for(int i=threadIdx.x;i<2*words;i+=blockDim.x)out[i]=leaves[i];
        return;
    }

    using Context=cgbn_context_t<tpi>;
    using Env=cgbn_env_t<Context,64*Words>;
    Context context(cgbn_no_checks);
    Env env(context);
    typename Env::cgbn_t mod;
    stage2_coop_load<Env>(mod,modulus,words);
    const Stage2CoopField<Env> field{env,mod,0,bits};
    if(n==2) {
        if(group<3)groot_coefficient(field,leaves,1,leaves+2*words,1,group,words,out);
        return;
    }

    extern __shared__ unsigned long long pairs[];
    if(group<3) {
        groot_coefficient(field,leaves,1,leaves+2*words,1,group,words,pairs);
    } else if(n==4 && group<6) {
        groot_coefficient(field,leaves+4*words,1,leaves+6*words,1,group-3,words,pairs+3*words);
    }
    __syncthreads();
    if(group<=n) {
        const auto *right=n==4 ? pairs+3*words : leaves+4*words;
        groot_coefficient(field,pairs,2,right,n-2,group,words,out);
    }
}

template<int Words>
void launch_groot_subtree(unsigned roots,const unsigned long long *leaves,unsigned long long *out,
    const unsigned long long *modulus,int words,int bits,size_t count)
{
    if(roots==2) {
        groot_subtree<Words,2><<<unsigned((count+1)/2),64>>>(leaves,out,modulus,words,bits,count);
    } else {
        groot_subtree<Words,4><<<unsigned((count+3)/4),128,6*words*sizeof(unsigned long long)>>>(
            leaves,out,modulus,words,bits,count);
    }
}
