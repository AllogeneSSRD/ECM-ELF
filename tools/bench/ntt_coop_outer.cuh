/* Included after Goldilocks arithmetic and the original outer kernels.
   A CTA owns V adjacent coarse offsets, each with R radix coordinates. The
   radix axis lives in shared memory, so increasing M never creates x[2^M]
   registers per thread. The offset axis stays contiguous in global memory.
   Forward is the original DIF; inverse is the original DIT. No permutation,
   additional full-length buffer, pointwise operation or normalization here. */
template <int M, bool INVERSE>
__global__ __launch_bounds__(256, M==8 ? 2 : 3)
void outer_coop_kernel(unsigned long long *a, unsigned long long n, int stage,
                       const unsigned long long *coarse, const unsigned long long *radix,
                       unsigned long long stride)
{
    constexpr int R=1<<M, V=M==8 ? 16 : 32, ROWS=256/V;
    static_assert(M>=5 && M<=8,"cooperative radix supports 5..8 stages");
    __shared__ unsigned long long x[R*V], roots[R-1];
    // Later stages share d<ROWS roots across radix groups. Cache only those
    // roots (at most 128 words), instead of duplicating gl_mul per butterfly.
    __shared__ unsigned long long stage_roots[128], base_roots[(INVERSE ? M : 2)*V];
    const unsigned int v=threadIdx.x%V, row=threadIdx.x/V;
    // stage is L for DIF, and s0 for DIT; S is the common coarse-offset stride.
    const unsigned long long S=INVERSE ? (1ull<<stage) : (n>>(stage+M));
    const unsigned long long tiles=S/V, blk=blockIdx.x/tiles;
    const unsigned long long offset=(blockIdx.x%tiles)*V+v;
    a+=(size_t)blockIdx.y*stride+blk*(S*R)+offset;
    for(int j=threadIdx.x;j<R-1;j+=256)roots[j]=radix[j];
    for(int r=row;r<R;r+=ROWS)x[r*V+v]=a[(unsigned long long)r*S];
    if(row==0) {
        unsigned long long tb=coarse[offset];
        if constexpr(INVERSE) {
            base_roots[(M-1)*V+v]=tb;
#pragma unroll
            for(int i=M-2;i>=0;--i){tb=gl_mul(tb,tb);base_roots[i*V+v]=tb;}
        } else base_roots[v]=tb;
    }
    __syncthreads();
#pragma unroll
    for(int i=0;i<M;++i) {
        const int d=INVERSE ? (1<<i) : (1<<(M-1-i));
        const int off=INVERSE ? d-1 : R-(R>>i);
        const unsigned long long base_root=base_roots[(INVERSE ? i*V : (i&1)*V)+v];
        if(d>=ROWS) {
            // Every row has work. Compute w once for u, use it for all groups.
            for(int u=row;u<d;u+=ROWS) {
                const auto w=gl_mul(base_root,roots[off+u]);
                for(int group=0;group<R/(2*d);++group) {
                    const int first=group*(2*d)+u,second=first+d;
                    const auto aa=x[first*V+v],bb=x[second*V+v];
                    if constexpr(INVERSE) {
                        const auto t=gl_mul(bb,w);
                        x[first*V+v]=gl_add_dev(aa,t);x[second*V+v]=gl_sub_dev(aa,t);
                    } else {
                        x[first*V+v]=gl_add_dev(aa,bb);
                        x[second*V+v]=gl_mul(gl_sub_dev(aa,bb),w);
                    }
                }
            }
        } else {
            // Share w across rows, keeping all rows active in the butterflies.
            if(row<(unsigned int)d)stage_roots[row*V+v]=gl_mul(base_root,roots[off+row]);
            __syncthreads();
            for(int j=row;j<R/2;j+=ROWS) {
                const int u=j&(d-1),first=(j/d)*(2*d)+u,second=first+d;
                const auto w=stage_roots[u*V+v],aa=x[first*V+v],bb=x[second*V+v];
                if constexpr(INVERSE) {
                    const auto t=gl_mul(bb,w);
                    x[first*V+v]=gl_add_dev(aa,t);x[second*V+v]=gl_sub_dev(aa,t);
                } else {
                    x[first*V+v]=gl_add_dev(aa,bb);
                    x[second*V+v]=gl_mul(gl_sub_dev(aa,bb),w);
                }
            }
        }
        // Each butterfly owns its two destinations; next stage can change owners.
        // Ping-pong coarse roots: never overwrite a slot another warp may still
        // be reading in this stage. The existing end barrier publishes the next.
        if constexpr(!INVERSE)if(row==0 && i<M-1)base_roots[((i+1)&1)*V+v]=gl_mul(base_root,base_root);
        __syncthreads();
    }
    for(int r=row;r<R;r+=ROWS)a[(unsigned long long)r*S]=x[r*V+v];
}

template <bool INVERSE>
static void launch_outer_coop(int m,unsigned long long *a,unsigned long long n,int stage,
                              const unsigned long long *coarse,const unsigned long long *radix,
                              unsigned long long nbatch,unsigned long long stride)
{
    const auto S=INVERSE ? (1ull<<stage) : (n>>(stage+m));
    const int v=m==8 ? 16 : 32;
    if(m<5 || m>8 || S<(unsigned long long)v) {
        std::fprintf(stderr,NTT_PROBE_NAME ": invalid cooperative outer M=%d S=%llu\n",m,S);
        std::exit(3);
    }
    dim3 grid((unsigned int)(n/((1ull<<m)*v)),(unsigned int)nbatch);
    switch(m) {
    case 5:outer_coop_kernel<5,INVERSE><<<grid,256>>>(a,n,stage,coarse,radix,stride);break;
    case 6:outer_coop_kernel<6,INVERSE><<<grid,256>>>(a,n,stage,coarse,radix,stride);break;
    case 7:outer_coop_kernel<7,INVERSE><<<grid,256>>>(a,n,stage,coarse,radix,stride);break;
    case 8:outer_coop_kernel<8,INVERSE><<<grid,256>>>(a,n,stage,coarse,radix,stride);break;
    }
}
