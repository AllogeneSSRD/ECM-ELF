#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
#include "../bench/ntt_tensor_goldilocks.cuh"

using Word=unsigned long long;
static Word live_bytes=0,peak_bytes=0;
struct Buffer {
    void *p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){CK(cudaMalloc(&p,n));live_bytes+=n;peak_bytes=std::max(peak_bytes,live_bytes);}
    ~Buffer(){CK(cudaFree(p));live_bytes-=bytes;}
    template<class T=Word>T *get(){return static_cast<T *>(p);}
};
static void put64(mpz_t z,Word x){mpz_import(z,1,-1,8,0,0,&x);}
static Word get64(const mpz_t z){Word x=0;size_t n=0;mpz_export(&x,&n,-1,8,0,0,z);return x;}
static unsigned int rev4(unsigned int x){return ((x&1)<<3)|((x&2)<<1)|((x&4)>>1)|((x&8)>>3);}
__device__ unsigned int dev_rev4(unsigned int x){return ((x&1)<<3)|((x&2)<<1)|((x&4)>>1)|((x&8)>>3);}

// A useful CUDA control: one full butterfly/lane, four 8-lane groups, two
// vectors/group, register-pair/lane-bit transposes, natural-order boundary.
// The low-stage twiddle helper is the one used by the production warp tile.
template<bool INV>
__global__ void cuda_goldilocks_fft16(const Word *in,Word *out,Word batches,
                                     const Word *table,Word scale)
{
    const Word batch=((Word)blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(batch>=batches)return;
    const unsigned int lane=threadIdx.x&31,j0=lane&7,vector=lane>>3;
    #pragma unroll
    for(int v=0;v<2;++v) {
        const unsigned int col=vector+4*v;
        Word lo=in[batch*128+(INV ? dev_rev4(2*j0) : j0)*8+col];
        Word hi=in[batch*128+(INV ? dev_rev4(2*j0+1) : j0+8)*8+col];
        #pragma unroll
        for(int q=0;q<4;++q) {
            const int st=INV ? q : 3-q;
            const unsigned int half=1u<<st,j=lane&(half-1);
            const Word w=__ldg(table+half-1+j),u=lo,t=INV ? tile_warp_twiddle(hi,w,st) : hi;
            lo=gl_add_dev(u,t);hi=INV ? gl_sub_dev(u,t) : tile_warp_twiddle(gl_sub_dev(u,t),w,st);
            if(INV ? st<3 : st>0) {
                const unsigned int mask=1u<<(INV ? st : st-1);
                const bool lower=(lane&mask)==0;
                const Word peer=__shfl_xor_sync(0xffffffffu,lower ? hi : lo,mask);
                lo=lower ? lo : peer;hi=lower ? peer : hi;
            }
        }
        if(INV){lo=gl_mul(lo,scale);hi=gl_mul(hi,scale);}
        out[batch*128+(INV ? j0 : dev_rev4(2*j0))*8+col]=lo;
        out[batch*128+(INV ? j0+8 : dev_rev4(2*j0+1))*8+col]=hi;
    }
}
__global__ void cuda_goldilocks_dft16(const Word *matrix,const Word *in,Word *out,Word batches)
{
    const Word index=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=batches*128)return;
    const unsigned int row=(index%128)/8,col=index&7;const Word base=(index/128)*128;
    Word value=0;
    #pragma unroll
    for(int k=0;k<16;++k)value=gl_add_dev(value,gl_mul(matrix[row*16+k],in[base+k*8+col]));
    out[index]=value;
}
__global__ void corrupt_first(Word *x){if(threadIdx.x==0 && blockIdx.x==0)x[0]^=1;}
__global__ void fill_patterns(Word *x,Word count,const Word *patterns)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)x[i]=patterns[((i/128)&15)*128+i%128];
}
__global__ void check_patterns(const Word *x,Word count,const Word *patterns,Word *bad)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && x[i]!=patterns[((i/128)&15)*128+i%128])atomicAdd(bad,1ull);
}
__global__ void fill_tile_patterns(Word *x,Word count,const Word *patterns,Word pattern_words)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]=patterns[i%pattern_words];
}
__global__ void check_tile_patterns(const Word *x,Word count,const Word *patterns,Word pattern_words,Word *bad)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && x[i]!=patterns[i%pattern_words])atomicAdd(bad,1ull);
}
static unsigned int warp_grid(Word batches,int threads){return (unsigned int)((batches*32+threads-1)/threads);}

struct Transform {
    std::vector<Word> matrix,table;Word scale;
    explicit Transform(bool inverse,bool normalized=true):matrix(256),table(15),scale(1) {
        mpz_t p,base,exponent,root,z,k;mpz_inits(p,base,exponent,root,z,k,nullptr);
        put64(p,GL_P);mpz_set_ui(base,7);put64(exponent,(GL_P-1)/16);mpz_powm(root,base,exponent,p);
        mpz_powm_ui(z,root,16,p);if(mpz_cmp_ui(z,1))std::exit(3);
        mpz_powm_ui(z,root,8,p);mpz_add_ui(z,z,1);if(mpz_cmp(z,p))std::exit(3);
        if(inverse){mpz_invert(root,root,p);mpz_set_ui(k,16);mpz_invert(k,k,p);scale=get64(k);}
        for(int row=0;row<16;++row)for(int col=0;col<16;++col) {
            mpz_powm_ui(z,root,row*col,p);
            if(inverse && normalized){mpz_mul(z,z,k);mpz_mod(z,z,p);}
            matrix[row*16+col]=get64(z);
        }
        for(int h=1;h<=8;h*=2)for(int j=0;j<h;++j) {
            mpz_powm_ui(z,root,j*(8/h),p);table[h-1+j]=get64(z);
        }
        mpz_clears(p,base,exponent,root,z,k,nullptr);
    }
};
static Word random64(Word &x){x^=x>>12;x^=x<<25;x^=x>>27;return (x*2685821657736338717ull)%GL_P;}
static std::vector<Word> fixture(int type,size_t size,Word seed)
{
    const Word edge[]={0,1,2,GL_P-1,GL_P-2,0xffffffffull,1ull<<32,1ull<<63,GL_P/2,0x0001000100010001ull};
    std::vector<Word> out(size);seed+=1;
    for(size_t i=0;i<size;++i)switch(type) {
        case 0:out[i]=0;break;case 1:out[i]=1;break;case 2:out[i]=GL_P-1;break;
        case 3:out[i]=1ull<<63;break;case 4:out[i]=edge[(i+seed)%10];break;
        case 5:out[i]=random64(seed);break;case 6:out[i]=(i%128==0 || i%128==127) ? GL_P-2 : 0;break;
        default:out[i]=GL_P-1-(i%7);break;
    }
    return out;
}
static std::vector<Word> reference(const std::vector<Word> &a,const std::vector<Word> &b,
                                   std::vector<unsigned int> *tops=nullptr)
{
    std::vector<Word> result(b.size());if(tops)tops->resize(b.size());
    mpz_t p,sum,x,y,z;mpz_inits(p,sum,x,y,z,nullptr);put64(p,GL_P);
    for(size_t batch=0;batch<b.size()/128;++batch)for(int row=0;row<16;++row)for(int col=0;col<8;++col) {
        mpz_set_ui(sum,0);
        for(int k=0;k<16;++k){put64(x,a[row*16+k]);put64(y,b[batch*128+k*8+col]);mpz_addmul(sum,x,y);}
        const size_t i=batch*128+row*8+col;
        if(tops){mpz_fdiv_q_2exp(z,sum,128);(*tops)[i]=(unsigned int)mpz_get_ui(z);}
        mpz_mod(z,sum,p);result[i]=get64(z);
    }
    mpz_clears(p,sum,x,y,z,nullptr);return result;
}
struct Checks {Word cases=0,words=0,bad=0,top_words=0,nonzero_tops=0;};
static void compare(Checks &stats,const std::vector<Word> &got,const std::vector<Word> &want,
                    const std::vector<unsigned int> *tops=nullptr,const std::vector<unsigned int> *expected=nullptr)
{
    ++stats.cases;stats.words+=want.size();
    for(size_t i=0;i<want.size();++i) {
        if(got[i]!=want[i])++stats.bad;
        if(tops){++stats.top_words;stats.nonzero_tops+=(*expected)[i]!=0;if((*tops)[i]!=(*expected)[i])++stats.bad;}
    }
}
static void print_checks(const char *name,const Checks &s)
{
    std::printf("tc_gold_check: group=%s cases=%llu words=%llu top_words=%llu nonzero_tops=%llu bad=%llu\n",
        name,s.cases,s.words,s.top_words,s.nonzero_tops,s.bad);
}
static std::vector<unsigned int> pack_roots(const std::vector<Word> &matrix)
{
    std::vector<unsigned int> out(512);
    for(int byte=0;byte<8;++byte)for(int half=0;half<2;++half)for(int lane=0;lane<32;++lane) {
        const int row=(lane>>2)+8*half,col=(lane&3)*4;unsigned int value=0;
        for(int i=0;i<4;++i)value|=(unsigned int)((matrix[row*16+col+i]>>(byte*8))&255ull)<<(8*i);
        out[byte*64+half*32+lane]=value;
    }
    return out;
}
static unsigned int reverse_bits(unsigned int x,int bits)
{
    unsigned int out=0;for(int i=0;i<bits;++i){out=(out<<1)|(x&1);x>>=1;}return out;
}
struct TilePlan {
    std::vector<Word> forward,inverse;std::vector<unsigned int> packed_f,packed_i;Word scale;
    explicit TilePlan(int t):forward((1u<<t)-1),inverse((1u<<t)-1) {
        const unsigned int n=1u<<t;
        mpz_t p,base,e,root,ri,w,z;mpz_inits(p,base,e,root,ri,w,z,nullptr);
        put64(p,GL_P);mpz_set_ui(base,7);put64(e,(GL_P-1)/n);mpz_powm(root,base,e,p);mpz_invert(ri,root,p);
        for(unsigned int h=1;h<n;h*=2)for(unsigned int j=0;j<h;++j) {
            mpz_powm_ui(z,root,j*(n/(2*h)),p);forward[h-1+j]=get64(z);
            mpz_powm_ui(z,ri,j*(n/(2*h)),p);inverse[h-1+j]=get64(z);
        }
        mpz_set_ui(z,n);mpz_invert(z,z,p);scale=get64(z);mpz_clears(p,base,e,root,ri,w,z,nullptr);
        Transform f(false),i(true,false);packed_f=pack_roots(f.matrix);packed_i=pack_roots(i.matrix);
    }
};
// GMP iterative DIT oracle. Forward starts with bit-reversed input and permutes
// its natural result to the tile's DIF order; inverse starts with physical
// bit-reversed input, after an independently computed B/scale product.
static std::vector<Word> tile_reference(const std::vector<Word> &input,const std::vector<Word> &multiplier,
                                        int t,bool inverse,unsigned int tiles,unsigned int batches,unsigned int padding)
{
    const unsigned int n=1u<<t,stride=n*tiles+padding;auto out=input;
    std::vector<__mpz_struct> x(n);for(auto &z:x)mpz_init(&z);
    mpz_t p,root,base,e,w,step,u,v,z,scale;mpz_inits(p,root,base,e,w,step,u,v,z,scale,nullptr);
    put64(p,GL_P);mpz_set_ui(base,7);put64(e,(GL_P-1)/n);mpz_powm(root,base,e,p);
    mpz_set_ui(scale,n);mpz_invert(scale,scale,p);if(inverse)mpz_invert(root,root,p);
    for(unsigned int batch=0;batch<batches;++batch)for(unsigned int tile=0;tile<tiles;++tile) {
        const size_t offset=(size_t)batch*stride+tile*n;
        for(unsigned int i=0;i<n;++i) {
            auto *dst=&x[inverse ? i : reverse_bits(i,t)];put64(dst,input[offset+i]);
            if(inverse){put64(z,multiplier[offset+i]);mpz_mul(dst,dst,z);mpz_mul(dst,dst,scale);mpz_mod(dst,dst,p);}
        }
        for(unsigned int len=2;len<=n;len*=2) {
            mpz_powm_ui(step,root,n/len,p);mpz_set_ui(w,1);
            for(unsigned int j=0;j<len/2;++j) {
                for(unsigned int start=0;start<n;start+=len) {
                    mpz_set(u,&x[start+j]);mpz_mul(v,&x[start+j+len/2],w);mpz_mod(v,v,p);
                    mpz_add(&x[start+j],u,v);mpz_mod(&x[start+j],&x[start+j],p);
                    mpz_sub(&x[start+j+len/2],u,v);mpz_mod(&x[start+j+len/2],&x[start+j+len/2],p);
                }
                mpz_mul(w,w,step);mpz_mod(w,w,p);
            }
        }
        for(unsigned int i=0;i<n;++i)out[offset+(inverse ? i : reverse_bits(i,t))]=get64(&x[i]);
    }
    for(auto &z:x)mpz_clear(&z);mpz_clears(p,root,base,e,w,step,u,v,z,scale,nullptr);return out;
}
static int tile_gate(bool fault)
{
    Checks ftc,itc,rtc,fcu,icu,rcu,b_readonly,guards;bool injected=false;
    for(int t:{6,8,10,12}) {
        const unsigned int tile=1u<<t;TilePlan plan(t);
        Buffer tf(plan.forward.size()*8),ti(plan.inverse.size()*8),rf(2048),ri(2048);
        CK(cudaMemcpy(tf.p,plan.forward.data(),tf.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(ti.p,plan.inverse.data(),ti.bytes,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(rf.p,plan.packed_f.data(),rf.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(ri.p,plan.packed_i.data(),ri.bytes,cudaMemcpyHostToDevice));
        ntt_tile_attributes(true,tile*8,true);
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,tile*8));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,tile*8));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        for(int type=0;type<8;++type)for(unsigned int batches:{1u,3u})for(unsigned int tiles:{1u,3u}) {
            const unsigned int padding=17,stride=tile*tiles+padding;const size_t count=(size_t)batches*stride;
            auto input=fixture(type,count,0x918),multiplier=fixture((type+4)%8,count,0x211),ones=std::vector<Word>(count,1);
            const auto forward=tile_reference(input,multiplier,t,false,tiles,batches,padding);
            const auto inverse=tile_reference(input,multiplier,t,true,tiles,batches,padding);
            Buffer a(count*8),b(count*8);std::vector<Word> got(count),bgot(count);
            const dim3 grid(tiles,batches);const auto launch_tile=[&](bool inv,int threads,bool tensor) {
                if(tensor) {
                    if(inv)tc_goldilocks_tile<true><<<grid,threads,tile*8>>>(a.get(),b.get(),tile*tiles,t,ti.get(),plan.scale,stride,ri.get<unsigned int>());
                    else tc_goldilocks_tile<false><<<grid,threads,tile*8>>>(a.get(),nullptr,tile*tiles,t,tf.get(),0,stride,rf.get<unsigned int>());
                } else {
                    if(inv)tile_kernel<true,true><<<grid,512,tile*8>>>(a.get(),b.get(),tile*tiles,t,t,ti.get(),plan.scale,stride);
                    else tile_kernel<false,true><<<grid,512,tile*8>>>(a.get(),nullptr,tile*tiles,t,t,tf.get(),0,stride);
                }
            };
            for(int threads:{128,256,512,0}) {
                const bool tensor=threads!=0;
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(b.p,multiplier.data(),b.bytes,cudaMemcpyHostToDevice));
                launch_tile(false,threads,tensor);
                if(fault && tensor && !injected){corrupt_first<<<1,32>>>(a.get());injected=true;}
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(tensor ? ftc : fcu,got,forward);
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));launch_tile(true,threads,tensor);
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(tensor ? itc : icu,got,inverse);
                CK(cudaMemcpy(bgot.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));compare(b_readonly,bgot,multiplier);
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(b.p,ones.data(),b.bytes,cudaMemcpyHostToDevice));
                launch_tile(false,threads,tensor);launch_tile(true,threads,tensor);
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(tensor ? rtc : rcu,got,input);
                std::vector<Word> guard_got,guard_want;
                for(unsigned int batch=0;batch<batches;++batch)for(unsigned int j=tile*tiles;j<stride;++j) {
                    guard_got.push_back(got[(size_t)batch*stride+j]);guard_want.push_back(input[(size_t)batch*stride+j]);
                }
                compare(guards,guard_got,guard_want);
            }
        }
    }
    Word bad=0;for(const auto &s:{ftc,itc,rtc,fcu,icu,rcu,b_readonly,guards})bad+=s.bad;
    print_checks("tile_forward_tc",ftc);print_checks("tile_inverse_tc",itc);print_checks("tile_roundtrip_tc",rtc);
    print_checks("tile_forward_cuda",fcu);print_checks("tile_inverse_cuda",icu);print_checks("tile_roundtrip_cuda",rcu);
    print_checks("tile_B_readonly",b_readonly);print_checks("tile_stride_guards",guards);
    std::printf("tc_gold_tile_gate: fault=%d injected=%d bad=%llu\n",(int)fault,(int)injected,bad);return bad ? 3 : 0;
}
struct TileBuffers {
    int t;Word n;TilePlan plan;
    std::vector<Word> input,multiplier,expected;
    Buffer a,b,tf,ti,rf,ri,patterns,bpatterns,want,bad;
    TileBuffers(int k,int tile_bits,int operation):t(tile_bits),n(1ull<<k),plan(t),
        input(fixture(5,4ull<<t,0x9156)),multiplier(fixture(operation==2 ? 1 : 5,4ull<<t,0x5318)),
        expected(operation==2 ? input : tile_reference(input,multiplier,t,operation==1,4,1,0)),
        a(n*8),b(n*8),tf(plan.forward.size()*8),ti(plan.inverse.size()*8),rf(2048),ri(2048),
        patterns(input.size()*8),bpatterns(multiplier.size()*8),want(expected.size()*8),bad(8)
    {
        const auto upload=[](Buffer &dst,const void *src){CK(cudaMemcpy(dst.p,src,dst.bytes,cudaMemcpyHostToDevice));};
        upload(tf,plan.forward.data());upload(ti,plan.inverse.data());upload(rf,plan.packed_f.data());upload(ri,plan.packed_i.data());
        upload(patterns,input.data());upload(bpatterns,multiplier.data());upload(want,expected.data());
        ntt_tile_attributes(true,(1u<<t)*8,true);
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(1u<<t)*8));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(1u<<t)*8));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        CK(cudaFuncSetAttribute(tc_goldilocks_tile<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    }
    void fill() {
        fill_tile_patterns<<<(unsigned int)((n+255)/256),256>>>(a.get(),n,patterns.get(),input.size());
        fill_tile_patterns<<<(unsigned int)((n+255)/256),256>>>(b.get(),n,bpatterns.get(),multiplier.size());
    }
    Word check() {
        CK(cudaMemset(bad.p,0,8));
        check_tile_patterns<<<(unsigned int)((n+255)/256),256>>>(a.get(),n,want.get(),expected.size(),bad.get());
        check_tile_patterns<<<(unsigned int)((n+255)/256),256>>>(b.get(),n,bpatterns.get(),multiplier.size(),bad.get());
        Word count=0;CK(cudaMemcpy(&count,bad.p,8,cudaMemcpyDeviceToHost));return count;
    }
    void launch(bool tensor,bool inverse,int threads,cudaStream_t stream=0) {
        const unsigned int grid=(unsigned int)(n>>t),shared=(1u<<t)*8;
        if(tensor) {
            if(inverse)tc_goldilocks_tile<true><<<grid,threads,shared,stream>>>(a.get(),b.get(),n,t,ti.get(),plan.scale,n,ri.get<unsigned int>());
            else tc_goldilocks_tile<false><<<grid,threads,shared,stream>>>(a.get(),nullptr,n,t,tf.get(),0,n,rf.get<unsigned int>());
        } else {
            if(inverse)tile_kernel<true,true><<<grid,512,shared,stream>>>(a.get(),b.get(),n,t,t,ti.get(),plan.scale,n);
            else tile_kernel<false,true><<<grid,512,shared,stream>>>(a.get(),nullptr,n,t,t,tf.get(),0,n);
        }
    }
};
static int tile_bench(int k,int t,int threads,int operation)
{
    TileBuffers state(k,t,operation);cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));int run=0;
    for(int mode:{0,1,1,0,1,0,0,1}) {
        double total=0;Word wrong=0;
        for(int repeat=0;repeat<4;++repeat) {
            state.fill();CK(cudaEventRecord(start));state.launch(mode,operation==1,threads);
            if(operation==2)state.launch(mode,true,threads);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
            if(repeat)total+=ms/1000.0;wrong=state.check();if(wrong)return 3;
        }
        std::printf("tc_gold_tile_bench: run=%d mode=%d operation=%d t=%d threads=%d N=%llu seconds=%.9f bad=%llu\n",
            ++run,mode,operation,t,threads,state.n,total/3,wrong);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return 0;
}
// Two independent buffers/tasks; this measures tile throughput and overlap,
// not two ECM curves. Mode 0/1: CUDA+CUDA serial/parallel; 2/3: CUDA+TC serial/parallel.
static int mixed_bench(int k,int t,int threads)
{
    TileBuffers first(k,t,0),second(k,t,0);cudaStream_t sa,sb;
    CK(cudaStreamCreateWithFlags(&sa,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&sb,cudaStreamNonBlocking));
    cudaEvent_t start,end,ea,eb;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));CK(cudaEventCreate(&ea));CK(cudaEventCreate(&eb));
    int run=0;
    for(int mode:{0,2,3,1,1,3,2,0,1,3,2,0,0,2,3,1}) {
        double total=0;Word wrong=0;
        for(int repeat=0;repeat<4;++repeat) {
            first.fill();second.fill();CK(cudaEventRecord(start));
            CK(cudaStreamWaitEvent(sa,start));first.launch(false,false,512,sa);CK(cudaEventRecord(ea,sa));
            CK(cudaStreamWaitEvent(sb,start));if(!(mode&1))CK(cudaStreamWaitEvent(sb,ea));
            second.launch(mode>=2,false,threads,sb);CK(cudaEventRecord(eb,sb));
            CK(cudaStreamWaitEvent(0,ea));CK(cudaStreamWaitEvent(0,eb));CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
            float ms=0;CK(cudaEventElapsedTime(&ms,start,end));if(repeat)total+=ms/1000.0;
            wrong=first.check()+second.check();if(wrong)return 3;
        }
        std::printf("tc_gold_mixed_bench: run=%d mode=%d t=%d threads=%d N=%llu seconds=%.9f tasks=2 bad=%llu\n",
            ++run,mode,t,threads,first.n,total/3,wrong);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));CK(cudaEventDestroy(ea));CK(cudaEventDestroy(eb));
    CK(cudaStreamDestroy(sa));CK(cudaStreamDestroy(sb));return 0;
}
static int gate(bool fault)
{
    Checks matrix_tc,matrix_cuda,transform_tc,transform_cuda,roundtrip_tc,roundtrip_cuda;
    bool injected=false;
    Transform forward(false),inverse(true);
    Buffer ft(15*8),it(15*8),fm(256*8),im(256*8);
    CK(cudaMemcpy(ft.p,forward.table.data(),ft.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(it.p,inverse.table.data(),it.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fm.p,forward.matrix.data(),fm.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(im.p,inverse.matrix.data(),im.bytes,cudaMemcpyHostToDevice));
    for(int type=0;type<8;++type)for(Word batches:{1ull,3ull,4ull,17ull}) {
        auto a=fixture(type,256,0x1935),input=fixture(type,batches*128,0x572a);
        std::vector<unsigned int> top_want;const auto want=reference(a,input,&top_want);
        Buffer ma(256*8),b(input.size()*8),c(input.size()*8),top(input.size()*4);
        CK(cudaMemcpy(ma.p,a.data(),ma.bytes,cudaMemcpyHostToDevice));
        std::vector<Word> got(input.size());std::vector<unsigned int> got_top(input.size());
        for(int threads:{32,64,128,256}) {
            for(int alias:{0,1}) {
                CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
                Word *out=alias ? b.get() : c.get();
                tc_goldilocks_mat16<<<warp_grid(batches,threads),threads>>>(ma.get(),b.get(),out,batches,top.get<unsigned int>());
                if(fault && !injected){corrupt_first<<<1,32>>>(out);injected=true;}
                CK(cudaMemcpy(got.data(),out,b.bytes,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(got_top.data(),top.p,top.bytes,cudaMemcpyDeviceToHost));
                compare(matrix_tc,got,want,&got_top,&top_want);
            }
            CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
            cuda_goldilocks_dft16<<<(unsigned int)((input.size()+threads-1)/threads),threads>>>(ma.get(),b.get(),c.get(),batches);
            CK(cudaMemcpy(got.data(),c.p,c.bytes,cudaMemcpyDeviceToHost));compare(matrix_cuda,got,want);
        }
        for(int inv:{0,1}) {
            const auto &t=inv ? inverse : forward;auto expected=reference(t.matrix,input,&top_want);
            for(int threads:{32,64,128,256})for(int alias:{0,1}) {
                Word *out=alias ? b.get() : c.get();
                CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
                tc_goldilocks_mat16<<<warp_grid(batches,threads),threads>>>(inv ? im.get() : fm.get(),b.get(),out,batches,top.get<unsigned int>());
                CK(cudaMemcpy(got.data(),out,b.bytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got_top.data(),top.p,top.bytes,cudaMemcpyDeviceToHost));
                compare(transform_tc,got,expected,&got_top,&top_want);
                CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
                if(inv)cuda_goldilocks_fft16<true><<<warp_grid(batches,threads),threads>>>(b.get(),out,batches,it.get(),inverse.scale);
                else cuda_goldilocks_fft16<false><<<warp_grid(batches,threads),threads>>>(b.get(),out,batches,ft.get(),1);
                CK(cudaMemcpy(got.data(),out,b.bytes,cudaMemcpyDeviceToHost));compare(transform_cuda,got,expected);
            }
        }
        for(int threads:{32,64,128,256})for(int alias:{0,1}) {
            Word *middle=alias ? b.get() : c.get();
            CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
            tc_goldilocks_mat16<<<warp_grid(batches,threads),threads>>>(fm.get(),b.get(),middle,batches);
            tc_goldilocks_mat16<<<warp_grid(batches,threads),threads>>>(im.get(),middle,b.get(),batches);
            CK(cudaMemcpy(got.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));compare(roundtrip_tc,got,input);
            CK(cudaMemcpy(b.p,input.data(),b.bytes,cudaMemcpyHostToDevice));
            cuda_goldilocks_fft16<false><<<warp_grid(batches,threads),threads>>>(b.get(),middle,batches,ft.get(),1);
            cuda_goldilocks_fft16<true><<<warp_grid(batches,threads),threads>>>(middle,b.get(),batches,it.get(),inverse.scale);
            CK(cudaMemcpy(got.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));compare(roundtrip_cuda,got,input);
        }
    }
    Word bad=0;
    for(const auto &s:{matrix_tc,matrix_cuda,transform_tc,transform_cuda,roundtrip_tc,roundtrip_cuda})bad+=s.bad;
    print_checks("matrix_tc",matrix_tc);print_checks("matrix_cuda",matrix_cuda);
    print_checks("transform_tc",transform_tc);print_checks("transform_cuda",transform_cuda);
    print_checks("roundtrip_tc",roundtrip_tc);print_checks("roundtrip_cuda",roundtrip_cuda);
    std::printf("tc_gold_gate: fault=%d injected=%d bad=%llu\n",(int)fault,(int)injected,bad);
    return bad ? 3 : 0;
}
static void launch(int mode,int operation,int threads,Word batches,Word *a,Word *b,
                   Word *fm,Word *im,Word *ft,Word *it,Word scale)
{
    const auto grid=warp_grid(batches,threads);
    if(mode==1) {
        if(operation!=1)tc_goldilocks_mat16<<<grid,threads>>>(fm,a,b,batches);
        else tc_goldilocks_mat16<<<grid,threads>>>(im,a,b,batches);
        if(operation==2)tc_goldilocks_mat16<<<grid,threads>>>(im,b,a,batches);
    } else if(mode==0) {
        if(operation!=1)cuda_goldilocks_fft16<false><<<grid,threads>>>(a,b,batches,ft,1);
        else cuda_goldilocks_fft16<true><<<grid,threads>>>(a,b,batches,it,scale);
        if(operation==2)cuda_goldilocks_fft16<true><<<grid,threads>>>(b,a,batches,it,scale);
    } else {
        const auto dg=(unsigned int)((batches*128+threads-1)/threads);
        cuda_goldilocks_dft16<<<dg,threads>>>(operation==1 ? im : fm,a,b,batches);
        if(operation==2)cuda_goldilocks_dft16<<<dg,threads>>>(im,b,a,batches);
    }
}
static int bench(Word batches,int threads,int operation)
{
    Transform f(false),i(true);const auto patterns=fixture(5,16*128,0x789f);
    const auto expected=operation==2 ? patterns : reference(operation==1 ? i.matrix : f.matrix,patterns);
    Buffer a(batches*128*8),b(batches*128*8),fm(256*8),im(256*8),ft(15*8),it(15*8),pat(patterns.size()*8),want(expected.size()*8),bad(8);
    const auto upload=[](Buffer &dst,const std::vector<Word> &src) {
        CK(cudaMemcpy(dst.p,src.data(),dst.bytes,cudaMemcpyHostToDevice));
    };
    upload(fm,f.matrix);upload(im,i.matrix);upload(ft,f.table);upload(it,i.table);upload(pat,patterns);upload(want,expected);
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));int run=0;
    for(int mode:{0,1,1,0,1,0,0,1,2}) {
        double total=0;Word wrong=0;
        for(int repeat=0;repeat<4;++repeat) {
            const Word count=batches*128;
            fill_patterns<<<(unsigned int)((count+255)/256),256>>>(a.get(),count,pat.get());
            CK(cudaEventRecord(start));launch(mode,operation,threads,batches,a.get(),b.get(),fm.get(),im.get(),ft.get(),it.get(),i.scale);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
            if(repeat)total+=ms/1000.0;
            CK(cudaMemset(bad.p,0,8));check_patterns<<<(unsigned int)((count+255)/256),256>>>(operation==2 ? a.get() : b.get(),count,want.get(),bad.get());
            CK(cudaMemcpy(&wrong,bad.p,8,cudaMemcpyDeviceToHost));if(wrong)return 3;
        }
        std::printf("tc_gold_bench: run=%d mode=%d operation=%d threads=%d batches=%llu words=%llu seconds=%.9f bad=%llu\n",
            ++run,mode,operation,threads,batches,batches*128,total/3,wrong);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return 0;
}
template<class Kernel>static void resources(const char *name,Kernel kernel,int threads)
{
    cudaFuncAttributes a{};int blocks=0;CK(cudaFuncGetAttributes(&a,kernel));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,0));
    std::printf("tc_gold_resources: kernel=%s threads=%d regs=%d local=%zu shared=%zu max_blocks=%d\n",
        name,threads,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,blocks);
}
template<class Kernel>static void tile_resources(const char *name,Kernel kernel,int threads)
{
    cudaFuncAttributes a{};int blocks=0;CK(cudaFuncGetAttributes(&a,kernel));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,32768));
    std::printf("tc_gold_tile_resources: kernel=%s threads=%d regs=%d local=%zu dynamic_shared=32768 max_blocks=%d\n",
        name,threads,a.numRegs,a.localSizeBytes,blocks);
}
#ifndef NTT_TENSOR_GOLDILOCKS_PROBE_NO_MAIN
int main(int argc,char **argv)
{
    const int device=argc>1 ? std::atoi(argv[1]) : 1;const char *mode=argc>2 ? argv[2] : "--check";
    CK(cudaSetDevice(device));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    if(prop.major<8){std::fprintf(stderr,"requires sm80+ integer MMA\n");return 2;}
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::printf("tc_gold_device: device=%d name=%s sm=%d%d\n",device,prop.name,prop.major,prop.minor);
    for(int threads:{32,64,128,256}) {
        resources("tensor",tc_goldilocks_mat16,threads);resources("fft_forward",cuda_goldilocks_fft16<false>,threads);
        resources("fft_inverse",cuda_goldilocks_fft16<true>,threads);resources("scalar_dft",cuda_goldilocks_dft16,threads);
    }
    for(int threads:{128,256,512}) {
        tile_resources("tensor_forward",tc_goldilocks_tile<false>,threads);
        tile_resources("tensor_inverse",tc_goldilocks_tile<true>,threads);
    }
    tile_resources("cuda_forward",tile_kernel<false,true>,512);
    tile_resources("cuda_inverse",tile_kernel<true,true>,512);
    int code=2;
    if(!std::strcmp(mode,"--check") || !std::strcmp(mode,"--fault"))code=gate(!std::strcmp(mode,"--fault"));
    else if(!std::strcmp(mode,"--tile-check") || !std::strcmp(mode,"--tile-fault"))code=tile_gate(!std::strcmp(mode,"--tile-fault"));
    else if(!std::strcmp(mode,"--tile-bench") || !std::strcmp(mode,"--mixed-bench")) {
        const int k=argc>3 ? std::atoi(argv[3]) : 24,t=argc>4 ? std::atoi(argv[4]) : 12;
        const int threads=argc>5 ? std::atoi(argv[5]) : 256,operation=argc>6 ? std::atoi(argv[6]) : 0;
        const bool mixed=!std::strcmp(mode,"--mixed-bench");
        if(k>=t && k>=12 && k<=26 && t>=6 && t<=12 && (threads==128 || threads==256 || threads==512) &&
           operation>=0 && operation<=2 && (!mixed || operation==0))
            code=mixed ? mixed_bench(k,t,threads) : tile_bench(k,t,threads,operation);
    }
    else if(!std::strcmp(mode,"--bench")) {
        const Word batches=argc>3 ? std::strtoull(argv[3],nullptr,10) : 65536;
        const int threads=argc>4 ? std::atoi(argv[4]) : 128,operation=argc>5 ? std::atoi(argv[5]) : 0;
        if(batches && batches<=262144 && (threads==32 || threads==64 || threads==128 || threads==256) && operation>=0 && operation<=2)
            code=bench(batches,threads,operation);
    }
    CK(cudaDeviceSynchronize());
    std::printf("tc_gold_memory: requested_peak_bytes=%llu live_bytes=%llu\n",peak_bytes,live_bytes);
    return code ? code : live_bytes ? 4 : 0;
}
#endif
