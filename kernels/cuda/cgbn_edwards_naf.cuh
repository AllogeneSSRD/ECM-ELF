#pragma once

// GPU w-NAF dictionary and execution. Device state/dictionary remain in the
// Montgomery domain across slices; only the four checkpoint coordinates are
// converted on publication. The ordinary-domain API never exposes this state.
template<class P> __global__ void edwards_naf_prepare(cgbn_error_report_t *report,
        cgbn_mem_t<P::BITS> *data,cgbn_mem_t<P::BITS> *dictionary,
        uint32_t entries,uint32_t count) {
    using Context=cgbn_context_t<P::TPI,P>;
    using Env=cgbn_env_t<Context,P::BITS>;
    const uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/P::TPI;
    if(i>=count)return;
    Context ctx(cgbn_report_monitor,report,i);Env env(ctx);
    auto *slot=data+size_t(10)*i;
    typename Env::cgbn_t n,d,value;
    cgbn_load(env,n,slot);cgbn_load(env,d,slot+1);
    const uint32_t np0=cgbn_bn2mont(env,d,d,n);cgbn_store(env,slot+1,d);
    for(unsigned j=2;j<10;++j){cgbn_load(env,value,slot+j);cgbn_bn2mont(env,value,value,n);cgbn_store(env,slot+j,value);}
    cgbn_edwards_point<Env> odd,two;
    cgbn_load(env,odd.x,slot+2);cgbn_load(env,odd.y,slot+3);
    cgbn_load(env,odd.z,slot+4);cgbn_load(env,odd.t,slot+5);
    cgbn_edwards_field<Env> f{env,n,np0};f.dbl(two,odd);
    auto *dest=dictionary+size_t(i)*entries*4;
    for(uint32_t j=0;j<entries;++j) {
        cgbn_store(env,dest+4*j,odd.x);cgbn_store(env,dest+4*j+1,odd.y);
        cgbn_store(env,dest+4*j+2,odd.z);cgbn_store(env,dest+4*j+3,odd.t);
        if(j+1<entries)f.add_point(odd,odd,two,d);
    }
}

template<class Env> __device__ __forceinline__ void edwards_naf_double(
        const cgbn_edwards_field<Env> &f,cgbn_edwards_point<Env> &r,
        bool extended,int algorithm) {
    using bn=typename Env::cgbn_t;
    bn A,B,C,E,F,G,H,tmp;
    f.sqr(A,r.x);f.sqr(B,r.y);f.sqr(C,r.z);f.add(C,C,C);
    if(algorithm==5){f.mul(E,r.x,r.y);f.add(E,E,E);}
    else {f.add(tmp,r.x,r.y);f.sqr(E,tmp);f.sub(E,E,A);f.sub(E,E,B);}
    f.add(G,A,B);f.sub(F,G,C);f.sub(H,A,B);
    f.mul(r.x,E,F);f.mul(r.y,G,H);f.mul(r.z,F,G);
    // T is unnecessary until an addition or checkpoint boundary. The next
    // doubling uses X,Y,Z alone. Published points always satisfy XY=ZT.
    if(extended)f.mul(r.t,E,H);
}

template<class P> __global__ void edwards_naf_slice(cgbn_error_report_t *report,
        cgbn_mem_t<P::BITS> *data,cgbn_mem_t<P::BITS> *dictionary,
        cgbn_mem_t<P::BITS> *ordinary,const uint32_t *nonzero,const uint32_t *codes,
        uint32_t total_bits,uint32_t start,uint32_t stop,uint32_t entries,
        uint32_t window,uint32_t count,int algorithm) {
    using Context=cgbn_context_t<P::TPI,P>;
    using Env=cgbn_env_t<Context,P::BITS>;
    const uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/P::TPI;
    if(i>=count)return;
    Context ctx(cgbn_report_monitor,report,i);Env env(ctx);
    auto *slot=data+size_t(10)*i;
    typename Env::cgbn_t n,d,tmp,zero;
    cgbn_load(env,n,slot);cgbn_load(env,d,slot+1);
    // np0 depends only on the low limb of N, not on d's representation.
    const uint32_t np0=cgbn_binary_inverse_ui32(env,0u-cgbn_get_ui32(env,n));
    cgbn_set_ui32(env,zero,0);
    cgbn_edwards_point<Env> r,q;
    cgbn_load(env,r.x,slot+6);cgbn_load(env,r.y,slot+7);
    cgbn_load(env,r.z,slot+8);cgbn_load(env,r.t,slot+9);
    cgbn_edwards_field<Env> f{env,n,np0};
    for(uint32_t pos=start;pos<stop;++pos) {
        const uint32_t bit=total_bits-1-pos,word=bit/32,shift=bit%32;
        const bool add=(nonzero[word]>>shift)&1u;
        edwards_naf_double(f,r,add || pos+1==stop,algorithm);
        if(add) {
            const uint64_t packed=uint64_t(codes[word])|(uint64_t(codes[word+1])<<32);
            const uint32_t code=uint32_t(packed>>shift)&((1u<<(window-1))-1);
            auto *source=dictionary+(size_t(i)*entries+(code&(entries-1)))*4;
            cgbn_load(env,q.x,source);cgbn_load(env,q.y,source+1);
            cgbn_load(env,q.z,source+2);cgbn_load(env,q.t,source+3);
            if(code&entries){f.sub(q.x,zero,q.x);f.sub(q.t,zero,q.t);}
            f.add_point(r,r,q,d);
        }
    }
    cgbn_store(env,slot+6,r.x);cgbn_store(env,slot+7,r.y);
    cgbn_store(env,slot+8,r.z);cgbn_store(env,slot+9,r.t);
    cgbn_mont2bn(env,tmp,r.x,n,np0);cgbn_store(env,ordinary+size_t(i)*4,tmp);
    cgbn_mont2bn(env,tmp,r.y,n,np0);cgbn_store(env,ordinary+size_t(i)*4+1,tmp);
    cgbn_mont2bn(env,tmp,r.z,n,np0);cgbn_store(env,ordinary+size_t(i)*4+2,tmp);
    cgbn_mont2bn(env,tmp,r.t,n,np0);cgbn_store(env,ordinary+size_t(i)*4+3,tmp);
}
