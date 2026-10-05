#pragma once
#include <sstream>
#include <iomanip>

namespace stage2_tune {
inline std::string quote(const char *s) {
    std::string out="\"";
    for(;*s;++s) {
        const unsigned char c=(unsigned char)*s;
        if(c=='"' || c=='\\') { out+='\\';out+=(char)c; }
        else if(c<32) { char b[7];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b; }
        else out+=(char)c;
    }
    return out+'"';
}
__global__ void seed(unsigned long long *a,unsigned long long *b,unsigned long long n) {
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n) {
        a[i]=i==0 ? GL_P-1 : i==1 ? 0x8000000000000000ull : i==2 ? GL_P-2 : 0;
        b[i]=i==0 ? 3 : i==1 ? 5 : i==2 ? 7 : 0;
    }
}
__global__ void verdict(const unsigned long long *a,unsigned long long n,
                        const unsigned long long *want,unsigned long long *bad) {
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n && a[i]!=(i<5 ? want[i] : 0))atomicAdd(bad,1ull);
}
struct Resources {
    unsigned long long *a=nullptr,*b=nullptr,*want=nullptr,*bad=nullptr;
    cudaEvent_t start=nullptr,end=nullptr;
    NttArena arena;
    ~Resources() {
        // Finish readers before releasing the tables and their input arrays.
        cudaDeviceSynchronize(); arena.release();
        if(a)cudaFree(a);if(b)cudaFree(b);if(want)cudaFree(want);if(bad)cudaFree(bad);
        if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);
    }
};
static int run(int device,int first,int last,int repeats,unsigned long long memory,
               void (*report)(const char*,void*),void *context) {
    if(!report || device<0 || first<16 || last>27 || first>last ||
       repeats<1 || repeats>1000 || !memory)return 2;
    if(NTT_GL_FIXED_MODE<0 || fuse_trace_on()) {
        std::fprintf(stderr,"NTT tune requires a fixed Goldilocks build and NTT_FUSE_TRACE=0\n");
        return 2;
    }
    CK(cudaSetDevice(device));CK(cudaFree(0));ntt_gl_reduce_configure();
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    int runtime=0,driver=0;CK(cudaRuntimeGetVersion(&runtime));CK(cudaDriverGetVersion(&driver));
    char uuid[33]={};for(int i=0;i<16;++i)std::snprintf(uuid+2*i,3,"%02x",(unsigned char)prop.uuid.bytes[i]);
    std::ostringstream identity;
    identity<<"{\"type\":\"device\",\"device_index\":"<<device<<",\"name\":"<<quote(prop.name)
        <<",\"uuid_hex\":\""<<uuid<<"\",\"sm_major\":"<<prop.major<<",\"sm_minor\":"<<prop.minor
        <<",\"cuda_runtime\":"<<runtime<<",\"cuda_driver\":"<<driver
        <<",\"gl_fixed_mode\":"<<NTT_GL_FIXED_MODE<<",\"outer_unroll_u\":"<<NTT_OUTER_UNROLL_U
        <<",\"memory_budget_bytes\":"<<memory<<",\"accounting_version\":2}";
    report(identity.str().c_str(),context);
    unsigned long long expected[5]={},aa[3]={GL_P-1,0x8000000000000000ull,GL_P-2},bb[3]={3,5,7};
    mpz_t p,z,x,y;mpz_inits(p,z,x,y,nullptr);
    const auto prime=GL_P;mpz_import(p,1,-1,8,0,0,&prime);
    for(int c=0;c<5;++c) {
        mpz_set_ui(z,0);
        for(int i=0;i<3;++i)if(c-i>=0 && c-i<3) {
            mpz_import(x,1,-1,8,0,0,aa+i);mpz_import(y,1,-1,8,0,0,bb+c-i);mpz_addmul(z,x,y);
        }
        mpz_mod(z,z,p);size_t count=0;mpz_export(expected+c,&count,-1,8,0,0,z);
    }
    mpz_clears(p,z,x,y,nullptr);
    int measured=0,skipped=0;
    for(int k=first;k<=last;++k) {
        const auto n=1ull<<k,om=gl_pow_host(7,(GL_P-1)/n),omi=gl_pow_host(om,GL_P-2);
        const auto scale=gl_pow_host(n,GL_P-2);
        FuseCtx description;fuse_describe(description,n,k,om,omi);
        const auto base=fuse_planned_base_words(description)*8;
        const auto tables=fuse_planned_table_words(description)*8;
        const auto payload=16*n+base+tables+48;
        size_t free=0,total=0;CK(cudaMemGetInfo(&free,&total));
        const size_t reserve=768ull<<20;
        const auto available=free>reserve ? free-reserve : 0;
        if(payload>memory || payload>available) {
            std::ostringstream row;row<<"{\"type\":\"sample\",\"status\":\"skipped_memory\",\"log2_length\":"
                <<k<<",\"length\":"<<n<<",\"payload_bytes\":"<<payload<<",\"free_bytes\":"<<free
                <<",\"reserve_bytes\":"<<reserve<<"}";
            report(row.str().c_str(),context);++skipped;continue;
        }
        Resources r;r.arena.device=device;r.arena.cap_bytes=memory-16*n-48;
        const double setup_start=now_s();
        CK(cudaMalloc(&r.a,n*8));CK(cudaMalloc(&r.b,n*8));
        CK(cudaMalloc(&r.want,40));CK(cudaMalloc(&r.bad,8));
        CK(cudaMemcpy(r.want,expected,40,cudaMemcpyHostToDevice));
        FuseCtx fc;ntt_arena_fuse(&r.arena,n,k,om,omi,fc);
        FuseCallGuard guard(fc);
        CK(cudaDeviceSynchronize());const double setup=now_s()-setup_start;
        CK(cudaEventCreate(&r.start));CK(cudaEventCreate(&r.end));
        std::vector<double> samples;double check_seconds=0,warmup=0;
        for(int repeat=0;repeat<=repeats;++repeat) {
            seed<<<(unsigned int)((n+255)/256),256>>>(r.a,r.b,n);CK(cudaGetLastError());
            CK(cudaEventRecord(r.start));
            ntt_forward_fused(r.a,fc,om);ntt_forward_fused(r.b,fc,om);
            ntt_inverse_fused(r.a,r.b,fc,omi,scale);
            CK(cudaEventRecord(r.end));CK(cudaEventSynchronize(r.end));
            float ms=0;CK(cudaEventElapsedTime(&ms,r.start,r.end));
            if(!(ms>0) || !std::isfinite(ms))return 3;
            if(repeat)samples.push_back(ms/1000.0);else warmup=ms/1000.0;
            const double check_start=now_s();
            CK(cudaMemset(r.bad,0,8));
            verdict<<<(unsigned int)((n+255)/256),256>>>(r.a,n,r.want,r.bad);CK(cudaGetLastError());
            unsigned long long bad=0;CK(cudaMemcpy(&bad,r.bad,8,cudaMemcpyDeviceToHost));
            check_seconds+=now_s()-check_start;
            if(bad) {
                std::ostringstream row;row<<"{\"type\":\"sample\",\"status\":\"failed_verification\",\"length\":"
                    <<n<<",\"bad\":"<<bad<<"}";report(row.str().c_str(),context);return 3;
            }
        }
        auto sorted=samples;std::sort(sorted.begin(),sorted.end());
        const size_t mid=sorted.size()/2;
        const double median=sorted.size()%2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
        std::ostringstream row;row<<std::setprecision(17)
            <<"{\"type\":\"sample\",\"status\":\"measured\",\"unit\":\"field_convolution\",\"batch\":1,\"log2_length\":"
            <<k<<",\"length\":"<<n<<",\"median_seconds\":"<<median<<",\"conv_iter_per_s\":"<<1/median
            <<",\"min_seconds\":"<<sorted.front()<<",\"max_seconds\":"<<sorted.back()
            <<",\"setup_seconds\":"<<setup<<",\"warmup_seconds\":"<<warmup
            <<",\"verification_seconds\":"<<check_seconds<<",\"bad\":0,\"verified_words_per_sample\":"<<n
            <<",\"external_arrays_bytes\":"<<16*n+48<<",\"arena_payload_bytes\":"<<r.arena.bytes
            <<",\"payload_bytes\":"<<payload<<",\"tile\":"<<fc.t<<",\"max_radix_bits\":"<<fc.m_max
            <<",\"coop_outer\":"<<(fc.coop_outer ? "true" : "false")<<",\"warp_tail\":"<<(fc.warp_tail ? "true" : "false")
            <<",\"compact_scratch\":"<<(fc.compact_scratch ? "true" : "false")<<",\"passes_forward\":"<<fc.passes_fwd
            <<",\"outer_radix_bits\":[";
        for(int i=0;i<fc.nms;++i){if(i)row<<',';row<<fc.ms[i];}
        row<<"],\"seconds\":[";
        for(size_t i=0;i<samples.size();++i){if(i)row<<',';row<<samples[i];}row<<"]}";
        report(row.str().c_str(),context);++measured;
    }
    std::ostringstream end;end<<"{\"type\":\"complete\",\"measured\":"<<measured<<",\"skipped\":"<<skipped
        <<",\"failed\":0,\"usable\":"<<(measured ? "true" : "false")<<"}";
    report(end.str().c_str(),context);
    return measured ? 0 : 2;
}
} // namespace stage2_tune
