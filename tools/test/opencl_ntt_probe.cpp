#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#include "ecm/opencl/ntt/runtime.h"
#include "ecm/opencl/ntt/native.h"
#include "ecm/platform/program_version.h"
#include <gmp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt=ecm_opencl::ntt;
using Word=ntt::Word;
namespace {
struct Big {mpz_t v;Big(){mpz_init(v);}~Big(){mpz_clear(v);}Big(const Big&)=delete;};
void set(Big &z,Word v){mpz_import(z.v,1,-1,sizeof(v),0,0,&v);}
Word word(const Big &z){Word v=0;std::size_t n=0;if(mpz_sizeinbase(z.v,2)>64)throw std::runtime_error("oracle value exceeds 64 bits");mpz_export(&v,&n,-1,sizeof(v),0,0,z.v);return v;}
Word oracle(Word a,Word b,unsigned op) {
    Big x,y,q,z;set(x,a);set(y,b);set(q,ntt::modulus);
    if(op==0)mpz_add(z.v,x.v,y.v);
    else if(op==1)mpz_sub(z.v,x.v,y.v);
    else if(op==2 || op==4){mpz_mul(z.v,x.v,y.v);if(op==4){mpz_fdiv_q_2exp(z.v,z.v,64);return word(z);}}
    else {mpz_mul_2exp(z.v,y.v,64);mpz_add(z.v,z.v,x.v);}
    mpz_mod(z.v,z.v,q.v);return word(z);
}
Word root(unsigned k) {
    Big a,e,q,z;set(a,7);set(q,ntt::modulus);set(e,(ntt::modulus-1)/(Word(1)<<k));
    mpz_powm(z.v,a.v,e.v,q.v);return word(z);
}
Word power(Word a,Word e){Big x,y,q,z;set(x,a);set(y,e);set(q,ntt::modulus);mpz_powm(z.v,x.v,y.v,q.v);return word(z);}
Word random_word(Word &s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
std::string quote(const std::string &s){std::string out="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c<32)out+='?';else out+=char(c);}return out+'"';}
void clcheck(cl_int c,const char *where){if(c!=CL_SUCCESS)throw std::runtime_error(std::string(where)+": OpenCL error "+std::to_string(c));}
std::string text_file(const std::filesystem::path &p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
template<class T>void argument(cl_kernel k,unsigned i,const T &v){clcheck(clSetKernelArg(k,i,sizeof(v),&v),"arithmetic argument");}

struct Fixture {
    cl_context context=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;cl_kernel kernel=nullptr;
    std::array<cl_mem,3> memory{};
    ~Fixture(){if(queue)(void)clFinish(queue);for(auto m:memory)if(m)clReleaseMemObject(m);if(kernel)clReleaseKernel(kernel);
        if(program)clReleaseProgram(program);if(queue)clReleaseCommandQueue(queue);if(context)clReleaseContext(context);}
};
struct Counts {Word arithmetic=0,roundtrip=0,convolution=0,direct=0,guards=0,plans=0;};

void arithmetic(std::size_t index,const std::filesystem::path &resources,bool split,Counts &counts) {
    cl_uint np=0;clcheck(clGetPlatformIDs(0,nullptr,&np),"fixture platforms");std::vector<cl_platform_id> ps(np);
    clcheck(clGetPlatformIDs(np,ps.data(),nullptr),"fixture platform list");std::vector<cl_device_id> ds;
    for(auto p:ps){cl_uint nd=0;const auto c=clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,0,nullptr,&nd);if(c==CL_DEVICE_NOT_FOUND)continue;
        clcheck(c,"fixture device count");std::vector<cl_device_id> row(nd);clcheck(clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,nd,row.data(),nullptr),"fixture devices");ds.insert(ds.end(),row.begin(),row.end());}
    if(index>=ds.size())throw std::runtime_error("fixture device unavailable");const auto d=ds[index];Fixture f;cl_int c=0;
    f.context=clCreateContext(nullptr,1,&d,nullptr,nullptr,&c);clcheck(c,"fixture context");
    f.queue=clCreateCommandQueue(f.context,d,0,&c);clcheck(c,"fixture queue");
    const auto source=text_file(resources/"goldilocks.clh")+R"CLC(
__kernel void arithmetic(__global const ulong *a,__global const ulong *b,__global ulong *out) {
    const size_t i=get_global_id(0);const ulong x=a[i],y=b[i];
    const ulong ca=x>=ECM_GL_Q?x-ECM_GL_Q:x,cb=y>=ECM_GL_Q?y-ECM_GL_Q:y;
    out[5*i+0]=ecm_gl_add(ca,cb);out[5*i+1]=ecm_gl_sub(ca,cb);
    out[5*i+2]=ecm_gl_mul(x,y);out[5*i+3]=ecm_gl_reduce(x,y);
    out[5*i+4]=ecm_gl_product_hi(x,y);
}
)CLC";
    const char *src=source.c_str();const auto len=source.size();f.program=clCreateProgramWithSource(f.context,1,&src,&len,&c);clcheck(c,"fixture program");
    const auto options="-cl-std=CL1.2 -DECM_NTT_SPLIT_MUL="+std::to_string(split?1:0);
    c=clBuildProgram(f.program,1,&d,options.c_str(),nullptr,nullptr);
    if(c!=CL_SUCCESS){std::size_t n=0;clGetProgramBuildInfo(f.program,d,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);std::string log(n,'\0');
        clGetProgramBuildInfo(f.program,d,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);throw std::runtime_error("arithmetic compile: "+log);}
    f.kernel=clCreateKernel(f.program,"arithmetic",&c);clcheck(c,"fixture kernel");
    const std::array<Word,15> edge={0,1,2,3,0xffffffffULL,0x100000000ULL,0x100000001ULL,
        0x7fffffffffffffffULL,0x8000000000000000ULL,0xfffffffeffffffffULL,ntt::modulus-2,ntt::modulus-1,
        ntt::modulus,ntt::modulus+1,~Word(0)};
    std::vector<Word> a,b;for(auto x:edge)for(auto y:edge){a.push_back(x);b.push_back(y);}
    Word seed=0x89264c132f07ab51ULL;
    for(unsigned i=0;i<8192;++i){a.push_back(random_word(seed));b.push_back(random_word(seed));}
    for(unsigned i=0;i<64;++i){a.push_back(Word(1)<<i);b.push_back(~(Word(1)<<i));}
    std::vector<Word> out(5*a.size());
    for(unsigned i=0;i<3;++i){const auto bytes=(i==2?out.size():a.size())*sizeof(Word);f.memory[i]=clCreateBuffer(f.context,CL_MEM_READ_WRITE,bytes,nullptr,&c);clcheck(c,"fixture allocation");argument(f.kernel,i,f.memory[i]);}
    clcheck(clEnqueueWriteBuffer(f.queue,f.memory[0],CL_TRUE,0,a.size()*8,a.data(),0,nullptr,nullptr),"fixture input A");
    clcheck(clEnqueueWriteBuffer(f.queue,f.memory[1],CL_TRUE,0,b.size()*8,b.data(),0,nullptr,nullptr),"fixture input B");
    const auto global=a.size();clcheck(clEnqueueNDRangeKernel(f.queue,f.kernel,1,nullptr,&global,nullptr,0,nullptr,nullptr),"fixture launch");
    clcheck(clEnqueueReadBuffer(f.queue,f.memory[2],CL_TRUE,0,out.size()*8,out.data(),0,nullptr,nullptr),"fixture output");
    for(std::size_t i=0;i<a.size();++i)for(unsigned op=0;op<5;++op){
        const auto expected=oracle(a[i],b[i],op);if(out[5*i+op]!=expected)throw std::runtime_error("Goldilocks mismatch case="+std::to_string(i)+" op="+std::to_string(op)+" split="+std::to_string(split));
        ++counts.arithmetic;
    }
    std::cout<<"opencl_arithmetic: multiply="<<(split?"split32":"mul_hi")<<" checked="<<out.size()<<" bad=0\n";
}

template<class F>void rejects(const char *name,F action,Counts &counts){try{action();}catch(const std::exception&){++counts.guards;return;}throw std::runtime_error(std::string("missing rejection: ")+name);}

void program_cache(const ntt::Options&options,Counts&counts){
    // Exercise the real program-creation seam, including both key components
    // and context isolation. All transform tests still check actual outputs.
    const std::string source="__kernel void cache_probe(__global ulong*x){x[get_global_id(0)]+=1UL;}";
    const std::string flags="-cl-std=CL1.2";
    struct Owners{std::vector<cl_program> values;~Owners(){for(auto p:values)clReleaseProgram(p);}} owners;
    auto get=[&](ntt::Runtime&r,const std::string&s,const std::string&o){auto p=ntt::NativeAccess::program(r,s,o);owners.values.push_back(p);return p;};
    ntt::Runtime r(options);const auto first=get(r,source,flags);
    if(get(r,source,flags)!=first)throw std::runtime_error("identical program was not reused");
    if(get(r,source+"\n// different source",flags)==first)throw std::runtime_error("program cache ignored source");
    if(get(r,source,flags+" -DCACHE_VARIANT=1")==first)throw std::runtime_error("program cache ignored options");
    const auto stats=r.programs();if(stats.builds!=3||stats.hits!=1)throw std::runtime_error("program cache accounting");
    ntt::Runtime other(options);if(get(other,source,flags)==first||other.programs().builds!=1||other.programs().hits)throw std::runtime_error("program escaped its context");
    auto disabled=options;disabled.cache_programs=false;ntt::Runtime uncached(disabled);
    get(uncached,source,flags);get(uncached,source,flags);if(uncached.programs().builds!=2||uncached.programs().hits)throw std::runtime_error("disabled program cache reused a program");
    counts.guards+=6;
}
void unchanged_padding(const std::vector<Word> &before,const std::vector<Word> &after,
                       std::size_t n,std::size_t batch,std::size_t stride,std::size_t offset) {
    for(std::size_t i=0;i<before.size();++i){const auto local=i>=offset?(i-offset)%stride:stride;
        const bool active=i>=offset && (i-offset)/stride<batch && local<n;
        if(!active && before[i]!=after[i])throw std::runtime_error("NTT wrote outside view");}
}
void ntt_case(ntt::Runtime &runtime,unsigned k,std::size_t batch,std::size_t group,unsigned tile,bool split,Counts &counts) {
    ntt::PlanOptions options;options.log2_length=k;options.work_group=group;options.tile_bits=tile;options.split_multiply=split;
    ntt::Plan plan(runtime,options);const auto n=plan.length();const std::size_t offset=3,stride=n+7,total=offset+batch*stride+5;
    auto a=runtime.allocate(total),b=runtime.allocate(total);std::vector<Word> input(total,0xbad1234511223344ULL),after(total);
    Word seed=0x72b3ce1a98347612ULL+k;
    for(std::size_t slice=0;slice<batch;++slice)for(std::size_t i=0;i<n;++i){auto v=random_word(seed);if(v>=ntt::modulus)v-=ntt::modulus;input[offset+slice*stride+i]=v;}
    runtime.upload(a,input.data(),total);auto forward=plan.forward(a,batch,stride,offset);
    if(k<=5){
        runtime.download(a,after.data(),total);const auto omega=root(k);
        for(std::size_t slice=0;slice<batch;++slice)for(std::size_t frequency=0;frequency<n;++frequency){
            Word sum=0;for(std::size_t j=0;j<n;++j)sum=oracle(sum,oracle(input[offset+slice*stride+j],power(omega,Word(j*frequency)),2),0);
            std::size_t reversed=0,v=frequency;for(unsigned bit=0;bit<k;++bit){reversed=(reversed<<1)|(v&1);v>>=1;}
            if(after[offset+slice*stride+reversed]!=sum)throw std::runtime_error("forward DFT ordering/value mismatch");++counts.direct;
        }
    }
    auto inverse=plan.inverse(a,batch,stride,offset);inverse.wait();runtime.download(a,after.data(),total);
    if(input!=after)throw std::runtime_error("NTT round-trip or padding mismatch");counts.roundtrip+=n*batch;
    std::vector<Word> left(total,0x91aa23bc11889977ULL),right(total,0x19bb54ce11002233ULL),expected(n*batch,0);
    const auto ac=std::min<std::size_t>(17,n),bc=std::min<std::size_t>(19,n);
    for(std::size_t slice=0;slice<batch;++slice){
        const auto base=offset+slice*stride;std::fill_n(left.begin()+base,n,0);std::fill_n(right.begin()+base,n,0);
        for(std::size_t j=0;j<ac;++j)left[base+j*n/ac]=random_word(seed)%ntt::modulus;
        for(std::size_t j=0;j<bc;++j)right[base+j*n/bc]=random_word(seed)%ntt::modulus;
        for(std::size_t i=0;i<ac;++i)for(std::size_t j=0;j<bc;++j){const auto x=i*n/ac,y=j*n/bc,at=slice*n+(x+y)%n;
            expected[at]=oracle(expected[at],oracle(left[base+x],right[base+y],2),0);}
    }
    runtime.upload(a,left.data(),total);runtime.upload(b,right.data(),total);
    auto convolution=plan.convolve(a,b,batch,stride,offset);convolution.wait();runtime.download(a,after.data(),total);
    for(std::size_t slice=0;slice<batch;++slice)for(std::size_t i=0;i<n;++i){if(after[offset+slice*stride+i]!=expected[slice*n+i])
        throw std::runtime_error("convolution mismatch k="+std::to_string(k)+" slice="+std::to_string(slice)+" i="+std::to_string(i));++counts.convolution;}
    unchanged_padding(left,after,n,batch,stride,offset);runtime.download(b,after.data(),total);unchanged_padding(right,after,n,batch,stride,offset);
    rejects("batch=0",[&]{plan.forward(a,0);},counts);rejects("short stride",[&]{plan.forward(a,1,n-1);},counts);
    rejects("oversize span",[&]{plan.forward(a,batch+2,stride,offset);},counts);rejects("same owner",[&]{plan.convolve(a,a);},counts);
    rejects("overflow span",[&]{plan.forward(a,std::numeric_limits<std::size_t>::max(),n);},counts);
    ++counts.plans;std::cout<<"opencl_ntt_verified: log2="<<k<<" batch="<<batch<<" group="<<group<<" tile="<<plan.tile_bits()
        <<" multiply="<<(split?"split32":"mul_hi")<<" convolution_event_seconds="<<convolution.seconds()<<" bad=0\n";
}
Word number(const std::string &s){if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("unsigned integer required: "+s);return std::stoull(s);}
} // namespace

int main(int argc,char **argv) {
    std::cout<<std::unitbuf;std::cerr<<std::unitbuf;
    try {
        std::size_t device=0;unsigned max_log2=16;Word budget_mb=256;bool verify=false;
        std::string output,resources=ECM_OPENCL_NTT_RESOURCE_ROOT;
        for(int i=1;i<argc;++i){const std::string arg=argv[i];
            if(arg=="--list")continue;if(arg=="--verify"){verify=true;continue;}
            if(arg=="--help"){std::cout<<"opencl_ntt_probe [--list] [--verify --device N --max-log2 1..20 --budget-mb N --output FILE --resource-root DIR]\n";return 0;}
            if(i+1>=argc)throw std::invalid_argument("missing value for "+arg);const std::string value=argv[++i];
            if(arg=="--device")device=std::size_t(number(value));else if(arg=="--max-log2"){
                const auto n=number(value);if(n<1 || n>20)throw std::invalid_argument("verification max-log2 outside 1..20");max_log2=unsigned(n);}
            else if(arg=="--budget-mb")budget_mb=number(value);else if(arg=="--output")output=value;
            else if(arg=="--resource-root")resources=value;else throw std::invalid_argument("unknown option "+arg);
        }
        if(max_log2<1 || max_log2>20)throw std::invalid_argument("verification max-log2 outside 1..20");
        if(!budget_mb || budget_mb>(std::numeric_limits<Word>::max()>>20))throw std::invalid_argument("invalid payload budget");
        if(!output.empty() && std::filesystem::exists(output))throw std::invalid_argument("verification output already exists; use a fresh evidence path");
        const auto list=ntt::devices();for(const auto &d:list)std::cout<<"opencl_device: index="<<d.index<<" platform="<<quote(d.platform)<<" name="<<quote(d.name)
            <<" vendor="<<quote(d.vendor)<<" version="<<quote(d.version)<<" driver="<<quote(d.driver)
            <<" profile="<<d.profile<<" address_bits="<<d.address_bits<<" local_bytes="<<d.local_bytes
            <<" max_alloc_bytes="<<d.max_allocation_bytes<<" global_bytes="<<d.global_bytes
            <<" max_group="<<d.max_work_group<<" supported="<<d.supported<<" reason="<<quote(d.reason)<<'\n';
        if(!verify)return 0;
        if(device>=list.size() || !list[device].supported)throw std::invalid_argument("selected OpenCL device unsupported/unavailable");
        Counts counts;const auto started=std::chrono::steady_clock::now();
        for(bool split:{false,true})arithmetic(device,resources,split,counts);
        ntt::Options options;options.device=device;options.budget_bytes=budget_mb<<20;options.resource_root=resources;
        program_cache(options,counts);
        ntt::Runtime runtime(options);
        for(bool split:{false,true})for(unsigned k:std::vector<unsigned>{1,2,3,4,5,8,10,12,16,20}){
            if(k>max_log2)continue;for(std::size_t batch:{std::size_t(1),std::size_t(3)})ntt_case(runtime,k,batch,128,8,split,counts);
            if(k==8 || k==12)ntt_case(runtime,k,3,64,5,split,counts);
        }
        // The completion must keep both the borrowed buffer and plan alive.
        ntt::Completion pending;
        {ntt::PlanOptions p;p.log2_length=10;ntt::Plan plan(runtime,p);auto a=runtime.allocate(plan.length());
            std::vector<Word> ones(plan.length(),1);runtime.upload(a,ones.data(),ones.size());pending=plan.forward(a);}
        pending.wait();if(!runtime.memory().live_bytes)throw std::runtime_error("submission lease released too early");
        pending={};if(runtime.memory().live_bytes)throw std::runtime_error("OpenCL payload leak after completion");++counts.guards;
        ntt::PlanOptions bad;bad.log2_length=0;rejects("invalid length",[&]{ntt::Plan p(runtime,bad);},counts);
        bad.log2_length=10;bad.work_group=3;rejects("invalid group",[&]{ntt::Plan p(runtime,bad);},counts);
        rejects("empty allocation",[&]{runtime.allocate(0);},counts);
        rejects("single-object limit",[&]{runtime.allocate(std::size_t(runtime.device().max_allocation_bytes/8+1));},counts);
        {auto a=runtime.allocate(2);Word value=1;rejects("upload view",[&]{runtime.upload(a,&value,2,1);},counts);
            rejects("null upload",[&]{runtime.upload(a,nullptr,1);},counts);}
        auto tiny=options;tiny.budget_bytes=8;ntt::Runtime small(tiny);rejects("payload budget",[&]{small.allocate(2);},counts);
        {ntt::PlanOptions p;ntt::Plan plan(runtime,p);auto foreign=small.allocate(1);rejects("foreign owner",[&]{plan.forward(foreign);},counts);}
        runtime.finish();const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::ostringstream report;report<<"{\"status\":\"passed\",\"program_version\":"<<quote(ecm_program::version)
            <<",\"backend\":\"native-opencl-goldilocks64\",\"device\":"<<quote(runtime.device().name)
            <<",\"vendor\":"<<quote(runtime.device().vendor)<<",\"driver\":"<<quote(runtime.device().driver)
            <<",\"source_fnv\":"<<quote(runtime.source_identity())<<",\"arithmetic_values\":"<<counts.arithmetic
            <<",\"roundtrip_words\":"<<counts.roundtrip<<",\"convolution_words\":"<<counts.convolution
            <<",\"direct_dft_words\":"<<counts.direct<<",\"guard_cases\":"<<counts.guards<<",\"plans\":"<<counts.plans
            <<",\"payload_peak_bytes\":"<<runtime.memory().peak_bytes<<",\"verification_wall_seconds\":"<<elapsed
            <<",\"full_stage2\":false,\"bad\":0}";
        if(!output.empty()){std::ofstream f(output,std::ios::binary);if(!f)throw std::runtime_error("cannot write verification output");f<<report.str()<<'\n';f.close();if(!f)throw std::runtime_error("verification output write failed");}
        std::cout<<"opencl_ntt_summary: "<<report.str()<<'\n';return 0;
    }catch(const std::exception &e){std::cerr<<"opencl_ntt_failed: "<<e.what()<<'\n';return 2;}
}
