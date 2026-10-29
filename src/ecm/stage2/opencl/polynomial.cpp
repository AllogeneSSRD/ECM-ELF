#include "polynomial.h"
#include "ecm/opencl/ntt/native.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ecm_opencl::stage2 {
namespace {
using Native=ntt::NativeAccess;
void check(cl_int code,const char *where){if(code!=CL_SUCCESS)throw std::runtime_error(std::string(where)+": OpenCL error "+std::to_string(code));}
template<class T>void arg(cl_kernel k,cl_uint i,const T &v){check(clSetKernelArg(k,i,sizeof(v),&v),"polynomial argument");}
std::size_t product(std::size_t a,std::size_t b) {
    if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("polynomial size overflow");return a*b;
}
unsigned bit_length(Word v){unsigned bits=0;for(;v;v>>=1)++bits;return bits;}
std::string read(const std::filesystem::path &p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
Word reciprocal(Word d) {
    // floor((2^128-1)/d)-2^64, d normalized. Exact restoring division,
    // performed once per plan; no host __int128 or GMP dependency.
    Word r=~d,q=0;
    for(unsigned i=0;i<64;++i){const bool high=(r>>63)!=0;r=(r<<1)|1;const bool take=high || r>=d;if(take)r-=d;q=(q<<1)|Word(take);}
    return q;
}
void view(const ntt::Runtime &r,PolynomialView &v,std::size_t words,std::size_t batch) {
    if(!v.stride)v.stride=words;
    if(!batch || v.stride<words)throw std::invalid_argument("invalid polynomial batch/stride");
    const auto tail=product(batch-1,v.stride);
    if(words>std::numeric_limits<std::size_t>::max()-tail)throw std::overflow_error("polynomial view span overflow");
    Native::span(r,v.buffer,tail+words,v.offset);
}
struct Drain {ntt::Runtime &r;~Drain(){try{r.finish();}catch(...){}}};
} // namespace

IntegerShape integer_shape(unsigned bits,std::size_t p,unsigned forced) {
    if(bits<2 || bits>16384 || !p || p>(std::size_t(1)<<28))throw std::invalid_argument("integer shape outside supported bit/capacity range");
    if(forced>62)throw std::invalid_argument("forced digit bits outside 1..62");
    IntegerShape s;s.coefficient_bits=bits;s.capacity=p;
    unsigned logp=1;while((std::size_t(1)<<logp)<p)++logp;
    s.slot_bits=2*std::size_t(bits)+logp;
    for(unsigned b=forced?forced:62;b;--b) {
        const Word mask=(Word(1)<<b)-1;const auto sw=(s.slot_bits+b-1)/b;
        const auto terms=product(p,sw);
        // L*mask^2 < q iff L <= floor((q-1)/mask^2).
        // Sequential unsigned division avoids overflow and is exact.
        if(terms>(ntt::modulus-1)/mask/mask){if(forced)break;continue;}
        const auto need=product(2,terms)+1;
        unsigned k=1;while(k<=29 && (std::size_t(1)<<k)<need)++k;
        if(k>29)break;
        s.digit_bits=b;s.slot_words=sw;s.length=std::size_t(1)<<k;s.log2_length=k;return s;
    }
    throw std::invalid_argument("no exact integer shape up to NTT 2^29");
}
struct PolynomialState {
    ntt::Runtime runtime;
    IntegerShape shape;
    std::unique_ptr<ntt::Plan> transform;
    ntt::Buffer modulus,normalized;
    std::shared_ptr<PolynomialWorkspace> workspace;
    std::array<cl_kernel,7> kernels{};
    cl_program program=nullptr;
    unsigned shift=0;
    Word inverse=0;
    std::size_t nw=0,group=0,reduce_group=0;
    std::string log;
    explicit PolynomialState(ntt::Runtime &r):runtime(r){}
    ~PolynomialState(){for(auto k:kernels)if(k)clReleaseKernel(k);if(program)clReleaseProgram(program);}
};
PolynomialPlan::PolynomialPlan(ntt::Runtime &runtime,const PolynomialOptions &o):state_(std::make_shared<PolynomialState>(runtime)) {
    auto &p=*state_;if(o.modulus.empty() || !o.modulus.back())throw std::invalid_argument("modulus must have minimal nonzero limbs");
    p.workspace=o.workspace;
    if(o.modulus.size()>256)throw std::invalid_argument("modulus exceeds 16384 bits");
    const unsigned bits=unsigned((o.modulus.size()-1)*64)+bit_length(o.modulus.back());
    p.shape=integer_shape(bits,o.capacity,o.forced_digit_bits);p.nw=o.modulus.size();
    if(o.mersenne)for(std::size_t i=0;i<p.nw;++i){const Word expected=i+1==p.nw && bits%64?((Word(1)<<(bits%64))-1):~Word(0);
        if(o.modulus[i]!=expected)throw std::invalid_argument("Mersenne reducer requires N == 2^S-1");}
    auto plan=o.transform;plan.log2_length=p.shape.log2_length;p.transform=std::make_unique<ntt::Plan>(runtime,plan);
    p.group=std::min(plan.work_group,p.shape.length);p.reduce_group=std::min(p.group,std::size_t(64));
    p.shift=64-bit_length(o.modulus.back());std::vector<Word> ns(p.nw);Word carry=0;
    for(std::size_t i=0;i<p.nw;++i){const Word v=o.modulus[i];ns[i]=p.shift?((v<<p.shift)|carry):v;carry=p.shift?v>>(64-p.shift):0;}
    p.inverse=reciprocal(ns.back());p.modulus=runtime.allocate(p.nw);p.normalized=runtime.allocate(p.nw);
    runtime.upload(p.modulus,o.modulus.data(),p.nw);runtime.upload(p.normalized,ns.data(),p.nw);
    const std::filesystem::path root=Native::resource_root(runtime);
    const auto source=read(root/"goldilocks.clh")+"\n#line 1 \"polynomial.cl\"\n"+read(root.parent_path()/"stage2/polynomial.cl");
    cl_int code=0;
    const auto options="-cl-std=CL1.2 -DECM_S4_NW="+std::to_string(p.nw)+" -DECM_S4_BITS="+std::to_string(bits)+
        " -DECM_S4_MERSENNE="+std::to_string(o.mersenne?1:0)+" -DECM_CARRY_GROUP="+std::to_string(p.group)+
        " -DECM_NTT_SPLIT_MUL="+std::to_string(plan.split_multiply?1:0);
    const auto device=Native::device(runtime);p.program=Native::program(runtime,source,options);
    std::size_t log_size=0;check(clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&log_size),"polynomial log size");
    std::string log(log_size,'\0');if(log_size)check(clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,log_size,log.data(),nullptr),"polynomial log");
    while(!log.empty() && log.back()=='\0')log.pop_back();p.log=options+"\n"+log;
    const std::array<const char*,7> names={"ecm_poly_pack","ecm_poly_inputs","ecm_carry_local","ecm_carry_blocks","ecm_carry_finish","ecm_poly_reduce","ecm_poly_pack"};
    for(std::size_t i=0;i<names.size();++i){p.kernels[i]=clCreateKernel(p.program,names[i],&code);check(code,names[i]);
        std::size_t limit=0;cl_ulong local=0;check(clGetKernelWorkGroupInfo(p.kernels[i],device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(limit),&limit,nullptr),"polynomial group limit");
        check(clGetKernelWorkGroupInfo(p.kernels[i],device,CL_KERNEL_LOCAL_MEM_SIZE,sizeof(local),&local,nullptr),"polynomial local memory");
        const auto group=i==3?1:(i==5?p.reduce_group:p.group);
        if(group>limit || local>runtime.device().local_bytes)throw std::runtime_error("polynomial kernel resource limit exceeded");}
}
const IntegerShape &PolynomialPlan::shape()const{return state_->shape;}
std::size_t PolynomialPlan::coefficient_words()const{return state_->nw;}
std::string PolynomialPlan::build_log()const{return state_->log;}
ProductInfo PolynomialPlan::multiply(PolynomialView a,std::size_t na,PolynomialView b,std::size_t nb,PolynomialView out,std::size_t batch) {
    auto &p=*state_;auto &r=p.runtime;const auto &s=p.shape;
    if(!na || !nb || na>s.capacity || nb>s.capacity)throw std::invalid_argument("polynomial coefficient count outside plan capacity");
    // Both packs only read their source and use distinct digit buffers. A/B
    // can therefore share a coefficient owner, including true squares.
    if(Native::same_owner(a.buffer,out.buffer) || Native::same_owner(b.buffer,out.buffer))
        throw std::invalid_argument("polynomial product requires a distinct output buffer");
    const auto nc=na+nb-1;
    view(r,a,product(na,p.nw),batch);view(r,b,product(nb,p.nw),batch);view(r,out,product(nc,p.nw),batch);
    const auto groups=s.length/p.group,fs=na+nb+nc;
    PolynomialWorkspace transient;auto &work=p.workspace?*p.workspace:transient;
    auto ensure=[&](ntt::Buffer &buffer,std::size_t words){
        if(buffer.words()<words){buffer={};buffer=r.allocate(words);}
        // Shared workspaces cannot be handed to a plan on a different context.
        Native::span(r,buffer,words,0);
    };
    auto admit=[&]{ensure(work.a_,product(s.length,batch));ensure(work.b_,product(s.length,batch));
        ensure(work.blocks_,product(product(3,groups),batch));ensure(work.flags_,product(fs,batch));};
    try{admit();}catch(const std::runtime_error &e){
        if(!p.workspace || std::string(e.what()).find("owned payload budget exceeded")==std::string::npos)throw;
        // Old capacities can belong to different (length,batch) shapes. They
        // are idle at this synchronous boundary; retry the exact request once.
        work.release();admit();
    }
    auto da=work.a_,db=work.b_,blocks=work.blocks_,flags=work.flags_;
    Drain drain{r}; // Error paths also wait before destroying borrowed owners.
    const auto queue=Native::queue(r);const Word n=s.length,sw=s.slot_words,slot_bits=s.slot_bits,ng=groups,flag_stride=fs;
    const cl_uint bpw=s.digit_bits;const Word wa=na,wb=nb,wc=nc;
    const auto ma=Native::memory(a.buffer),mb=Native::memory(b.buffer),mo=Native::memory(out.buffer);
    const auto md=Native::memory(da),ms=Native::memory(db),mg=Native::memory(blocks),mf=Native::memory(flags);
    auto launch=[&](cl_kernel k,std::size_t items,std::size_t group){const std::size_t local[]={group,1},global[]={((items+group-1)/group)*group,batch};
        check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"polynomial launch");};
    auto k=p.kernels[1];const Word ao=a.offset,ast=a.stride,bo=b.offset,bst=b.stride;
    arg(k,0,ma);arg(k,1,ao);arg(k,2,ast);arg(k,3,wa);arg(k,4,mb);arg(k,5,bo);arg(k,6,bst);arg(k,7,wb);arg(k,8,mf);arg(k,9,flag_stride);launch(k,na+nb,p.group);
    auto pack=[&](unsigned ki,cl_mem input,Word offset,Word stride,Word count,cl_mem dest){auto pk=p.kernels[ki];arg(pk,0,input);arg(pk,1,offset);arg(pk,2,stride);arg(pk,3,count);
        arg(pk,4,dest);arg(pk,5,n);arg(pk,6,sw);arg(pk,7,bpw);launch(pk,s.length,p.group);};
    pack(0,ma,ao,ast,wa,md);pack(6,mb,bo,bst,wb,ms);
    auto convolution=p.transform->convolve(da,db,batch);
    k=p.kernels[2];arg(k,0,md);arg(k,1,ms);arg(k,2,mg);arg(k,3,n);arg(k,4,ng);arg(k,5,bpw);launch(k,s.length,p.group);
    k=p.kernels[3];arg(k,0,mg);arg(k,1,ng);arg(k,2,md);arg(k,3,n);arg(k,4,bpw);const auto global=batch;check(clEnqueueNDRangeKernel(queue,k,1,nullptr,&global,nullptr,0,nullptr,nullptr),"carry block prefix");
    k=p.kernels[4];const Word active=nc*s.slot_words;arg(k,0,ms);arg(k,1,md);arg(k,2,mg);arg(k,3,n);arg(k,4,ng);arg(k,5,active);arg(k,6,bpw);launch(k,s.length,p.group);
    k=p.kernels[5];const auto mn=Native::memory(p.modulus),mns=Native::memory(p.normalized);const cl_int shift=cl_int(p.shift);
    const Word oo=out.offset,ost=out.stride,flag_offset=na+nb;
    arg(k,0,md);arg(k,1,n);arg(k,2,bpw);arg(k,3,sw);arg(k,4,slot_bits);arg(k,5,wc);arg(k,6,mn);arg(k,7,mns);arg(k,8,shift);arg(k,9,p.inverse);
    arg(k,10,mo);arg(k,11,oo);arg(k,12,ost);arg(k,13,mf);arg(k,14,flag_stride);arg(k,15,flag_offset);launch(k,nc,p.reduce_group);
    check(clFlush(queue),"polynomial flush");
    std::vector<Word> verdict(product(fs,batch)),carry_flags(product(product(3,groups),batch));
    r.download(flags,verdict.data(),verdict.size());r.download(blocks,carry_flags.data(),carry_flags.size());convolution.wait();
    for(std::size_t i=0;i<verdict.size();++i)if(verdict[i])throw std::runtime_error("OpenCL polynomial verdict failed at "+std::to_string(i)+" flags="+std::to_string(verdict[i]));
    for(std::size_t j=0;j<batch;++j)for(std::size_t g=0;g<groups;++g)if(carry_flags[j*3*groups+2*groups+g])
        throw std::runtime_error("OpenCL carry overflow/nonzero padding at batch="+std::to_string(j)+" block="+std::to_string(g));
    return {nc,batch,Word(verdict.size()),Word(groups*batch)};
}
} // namespace ecm_opencl::stage2
