#include "ecm/stage2/opencl/polynomial.h"
#include "ecm/opencl/ntt/native.h"
#include "ecm/platform/program_version.h"
#include <gmp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ntt=ecm_opencl::ntt;
namespace s4=ecm_opencl::stage2;
using Native=ntt::NativeAccess;
using Word=ntt::Word;
namespace {
struct Big {mpz_t v;Big(){mpz_init(v);}~Big(){mpz_clear(v);}Big(const Big&)=delete;};
void set(Big &z,Word v){mpz_import(z.v,1,-1,8,0,0,&v);}
Word word(const Big &z){Word v=0;std::size_t n=0;if(mpz_sizeinbase(z.v,2)>64)throw std::runtime_error("oracle exceeds word");mpz_export(&v,&n,-1,8,0,0,z.v);return v;}
void load(Big &z,const Word *p,std::size_t n){mpz_import(z.v,n,-1,8,0,0,p);}
std::vector<Word> limbs(const Big &z,std::size_t n){std::vector<Word> out(n);std::size_t count=0;if(mpz_sizeinbase(z.v,2)>n*64)throw std::runtime_error("oracle exceeds limb buffer");mpz_export(out.data(),&count,-1,8,0,0,z.v);return out;}
Word random(Word &s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
std::string quote(const std::string &s){std::string out="\"";for(unsigned char c:s){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c<32)out+='?';else out+=char(c);}return out+'"';}
void check(cl_int c,const char *s){if(c!=CL_SUCCESS)throw std::runtime_error(std::string(s)+": OpenCL "+std::to_string(c));}
template<class T>void arg(cl_kernel k,unsigned i,const T &v){check(clSetKernelArg(k,i,sizeof(v),&v),"fixture argument");}
std::string read(const std::filesystem::path &p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
struct Counts {Word shapes=0,carry_words=0,reduction_words=0,integer_coefficients=0,polynomial_words=0,products=0,plans=0,guards=0;};
template<class F>void rejects(const char *tag,const char *message,F f,Counts &counts){try{f();}catch(const std::exception &e){if(std::string(e.what()).find(message)==std::string::npos)throw std::runtime_error(std::string("wrong rejection ")+tag+": "+e.what());++counts.guards;return;}throw std::runtime_error(std::string("missing rejection ")+tag);}
void shapes(Counts &counts) {
    // Independent original CUDA formula using GMP, including bpw=32 and overflow gates.
    for(unsigned bits:{2u,31u,64u,65u,512u,1009u,4423u,8192u,16384u})
    for(std::size_t p:{std::size_t(1),std::size_t(2),std::size_t(7),std::size_t(128),std::size_t(8192),std::size_t(155520),std::size_t(1)<<28})
    for(unsigned forced=0;forced<=62;++forced) {
        unsigned lg=1;while((std::size_t(1)<<lg)<p)++lg;const auto slot=2*std::size_t(bits)+lg;
        unsigned b=0,k=0;std::size_t sw=0;Big terms,digit,bound,q;set(q,ntt::modulus);
        for(unsigned j=forced?forced:62;j;--j){sw=(slot+j-1)/j;set(terms,p*sw);mpz_set_ui(digit.v,1);mpz_mul_2exp(digit.v,digit.v,j);mpz_sub_ui(digit.v,digit.v,1);
            mpz_mul(bound.v,digit.v,digit.v);mpz_mul(bound.v,bound.v,terms.v);
            if(mpz_cmp(bound.v,q.v)<0){b=j;break;}if(forced)break;}
        if(b){for(k=1;k<=29 && (std::size_t(1)<<k)<2*p*sw+1;++k){}if(k>29)b=0;}
        if(!b){rejects("shape exactness","no exact integer shape",[&]{s4::integer_shape(bits,p,forced);},counts);}
        else {const auto got=s4::integer_shape(bits,p,forced);if(got.slot_bits!=slot || got.digit_bits!=b || got.slot_words!=sw || got.log2_length!=k)throw std::runtime_error("shape disagrees with GMP/CUDA contract");}
        ++counts.shapes;
    }
}
struct Program {
    cl_program program=nullptr;std::array<cl_kernel,3> kernels{};
    ~Program(){for(auto k:kernels)if(k)clReleaseKernel(k);if(program)clReleaseProgram(program);}
};
void carry_cases(ntt::Runtime &r,const std::filesystem::path &root,Counts &counts) {
    Program p;const auto source=read(root/"goldilocks.clh")+"\n"+read(root.parent_path()/"stage2/polynomial.cl");
    const char *text=source.c_str();const auto bytes=source.size();cl_int code=0;const auto device=Native::device(r);
    p.program=clCreateProgramWithSource(Native::context(r),1,&text,&bytes,&code);check(code,"carry fixture program");
    code=clBuildProgram(p.program,1,&device,"-cl-std=CL1.2 -DECM_S4_NW=1 -DECM_S4_BITS=64 -DECM_S4_MERSENNE=0 -DECM_CARRY_GROUP=64",nullptr,nullptr);
    if(code!=CL_SUCCESS){std::size_t len=0;clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&len);std::string log(len,'\0');clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,len,log.data(),nullptr);throw std::runtime_error("carry fixture compile: "+log);}
    unsigned ki=0;for(const char *name:{"ecm_carry_local","ecm_carry_blocks","ecm_carry_finish"}){p.kernels[ki++]=clCreateKernel(p.program,name,&code);check(code,name);}
    const std::size_t n=4096,batch=4,group=64,ng=n/group;const Word wn=n,wg=ng;
    auto c=r.allocate(n*batch),scratch=r.allocate(n*batch),blocks=r.allocate(3*ng*batch);
    const auto mc=Native::memory(c),ms=Native::memory(scratch),mb=Native::memory(blocks);
    Word seed=0x32443529826bb6abULL;
    for(cl_uint b:{1u,2u,7u,16u,24u,31u,32u}) {
        const Word active=b==7 || b==24 || b==32?n-67:n;
        const Word mask=(Word(1)<<b)-1;std::vector<Word> input(n*batch);
        for(std::size_t j=0;j<batch;++j)for(std::size_t i=0;i<n;++i){Word v=0;
            if(j==0)v=i?mask:mask+1; // Carry travels across all work-group boundaries.
            if(j==1)v=~Word(0);if(j==2)v=random(seed);if(j==3)v=i%71==0?mask+1:(i%67==0?0:mask);
            input[j*n+i]=v;}
        r.upload(c,input.data(),input.size());
        auto k=p.kernels[0];arg(k,0,mc);arg(k,1,ms);arg(k,2,mb);arg(k,3,wn);arg(k,4,wg);arg(k,5,b);
        const std::size_t global[]={n,batch},local[]={group,1};const auto queue=Native::queue(r);
        check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"carry fixture local");
        k=p.kernels[1];arg(k,0,mb);arg(k,1,wg);arg(k,2,mc);arg(k,3,wn);arg(k,4,b);const auto total=batch;check(clEnqueueNDRangeKernel(queue,k,1,nullptr,&total,nullptr,0,nullptr,nullptr),"carry fixture blocks");
        k=p.kernels[2];arg(k,0,ms);arg(k,1,mc);arg(k,2,mb);arg(k,3,wn);arg(k,4,wg);arg(k,5,active);arg(k,6,b);
        check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"carry fixture finish");
        std::vector<Word> out(input.size()),status(3*ng*batch);r.download(c,out.data(),out.size());r.download(blocks,status.data(),status.size());
        Big carry,x,digit;
        for(std::size_t j=0;j<batch;++j){mpz_set_ui(carry.v,0);
            for(std::size_t i=0;i<n;++i){set(x,input[j*n+i]);mpz_add(x.v,x.v,carry.v);mpz_fdiv_r_2exp(digit.v,x.v,b);mpz_fdiv_q_2exp(carry.v,x.v,b);
                if(out[j*n+i]!=word(digit))throw std::runtime_error("carry mismatch bpw="+std::to_string(b)+" batch="+std::to_string(j)+" digit="+std::to_string(i));++counts.carry_words;}
            const bool overflow=mpz_sgn(carry.v)!=0;
            for(std::size_t g=0;g<ng;++g){bool expected=g==ng-1 && overflow;for(std::size_t i=g*group;i<(g+1)*group;++i)if(i>=active && out[j*n+i])expected=true;
                if(status[j*3*ng+2*ng+g]!=Word(expected))throw std::runtime_error("carry overflow/padding verdict mismatch bpw="+std::to_string(b)+" batch="+std::to_string(j)+" block="+std::to_string(g)+" got="+std::to_string(status[j*3*ng+2*ng+g])+" expected="+std::to_string(expected));}
        }
        std::cout<<"opencl_carry_verified: bpw="<<b<<" words="<<input.size()<<" active="<<active<<" long_chain=4096 bad=0\n";
    }
    // Check complete integer coefficients BEFORE S4, with the real pack kernel.
    // This uses dense 64-bit operands; the polynomial tests cover wider inputs.
    cl_int pack_code=0;auto pack_kernel=p.kernels[0]; // release local kernel only after carry fixtures.
    clReleaseKernel(pack_kernel);p.kernels[0]=clCreateKernel(p.program,"ecm_poly_pack",&pack_code);check(pack_code,"integer fixture pack");
    const std::size_t capacity=17,na=17,nb=13,nc=na+nb-1,ibatch=3;
    const auto shape=s4::integer_shape(64,capacity);const Word len=shape.length,sw=shape.slot_words,blocks_count=len/group;
    auto left=r.allocate(na*ibatch),right=r.allocate(nb*ibatch),da=r.allocate(len*ibatch),db=r.allocate(len*ibatch),bs=r.allocate(3*blocks_count*ibatch);
    const auto ml=Native::memory(left),mr=Native::memory(right),md=Native::memory(da),mt=Native::memory(db),mg=Native::memory(bs);
    cl_kernel cone=clCreateKernel(p.program,"ecm_carry_local",&pack_code);check(pack_code,"integer fixture cone");
    // Put the temporary kernel in a separate RAII owner.
    Program extra;extra.kernels[0]=cone;
    for(bool split:{false,true}) {
        ntt::PlanOptions po;po.log2_length=shape.log2_length;po.work_group=64;po.split_multiply=split;ntt::Plan plan(r,po);
        std::vector<Word> ah(na*ibatch),bh(nb*ibatch);
        for(std::size_t j=0;j<ibatch;++j){for(std::size_t i=0;i<na;++i)ah[j*na+i]=j==0?~Word(0):random(seed);for(std::size_t i=0;i<nb;++i)bh[j*nb+i]=j==0?~Word(0):random(seed);}
        r.upload(left,ah.data(),ah.size());r.upload(right,bh.data(),bh.size());const cl_uint bpw=shape.digit_bits;const Word zero=0,wna=na,wnb=nb;
        const std::size_t global[]={std::size_t(len),ibatch},local[]={group,1};const auto queue=Native::queue(r);
        auto pack=[&](cl_mem input,Word count,cl_mem dest){auto k=p.kernels[0];arg(k,0,input);arg(k,1,zero);arg(k,2,count);arg(k,3,count);arg(k,4,dest);arg(k,5,len);arg(k,6,sw);arg(k,7,bpw);check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"integer pack");};
        pack(ml,wna,md);pack(mr,wnb,mt);auto ticket=plan.convolve(da,db,ibatch);
        auto k=cone;arg(k,0,md);arg(k,1,mt);arg(k,2,mg);arg(k,3,len);arg(k,4,blocks_count);arg(k,5,bpw);check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"integer cone");
        k=p.kernels[1];arg(k,0,mg);arg(k,1,blocks_count);arg(k,2,md);arg(k,3,len);arg(k,4,bpw);const auto total=ibatch;check(clEnqueueNDRangeKernel(queue,k,1,nullptr,&total,nullptr,0,nullptr,nullptr),"integer prefix");
        k=p.kernels[2];const Word active_digits=nc*sw;arg(k,0,mt);arg(k,1,md);arg(k,2,mg);arg(k,3,len);arg(k,4,blocks_count);arg(k,5,active_digits);arg(k,6,bpw);check(clEnqueueNDRangeKernel(queue,k,2,nullptr,global,local,0,nullptr,nullptr),"integer finish");
        std::vector<Word> result(len*ibatch),verdict(3*blocks_count*ibatch);r.download(da,result.data(),result.size());r.download(bs,verdict.data(),verdict.size());ticket.wait();
        Big exact,actual,x,y,z;
        for(std::size_t j=0;j<ibatch;++j){for(std::size_t g=0;g<blocks_count;++g)if(verdict[j*3*blocks_count+2*blocks_count+g])throw std::runtime_error("integer fixture carry verdict");
            for(std::size_t k2=0;k2<nc;++k2){mpz_set_ui(exact.v,0);for(std::size_t i=0;i<na;++i)if(k2>=i && k2-i<nb){set(x,ah[j*na+i]);set(y,bh[j*nb+k2-i]);mpz_mul(z.v,x.v,y.v);mpz_add(exact.v,exact.v,z.v);}
                mpz_set_ui(actual.v,0);for(std::size_t d=sw;d>0;--d){mpz_mul_2exp(actual.v,actual.v,bpw);set(x,result[j*len+k2*sw+d-1]);mpz_add(actual.v,actual.v,x.v);}
                if(mpz_cmp(exact.v,actual.v))throw std::runtime_error("integer coefficient mismatch before S4");++counts.integer_coefficients;}
            for(std::size_t i=nc*sw;i<len;++i)if(result[j*len+i])throw std::runtime_error("integer fixture nonzero padding");}
    }
    std::cout<<"opencl_integer_verified: bits=64 dense=1 coefficients="<<2*ibatch*nc<<" before_s4=1 bad=0\n";
}
void reduction_cases(ntt::Runtime &r,const std::filesystem::path &root,unsigned bits,bool mersenne,bool split,Counts &counts) {
    const std::size_t nw=(bits+63)/64,cases=128,stride=2*nw+2;Word seed=0x923661b6ce1a30fdULL+bits;
    std::vector<Word> modulus(nw),normalized(nw),input(cases*stride);
    for(auto &v:modulus)v=mersenne?~Word(0):random(seed);
    if(bits%64)modulus.back()&=(Word(1)<<(bits%64))-1;modulus.back()|=Word(1)<<((bits-1)%64);modulus[0]|=1;
    Big mod,ns,inverse,top,power;load(mod,modulus.data(),nw);unsigned shift=0;for(Word t=modulus.back();!(t>>63);t<<=1)++shift;
    mpz_mul_2exp(ns.v,mod.v,shift);normalized=limbs(ns,nw);set(top,normalized.back());mpz_set_ui(power.v,1);mpz_mul_2exp(power.v,power.v,128);mpz_sub_ui(power.v,power.v,1);
    mpz_fdiv_q(inverse.v,power.v,top.v);mpz_set_ui(power.v,1);mpz_mul_2exp(power.v,power.v,64);mpz_sub(inverse.v,inverse.v,power.v);const Word reciprocal=word(inverse);
    for(std::size_t j=0;j<cases;++j)for(std::size_t i=0;i<stride;++i)input[j*stride+i]=j<4?0:random(seed);
    input[stride]=1;std::copy(modulus.begin(),modulus.end(),input.begin()+2*stride);
    for(std::size_t i=0;i<stride;++i)input[3*stride+i]=~Word(0);
    Program p;const auto source=read(root/"goldilocks.clh")+"\n"+read(root.parent_path()/"stage2/polynomial.cl")+R"CLC(
__kernel void s4_fixture(__global const ulong *input,ulong stride,__global const ulong *ns,
    int shift,ulong reciprocal,__global ulong *out,__global ulong *status) {
    const ulong k=get_global_id(0);ulong t[2*ECM_S4_NW+4],u[ECM_S4_NW];
    for(int i=0;i<2*ECM_S4_NW+4;++i)t[i]=i<stride?input[k*stride+i]:0;
#if ECM_S4_MERSENNE
    ecm_s4_mersenne(t,(int)stride,u);status[k]=0;
#else
    status[k]=(ulong)(ecm_s4_div(t,(int)stride,ns,shift,reciprocal,u)<0);
#endif
    for(int i=0;i<ECM_S4_NW;++i)out[k*ECM_S4_NW+i]=u[i];
}
)CLC";
    const char *text=source.c_str();const auto len=source.size();cl_int code=0;const auto device=Native::device(r);
    p.program=clCreateProgramWithSource(Native::context(r),1,&text,&len,&code);check(code,"S4 fixture program");
    const auto options="-cl-std=CL1.2 -DECM_S4_NW="+std::to_string(nw)+" -DECM_S4_BITS="+std::to_string(bits)+
        " -DECM_S4_MERSENNE="+std::to_string(mersenne?1:0)+" -DECM_CARRY_GROUP=64 -DECM_NTT_SPLIT_MUL="+std::to_string(split?1:0);
    code=clBuildProgram(p.program,1,&device,options.c_str(),nullptr,nullptr);
    if(code!=CL_SUCCESS){std::size_t bytes=0;clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&bytes);std::string log(bytes,'\0');clGetProgramBuildInfo(p.program,device,CL_PROGRAM_BUILD_LOG,bytes,log.data(),nullptr);throw std::runtime_error("S4 fixture compile: "+log);}
    auto k=p.kernels[0]=clCreateKernel(p.program,"s4_fixture",&code);check(code,"S4 fixture kernel");
    auto a=r.allocate(input.size()),b=r.allocate(nw),out=r.allocate(cases*nw),status=r.allocate(cases);
    r.upload(a,input.data(),input.size());r.upload(b,normalized.data(),nw);
    const auto ma=Native::memory(a),mb=Native::memory(b),mo=Native::memory(out),ms=Native::memory(status);const Word ws=stride;const cl_int sh=cl_int(shift);
    arg(k,0,ma);arg(k,1,ws);arg(k,2,mb);arg(k,3,sh);arg(k,4,reciprocal);arg(k,5,mo);arg(k,6,ms);
    const std::size_t group=32;check(clEnqueueNDRangeKernel(Native::queue(r),k,1,nullptr,&cases,&group,0,nullptr,nullptr),"S4 fixture launch");
    std::vector<Word> actual(cases*nw),flags(cases);r.download(out,actual.data(),actual.size());r.download(status,flags.data(),flags.size());
    Big z;for(std::size_t j=0;j<cases;++j){if(flags[j])throw std::runtime_error("S4 fixture correction failed");load(z,input.data()+j*stride,stride);mpz_mod(z.v,z.v,mod.v);const auto expected=limbs(z,nw);
        for(std::size_t i=0;i<nw;++i)if(actual[j*nw+i]!=expected[i])throw std::runtime_error("S4 raw mismatch bits="+std::to_string(bits)+" mode="+std::to_string(mersenne)+" case="+std::to_string(j));counts.reduction_words+=nw;}
    std::cout<<"opencl_s4_raw_verified: bits="<<bits<<" mode="<<(mersenne?"mersenne":"ordinary")<<" split="<<split<<" cases="<<cases<<" bad=0\n";
}
void polynomial_case(ntt::Runtime &r,unsigned bits,bool mersenne,bool split,Counts &counts) {
    const std::size_t nw=(bits+63)/64,capacity=bits<=129?7:(bits<8192?17:3),batch=3;
    Word seed=0x1293964726d54627ULL+bits;std::vector<Word> modulus(nw);
    for(auto &v:modulus)v=mersenne?~Word(0):random(seed);
    if(bits%64)modulus.back()&=(Word(1)<<(bits%64))-1;
    modulus.back()|=Word(1)<<((bits-1)%64);modulus[0]|=1;
    if(!mersenne && bits==2)modulus[0]=2;
    s4::PolynomialOptions options;options.modulus=modulus;options.capacity=capacity;options.mersenne=mersenne;
    options.transform.split_multiply=split;options.transform.work_group=split?64:128;options.transform.tile_bits=split?5:8;
    s4::PolynomialPlan plan(r,options);++counts.plans;const auto &shape=plan.shape();
    if(plan.build_log().find("warning:")!=std::string::npos)std::cout<<"opencl_s4_build_log: "<<plan.build_log()<<'\n';
    const std::size_t na=capacity,nb=capacity-1,nc=na+nb-1,off=3,ast=na*nw+5,bst=nb*nw+7,ost=nc*nw+9;
    constexpr Word guard=0xca703f1b556edeadULL;
    auto a=r.allocate(off+batch*ast+2),b=r.allocate(off+batch*bst+2),out=r.allocate(off+batch*ost+2);
    std::vector<Word> ah(a.words(),guard),bh(b.words(),guard),oh(out.words(),guard);Big mod,x,y,z,sum;
    load(mod,modulus.data(),nw);
    for(unsigned pattern=0;pattern<4;++pattern) {
        auto fill=[&](std::vector<Word> &host,std::size_t count,std::size_t stride){
            for(std::size_t j=0;j<batch;++j)for(std::size_t i=0;i<count;++i){auto at=host.data()+off+j*stride+i*nw;
                for(std::size_t w=0;w<nw;++w)at[w]=pattern==0?0:(pattern==1?~Word(0):random(seed));
                if(bits%64)at[nw-1]&=(Word(1)<<(bits%64))-1;
                if(pattern==3){if(i%3==0){for(std::size_t w=0;w<nw;++w)at[w]=modulus[w];}
                    else if(i%3==1){for(std::size_t w=0;w<nw;++w)at[w]=0;at[0]=1;}}
            }};
        fill(ah,na,ast);fill(bh,nb,bst);r.upload(a,ah.data(),ah.size());r.upload(b,bh.data(),bh.size());r.upload(out,oh.data(),oh.size());
        const auto info=plan.multiply({a,off,ast},na,{b,off,bst},nb,{out,off,ost},batch);
        if(info.coefficients!=nc || info.batch!=batch)throw std::runtime_error("product metadata mismatch");
        std::vector<Word> expected=oh,actual(oh.size());r.download(out,actual.data(),actual.size());
        for(std::size_t j=0;j<batch;++j)for(std::size_t k=0;k<nc;++k){mpz_set_ui(sum.v,0);
            for(std::size_t i=0;i<na;++i)if(k>=i && k-i<nb){load(x,ah.data()+off+j*ast+i*nw,nw);load(y,bh.data()+off+j*bst+(k-i)*nw,nw);mpz_mul(z.v,x.v,y.v);mpz_add(sum.v,sum.v,z.v);}
            mpz_mod(sum.v,sum.v,mod.v);const auto e=limbs(sum,nw);std::copy(e.begin(),e.end(),expected.begin()+off+j*ost+k*nw);counts.polynomial_words+=nw;}
        for(std::size_t i=0;i<actual.size();++i)if(actual[i]!=expected[i])throw std::runtime_error("modular product mismatch bits="+std::to_string(bits)+" mode="+(mersenne?"mersenne":"ordinary")+" pattern="+std::to_string(pattern)+" word="+std::to_string(i));
        ++counts.products;
    }
    // View and input failures must be observable; a failed request must not poison reuse.
    rejects("count","coefficient count",[&]{plan.multiply({a},capacity+1,{b},nb,{out});},counts);
    rejects("alias","distinct output buffer",[&]{plan.multiply({a},1,{b},1,{a});},counts);
    rejects("batch","batch/stride",[&]{plan.multiply({a},na,{b},nb,{out},0);},counts);
    rejects("stride","batch/stride",[&]{plan.multiply({a,off,1},na,{b},nb,{out});},counts);
    rejects("out view","out of bounds",[&]{plan.multiply({a},na,{b},nb,{out,out.words(),0});},counts);
    rejects("span overflow","overflow",[&]{plan.multiply({a,0,std::numeric_limits<std::size_t>::max()},na,{b},nb,{out},3);},counts);
    if(bits%64){const auto index=off+nw-1;const auto saved=ah[index];ah[index]|=Word(1)<<(bits%64);r.upload(a,ah.data(),ah.size());
        rejects("input top bits","verdict failed",[&]{plan.multiply({a,off,ast},na,{b,off,bst},nb,{out,off,ost},batch);},counts);
        ah[index]=saved;r.upload(a,ah.data(),ah.size());plan.multiply({a,off,ast},na,{b,off,bst},nb,{out,off,ost},batch);++counts.guards;}
    std::cout<<"opencl_s4_verified: bits="<<bits<<" mode="<<(mersenne?"mersenne":"ordinary")<<" split="<<split<<" capacity="<<capacity<<" bpw="<<shape.digit_bits<<" length="<<shape.length<<" batch=3 patterns=4 bad=0\n";
}
void large_product(ntt::Runtime &r,unsigned bits,std::size_t capacity,bool mersenne,Counts &counts) {
    const std::size_t nw=(bits+63)/64,nc=2*capacity-1;Word seed=0x841376921e82a29dULL+bits;
    std::vector<Word> modulus(nw,~Word(0));if(bits%64)modulus.back()=(Word(1)<<(bits%64))-1;
    if(!mersenne)modulus[0]-=32;
    s4::PolynomialOptions po;po.modulus=modulus;po.capacity=capacity;po.mersenne=mersenne;po.transform.split_multiply=mersenne;
    s4::PolynomialPlan plan(r,po);++counts.plans;Big mod;load(mod,modulus.data(),nw);
    const std::array<std::size_t,4> ai={0,capacity/3,capacity/2,capacity-1},bi={0,capacity/4,capacity/2+1,capacity-1};
    std::vector<Word> ah(capacity*nw),bh(capacity*nw),actual(nc*nw),expected(nc*nw);
    for(auto i:ai)for(std::size_t w=0;w<nw;++w)ah[i*nw+w]=random(seed);
    for(auto i:bi)for(std::size_t w=0;w<nw;++w)bh[i*nw+w]=random(seed);
    for(auto *v:{&ah,&bh})for(std::size_t i=0;i<capacity;++i)if(bits%64)(*v)[i*nw+nw-1]&=(Word(1)<<(bits%64))-1;
    Big x,y,z,sum;for(auto i:ai)for(auto j:bi){load(x,ah.data()+i*nw,nw);load(y,bh.data()+j*nw,nw);mpz_mul(z.v,x.v,y.v);load(sum,expected.data()+(i+j)*nw,nw);mpz_add(z.v,z.v,sum.v);mpz_mod(z.v,z.v,mod.v);const auto v=limbs(z,nw);std::copy(v.begin(),v.end(),expected.begin()+(i+j)*nw);}
    auto a=r.allocate(ah.size()),b=r.allocate(bh.size()),out=r.allocate(actual.size());r.upload(a,ah.data(),ah.size());r.upload(b,bh.data(),bh.size());
    plan.multiply({a},capacity,{b},capacity,{out});r.download(out,actual.data(),actual.size());
    for(std::size_t i=0;i<actual.size();++i)if(actual[i]!=expected[i])throw std::runtime_error("large sparse product mismatch bits="+std::to_string(bits)+" word="+std::to_string(i));
    counts.polynomial_words+=actual.size();++counts.products;
    std::cout<<"opencl_s4_large_verified: bits="<<bits<<" mode="<<(mersenne?"mersenne":"ordinary")<<" capacity="<<capacity<<" length="<<plan.shape().length<<" checked_words="<<actual.size()<<" bad=0\n";
}
Word number(const std::string &v){if(v.empty() || v.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("unsigned integer required");return std::stoull(v);}
} // namespace
int main(int argc,char **argv) {
    std::cout<<std::unitbuf;std::cerr<<std::unitbuf;
    try {
        bool verify=false;std::size_t device=0;std::string output,root=ECM_OPENCL_NTT_RESOURCE_ROOT;unsigned max_bits=16384;
        for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--verify"){verify=true;continue;}if(key=="--list")continue;
            if(key=="--help"){std::cout<<"opencl_s4_probe --verify --device N [--max-bits 2..16384] [--output FILE] [--resource-root DIR]\n";return 0;}
            if(i+1>=argc)throw std::invalid_argument("missing value");const std::string v=argv[++i];if(key=="--device")device=std::size_t(number(v));else if(key=="--output")output=v;
            else if(key=="--resource-root")root=v;else if(key=="--max-bits"){const auto n=number(v);if(n<2 || n>16384)throw std::invalid_argument("max-bits outside 2..16384");max_bits=unsigned(n);}else throw std::invalid_argument("unknown option "+key);}
        const auto list=ntt::devices();for(const auto &d:list)std::cout<<"opencl_device: index="<<d.index<<" name="<<quote(d.name)<<" supported="<<d.supported<<'\n';
        if(!verify)return 0;if(!output.empty() && std::filesystem::exists(output))throw std::invalid_argument("output already exists; use a fresh evidence path");
        Counts counts;const auto start=std::chrono::steady_clock::now();shapes(counts);
        ntt::Options ro;ro.device=device;ro.budget_bytes=256ULL<<20;ro.resource_root=root;ntt::Runtime runtime(ro);carry_cases(runtime,root,counts);
        unsigned index=0;for(unsigned bits:{2u,31u,63u,64u,65u,127u,128u,129u,503u,512u,1009u,2203u,4423u,8191u,8192u,16384u}){
            if(bits>max_bits)continue;for(bool mersenne:{false,true})polynomial_case(runtime,bits,mersenne,(index%2)!=0,counts);++index;}
        for(unsigned bits:{64u,65u,503u,4423u,8192u,16384u})if(bits<=max_bits)for(bool mersenne:{false,true})for(bool split:{false,true})reduction_cases(runtime,root,bits,mersenne,split,counts);
        if(max_bits>=512)for(bool mersenne:{false,true})large_product(runtime,512,8192,mersenne,counts);
        if(max_bits>=4423)for(bool mersenne:{false,true})large_product(runtime,4423,1024,mersenne,counts);
        // Invalid modulus, exactness and ownership checks, before any product launch.
        rejects("zero capacity","shape outside",[&]{s4::integer_shape(512,0);},counts);
        rejects("oversize bits","shape outside",[&]{s4::integer_shape(16385,1);},counts);
        rejects("bad bpw","outside 1..62",[&]{s4::integer_shape(512,1,63);},counts);
        s4::PolynomialOptions po;rejects("empty modulus","minimal",[&]{s4::PolynomialPlan p(runtime,po);},counts);
        po.modulus={5,0};rejects("nonminimal modulus","minimal",[&]{s4::PolynomialPlan p(runtime,po);},counts);
        po.modulus={5};po.mersenne=true;rejects("false Mersenne","requires N",[&]{s4::PolynomialPlan p(runtime,po);},counts);
        {po.mersenne=false;po.capacity=1;s4::PolynomialPlan p(runtime,po);auto a=runtime.allocate(1),b=runtime.allocate(1),c=runtime.allocate(1);Word av=4,bv=3;
            runtime.upload(a,&av,1);runtime.upload(b,&bv,1);p.multiply({a},1,{b},1,{c});Word actual=0;runtime.download(c,&actual,1);if(actual!=2)throw std::runtime_error("capacity-one product failed");++counts.products;
            auto other=ro;ntt::Runtime foreign(other);auto f=foreign.allocate(1);rejects("foreign owner","another runtime",[&]{p.multiply({f},1,{b},1,{c});},counts);}
        {po.forced_digit_bits=1;s4::PolynomialPlan p(runtime,po);auto a=runtime.allocate(1),b=runtime.allocate(1),c=runtime.allocate(1);Word av=7,bv=7;
            runtime.upload(a,&av,1);runtime.upload(b,&bv,1);p.multiply({a},1,{b},1,{c});Word actual=0;runtime.download(c,&actual,1);if(actual!=4)throw std::runtime_error("forced bpw-one product failed");++counts.products;}
        {auto limited_options=ro;limited_options.budget_bytes=96*1024;ntt::Runtime limited(limited_options);
            auto workspace=std::make_shared<s4::PolynomialWorkspace>();s4::PolynomialOptions large,small;
            large.modulus={~Word(0),(Word(1)<<63)-1};large.capacity=33;large.workspace=workspace;small=large;small.capacity=1;
            s4::PolynomialPlan big(limited,large),little(limited,small);
            auto a=limited.allocate(66),b=limited.allocate(66),out=limited.allocate(130);
            std::vector<Word> input(66);for(std::size_t i=0;i<33;++i)input[2*i]=1;
            limited.upload(a,input.data(),input.size());limited.upload(b,input.data(),input.size());big.multiply({a},33,{b},33,{out});
            auto x=limited.allocate(64),y=limited.allocate(64),z=limited.allocate(64);std::vector<Word> xv(64),yv(64);
            for(std::size_t i=0;i<32;++i){xv[2*i]=2;yv[2*i]=3;}limited.upload(x,xv.data(),64);limited.upload(y,yv.data(),64);
            // The new shape needs a larger summary but smaller verdict. With
            // only 400 bytes free, retaining mismatched old capacities fails.
            const auto room=limited.memory().budget_bytes-limited.memory().live_bytes;
            if(room<=400)throw std::runtime_error("workspace budget fixture size");auto pressure=limited.allocate(std::size_t((room-400)/8));
            little.multiply({x},1,{y},1,{z},32);std::vector<Word> actual(64);limited.download(z,actual.data(),64);
            for(std::size_t i=0;i<64;++i)if(actual[i]!=(i%2?0:6))throw std::runtime_error("workspace budget reclaim mismatch");++counts.guards;
        }
        if(runtime.memory().live_bytes)throw std::runtime_error("native polynomial payload leak");++counts.guards;
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::ostringstream report;
        report<<"{\"status\":\"passed\",\"program_version\":"<<quote(ecm_program::version)<<",\"backend\":\"native-opencl-pack-ntt-carry-s4\",\"device\":"<<quote(runtime.device().name)
            <<",\"driver\":"<<quote(runtime.device().driver)<<",\"max_bits\":"<<max_bits<<",\"shape_cases\":"<<counts.shapes<<",\"carry_words\":"<<counts.carry_words
            <<",\"reduction_words\":"<<counts.reduction_words<<",\"integer_coefficients\":"<<counts.integer_coefficients<<",\"polynomial_words\":"<<counts.polynomial_words<<",\"products\":"<<counts.products<<",\"plans\":"<<counts.plans<<",\"guard_cases\":"<<counts.guards
            <<",\"payload_peak_bytes\":"<<runtime.memory().peak_bytes<<",\"verification_wall_seconds\":"<<elapsed<<",\"full_stage2\":false,\"bad\":0}";
        if(!output.empty()){std::ofstream f(output,std::ios::binary);if(!f)throw std::runtime_error("cannot write result");f<<report.str()<<'\n';f.close();if(!f)throw std::runtime_error("result write failed");}
        std::cout<<"opencl_s4_summary: "<<report.str()<<'\n';return 0;
    }catch(const std::exception &e){std::cerr<<"opencl_s4_failed: "<<e.what()<<'\n';return 2;}
}
