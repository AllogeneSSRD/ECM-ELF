#include "points.h"
#include "ecm/opencl/ntt/native.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ecm_opencl::stage2 {
namespace {
using Native=ntt::NativeAccess;
void check(cl_int c,const char*s){if(c!=CL_SUCCESS)throw std::runtime_error(std::string(s)+": OpenCL "+std::to_string(c));}
// Argument zero is the runtime limb count, set once at program construction.
template<class T>void arg(cl_kernel k,unsigned i,const T&v){check(clSetKernelArg(k,i+1,sizeof(v),&v),"point argument");}
std::size_t times(std::size_t a,std::size_t b){if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("point span overflow");return a*b;}
bool ge(const Word*a,const std::vector<Word>&n){for(std::size_t i=n.size();i-->0;)if(a[i]!=n[i])return a[i]>n[i];return true;}
void double_mod(std::vector<Word>&a,const std::vector<Word>&n){Word c=0;for(auto&w:a){const auto next=w>>63;w=(w<<1)|c;c=next;}
    if(c||ge(a.data(),n)){Word borrow=0;for(std::size_t i=0;i<a.size();++i){const Word x=a[i]-n[i],b1=a[i]<n[i],y=x-borrow,b2=x<borrow;a[i]=y;borrow=b1|b2;}}}
}
struct PointState {
    ntt::Runtime runtime;std::vector<Word> n;std::size_t nw,ladder_group=0;Word ninv=0;
    ntt::Buffer constants;cl_program program=nullptr;std::array<cl_kernel,5> kernels{};PointTiming timing;
    PointState(ntt::Runtime&r,const std::vector<Word>&v):runtime(r),n(v),nw(v.size()){}
    ~PointState(){try{runtime.finish();}catch(...){}for(auto k:kernels)if(k)clReleaseKernel(k);if(program)clReleaseProgram(program);}
    Polynomial allocate(std::size_t count){if(!count)throw std::invalid_argument("empty point request");return {runtime.allocate(times(count,nw)),1,count,0,nw};}
    std::size_t validate(const Polynomial&p)const {
        // Flat vectors and strided batches of scalar coefficients are accepted.
        if(!p.size||!p.batch || (p.size>1&&p.batch>1))throw std::invalid_argument("point scalar vector required");
        const auto count=p.size>1?p.size:p.batch,stride=p.size>1?nw:p.stride;
        if(stride<nw)throw std::invalid_argument("point stride too short");
        const auto tail=times(count-1,stride);if(tail>std::numeric_limits<std::size_t>::max()-nw)throw std::overflow_error("point span overflow");
        Native::span(runtime,p.buffer,tail+nw,p.offset);return count;
    }
    Word stride(const Polynomial&p)const{return p.size>1?nw:p.stride;}
    void canonical(const std::vector<Word>&v)const {if(v.empty()||v.size()%nw)throw std::invalid_argument("point limb shape");
        for(std::size_t i=0;i<v.size();i+=nw)if(ge(v.data()+i,n))throw std::invalid_argument("noncanonical point input");}
    void launch(cl_kernel k,std::size_t count,std::size_t group=0){cl_event event=nullptr;
        const auto global=group?((count+group-1)/group)*group:count;
        check(clEnqueueNDRangeKernel(Native::queue(runtime),k,1,nullptr,&global,group?&group:nullptr,0,nullptr,&event),"point launch");
        // Always drain before local buffer owners leave scope, including errors.
        const auto code=clWaitForEvents(1,&event);cl_int status=0;const auto query=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,nullptr);
        cl_ulong start=0,end=0;const auto ps=clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,nullptr);
        const auto pe=clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr);clReleaseEvent(event);
        if(code!=CL_SUCCESS||query!=CL_SUCCESS||status<0){try{runtime.finish();}catch(...){}check(code,"point wait");check(query,"point status");check(status,"point execution");}
        if(ps==CL_SUCCESS&&pe==CL_SUCCESS)timing.kernel_seconds+=double(end-start)*1e-9;
    }
    void verdict(const ntt::Buffer&bad,std::size_t count,bool recovery=false){std::vector<Word>v(count);runtime.download(bad,v.data(),count);timing.download_bytes+=8*count;
        if(std::any_of(v.begin(),v.end(),[&](Word w){return w!=0&&!(recovery&&w==2);}))throw std::runtime_error("device binary GCD iteration bound");}
};
Points::Points(ntt::Runtime&r,const std::vector<Word>&n,std::size_t ladder_group):state_(std::make_unique<PointState>(r,n)){
    auto&p=*state_;if(n.empty()||n.size()>256||!n.back()||!(n[0]&1)||(n.size()==1&&n[0]<3))throw std::invalid_argument("point modulus must be odd, minimal, 3..16384 bits");
    const auto&vendor=r.device().vendor;const bool amd=vendor.find("Advanced Micro Devices")!=std::string::npos||vendor=="AMD";
    // Qualification is narrower than integer capacity. Wide AMD inverse/GCD
    // results are not stable across runs, including local=1 Mersenne fixtures;
    // -cl-opt-disable does not repair generic inputs. Reject before producing
    // coordinates/factors. This does not restrict the separately tested NTT/S4.
    if(amd&&p.nw>8)throw std::invalid_argument("AMD point arithmetic above 512 bits is not qualified; use CPU/CUDA for this target");
    Word inv=1;for(int i=0;i<6;++i)inv*=2-n[0]*inv;p.ninv=0-inv;
    std::vector<Word> one(p.nw);one[0]=1;for(std::size_t i=0;i<64*p.nw;++i)double_mod(one,n);auto r2=one;
    for(std::size_t i=0;i<64*p.nw;++i)double_mod(r2,n);
    std::vector<Word>c=n;c.insert(c.end(),one.begin(),one.end());c.insert(c.end(),r2.begin(),r2.end());c.resize(6*p.nw);
    p.constants=r.allocate(c.size());r.upload(p.constants,c.data(),c.size());
    const auto path=std::filesystem::path(Native::resource_root(r)).parent_path()/"stage2/points.cl";
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+path.string());const std::string source{std::istreambuf_iterator<char>(f),{}};
    cl_int code=0;const auto device=Native::device(r);const auto opts="-cl-std=CL1.2 -DECM_POINT_NW="+std::to_string(p.nw);p.program=Native::program(r,source,opts);
    unsigned i=0;const cl_int nw=static_cast<cl_int>(p.nw);
    for(const char*name:{"point_ladder","point_normalize","point_products","point_gcd","point_recover"}){auto k=clCreateKernel(p.program,name,&code);p.kernels[i++]=k;check(code,name);check(clSetKernelArg(k,0,sizeof(nw),&nw),"point runtime width");}
    std::size_t limit=0;check(clGetKernelWorkGroupInfo(p.kernels[0],device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(limit),&limit,nullptr),"point ladder group limit");
    if(ladder_group>limit||ladder_group>r.device().max_work_group)throw std::invalid_argument("point ladder group exceeds device/kernel limit");
    p.ladder_group=ladder_group;
}
Points::~Points()=default;
std::size_t Points::words()const{return state_->nw;}
const PointTiming&Points::timing()const{return state_->timing;}
Polynomial Points::upload(const std::vector<Word>&v){auto&p=*state_;p.canonical(v);auto out=p.allocate(v.size()/p.nw);p.runtime.upload(out.buffer,v.data(),v.size());return out;}
std::vector<Word>Points::download(const Polynomial&a){auto&p=*state_;const auto count=p.validate(a);std::vector<Word>v(times(count,p.nw));
    if(p.stride(a)==p.nw)p.runtime.download(a.buffer,v.data(),v.size(),a.offset);
    else for(std::size_t i=0;i<count;++i)p.runtime.download(a.buffer,v.data()+i*p.nw,p.nw,a.offset+i*p.stride(a));return v;}
ProjectivePoints Points::ladder(const std::vector<Word>&x,const std::vector<Word>&z,const std::vector<Word>&a24,const std::vector<Word>&scalars){
    auto&p=*state_;p.timing={};for(auto*v:{&x,&z,&a24}){if(v->size()!=p.nw)throw std::invalid_argument("point base limb shape");p.canonical(*v);}
    if(scalars.empty())throw std::invalid_argument("empty point scalars");const Word count=scalars.size();
    auto ks=p.runtime.allocate(scalars.size());p.runtime.upload(ks,scalars.data(),scalars.size());std::vector<Word>base=x;base.insert(base.end(),z.begin(),z.end());base.insert(base.end(),a24.begin(),a24.end());
    p.runtime.upload(p.constants,base.data(),base.size(),3*p.nw);p.timing.upload_bytes=8*(base.size()+scalars.size());ProjectivePoints out{p.allocate(scalars.size()),p.allocate(scalars.size())};
    auto k=p.kernels[0];const auto c=Native::memory(p.constants),s=Native::memory(ks),ox=Native::memory(out.x.buffer),oz=Native::memory(out.z.buffer);
    arg(k,0,c);arg(k,1,p.ninv);arg(k,2,s);arg(k,3,count);arg(k,4,ox);arg(k,5,oz);p.launch(k,scalars.size(),p.ladder_group);return out;
}
AffinePoints Points::normalize(const ProjectivePoints&a,std::size_t segment){auto&p=*state_;p.timing={};const Word count=p.validate(a.x);
    if(!segment||p.validate(a.z)!=count||a.x.offset!=a.z.offset||p.stride(a.x)!=p.stride(a.z))throw std::invalid_argument("normalization point layout/segment");
    segment=std::min(segment,std::size_t(count));const auto groups=1+(std::size_t(count)-1)/segment;auto prefix=p.runtime.allocate(times(count,p.nw)),bad=p.runtime.allocate(groups+count);AffinePoints out{p.allocate(count),p.allocate(count)};
    auto k=p.kernels[1];const auto c=Native::memory(p.constants),x=Native::memory(a.x.buffer),z=Native::memory(a.z.buffer),pr=Native::memory(prefix),o=Native::memory(out.x.buffer),g=Native::memory(out.gcd.buffer),b=Native::memory(bad);
    const Word offset=a.x.offset,stride=p.stride(a.x),seg=segment;arg(k,0,c);arg(k,1,p.ninv);arg(k,2,x);arg(k,3,z);arg(k,4,offset);arg(k,5,stride);arg(k,6,count);arg(k,7,seg);arg(k,8,pr);arg(k,9,o);arg(k,10,g);arg(k,11,b);p.launch(k,groups);
    k=p.kernels[4];arg(k,0,c);arg(k,1,p.ninv);arg(k,2,x);arg(k,3,z);arg(k,4,offset);arg(k,5,stride);arg(k,6,count);arg(k,7,seg);arg(k,8,o);arg(k,9,g);arg(k,10,b);p.launch(k,count);p.verdict(bad,groups+count,true);return out;
}
Polynomial Points::products(const Polynomial&a,std::size_t block){auto&p=*state_;p.timing={};const Word count=p.validate(a);if(!block)throw std::invalid_argument("zero product block");block=std::min(block,std::size_t(count));
    auto out=p.allocate(1+(count-1)/block);auto k=p.kernels[2];const auto c=Native::memory(p.constants),v=Native::memory(a.buffer),o=Native::memory(out.buffer);const Word off=a.offset,stride=p.stride(a),b=block;
    arg(k,0,c);arg(k,1,p.ninv);arg(k,2,v);arg(k,3,off);arg(k,4,stride);arg(k,5,count);arg(k,6,b);arg(k,7,o);p.launch(k,out.batch);return out;
}
Polynomial Points::gcd(const Polynomial&a){auto&p=*state_;p.timing={};const Word count=p.validate(a);auto out=p.allocate(count);auto bad=p.runtime.allocate(count);auto k=p.kernels[3];
    const auto c=Native::memory(p.constants),v=Native::memory(a.buffer),o=Native::memory(out.buffer),b=Native::memory(bad);const Word off=a.offset,stride=p.stride(a);
    arg(k,0,c);arg(k,1,v);arg(k,2,off);arg(k,3,stride);arg(k,4,count);arg(k,5,o);arg(k,6,b);p.launch(k,count);p.verdict(bad,count);return out;
}
} // namespace ecm_opencl::stage2
