#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#include "runtime.h"
#include "native.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ecm_opencl::ntt {
namespace {
static_assert(sizeof(Word)==sizeof(cl_ulong));
void checked(cl_int code,const char *operation) {
    if(code!=CL_SUCCESS)throw std::runtime_error(std::string(operation)+": OpenCL error "+std::to_string(code));
}
template<class T>T info(cl_device_id d,cl_device_info key) {
    T value{};checked(clGetDeviceInfo(d,key,sizeof(value),&value,nullptr),"clGetDeviceInfo");return value;
}
std::string device_text(cl_device_id d,cl_device_info key) {
    std::size_t bytes=0;checked(clGetDeviceInfo(d,key,0,nullptr,&bytes),"clGetDeviceInfo size");
    std::string out(bytes,'\0');if(bytes)checked(clGetDeviceInfo(d,key,bytes,out.data(),nullptr),"clGetDeviceInfo text");
    while(!out.empty() && out.back()=='\0')out.pop_back();return out;
}
std::string platform_text(cl_platform_id p) {
    std::size_t bytes=0;checked(clGetPlatformInfo(p,CL_PLATFORM_NAME,0,nullptr,&bytes),"clGetPlatformInfo size");
    std::string out(bytes,'\0');if(bytes)checked(clGetPlatformInfo(p,CL_PLATFORM_NAME,bytes,out.data(),nullptr),"clGetPlatformInfo");
    while(!out.empty() && out.back()=='\0')out.pop_back();return out;
}
struct Device {cl_platform_id platform;cl_device_id device;DeviceInfo description;};
std::vector<Device> enumerate_devices() {
    cl_uint count=0;const auto status=clGetPlatformIDs(0,nullptr,&count);
    if(status==-1001)return {}; // CL_PLATFORM_NOT_FOUND_KHR; no ICD installed.
    checked(status,"clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(count);
    if(count)checked(clGetPlatformIDs(count,platforms.data(),nullptr),"clGetPlatformIDs list");
    std::vector<Device> result;
    for(auto platform:platforms) {
        cl_uint n=0;const auto code=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,nullptr,&n);
        if(code==CL_DEVICE_NOT_FOUND)continue;checked(code,"clGetDeviceIDs count");
        std::vector<cl_device_id> list(n);
        if(n)checked(clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,n,list.data(),nullptr),"clGetDeviceIDs list");
        for(auto d:list) {
            DeviceInfo v;v.index=result.size();v.platform=platform_text(platform);
            v.name=device_text(d,CL_DEVICE_NAME);v.vendor=device_text(d,CL_DEVICE_VENDOR);
            v.version=device_text(d,CL_DEVICE_VERSION);v.driver=device_text(d,CL_DRIVER_VERSION);
            v.profile=device_text(d,CL_DEVICE_PROFILE);
            v.global_bytes=info<cl_ulong>(d,CL_DEVICE_GLOBAL_MEM_SIZE);
            v.max_allocation_bytes=info<cl_ulong>(d,CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            v.local_bytes=info<cl_ulong>(d,CL_DEVICE_LOCAL_MEM_SIZE);
            v.max_work_group=info<std::size_t>(d,CL_DEVICE_MAX_WORK_GROUP_SIZE);
            v.address_bits=info<cl_uint>(d,CL_DEVICE_ADDRESS_BITS);
            v.little_endian=info<cl_bool>(d,CL_DEVICE_ENDIAN_LITTLE)!=0;
            v.available=info<cl_bool>(d,CL_DEVICE_AVAILABLE)!=0;
            v.compiler=info<cl_bool>(d,CL_DEVICE_COMPILER_AVAILABLE)!=0;
            unsigned major=0,minor=0;std::string tag;char dot=0;
            std::istringstream version(v.version);version>>tag>>major>>dot>>minor;
            if(tag!="OpenCL" || dot!='.')major=minor=0;
            if(!v.available)v.reason="device unavailable";
            else if(!v.compiler)v.reason="OpenCL compiler unavailable";
            else if(major<1 || (major==1 && minor<2))v.reason="OpenCL 1.2 required";
            else if(v.profile!="FULL_PROFILE")v.reason="first native backend requires FULL_PROFILE integer support";
            else if(v.address_bits!=64)v.reason="64-bit device addresses required";
            else if(!v.little_endian)v.reason="first native backend requires little-endian buffers";
            else if(info<cl_uint>(d,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)<2)v.reason="two NDRange dimensions required";
            v.supported=v.reason.empty();result.push_back({platform,d,std::move(v)});
        }
    }
    return result;
}
const std::vector<Device>& inventory() {
    // Stable indices within one process. Curve workers are isolated processes;
    // driver/device errors still propagate through context/command creation.
    static const auto snapshot=enumerate_devices();return snapshot;
}
std::string read_source(const std::filesystem::path &p) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read OpenCL resource: "+p.string());
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
std::string identity(const std::string &s) {
    Word h=14695981039346656037ULL;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}
    std::ostringstream out;out<<std::hex<<h;return out.str();
}
Word reduce(Word lo,Word hi) {
    constexpr Word e=0xffffffffULL;
    const Word minus=lo-(hi>>32),a=minus-(minus>lo?e:0);
    const Word b=((hi&e)<<32)-(hi&e),sum=a+b,value=sum+(sum<a?e:0);
    return value>=modulus?value-modulus:value;
}
Word multiply(Word a,Word b) {
    const Word a0=std::uint32_t(a),a1=a>>32,b0=std::uint32_t(b),b1=b>>32;
    const Word p00=a0*b0,p01=a0*b1,p10=a1*b0;
    const Word mid=(p00>>32)+std::uint32_t(p01)+Word(std::uint32_t(p10));
    return reduce(a*b,a1*b1+(p01>>32)+(p10>>32)+(mid>>32));
}
Word power(Word a,Word e) {Word r=1;for(;e;e>>=1,a=multiply(a,a))if(e&1)r=multiply(r,a);return r;}
std::size_t bytes_for(std::size_t words) {
    if(words>std::numeric_limits<std::size_t>::max()/sizeof(Word))throw std::overflow_error("OpenCL buffer size overflow");
    return words*sizeof(Word);
}
template<class T>void arg(cl_kernel k,cl_uint i,const T &v) {checked(clSetKernelArg(k,i,sizeof(v),&v),"clSetKernelArg");}
} // namespace

struct RuntimeState {
    DeviceInfo device;
    cl_device_id id=nullptr;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    MemoryInfo memory;
    bool profiling=false;
    bool cache_programs=true;
    ProgramInfo programs;
    std::map<std::pair<std::string,std::string>,cl_program> program_cache;
    std::string source,resource_root;
    ~RuntimeState() {
        if(queue){(void)clFinish(queue);clReleaseCommandQueue(queue);}
        for(const auto& entry:program_cache)clReleaseProgram(entry.second);
        if(context)clReleaseContext(context);
    }
};
struct BufferState {
    std::shared_ptr<RuntimeState> runtime;
    cl_mem memory=nullptr;
    std::size_t count=0,bytes=0;
    ~BufferState() {if(memory){clReleaseMemObject(memory);runtime->memory.live_bytes-=bytes;}}
};
namespace {
std::shared_ptr<BufferState> allocate(const std::shared_ptr<RuntimeState> &r,std::size_t words) {
    const auto bytes=bytes_for(words);
    if(!bytes)throw std::invalid_argument("empty OpenCL allocation");
    if(bytes>r->device.max_allocation_bytes)throw std::runtime_error("OpenCL allocation exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE");
    if(bytes>r->memory.budget_bytes || r->memory.live_bytes>r->memory.budget_bytes-bytes)
        throw std::runtime_error("OpenCL owned payload budget exceeded");
    auto b=std::make_shared<BufferState>();b->runtime=r;b->count=words;b->bytes=bytes;
    cl_int code=0;b->memory=clCreateBuffer(r->context,CL_MEM_READ_WRITE,bytes,nullptr,&code);
    if(code!=CL_SUCCESS){b->memory=nullptr;checked(code,"clCreateBuffer");}
    r->memory.live_bytes+=bytes;r->memory.peak_bytes=std::max(r->memory.peak_bytes,r->memory.live_bytes);return b;
}
void span(const std::shared_ptr<BufferState> &b,const std::shared_ptr<RuntimeState> &r,
          std::size_t count,std::size_t offset) {
    if(!b || b->runtime!=r)throw std::invalid_argument("OpenCL buffer belongs to another runtime");
    if(offset>b->count || count>b->count-offset)throw std::out_of_range("OpenCL buffer view out of bounds");
}
void transfer(const std::shared_ptr<BufferState> &b,Word *data,std::size_t count,std::size_t offset,bool write) {
    if(count && !data)throw std::invalid_argument("null host transfer buffer");
    if(!count)return;
    const auto r=b->runtime;
    checked(write?clEnqueueWriteBuffer(r->queue,b->memory,CL_TRUE,bytes_for(offset),bytes_for(count),data,0,nullptr,nullptr):
                  clEnqueueReadBuffer(r->queue,b->memory,CL_TRUE,bytes_for(offset),bytes_for(count),data,0,nullptr,nullptr),
            write?"clEnqueueWriteBuffer":"clEnqueueReadBuffer");
}
} // namespace

struct PlanState {
    std::shared_ptr<RuntimeState> runtime;
    std::shared_ptr<BufferState> low,high;
    cl_program program=nullptr;
    cl_kernel global=nullptr,tile=nullptr;
    Word n=0,scale=0,low_count=0,high_count=0;
    unsigned low_bits=0,tile_bits=0;
    std::size_t group=0;
    std::string log;
    ~PlanState() {
        if(global)clReleaseKernel(global);if(tile)clReleaseKernel(tile);
        if(program)clReleaseProgram(program);
    }
};
struct CompletionState {
    std::shared_ptr<PlanState> plan;
    std::shared_ptr<BufferState> a,b;
    cl_event first=nullptr,last=nullptr;
    bool flushed=false;
    ~CompletionState() {
        // Keep borrowed buffers/tables alive until actual device completion.
        if(last){if(!flushed)(void)clFlush(plan->runtime->queue);(void)clWaitForEvents(1,&last);}
        if(first)clReleaseEvent(first);if(last)clReleaseEvent(last);
    }
    void wait() const {
        if(!last)throw std::logic_error("empty OpenCL submission");
        checked(clWaitForEvents(1,&last),"clWaitForEvents");
        cl_int status=0;checked(clGetEventInfo(last,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,nullptr),"clGetEventInfo");
        if(status!=CL_COMPLETE)throw std::runtime_error("OpenCL command failed: execution status "+std::to_string(status));
    }
    void record(cl_event event) {
        if(!first){checked(clRetainEvent(event),"clRetainEvent");first=event;}
        if(last)clReleaseEvent(last);last=event;
    }
};

std::vector<DeviceInfo> devices() {std::vector<DeviceInfo> out;for(const auto &d:inventory())out.push_back(d.description);return out;}
Buffer::Buffer(std::shared_ptr<BufferState> s):state_(std::move(s)){}
std::size_t Buffer::words()const{return state_?state_->count:0;}
Completion::Completion(std::shared_ptr<CompletionState> s):state_(std::move(s)){}
void Completion::wait()const {if(!state_)throw std::logic_error("empty completion");state_->wait();}
double Completion::seconds()const {
    wait();if(!state_->plan->runtime->profiling)throw std::logic_error("OpenCL queue profiling disabled");
    cl_ulong begin=0,end=0;
    checked(clGetEventProfilingInfo(state_->first,CL_PROFILING_COMMAND_START,sizeof(begin),&begin,nullptr),"profiling start");
    checked(clGetEventProfilingInfo(state_->last,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr),"profiling end");
    if(end<begin)throw std::runtime_error("nonmonotonic OpenCL profiling timestamps");return double(end-begin)*1e-9;
}
Runtime::Runtime(const Options &o):state_(std::make_shared<RuntimeState>()) {
    const auto& list=inventory();if(o.device>=list.size())throw std::out_of_range("OpenCL device index unavailable");
    const auto &d=list[o.device];if(!d.description.supported)throw std::runtime_error(d.description.reason);
    if(!o.budget_bytes)throw std::invalid_argument("OpenCL payload budget must be positive");
    const Word endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("little-endian host required");
    state_->device=d.description;state_->id=d.device;state_->profiling=o.profiling;
    state_->cache_programs=o.cache_programs;
    state_->memory.budget_bytes=std::min(o.budget_bytes,d.description.global_bytes);
    cl_int code=0;const cl_context_properties properties[]={CL_CONTEXT_PLATFORM,reinterpret_cast<cl_context_properties>(d.platform),0};
    state_->context=clCreateContext(properties,1,&d.device,nullptr,nullptr,&code);checked(code,"clCreateContext");
    state_->queue=clCreateCommandQueue(state_->context,d.device,o.profiling?CL_QUEUE_PROFILING_ENABLE:0,&code);
    checked(code,"clCreateCommandQueue");
    std::filesystem::path root=o.resource_root;
#ifdef ECM_OPENCL_NTT_RESOURCE_ROOT
    if(root.empty())root=ECM_OPENCL_NTT_RESOURCE_ROOT;
#endif
    if(root.empty())throw std::invalid_argument("OpenCL NTT resource root required");
    state_->resource_root=root.string();
    state_->source=read_source(root/"goldilocks.clh")+"\n#line 1 \"transform.cl\"\n"+read_source(root/"transform.cl");
}
const DeviceInfo &Runtime::device()const{return state_->device;}
MemoryInfo Runtime::memory()const{return state_->memory;}
ProgramInfo Runtime::programs()const{return state_->programs;}
Buffer Runtime::allocate(std::size_t words){return Buffer(ntt::allocate(state_,words));}
void Runtime::upload(const Buffer &b,const Word *p,std::size_t n,std::size_t offset) {
    span(b.state_,state_,n,offset);transfer(b.state_,const_cast<Word*>(p),n,offset,true);
}
void Runtime::download(const Buffer &b,Word *p,std::size_t n,std::size_t offset) {
    span(b.state_,state_,n,offset);transfer(b.state_,p,n,offset,false);
}
void Runtime::finish(){checked(clFinish(state_->queue),"clFinish");}
std::string Runtime::source_identity()const{return identity(state_->source);}
cl_context NativeAccess::context(const Runtime &r){return r.state_->context;}
cl_command_queue NativeAccess::queue(const Runtime &r){return r.state_->queue;}
cl_device_id NativeAccess::device(const Runtime &r){return r.state_->id;}
cl_mem NativeAccess::memory(const Buffer &b){if(!b.state_)throw std::invalid_argument("empty native buffer");return b.state_->memory;}
bool NativeAccess::same_owner(const Buffer &a,const Buffer &b){return a.state_==b.state_;}
void NativeAccess::span(const Runtime &r,const Buffer &b,std::size_t n,std::size_t offset){ntt::span(b.state_,r.state_,n,offset);}
std::string NativeAccess::resource_root(const Runtime &r){return r.state_->resource_root;}
cl_program NativeAccess::program(const Runtime &r,const std::string &source,const std::string &options) {
    auto& state=*r.state_;const auto key=std::make_pair(source,options);
    if(state.cache_programs){const auto found=state.program_cache.find(key);
        if(found!=state.program_cache.end()){checked(clRetainProgram(found->second),"retain cached program");++state.programs.hits;return found->second;}}
    const auto start=std::chrono::steady_clock::now();const char*text=source.c_str();const auto bytes=source.size();cl_int code=0;
    auto program=clCreateProgramWithSource(state.context,1,&text,&bytes,&code);checked(code,"clCreateProgramWithSource");
    const auto built=clBuildProgram(program,1,&state.id,options.c_str(),nullptr,nullptr);
    state.programs.build_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();++state.programs.builds;
    if(built!=CL_SUCCESS){std::size_t log_bytes=0;clGetProgramBuildInfo(program,state.id,CL_PROGRAM_BUILD_LOG,0,nullptr,&log_bytes);std::string log(log_bytes,'\0');
        if(log_bytes)clGetProgramBuildInfo(program,state.id,CL_PROGRAM_BUILD_LOG,log_bytes,log.data(),nullptr);clReleaseProgram(program);
        throw std::runtime_error("OpenCL program compilation failed: "+std::to_string(built)+"\n"+options+"\n"+log);}
    if(state.cache_programs){
        // The cache takes the initial reference, the caller takes a second one.
        // Roll back on host allocation failure without leaking the new program.
        try{state.program_cache.emplace(key,program);}catch(...){clReleaseProgram(program);throw;}
        checked(clRetainProgram(program),"retain new cached program");
    }
    return program;
}

Plan::Plan(Runtime &r,const PlanOptions &o):state_(std::make_shared<PlanState>()) {
    auto &p=*state_;p.runtime=r.state_;
    if(o.log2_length<1 || o.log2_length>29)throw std::invalid_argument("OpenCL NTT log2_length outside 1..29");
    if(o.tile_bits<1 || o.tile_bits>12)throw std::invalid_argument("OpenCL NTT tile_bits outside 1..12");
    if(!o.work_group || (o.work_group&(o.work_group-1)))throw std::invalid_argument("work-group must be a power of two");
    if(o.work_group>r.device().max_work_group)throw std::invalid_argument("work-group exceeds device limit");
    const auto dims=info<cl_uint>(p.runtime->id,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    std::vector<std::size_t> sizes(dims);checked(clGetDeviceInfo(p.runtime->id,CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizes.size()*sizeof(sizes[0]),sizes.data(),nullptr),"CL_DEVICE_MAX_WORK_ITEM_SIZES");
    if(o.work_group>sizes[0])throw std::invalid_argument("work-group exceeds first-dimension limit");
    p.n=Word(1)<<o.log2_length;p.tile_bits=std::min(o.tile_bits,o.log2_length);p.group=o.work_group;
    if((Word(8)<<p.tile_bits)>r.device().local_bytes)throw std::invalid_argument("NTT tile exceeds local memory");
    const std::string options="-cl-std=CL1.2 -DECM_NTT_TILE_BITS="+std::to_string(p.tile_bits)+
        " -DECM_NTT_SPLIT_MUL="+std::to_string(o.split_multiply?1:0);
    cl_int code=0;p.program=NativeAccess::program(r,p.runtime->source,options);
    std::size_t log_size=0;checked(clGetProgramBuildInfo(p.program,p.runtime->id,CL_PROGRAM_BUILD_LOG,0,nullptr,&log_size),"build log size");
    std::string log(log_size,'\0');if(log_size)checked(clGetProgramBuildInfo(p.program,p.runtime->id,CL_PROGRAM_BUILD_LOG,log_size,log.data(),nullptr),"build log");
    while(!log.empty() && log.back()=='\0')log.pop_back();p.log=options+"\n"+log;
    p.global=clCreateKernel(p.program,"ecm_ntt_global",&code);checked(code,"global kernel");
    p.tile=clCreateKernel(p.program,"ecm_ntt_tile",&code);checked(code,"tile kernel");
    for(auto k:{p.global,p.tile}) {
        std::size_t max_group=0;cl_ulong local=0;
        checked(clGetKernelWorkGroupInfo(k,p.runtime->id,CL_KERNEL_WORK_GROUP_SIZE,sizeof(max_group),&max_group,nullptr),"kernel group limit");
        checked(clGetKernelWorkGroupInfo(k,p.runtime->id,CL_KERNEL_LOCAL_MEM_SIZE,sizeof(local),&local,nullptr),"kernel local memory");
        if(p.group>max_group || local>r.device().local_bytes)throw std::runtime_error("NTT kernel resource limit exceeded");
    }
    const Word omega=power(7,(modulus-1)/p.n),inverse=power(omega,modulus-2);
    p.scale=power(p.n,modulus-2);
    if(power(omega,p.n)!=1 || power(omega,p.n/2)!=modulus-1 || multiply(p.n,p.scale)!=1)
        throw std::runtime_error("invalid Goldilocks root or normalization");
    p.low_bits=std::min(12u,o.log2_length-1);p.low_count=Word(1)<<p.low_bits;
    p.high_count=(p.n/2+p.low_count-1)/p.low_count;
    std::vector<Word> low(2*p.low_count),high(2*p.high_count);
    for(unsigned direction=0;direction<2;++direction) {
        const Word root=direction?inverse:omega,step=power(root,p.low_count);Word value=1;
        for(Word i=0;i<p.low_count;++i){low[direction*p.low_count+i]=value;value=multiply(value,root);}
        value=1;for(Word i=0;i<p.high_count;++i){high[direction*p.high_count+i]=value;value=multiply(value,step);}
    }
    p.low=ntt::allocate(p.runtime,low.size());p.high=ntt::allocate(p.runtime,high.size());
    transfer(p.low,low.data(),low.size(),0,true);transfer(p.high,high.data(),high.size(),0,true);
}
std::size_t Plan::length()const{return std::size_t(state_->n);}
unsigned Plan::tile_bits()const{return state_->tile_bits;}
Word Plan::table_bytes()const{return state_->low->bytes+state_->high->bytes;}
std::string Plan::build_log()const{return state_->log;}

namespace {
void view(const std::shared_ptr<PlanState> &p,const std::shared_ptr<BufferState> &b,
          std::size_t batch,std::size_t stride,std::size_t offset) {
    if(!batch || stride<p->n)throw std::invalid_argument("invalid NTT batch/stride");
    const auto max=std::numeric_limits<std::size_t>::max();
    if(batch-1>(max-std::size_t(p->n))/stride)throw std::overflow_error("NTT batch span overflow");
    span(b,p->runtime,(batch-1)*stride+std::size_t(p->n),offset);
}
void launch(const std::shared_ptr<CompletionState> &ticket,cl_kernel kernel,
            std::size_t x,std::size_t batch) {
    const std::size_t local[]={ticket->plan->group,1},global[]={x,batch};cl_event event=nullptr;
    checked(clEnqueueNDRangeKernel(ticket->plan->runtime->queue,kernel,2,nullptr,global,local,0,nullptr,&event),"clEnqueueNDRangeKernel");
    ticket->record(event);
}
void global_pass(const std::shared_ptr<CompletionState> &t,Word half,bool inverse,Word offset,Word stride,std::size_t batch) {
    const auto &p=*t->plan;auto k=p.global;cl_uint direction=inverse?1:0;
    arg(k,0,t->a->memory);arg(k,1,offset);arg(k,2,stride);arg(k,3,p.n);arg(k,4,half);arg(k,5,direction);
    arg(k,6,p.low->memory);arg(k,7,p.high->memory);arg(k,8,p.low_bits);arg(k,9,p.low_count);arg(k,10,p.high_count);
    const auto items=std::size_t(p.n/2),groups=(items+p.group-1)/p.group;launch(t,k,groups*p.group,batch);
}
void tile_pass(const std::shared_ptr<CompletionState> &t,bool inverse,bool product,Word offset,Word stride,std::size_t batch) {
    const auto &p=*t->plan;auto k=p.tile;cl_uint direction=inverse?1:0,mul=product?1:0;
    const auto other=product?t->b->memory:t->a->memory;
    arg(k,0,t->a->memory);arg(k,1,other);arg(k,2,offset);arg(k,3,stride);arg(k,4,p.n);arg(k,5,direction);
    arg(k,6,mul);arg(k,7,p.scale);arg(k,8,p.low->memory);arg(k,9,p.high->memory);arg(k,10,p.low_bits);
    arg(k,11,p.low_count);arg(k,12,p.high_count);
    const auto tiles=std::size_t(p.n>>p.tile_bits);launch(t,k,tiles*p.group,batch);
}
void forward_passes(const std::shared_ptr<CompletionState> &t,Word offset,Word stride,std::size_t batch) {
    const Word tile=Word(1)<<t->plan->tile_bits;
    for(Word half=t->plan->n/2;half>=tile;half/=2)global_pass(t,half,false,offset,stride,batch);
    tile_pass(t,false,false,offset,stride,batch);
}
void inverse_passes(const std::shared_ptr<CompletionState> &t,bool product,Word offset,Word stride,std::size_t batch) {
    tile_pass(t,true,product,offset,stride,batch);
    for(Word half=Word(1)<<t->plan->tile_bits;half<t->plan->n;half*=2)global_pass(t,half,true,offset,stride,batch);
}
std::shared_ptr<CompletionState> ticket(const std::shared_ptr<PlanState> &p,
    const std::shared_ptr<BufferState> &a,const std::shared_ptr<BufferState> &b,
    std::size_t batch,std::size_t stride,std::size_t offset) {
    view(p,a,batch,stride,offset);if(b)view(p,b,batch,stride,offset);
    auto t=std::make_shared<CompletionState>();t->plan=p;t->a=a;t->b=b;return t;
}
void flush(const std::shared_ptr<CompletionState> &t) {checked(clFlush(t->plan->runtime->queue),"clFlush");t->flushed=true;}
} // namespace
Completion Plan::forward(const Buffer &a,std::size_t batch,std::size_t stride,std::size_t offset) {
    if(!stride)stride=length();auto t=ticket(state_,a.state_,{},batch,stride,offset);
    forward_passes(t,offset,stride,batch);flush(t);return Completion(std::move(t));
}
Completion Plan::inverse(const Buffer &a,std::size_t batch,std::size_t stride,std::size_t offset) {
    if(!stride)stride=length();auto t=ticket(state_,a.state_,{},batch,stride,offset);
    inverse_passes(t,false,offset,stride,batch);flush(t);return Completion(std::move(t));
}
Completion Plan::convolve(const Buffer &a,const Buffer &b,std::size_t batch,std::size_t stride,std::size_t offset) {
    if(a.state_==b.state_)throw std::invalid_argument("convolution requires distinct input owners");
    if(!stride)stride=length();auto t=ticket(state_,a.state_,b.state_,batch,stride,offset);
    forward_passes(t,offset,stride,batch);
    std::swap(t->a,t->b);forward_passes(t,offset,stride,batch);std::swap(t->a,t->b);
    inverse_passes(t,true,offset,stride,batch);flush(t);return Completion(std::move(t));
}
} // namespace ecm_opencl::ntt
