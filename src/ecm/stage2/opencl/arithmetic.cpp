#include "arithmetic.h"
#include "ecm/opencl/ntt/native.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ecm_opencl::stage2 {
namespace {
using Native=ntt::NativeAccess;
void check(cl_int code,const char *where){if(code!=CL_SUCCESS)throw std::runtime_error(std::string(where)+": OpenCL "+std::to_string(code));}
template<class T>void arg(cl_kernel k,unsigned i,const T &v){check(clSetKernelArg(k,i,sizeof(v),&v),"coefficient argument");}
std::size_t times(std::size_t a,std::size_t b){if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("coefficient size overflow");return a*b;}
std::string read(const std::filesystem::path &p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
}
Polynomial Polynomial::element(std::size_t i)const {
    if(i>=batch)throw std::out_of_range("polynomial batch element");
    auto p=*this;p.offset+=times(i,stride);p.batch=1;return p;
}
struct ArithmeticState {
    ntt::Runtime runtime;ArithmeticOptions options;ArithmeticStats stats;
    std::size_t nw=0;ntt::Buffer modulus;
    cl_program program=nullptr;std::array<cl_kernel,3> kernels{};
    std::shared_ptr<PolynomialWorkspace> workspace;
    std::map<std::size_t,std::unique_ptr<PolynomialPlan>> products;
    // Commands retain OpenCL objects, but the payload ledger follows Buffer
    // owners. Keep every async input/output until an actual queue drain.
    std::vector<ntt::Buffer> pending;
    ArithmeticState(ntt::Runtime &r,const ArithmeticOptions &o):runtime(r),options(o){}
    ~ArithmeticState(){try{runtime.finish();}catch(...){}pending.clear();for(auto k:kernels)if(k)clReleaseKernel(k);if(program)clReleaseProgram(program);}
    void validate(const Polynomial &p)const {
        if(!p.size || !p.batch || p.stride<times(p.size,nw))throw std::invalid_argument("invalid coefficient view");
        const auto tail=times(p.batch-1,p.stride),words=times(p.size,nw);
        if(tail>std::numeric_limits<std::size_t>::max()-words)throw std::overflow_error("coefficient span overflow");
        Native::span(runtime,p.buffer,tail+words,p.offset);
    }
    Polynomial allocate(std::size_t count,std::size_t batch){
        const auto stride=times(count,nw),size=times(stride,batch);ntt::Buffer buffer;
        try{buffer=runtime.allocate(size);}catch(const std::runtime_error &e){
            if(!workspace || std::string(e.what()).find("owned payload budget exceeded")==std::string::npos)throw;
            // Workspace never owns result/coefficient leases and no product is
            // in flight here. Reclaim idle scratch, without changing math/batch.
            workspace->release();buffer=runtime.allocate(size);
        }
        return {buffer,count,batch,0,stride};
    }
    void hold(const Polynomial &p){pending.push_back(p.buffer);}
    void launch(cl_kernel k,std::size_t count,std::size_t batch){const std::size_t global[]={count,batch};
        check(clEnqueueNDRangeKernel(Native::queue(runtime),k,2,nullptr,global,nullptr,0,nullptr,nullptr),"coefficient launch");++stats.coefficient_launches;}
};
Arithmetic::Arithmetic(ntt::Runtime &r,const ArithmeticOptions &o):state_(std::make_unique<ArithmeticState>(r,o)) {
    auto &p=*state_;
    p.options.product.workspace.reset();
    // Validate modulus/NTT options once, even before the first multiplication.
    auto initial=o.product;initial.capacity=1;initial.workspace.reset();PolynomialPlan validation(r,initial);p.nw=validation.coefficient_words();
    if(o.reuse_workspace)p.workspace=std::make_shared<PolynomialWorkspace>();
    p.modulus=r.allocate(p.nw);r.upload(p.modulus,o.product.modulus.data(),p.nw);p.stats.upload_bytes+=p.nw*8;
    const std::filesystem::path root=Native::resource_root(r);
    const auto source=read(root.parent_path()/"stage2/coefficients.cl");cl_int code=0;
    const auto options="-cl-std=CL1.2 -DECM_COEFF_NW="+std::to_string(p.nw);
    p.program=Native::program(r,source,options);
    unsigned i=0;for(const char *name:{"ecm_poly_view","ecm_poly_sub","ecm_poly_linear"}){p.kernels[i++]=clCreateKernel(p.program,name,&code);check(code,name);}
}
Arithmetic::~Arithmetic()=default;
std::size_t Arithmetic::words()const{return state_->nw;}
bool Arithmetic::group_batches()const{return state_->options.group_batches;}
const ArithmeticStats &Arithmetic::stats()const{return state_->stats;}
void Arithmetic::finish(){state_->runtime.finish();state_->pending.clear();}
Polynomial Arithmetic::upload(const std::vector<Word> &v,std::size_t count,std::size_t batch) {
    auto &p=*state_;if(!count || !batch || v.size()!=times(times(count,batch),p.nw))throw std::invalid_argument("coefficient upload shape");
    // The small coefficient operators assume canonical inputs. Check at the
    // public host boundary; device products preserve the invariant thereafter.
    for(std::size_t k=0;k<count*batch;++k){int cmp=0;for(std::size_t j=p.nw;j>0;--j)if(!cmp && v[k*p.nw+j-1]!=p.options.product.modulus[j-1])cmp=v[k*p.nw+j-1]<p.options.product.modulus[j-1]?-1:1;
        if(cmp>=0)throw std::invalid_argument("noncanonical coefficient upload");}
    auto out=p.allocate(count,batch);p.runtime.upload(out.buffer,v.data(),v.size());p.pending.clear();p.stats.upload_bytes+=v.size()*8;return out;
}
std::vector<Word> Arithmetic::download(const Polynomial &a) {
    auto &p=*state_;p.validate(a);auto packed=a;
    // A scaled frontier may be a strided middle-product window. Compact once
    // on device instead of performing one blocking read for every leaf.
    if(a.batch>1 && a.stride!=a.size*p.nw)packed=reshape(a,a.size);
    std::vector<Word> out(times(times(a.size,a.batch),p.nw));
    p.runtime.download(packed.buffer,out.data(),out.size(),packed.offset);
    p.pending.clear();p.stats.download_bytes+=out.size()*8;return out;
}
Polynomial Arithmetic::constant(Word value){std::vector<Word> v(words());v[0]=words()==1?value%state_->options.product.modulus[0]:value;return upload(v,1);}
Polynomial Arithmetic::window(const Polynomial &a,std::size_t first,std::size_t count)const {
    state_->validate(a);if(!count || first>a.size || count>a.size-first)throw std::out_of_range("coefficient window");
    auto out=a;out.offset+=times(first,words());out.size=count;return out;
}
Polynomial Arithmetic::reshape(const Polynomial &a,std::size_t count,std::size_t first,bool reverse) {
    auto &p=*state_;p.validate(a);if(!count || first>std::numeric_limits<std::size_t>::max()-count)throw std::invalid_argument("coefficient reshape range");
    auto out=p.allocate(count,a.batch);p.hold(a);p.hold(out);auto k=p.kernels[0];const auto src=Native::memory(a.buffer),dst=Native::memory(out.buffer);
    const Word offset=a.offset,stride=a.stride,n=a.size,nout=count,start=first;const cl_uint rev=reverse?1:0;
    arg(k,0,src);arg(k,1,offset);arg(k,2,stride);arg(k,3,n);arg(k,4,dst);arg(k,5,nout);arg(k,6,start);arg(k,7,rev);p.launch(k,count,a.batch);return out;
}
Polynomial Arithmetic::gather(const std::vector<Polynomial> &a) {
    auto &p=*state_;if(a.empty())throw std::invalid_argument("empty coefficient gather");
    for(const auto &v:a){p.validate(v);if(v.size!=a[0].size || v.batch!=1)throw std::invalid_argument("coefficient gather shape");}
    if(p.options.borrow_contiguous) {
        // Product-tree children normally share one level allocation. Represent
        // regular slices as a strided batch; retain ownership through the view.
        auto borrowed=a[0];bool regular=a.size()==1;std::size_t step=0;
        if(a.size()>1 && a[1].offset>=a[0].offset){step=a[1].offset-a[0].offset;regular=step>=a[0].size*p.nw;}
        for(std::size_t i=1;regular && i<a.size();++i)regular=Native::same_owner(a[0].buffer,a[i].buffer) && a[i].offset>=a[0].offset && a[i].offset-a[0].offset==times(i,step);
        if(regular){borrowed.batch=a.size();if(a.size()>1)borrowed.stride=step;p.validate(borrowed);
            ++p.stats.borrowed_gathers;p.stats.avoided_copy_bytes+=times(times(a.size(),a[0].size),p.nw)*8;return borrowed;}
    }
    auto out=p.allocate(a[0].size,a.size());p.hold(out);const auto dst=Native::memory(out.buffer);
    for(std::size_t i=0;i<a.size();++i){p.hold(a[i]);const auto bytes=a[i].size*p.nw*8;
        check(clEnqueueCopyBuffer(Native::queue(p.runtime),Native::memory(a[i].buffer),dst,a[i].offset*8,i*bytes,bytes,0,nullptr,nullptr),"coefficient gather copy");
        ++p.stats.device_copies;p.stats.device_copy_bytes+=bytes;}
    return out;
}
Polynomial Arithmetic::subtract(const Polynomial &a,const Polynomial &b) {
    auto &p=*state_;p.validate(a);p.validate(b);if(a.batch!=b.batch)throw std::invalid_argument("coefficient subtract batch");
    auto out=p.allocate(std::max(a.size,b.size),a.batch);p.hold(a);p.hold(b);p.hold(out);auto k=p.kernels[1];
    const auto ma=Native::memory(a.buffer),mb=Native::memory(b.buffer),mn=Native::memory(p.modulus),mo=Native::memory(out.buffer);
    const Word ao=a.offset,ast=a.stride,na=a.size,bo=b.offset,bst=b.stride,nb=b.size,count=out.size;
    arg(k,0,ma);arg(k,1,ao);arg(k,2,ast);arg(k,3,na);arg(k,4,mb);arg(k,5,bo);arg(k,6,bst);arg(k,7,nb);arg(k,8,mn);arg(k,9,mo);arg(k,10,count);p.launch(k,out.size,out.batch);return out;
}
Polynomial Arithmetic::linear_factors(const Polynomial &roots) {
    auto &p=*state_;p.validate(roots);if(roots.batch!=1)throw std::invalid_argument("root list must have batch=1");
    auto out=p.allocate(2,roots.size);p.hold(roots);p.hold(out);auto k=p.kernels[2];const auto src=Native::memory(roots.buffer),mod=Native::memory(p.modulus),dst=Native::memory(out.buffer);
    const Word offset=roots.offset,count=roots.size;arg(k,0,src);arg(k,1,offset);arg(k,2,count);arg(k,3,mod);arg(k,4,dst);p.launch(k,roots.size,1);return out;
}
Polynomial Arithmetic::multiply(const Polynomial &a,const Polynomial &b,std::size_t first,std::size_t count) {
    auto &p=*state_;p.validate(a);p.validate(b);if(a.batch!=b.batch)throw std::invalid_argument("polynomial multiply batch");
    const auto nc=a.size+b.size-1;if(first>=nc)throw std::out_of_range("product window");count=std::min(count,nc-first);if(!count)throw std::invalid_argument("empty product window");
    const auto cap=std::max(a.size,b.size);auto &plan=p.products[cap];
    if(!plan){auto options=p.options.product;options.capacity=cap;options.workspace=p.workspace;plan=std::make_unique<PolynomialPlan>(p.runtime,options);}
    auto out=p.allocate(nc,a.batch);
    const auto info=plan->multiply({a.buffer,a.offset,a.stride},a.size,{b.buffer,b.offset,b.stride},b.size,{out.buffer,0,out.stride},a.batch);
    p.pending.clear();++p.stats.multiply_calls;p.stats.multiply_pairs+=a.batch;
    p.stats.ntt_transforms[plan->shape().log2_length]+=3*a.batch;
    p.stats.download_bytes+=(info.verdict_words+3*info.carry_blocks)*8;return window(out,first,count);
}
Polynomial Arithmetic::inverse_series(const Polynomial &a,std::size_t count) {
    if(!count || a.batch!=1)throw std::invalid_argument("inverse series shape");
    const auto c=download(window(a,0,1));if(c[0]!=1 || std::any_of(c.begin()+1,c.end(),[](Word w){return w!=0;}))throw std::invalid_argument("inverse series requires a(0)=1");
    auto g=constant(1);
    while(g.size<count){const auto next=std::min(2*g.size,count);auto ag=multiply(window(a,0,std::min(a.size,next)),g,0,next);
        ag=reshape(ag,next);auto h=subtract(constant(2),ag);g=multiply(g,h,0,next);g=reshape(g,next);}
    return g;
}
Polynomial Arithmetic::remainder(const Polynomial &a,const Polynomial &b,const Polynomial &inverse) {
    if(a.batch!=1 || b.batch!=1 || inverse.batch!=1 || b.size<2)throw std::invalid_argument("monic remainder shape");
    const auto degree=b.size-1;if(a.size<=degree)return reshape(a,degree);
    const auto k=a.size-degree;if(inverse.size<k)throw std::invalid_argument("short monic inverse");
    // q = reverse(low_k(reverse(a) * inverse(reverse(b)))); r = low_d(a-q*b).
    auto ra=reshape(a,k,a.size-1,true),qr=multiply(ra,window(inverse,0,k),0,k);
    auto q=reshape(qr,k,k-1,true),qb=multiply(q,b,0,degree);
    return window(subtract(reshape(a,degree),reshape(qb,degree)),0,degree);
}
} // namespace ecm_opencl::stage2
