// Native OpenCL engine behind the existing save/queue driver ABI. The legacy
// ecm_cuda_stage2_* symbol names are a link seam, not a CUDA dependency.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "engine.h"
#include "tree.h"
#include "points.h"
#include "point_plan.h"
#include "../cuda/api.h"
#include "../modulus.h"
#include "../planning/d_catalog.h"
#include "ecm/platform/compute_log.h"
#include <gmp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>

namespace {
namespace cln=ecm_opencl::ntt;
namespace cls=ecm_opencl::stage2;
using Word=cln::Word;
struct Big {
    mpz_t v;
    Big(){mpz_init(v);} explicit Big(Word n):Big(){mpz_import(v,1,-1,8,0,0,&n);}
    Big(const Big&b):Big(){mpz_set(v,b.v);} Big&operator=(const Big&b){mpz_set(v,b.v);return *this;}
    ~Big(){mpz_clear(v);}
};
using Values=std::vector<Big>;
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
Word setting(const char*key,Word fallback){const char*s=std::getenv(key);if(!s||!*s)return fallback;char*end=nullptr;const auto n=std::strtoull(s,&end,10);if(*s=='-'||*end)throw std::runtime_error(std::string("invalid ")+key);return n;}
bool overridden(const char*key){const char*s=std::getenv(key);return s&&*s;}
unsigned scalar_bits(Word value){unsigned bits=0;do{++bits;}while(value>>=1);return bits;}
std::string decimal(mpz_srcptr v){std::string s(mpz_sizeinbase(v,10)+2,'\0');mpz_get_str(s.data(),10,v);s.resize(s.find('\0'));return s;}
std::vector<Word> flatten(const Values&a,std::size_t w){std::vector<Word> out(a.size()*w);for(std::size_t i=0;i<a.size();++i){std::size_t n=0;mpz_export(out.data()+i*w,&n,-1,8,0,0,a[i].v);if(n>w)throw std::runtime_error("coefficient width overflow");}return out;}
Values import(const std::vector<Word>&a,std::size_t w){if(a.size()%w)throw std::runtime_error("coordinate width");Values out(a.size()/w);for(std::size_t i=0;i<out.size();++i)mpz_import(out[i].v,w,-1,8,0,0,a.data()+i*w);return out;}
Big add(const Big&a,const Big&b,mpz_srcptr n){Big r;mpz_add(r.v,a.v,b.v);mpz_mod(r.v,r.v,n);return r;}
Big sub(const Big&a,const Big&b,mpz_srcptr n){Big r;mpz_sub(r.v,a.v,b.v);mpz_mod(r.v,r.v,n);return r;}
Big mul(const Big&a,const Big&b,mpz_srcptr n){Big r;mpz_mul(r.v,a.v,b.v);mpz_mod(r.v,r.v,n);return r;}
struct Point {Big x,z;};
Point twice(const Point&p,const Big&a24,mpz_srcptr n){auto a=add(p.x,p.z,n),b=sub(p.x,p.z,n);a=mul(a,a,n);b=mul(b,b,n);auto e=sub(a,b,n);return {mul(a,b,n),mul(e,add(b,mul(a24,e,n),n),n)};}
Point sum(const Point&a,const Point&b,const Point&diff,mpz_srcptr n){auto u=mul(sub(b.x,b.z,n),add(a.x,a.z,n),n),v=mul(add(b.x,b.z,n),sub(a.x,a.z,n),n);auto s=add(u,v,n),t=sub(u,v,n);return {mul(diff.z,mul(s,s,n),n),mul(diff.x,mul(t,t,n),n)};}
Point ladder(const Point&p,Word scalar,const Big&a24,mpz_srcptr n){if(!scalar)return {Big(1),Big(0)};auto a=p,b=twice(p,a24,n);unsigned bit=63;while(!((scalar>>bit)&1))--bit;while(bit--){auto c=sum(a,b,p,n);if((scalar>>bit)&1){b=twice(b,a24,n);a=c;}else{a=twice(a,a24,n);b=c;}}return a;}
void factor(std::set<std::string>&f,mpz_srcptr v,mpz_srcptr n){if(mpz_cmp_ui(v,1)>0&&mpz_cmp(v,n)<0){if(!mpz_divisible_p(n,v))throw std::runtime_error("invalid factor");f.insert(decimal(v));}}

// A batch of 256 projective denominators needs one GMP inverse.
// On inverse failure inspect every denominator before accepting any result.
Values normalize(const Values&x,const Values&z,mpz_srcptr n,std::set<std::string>&f){
    Values out(x.size()),prefix(257);Big inv,t,g;
    for(std::size_t lo=0;lo<x.size();lo+=256){const auto count=std::min(std::size_t(256),x.size()-lo);mpz_set_ui(prefix[0].v,1);
        for(std::size_t j=0;j<count;++j)prefix[j+1]=mul(prefix[j],z[lo+j],n);
        if(mpz_invert(inv.v,prefix[count].v,n)){
            for(std::size_t j=count;j-->0;){t=mul(inv,prefix[j],n);inv=mul(inv,z[lo+j],n);out[lo+j]=mul(x[lo+j],t,n);}
        }else{
            bool saturated=false;
            for(std::size_t j=0;j<count;++j){if(mpz_invert(inv.v,z[lo+j].v,n))out[lo+j]=mul(x[lo+j],inv,n);else{mpz_gcd(g.v,z[lo+j].v,n);factor(f,g.v,n);saturated|=mpz_cmp(g.v,n)==0;}}
            // Never manufacture an affine root for the point at infinity.
            if(saturated&&f.empty())throw std::runtime_error("saturated projective denominator; no affine Stage2 result can be published");
        }
    }return out;
}
std::string resources(){std::vector<wchar_t> p(32768);auto n=GetModuleFileNameW(nullptr,p.data(),DWORD(p.size()));if(!n||n==p.size())throw std::runtime_error("executable path");return (std::filesystem::path(std::wstring(p.data(),n)).parent_path()/"opencl_resources/ntt").string();}
Word budget(const cln::DeviceInfo&d){const auto kb=setting("NTT_ARENA_CAP_KB",0);if(kb>std::numeric_limits<Word>::max()/1024)throw std::runtime_error("arena budget overflow");return kb?kb*1024:std::min(Word(1024)<<20,d.global_bytes/2);}
struct Shape {Word d=0,p=0,giants=0,chunk=0,bytes=0;double score=0;};
Shape shape(Word d,Word b2,unsigned bits,Word limit){
    Shape s;s.d=d;s.p=ecm_stage2::d_catalog::degree(d);s.giants=b2/d+2;
    const auto batch_mb=setting("NTT_S4_BATCH_MB",64);
    if(!batch_mb||batch_mb>(Word(1)<<20))throw std::runtime_error("invalid OpenCL batch budget");
    s.chunk=std::min(s.p,std::max(Word(1),(batch_mb<<20)/(6*((bits+63)/64)*8)));
    const auto packed=cls::integer_shape(bits,std::size_t(s.p+1));
    // Conservative payload admission: retained F, concurrent G/fold/descent,
    // shared NTT high-water scratch and cached root tables. Runtime enforces
    // the hard owned-payload budget independently; this is not free VRAM.
    const double l=std::ceil(std::log2(double(s.p)))+1;
    const double bytes=64.*packed.length+double(s.p)*(bits+63)/64*8*(4*l+32)+(32u<<20);
    if(bytes>double(limit)||bytes>double(std::numeric_limits<Word>::max()))return {};
    s.bytes=Word(bytes);const double groups=std::ceil(double(s.giants)/s.chunk);
    s.score=double(s.p)*l*l+groups*(double(s.chunk)*std::pow(std::log2(double(s.chunk)+1),2)+double(s.p)*l);
    s.score+=double(s.p+s.giants)*std::log2(double(b2)+1);return s;
}
Shape choose(Word requested,Word b2,unsigned bits,Word limit){
    if(requested){if(requested<6||requested%2||requested>200000000)throw std::runtime_error("D must be even, 6..200000000");auto s=shape(requested,b2,bits,limit);if(!s.d)throw std::runtime_error("OpenCL D exceeds conservative payload budget");return s;}
    Shape best;for(const auto&e:ecm_stage2::d_catalog::entries()){try{const auto s=shape(e.d,b2,bits,limit);if(s.d&&(!best.d||s.score<best.score))best=s;}catch(const std::exception&){}}
    if(!best.d)throw std::runtime_error("no OpenCL D fits the payload budget");return best;
}
bool counts_enabled=false;
void print_shape(const Shape&s,Word b1,Word b2,unsigned bits){stage2_log::print(stage2_log::phases,"real_shape: D=%llu P=phi(D)/2=%llu baby_points=%llu giant_points=%llu num_poly_g=%llu loops=%llu B1=%llu B2=%llu S_bits=%u\n",s.d,s.p,s.p,s.giants,(s.giants+s.chunk-1)/s.chunk,(s.giants+s.chunk-1)/s.chunk-1,b1,b2,bits);}
struct PointGenerator {
    cln::Runtime &runtime;mpz_srcptr n;Big a24;Point base;std::size_t w;std::unique_ptr<cls::Points> device;
    double kernel_seconds=0,kernel_max_seconds=0,coordinate_seconds=0,check_seconds=0,normalize_seconds=0,seed_seconds=0;
    Word launches=0,points=0,upload_bytes=0,download_bytes=0;
    std::size_t chunk=0,group=0;
    cls::PointPlan plan{256,32,0,false};Word promoted_launches=0,max_command_points=0,budget_clamps=0;
    bool chain_requested=false,chain_enabled=false;
    Word chain_launches=0,chain_points=0,chain_seeds=0,chain_recoveries=0;
    double chain_recovery_seconds=0,chain_kernel_seconds=0,chain_max_seconds=0,giant_kernel_seconds=0;
    Word giant_step=0;Point giant_base;bool use_giant_base=false;
    PointGenerator(cln::Runtime&r,mpz_srcptr modulus,const Big&x,const Big&a):runtime(r),n(modulus),a24(a),base{x,Big(1)},w((mpz_sizeinbase(n,2)+63)/64){
        const auto&v=r.device().vendor;const bool amd=v.find("Advanced Micro Devices")!=v.npos||v=="AMD";
        const bool nvidia=v.find("NVIDIA")!=v.npos;
        const auto automatic=setting("ECM_OPENCL_POINT_AUTO",1);
        if(automatic>1)throw std::runtime_error("ECM_OPENCL_POINT_AUTO must be 0 or 1");
        // The bounded ramp is qualified separately from giant chains. Its
        // actual eligibility below still requires NVIDIA device points and
        // honors explicit chunk/group plans and POINT_AUTO=0.
        const auto wide_auto=setting("ECM_OPENCL_WIDE_AUTO",1);
        if(wide_auto>1)throw std::runtime_error("ECM_OPENCL_WIDE_AUTO must be 0 or 1");
        // The existing wide-AMD point qualification guard remains intact.
        // CPU points still feed the same native OpenCL polynomial pipeline.
        const char* mode=std::getenv("ECM_OPENCL_POINTS");
        if(mode&&std::string(mode)!="cpu"&&std::string(mode)!="device")throw std::runtime_error("ECM_OPENCL_POINTS must be cpu or device");
        group=std::size_t(setting("ECM_OPENCL_POINT_GROUP",nvidia?32:0));
        if((!mode||std::string(mode)!="cpu")&&(!amd||w<=8))device=std::make_unique<cls::Points>(runtime,flatten({BigValue()},w),group);
        const std::size_t default_chunk=w<=8?1024:(nvidia&&w<=128)?256:w<=32?256:w<=64?128:w<=128?64:16;
        chunk=std::size_t(setting("ECM_OPENCL_POINT_CHUNK",default_chunk));
        if(!chunk||chunk>4096)throw std::runtime_error("ECM_OPENCL_POINT_CHUNK must be 1..4096");
        // Respect manual chunk/group plans and keep the existing wide/vendor
        // qualification guards. Only ordinary useful commands train the plan.
        const bool wide_ramp=wide_auto&&w>128&&w<=256;
        const bool adaptive=automatic&&device&&nvidia&&w>8&&(w<=128||wide_ramp)&&group==32&&
            !overridden("ECM_OPENCL_POINT_CHUNK")&&!overridden("ECM_OPENCL_POINT_GROUP");
        plan=cls::PointPlan(chunk,group,r.device().compute_units,adaptive,wide_ramp);
        const auto use=setting("ECM_OPENCL_GIANT_BASE",1);if(use>1)throw std::runtime_error("ECM_OPENCL_GIANT_BASE must be 0 or 1");use_giant_base=use!=0;
        const auto chain=setting("ECM_OPENCL_GIANT_CHAIN",0);if(chain>1)throw std::runtime_error("ECM_OPENCL_GIANT_CHAIN must be 0 or 1");
        chain_requested=chain!=0;chain_enabled=chain_requested&&device&&use_giant_base;
    }
    Word chain_capacity()const {
        if(overridden("ECM_OPENCL_POINT_CHUNK"))return chunk; // Manual command bounds also govern chains.
        return std::max(Word(chunk),std::min(Word(4096),std::max(Word(1),Word(runtime.device().compute_units))*std::max(Word(1),Word(group))*4));
    }
    Big BigValue(){Big value;mpz_set(value.v,n);return value;}
    Values generate(const std::vector<Word>&scalars,std::set<std::string>&f,Word&checked,Word step=1,std::size_t sample_span=0){
        if(scalars.empty())throw std::runtime_error("empty point generation");
        const Point* origin=&base;std::vector<Word> reduced;
        if(step>1&&use_giant_base){
            if(giant_step!=step){
                const auto start=now();auto seed=ladder(base,step,a24,n);
                // Nonunit [D]Q is already one of the required giant points.
                // Inspect it before changing the base; never hide its factor.
                auto affine=normalize({seed.x},{seed.z},n,f);seed_seconds+=now()-start;
                if(!f.empty())return {};
                giant_base={affine[0],Big(1)};giant_step=step;
            }
            origin=&giant_base;reduced.reserve(scalars.size());
            for(auto scalar:scalars){if(scalar%step)throw std::runtime_error("nonmultiple giant scalar");reduced.push_back(scalar/step);}
        }
        const auto&work=reduced.empty()?scalars:reduced;
        const bool chained=chain_enabled&&step>1&&!reduced.empty()&&work.size()>1;
        if(chained)for(std::size_t i=1;i<work.size();++i)if(work[i-1]==std::numeric_limits<Word>::max()||work[i]!=work[i-1]+1)throw std::runtime_error("nonconsecutive giant chain scalars");
        points+=scalars.size();const auto begin=now();Values x,z;if(device){
            const auto bits=scalar_bits(*std::max_element(work.begin(),work.end()));
            // Bound each scalar-ladder submission on display GPUs. Wide scalar
            // kernels are slow; a whole production giant batch can otherwise
            // become a single multi-second command. This is separate from the
            // polynomial batch size and is not a calibrated watchdog guarantee.
            for(std::size_t first=0;first<scalars.size();){const auto wanted=chained?std::size_t(chain_capacity()):plan.chunk(bits),remaining=scalars.size()-first;
                const auto mem=runtime.memory();const auto available=(mem.budget_bytes-mem.live_bytes)/(8*(2*w+1));
                const auto count=chained?std::min(remaining,std::max(std::min(chunk,remaining),std::min(wanted,std::size_t(available)))):plan.admit(wanted,remaining,std::size_t(available)),last=first+count;
                budget_clamps+=count<std::min(wanted,remaining);
                std::vector<Word> slice(work.begin()+first,work.begin()+last);
                const auto lanes=std::max(Word(1),Word(runtime.device().compute_units))*std::max(Word(1),Word(group));
                const auto segment=std::size_t(std::clamp((Word(count)+lanes-1)/lanes,Word(2),Word(32)));
                auto points=chained?device->consecutive(flatten({origin->x},w),flatten({origin->z},w),flatten({a24},w),slice.front(),count,segment):device->ladder(flatten({origin->x},w),flatten({origin->z},w),flatten({a24},w),slice);
                kernel_seconds+=device->timing().kernel_seconds;upload_bytes+=device->timing().upload_bytes;++launches;
                if(step>1)giant_kernel_seconds+=device->timing().kernel_seconds;
                kernel_max_seconds=std::max(kernel_max_seconds,device->timing().kernel_seconds);
                if(chained){++chain_launches;chain_points+=count;chain_seeds+=1+(count-1)/segment;
                    chain_kernel_seconds+=device->timing().kernel_seconds;chain_max_seconds=std::max(chain_max_seconds,device->timing().kernel_seconds);}
                else{plan.observe(device->timing().kernel_seconds,scalar_bits(*std::max_element(slice.begin(),slice.end())),slice.size());
                    promoted_launches+=slice.size()>chunk;max_command_points=std::max(max_command_points,Word(slice.size()));}
                auto xx=import(device->download(points.x),w),zz=import(device->download(points.z),w);
                download_bytes+=2*slice.size()*w*8;
                if(chained){
                    // A differential chain may acquire a nonunit projective
                    // scale on a composite target. Never interpret that scale
                    // as a factor: recompute the entire affected command from
                    // the ORIGINAL Q and full iD before normalization/oracles.
                    const auto recovery_begin=now();Big product(1),g;
                    for(const auto&v:zz)product=mul(product,v,n);
                    mpz_gcd(g.v,product.v,n);
                    if(mpz_cmp_ui(g.v,1)){++chain_recoveries;for(std::size_t j=0;j<count;++j){auto p=ladder(base,scalars[first+j],a24,n);xx[j]=p.x;zz[j]=p.z;}}
                    chain_recovery_seconds+=now()-recovery_begin;
                }
                x.insert(x.end(),xx.begin(),xx.end());z.insert(z.end(),zz.begin(),zz.end());
                first=last;
            }
        }else{for(auto scalar:work){auto p=ladder(*origin,scalar,a24,n);x.push_back(p.x);z.push_back(p.z);}}
        coordinate_seconds+=now()-begin;
        // Always compare rebased points with the original [iD]Q ladder,
        // including the CPU fallback. This oracle does not reuse giant_base.
        if(device||origin!=&base){const auto check_begin=now();
            // A wider generation window must preserve the three original
            // samples of EVERY G batch, including the final short batch.
            const auto span=sample_span?sample_span:scalars.size();
            for(std::size_t lo=0;lo<scalars.size();lo+=span){const auto count=std::min(span,scalars.size()-lo);
                for(auto k:std::set<std::size_t>{lo,lo+count/2,lo+count-1}){const auto oracle=ladder(base,scalars[k],a24,n);auto cross=sub(mul(x[k],oracle.z,n),mul(z[k],oracle.x,n),n);if(mpz_sgn(cross.v))throw std::runtime_error("OpenCL ladder GMP mismatch");checked+=w*2;}}
            check_seconds+=now()-check_begin;
        }
        const auto norm_begin=now();auto out=normalize(x,z,n,f);normalize_seconds+=now()-norm_begin;return out;
    }
};
Big curve_a24(Word sigma,mpz_srcptr n,std::set<std::string>&f){Big s(sigma),u=sub(mul(s,s,n),Big(5),n),v=mul(Big(4),s,n);auto u3=mul(mul(u,u,n),u,n),d=sub(v,u,n);auto numerator=mul(mul(mul(d,d,n),d,n),add(mul(Big(3),u,n),v,n),n),den=mul(Big(16),mul(u3,v,n),n);Big inv,g;if(!mpz_invert(inv.v,den.v,n)){mpz_gcd(g.v,den.v,n);factor(f,g.v,n);if(f.empty())throw std::runtime_error("degenerate Suyama curve");return {}; }return mul(numerator,inv,n);}
std::string result_json(const std::set<std::string>&f){std::string text="\"hits\":"+std::to_string(f.size())+",\"bad_factors\":0,\"factors\":[";for(const auto&v:f){if(text.back()!='[')text+=',';text+='"'+v+'"';}return text+']';}
}

int ecm_opencl_stage2_list_devices(){try{for(const auto&d:cln::devices())std::cout<<"opencl_device: index="<<d.index<<" name="<<d.name<<" vendor="<<d.vendor<<" driver="<<d.driver<<" supported="<<d.supported<<" global_bytes="<<d.global_bytes<<" reason="<<d.reason<<'\n';return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
int ecm_cuda_stage2_default_log_level(){return stage2_log::batches;}
int ecm_cuda_stage2_set_log_level(int level){if(level<0||level>4)return 2;stage2_log::level=level;return 0;}
int ecm_cuda_stage2_set_ntt_counts(bool on){counts_enabled=on;return 0;}
int ecm_cuda_stage2_set_relocation(bool on){return on?2:0;}
int ecm_cuda_stage2_check_configuration(){return 0;}
int ecm_cuda_stage2_set_short_ntt_batches(const EcmStage2NttBatchSample*,size_t count){return count?2:0;}
bool ecm_cuda_stage2_shape_query(uint64_t coefficients,int bits,uint64_t*length,uint64_t*slots){try{auto s=cls::integer_shape(bits,coefficients);if(length)*length=s.length;if(slots)*slots=coefficients*2-1;return true;}catch(...){return false;}}
int ecm_cuda_stage2_device_info(int index,EcmStage2DeviceInfo*info,const char*expected){try{const auto ds=cln::devices();if(index<0||size_t(index)>=ds.size()||!info)return 2;const auto&d=ds[index];*info={};info->total_bytes=d.global_bytes;info->fixed_mode=-2;
    // OpenCL 1.2 has no portable free-memory or UUID query. This is a device
    // identity token, not a hardware UUID and never a free-memory promise.
    Word a=14695981039346656037ull,b=1099511628211ull;for(unsigned char c:d.platform+d.vendor+d.name+d.driver+std::to_string(index)){a=(a^c)*1099511628211ull;b=(b^c)*1099511628211ull;}
    std::snprintf(info->uuid_hex,sizeof(info->uuid_hex),"%016llx%016llx",a,b);return expected&&std::string(expected)!=info->uuid_hex?2:0;}catch(...){return 2;}}
int ecm_cuda_stage2_plan(const char*n, uint64_t sigma,uint64_t b1,uint64_t b2,uint64_t d,int device,void(*report)(const char*,void*),void*ctx,unsigned carrier){try{ecm_stage2::ModulusContext m;std::string error;if(!m.configure(n,16,carrier,error))throw std::runtime_error(error);auto ds=cln::devices();if(device<0||size_t(device)>=ds.size())throw std::runtime_error("OpenCL device index");const auto s=choose(d,b2,unsigned(m.S),budget(ds[device]));print_shape(s,b1,b2,unsigned(m.S));std::ostringstream text;text<<"{\"backend\":\"opencl-ntt\",\"D\":"<<s.d<<",\"P\":"<<s.p<<",\"estimated_payload_bytes\":"<<s.bytes<<",\"payload_budget_bytes\":"<<budget(ds[device])<<",\"live_free_memory_known\":false}";report(text.str().c_str(),ctx);return 0;}catch(const std::exception&e){std::cerr<<"OpenCL plan: "<<e.what()<<'\n';return 2;}}
int ecm_cuda_stage2_tune_ntt(int,int,int,int,uint64_t,void(*)(const char*,void*),void*,bool){return 2;}
int ecm_cuda_stage2_tune_ntt_batches(int,int,int,int,uint64_t,const uint64_t*,size_t,void(*)(const char*,void*),void*,bool){return 2;}

int ecm_cuda_stage2_run(const char*n,const char*x,uint64_t sigma,uint64_t b1,uint64_t b2,uint64_t d,int index,void(*report)(const char*,void*),void*ctx,unsigned carrier,EcmStage2Metrics*metrics,const char*a24){
    try{
        if(!n||!x||!report||b1<2||b2<=b1||b2>uint64_t(INT64_MAX)-8192)throw std::runtime_error("invalid Stage2 bounds");
        stage2_log::phase("Plan Stage2 bounds and OpenCL payload");
        const double begin=now();ecm_stage2::ModulusContext m;std::string error;if(!m.configure(n,16,carrier,error))throw std::runtime_error(error);
        Big saved;if(mpz_set_str(saved.v,x,16)||mpz_sgn(saved.v)<0||mpz_cmp(saved.v,m.target())>=0)throw std::runtime_error("invalid Stage1 X");
        auto ds=cln::devices();if(index<0||size_t(index)>=ds.size()||!ds[index].supported)throw std::runtime_error("unsupported OpenCL device index");const auto s=choose(d,b2,unsigned(m.S),budget(ds[index]));
        const auto requested_window=setting("ECM_OPENCL_POINT_WINDOW",0);
        if(requested_window>65536)throw std::runtime_error("ECM_OPENCL_POINT_WINDOW must be 0..65536");
        EcmStage2Metrics result;result.d=s.d;result.p=s.p;result.giant_points=s.giants;result.giant_last=s.giants;result.phases.seconds[0]=now()-begin;
        stage2_log::phase("Initialize OpenCL arithmetic and validate");
        cln::Options options;options.device=index;options.budget_bytes=budget(ds[index]);options.resource_root=resources();
        const auto cache=setting("ECM_OPENCL_CACHE_PROGRAMS",1);if(cache>1)throw std::runtime_error("ECM_OPENCL_CACHE_PROGRAMS must be 0 or 1");options.cache_programs=cache!=0;cln::Runtime runtime(options);
        cls::ArithmeticOptions arithmetic;Big modulus;mpz_set(modulus.v,m.N);arithmetic.product.modulus=flatten({modulus},m.W);arithmetic.product.mersenne=carrier!=0;
        cls::Arithmetic field(runtime,arithmetic);std::set<std::string> factors;Big a;if(a24){if(mpz_set_str(a.v,a24,16)||mpz_sgn(a.v)<0||mpz_cmp(a.v,m.target())>=0)throw std::runtime_error("invalid curve a24");}else a=curve_a24(sigma,m.target(),factors);
        PointGenerator generator(runtime,m.target(),saved,a);double sentinel_seconds=0;
        // Real device multiply, independently checked with GMP before any result.
        const auto self=field.download(field.multiply(field.upload(flatten({Big(1),Big(2)},m.W),2),field.upload(flatten({Big(3),Big(1)},m.W),2)));
        auto check=import(self,m.W);for(std::size_t k=0;k<3;++k){Big v(k==0?3:k==1?7:2);mpz_mod(v.v,v.v,m.N);if(mpz_cmp(v.v,check[k].v))throw std::runtime_error("OpenCL polynomial startup selftest");}result.selftest_cases=1;result.checked+=self.size();
        result.phases.seconds[1]=now()-begin-result.phases.seconds[0];
        std::cout<<"stage2_opencl: device="<<index<<" name="<<ds[index].name<<" target_bits="<<m.target_bits()<<" arithmetic_bits="<<m.S<<" carrier="<<carrier<<" point_backend="<<(generator.device?"opencl":"cpu_gmp")<<" point_chunk="<<generator.chunk<<" point_group="<<generator.group<<" giant_base="<<generator.use_giant_base<<" normalization=cpu_batch256 gcd=cpu_gmp payload_budget_bytes="<<options.budget_bytes<<" free_memory_known=0\n";
        print_shape(s,b1,b2,unsigned(m.S));
        auto finish=[&](){field.finish();runtime.finish();for(const auto&f:factors){Big v;if(mpz_set_str(v.v,f.c_str(),10))throw std::runtime_error("factor encoding");if(mpz_cmp_ui(v.v,1)<=0||mpz_cmp(v.v,m.target())>=0||!mpz_divisible_p(m.target(),v.v))throw std::runtime_error("factor validation");}
            const double tail=now()-begin-result.phases.total();result.phases.seconds[9]=std::max(0.,tail);result.phases.complete=true;result.total_seconds=result.phases.total();result.init_seconds=0;for(unsigned k=0;k<4;++k)result.init_seconds+=result.phases.seconds[k];result.main_seconds=result.total_seconds-result.init_seconds;result.clean=true;result.fold_resident=true;result.frontier_resident=true;
            const auto mem=runtime.memory();stage2_log::print(stage2_log::curve,"stage2_full_wall: curve=1 shape=%.6f init=%.6f main=%.6f total=%.6f backend=opencl-ntt (planning/JIT/selftest included; Stage1 excluded)\n",result.phases.seconds[0],result.init_seconds,result.main_seconds,result.total_seconds);
            stage2_log::print(stage2_log::phases,"opencl_payload: live_bytes=%llu peak_bytes=%llu budget_bytes=%llu scope=owned_buffers\n",mem.live_bytes,mem.peak_bytes,mem.budget_bytes);
            stage2_log::print(stage2_log::phases,"opencl_points: points=%llu launches=%llu coordinates=%.6f kernel=%.6f sample_checks=%.6f normalize=%.6f seed=%.6f\n",generator.points,generator.launches,generator.coordinate_seconds,generator.kernel_seconds,generator.check_seconds,generator.normalize_seconds,generator.seed_seconds);
            stage2_log::print(stage2_log::phases,"opencl_point_io: h2d_bytes=%llu d2h_bytes=%llu (kernel nested in coordinates; point timings nested in baby/giant)\n",generator.upload_bytes,generator.download_bytes);
            stage2_log::print(stage2_log::phases,"opencl_point_command: max_kernel_seconds=%.6f chunk=%llu group=%llu\n",generator.kernel_max_seconds,Word(generator.chunk),Word(generator.group));
            stage2_log::print(stage2_log::phases,"opencl_point_plan: enabled=%d compute_units=%u seed_chunk=%llu ceiling=%llu samples=%llu promotions=%llu max_points=%llu budget_clamps=%llu target_seconds=%.3f seconds_per_bit=%.6f wide_ramp=%d ramp_stopped=%d growth_limit=%.3f ramp_accepted=%llu ramp_next=%llu seed_efficiency_limit=%.3f\n",int(generator.plan.enabled),runtime.device().compute_units,Word(generator.chunk),Word(generator.plan.ceiling),Word(generator.plan.samples),generator.promoted_launches,generator.max_command_points,generator.budget_clamps,generator.plan.wide_ramp?0.:cls::PointPlan::target_seconds,generator.plan.seconds_per_bit,int(generator.plan.wide_ramp),int(generator.plan.ramp_stopped),cls::PointPlan::growth_limit,Word(generator.plan.accepted),Word(generator.plan.proposed),cls::PointPlan::seed_efficiency_limit);
            stage2_log::print(stage2_log::phases,"opencl_giant_chain: requested=%d enabled=%d launches=%llu points=%llu seed_pairs=%llu recovery_commands=%llu guard_and_recovery_seconds=%.6f kernel=%.6f max_kernel_seconds=%.6f giant_kernel=%.6f\n",int(generator.chain_requested),int(generator.chain_enabled),generator.chain_launches,generator.chain_points,generator.chain_seeds,generator.chain_recoveries,generator.chain_recovery_seconds,generator.chain_kernel_seconds,generator.chain_max_seconds,generator.giant_kernel_seconds);
            stage2_log::print(stage2_log::phases,"opencl_split: baby=%.6f ftree_inverse=%.6f giant=%.6f gtree_fold=%.6f\n",result.phases.seconds[2],result.phases.seconds[3],result.giant_seconds,result.gtrees_seconds);
            stage2_log::print(stage2_log::phases,"opencl_tail: leaf_oracle=%.6f descent=%.6f accum=%.6f\n",sentinel_seconds,result.descent_seconds,result.accum_seconds);
            const auto programs=runtime.programs();stage2_log::print(stage2_log::phases,"opencl_programs: builds=%llu hits=%llu build_seconds=%.6f cache=%d scope=context\n",programs.builds,programs.hits,programs.build_seconds,int(options.cache_programs));
            stage2_log::print(stage2_log::phases,"opencl_checks: startup_cases=%llu checked_words=%llu bad=0 point_samples=GMP leaf_sentinels=GMP factors=divisibility\n",result.selftest_cases,result.checked);
            if(counts_enabled){std::cout<<"ntt_lengths:";for(const auto&entry:field.stats().ntt_transforms)std::cout<<' '<<entry.first<<':'<<entry.second;std::cout<<" backend=opencl scope=transforms\n";}
            if(metrics)*metrics=result;const auto json=result_json(factors);report(json.c_str(),ctx);return 0;};
        if(!factors.empty())return finish();
        stage2_log::phase("Generate and normalize baby points");double t=now();std::vector<Word> indices;indices.reserve(s.p);for(Word j=1;j<=s.d/2;++j)if(std::gcd(j,s.d)==1)indices.push_back(j);if(indices.size()!=s.p)throw std::runtime_error("D/phi baby count");auto babies=generator.generate(indices,factors,result.checked);result.phases.seconds[2]=now()-t;if(!factors.empty())return finish();
        stage2_log::phase("Build baby-point F tree and polynomial inverse");t=now();cls::Resultant tree(field,field.upload(flatten(babies,m.W),babies.size()));result.phases.seconds[3]=now()-t;
        std::vector<std::size_t> sentinels;for(auto k:std::set<std::size_t>{0,std::size_t(s.p/2),std::size_t(s.p-1)})sentinels.push_back(k);Values expected(sentinels.size(),Big(1));
        // The point window may span several G batches. Keep G boundaries
        // unchanged and admit the host coordinate staging independently.
        const auto window_cap=(setting("NTT_S4_BATCH_MB",64)<<20)/(6*m.W*8);
        auto window_request=overridden("ECM_OPENCL_POINT_WINDOW")?requested_window:generator.plan.window(s.chunk,window_cap);
        if(!overridden("ECM_OPENCL_POINT_WINDOW")&&generator.chain_enabled)window_request=std::max(window_request,std::min(window_cap,generator.chain_capacity()));
        const auto admitted_window=std::min(window_request,window_cap);
        const auto point_window=std::max(s.chunk,(admitted_window/s.chunk)*s.chunk);
        stage2_log::print(stage2_log::phases,"opencl_point_window: requested=%llu effective=%llu cap=%llu G_batch=%llu source=%s\n",requested_window,point_window,window_cap,s.chunk,overridden("ECM_OPENCL_POINT_WINDOW")?"explicit":generator.chain_enabled?"chain":generator.plan.enabled?"adaptive":"default");
        stage2_log::phase("Giant points, G trees and fold");const double loop=now();Word batches=0;
        for(Word first=1;first<=s.giants;){const auto amount=std::min(point_window,s.giants-first+1);indices.resize(amount);for(Word k=0;k<amount;++k)indices[k]=(first+k)*s.d;
            t=now();auto window=generator.generate(indices,factors,result.checked,s.d,std::size_t(s.chunk));result.giant_seconds+=now()-t;
            if(!factors.empty()){result.phases.seconds[6]=now()-loop;return finish();}
            for(Word offset=0;offset<amount;){const auto count=std::min(s.chunk,amount-offset);
                Values giants(window.begin()+std::size_t(offset),window.begin()+std::size_t(offset+count));
                t=now();for(std::size_t k=0;k<sentinels.size();++k)for(const auto&g:giants)expected[k]=mul(expected[k],sub(babies[sentinels[k]],g,m.target()),m.target());
                sentinel_seconds+=now()-t;t=now();tree.append(field.upload(flatten(giants,m.W),giants.size()));result.gtrees_seconds+=now()-t;++batches;first+=count;offset+=count;
                if(stage2_log::enabled(stage2_log::batches))std::cout<<"batched_progress: batch="<<batches<<'/'<<(s.giants+s.chunk-1)/s.chunk<<" points="<<first-1<<'/'<<s.giants<<" backend=opencl\n";
            }
        }result.phases.seconds[6]=now()-loop;
        stage2_log::phase("Normalize fold and descend F tree");t=now();const auto leaves=import(field.download(tree.values()),m.W);result.descent_seconds=now()-t;result.phases.seconds[7]=result.descent_seconds;
        for(std::size_t k=0;k<sentinels.size();++k){Big v;mpz_mod(v.v,leaves[sentinels[k]].v,m.target());if(mpz_cmp(v.v,expected[k].v))throw std::runtime_error("OpenCL full-range leaf sentinel mismatch");result.checked+=m.W;}
        stage2_log::phase("Accumulate leaf products and GCD");t=now();Big product,g;
        for(std::size_t lo=0;lo<leaves.size();lo+=64){const auto hi=std::min(lo+64,leaves.size());mpz_set_ui(product.v,1);for(std::size_t k=lo;k<hi;++k){mpz_mul(product.v,product.v,leaves[k].v);mpz_mod(product.v,product.v,m.target());}mpz_gcd(g.v,product.v,m.target());factor(factors,g.v,m.target());
            if(mpz_cmp_ui(g.v,1)>0)for(std::size_t k=lo;k<hi;++k){mpz_gcd(g.v,leaves[k].v,m.target());factor(factors,g.v,m.target());}}
        result.accum_seconds=now()-t;result.phases.seconds[8]=result.accum_seconds;return finish();
    }catch(const std::exception&e){std::cerr<<"OpenCL Stage2 failed: "<<e.what()<<'\n';return 2;}
}
