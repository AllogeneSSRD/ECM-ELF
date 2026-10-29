// Native OpenCL engine behind the existing save/queue driver ABI. The legacy
// ecm_cuda_stage2_* symbol names are a link seam, not a CUDA dependency.
#define NOMINMAX
#include <windows.h>
#include "engine.h"
#include "tree.h"
#include "points.h"
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

// The device computes points; a batch of 256 denominators needs one GMP inverse.
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
    PointGenerator(cln::Runtime&r,mpz_srcptr modulus,const Big&x,const Big&a):runtime(r),n(modulus),a24(a),base{x,Big(1)},w((mpz_sizeinbase(n,2)+63)/64){
        const auto&v=r.device().vendor;const bool amd=v.find("Advanced Micro Devices")!=v.npos||v=="AMD";
        // The existing wide-AMD point qualification guard remains intact.
        // CPU points still feed the same native OpenCL polynomial pipeline.
        if(!amd||w<=8)device=std::make_unique<cls::Points>(runtime,flatten({BigValue()},w));
    }
    Big BigValue(){Big value;mpz_set(value.v,n);return value;}
    Values generate(const std::vector<Word>&scalars,std::set<std::string>&f,Word&checked){
        Values x,z;if(device){
            // Bound each scalar-ladder submission on display GPUs. Wide scalar
            // kernels are slow; a whole production giant batch can otherwise
            // become a single multi-second command. This is separate from the
            // polynomial batch size and is not a calibrated watchdog guarantee.
            const std::size_t chunk=w<=8?1024:w<=32?256:w<=64?128:w<=128?64:16;
            for(std::size_t first=0;first<scalars.size();first+=chunk){const auto last=std::min(first+chunk,scalars.size());
                std::vector<Word> slice(scalars.begin()+first,scalars.begin()+last);
                auto points=device->ladder(flatten({base.x},w),flatten({base.z},w),flatten({a24},w),slice);
                auto xx=import(device->download(points.x),w),zz=import(device->download(points.z),w);
                x.insert(x.end(),xx.begin(),xx.end());z.insert(z.end(),zz.begin(),zz.end());
            }
            for(auto k:std::set<std::size_t>{0,scalars.size()/2,scalars.size()-1}){const auto oracle=ladder(base,scalars[k],a24,n);auto cross=sub(mul(x[k],oracle.z,n),mul(z[k],oracle.x,n),n);if(mpz_sgn(cross.v))throw std::runtime_error("OpenCL ladder GMP mismatch");checked+=w*2;}
        }else{for(auto scalar:scalars){auto p=ladder(base,scalar,a24,n);x.push_back(p.x);z.push_back(p.z);}}
        return normalize(x,z,n,f);
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
        EcmStage2Metrics result;result.d=s.d;result.p=s.p;result.giant_points=s.giants;result.giant_last=s.giants;result.phases.seconds[0]=now()-begin;
        stage2_log::phase("Initialize OpenCL arithmetic and validate");
        cln::Options options;options.device=index;options.budget_bytes=budget(ds[index]);options.resource_root=resources();cln::Runtime runtime(options);
        cls::ArithmeticOptions arithmetic;Big modulus;mpz_set(modulus.v,m.N);arithmetic.product.modulus=flatten({modulus},m.W);arithmetic.product.mersenne=carrier!=0;
        cls::Arithmetic field(runtime,arithmetic);std::set<std::string> factors;Big a;if(a24){if(mpz_set_str(a.v,a24,16)||mpz_sgn(a.v)<0||mpz_cmp(a.v,m.target())>=0)throw std::runtime_error("invalid curve a24");}else a=curve_a24(sigma,m.target(),factors);
        PointGenerator generator(runtime,m.target(),saved,a);
        // Real device multiply, independently checked with GMP before any result.
        const auto self=field.download(field.multiply(field.upload(flatten({Big(1),Big(2)},m.W),2),field.upload(flatten({Big(3),Big(1)},m.W),2)));
        auto check=import(self,m.W);for(std::size_t k=0;k<3;++k){Big v(k==0?3:k==1?7:2);mpz_mod(v.v,v.v,m.N);if(mpz_cmp(v.v,check[k].v))throw std::runtime_error("OpenCL polynomial startup selftest");}result.selftest_cases=1;result.checked+=self.size();
        result.phases.seconds[1]=now()-begin-result.phases.seconds[0];
        std::cout<<"stage2_opencl: device="<<index<<" name="<<ds[index].name<<" target_bits="<<m.target_bits()<<" arithmetic_bits="<<m.S<<" carrier="<<carrier<<" point_backend="<<(generator.device?"opencl":"cpu_gmp_amd_wide_guard")<<" normalization=cpu_batch256 gcd=cpu_gmp payload_budget_bytes="<<options.budget_bytes<<" free_memory_known=0\n";
        print_shape(s,b1,b2,unsigned(m.S));
        auto finish=[&](){field.finish();runtime.finish();for(const auto&f:factors){Big v;if(mpz_set_str(v.v,f.c_str(),10))throw std::runtime_error("factor encoding");if(mpz_cmp_ui(v.v,1)<=0||mpz_cmp(v.v,m.target())>=0||!mpz_divisible_p(m.target(),v.v))throw std::runtime_error("factor validation");}
            const double tail=now()-begin-result.phases.total();result.phases.seconds[9]=std::max(0.,tail);result.phases.complete=true;result.total_seconds=result.phases.total();result.init_seconds=0;for(unsigned k=0;k<4;++k)result.init_seconds+=result.phases.seconds[k];result.main_seconds=result.total_seconds-result.init_seconds;result.clean=true;result.fold_resident=true;result.frontier_resident=true;
            const auto mem=runtime.memory();stage2_log::print(stage2_log::curve,"stage2_full_wall: curve=1 shape=%.6f init=%.6f main=%.6f total=%.6f backend=opencl-ntt (planning/JIT/selftest included; Stage1 excluded)\n",result.phases.seconds[0],result.init_seconds,result.main_seconds,result.total_seconds);
            stage2_log::print(stage2_log::phases,"opencl_payload: live_bytes=%llu peak_bytes=%llu budget_bytes=%llu scope=owned_buffers\n",mem.live_bytes,mem.peak_bytes,mem.budget_bytes);
            stage2_log::print(stage2_log::phases,"opencl_checks: startup_cases=%llu checked_words=%llu bad=0 point_samples=GMP leaf_sentinels=GMP factors=divisibility\n",result.selftest_cases,result.checked);
            if(counts_enabled){std::cout<<"ntt_lengths:";for(const auto&entry:field.stats().ntt_transforms)std::cout<<' '<<entry.first<<':'<<entry.second;std::cout<<" backend=opencl scope=transforms\n";}
            if(metrics)*metrics=result;const auto json=result_json(factors);report(json.c_str(),ctx);return 0;};
        if(!factors.empty())return finish();
        stage2_log::phase("Generate and normalize baby points");double t=now();std::vector<Word> indices;indices.reserve(s.p);for(Word j=1;j<=s.d/2;++j)if(std::gcd(j,s.d)==1)indices.push_back(j);if(indices.size()!=s.p)throw std::runtime_error("D/phi baby count");auto babies=generator.generate(indices,factors,result.checked);result.phases.seconds[2]=now()-t;if(!factors.empty())return finish();
        stage2_log::phase("Build baby-point F tree and polynomial inverse");t=now();cls::Resultant tree(field,field.upload(flatten(babies,m.W),babies.size()));result.phases.seconds[3]=now()-t;
        std::vector<std::size_t> sentinels;for(auto k:std::set<std::size_t>{0,std::size_t(s.p/2),std::size_t(s.p-1)})sentinels.push_back(k);Values expected(sentinels.size(),Big(1));
        stage2_log::phase("Giant points, G trees and fold");const double loop=now();Word batches=0;
        for(Word first=1;first<=s.giants;){const auto count=std::min(s.chunk,s.giants-first+1);indices.resize(count);for(Word k=0;k<count;++k)indices[k]=(first+k)*s.d;t=now();auto giants=generator.generate(indices,factors,result.checked);result.giant_seconds+=now()-t;if(!factors.empty()){result.phases.seconds[6]=now()-loop;return finish();}
            t=now();for(std::size_t k=0;k<sentinels.size();++k)for(const auto&g:giants)expected[k]=mul(expected[k],sub(babies[sentinels[k]],g,m.target()),m.target());
            tree.append(field.upload(flatten(giants,m.W),giants.size()));result.gtrees_seconds+=now()-t;++batches;first+=count;
            if(stage2_log::enabled(stage2_log::batches))std::cout<<"batched_progress: batch="<<batches<<'/'<<(s.giants+s.chunk-1)/s.chunk<<" points="<<first-1<<'/'<<s.giants<<" backend=opencl\n";
        }result.phases.seconds[6]=now()-loop;
        stage2_log::phase("Normalize fold and descend F tree");t=now();const auto leaves=import(field.download(tree.values()),m.W);result.descent_seconds=now()-t;result.phases.seconds[7]=result.descent_seconds;
        for(std::size_t k=0;k<sentinels.size();++k){Big v;mpz_mod(v.v,leaves[sentinels[k]].v,m.target());if(mpz_cmp(v.v,expected[k].v))throw std::runtime_error("OpenCL full-range leaf sentinel mismatch");result.checked+=m.W;}
        stage2_log::phase("Accumulate leaf products and GCD");t=now();Big product,g;
        for(std::size_t lo=0;lo<leaves.size();lo+=64){const auto hi=std::min(lo+64,leaves.size());mpz_set_ui(product.v,1);for(std::size_t k=lo;k<hi;++k){mpz_mul(product.v,product.v,leaves[k].v);mpz_mod(product.v,product.v,m.target());}mpz_gcd(g.v,product.v,m.target());factor(factors,g.v,m.target());
            if(mpz_cmp_ui(g.v,1)>0)for(std::size_t k=lo;k<hi;++k){mpz_gcd(g.v,leaves[k].v,m.target());factor(factors,g.v,m.target());}}
        result.accum_seconds=now()-t;result.phases.seconds[8]=result.accum_seconds;return finish();
    }catch(const std::exception&e){std::cerr<<"OpenCL Stage2 failed: "<<e.what()<<'\n';return 2;}
}
