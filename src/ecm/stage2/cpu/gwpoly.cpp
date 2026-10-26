#include "gwpoly.h"
#include "../../curve/edwards.h"
extern "C" {
#include <cpuid.h>
#include <gwnum.h>
#include <polymult.h>
}
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace ecm_gwpoly {
namespace {
using ecm_edwards::Integer;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Hit {};
struct Context;
struct Array {
    gwhandle *gw=nullptr;gwnum *v=nullptr;size_t size=0;bool owned=true;
    Array()=default;
    Array(gwhandle &h,size_t count):gw(&h),size(count){
        if(count && !(v=gwalloc_array(&h,count)))throw std::bad_alloc();
    }
    Array(gwhandle &h,gwnum *data,size_t count):gw(&h),v(data),size(count),owned(false){}
    ~Array(){if(v && owned)gwfree_array(gw,v);}
    Array(const Array &)=delete;Array &operator=(const Array &)=delete;
    Array(Array &&a) noexcept:gw(a.gw),v(a.v),size(a.size),owned(a.owned){a.v=nullptr;a.size=0;}
    Array &operator=(Array &&a) noexcept {if(v && owned)gwfree_array(gw,v);gw=a.gw;v=a.v;size=a.size;owned=a.owned;a.v=nullptr;a.size=0;return *this;}
    gwnum operator[](size_t i)const{return v[i];}
};
using View=std::vector<gwnum>;
View view(const Array &a){return View(a.v,a.v+a.size);}
View reverse(const Array &a){View v=view(a);std::reverse(v.begin(),v.end());return v;}
struct Context {
    gwhandle gw{};pmhandle pm{};bool ready=false,pmready=false;
    const Options &o;Stats &stats;mpz_srcptr n;mpz_ptr factor;
    std::vector<uint32_t> binary;
    using PlanKey=std::tuple<size_t,size_t,size_t,size_t,int,unsigned>;
    std::map<PlanKey,void *> plans;
    Context(const mpz_t modulus,mpz_t f,const Options &opt,Stats &s,size_t max_poly):o(opt),stats(s),n(modulus),factor(f){
        try {
        guessCpuType();
        if(o.exponent){Integer m;mpz_set_ui(m.v,1);mpz_mul_2exp(m.v,m.v,o.exponent);mpz_sub_ui(m.v,m.v,1);
            if(!mpz_divisible_p(m.v,n))throw std::runtime_error("GW carrier does not contain actual N");}
        for(unsigned extra=0;extra<16;++extra){
            gwinit(&gw);ready=true;
            gwset_using_polymult(&gw);gwset_num_threads(&gw,1);gwclear_use_benchmarks(&gw);
            // Select against unreserved capacity. Reserving the polynomial
            // margin subtracts it from EXTRA_BITS; testing it again afterwards
            // would count the same requirement twice.
            gwset_polymult_safety_margin(&gw,0.0f);
            gwset_larger_fftlen_count(&gw,extra);gwset_will_error_check(&gw);
            int rc;
            if(o.exponent)rc=gwsetup(&gw,1.0,2,static_cast<unsigned long>(o.exponent),-1);
            else{
                size_t count=0;binary.resize((mpz_sizeinbase(n,2)+31)/32);
                mpz_export(binary.data(),&count,-1,4,0,0,n);
                rc=gwsetup_general_mod(&gw,binary.data(),static_cast<uint32_t>(count));
            }
            if(rc)throw std::runtime_error("GWNUM setup failed: "+std::to_string(rc));
            if(gw_passes_safety_margin(&gw,polymult_safety_margin(max_poly+1,max_poly+1)+0.25f))break;
            gwdone(&gw);ready=false;
        }
        if(!ready)throw std::runtime_error("No safe CPU polynomial FFT length");
        gwerror_checking(&gw,1);
        gwset_polymult_safety_margin(&gw,polymult_safety_margin(max_poly+1,max_poly+1)+0.25f);
        binary.resize((static_cast<size_t>(gw.bit_length)+31)/32+8);
        gwuser_init_FFT1(&gw);
        polymult_init(&pm,&gw);pmready=true;
        polymult_set_cpu_flags(&pm,CPU_FLAGS);polymult_set_max_num_threads(&pm,o.threads);
        polymult_default_tuning(&pm,std::max(256,CPU_L2_CACHE_SIZE),std::max(6144,CPU_L3_CACHE_SIZE));
        // polymult launches its own internal tasks. A naked launch_helpers call
        // instead dispatches the caller's user callback, which we do not set.
        stats.fft_length=gwfftlen(&gw);
        }catch(...){if(pmready)polymult_done(&pm);if(ready)gwdone(&gw);throw;}
    }
    ~Context(){if(pmready)polymult_done(&pm);for(auto &plan:plans)std::free(plan.second);if(ready)gwdone(&gw);}
    Context(Context &parent,mpz_t f,Stats &s):o(parent.o),stats(s),n(parent.n),factor(f){
        if(gwclone(&gw,&parent.gw))throw std::runtime_error("Cannot clone GWNUM arithmetic context");
        ready=true;binary.resize(parent.binary.size());
        stats.fft_length=parent.stats.fft_length;
        polymult_init(&pm,&gw);pmready=true;
        polymult_set_cpu_flags(&pm,parent.pm.cpu_flags);polymult_set_max_num_threads(&pm,1);
        polymult_default_tuning(&pm,std::max(256,CPU_L2_CACHE_SIZE),std::max(6144,CPU_L3_CACHE_SIZE));
    }
    void poll(){if(o.stop && o.stop())throw std::runtime_error("CPU Stage2 interrupted");}
    void check(const char *operation="point arithmetic"){
        poll();stats.max_roundoff=std::max(stats.max_roundoff,gw_get_maxerr(&gw));
        const auto error=gw_test_for_error(&gw);
        if(error || stats.max_roundoff>0.40)throw std::runtime_error(std::string("GWNUM arithmetic/roundoff error in ")+operation+
            "; flags="+std::to_string(error)+" maxerr="+std::to_string(stats.max_roundoff)+
            " fft="+std::to_string(stats.fft_length)+"; no NF result");
    }
    void put(const mpz_t z,gwnum out){
        size_t count=0;mpz_export(binary.data(),&count,-1,4,0,0,z);
        if(!count)u64togw(&gw,0,out);else binarytogw(&gw,binary.data(),static_cast<uint32_t>(count),out);
    }
    void get(mpz_t z,gwnum in){
        Array tmp(gw,1);gwunfft(&gw,in,tmp[0]);
        const auto len=gwtobinary(&gw,tmp[0],binary.data(),static_cast<uint32_t>(binary.size()));
        if(len<0)throw std::runtime_error("GWNUM conversion error");
        mpz_import(z,static_cast<size_t>(len),-1,4,0,0,binary.data());mpz_mod(z,z,n);
    }
    void mul(gwnum a,gwnum b,gwnum out){gwmul3(&gw,a,b,out,GWMUL_PRESERVE_S1|GWMUL_PRESERVE_S2);}
    Array constant(uint64_t x){Array a(gw,1);u64togw(&gw,x,a[0]);return a;}
    Array copy(const Array &a){Array b(gw,a.size);for(size_t i=0;i<a.size;++i)gwcopy(&gw,a[i],b[i]);return b;}
    Array convolution(View a,View b,size_t first,size_t count){
        if(a.empty() || b.empty() || !count)throw std::runtime_error("Empty polynomial multiplication");
        Array out(gw,count);++stats.polymults;
        polymult_set_num_threads(&pm,a.size()+b.size()<64?1:pm.max_num_threads);
        const int base_options=POLYMULT_MULMID|POLYMULT_NEXTFFT;
        auto &plan=plans[{a.size(),b.size(),count,first,base_options,pm.num_threads}];
        int options=base_options|POLYMULT_SAVE_PLAN;
        if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult2(&pm,a.data(),a.size(),b.data(),b.size(),out.v,count,nullptr,0,first,
                  options);
        plan=pm.plan;
        check("polynomial convolution");return out;
    }
    void monic_product(const Array &a,const Array &b,Array &out,bool defer=false){
        const size_t degree=a.size+b.size-2;++stats.polymults;
        polymult_set_num_threads(&pm,degree<64?1:pm.max_num_threads);
        const int base_options=POLYMULT_INVEC1_MONIC|POLYMULT_INVEC2_MONIC|(defer?POLYMULT_NO_UNFFT:POLYMULT_NEXTFFT);
        auto &plan=plans[{a.size-1,b.size-1,degree,0,base_options,static_cast<unsigned>(pm.num_threads)}];
        int options=base_options|POLYMULT_SAVE_PLAN;
        if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult(&pm,a.v,a.size-1,b.v,b.size-1,out.v,degree,
                 options);
        plan=pm.plan;
        u64togw(&gw,1,out[degree]);check("monic polynomial product");
    }
    Array convolution(const Array &a,const Array &b){return convolution(view(a),view(b),0,a.size+b.size-1);}
    Array scaled_child(const Array &parent,const Array &sibling,size_t count){
        // In reversed scaled coordinates, only the high child interval of a
        // cyclic product of length parent.size is needed. Wrapped coefficients
        // lie below this interval and do not affect the result.
        Array out(gw,count);++stats.polymults;
        polymult_set_num_threads(&pm,parent.size+sibling.size<64?1:pm.max_num_threads);
        const int base=POLYMULT_INVEC2_MONIC|POLYMULT_CIRCULAR|POLYMULT_MULHI|POLYMULT_NEXTFFT;
        auto &plan=plans[{parent.size,sibling.size-1,count,parent.size,base,static_cast<unsigned>(pm.num_threads)}];
        int options=base|POLYMULT_SAVE_PLAN;if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult2(&pm,parent.v,parent.size,sibling.v,sibling.size-1,out.v,count,nullptr,parent.size,0,options);
        plan=pm.plan;check("scaled polynomial descent");return out;
    }
    Array inverse(const Array &f,size_t count){
        // f[0]=1. Newton g <- g*(2-f*g) mod X^k, over Z/NZ.
        auto g=constant(1);Array two=constant(2),zero=constant(0);
        gwfft(&gw,two[0],two[0]);gwfft(&gw,zero[0],zero[0]);
        while(g.size<count){
            const size_t k=std::min(count,2*g.size);auto fv=view(f);fv.resize(std::min(k,f.size));
            auto e=convolution(fv,view(g),0,k);
            for(size_t i=0;i<k;++i){
                if(!i)gwsub3o(&gw,two[0],e[i],e[i],GWADD_FORCE_NORMALIZE);
                else gwsub3o(&gw,zero[0],e[i],e[i],GWADD_FORCE_NORMALIZE);
            }
            g=convolution(view(g),view(e),0,k);
        }
        return g;
    }
    Array remainder(const Array &t,const Array &f,const Array &inv){
        const size_t degree=f.size-1;
        if(t.size<=degree)return copy(t);
        const size_t k=t.size-degree;auto rt=reverse(t);rt.resize(k);auto rv=view(inv);rv.resize(k);
        auto qr=convolution(rt,rv,0,k);auto q=reverse(qr);
        auto low=convolution(q,view(f),0,degree);
        for(size_t i=0;i<degree;++i){gwfft(&gw,t[i],t[i]);gwsub3o(&gw,t[i],low[i],low[i],GWADD_FORCE_NORMALIZE);}
        return low;
    }
    void divisor(gwnum z){
        Integer v;get(v.v,z);mpz_gcd(factor,v.v,n);
        if(mpz_cmp_ui(factor,1)>0 && mpz_cmp(factor,n)<0)throw Hit{};
    }
};
struct Tree {
    size_t lo,hi;Array polynomial;
    std::unique_ptr<Tree> left,right;
    std::vector<Array> rows;
    explicit Tree(size_t a,size_t b):lo(a),hi(b){}
};
std::unique_ptr<Tree> tree(Context &,const Array &,size_t,size_t,bool,Array *a=nullptr,Array *b=nullptr);
void descend(Context &c,const Tree &t,const Array &s,Array &values){
    if(!t.left){gwcopy(&c.gw,s[0],values[t.lo]);return;}
    const auto a=t.left->hi-t.left->lo,b=t.right->hi-t.right->lo;
    auto sl=c.scaled_child(s,t.right->polynomial,a);
    auto sr=c.scaled_child(s,t.left->polynomial,b);
    descend(c,*t.left,sl,values);descend(c,*t.right,sr,values);
}
struct Point {gwnum x,z;};
struct Points {
    Context &c;Array work;
    explicit Points(Context &context):c(context),work(c.gw,12){}
    void copy(Point a,Point b){gwcopy(&c.gw,a.x,b.x);gwcopy(&c.gw,a.z,b.z);}
    void dbl(Point p,Point out,gwnum a24){
        auto &h=c.gw;auto &t=work;
        gwaddsub4o(&h,p.x,p.z,t[0],t[1],GWADD_MUL_INPUT);
        c.mul(t[0],t[0],t[2]);c.mul(t[1],t[1],t[3]);
        gwsub3o(&h,t[2],t[3],t[4],GWADD_MUL_INPUT);
        c.mul(t[4],a24,t[5]);gwadd3o(&h,t[3],t[5],t[6],GWADD_MUL_INPUT);
        c.mul(t[2],t[3],out.x);c.mul(t[4],t[6],out.z);
    }
    void add(Point p,Point q,Point difference,Point out){
        auto &h=c.gw;auto &t=work;
        gwaddsub4o(&h,p.x,p.z,t[0],t[1],GWADD_MUL_INPUT);
        gwaddsub4o(&h,q.x,q.z,t[2],t[3],GWADD_MUL_INPUT);
        c.mul(t[0],t[3],t[4]);c.mul(t[1],t[2],t[5]);
        gwaddsub4o(&h,t[4],t[5],t[6],t[7],GWADD_SQUARE_INPUT);
        c.mul(t[6],t[6],t[8]);c.mul(t[7],t[7],t[9]);
        c.mul(t[8],difference.z,t[10]);c.mul(t[9],difference.x,t[11]);
        gwcopy(&h,t[10],out.x);gwcopy(&h,t[11],out.z);
    }
    void ladder(Point p,uint64_t k,Point out,gwnum a24){
        if(!k){u64togw(&c.gw,1,out.x);u64togw(&c.gw,0,out.z);return;}
        Array storage(c.gw,6);Point a{storage[0],storage[1]},b{storage[2],storage[3]},t{storage[4],storage[5]};
        copy(p,a);dbl(p,b,a24);
        uint64_t bit=1;while(bit<=k/2)bit<<=1;
        for(bit>>=1;bit;bit>>=1){
            if(k&bit){add(a,b,p,t);dbl(b,b,a24);copy(t,a);}
            else {add(a,b,p,t);dbl(a,a,a24);copy(t,b);}
        }
        copy(a,out);
    }
    void normalize(gwnum *x,gwnum *z,size_t count){
        Array prefix(c.gw,count);gwcopy(&c.gw,z[0],prefix[0]);
        for(size_t i=1;i<count;++i)c.mul(prefix[i-1],z[i],prefix[i]);
        Integer product,inv;c.get(product.v,prefix[count-1]);
        if(!mpz_invert(inv.v,product.v,c.n)){
            for(size_t i=0;i<count;++i)c.divisor(z[i]);
            throw std::runtime_error("Saturated point denominator; no NF result");
        }
        Array scratch(c.gw,3);c.put(inv.v,scratch[0]);
        for(size_t i=count;i-->0;){
            if(i)c.mul(scratch[0],prefix[i-1],scratch[1]);else gwcopy(&c.gw,scratch[0],scratch[1]);
            c.mul(x[i],scratch[1],scratch[2]);gwcopy(&c.gw,scratch[2],x[i]);
            if(i){c.mul(scratch[0],z[i],scratch[2]);gwcopy(&c.gw,scratch[2],scratch[0]);}
        }
        c.check("batch point normalization");
    }
};
template<class Function> void parallel_points(Context &parent,size_t count,Function function,size_t min_chunk=256){
    struct Worker {
        Integer factor;Stats stats;Context context;std::exception_ptr error;
        explicit Worker(Context &p):context(p,factor.v,stats){}
    };
    const size_t threads=std::min<size_t>(parent.o.threads,std::max<size_t>(1,count/min_chunk));
    std::vector<std::unique_ptr<Worker>> workers;
    for(size_t i=0;i<threads;++i)workers.push_back(std::make_unique<Worker>(parent));
    auto execute=[&](size_t i){try{function(workers[i]->context,count*i/threads,count*(i+1)/threads);}
        catch(...){workers[i]->error=std::current_exception();}};
    std::vector<std::thread> jobs;
    try{for(size_t i=1;i<threads;++i)jobs.emplace_back(execute,i);}
    catch(...){for(auto &job:jobs)job.join();throw;}
    execute(0);for(auto &job:jobs)job.join();
    for(auto &worker:workers){
        gwclone_merge_stats(&parent.gw,&worker->context.gw);
        parent.stats.polymults+=worker->stats.polymults;
        if(worker->error){mpz_set(parent.factor,worker->factor.v);std::rethrow_exception(worker->error);}
    }
    parent.check("parallel worker merge");
}
template<class Function> void tree_jobs(Context &c,size_t count,Function function){
    if(count<128 || c.o.threads==1)function(c,0,count);
    else parallel_points(c,count,function,1);
}
void tree_layout(size_t lo,size_t hi,size_t depth,std::vector<size_t> &sizes){
    if(sizes.size()<=depth)sizes.resize(depth+1);
    sizes[depth]+=hi-lo+1;
    if(hi-lo>1){const auto mid=lo+(hi-lo)/2;tree_layout(lo,mid,depth+1,sizes);tree_layout(mid,hi,depth+1,sizes);}
}
std::unique_ptr<Tree> tree_skeleton(Context &c,size_t lo,size_t hi,size_t depth,std::vector<Array> &rows,
                                    std::vector<size_t> &offsets,std::vector<std::vector<Tree *>> &levels){
    auto t=std::make_unique<Tree>(lo,hi);
    t->polynomial=Array(c.gw,rows[depth].v+offsets[depth],hi-lo+1);offsets[depth]+=hi-lo+1;
    levels[depth].push_back(t.get());
    if(hi-lo>1){const auto mid=lo+(hi-lo)/2;
        t->left=tree_skeleton(c,lo,mid,depth+1,rows,offsets,levels);
        t->right=tree_skeleton(c,mid,hi,depth+1,rows,offsets,levels);}
    return t;
}
std::unique_ptr<Tree> tree(Context &c,const Array &roots,size_t lo,size_t hi,bool retain,Array *workspace_a,Array *workspace_b){
    if(!retain){
        const auto count=hi-lo;Array local_a,local_b;
        if(!workspace_a){local_a=Array(c.gw,2*count);local_b=Array(c.gw,2*count);workspace_a=&local_a;workspace_b=&local_b;}
        if(!workspace_b || workspace_a->size<2*count || workspace_b->size<2*count)throw std::runtime_error("G tree workspace too small");
        Array *source=workspace_a,*destination=workspace_b;
        Array row(c.gw,source->v,2*count);std::vector<size_t> degrees(count,1);
        for(size_t i=0;i<count;++i){gwcopy(&c.gw,roots[lo+i],row[2*i]);gwsmallmul(&c.gw,-1,row[2*i]);u64togw(&c.gw,1,row[2*i+1]);}
        while(degrees.size()>1){
            struct Job {size_t a,da,b,db,out;};std::vector<Job> jobs;std::vector<size_t> next;
            size_t offset=0,out=0;
            for(size_t i=0;i<degrees.size();i+=2){
                const auto a=degrees[i],b=i+1<degrees.size()?degrees[i+1]:0;
                jobs.push_back({offset,a,offset+a+1,b,out});next.push_back(a+b);
                offset+=a+1+(b?b+1:0);out+=a+b+1;
            }
            Array result(c.gw,destination->v,out);
            tree_jobs(c,jobs.size(),[&](Context &worker,size_t begin,size_t end){for(size_t j=begin;j<end;++j){
                const auto &job=jobs[j];Array dest(worker.gw,result.v+job.out,job.da+job.db+1);
                if(!job.db){for(size_t i=0;i<=job.da;++i)gwcopy(&worker.gw,row[job.a+i],dest[i]);}
                else {Array a(worker.gw,row.v+job.a,job.da+1),b(worker.gw,row.v+job.b,job.db+1);worker.monic_product(a,b,dest,jobs.size()>=128);}
            }});
            if(jobs.size()>=128){
                View coefficients;coefficients.reserve(count);
                for(const auto &job:jobs)for(size_t i=0;i<job.da+job.db;++i)coefficients.push_back(result[job.out+i]);
                polymult_set_num_threads(&c.pm,c.pm.max_num_threads);
                poly_unfft_fft_coefficients(&c.pm,coefficients.data(),coefficients.size());c.check("G tree coefficient normalization");
            }
            row=std::move(result);degrees=std::move(next);std::swap(source,destination);
        }
        auto t=std::make_unique<Tree>(lo,hi);t->polynomial=Array(c.gw,row.v,count+1);
        if(local_a.v){t->rows.push_back(std::move(local_a));t->rows.push_back(std::move(local_b));}
        return t;
    }
    std::vector<size_t> sizes;tree_layout(lo,hi,0,sizes);
    std::vector<Array> rows;for(auto size:sizes)rows.emplace_back(c.gw,size);
    std::vector<size_t> offsets(sizes.size());std::vector<std::vector<Tree *>> levels(sizes.size());
    auto t=tree_skeleton(c,lo,hi,0,rows,offsets,levels);t->rows=std::move(rows);
    for(size_t depth=levels.size();depth-->0;){
        auto &nodes=levels[depth];
        tree_jobs(c,nodes.size(),[&](Context &worker,size_t begin,size_t end){for(size_t i=begin;i<end;++i){
            auto &node=*nodes[i];
            if(node.left)worker.monic_product(node.left->polynomial,node.right->polynomial,node.polynomial,nodes.size()>=128);
            else {gwcopy(&worker.gw,roots[node.lo],node.polynomial[0]);gwsmallmul(&worker.gw,-1,node.polynomial[0]);u64togw(&worker.gw,1,node.polynomial[1]);}
        }});
        if(nodes.size()>=128){
            View coefficients;
            for(auto node:nodes)for(size_t i=0;i+1<node->polynomial.size;++i)coefficients.push_back(node->polynomial[i]);
            polymult_set_num_threads(&c.pm,c.pm.max_num_threads);
            poly_unfft_fft_coefficients(&c.pm,coefficients.data(),coefficients.size());c.check("F tree coefficient normalization");
        }
    }
    return t;
}
uint64_t phi(uint64_t d){uint64_t p=d;for(uint64_t f=2;f<=d/f;++f)if(d%f==0){p=p/f*(f-1);while(d%f==0)d/=f;}if(d>1)p=p/d*(d-1);return p;}
uint64_t choose_d(uint64_t b2,uint64_t forced,double bytes,uint64_t budget){
    // Smooth wheels including repeated odd factors; compare actual phi(D)/2.
    std::vector<uint64_t> choices;
    for(uint64_t wheel:{30ull,210ull,2310ull,30030ull,510510ull})
        for(uint64_t k:{1ull,2ull,3ull,4ull,5ull,6ull,7ull,8ull,9ull,11ull,13ull,15ull,21ull,27ull,35ull,45ull,63ull,81ull})
            if(wheel*k<=20000000)choices.push_back(wheel*k);
    if(forced)choices={forced};
    double best=1e300;uint64_t result=0;
    for(auto d:choices){
        if(d<6 || d%2 || d>20000000)continue;
        const double p=double(phi(d)/2),levels=std::ceil(std::log2(p));
        const double memory=(p*(levels+18)+256)*bytes;
        if(memory>double(budget)*1048576)continue;
        const double cost=6*p*levels+(double(b2)/d+p)*(levels+8);
        if(cost<best){best=cost;result=d;}
    }
    if(!result)throw std::runtime_error("No CPU polynomial plan within memory budget");
    return result;
}
}

const char *version(){return GWNUM_VERSION;}
bool stage2(mpz_t factor,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &o,Stats &stats){
    stats={};const auto begin=Clock::now();
    if(!o.threads || b2<=b1 || b2>max_b2)throw std::runtime_error("Invalid GW polynomial bounds/threads");
    // Probe a real FFT layout before applying the memory budget. A guessed
    // per-coefficient footprint can otherwise reject a valid small-modulus plan.
    if(o.d && (o.d<6 || o.d%2 || o.d>20000000))throw std::runtime_error("Invalid CPU polynomial wheel D");
    uint64_t d=o.d?o.d:30,p=phi(d)/2;
    std::unique_ptr<Context> owner;
    for(unsigned attempt=0;attempt<6;++attempt){
        owner=std::make_unique<Context>(n,factor,o,stats,p);
        const auto candidate=choose_d(b2,o.d,double(gwnum_size(&owner->gw)+64),o.memory_mib);
        const auto candidate_p=phi(candidate)/2;
        const auto required=polymult_safety_margin(candidate_p+1,candidate_p+1)+0.25f;
        if(owner->gw.EXTRA_BITS/2.0f+owner->gw.polymult_safety_margin>required){
            gwset_polymult_safety_margin(&owner->gw,required);
            d=candidate;p=candidate_p;break;
        }
        owner.reset();d=candidate;p=candidate_p;
    }
    if(!owner)throw std::runtime_error("CPU polynomial FFT/memory planning did not converge");
    auto &c=*owner;
    stats.d=d;stats.poly_size=p;
    stats.estimated_memory_mib=(p*(std::ceil(std::log2(double(p)))+18)+256)*double(gwnum_size(&c.gw)+64)/1048576.0;
    if(stats.estimated_memory_mib>o.memory_mib)throw std::runtime_error("Actual GWNUM layout exceeds planned memory budget; choose a smaller D");
    bool main_started=false;auto main_begin=begin;
    try {
        Points points(c);Array storage(c.gw,14);Point q{storage[0],storage[1]},step{storage[2],storage[3]},
            prev{storage[4],storage[5]},cur{storage[6],storage[7]},next{storage[8],storage[9]},base{storage[10],storage[11]};
        c.put(x,q.x);u64togw(&c.gw,1,q.z);c.put(a24,storage[12]);
        Array roots(c.gw,p);
        {
            const auto t=Clock::now();Array z(c.gw,p);points.dbl(q,step,storage[12]);
            std::vector<uint64_t> residues;residues.reserve(p);
            for(uint64_t j=1;j<=d/2;j+=2)if(std::gcd(j,d)==1)residues.push_back(j);
            if(residues.size()!=p)throw std::runtime_error("Baby wheel size differs");
            parallel_points(c,p,[&](Context &worker,size_t lo,size_t hi){
                // GW add/subtract may transform source operands in place to
                // reconcile their FFT states. Each worker owns its constants.
                Points ops(worker);Array work(worker.gw,11);Point a{work[0],work[1]},b{work[2],work[3]},v{work[4],work[5]},
                    local_q{work[6],work[7]},local_step{work[8],work[9]};
                ops.copy(q,local_q);ops.copy(step,local_step);gwcopy(&worker.gw,storage[12],work[10]);
                const auto first=residues[lo],last=residues[hi-1];
                ops.ladder(local_q,first,b,work[10]);ops.ladder(local_q,first>1?first-2:1,a,work[10]);
                size_t index=lo;
                for(uint64_t j=first;j<=last;j+=2){
                    if(std::gcd(j,d)==1){gwcopy(&worker.gw,b.x,roots[index]);gwcopy(&worker.gw,b.z,z[index++]);}
                    if(j==1)ops.add(local_step,local_q,local_q,v);else ops.add(b,local_step,a,v);
                    ops.copy(b,a);ops.copy(v,b);if((j&255)==1)worker.check();
                }
                ops.normalize(roots.v+lo,z.v+lo,hi-lo);
            });
            stats.point_seconds+=seconds(t);
        }
        auto init_tree_begin=Clock::now();
        auto ftree=tree(c,roots,0,p,true);roots=Array();
        struct Probe {Integer root,product;size_t index;};
        std::array<Probe,3> probes;
        if(o.check){
            for(size_t i=0;i<probes.size();++i){
                auto &probe=probes[i];probe.index=std::min<size_t>(p-1,i*(p/2));
                const Tree *leaf=ftree.get();while(leaf->left)leaf=probe.index<leaf->left->hi?leaf->left.get():leaf->right.get();
                c.get(probe.root.v,leaf->polynomial[0]);mpz_neg(probe.root.v,probe.root.v);mpz_mod(probe.root.v,probe.root.v,n);
                mpz_set_ui(probe.product.v,1);
            }
        }
        Array reversed_f(c.gw,p+1);
        for(size_t i=0;i<=p;++i)gwcopy(&c.gw,ftree->polynomial[p-i],reversed_f[i]);
        auto inv=c.inverse(reversed_f,p);reversed_f=Array();
        stats.tree_seconds+=seconds(init_tree_begin);
        points.ladder(q,d,base,storage[12]);
        c.check("giant base ladder");
        const uint64_t first=std::max<uint64_t>(1,b1>d/2?(b1-d/2)/d:1),last=(b2+d/2)/d;
        // Primes <= D/2 are not represented by a positive giant index. Cover
        // them explicitly, without sieving the production B2 interval.
        const uint64_t small_end=std::min<uint64_t>(b2,d/2);
        if(b1<small_end){
            std::vector<bool> composite(small_end+1);
            for(uint64_t v=2;v<=small_end;++v)if(!composite[v]){
                if(v>b1){points.ladder(q,v,next,storage[12]);c.divisor(next.z);c.check("small-prime ladder");}
                if(v<=small_end/v)for(uint64_t k=v*v;k<=small_end;k+=v)composite[k]=true;
            }
        }
        Array g_workspace_a(c.gw,2*p),g_workspace_b(c.gw,2*p);
        stats.init_seconds=seconds(begin);main_begin=Clock::now();main_started=true;
        Array h;
        for(uint64_t pos=first;pos<=last;){
            c.poll();const size_t count=static_cast<size_t>(std::min<uint64_t>(p,last-pos+1));
            auto part_begin=Clock::now();
            Array gx(c.gw,count),gz(c.gw,count);
            parallel_points(c,count,[&](Context &worker,size_t lo,size_t hi){
                Points ops(worker);Array work(worker.gw,9);Point a{work[0],work[1]},b{work[2],work[3]},v{work[4],work[5]},
                    local_base{work[6],work[7]};
                ops.copy(base,local_base);gwcopy(&worker.gw,storage[12],work[8]);
                ops.ladder(local_base,pos+lo,b,work[8]);ops.ladder(local_base,pos+lo-1,a,work[8]);
                for(size_t i=lo;i<hi;++i){
                    gwcopy(&worker.gw,b.x,gx[i]);gwcopy(&worker.gw,b.z,gz[i]);
                    if(pos+i==1)ops.dbl(local_base,v,work[8]);else ops.add(b,local_base,a,v);
                    ops.copy(b,a);ops.copy(v,b);if(!(i&127))worker.check();
                }
                ops.normalize(gx.v+lo,gz.v+lo,hi-lo);
            });gz=Array();stats.point_seconds+=seconds(part_begin);
            if(o.check){
                Integer giant,difference;
                for(size_t i=0;i<count;++i){c.get(giant.v,gx[i]);for(auto &probe:probes){
                    mpz_sub(difference.v,probe.root.v,giant.v);mpz_mul(probe.product.v,probe.product.v,difference.v);mpz_mod(probe.product.v,probe.product.v,n);
                }}
            }
            part_begin=Clock::now();auto g=tree(c,gx,0,count,false,&g_workspace_a,&g_workspace_b);gx=Array();stats.tree_seconds+=seconds(part_begin);
            part_begin=Clock::now();
            if(!h.size){h=c.remainder(g->polynomial,ftree->polynomial,inv);}
            else {auto product=c.convolution(h,g->polynomial);h=c.remainder(product,ftree->polynomial,inv);}
            stats.fold_seconds+=seconds(part_begin);
            pos+=count;++stats.blocks;stats.giants+=count;
        }
        if(!h.size)h=c.constant(1);
        auto descent_begin=Clock::now();
        // Scaled remainder: s=rev(H)*inv(rev(F)) mod X^P. Each child takes
        // middle coefficients of s*rev(sibling F), requiring no node inverses.
        Array padded(c.gw,p);
        for(size_t i=0;i<p;++i){if(i<h.size)gwcopy(&c.gw,h[i],padded[i]);else u64togw(&c.gw,0,padded[i]);}
        h=Array();auto forward_scaled=c.convolution(reverse(padded),view(inv),0,p);padded=Array();inv=Array();
        Array s(c.gw,p);for(size_t i=0;i<p;++i)gwcopy(&c.gw,forward_scaled[p-i-1],s[i]);forward_scaled=Array();
        Array values(c.gw,p);descend(c,*ftree,s,values);s=Array();ftree.reset();
        stats.descent_seconds=seconds(descent_begin);
        if(o.check){Integer got;for(auto &probe:probes){
            c.get(got.v,values[probe.index]);if(mpz_cmp(got.v,probe.product.v))throw std::runtime_error("Stage2 polynomial differs from direct GMP products");
            ++stats.checked_values;
        }}
        Array product=c.constant(1);
        for(size_t i=0;i<p;++i)c.mul(product[0],values[i],product[0]);
        c.check();Integer gcd;c.get(gcd.v,product[0]);mpz_gcd(gcd.v,gcd.v,n);
        stats.main_seconds=seconds(main_begin);
        if(mpz_cmp_ui(gcd.v,1)==0)return false;
        if(mpz_cmp(gcd.v,n)<0){mpz_set(factor,gcd.v);return true;}
        for(size_t i=0;i<p;++i)c.divisor(values[i]);
        throw std::runtime_error("Saturated Stage2 polynomial GCD; no NF result");
    }catch(const Hit &){if(main_started)stats.main_seconds=seconds(main_begin);else stats.init_seconds=seconds(begin);c.check();return true;}
}

void selftest(unsigned threads){
    Integer n,f,x,y,z;mpz_set_ui(n.v,1000003);mpz_mul_ui(n.v,n.v,1000033);
    Options o;o.threads=threads;Stats stats;Context c(n.v,f.v,o,stats,512);
    for(size_t count:{1u,2u,3u,4u,5u,8u,17u,32u,65u,129u,257u}){
        Array roots(c.gw,count);std::vector<uint64_t> expected_roots(count);
        for(size_t i=0;i<count;++i){expected_roots[i]=7919*(i+1)+17;u64togw(&c.gw,expected_roots[i],roots[i]);}
        auto t=tree(c,roots,0,count,true);auto rf=reverse(t->polynomial);
        Array rfa(c.gw,rf.size());for(size_t i=0;i<rf.size();++i)gwcopy(&c.gw,rf[i],rfa[i]);
        auto inv=c.inverse(rfa,count);
        Array h(c.gw,count);for(size_t i=0;i<count;++i)u64togw(&c.gw,103*(i+3),h[i]);
        auto forward=c.convolution(reverse(h),view(inv),0,count);Array s(c.gw,count);
        for(size_t i=0;i<count;++i)gwcopy(&c.gw,forward[count-i-1],s[i]);
        Array out(c.gw,count);descend(c,*t,s,out);
        for(size_t i=0;i<count;++i){
            mpz_set_ui(y.v,0);for(size_t k=count;k-->0;){mpz_mul_ui(y.v,y.v,expected_roots[i]);mpz_add_ui(y.v,y.v,103*(k+3));mpz_mod(y.v,y.v,n.v);}
            c.get(x.v,out[i]);if(mpz_cmp(x.v,y.v))throw std::runtime_error("Scaled polynomial descent differs from GMP Horner");
        }
        auto product=c.convolution(h,t->polynomial);auto rem=c.remainder(product,t->polynomial,inv);
        for(size_t i=0;i<rem.size;++i){c.get(x.v,rem[i]);if(mpz_sgn(x.v))throw std::runtime_error("Polynomial remainder differs from exact multiple");}
        // Nonzero remainders across several folds, including packed G workspaces.
        Array accumulator=c.copy(h),wa(c.gw,2*count),wb(c.gw,2*count);
        std::vector<std::unique_ptr<Integer>> expected;
        for(size_t i=0;i<count;++i){
            auto value=std::make_unique<Integer>();mpz_set_ui(value->v,0);
            for(size_t k=count;k-->0;){mpz_mul_ui(value->v,value->v,expected_roots[i]);mpz_add_ui(value->v,value->v,103*(k+3));mpz_mod(value->v,value->v,n.v);}
            expected.push_back(std::move(value));
        }
        for(unsigned block=0;block<3;++block){
            for(size_t j=0;j<count;++j)u64togw(&c.gw,1237*(j+5)+97*block,roots[j]);
            auto g=tree(c,roots,0,count,false,&wa,&wb);auto full=c.convolution(accumulator,g->polynomial);
            accumulator=c.remainder(full,t->polynomial,inv);
            auto scaled=c.convolution(reverse(accumulator),view(inv),0,count);Array reversed(c.gw,count);
            for(size_t i=0;i<count;++i)gwcopy(&c.gw,scaled[count-i-1],reversed[i]);
            descend(c,*t,reversed,out);
            for(size_t i=0;i<count;++i){
                for(size_t j=0;j<count;++j){mpz_set_ui(z.v,expected_roots[i]);mpz_sub_ui(z.v,z.v,1237*(j+5)+97*block);mpz_mul(expected[i]->v,expected[i]->v,z.v);mpz_mod(expected[i]->v,expected[i]->v,n.v);}
                c.get(x.v,out[i]);if(mpz_cmp(x.v,expected[i]->v))throw std::runtime_error("Polynomial fold differs from direct GMP products");
            }
        }
    }
    c.check();std::printf("gwpoly selftest: composite-ring Newton/remainder/scaled-Horner PASS; threads=%u maxerr=%.9g\n",threads,stats.max_roundoff);
}
}
