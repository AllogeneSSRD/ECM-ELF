#include "gwpoly.h"
#include "giant_range.h"
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
#include <limits>
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
struct Saturated : std::runtime_error {using std::runtime_error::runtime_error;};
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
    Context(const mpz_t modulus,mpz_t f,const Options &opt,Stats &s,size_t max_poly,bool preflight=false)
        :o(opt),stats(s),n(modulus),factor(f){
        try {
        guessCpuType();
        int available_flags=CPU_FLAGS;
#ifdef ECM_GWPOLY_TEST_NO_AVX512
        // Qualification build: exercise the normal dispatch under an AVX2/FMA3 ceiling.
        available_flags &= ~(CPU_AVX512F | CPU_AVX512PF | CPU_AVX512DQ | CPU_AVX512VL);
#endif
        if(o.exponent){Integer m;mpz_set_ui(m.v,1);mpz_mul_2exp(m.v,m.v,o.exponent);mpz_sub_ui(m.v,m.v,1);
            if(!mpz_divisible_p(m.v,n))throw std::runtime_error("GW carrier does not contain actual N");}
        for(unsigned extra=0;extra<16;++extra){
            gwinit(&gw);ready=true;
            gw.cpu_flags &= available_flags;
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
        stats.fft_length=gwfftlen(&gw);
        gwfft_description(&gw,stats.fft_description);
        const int planned_flags = (gw.cpu_flags & CPU_AVX) && (available_flags & CPU_AVX512F)
                                 ? available_flags : gw.cpu_flags;
        set_polymult_isa(planned_flags);
        if(preflight)return;
        binary.resize((static_cast<size_t>(gw.bit_length)+31)/32+8);
        gwuser_init_FFT1(&gw);
        polymult_init(&pm,&gw);pmready=true;
        polymult_set_cpu_flags(&pm,available_flags);polymult_set_max_num_threads(&pm,o.threads);
        polymult_default_tuning(&pm,std::max(256,CPU_L2_CACHE_SIZE),std::max(6144,CPU_L3_CACHE_SIZE));
        // polymult launches its own internal tasks. A naked launch_helpers call
        // instead dispatches the caller's user callback, which we do not set.
        set_polymult_isa(pm.cpu_flags);
        }catch(...){if(pmready)polymult_done(&pm);if(ready)gwdone(&gw);throw;}
    }
    void set_polymult_isa(int flags) {
        // Match the upstream HELPER_POLYMULT_LINE dispatch.
        const char* isa = (flags & CPU_AVX512F) ? "AVX-512" : (flags & CPU_FMA3) ? "FMA3" :
                          (flags & CPU_AVX) ? "AVX" : "SSE2";
        std::snprintf(stats.polymult_isa,sizeof(stats.polymult_isa),"%s",isa);
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
    void product(gwnum *a,size_t a_size,gwnum *b,size_t b_size,gwnum *out,
                 size_t first,size_t count,int operation,gwnum *subtract_from=nullptr){
        ++stats.polymults;
        polymult_set_num_threads(&pm,a_size+b_size<64?1:pm.max_num_threads);
        // GWNUM 31.6's SIMD FMA writer reads the zero-padding words even on
        // unpadded FFTs. Our addends live in GW arrays, whose pointer table
        // must cover that read at the first coefficient. FFT(1) can instead
        // be a separately allocated number in the scaled and MMGW layouts.
        const bool separate_subtract=subtract_from && (gw.GENERAL_MMGW_MOD ||
            (!gw.ZERO_PADDED_FFT && (gw.k!=1.0 || count*sizeof(gwnum)<GW_LARGE_HEADER_SIZE)));
        if(separate_subtract)operation &= ~POLYMULT_FNMADD;
        const int base_options=(operation & POLYMULT_NO_UNFFT) ? operation : operation|POLYMULT_NEXTFFT;
        auto &plan=plans[{a_size,b_size,count,first,base_options,pm.num_threads}];
        int options=base_options|POLYMULT_SAVE_PLAN;
        if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult2(&pm,a,a_size,b,b_size,out,count,separate_subtract?nullptr:subtract_from,0,first,options);
        plan=pm.plan;
        if(separate_subtract){
            for(size_t i=0;i<count;++i)
                gwsub3o(&gw,subtract_from[i],out[i],out[i],GWADD_FORCE_NORMALIZE);
        }
        check("polynomial convolution");
    }
    Array convolution(View a,View b,size_t first,size_t count,const Array *subtract_from=nullptr){
        if(a.empty() || b.empty() || !count)throw std::runtime_error("Empty polynomial multiplication");
        Array out(gw,count);
        const int operation=POLYMULT_MULMID|(subtract_from?POLYMULT_FNMADD:0);
        product(a.data(),a.size(),b.data(),b.size(),out.v,first,count,operation,
                subtract_from?subtract_from->v:nullptr);
        return out;
    }
    void monic_coefficients(gwnum *a,size_t a_degree,gwnum *b,size_t b_degree,gwnum *out,bool defer,bool negate_monic=false){
        const size_t degree=a_degree+b_degree;
        ++stats.polymults;
        polymult_set_num_threads(&pm,degree<64?1:pm.max_num_threads);
        const int monic=negate_monic?POLYMULT_INVEC1_MONIC_NEGATE|POLYMULT_INVEC2_MONIC_NEGATE:
            POLYMULT_INVEC1_MONIC|POLYMULT_INVEC2_MONIC;
        const int base_options=monic|(defer?POLYMULT_NO_UNFFT:POLYMULT_NEXTFFT);
        auto &plan=plans[{a_degree,b_degree,degree,0,base_options,static_cast<unsigned>(pm.num_threads)}];
        int options=base_options|POLYMULT_SAVE_PLAN;
        if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult(&pm,a,a_degree,b,b_degree,out,degree,options);
        plan=pm.plan;
        check("monic polynomial product");
    }
    void monic_product(const Array &a,const Array &b,Array &out,bool defer=false){
        monic_coefficients(a.v,a.size-1,b.v,b.size-1,out.v,defer);
        u64togw(&gw,1,out[a.size+b.size-2]);
    }
    Array convolution(const Array &a,const Array &b){return convolution(view(a),view(b),0,a.size+b.size-1);}
    Array scaled_remainder(const Array &h,const Array &inv){
        // rev_P(H) starts with P-|H| zero coefficients. Truncation leaves
        // only |H| useful outputs; write them directly in descent order.
        Array scaled(gw,inv.size);
        auto input=reverse(h);
        View output(scaled.v,scaled.v+h.size);
        std::reverse(output.begin(),output.end());
        product(input.data(),h.size,inv.v,h.size,output.data(),0,h.size,POLYMULT_MULLO);
        for(size_t i=h.size;i<scaled.size;++i)u64togw(&gw,0,scaled[i]);
        return scaled;
    }
    void scaled_children(Array &parent,const Array &left,const Array &right,size_t a,size_t b){
        // Both circular products use the same transformed parent. Wrapped
        // coefficients lie below each requested high interval. Like the
        // upstream F descent the outputs may occupy the parent's own row.
        constexpr int child=POLYMULT_INVEC2_MONIC|POLYMULT_CIRCULAR|POLYMULT_MULHI;
        polymult_arg args[2]={{right.v,b,parent.v,a,nullptr,parent.size,0,child},
                              {left.v,a,parent.v+a,b,nullptr,parent.size,0,child}};
        polymult_set_num_threads(&pm,parent.size+std::max(a,b)<64?1:pm.max_num_threads);
        // Negative tag distinguishes these two-product plans from product().
        auto &plan=plans[{a,b,parent.size,0,-1,pm.num_threads}];
        int options=POLYMULT_NEXTFFT|POLYMULT_SAVE_PLAN;
        if(plan){pm.plan=plan;options|=POLYMULT_USE_PLAN;}
        polymult_several(&pm,parent.v,parent.size,args,2,options);
        plan=pm.plan;stats.polymults+=2;check("scaled polynomial descent");
    }
    Array inverse(View f,size_t count){
        // F*g = 1 + X^known * error (mod X^(known+extra)). The existing
        // inverse stays in place; only -g*error mod X^extra is appended.
        Array g(gw,count);u64togw(&gw,1,g[0]);
        Array zero=constant(0);gwfft(&gw,zero[0],zero[0]);
        for(size_t known=1;known<count;){
            const auto extra=std::min(known,count-known);
            Array error(gw,extra);
            product(f.data(),std::min(f.size(),known+extra),g.v,known,
                    error.v,known,extra,POLYMULT_MULMID);
            for(size_t i=0;i<extra;++i)
                gwsub3o(&gw,zero[0],error[i],error[i],GWADD_FORCE_NORMALIZE);
            product(g.v,extra,error.v,extra,g.v+known,0,extra,POLYMULT_MULLO);
            known+=extra;
        }
        return g;
    }
    Array remainder(const Array &t,const Array &f,const Array &inv){
        const size_t degree=f.size-1;
        if(t.size<=degree)return copy(t);
        const size_t k=t.size-degree;auto rt=reverse(t);rt.resize(k);auto rv=view(inv);rv.resize(k);
        auto qr=convolution(rt,rv,0,k);auto q=reverse(qr);
        // The monic term of F contributes only at degree and above.
        return convolution(q,View(f.v,f.v+degree),0,degree,&t);
    }

    bool supports_deferred_h_fold() const {
        return o.exponent && gw.k==1.0 && !gw.GENERAL_MOD && !gw.GENERAL_MMGW_MOD &&
            !gw.ZERO_PADDED_FFT && !gw.NEGACYCLIC_FFT;
    }
    bool use_deferred_h_fold(size_t degree) const {
        if(!supports_deferred_h_fold() || degree*sizeof(gwnum)<GW_LARGE_HEADER_SIZE)return false;
        // A low remainder coefficient sums products from both H*G and Q*F.
        const auto required=polymult_safety_margin(2*(degree+1),2*(degree+1))+0.25f;
        return gw.EXTRA_BITS/2.0f+gw.polymult_safety_margin>required;
    }
    void fold(Array &h,Array &workspace,size_t g_degree,const Array &f,View &reciprocal,bool defer_low){
        const size_t degree=f.size-1;
        if(!h.size){
            if(g_degree<degree){
                h=copy(Array(gw,workspace.v,g_degree+1));
            }else{
                h=Array(gw,degree);
                // Monic F and G have the same degree so the quotient is 1.
                for(size_t i=0;i<degree;++i)
                    gwsub3o(&gw,workspace[i],f[i],h[i],GWADD_FORCE_NORMALIZE);
                check("first polynomial fold");
            }
            return;
        }

        // Use the two existing buffers for the low and high parts of H*G.
        // Even for a short G block, workspace has degree+1 allocated slots.
        View coefficients(degree+g_degree);
        std::copy_n(workspace.v,degree,coefficients.begin());
        std::copy_n(h.v,g_degree,coefficients.begin()+degree);
        // On a cyclic k=1 transform the low products already have the
        // Fourier scaling expected of the next FNMADD addend. Reserve
        // enough headroom to subtract two sums of coefficient products.
        if(o.check)std::printf("gwpoly-fold: deferred_low=%d\n",defer_low);
        product(workspace.v,g_degree,h.v,degree,coefficients.data(),0,degree+g_degree,
                     POLYMULT_INVEC1_MONIC|(defer_low?POLYMULT_NO_UNFFT:0));
        if(defer_low){
            poly_unfft_fft_coefficients(&pm,h.v,g_degree);
            check("polynomial quotient input normalization");
        }

        // Q is the high g_degree coefficients of the product of the high
        // block and the reversed truncated reciprocal. Its leading one is
        // implicit. When g_degree=1 the high block already equals Q.
        if(g_degree>1)
            product(h.v,g_degree,reciprocal.data()+degree-g_degree,g_degree-1,
                         h.v,0,g_degree,POLYMULT_INVEC2_MONIC|POLYMULT_MULHI);
        // Only the low coefficients of F can contribute here. Keep the
        // low H*G block intact as the FMA input and overwrite the quotient.
        product(h.v,g_degree,f.v,degree,h.v,0,degree,
                     POLYMULT_MULLO|POLYMULT_FNMADD,workspace.v);
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
std::unique_ptr<Tree> retained_ftree(Context &,const Array &,size_t,size_t,bool need_root=true,std::vector<Array> rows={});
void descend_node(Context &c,const Tree &t,Array &values,const Array *roots,gwnum scratch,double &rebuild_seconds,std::vector<Array> *sibling_rows=nullptr){
    if(!t.left){
        if(t.hi-t.lo==1)return;
        if(!roots)throw std::logic_error("Missing roots for F subtree rebuild");
        // The parent remainder is already available. Only F polynomials
        // strictly below this checkpoint are consumed by descent.
        const auto start=Clock::now();
        auto segment=retained_ftree(c,*roots,t.lo,t.hi,false,
                                    sibling_rows ? std::move(*sibling_rows) : std::vector<Array>{});
        rebuild_seconds+=seconds(start);
        descend_node(c,*segment,values,roots,scratch,rebuild_seconds);
        if(sibling_rows)*sibling_rows=std::move(segment->rows);
        return;
    }
    const auto a=t.left->hi-t.left->lo,b=t.right->hi-t.right->lo;
    Array s(c.gw,values.v+t.lo,a+b);
    if(a==1 && b==1){
        // High coefficient of S*(X-r) mod X^2-1 equals S0-r*S1.
        // Hold the first result until both overlapping inputs have been read.
        gwmuladd4(&c.gw,s[1],t.right->polynomial[0],s[0],scratch,GWMUL_FFT_S13);
        gwmuladd4(&c.gw,s[1],t.left->polynomial[0],s[0],values[t.lo+1],0);
        gwcopy(&c.gw,scratch,values[t.lo]);
        c.check("scaled leaf pair");
        return;
    }
    c.scaled_children(s,t.left->polynomial,t.right->polynomial,a,b);
    // Equally sized checkpoint siblings rebuild the same row layout.
    // Release the spare before returning to their parent's caller.
    std::vector<Array> spare;
    auto *reuse=(roots && a==b && !t.left->left && !t.right->left) ? &spare : nullptr;
    descend_node(c,*t.left,values,roots,scratch,rebuild_seconds,reuse);
    descend_node(c,*t.right,values,roots,scratch,rebuild_seconds,reuse);
}
double descend(Context &c,const Tree &t,Array &values,const Array *roots=nullptr){
    Array scratch(c.gw,1);
    double rebuild_seconds=0;
    descend_node(c,t,values,roots,scratch[0],rebuild_seconds);
    return rebuild_seconds;
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
    void add_fused(Point p,Point q,Point difference,Point out){
        auto &gw=c.gw;
        // These inputs belong to this worker and may keep their forward FFTs.
        // The two squares omit the same factor of four in the projective output.
        gwmulmulsub5(&gw,p.x,q.z,p.z,q.x,out.z,GWMUL_FFT_S1234|GWMUL_STARTNEXTFFT);
        gwmulmulsub5(&gw,p.x,q.x,p.z,q.z,out.x,GWMUL_STARTNEXTFFT);
        gwsquare2(&gw,out.z,out.z,GWMUL_STARTNEXTFFT);
        gwsquare2(&gw,out.x,out.x,GWMUL_STARTNEXTFFT);
        gwmul3(&gw,out.z,difference.x,out.z,GWMUL_FFT_S2|GWMUL_STARTNEXTFFT);
        gwmul3(&gw,out.x,difference.z,out.x,GWMUL_FFT_S2|GWMUL_STARTNEXTFFT);
    }
    void ladder(Point p,uint64_t k,Point out,gwnum a24,Point* successor=nullptr){
        if(!k){
            u64togw(&c.gw,1,out.x);u64togw(&c.gw,0,out.z);
            if(successor)copy(p,*successor);
            return;
        }
        Array storage(c.gw,6);Point a{storage[0],storage[1]},b{storage[2],storage[3]},t{storage[4],storage[5]};
        copy(p,a);dbl(p,b,a24);
        uint64_t bit=1;while(bit<=k/2)bit<<=1;
        for(bit>>=1;bit;bit>>=1){
            if(k&bit){add(a,b,p,t);dbl(b,b,a24);copy(t,a);}
            else {add(a,b,p,t);dbl(a,a,a24);copy(t,b);}
        }
        copy(a,out);
        if(successor)copy(b,*successor);
    }
    void normalize(gwnum *x,gwnum *z,size_t count,bool negate=false){
        auto &gw=c.gw;
        // Multiply each numerator by the product of preceding denominators
        // while that product is live. Keep the original z values for GCDs.
        Array prefix(gw,1);
        gwcopy(&gw,z[0],prefix[0]);
        for(size_t i=1;i<count;++i){
            gwmul3(&gw,prefix[0],x[i],x[i],GWMUL_FFT_S1|GWMUL_STARTNEXTFFT);
            gwmul3(&gw,prefix[0],z[i],prefix[0],GWMUL_FFT_S2|GWMUL_STARTNEXTFFT);
        }
        Integer product,inv;c.get(product.v,prefix[0]);
        if(!mpz_invert(inv.v,product.v,c.n)){
            for(size_t i=0;i<count;++i)c.divisor(z[i]);
            throw Saturated("Saturated point denominator; no NF result");
        }
        // Walk the inverse prefix products backward to finish x[i]/z[i].
        // Giant leaves use X - x/z. Negate the shared inverse once per batch.
        if(negate)mpz_sub(inv.v,c.n,inv.v);
        c.put(inv.v,prefix[0]);
        for(size_t i=count-1;i>0;--i){
            gwmul3(&gw,prefix[0],x[i],x[i],GWMUL_FFT_S1);
            gwmul3(&gw,prefix[0],z[i],prefix[0],GWMUL_STARTNEXTFFT);
        }
        gwmul3(&gw,prefix[0],x[0],x[0],0);
        c.check("batch point normalization");
    }
};
template<class Function> void parallel_points(Context &parent,size_t count,Function function,size_t min_chunk=256){
    struct Worker {
        Integer factor;Stats stats{};Context context;std::exception_ptr error;
        explicit Worker(Context &p):context(p,factor.v,stats){}
    };
    const size_t threads=std::min<size_t>(parent.o.threads,std::max<size_t>(1,count/min_chunk));
    if(threads==1){
        function(parent,0,count);
        parent.check("point worker");
        return;
    }
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
    sizes[depth]+=hi-lo;
    if(hi-lo>1){const auto mid=lo+(hi-lo)/2;tree_layout(lo,mid,depth+1,sizes);tree_layout(mid,hi,depth+1,sizes);}
}
std::unique_ptr<Tree> tree_skeleton(Context &c,size_t lo,size_t hi,size_t depth,std::vector<Array> &rows,
                                    std::vector<size_t> &offsets,std::vector<std::vector<Tree *>> &levels){
    auto t=std::make_unique<Tree>(lo,hi);
    if(rows[depth].size){
        const auto count=depth==0?rows[0].size:hi-lo;
        t->polynomial=Array(c.gw,rows[depth].v+offsets[depth],count);
        offsets[depth]+=count;
    }
    levels[depth].push_back(t.get());
    if(hi-lo>1){const auto mid=lo+(hi-lo)/2;
        t->left=tree_skeleton(c,lo,mid,depth+1,rows,offsets,levels);
        t->right=tree_skeleton(c,mid,hi,depth+1,rows,offsets,levels);}
    return t;
}
std::unique_ptr<Tree> product_tree(Context &c,const Array &roots,size_t lo,size_t hi,Array *workspace=nullptr,bool negative_roots=false){
    const size_t count=hi-lo, row_size=count+1;
    if(workspace && workspace->size<row_size)
        throw std::runtime_error("G tree workspace too small");
    auto t=std::make_unique<Tree>(lo,hi);
    t->polynomial=workspace?Array(c.gw,workspace->v,row_size):Array(c.gw,row_size);
    gwnum *row=t->polynomial.v;
    // (X-x)(X-y) = X^2 - (x+y)X + xy.
    const bool in_place=roots.v+lo==row;
    auto pair_roots=[&](Context &worker,Array &sum,size_t first,size_t last){
        for(size_t offset=first;offset<last;offset+=2){
            const auto x=roots[lo+offset];
            if(offset+1==count){
                if(!in_place)gwcopy(&worker.gw,x,row[offset]);
                if(!negative_roots)gwsmallmul(&worker.gw,-1,row[offset]);
            }else if(negative_roots){
                // Keep the sum before overwriting either signed affine input.
                const auto y=roots[lo+offset+1];
                gwadd3o(&worker.gw,x,y,sum[0],GWADD_FORCE_NORMALIZE);
                gwmul3(&worker.gw,x,y,row[offset],GWMUL_STARTNEXTFFT);
                gwfft(&worker.gw,sum[0],row[offset+1]);
                if(!(offset&255))worker.check("G leaf pairs");
            }else if(in_place){
                worker.monic_coefficients(row+offset,1,row+offset+1,1,row+offset,false,true);
            }else{
                const auto y=roots[lo+offset+1];
                worker.mul(x,y,row[offset]);
                gwadd3o(&worker.gw,x,y,row[offset+1],GWADD_FORCE_NORMALIZE);
                gwsmallmul(&worker.gw,-1,row[offset+1]);
            }
        }
    };
    const auto merge=[&](Context &worker,size_t offset,size_t width,size_t last,bool defer){
        if(last-offset<=width)return;
        const size_t right=std::min(width,last-offset-width);
        worker.monic_coefficients(row+offset+width,right,row+offset,width,row+offset,defer);
    };

    // Finish the lower levels of each signed G tile while its coefficients
    // are cache-local. Whole-curve tests qualified blocks through 20160
    // roots; larger blocks keep batched level normalization.
    constexpr size_t max_tiled_roots=20160;
    size_t tile=2;
    if(negative_roots && c.o.threads==1 && count<=max_tiled_roots){
        const auto bytes_per_root=gwnum_size(&c.gw)+64;
        while(tile<32 && 2*tile*bytes_per_root<=256*1024)tile*=2;
    }
    if(negative_roots && c.o.check)
        std::printf("gwpoly-gtree: roots=%zu tile=%zu\n",count,tile);
    if(tile==2){
        tree_jobs(c,(count+1)/2,[&](Context &worker,size_t begin,size_t end){
            Array sum(worker.gw,negative_roots?1:0);
            pair_roots(worker,sum,2*begin,std::min(count,2*end));
            worker.check("linear factor pairs");
        });
    }else{
        Array sum(c.gw,1);
        for(size_t first=0;first<count;first+=tile){
            const auto last=std::min(count,first+tile);
            pair_roots(c,sum,first,last);
            for(size_t width=2;width<last-first;width*=2)
                for(size_t offset=first;offset+width<last;offset+=2*width)
                    merge(c,offset,width,last,false);
            c.check("G tree tile");
        }
    }
    // Every node stores only the coefficients below its leading one. Left,
    // right and result can share a range of the packed row.
    for(size_t width=tile;width<count;width*=2){
        const size_t stride=2*width,jobs=(count+stride-1)/stride;
        tree_jobs(c,jobs,[&](Context &worker,size_t begin,size_t end){
            for(size_t j=begin;j<end;++j)merge(worker,j*stride,width,count,jobs>=128);
        });
        if(jobs>=128){
            polymult_set_num_threads(&c.pm,c.pm.max_num_threads);
            poly_unfft_fft_coefficients(&c.pm,row,count);
            c.check("G tree coefficient normalization");
        }
    }
    u64togw(&c.gw,1,row[count]);
    return t;
}
std::unique_ptr<Tree> retained_ftree(Context &c,const Array &roots,size_t lo,size_t hi,bool need_root,std::vector<Array> rows){
    if(rows.empty()){
        std::vector<size_t> sizes;tree_layout(lo,hi,0,sizes);
        // Every internal row stores just the low coefficients of its monic nodes.
        sizes[0]=need_root?hi-lo+1:0;
        for(auto size:sizes)rows.emplace_back(c.gw,size);
    }
    std::vector<size_t> offsets(rows.size());std::vector<std::vector<Tree *>> levels(rows.size());
    auto t=tree_skeleton(c,lo,hi,0,rows,offsets,levels);t->rows=std::move(rows);
    const size_t first_row=need_root?0:1;
    for(size_t depth=levels.size();depth>first_row;){
        --depth;
        auto &nodes=levels[depth];
        tree_jobs(c,nodes.size(),[&](Context &worker,size_t begin,size_t end){for(size_t i=begin;i<end;++i){
            auto &node=*nodes[i];
            if(node.left)worker.monic_coefficients(node.left->polynomial.v,node.left->hi-node.left->lo,
                node.right->polynomial.v,node.right->hi-node.right->lo,node.polynomial.v,nodes.size()>=128);
            else {gwcopy(&worker.gw,roots[node.lo],node.polynomial[0]);gwsmallmul(&worker.gw,-1,node.polynomial[0]);}
        }});
        if(nodes.size()>=128){
            polymult_set_num_threads(&c.pm,c.pm.max_num_threads);
            poly_unfft_fft_coefficients(&c.pm,t->rows[depth].v,t->rows[depth].size);c.check("F tree coefficient normalization");
        }
    }
    if(need_root)u64togw(&c.gw,1,t->polynomial[hi-lo]);
    return t;
}
template<class Function>
void for_each_wheel_prime(uint64_t wheel, Function visit) {
    for (uint64_t prime = 2; prime <= wheel / prime; ++prime) {
        if (wheel % prime != 0) continue;
        visit(prime);
        do {
            wheel /= prime;
        } while (wheel % prime == 0);
    }
    if (wheel > 1) visit(wheel);
}

uint64_t phi(uint64_t wheel) {
    uint64_t result = wheel;
    for_each_wheel_prime(wheel, [&](uint64_t prime) { result -= result / prime; });
    return result;
}
std::unique_ptr<Tree> partial_ftree(Context &c,const Array &roots,size_t lo,size_t hi,unsigned splits){
    auto t=std::make_unique<Tree>(lo,hi);
    if(!splits || hi-lo==1){
        auto segment=product_tree(c,roots,lo,hi);
        t->polynomial=std::move(segment->polynomial);
    }else{
        const auto mid=lo+(hi-lo)/2;
        t->left=partial_ftree(c,roots,lo,mid,splits-1);
        t->right=partial_ftree(c,roots,mid,hi,splits-1);
        t->polynomial=Array(c.gw,hi-lo+1);
        c.monic_product(t->left->polynomial,t->right->polynomial,t->polynomial);
    }
    return t;
}
struct FTreePlan {bool rebuild;double bytes;unsigned splits;};
FTreePlan ftree_plan(size_t p,double bytes,double budget,unsigned threads,int requested=automatic_tree){
    const auto levels=static_cast<unsigned>(std::ceil(std::log2(double(p))));
    // Each layer has at most P coefficients plus one monic term per node.
    // Allow four FFT vectors per worker (two inputs, monic adjustment and
    // pass scratch), plus twiddles and GW clone/point workspace.
    const double scratch=polymult_fft_size(2*p+2)*(4.0*128+sizeof(double))*threads;
    const auto estimate=[&](double slots){
        return (slots+256+48.0*threads)*bytes*1.06+scratch;
    };
    const double full_rows=(levels+3.0)*p;
    const double limit=budget*1048576;
    FTreePlan best{false,estimate(full_rows+4.0*p+8),0};
    if(requested==0 || (requested==automatic_tree && best.bytes<=limit))return best;
    if(requested>=static_cast<int>(levels))return {true,std::numeric_limits<double>::infinity(),0};
    if(requested!=automatic_tree)best={true,std::numeric_limits<double>::infinity(),0};
    // Keep the deepest affordable top rows. Lower segments are rebuilt
    // individually, after all giant and reciprocal buffers are released.
    const auto first=requested>0?static_cast<unsigned>(requested):1u;
    const auto end=requested>0?first+1:levels;
    for(unsigned splits=first;splits<end;++splits){
        const size_t segments=size_t(1)<<splits,chunk=(p+segments-1)/segments;
        const double retained=(splits+2.0)*p+2*segments; // includes the baby roots
        const double init=retained+2.0*p+8; // inverse and truncated Newton error
        const double main=retained+4.0*p+8; // inverse, H, G, giant denominators
        // The scaled root row becomes the leaf output. Keep two rows of GW allocation headroom.
        const double descent=retained+3.0*p+chunk*(std::ceil(std::log2(double(chunk)))+3);
        const double peak=estimate(std::max({init,main,descent}));
        if(peak<=limit || (best.bytes>limit && peak<best.bytes))best={true,peak,splits};
    }
    return best;
}
const std::vector<uint64_t>& wheel_choices(){
    static const auto choices=[] {
        // Smooth wheels including repeated odd factors.
        std::vector<uint64_t> wheels;
        for(uint64_t wheel:{30ull,210ull,2310ull,30030ull,510510ull})
            for(uint64_t k:{1ull,2ull,3ull,4ull,5ull,6ull,7ull,8ull,9ull,11ull,13ull,15ull,21ull,27ull,35ull,45ull,63ull,81ull})
                if(wheel*k<=20000000 && std::find(wheels.begin(),wheels.end(),wheel*k)==wheels.end())
                    wheels.push_back(wheel*k);
        // Intermediate geometries verified on wide cofactors.
        wheels.push_back(10920);
        wheels.push_back(43890);
        return wheels;
    }();
    return choices;
}
const std::vector<uint64_t>& dense_wheel_choices(){
    static const auto choices=[] {
        auto wheels=wheel_choices();
        constexpr std::array<uint64_t,6> primes{2,3,5,7,11,13};
        const auto add=[&](auto &self,uint64_t d,size_t first)->void {
            wheels.push_back(d);
            for(size_t i=first;i<primes.size();++i)
                if(d<=20000000/primes[i])self(self,d*primes[i],i);
        };
        add(add,30,0);
        std::sort(wheels.begin(),wheels.end());
        wheels.erase(std::unique(wheels.begin(),wheels.end()),wheels.end());
        return wheels;
    }();
    return choices;
}
double wheel_cost(uint64_t d,uint64_t b2,double bytes,uint64_t budget,unsigned threads,int splits=automatic_tree){
    const auto p=phi(d)/2;
    const auto plan=ftree_plan(static_cast<size_t>(p),bytes,budget,threads,splits);
    if(plan.bytes>double(budget)*1048576)return std::numeric_limits<double>::infinity();
    const double levels=std::ceil(std::log2(double(p)));
    return (plan.rebuild?8:6)*p*levels+(double(b2)/d+p)*(levels+8);
}
uint64_t choose_d(uint64_t b2,uint64_t forced,double bytes,uint64_t budget,unsigned threads,int splits){
    if(forced && std::isfinite(wheel_cost(forced,b2,bytes,budget,threads,splits)))return forced;
    double best=std::numeric_limits<double>::infinity();
    uint64_t result=0;
    if(!forced)for(auto d:wheel_choices()){
        const auto cost=wheel_cost(d,b2,bytes,budget,threads);
        if(cost<best){best=cost;result=d;}
    }
    if(!result)throw std::runtime_error("No CPU polynomial plan within memory budget");
    return result;
}

uint64_t fft_exponent_ceiling(const gwhandle& gw, double margin) {
    const auto upper = [margin](const gwhandle& fft) {
        // For base two the rational-word discount can add half a bit per
        // FFT word. Use that upper envelope here. The actual modulus always
        // goes through GWNUM's distribution-dependent safety test.
        const double bpw = fft.fft_max_bits_per_word - fft.safety_margin - margin + 0.5;
        return static_cast<uint64_t>(std::max(0.0, bpw) * fft.FFTLEN /
                                     (fft.ZERO_PADDED_FFT ? 2 : 1));
    };
    if (gw.GENERAL_MMGW_MOD)
        return std::min(upper(*gw.cyclic_gwdata), upper(*gw.negacyclic_gwdata));
    return upper(gw);
}
std::unique_ptr<Context> prepare_plan(mpz_srcptr n,mpz_ptr factor,uint64_t b1,uint64_t b2,
                                      const Options &o,Stats &stats,FTreePlan &fplan,bool preflight=false){
    if(!o.threads || b2<=b1 || b2>max_b2)throw std::runtime_error("Invalid GW polynomial bounds/threads");
    if(o.d && (o.d<6 || o.d%2 || o.d>20000000))throw std::runtime_error("Invalid CPU polynomial wheel D");
    if(o.f_tree_splits!=automatic_tree && (!o.d || o.f_tree_splits<segmented_tree || o.f_tree_splits>=64))
        throw std::runtime_error("A fixed F-tree layout requires a fixed D and valid split count");
    uint64_t degree=phi(o.d?o.d:30)/2;
    // A larger FFT can change which wheels fit the coefficient budget.
    for(unsigned attempt=0;attempt<6;++attempt){
        auto owner=std::make_unique<Context>(n,factor,o,stats,degree,preflight);
        owner->poll();
        const double bytes=double(gwnum_size(&owner->gw)+64);
        const auto candidate=choose_d(b2,o.d,bytes,o.memory_mib,o.threads,o.f_tree_splits);
        degree=phi(candidate)/2;
        const auto required=polymult_safety_margin(degree+1,degree+1)+0.25f;
        if(owner->gw.EXTRA_BITS/2.0f+owner->gw.polymult_safety_margin<=required)continue;
        // A memory re-plan can settle on a smaller D than the wheel that
        // selected this FFT. Build the shortest safe FFT for the settled D,
        // exactly as for a forced wheel. Keep the wheel fixed to avoid a cycle.
        if(required<owner->gw.polymult_safety_margin){
            owner.reset();
            owner=std::make_unique<Context>(n,factor,o,stats,degree,preflight);
            owner->poll();
        }
        gwset_polymult_safety_margin(&owner->gw,required);
        stats.d=candidate;
        stats.poly_size=degree;
        fplan=ftree_plan(static_cast<size_t>(degree),double(gwnum_size(&owner->gw)+64),o.memory_mib,o.threads,o.f_tree_splits);
        stats.estimated_memory_mib=fplan.bytes/1048576.0;
        if(stats.estimated_memory_mib>o.memory_mib)
            throw std::runtime_error("Actual GWNUM layout exceeds planned memory budget; choose a smaller D");
        return owner;
    }
    throw std::runtime_error("CPU polynomial FFT/memory planning did not converge");
}
}

const char* cpu() {
    static const std::string description = [] {
        guessCpuType();
        std::string name(CPU_BRAND);
        name.erase(name.find_last_not_of(' ') + 1);
        name += " supported:";
        if (CPU_FLAGS & CPU_SSE2) name += " SSE2";
        if (CPU_FLAGS & CPU_AVX) name += " AVX";
        if (CPU_FLAGS & CPU_FMA3) name += " FMA3";
        if (CPU_FLAGS & CPU_AVX2) name += " AVX2";
        if (CPU_FLAGS & CPU_AVX512F) name += " AVX-512F";
        return name;
    }();
    return description.c_str();
}
const char* identity(){return "static gwpoly";}
const char* load_error(){return nullptr;}
unsigned stage2_abi(){return 5;}
const char *version(){return GWNUM_VERSION;}
std::vector<TuneCandidate> tune_candidates(mpz_srcptr n,uint64_t b1,uint64_t b2,const Options &o){
    if(o.tune_candidate_limit>max_tune_candidates)throw std::invalid_argument("Invalid CPU Stage2 candidate limit");
    Integer unused_factor;
    Stats automatic{};
    FTreePlan tree_plan{};
    auto owner=prepare_plan(n,unused_factor.v,b1,b2,o,automatic,tree_plan,true);
    const double bytes=double(gwnum_size(&owner->gw)+64);
    const auto width = o.exponent ? o.exponent : mpz_sizeinbase(n, 2);
    const auto describe = [&](const Stats& stats, const gwhandle& gw) {
        const auto exponent = std::max<uint64_t>(width, gw.n);
        const double margin = polymult_safety_margin(stats.poly_size+1,stats.poly_size+1)+0.25;
        TuneCandidate plan{stats.d, stats.poly_size, stats.fft_length, exponent,
                           std::max(exponent, fft_exponent_ceiling(gw, margin)),
                           tree_plan.splits, stats.estimated_memory_mib, {}, {}};
        std::snprintf(plan.fft_layout, sizeof(plan.fft_layout), "%s", stats.fft_description);
        std::snprintf(plan.polymult_isa, sizeof(plan.polymult_isa), "%s", stats.polymult_isa);
        return plan;
    };
    std::vector<TuneCandidate> candidates{describe(automatic,owner->gw)};
    if(o.d || o.tune_candidate_limit==1)return candidates;
    owner.reset();

    Options forced=o;
    auto inspect=[&](uint64_t d,int splits=automatic_tree) {
        forced.d=d;
        forced.f_tree_splits=splits;
        Stats candidate{};
        auto context=prepare_plan(n,unused_factor.v,b1,b2,forced,candidate,tree_plan,true);
        return describe(candidate,context->gw);
    };
    auto add=[&](uint64_t d,int splits=automatic_tree) {
        try {candidates.push_back(inspect(d,splits));return true;}
        catch(const std::runtime_error&){
            if(o.stop && o.stop())throw;
            // The required coefficient FFT or F tree may fail admission.
            return false;
        }
    };
    if(o.tune_candidate_limit){
        struct Ranked {uint64_t d,blocks;double score;bool segmented;};
        const auto giant_count=[&](uint64_t d,uint64_t p) {
            return giant_range(b1,(b2+d/2)/d,d,p).size();
        };
        const auto rank=[&](uint64_t d,uint64_t p,bool segmented) {
            const auto giants=giant_count(d,p);
            const auto levels=std::ceil(std::log2(double(p)));
            return Ranked{d,(giants+p-1)/p,(segmented?8:6)*p*levels+(double(giants)+p)*(levels+8),segmented};
        };
        std::vector<Ranked> ranked;
        for(const auto d:o.dense_d?dense_wheel_choices():wheel_choices()){
            if(d==automatic.d)continue;
            const auto p=phi(d)/2;
            const auto plan=ftree_plan(static_cast<size_t>(p),bytes,o.memory_mib,o.threads);
            if(plan.bytes>double(o.memory_mib)*1048576)continue;
            ranked.push_back(rank(d,p,plan.rebuild));
        }
        std::sort(ranked.begin(),ranked.end(),[](const auto &a,const auto &b){
            return std::tie(a.score,a.d)<std::tie(b.score,b.d);
        });
        const auto baseline=candidates.front();
        if(baseline.f_tree_splits==0){
            // Test a layout that releases substantial memory. At the full
            // budget a very deep segmented tree can be nearly as large as full.
            const auto reduced=0.65*baseline.planned_mib;
            const auto slim=ftree_plan(baseline.poly_size,bytes,reduced,o.threads,segmented_tree);
            if(slim.bytes<=reduced*1048576)add(baseline.d,static_cast<int>(slim.splits));
        }
        // Reserve a trial for a larger segmented geometry that reduces G blocks.
        if(candidates.size()<o.tune_candidate_limit){
            const auto reference=rank(baseline.d,baseline.poly_size,baseline.f_tree_splits!=0);
            unsigned attempts=0;
            for(const auto &plan:ranked){
                // Fewer blocks alone do not justify a much more expensive tree.
                if(plan.score>1.5*reference.score)break;
                if(!plan.segmented || plan.d<=reference.d || plan.blocks>=reference.blocks)continue;
                if(add(plan.d,segmented_tree) || ++attempts==4)break;
            }
        }
        const size_t attempts=std::min<size_t>(3*o.tune_candidate_limit,ranked.size());
        for(size_t i=0;i<attempts && candidates.size()<o.tune_candidate_limit;++i){
            const auto d=ranked[i].d;
            if(std::none_of(candidates.begin(),candidates.end(),[&](const auto &plan){return plan.d==d;}))add(d);
        }
        return candidates;
    }
    std::vector<std::pair<double,uint64_t>> ranked;
    for(auto d:wheel_choices()){
        if(d==automatic.d)continue;
        const auto cost=wheel_cost(d,b2,bytes,o.memory_mib,o.threads);
        if(std::isfinite(cost))ranked.emplace_back(cost,d);
    }
    std::sort(ranked.begin(),ranked.end());
    const auto attempts=std::min<size_t>(4,ranked.size());
    for(size_t i=0;i<attempts && candidates.size()<3;++i){
        add(ranked[i].second);
    }
    return candidates;
}
static bool stage2_range(mpz_t factor,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &o,Stats &stats,bool allow_suffix,bool &shortened){
    stats={};const auto begin=Clock::now();
    const auto progress=[&](ecm_gw_stage2_progress_kind kind,uint64_t batch=0,uint64_t batches=0) {
        if(o.progress) {
            const ecm_gw_stage2_progress event{kind,batch,batches,stats.d,stats.poly_size,seconds(begin)};
            o.progress(o.progress_context,&event);
        }
    };
    progress(allow_suffix?ECM_GW_PLAN:ECM_GW_REPLAY);
    FTreePlan fplan{};
    auto owner=prepare_plan(n,factor,b1,b2,o,stats,fplan);
    auto &c=*owner;
    const auto d=stats.d,p=stats.poly_size;
    std::printf("gwpoly-plan: ftree=%s bytes_per_value=%zu retained_splits=%u\n",
                fplan.rebuild?"segmented":"full",static_cast<size_t>(gwnum_size(&c.gw))+64,fplan.splits);
    bool main_started=false;auto main_begin=begin;
    try {
        progress(ECM_GW_BABY);
        Points points(c);Array storage(c.gw,14);Point q{storage[0],storage[1]},step{storage[2],storage[3]},
            prev{storage[4],storage[5]},cur{storage[6],storage[7]},next{storage[8],storage[9]},base{storage[10],storage[11]};
        c.put(x,q.x);u64togw(&c.gw,1,q.z);c.put(a24,storage[12]);
        Array roots(c.gw,p);
        {
            const auto t=Clock::now();Array z(c.gw,p);
            // A stride dividing D needs only classes coprime to that stride.
            // Leave short slices on the two-step chain to avoid extra ladders.
            const auto stride=p/o.threads>=512 ? std::gcd(d,uint64_t{30}) : uint64_t{2};
            if(o.check)std::printf("gwpoly-baby: stride=%llu\n",static_cast<unsigned long long>(stride));
            if(stride==2)points.dbl(q,step,storage[12]);
            else points.ladder(q,stride,step,storage[12]);
            std::vector<uint64_t> residues;residues.reserve(p);
            std::array<std::vector<size_t>,30> lanes;
            for(uint64_t j=1;j<=d/2;j+=2)if(std::gcd(j,d)==1){
                if(stride>2)lanes[j%stride].push_back(residues.size());
                residues.push_back(j);
            }
            if(residues.size()!=p)throw std::runtime_error("Baby wheel size differs");
            parallel_points(c,p,[&](Context &worker,size_t lo,size_t hi){
                // GW add/subtract may transform source operands in place to
                // reconcile their FFT states. Each worker owns its constants.
                Points ops(worker);Array work(worker.gw,11);Point a{work[0],work[1]},b{work[2],work[3]},v{work[4],work[5]},
                    local_q{work[6],work[7]},local_step{work[8],work[9]};
                ops.copy(q,local_q);ops.copy(step,local_step);gwcopy(&worker.gw,storage[12],work[10]);
                const bool fused=worker.o.exponent && (worker.gw.cpu_flags & (CPU_AVX512F|CPU_FMA3));
                if(stride>2){
                    const std::array<Point,3> scratch{a,b,v};
                    size_t added=0;
                    for(const auto &lane:lanes){
                        auto index=std::lower_bound(lane.begin(),lane.end(),lo);
                        const auto end=std::lower_bound(index,lane.end(),hi);
                        if(index==end)continue;
                        worker.check("baby wheel chain");
                        const auto first=residues[*index],last=residues[*(end-1)];
                        Point previous=scratch[0],current{roots[*index],z[*index]};
                        ops.ladder(local_q,first,current,work[10]);
                        if(first==last)continue;
                        // The Montgomery x-coordinate does not depend on sign.
                        ops.ladder(local_q,first>stride ? first-stride : stride-first,previous,work[10]);
                        ++index;
                        // The two preceding points may still occupy scratch
                        // slots. Rotate through three to keep the output distinct.
                        size_t scratch_index=1;
                        for(uint64_t j=first+stride;index!=end;j+=stride){
                            Point next;
                            if(j==residues[*index]){
                                next={roots[*index],z[*index]};
                                ++index;
                            }else{
                                next=scratch[scratch_index];
                                scratch_index=(scratch_index+1)%scratch.size();
                            }
                            if(fused)ops.add_fused(current,local_step,previous,next);
                            else ops.add(current,local_step,previous,next);
                            previous=current;
                            current=next;
                            if(!(++added&127))worker.check("baby wheel chain");
                        }
                    }
                }else{
                    const auto first=residues[lo],last=residues[hi-1];
                    ops.ladder(local_q,first,b,work[10]);ops.ladder(local_q,first>1?first-2:1,a,work[10]);
                    size_t index=lo;
                    for(uint64_t j=first;j<=last;j+=2){
                        if(std::gcd(j,d)==1){gwcopy(&worker.gw,b.x,roots[index]);gwcopy(&worker.gw,b.z,z[index++]);}
                        if(j==1)ops.add(local_step,local_q,local_q,v);
                        else if(fused)ops.add_fused(b,local_step,a,v);
                        else ops.add(b,local_step,a,v);
                        std::swap(a,b);std::swap(b,v);
                        if((j&255)==1)worker.check();
                    }
                }
                ops.normalize(roots.v+lo,z.v+lo,hi-lo);
                if(o.check){
                    // Check the retained roots against independent ladders,
                    // including the boundaries of each lane in this slice.
                    Integer root,ref_x,ref_z;
                    const auto verify=[&](size_t sample){
                        ops.ladder(local_q,residues[sample],v,work[10]);
                        worker.get(root.v,roots[sample]);worker.get(ref_x.v,v.x);worker.get(ref_z.v,v.z);
                        mpz_mul(root.v,root.v,ref_z.v);mpz_mod(root.v,root.v,worker.n);
                        if(mpz_cmp(root.v,ref_x.v))throw std::runtime_error("Baby point differs from direct ladder");
                    };
                    if(stride==2){
                        for(const size_t sample:{lo,lo+(hi-lo)/2,hi-1})verify(sample);
                    }else{
                        for(const auto &lane:lanes){
                            const auto first=std::lower_bound(lane.begin(),lane.end(),lo);
                            const auto end=std::lower_bound(first,lane.end(),hi);
                            if(first!=end){
                                verify(*first);
                                verify(first[(end-first)/2]);
                                verify(end[-1]);
                            }
                        }
                    }
                }
            });
            stats.point_seconds+=seconds(t);
        }
        const double baby_seconds=stats.point_seconds;
        progress(ECM_GW_FTREE);
        auto init_tree_begin=Clock::now();
        auto ftree=fplan.rebuild?partial_ftree(c,roots,0,p,fplan.splits):retained_ftree(c,roots,0,p);
        struct Probe {Integer root,product;size_t index;};
        std::array<Probe,3> probes;
        if(o.check){
            for(size_t i=0;i<probes.size();++i){
                auto &probe=probes[i];probe.index=std::min<size_t>(p-1,i*(p/2));
                c.get(probe.root.v,roots[probe.index]);
                mpz_set_ui(probe.product.v,1);
            }
        }
        if(!fplan.rebuild)roots=Array();
        progress(ECM_GW_INVERSE);
        const auto inverse_begin=Clock::now();
        auto inv=c.inverse(reverse(ftree->polynomial),p);
        const double inverse_seconds=seconds(inverse_begin);
        const double init_tree_seconds=seconds(init_tree_begin);
        stats.tree_seconds+=init_tree_seconds;
        points.ladder(q,d,base,storage[12]);
        c.check("giant base ladder");
        const auto [original_first,first,last]=giant_range(b1,(b2+d/2)/d,d,p,allow_suffix);
        shortened=first>original_first;
        if(shortened)std::printf("gwpoly-range: original_first=%llu first=%llu last=%llu\n",
                                 (unsigned long long)original_first,(unsigned long long)first,(unsigned long long)last);
        // Successful baby normalization proved every Z([j]Q) with gcd(j,D)=1
        // invertible modulo the target N. Only prime divisors of D are missing
        // from that proof in the small-prime interval.
        const uint64_t small_end = std::min(b2, d / 2);
        if (b1 < small_end) {
            for_each_wheel_prime(d, [&](uint64_t prime) {
                if (prime <= b1 || prime > small_end) return;
                points.ladder(q, prime, next, storage[12]);
                c.divisor(next.z);
                c.check("small-prime ladder");
            });
        }
        const auto g_row_size=p+1;
        Array g_workspace(c.gw,g_row_size);
        stats.init_seconds=seconds(begin);main_begin=Clock::now();main_started=true;
        Array h;
        auto reciprocal=reverse(inv);
        // Wide AVX-512 coefficients benefit from avoiding a normalization
        // pass. Shorter FFTs and threaded folds keep the fused library path.
        const bool defer_low=o.threads==1 && (c.pm.cpu_flags & CPU_AVX512F) &&
                             stats.fft_length>=15360 && c.use_deferred_h_fold(p);
        progress(ECM_GW_GIANT);
        const uint64_t batches=first<=last?1+(last-first)/p:0;
        // With large coefficient FFTs even a short giant chain amortizes the
        // ladder and worker startup. Keep the smaller FFTs' existing threshold.
        const size_t giant_chunk=stats.fft_length>=65536?64:256;
        auto last_report=main_begin;
        for(uint64_t pos=first;pos<=last;){
            c.poll();const size_t count=static_cast<size_t>(std::min<uint64_t>(p,last-pos+1));
            auto part_begin=Clock::now();
            Array gx(c.gw,g_workspace.v,count),gz(c.gw,count);
            parallel_points(c,count,[&](Context &worker,size_t lo,size_t hi){
                Points ops(worker);
                Array work(worker.gw,5);
                Point previous{work[0],work[1]}, local_base{work[2],work[3]};
                Point current{gx[lo],gz[lo]};
                auto a24=work[4];
                ops.copy(base,local_base);
                gwcopy(&worker.gw,storage[12],a24);
                // A differential ladder already computes two adjacent
                // multiples; use both as the seed for the chain.
                ops.ladder(local_base,pos+lo-1,previous,a24,&current);
                const bool use_fused=worker.o.exponent &&
                    (worker.gw.cpu_flags & (CPU_AVX512F|CPU_FMA3));
                for(size_t i=lo+1;i<hi;++i){
                    Point next{gx[i],gz[i]};
                    if(pos+i==2)ops.dbl(local_base,next,a24);
                    else if(use_fused)ops.add_fused(current,local_base,previous,next);
                    else ops.add(current,local_base,previous,next);
                    previous=current;
                    current=next;
                    if(!(i&127))worker.check();
                }
                ops.normalize(gx.v+lo,gz.v+lo,hi-lo,true);
                if(o.check){
                    // Compare both ends of each lane with [iD]Q computed
                    // directly, independently of the base and the paired seed.
                    Point reference{work[0],work[1]};
                    ops.copy(q,local_base);
                    Integer negative,ref_x,ref_z;
                    for(const size_t sample:{lo,hi-1}){
                        ops.ladder(local_base,d*(pos+sample),reference,a24);
                        worker.get(negative.v,gx[sample]);
                        worker.get(ref_x.v,reference.x);
                        worker.get(ref_z.v,reference.z);
                        mpz_addmul(ref_x.v,negative.v,ref_z.v);
                        if(!mpz_divisible_p(ref_x.v,n))
                            throw std::runtime_error("Giant point differs from direct ladder");
                    }
                }
            },giant_chunk);gz=Array();stats.point_seconds+=seconds(part_begin);
            if(o.check){
                Integer negative_giant,difference;
                for(size_t i=0;i<count;++i){c.get(negative_giant.v,gx[i]);for(auto &probe:probes){
                    mpz_add(difference.v,probe.root.v,negative_giant.v);mpz_mul(probe.product.v,probe.product.v,difference.v);mpz_mod(probe.product.v,probe.product.v,n);
                }}
            }
            part_begin=Clock::now();auto g=product_tree(c,gx,0,count,&g_workspace,true);gx=Array();stats.tree_seconds+=seconds(part_begin);
            part_begin=Clock::now();
            c.fold(h,g_workspace,g->polynomial.size-1,ftree->polynomial,reciprocal,defer_low);
            stats.fold_seconds+=seconds(part_begin);
            pos+=count;++stats.blocks;stats.giants+=count;
            if(o.progress && (stats.blocks==1 || stats.blocks==batches || seconds(last_report)>=1.0)) {
                progress(ECM_GW_BATCH,stats.blocks,batches);
                last_report=Clock::now();
            }
        }
        // The descent only needs F, its reciprocal and the accumulated remainder.
        g_workspace=Array();
        if(!h.size)h=c.constant(1);
        progress(ECM_GW_DESCENT);
        auto descent_begin=Clock::now();
        // Scaled remainder: s=rev(H)*inv(rev(F)) mod X^P. Each child takes
        // middle coefficients of s*rev(sibling F), requiring no node inverses.
        // Parent and child coefficients occupy the same interval of this row.
        Array values=c.scaled_remainder(h,inv);h=Array();inv=Array();
        const double scale_seconds=seconds(descent_begin);
        const double rebuild_seconds=descend(c,*ftree,values,fplan.rebuild?&roots:nullptr);
        ftree.reset();roots=Array();
        stats.descent_seconds=seconds(descent_begin);
        if(o.check){Integer got;for(auto &probe:probes){
            c.get(got.v,values[probe.index]);if(mpz_cmp(got.v,probe.product.v))throw std::runtime_error("Stage2 polynomial differs from direct GMP products");
            ++stats.checked_values;
        }}
        progress(ECM_GW_GCD);
        const auto leaf_begin=Clock::now();
        Array product=c.constant(1);
        for(size_t i=0;i<p;++i)c.mul(product[0],values[i],product[0]);
        c.check();Integer gcd;c.get(gcd.v,product[0]);mpz_gcd(gcd.v,gcd.v,n);
        stats.main_seconds=seconds(main_begin);
        stats.phases={baby_seconds,init_tree_seconds-inverse_seconds,inverse_seconds,stats.point_seconds-baby_seconds,
                      stats.tree_seconds-init_tree_seconds,stats.fold_seconds,scale_seconds,
                      stats.descent_seconds-scale_seconds,seconds(leaf_begin)};
        const auto &phases=stats.phases;
        std::printf("gwpoly-descent: ftree_rebuild_seconds=%.6f\n",rebuild_seconds);
        std::printf("gwpoly-phases: baby=%.6f ftree=%.6f reciprocal=%.6f giant=%.6f gtree=%.6f fold=%.6f scale=%.6f descent=%.6f leaf_gcd=%.6f\n",
                    phases.baby_seconds,phases.ftree_seconds,phases.reciprocal_seconds,phases.giant_seconds,
                    phases.gtree_seconds,phases.fold_seconds,phases.scale_seconds,phases.descent_seconds,phases.leaf_gcd_seconds);
        if(mpz_cmp_ui(gcd.v,1)==0)return false;
        if(mpz_cmp(gcd.v,n)<0){mpz_set(factor,gcd.v);return true;}
        for(size_t i=0;i<p;++i)c.divisor(values[i]);
        throw Saturated("Saturated Stage2 polynomial GCD; no NF result");
    }catch(const Hit &){if(main_started)stats.main_seconds=seconds(main_begin);else stats.init_seconds=seconds(begin);c.check();return true;}
}

bool stage2(mpz_t factor,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &o,Stats &stats){
    bool shortened=false;
    try{
        const bool hit=stage2_range(factor,n,x,a24,b1,b2,o,stats,true,shortened);
        if(!shortened || !hit)return hit;
    }catch(const Saturated&){
        if(!shortened)throw;
    }
    // A suffix preserves zero support but can group factors differently.
    // Release the probe's resources and replay the original grid before
    // reporting any non-unit or factor. Arithmetic/stop errors propagate.
    std::printf("gwpoly-range: replay original grid\n");
    mpz_set_ui(factor,0);
    return stage2_range(factor,n,x,a24,b1,b2,o,stats,false,shortened);
}

static void selftest_polynomials(unsigned threads,unsigned long exponent);

void selftest(unsigned threads){
    // Independently enumerate discarded witnesses on small grids, including
    // non-unit giant denominators and both sides of the half-D boundary.
    for(uint64_t d:{6,10,30,42,210}){
        const auto p=phi(d)/2;
        for(uint64_t last=1;last<=100;++last){
            for(uint64_t original:{uint64_t(1),1+last/4}){
                const auto b1=original*d+d/2, requested=last*d, limit=200*d;
                const auto range=giant_range(b1,last,d,p);
                const auto suffix=range.first;
                if(requested>b1){
                    auto end=last;
                    while(giant_range(b1,end,d,p).size()%p)++end;
                    const auto actual=full_block_b2(b1,requested,d,p,limit);
                    if(actual<requested || (actual+d/2)/d!=end ||
                       full_block_b2(b1,actual,d,p,limit)!=actual ||
                       (giant_range(b1,end,d,p).size()+p-1)/p!=(range.size()+p-1)/p)
                        throw std::runtime_error("Invalid rounded G block geometry");
                    if(end!=last && full_block_b2(b1,requested,d,p,end*d-d/2-1)!=0)
                        throw std::runtime_error("Rounded B2 exceeds limit");
                }
                for(uint64_t i=original;i<suffix;++i){
                    for(uint64_t t=i*d-d/2+1;t<i*d+d/2;++t){
                        if(t!=i*d && std::gcd(t,d)!=1)continue;
                        bool covered=false;
                        for(uint64_t multiple=t;multiple<=(last*d+d/2);multiple+=t){
                            const auto centre=(multiple+d/2)/d;
                            const bool valid=t==i*d ? multiple%d==0 : std::gcd(multiple,d)==1;
                            if(centre>=suffix && centre<=last && valid){covered=true;break;}
                        }
                        if(!covered)throw std::runtime_error("Missing relocated giant witness");
                    }
                }
            }
        }
    }
    selftest_polynomials(threads,0);
    // (2^{697}-1) is composite and all denominators 2..510 are units.
    selftest_polynomials(threads,697);
}

static void selftest_polynomials(unsigned threads,unsigned long exponent){
    Integer n,f,x,y,z;
    if(exponent){
        mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,exponent);mpz_sub_ui(n.v,n.v,1);
    }else{
        mpz_set_ui(n.v,1000003);mpz_mul_ui(n.v,n.v,1000033);
    }
    Options o;o.threads=threads;o.exponent=exponent;
    Stats stats{};
    Context c(n.v,f.v,o,stats,exponent?1024:512);
    if(c.supports_deferred_h_fold() && !c.use_deferred_h_fold(509))
        throw std::runtime_error("Self-test carrier lacks margin for deferred H fold");
    // At 509 roots a lone tail crosses the 128-job batched normalization boundary.
    for(size_t count:{1u,2u,3u,4u,5u,8u,17u,32u,65u,129u,257u,509u}){
        Array roots(c.gw,count);std::vector<uint64_t> expected_roots(count);
        for(size_t i=0;i<count;++i){expected_roots[i]=7919*(i+1)+17;u64togw(&c.gw,expected_roots[i],roots[i]);}
        auto t=retained_ftree(c,roots,0,count);auto rf=reverse(t->polynomial);
        auto inv=c.inverse(rf,count);
        Array h(c.gw,count);for(size_t i=0;i<count;++i)u64togw(&c.gw,103*(i+3),h[i]);
        auto partial=partial_ftree(c,roots,0,count,3);
        for(size_t terms:{count,(count+1)/2}){
            auto out=c.scaled_remainder(Array(c.gw,h.v,terms),inv);
            auto rebuilt=c.copy(out);
            descend(c,*t,out);
            descend(c,*partial,rebuilt,&roots);
            for(size_t i=0;i<count;++i){
                mpz_set_ui(y.v,0);for(size_t k=terms;k-->0;){mpz_mul_ui(y.v,y.v,expected_roots[i]);mpz_add_ui(y.v,y.v,103*(k+3));mpz_mod(y.v,y.v,n.v);}
                c.get(x.v,out[i]);if(mpz_cmp(x.v,y.v))throw std::runtime_error("Scaled polynomial descent differs from GMP Horner");
                c.get(x.v,rebuilt[i]);if(mpz_cmp(x.v,y.v))throw std::runtime_error("Rebuilt F subtree differs from GMP Horner");
            }
        }
        auto product=c.convolution(h,t->polynomial);auto rem=c.remainder(product,t->polynomial,inv);
        for(size_t i=0;i<rem.size;++i){c.get(x.v,rem[i]);if(mpz_sgn(x.v))throw std::runtime_error("Polynomial remainder differs from exact multiple");}
        // Nonzero remainders across several folds, including packed G workspaces.
        Array accumulator,workspace(c.gw,count+1);
        auto reciprocal=reverse(inv);
        std::vector<std::unique_ptr<Integer>> expected;
        for(size_t i=0;i<count;++i){
            auto value=std::make_unique<Integer>();mpz_set_ui(value->v,1);
            expected.push_back(std::move(value));
        }
        for(unsigned block=0;block<4;++block){
            const size_t leaves=block<2?count:block==2?(count+1)/2:1;
            const bool signed_block=block!=1;
            if(signed_block){
                Array denominators(c.gw,leaves);
                for(size_t j=0;j<leaves;++j){
                    const auto affine=1237*(j+5)+97*block;
                    u64togw(&c.gw,(j+2)*affine,workspace[j]);
                    u64togw(&c.gw,j+2,denominators[j]);
                }
                Points(c).normalize(workspace.v,denominators.v,leaves,true);
            }else{
                for(size_t j=0;j<leaves;++j)u64togw(&c.gw,1237*(j+5)+97*block,roots[j]);
            }
            auto g=product_tree(c,signed_block?workspace:roots,0,leaves,&workspace,signed_block);
            c.fold(accumulator,workspace,g->polynomial.size-1,t->polynomial,reciprocal,c.use_deferred_h_fold(count));
            auto out=c.scaled_remainder(accumulator,inv);
            descend(c,*t,out);
            for(size_t i=0;i<count;++i){
                for(size_t j=0;j<leaves;++j){mpz_set_ui(z.v,expected_roots[i]);mpz_sub_ui(z.v,z.v,1237*(j+5)+97*block);mpz_mul(expected[i]->v,expected[i]->v,z.v);mpz_mod(expected[i]->v,expected[i]->v,n.v);}
                c.get(x.v,out[i]);if(mpz_cmp(x.v,expected[i]->v))throw std::runtime_error("Polynomial fold differs from direct GMP products");
            }
        }
    }
    c.check();
    if(exponent)
        std::printf("gwpoly selftest: M%lu Newton/remainder/scaled-Horner PASS; threads=%u deferred_low=%d maxerr=%.9g\n",
                    exponent,threads,c.use_deferred_h_fold(509),stats.max_roundoff);
    else
        std::printf("gwpoly selftest: composite-ring Newton/remainder/scaled-Horner PASS; threads=%u maxerr=%.9g\n",
                    threads,stats.max_roundoff);
}
}
