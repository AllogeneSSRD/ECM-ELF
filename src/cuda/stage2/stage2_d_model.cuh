#pragma once
#include <map>
#include <cmath>
#include <algorithm>
#include <cstdio>

/* Empirical v1 ranking, calibrated on sm89 RTX4060 Laptop/M4423 with xADD6,
   resident roots/fold, warp tail and unchanged checks. Use the multiply backend's
   actual N and count partial/unbalanced tree and Newton work; seconds are estimates.
   Allocation/eviction/headroom checks remain authoritative. */
struct DPhaseRates { double baby,affine,ftree,giant,gtrees,fold,descent,inv,accum,glue; };
static constexpr DPhaseRates d_original_rates={
    3.4253945078511067e-06,2.6393747066760531e-05,1.7447305084417047e-10,
    3.7068770158438959e-07,7.758513401074988e-11,2.1353197194212299e-10,
    3.9597124012586936e-10,1.5652213657735564e-10,1.486006614455004e-06,0.0382151508709496};
// Shape-policy fit: six D curves + four fixed-D anchors; independent B2=1e11 holdout.
// GPU1 RTX4060 Laptop/sm89, M4423, current resident/check configuration, 2026-10-04.
static constexpr bool d_shape_rates_valid=true;
static constexpr DPhaseRates d_shape_rates={
    3.4275264220727666e-06,2.2556244866830929e-05,1.841123950024502e-10,
    3.7060720495287843e-07,7.6212904428508932e-11,2.1330650275351144e-10,
    4.1921666198041151e-10,1.6365593523959591e-10,1.4263335239938989e-06,
    0.031158481644502741};
// Short-reducer fit: six D curves plus four same-binary short anchors.
// Pure-convolution weights frozen before fitting; independent B2 holdout excluded.
static constexpr bool d_short_rates_valid=true;
static constexpr DPhaseRates d_short_rates={
    3.4263623776208791e-06,2.6265051185028743e-05,2.3193561077716357e-10,
    3.719097111026145e-07,7.6762604042216047e-11,2.5730236195427515e-10,
    4.622033504014322e-10,2.38168142424563e-10,1.5229012671594509e-06,
    0.035282306705375549};
// GPU baby normalization fit: six D curves plus four GPU baby anchors.
// Same frozen short-reducer NTT weights; independent B2 holdout excluded.
static constexpr bool d_baby_rates_valid=true;
static constexpr DPhaseRates d_baby_rates={
    3.3934653003722419e-06,2.1360398410290981e-06,2.1215649385492397e-10,
    3.7066724433274104e-07,7.5968462945472818e-11,2.5768004089294959e-10,
    4.5033088482375454e-10,2.4903937851620365e-10,1.5104078895629704e-06,
    0.035801215481171531};
// Fixed PTX fit: six D curves plus four fixed-D anchors, save Q verified.
// Twelve frozen convolution weights; independent B2 holdouts excluded.
static constexpr bool d_fixed_ptx_rates_valid=true;
static constexpr DPhaseRates d_fixed_ptx_rates={
    3.3879008043088555e-06,1.8697002101155637e-06,1.7275480513992264e-10,
    3.7676813705981749e-07,7.58971125562758e-11,2.6138928941619862e-10,
    4.1422432018647603e-10,2.0318074545241819e-10,1.2423239031467308e-06,
    0.088798126218851692};
// Point Montgomery fold: six D curves plus four same-binary anchors, exact save Q.
// NTT component weights remain frozen; independent B2 holdout excluded.
static constexpr bool d_point_fold_rates_valid=true;
static constexpr DPhaseRates d_point_fold_rates={
    1.6015102480881092e-06,
    1.3761475061283704e-06,
    1.8288498316723563e-10,
    1.8211322281664145e-07,
    7.6003205812267765e-11,
    2.5677359806664745e-10,
    4.2265559208268865e-10,
    2.1641769288836323e-10,
    1.0426828505678124e-06,
    0.089200741603467026};
static constexpr double d_fixed_ptx_weights[]={
    0.64179829738066763,0.62148890553636571,0.59528834315532697,
    0.56151356265523122,0.55182990504274188,0.54979932983403246,
    0.63729435005048307,0.77229265985042306,0.54521273992154429,
    0.51914554071123042,0.52326150053667642,0.54879133892829246};
static constexpr double d_short_weights[]={
    0.79292702306276752,0.73558124224990329,0.7235194676663127,
    0.63928649882602806,0.63130908855801504,0.62116086613938892,
    0.6778661412185687,0.80961735026400383,0.59482539711948279,
    0.56878426010126348,0.57227693882061159,0.61342299190083338};
// Explicit temporary payload only: coordinates/output/constants/indices/tree/mask.
// Runtime free-memory and actual allocations remain authoritative.
static unsigned long long d_baby_payload_bytes(unsigned long long p,unsigned long long w) {
    unsigned long long count=p,nodes=0;
    for(int i=0;i<8;++i){count=(count+1)/2;nodes+=count;}
    return 8*((3*p+5)*w+p+nodes*w)+count;
}
struct DPhaseModel {
    int bits;
    int profile; // 0 original, 2 shape policy, 3 short reducer, 4 GPU baby plus short reducer, 5 fixed PTX/GPU baby, 6 point Montgomery fold
    const DPhaseRates &rates;
    unsigned long long bound;
    std::map<unsigned long long,double> unit_cache,tree_cache,inverse_cache;
    DPhaseModel(int s,unsigned long long b,int selected=0):bits(s),profile(selected),
        rates(selected==6 ? d_point_fold_rates : selected==5 ? d_fixed_ptx_rates : selected==4 ? d_baby_rates : selected==3 ? d_short_rates : selected==2 ? d_shape_rates : d_original_rates),bound(b) {}
    double unit(unsigned long long p) {
        auto it=unit_cache.find(p);if(it!=unit_cache.end())return it->second;
        unsigned long long n=0;
        double work=ntt_shape_query(p,bits,&n,nullptr,nullptr,nullptr,nullptr,nullptr)
                          ? (double)n*std::log2((double)n) : 1e90;
        if(profile==3 || profile==4 || profile==5 || profile==6) {
            for(int k=16;k<=27;++k)if(n==(1ull<<k)){work*=(profile==5 || profile==6) ? d_fixed_ptx_weights[k-16] : d_short_weights[k-16];break;}
        } else if(profile==2) {
            // Frozen pure-convolution ratios for the measured deterministic table.
            // Weights change the empirical feature, never the exactness/memory shape.
            double weight=1.0;
            if(n==(1ull<<24))weight=0.9386473792217225;
            else if(n==(1ull<<25))weight=0.8672913453433596;
            else if(n==(1ull<<26))weight=0.885360571656581;
            else if(n==(1ull<<27))weight=0.9302319270266772;
            work*=weight;
        }
        unit_cache[p]=work;return work;
    }
    double tree(unsigned long long p) {
        auto it=tree_cache.find(p);if(it!=tree_cache.end())return it->second;
        double work=0;
        for(unsigned long long h=1;h<p;h*=2)
            work+=(double)((p+h)/(2*h))*unit(h+1);
        tree_cache[p]=work;return work;
    }
    double inverse(unsigned long long k) {
        auto it=inverse_cache.find(k);if(it!=inverse_cache.end())return it->second;
        double work=0;unsigned long long m=1;
        while(m<k) {m=std::min(2*m,k);work+=2*unit(m);}
        inverse_cache[k]=work;return work;
    }
    struct Cost {double init,giant,gtrees,fold,descent,inv,accum,glue,total;};
    Cost cost(unsigned long long d,unsigned long long p) {
        const auto i=bound/d+2,g=(i+p-1)/p;
        const double tw=tree(p),iw=inverse(p+1);
        Cost c{};
        c.init=rates.baby*p*std::max(1.0,std::log2((double)d)-2.0)+rates.affine*p+rates.ftree*tw;
        c.giant=rates.giant*i*(6.0+22.0*std::log2((double)bound)/64.0);
        c.gtrees=rates.gtrees*((double)(i/p)*tw+tree(i%p));
        c.fold=rates.fold*(g-1)*unit(p+1);
        c.descent=rates.descent*tw;c.inv=rates.inv*iw;c.accum=rates.accum*p;c.glue=rates.glue*g;
        c.total=c.init+c.giant+c.gtrees+c.fold+c.descent+c.inv+c.accum+c.glue;
        return c;
    }
    void print(unsigned long long d,unsigned long long p,unsigned long long owner) {
        const auto c=cost(d,p);unsigned long long nf=0,nt=0;
        ntt_shape_query(p+1,bits,&nf,nullptr,nullptr,nullptr,nullptr,nullptr);
        ntt_shape_query(ecm_stage2::tree_operand_coefficients(p),bits,&nt,nullptr,nullptr,nullptr,nullptr,nullptr);
        stage2_log::print(stage2_log::debug, "d_model_features: D=%llu P=%llu n_fold=%llu n_tree=%llu tree_work=%.0f "
                    "inverse_work=%.0f init=%.6f giant=%.6f gtrees=%.6f fold=%.6f descent=%.6f "
                    "inv=%.6f accum=%.6f glue=%.6f total=%.6f owner_bytes=%llu\n",
                    d,p,nf,nt,tree(p),inverse(p+1),c.init,c.giant,c.gtrees,c.fold,c.descent,
                    c.inv,c.accum,c.glue,c.total,owner);
    }
};
