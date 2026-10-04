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
struct DPhaseModel {
    int bits;
    bool shape_ntt;
    const DPhaseRates &rates;
    unsigned long long bound;
    std::map<unsigned long long,double> unit_cache,tree_cache,inverse_cache;
    DPhaseModel(int s,unsigned long long b,bool shape=false):bits(s),shape_ntt(shape),
        rates(shape ? d_shape_rates : d_original_rates),bound(b) {}
    double unit(unsigned long long p) {
        auto it=unit_cache.find(p);if(it!=unit_cache.end())return it->second;
        unsigned long long n=0;
        double work=ntt_shape_query(p,bits,&n,nullptr,nullptr,nullptr,nullptr,nullptr)
                          ? (double)n*std::log2((double)n) : 1e90;
        if(shape_ntt) {
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
        const auto c=cost(d,p);unsigned long long nf=0,nt=0,h=1;
        while(2*h<p)h*=2;
        ntt_shape_query(p+1,bits,&nf,nullptr,nullptr,nullptr,nullptr,nullptr);
        ntt_shape_query(h+1,bits,&nt,nullptr,nullptr,nullptr,nullptr,nullptr);
        std::printf("d_model_features: D=%llu P=%llu n_fold=%llu n_tree=%llu tree_work=%.0f "
                    "inverse_work=%.0f init=%.6f giant=%.6f gtrees=%.6f fold=%.6f descent=%.6f "
                    "inv=%.6f accum=%.6f glue=%.6f total=%.6f owner_bytes=%llu\n",
                    d,p,nf,nt,tree(p),inverse(p+1),c.init,c.giant,c.gtrees,c.fold,c.descent,
                    c.inv,c.accum,c.glue,c.total,owner);
    }
};

