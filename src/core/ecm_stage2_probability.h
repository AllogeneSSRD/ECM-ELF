#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// Numerical reference: tools/ecm_prob/rho.py and GMP-ECM rho.c (Kruppa's
// local Dickman/semismooth model). The input size is a sought PRIME FACTOR,
// never the bit width of the entire number being factored. No Brent-Suyama
// extension is credited to the current polynomial Stage2 implementation.
namespace ecm_stage2 { namespace probability {
constexpr double suyama_delta=3.134;
constexpr double euler=0.577215664901532861;
constexpr double one_minus_euler=0.422784335098467139;
constexpr double pi2_over6=1.644934066848226436;
inline double dilog_series(double z) {
    double sum=0,power=z;
    for(int k=1;k<45;++k){sum+=power/(double(k)*k);power*=z;}
    return sum;
}
inline double dilog(double x) {
    if(x<=-2){const double l=std::log(-1/x);return -dilog_series(1/x)-pi2_over6-l*l/2;}
    const double l=std::log(1-x);
    return dilog_series(1/(1-x))-pi2_over6+l*(l/2-std::log(-x));
}
inline double rho_exact(double x) {
    if(x<=0)return 0;
    if(x<=1)return 1;
    if(x<=2)return 1-std::log(x);
    return 1-std::log(x)*(1-std::log(x-1))+dilog(1-x)+pi2_over6/2;
}
class Rho {
    static constexpr int invh=256,limit=2560;
    std::array<double,limit> values{};
public:
    Rho() {
        for(int i=0;i<3*invh;++i)values[i]=rho_exact(double(i)/invh);
        for(int i=3*invh;i<limit;++i)values[i]=std::max(0.,values[i-4]-2./45*(
            7*values[i-invh-4]/(i-4)+32*values[i-invh-3]/(i-3)+
            12*values[i-invh-2]/(i-2)+32*values[i-invh-1]/(i-1)+7*values[i-invh]/i));
    }
    double rho(double alpha)const {
        if(alpha<=3)return rho_exact(alpha);
        if(alpha>=10)return 0;
        const double index=alpha*invh;const int a=int(std::floor(index));
        const double next=a+1<limit?values[a+1]:0;
        return values[a]+(next-values[a])*(index-a);
    }
    double local(double alpha,double log_x)const {
        if(alpha<=1)return rho_exact(alpha);
        if(alpha>=10)return 0;
        return rho(alpha)-euler*rho(alpha-1)/log_x;
    }
    double local_i(int i,double log_x)const {
        if(i<=0 || i>=limit)return 0;
        if(i<=invh)return 1;
        if(i<=2*invh)return values[i]-euler/log_x;
        return values[i]-(euler*values[i-invh]+one_minus_euler*values[i-2*invh]/log_x)/log_x;
    }
    double mu(double alpha,double beta,double log_x)const {
        // A prime beyond the effective group-order size contributes nothing.
        beta=std::min(beta,alpha);
        constexpr double h=1./invh;
        const int ai=std::max(0,std::min(limit,int(std::ceil((alpha-beta)*invh))));
        const int bi=std::max(0,std::min(limit,int(std::floor((alpha-1)*invh))));
        const double a=ai*h,b=bi*h;
        double sum=0;
        for(int i=ai+1;i<bi;++i)sum+=local_i(i,log_x)/(alpha-i*h);
        sum+=.5*local_i(ai,log_x)/(alpha-a)+.5*local_i(bi,log_x)/(alpha-b);
        sum*=h;
        sum+=(a-alpha+beta)*.5*(local_i(ai,log_x)/(alpha-a)+local(alpha-beta,log_x)/beta);
        sum+=(alpha-1-b)*.5*(local(alpha-1,log_x)+local_i(bi,log_x)/(alpha-b));
        return sum;
    }
};
inline const Rho &table(){static const Rho instance;return instance;}
inline const std::vector<unsigned> &small_primes() {
    static const std::vector<unsigned> primes=[] {
        std::array<bool,20001> composite{};std::vector<unsigned> p;
        for(unsigned i=2;i<=20000;++i)if(!composite[i]) {
            p.push_back(i);if(i<=20000/i)for(unsigned j=i*i;j<=20000;j+=i)composite[j]=true;
        }
        return p;
    }();return primes;
}
inline double success(double b1,double b2,double factor_bits,double delta=suyama_delta) {
    if(!std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(factor_bits) ||
       !std::isfinite(delta) || b1<2 || b2<b1 || factor_bits<2)
        throw std::runtime_error("invalid ECM probability request");
    const double log_x=(factor_bits-.5)*std::log(2.)-delta,log_b1=std::log(b1);
    if(log_x<=log_b1)return 1;
    const double alpha=log_x/log_b1;
    double result=table().local(alpha,log_x),beta=1;
    if(b2>b1) {
        if(b1<20000) {
            const double upper=std::min(b2,20000.);
            for(unsigned p:small_primes())if(p>b1 && p<=upper) {
                const double l=std::log(double(p));result+=table().local((log_x-l)/log_b1,log_x-l)/p;
            }
            beta=std::log(b2)/std::log(upper);
        } else beta=std::log(b2)/log_b1;
        if(beta>1)result+=table().mu(alpha,beta,log_x);
    }
    return std::max(0.,std::min(1.,result));
}
// Inverse of model.py's GMP recommendation regression, used as a heuristic
// default, not as evidence that the actual unknown factor has this size.
inline double recommended_factor_bits(double b1,unsigned target_bits) {
    if(!std::isfinite(b1) || b1<2 || target_bits<4)throw std::runtime_error("invalid factor-size recommendation");
    return std::max(2.,std::min(double(target_bits)/2,std::max(10.,(std::log(b1)-5.332)/.075)));
}
} }
