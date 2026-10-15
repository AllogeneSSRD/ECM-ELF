#include "probability_model.h"
#include "../core/ecm_stage2_probability.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ecmgui { namespace probability_ui {
namespace {
constexpr double max_bound=1e18,max_curves=1e18;
void require(bool condition,const char *code) { if(!condition)throw std::runtime_error(code); }
bool finite_range(double x,double lo,double hi) {return std::isfinite(x)&&x>=lo&&x<=hi;}
double required_curves(double target,double p) {
    require(p>0,"unreachable");
    if(p>=1)return 1;
    const double count=std::ceil(std::log1p(-target)/std::log1p(-p));
    require(finite_range(count,1,max_curves),"unreachable");
    return count;
}
void validate(const Parameters &p) {
    require(finite_range(p.factor_bits,2,2048),"invalid_size");
    require(finite_range(p.b1,2,max_bound),"invalid_b1");
    require(finite_range(p.curves,0,max_curves),"invalid_curves");
    require(finite_range(p.effective_divisor,1e-9,max_bound),"invalid_divisor");
    if(p.stage2) {
        if(p.ratio_mode)require(finite_range(p.b2_ratio,1,1e9),"invalid_ratio");
        require(finite_range(effective_b2(p),p.b1,max_bound),"invalid_b2");
    }
}
}
Parameters::Parameters():effective_divisor(std::exp(ecm_stage2::probability::suyama_delta)){}
double bits_from_size(double value,bool digits) {
    return digits?(value-.5)*std::log2(10.)+.5:value;
}
double size_from_bits(double bits,bool digits) {
    return digits?(bits-.5)/std::log2(10.)+.5:bits;
}
double effective_b2(const Parameters &p) {
    return !p.stage2?p.b1:p.ratio_mode?p.b1*p.b2_ratio:p.b2;
}
double cumulative(double p,double curves) {
    if(!std::isfinite(p))return std::numeric_limits<double>::quiet_NaN();
    if(curves==0 || p<=0)return 0;
    if(p>=1)return 1;
    return std::clamp(-std::expm1(curves*std::log1p(-p)),0.,1.);
}
double miss(double p,double curves) {
    if(!std::isfinite(p))return std::numeric_limits<double>::quiet_NaN();
    if(curves==0 || p<=0)return 1;
    if(p>=1)return 0;
    return std::exp(curves*std::log1p(-p));
}
Probabilities evaluate(const Parameters &p) {
    validate(p);
    Probabilities out;
    const double delta=std::log(p.effective_divisor);
    out.stage1=ecm_stage2::probability::success(p.b1,p.b1,p.factor_bits,delta);
    out.combined=p.stage2?ecm_stage2::probability::success(p.b1,effective_b2(p),p.factor_bits,delta):out.stage1;
    out.conditional=out.stage1<1?std::clamp((out.combined-out.stage1)/(1-out.stage1),0.,1.):std::numeric_limits<double>::quiet_NaN();
    out.selected=p.scope==Scope::Stage1?out.stage1:p.scope==Scope::Combined?out.combined:out.conditional;
    require(std::isfinite(out.selected),"undefined_conditional");
    return out;
}
Result calculate(const Request &request) {
    Result out;out.request=request;out.parameters=request.parameters;
    try {
        auto &p=out.parameters;
        validate(p);
        require(finite_range(bits_from_size(request.size_min,request.decimal_digits),2,2048)&&
                finite_range(bits_from_size(request.size_max,request.decimal_digits),2,2048)&&
                request.size_min<request.size_max,"invalid_plot_range");
        require(finite_range(request.curves_max,1,max_curves),"invalid_plot_range");
        if(request.solve!=Solve::Forward) {
            require(std::isfinite(request.target) && request.target>0 && request.target<1,"invalid_target");
            if(request.solve==Solve::Curves) {
                require(request.cumulative_target,"curves_need_cumulative");
                p.curves=required_curves(request.target,evaluate(p).selected);
                out.solved_value=p.curves;
            } else {
                if(request.cumulative_target)require(p.curves>=1,"invalid_curves");
                const double target=request.cumulative_target?
                    -std::expm1(std::log1p(-request.target)/p.curves):request.target;
                require(target>0,"unreachable");
                require(!(p.scope==Scope::Stage2Conditional && request.solve==Solve::B1),"conditional_b1");
                double lo=0,hi=0;
                const bool decreasing=request.solve==Solve::FactorSize;
                switch(request.solve) {
                case Solve::FactorSize:lo=2;hi=2048;break;
                case Solve::B1:
                    lo=2;hi=!p.stage2?max_bound:p.ratio_mode?max_bound/p.b2_ratio:p.b2;break;
                case Solve::B2:
                    require(p.stage2,"b2_needs_stage2");p.ratio_mode=false;lo=p.b1;hi=max_bound;break;
                case Solve::EffectiveDivisor:lo=1e-9;hi=max_bound;break;
                default:throw std::runtime_error("invalid_solve");
                }
                if(p.scope==Scope::Stage2Conditional) {
                    const double log_order=(p.factor_bits-.5)*std::log(2.);
                    if(request.solve==Solve::FactorSize)
                        lo=std::max(lo,(std::log(p.b1)+std::log(p.effective_divisor))/std::log(2.)+.5+1e-7);
                    if(request.solve==Solve::EffectiveDivisor) {
                        const double log_limit=log_order-std::log(p.b1);
                        if(log_limit<std::log(hi))hi=std::exp(log_limit)*(1-1e-7);
                    }
                }
                require(lo<=hi,"unreachable");
                const auto assign=[&](double value) {
                    if(request.solve==Solve::FactorSize)p.factor_bits=value;
                    else if(request.solve==Solve::B1)p.b1=value;
                    else if(request.solve==Solve::B2)p.b2=value;
                    else p.effective_divisor=value;
                };
                const auto probe=[&](double value){assign(value);return evaluate(p).selected;};
                const double low_p=probe(lo),high_p=probe(hi);
                if(!decreasing && low_p>=target)out.lower_bound_sufficient=true;
                else {
                    require(decreasing?low_p>=target && high_p<=target:low_p<=target && high_p>=target,"unreachable");
                    // Log-space search for bounds/divisor covers the wide numeric range.
                    for(int iteration=0;iteration<80;++iteration) {
                        const double mid=decreasing?(lo+hi)*.5:std::exp((std::log(lo)+std::log(hi))*.5);
                        const double value=probe(mid);
                        if(decreasing?value>=target:value<target)lo=mid;else hi=mid;
                    }
                }
                assign(out.lower_bound_sufficient?lo:decreasing?lo:hi);
                // B1/B2 are integral in a worker command. Publish a sufficient bound.
                if(request.solve==Solve::B1)p.b1=std::ceil(p.b1);
                if(request.solve==Solve::B2)p.b2=std::ceil(p.b2);
                out.solved_value=request.solve==Solve::FactorSize?p.factor_bits:
                    request.solve==Solve::B1?p.b1:request.solve==Solve::B2?p.b2:p.effective_divisor;
            }
        }
        out.probability=evaluate(p);
        if(request.solve!=Solve::Forward) {
            const double achieved=request.cumulative_target?cumulative(out.probability.selected,p.curves):out.probability.selected;
            out.residual=std::abs(achieved-request.target);
        }
        // Graphs deliberately compare Stage1 and combined probability, regardless
        // of the inverse scope. A conditional probability needs a different cohort.
        auto plot_parameters=p;plot_parameters.scope=Scope::Combined;
        constexpr int samples=129;
        const double curves_upper=request.auto_curves_range?std::max(request.curves_max,std::min(max_curves,p.curves*2)):request.curves_max;
        for(int i=0;i<samples;++i) {
            const double t=double(i)/(samples-1);
            const double x=request.size_min+(request.size_max-request.size_min)*t;
            plot_parameters.factor_bits=bits_from_size(x,request.decimal_digits);
            const auto probabilities=evaluate(plot_parameters);
            out.sizes.push_back({x,probabilities.stage1,probabilities.combined,probabilities.conditional});
            out.size_totals.push_back({x,cumulative(probabilities.stage1,p.curves),cumulative(probabilities.combined,p.curves),cumulative(probabilities.conditional,p.curves)});
            const double count=curves_upper*t;
            out.curve_counts.push_back({count,cumulative(out.probability.stage1,count),cumulative(out.probability.combined,count),cumulative(out.probability.conditional,count)});
        }
    } catch(const std::exception &error){out.error=error.what();out.sizes.clear();out.size_totals.clear();out.curve_counts.clear();}
    return out;
}
std::string curve_csv(const Result &result) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"# model=local_dickman_semismooth; B1="<<result.parameters.b1<<"; B2="<<effective_b2(result.parameters)
       <<"; factor_bits="<<result.parameters.factor_bits<<"; curves="<<result.parameters.curves
       <<"; D_eff="<<result.parameters.effective_divisor<<"\n";
    out<<"plot,x_unit,x,stage1_probability,combined_probability,conditional_probability\n";
    const auto emit=[&](const Point &point) {
        out<<point.x<<','<<point.stage1<<','<<point.combined<<',';
        if(std::isfinite(point.conditional))out<<point.conditional;
        out<<'\n';
    };
    for(const auto &point:result.sizes){out<<"factor_size,"<<(result.request.decimal_digits?"digits":"bits")<<',';emit(point);}
    for(const auto &point:result.size_totals){out<<"factor_size_cumulative,"<<(result.request.decimal_digits?"digits":"bits")<<',';emit(point);}
    for(const auto &point:result.curve_counts){out<<"cumulative,curves,";emit(point);}
    return out.str();
}
} }
