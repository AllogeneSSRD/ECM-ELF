#include "probability_model.h"
#include "ecm/stage2/probability.h"
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

namespace {
constexpr double min_level=2,max_level=2048;
double curve_hazard(const Parameters &batch,double bits) {
    const double probability=ecm_stage2::probability::success(batch.b1,effective_b2(batch),bits,std::log(batch.effective_divisor));
    return -std::log1p(-probability);
}
double batch_hazard(const Parameters &batch,double bits) {
    return batch.curves>0?batch.curves*curve_hazard(batch,bits):0;
}
void validate_t_batch(Parameters batch) {
    batch.factor_bits=min_level;
    validate(batch);
    require(std::floor(batch.curves)==batch.curves,"invalid_curves");
    require(std::floor(batch.b1)==batch.b1,"invalid_b1");
    if(batch.stage2&&!batch.ratio_mode)require(std::floor(batch.b2)==batch.b2,"invalid_b2");
}
double t_level_max_b1(const Parameters &batch) {
    if(!batch.stage2)return max_bound;
    if(!batch.ratio_mode)return batch.b2;
    double bound=std::floor(max_bound/batch.b2_ratio);
    if(bound*batch.b2_ratio>max_bound)bound=std::floor(std::nextafter(bound,0.));
    return bound;
}
template<class Hazard>
std::optional<double> hazard_level(const Hazard &hazard) {
    double lo=min_level,hi=max_level;
    if(hazard(lo)<1)return std::nullopt;
    for(int iteration=0;iteration<52;++iteration) {
        const double mid=(lo+hi)*.5;
        if(hazard(mid)>=1)lo=mid;else hi=mid;
    }
    return lo;
}
}

TLevelResult calculate_t_level(const TLevelRequest &request) {
    TLevelResult out;out.request=request;out.batch=request.batch;
    try {
        auto &batch=out.batch;
        validate_t_batch(batch);
        require(request.history.size()<=max_t_level_batches,"history_limit");
        for(const auto &previous:request.history)validate_t_batch(previous);
        const bool use_estimate=request.starting_level!=0;
        double reference=0;
        if(use_estimate) {
            require(request.history.empty()&&finite_range(request.starting_level,min_level,max_level),"invalid_starting_level");
            reference=curve_hazard(request.batch,request.starting_level);
            require(reference>0&&std::isfinite(reference),"unreachable_starting_level");
        }
        const auto previous_hazard=[&](double bits) {
            if(use_estimate)return curve_hazard(request.batch,bits)/reference;
            double total=0;
            for(const auto &previous:request.history)total+=batch_hazard(previous,bits);
            return total;
        };
        out.previous_level=use_estimate?std::optional<double>(request.starting_level):hazard_level(previous_hazard);
        if(request.solve!=TLevelSolve::Forward) {
            require(!request.relative_target||out.previous_level.has_value(),"missing_starting_level");
            require(finite_range(request.target,request.relative_target?0:min_level,max_level),"invalid_t_level_target");
            const double target=request.target+(request.relative_target?*out.previous_level:0);
            require(finite_range(target,min_level,max_level),"invalid_t_level_target");
            out.target_level=target;
            const double remaining=1-previous_hazard(target);
            if(remaining<=0||(out.previous_level&&target<=*out.previous_level)) {
                batch.curves=0;out.already_sufficient=true;
            } else if(request.solve==TLevelSolve::Curves||request.solve==TLevelSolve::Recommend) {
                if(request.solve==TLevelSolve::Recommend) {
                    // Invert the existing GMP B1 recommendation regression at the target factor size.
                    batch.b1=std::min(std::ceil(std::exp(.075*target+5.332)),t_level_max_b1(batch));
                }
                const double per_curve=curve_hazard(batch,target);
                require(per_curve>0,"unreachable");
                batch.curves=std::max(1.,std::ceil(remaining/per_curve));
                if(batch.curves*per_curve<remaining)
                    batch.curves=std::max(batch.curves+1,std::nextafter(batch.curves,std::numeric_limits<double>::infinity()));
                require(finite_range(batch.curves,1,max_curves),"unreachable");
            } else {
                const bool solve_b1=request.solve==TLevelSolve::B1;
                require(solve_b1||request.solve==TLevelSolve::B2,"invalid_solve");
                if(!solve_b1)require(batch.stage2,"b2_needs_stage2");
                require(batch.curves>=1,"invalid_curves");
                if(!solve_b1)batch.ratio_mode=false;
                double &bound=solve_b1?batch.b1:batch.b2;
                double lo=solve_b1?2:batch.b1;
                double hi=solve_b1?t_level_max_b1(batch):max_bound;
                bound=hi;
                require(batch_hazard(batch,target)>=remaining,"unreachable");
                while(lo<hi) {
                    const double mid=std::floor(lo+(hi-lo)*.5);
                    if(mid>=hi)break;
                    bound=mid;
                    if(batch_hazard(batch,target)>=remaining)hi=mid;
                    else {
                        const double next=std::ceil(std::nextafter(mid,hi));
                        if(next<=lo)break;
                        lo=next;
                    }
                }
                bound=hi;
            }
        }
        const auto new_hazard=[&](double bits){return batch_hazard(batch,bits);};
        const auto total_hazard=[&](double bits){return previous_hazard(bits)+new_hazard(bits);};
        out.batch_level=hazard_level(new_hazard);
        out.total_level=hazard_level(total_hazard);
        double lo=std::min({out.previous_level.value_or(max_level),out.total_level.value_or(max_level),out.target_level?out.target_level:max_level});
        double hi=std::max({out.previous_level.value_or(min_level),out.total_level.value_or(min_level),out.target_level});
        if(lo>hi){lo=60;hi=200;}
        lo=std::max(min_level,lo-20);hi=std::min(max_level,hi+20);
        constexpr int samples=129;
        for(int i=0;i<samples;++i) {
            const double bits=lo+(hi-lo)*i/(samples-1);
            out.points.push_back({bits,std::exp(-previous_hazard(bits)),std::exp(-total_hazard(bits))});
        }
    } catch(const std::exception &error) {out.error=error.what();out.points.clear();}
    return out;
}

std::string t_level_csv(const TLevelResult &result) {
    std::ostringstream out;out<<std::setprecision(17);
    const auto write_batch=[&](const char *label,const Parameters &batch) {
        out<<"# "<<label<<"; curves="<<batch.curves<<"; B1="<<batch.b1<<"; B2="<<effective_b2(batch)
           <<"; D_eff="<<batch.effective_divisor<<"\n";
    };
    out<<"# model=local_dickman_semismooth; t_level_unit=bits; miss_threshold="<<std::exp(-1.)<<"\n";
    if(result.request.starting_level) {
        out<<"# estimated_prior_t_level="<<result.request.starting_level<<"\n";
        write_batch("estimate_reference",result.request.batch);
    }
    for(const auto &batch:result.request.history)write_batch("history",batch);
    write_batch("new_batch",result.batch);
    out<<"# previous_t_level=";if(result.previous_level)out<<*result.previous_level;
    out<<"; batch_t_level=";if(result.batch_level)out<<*result.batch_level;
    out<<"; total_t_level=";if(result.total_level)out<<*result.total_level;
    out<<"\nfactor_bits,previous_miss,total_miss\n";
    for(const auto &point:result.points)out<<point.bits<<','<<point.previous_miss<<','<<point.total_miss<<'\n';
    return out.str();
}
} }
