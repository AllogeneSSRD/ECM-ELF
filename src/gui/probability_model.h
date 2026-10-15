#pragma once
#include <string>
#include <vector>

namespace ecmgui { namespace probability_ui {
enum class Scope { Stage1, Combined, Stage2Conditional };
enum class Solve { Forward, Curves, FactorSize, B1, B2, EffectiveDivisor };
struct Parameters {
    Parameters();
    double factor_bits=130;
    double b1=110e6,b2=11e9,b2_ratio=100;
    double curves=960,effective_divisor=0;
    bool stage2=true,ratio_mode=true;
    Scope scope=Scope::Combined;
};
struct Request {
    Parameters parameters;
    Solve solve=Solve::Forward;
    double target=.95;
    bool cumulative_target=true;
    bool decimal_digits=false;
    bool auto_curves_range=true;
    double size_min=60,size_max=200,curves_max=1920;
};
struct Probabilities {
    double stage1=0,combined=0,conditional=0,selected=0;
};
struct Point { double x=0,stage1=0,combined=0,conditional=0; };
struct Result {
    Request request;
    Parameters parameters;
    Probabilities probability;
    double solved_value=0,residual=0;
    bool lower_bound_sufficient=false;
    std::string error;
    std::vector<Point> sizes,size_totals,curve_counts;
};
double bits_from_size(double value,bool decimal_digits);
double size_from_bits(double bits,bool decimal_digits);
double cumulative(double p,double curves);
double miss(double p,double curves);
double effective_b2(const Parameters &p);
Probabilities evaluate(const Parameters &p);
Result calculate(const Request &request);
std::string curve_csv(const Result &result);
} }
