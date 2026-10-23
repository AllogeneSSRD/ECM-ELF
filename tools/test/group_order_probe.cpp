#include "ecm/core/group_order/group_order.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace ecm_group_order;
namespace {
void require(bool condition,const char*message){if(!condition)throw std::runtime_error(message);}
std::string quoted(const std::string&s){
    std::string out="\"";for(unsigned char c:s){
        if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c=='\n')out+="\\n";
        else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c<32)out+='?';else out+=char(c);
    }return out+'"';
}
std::vector<std::string> fields(const std::string&s,char delimiter){
    std::vector<std::string> out;size_t begin=0;
    for(;;){size_t end=s.find(delimiter,begin);out.push_back(s.substr(begin,end-begin));if(end==std::string::npos)break;begin=end+1;}return out;
}
uint64_t number(const std::string&s){
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid uint64");
    size_t end=0;auto n=std::stoull(s,&end);if(end!=s.size())throw std::invalid_argument("invalid uint64");return n;
}
uint32_t number32(const std::string&s){auto n=number(s);if(n>UINT32_MAX)throw std::invalid_argument("invalid uint32");return uint32_t(n);}
std::vector<PrimePower> parse_parts(const std::string&s){
    std::vector<PrimePower> out;if(s.empty())return out;
    for(const auto&token:fields(s,',')){auto pair=fields(token,':');if(pair.size()!=2)throw std::invalid_argument("expected prime:exponent");out.push_back({Integer(pair[0]),number32(pair[1]),Primality::Probable});}return out;
}
void parts_json(const std::vector<PrimePower>&parts){
    std::cout<<'[';bool first=true;for(const auto&f:parts){if(!first)std::cout<<',';first=false;
        std::cout<<"["<<quoted(f.prime.str())<<','<<f.exponent<<','<<quoted(name(f.primality))<<"]";
    }std::cout<<']';
}
void stats_json(const Statistics&s){
    std::cout<<"{\"total_ms\":"<<s.total_ms<<",\"construction_ms\":"<<s.construction_ms
        <<",\"factor_ms\":"<<s.factor_ms<<",\"count_ms\":"<<s.count_ms<<",\"point_ms\":"<<s.point_ms
        <<",\"rho_iterations\":"<<s.rho_iterations<<",\"baby_steps\":"<<s.baby_steps
        <<",\"giant_steps\":"<<s.giant_steps<<",\"point_additions\":"<<s.point_additions
        <<",\"sampled_points\":"<<s.sampled_points<<'}';
}
void factor_json(const Factorization&f){
    std::cout<<"{\"status\":"<<quoted(name(f.status))<<",\"reason\":"<<quoted(f.reason)
        <<",\"cofactor\":"<<quoted(f.cofactor.str())<<",\"all_proven\":"<<(f.all_proven?"true":"false")<<",\"factors\":";
    parts_json(f.factors);std::cout<<",\"statistics\":";stats_json(f.statistics);std::cout<<'}';
}
void analysis_json(const Analysis&a){
    std::cout<<"{\"status\":"<<quoted(name(a.status))<<",\"reason\":"<<quoted(a.reason)
        <<",\"modulus_primality\":"<<quoted(name(a.modulus_primality))
        <<",\"group_order_computed\":"<<(a.group_order_computed?"true":"false")
        <<",\"group_order\":"<<quoted(a.group_order.str())<<",\"multiple\":"<<quoted(a.multiple.str())
        <<",\"point_order\":"<<quoted(a.point_order.str())
        <<",\"point_minimality_checked\":"<<(a.point_minimality_checked?"true":"false")
        <<",\"all_proven\":"<<(a.all_proven?"true":"false")<<",\"multiple_factorization\":";
    factor_json(a.multiple_factorization);std::cout<<",\"point_factorization\":";parts_json(a.point_factorization);
    std::cout<<",\"statistics\":";stats_json(a.statistics);std::cout<<'}';
}
void self_tests(){
    require(Integer(UINT64_MAX).str()=="18446744073709551615","uint64 construction truncated");
    Integer copy(7);Integer moved(std::move(copy));copy=Integer(11);require(moved.str()=="7"&&copy.str()=="11","integer ownership");
    for(const char*n:{"0","1","4","561","3215031751","3825123056546413051","18446744073709551615"})
        require(classify_prime(Integer(n))==Primality::Composite,"composite primality classification");
    require(classify_prime(Integer("18446744073709551557"))==Primality::Proven,"uint64 prime proof");
    require(classify_prime(Integer("170141183460469231731687303715884105727"))==Primality::Probable,"probable is not proven");
    Limits limits;limits.milliseconds=3000;
    for(const char*n:{"1","34012224","1030303027","1000036000099","59649589504338456"}){
        auto f=factor_integer(Integer(n),limits);require(f.status==Status::Complete,"integer factorization incomplete");
        Integer product(1);for(const auto&p:f.factors){Integer v;mpz_pow_ui(v.v,p.prime.v,p.exponent);mpz_mul(product.v,product.v,v.v);}
        require(product.str()==n && f.cofactor.str()=="1","factor product");
    }
    require(factor_integer(Integer(0)).status==Status::InvalidInput,"zero factorization accepted");
    Limits stopped=limits;stopped.rho_iterations=0;stopped.trial_limit=2;
    auto partial=factor_integer(Integer("2000072000198"),stopped);
    require(partial.status==Status::Partial && partial.cofactor.str()=="1000036000099" && partial.factors.size()==1,"partial cofactor lost");
    CurveRequest r;r.prime=Integer("59649589127497217");r.sigma=26;
    auto a=analyze(r,limits);require(a.status==Status::Complete && a.group_order_computed && a.all_proven,"BSGS count failed");
    require(a.group_order.str()=="59649589504338456" && a.point_order.str()=="14912397376084614","BSGS reference mismatch");
    Integer multiple("59649589504338456");auto supplied=analyze(r,limits,&multiple,&a.multiple_factorization.factors);
    require(supplied.status==Status::Complete && !supplied.group_order_computed && supplied.group_order.str()=="0","supplied multiple relabelled group order");
    require(supplied.point_order.str()==a.point_order.str(),"supplied factor point order");
    auto wrong=a.multiple_factorization.factors;wrong[0].exponent++;
    require(analyze(r,limits,&multiple,&wrong).status==Status::InvalidInput,"wrong supplied product accepted");
    std::vector<PrimePower> composite={{Integer(4),1,Primality::Proven}};Integer four(4);
    require(analyze(r,limits,&four,&composite).status==Status::InvalidInput,"forged factor proof accepted");
    require(analyze(r,limits,&four).status==Status::InvalidInput,"false annihilator accepted");
    auto capped=limits;capped.baby_steps=2;
    require(analyze(r,capped).status==Status::ResourceLimit,"table budget ignored");
    r.prime=Integer(91);require(analyze(r).status==Status::InvalidInput,"composite modulus accepted");
    r.prime=Integer(101);r.sigma=0;require(analyze(r).status==Status::InvalidInput,"zero sigma accepted");
    r.sigma=UINT64_MAX;r.family=Family::Edwards;r.parameter=0;
    require(analyze(r).status==Status::InvalidInput,"Edwards parameter accepted");
    Curve c;c.prime=Integer(101);c.a=Integer(100);c.b=Integer(0);c.point.infinity=false;c.point.x=Integer(0);c.point.y=Integer(0);
    auto noncyclic=analyze_curve(c,limits);
    require(noncyclic.status==Status::Complete && noncyclic.group_order.str()=="104" && noncyclic.point_order.str()=="2","noncyclic curve count");
    c.point.infinity=true;auto infinity=analyze_curve(c,limits);require(infinity.status==Status::Complete&&infinity.point_order.str()=="1","infinity order");
    c.prime=Integer(10007);c.a=Integer(10006);c.point.infinity=false;
    auto mestre=analyze_curve(c,limits);
    require(mestre.status==Status::Complete&&mestre.group_order.str()=="10008"&&mestre.point_order.str()=="2","Mestre low-order starting point");
    c.a=Integer(0);require(analyze_curve(c,limits).status==Status::InvalidInput,"singular curve accepted");
    std::cout<<"{\"self_tests\":\"passed\",\"gmp\":"<<quoted(gmp_version)<<"}\n";
}
void request(const std::string&line){
    const auto f=fields(line,'\t');Limits limits;
    if(f[0]=="FACTOR"){
        if(f.size()!=5)throw std::invalid_argument("FACTOR expects n, milliseconds, rho iterations, trial limit");
        limits.milliseconds=number(f[2]);limits.rho_iterations=number(f[3]);limits.trial_limit=number32(f[4]);
        factor_json(factor_integer(Integer(f[1]),limits));return;
    }
    if(f[0]!="ANALYZE" || f.size()!=13)throw std::invalid_argument("ANALYZE expects family,param,sigma,p,multiple,factors,ms,baby,rho,points,enumeration");
    CurveRequest r;
    if(f[1]=="mont")r.family=Family::Montgomery;else if(f[1]=="edwards")r.family=Family::Edwards;else throw std::invalid_argument("unknown curve family");
    if(f[2]!="-1" && number(f[2])>3)throw std::invalid_argument("parameter must be -1 or 0..3");
    r.parameter=f[2]=="-1"?-1:int(number32(f[2]));r.sigma=number(f[3]);r.prime=Integer(f[4]);
    Integer M;auto factors=parse_parts(f[6]);
    limits.milliseconds=number(f[7]);limits.baby_steps=number(f[8]);limits.rho_iterations=number(f[9]);
    limits.random_points=number32(f[10]);limits.enumeration_limit=number32(f[11]);limits.trial_limit=number32(f[12]);
    if(f[5].empty()){
        if(!f[6].empty())throw std::invalid_argument("factorization requires a known multiple");
        analysis_json(analyze(r,limits));
    }else{M=Integer(f[5]);analysis_json(analyze(r,limits,&M,f[6].empty()?nullptr:&factors));}
}
}
int main(int argc,char**argv){
    std::cout<<std::setprecision(12);
    try{
        if(argc==2 && std::string(argv[1])=="--self-test"){self_tests();return 0;}
        if(argc!=2 || std::string(argv[1])!="--stream")throw std::invalid_argument("usage: group_order_probe --self-test | --stream");
        std::string line;while(std::getline(std::cin,line)){
            if(!line.empty()&&line.back()=='\r')line.pop_back();
            try{request(line);}catch(const std::exception&e){std::cout<<"{\"status\":\"invalid_input\",\"reason\":"<<quoted(e.what())<<'}';}
            std::cout<<'\n'<<std::flush;
        }return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
