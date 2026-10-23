#include "group_order_job.h"
#include "../../service/diagnostics/order_analysis.h"
#include <filesystem>
#include <fstream>
#include <iostream>
namespace j=ecm_results::json;
namespace g=ecm_group_order;
namespace {
j::Value str(const std::string&s){return j::Value::string(s);}
j::Value boolean(bool b){j::Value v;v.scalar=b?"true":"false";return v;}
j::Value factors(const std::vector<g::PrimePower>&values){j::Value out;out.type='a';for(const auto&f:values){j::Value item;item.type='o';item.set("prime",str(f.prime.str()));item.set("exponent",j::Value::number(std::to_string(f.exponent)));item.set("primality",str(g::name(f.primality)));out.items.push_back(item);}return out;}
}
int wmain(int argc,wchar_t**argv){
    if(argc!=3)return 2;j::Value out;out.type='o';out.set("protocol",j::Value::number("1"));
    try{
        if(std::filesystem::file_size(argv[1])>65536)throw std::runtime_error("Request exceeds limit");std::ifstream in(argv[1],std::ios::binary);if(!in)throw std::runtime_error("Cannot read request");std::string raw{std::istreambuf_iterator<char>(in),{}};auto v=j::parse(raw);
        ecmgui_v2::OrderInput i{v.get("factor").text(),v.get("sigma").text(),v.get("gp").text(),std::stoi(v.get("profile").text()),std::stoi(v.get("backend").text())};std::string error;if(!ecmgui_v2::valid_order_input(i,error))throw std::runtime_error(error);
        g::CurveRequest request;request.prime=g::Integer(i.factor);request.sigma=std::stoull(i.sigma);request.family=i.profile==1?g::Family::Edwards:g::Family::Montgomery;request.parameter=i.profile==1?-1:i.profile==3?1:i.profile==5?2:i.profile==4?3:0;
        ecm_diagnostics::Settings settings;settings.backend=i.backend==1?ecm_diagnostics::Backend::Gp:i.backend==2?ecm_diagnostics::Backend::Native:ecm_diagnostics::Backend::Auto;settings.gp=i.gp.empty()?"gp":i.gp;
        const auto result=ecm_diagnostics::Session(settings).group(request);const auto&a=result.value;
        out.set("status",str(g::name(a.status)));out.set("reason",str(a.reason));out.set("backend",str(result.backend));out.set("fallback",str(result.gp_failure));out.set("all_proven",boolean(a.all_proven));out.set("modulus_primality",str(g::name(a.modulus_primality)));out.set("group_computed",boolean(a.group_order_computed));out.set("group_order",str(a.group_order.str()));out.set("point_order",str(a.point_order.str()));out.set("point_minimality",boolean(a.point_minimality_checked));out.set("group_factors",factors(a.multiple_factorization.factors));out.set("point_factors",factors(a.point_factorization));out.set("cofactor",str(a.multiple_factorization.cofactor.str()));
    }catch(const std::exception&e){out.set("error",str(e.what()));}
    std::ofstream file(argv[2],std::ios::binary|std::ios::trunc);file<<out.dump();file.close();return !file?3:out.has("error")?1:0;
}
