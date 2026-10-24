#include "group_order_job.h"
#include "group_order_result.h"
#include "../../service/diagnostics/order_analysis.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <algorithm>
namespace j=ecm_results::json;
namespace g=ecm_group_order;
namespace {
j::Value str(const std::string&s){return j::Value::string(s);}
j::Value boolean(bool b){j::Value v;v.scalar=b?"true":"false";return v;}
j::Value factors(const std::vector<g::PrimePower>&values){j::Value out;out.type='a';for(const auto&f:values){j::Value item;item.type='o';item.set("prime",str(f.prime.str()));item.set("exponent",j::Value::number(std::to_string(f.exponent)));item.set("primality",str(g::name(f.primality)));out.items.push_back(item);}return out;}
j::Value analysis(const ecm_diagnostics::GroupResult& result){
    const auto&a=result.value;j::Value out;out.type='o';out.set("protocol",j::Value::number("1"));
    out.set("native_engine",str(result.backend=="pari"?"pari":result.backend=="gmp"?"gmp":""));out.set("native_failure",str(result.pari_failure));out.set("status",str(g::name(a.status)));out.set("reason",str(a.reason));out.set("backend",str(result.backend));out.set("fallback",str(result.gp_failure));out.set("all_proven",boolean(a.all_proven));out.set("modulus_primality",str(g::name(a.modulus_primality)));out.set("group_computed",boolean(a.group_order_computed));out.set("group_order",str(a.group_order.str()));out.set("point_order",str(a.point_order.str()));out.set("point_minimality",boolean(a.point_minimality_checked));out.set("group_factors",factors(a.multiple_factorization.factors));out.set("point_factors",factors(a.point_factorization));out.set("cofactor",str(a.multiple_factorization.cofactor.str()));return out;
}
void save(const std::filesystem::path&path,const j::Value&value){
    auto next=path;next+=L".next";std::ofstream file(next,std::ios::binary|std::ios::trunc);file<<value.dump();file.close();
    if(!file||!MoveFileExW(next.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot publish analysis checkpoint");
}

}
int wmain(int argc,wchar_t**argv){
    if(argc!=3)return 2;j::Value out;out.type='o';out.set("protocol",j::Value::number("1"));
    try{
        if(std::filesystem::file_size(argv[1])>65536)throw std::runtime_error("Request exceeds limit");std::ifstream in(argv[1],std::ios::binary);if(!in)throw std::runtime_error("Cannot read request");std::string raw{std::istreambuf_iterator<char>(in),{}};auto v=j::parse(raw);
        ecmgui_v2::OrderInput i{v.get("factor").text(),v.get("sigma").text(),v.get("gp").text(),std::stoi(v.get("profile").text()),std::stoi(v.get("backend").text())};std::string error;if(!ecmgui_v2::valid_order_input(i,error))throw std::runtime_error(error);
        g::CurveRequest request;request.prime=g::Integer(i.factor);request.sigma=std::stoull(i.sigma);request.family=i.profile==1?g::Family::Edwards:g::Family::Montgomery;request.parameter=i.profile==1?-1:i.profile==3?1:i.profile==5?2:i.profile==4?3:0;
        ecm_diagnostics::Settings settings;settings.backend=i.backend==1?ecm_diagnostics::Backend::Gp:i.backend==2?ecm_diagnostics::Backend::Native:i.backend==3?ecm_diagnostics::Backend::Gmp:ecm_diagnostics::Backend::Auto;settings.gp=i.gp.empty()?"gp":i.gp;settings.native_total_ms=5000;
        const auto output=std::filesystem::path(argv[2]);
        out.set("status",str("resource_limit"));out.set("backend",str(ecm_diagnostics::name(settings.backend)));out.set("reason",str("analysis initialization in progress"));save(output,out);
        ecm_diagnostics::Session session(settings);
        if(g::classify_prime(request.prime)!=g::Primality::Composite||mpz_cmp_ui(request.prime.v,2)<0)out=analysis(session.group(request));
        else{
            j::Value components;components.type='a';g::Factorization split;split.cofactor=request.prime;
            std::string backend=ecm_diagnostics::name(settings.backend),pari_failure,gp_failure;
            const auto batch=[&](){auto value=ecmgui_v2::order_components(components,factors(split.factors),split.cofactor.str(),backend,split.all_proven);value.set("native_engine",str(backend=="pari"?"pari":backend=="gmp"?"gmp":""));value.set("native_failure",str(pari_failure));value.set("fallback",str(gp_failure));value.set("reason",str(split.reason));return value;};
            const auto analyze_known=[&](){for(auto& factor:split.factors){
                if(std::any_of(components.items.begin(),components.items.end(),[&](const auto&row){return row.get("prime").text()==factor.prime.str();}))continue;
                auto field=request;field.prime=factor.prime;const auto result=session.group(field);backend=result.backend;pari_failure=result.pari_failure;gp_failure=result.gp_failure;
                if(result.value.modulus_primality==g::Primality::Proven)factor.primality=g::Primality::Proven;
                auto row=analysis(result);row.set("prime",str(factor.prime.str()));row.set("exponent",j::Value::number(std::to_string(factor.exponent)));components.items.push_back(row);save(output,batch());
            }};
            // Cheap decomposition exposes known small fields before an expensive
            // residual operation. This is preparation, not a GMP group-count fallback.
            if(i.backend!=1){g::Limits cheap;cheap.milliseconds=50;cheap.rho_iterations=8192;split=g::factor_integer(request.prime,cheap);analyze_known();}
            if(mpz_cmp_ui(split.cofactor.v,1)>0){
                const auto known=split.factors;const auto result=session.factor(split.cofactor);backend=result.backend;pari_failure=result.pari_failure;gp_failure=result.gp_failure;split=result.value;
                split.factors.insert(split.factors.end(),known.begin(),known.end());std::sort(split.factors.begin(),split.factors.end(),[](const auto&a,const auto&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});
            }
            analyze_known();split.all_proven=mpz_cmp_ui(split.cofactor.v,1)==0&&std::all_of(split.factors.begin(),split.factors.end(),[](const auto& p){return p.primality==g::Primality::Proven;});out=batch();
        }
    }catch(const std::exception&e){out.set("error",str(e.what()));}
    try{save(argv[2],out);}catch(...){return 3;}return out.has("error")?1:0;
}
