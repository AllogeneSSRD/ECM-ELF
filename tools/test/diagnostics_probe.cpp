#include "ecm/diagnostics/order_analysis.h"
#include "ecm/diagnostics/pari_protocol.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
int main(int argc,char** argv){
    namespace d=ecm_diagnostics;namespace p=d::pari_protocol;using V=p::V;
    // Isolated fault injection executable. No hooks in production binaries.
    if(argc>1&&std::string(argv[1])=="-q"){
        if(const char* path=std::getenv("ECM_DIAG_GP_COUNTER"))std::ofstream(path,std::ios::app)<<"gp\n";
        std::cout<<"ECM_GP_ERROR\n";return 0;
    }
    if(argc==2){
        if(const char* path=std::getenv("ECM_DIAG_HOST_COUNTER"))std::ofstream(path,std::ios::app)<<"pari\n";
        const std::string mode=std::getenv("ECM_DIAG_HOST_MODE")?std::getenv("ECM_DIAG_HOST_MODE"):"unavailable";
        if(mode=="timeout"){
            ecm_group_order::Factorization partial;partial.status=ecm_group_order::Status::Partial;partial.cofactor=ecm_group_order::Integer(1013);partial.factors.push_back({ecm_group_order::Integer(1009),1,ecm_group_order::Primality::Proven});auto v=p::encode(partial);v.set("protocol",V::number("2"));std::cout<<v.dump()<<std::endl;std::this_thread::sleep_for(std::chrono::seconds(30));return 0;
        }
        if(mode=="invalid_product"){
            ecm_group_order::Factorization f;f.status=ecm_group_order::Status::Complete;f.all_proven=true;f.factors.push_back({ecm_group_order::Integer(7),1,ecm_group_order::Primality::Proven});auto v=p::encode(f);v.set("protocol",V::number("2"));std::cout<<v.dump()<<std::endl;return 0;
        }
        std::cout<<"{\"protocol\":2,\"error\":\"fixture PARI unavailable\"}\n";return 3;
    }
    if(argc<7)return 2;
    d::Settings settings;settings.backend=d::parse_backend(argv[1]);settings.gp=argv[2];
    const std::string op=argv[3];ecm_group_order::Integer n(argv[4]);
    if(argc>7&&std::string(argv[7])!="default")settings.pari_host=argv[7];
    if(argc>8)settings.pari_timeout_ms=std::stoull(argv[8]);
    d::Session session(settings);V out;
    if(op=="factor"){auto result=session.factor(n);out=p::encode(result.value);out.set("backend",V::string(result.backend));out.set("pari_failure",V::string(result.pari_failure));out.set("gp_failure",V::string(result.gp_failure));}
    else{ecm_group_order::CurveRequest r;r.prime=n;r.sigma=std::stoull(argv[5]);const int profile=std::stoi(argv[6]);r.family=profile==-1?ecm_group_order::Family::Edwards:ecm_group_order::Family::Montgomery;r.parameter=profile;auto result=session.group(r);out=p::encode(result.value);out.set("backend",V::string(result.backend));out.set("pari_failure",V::string(result.pari_failure));out.set("gp_failure",V::string(result.gp_failure));}
    std::cout<<out.dump()<<std::endl;return 0;
}
