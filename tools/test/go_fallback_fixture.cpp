#include "ecm/diagnostics/order_analysis.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
int main(int argc,char**argv){
    if(argc>1 && std::string(argv[1])=="-q"){
        if(const char*counter=std::getenv("ECM_TEST_GP_COUNTER"))std::ofstream(counter,std::ios::app)<<"call\n";
        std::string mode=std::getenv("ECM_TEST_GP_MODE")?std::getenv("ECM_TEST_GP_MODE"):"protocol";
        if(mode=="timeout"){std::this_thread::sleep_for(std::chrono::seconds(10));return 0;}
        if(mode=="flood"){std::cout<<std::string(3*1024*1024,'x');return 0;}
        if(mode=="invalid")std::cout<<"GROUP|1\nPOINT|1\nECM_OK\n";
        else if(mode=="factor_invalid")std::cout<<"F|4|1\nECM_OK\n";
        else if(mode=="math")std::cout<<"ECM_MATH_ERROR|invalid_curve\n";
        else if(mode=="exit")return 4;
        else std::cout<<"not a group-order protocol\n";
        return 0;
    }
    if(argc!=4)return 2;
    ecm_diagnostics::Settings settings;settings.gp=argv[1];settings.gp_timeout_ms=1000;
    settings.backend=ecm_diagnostics::parse_backend(argv[2]);settings.native_total_ms=std::stoull(argv[3]);
    ecm_diagnostics::Session session(settings);
    ecm_group_order::CurveRequest request;request.prime=ecm_group_order::Integer("59649589127497217");request.sigma=26;
    if(std::getenv("ECM_TEST_BAD_CURVE_FIRST")){
        auto bad=request;bad.sigma=0;
        if(session.group(bad).value.status!=ecm_group_order::Status::InvalidInput)return 7;
    }
    for(int i=0;i<2;++i){
        auto result=session.group(request);
        std::cout<<"{\"backend\":"<<ecm_diagnostics::quote(result.backend)<<",\"status\":"<<ecm_diagnostics::quote(ecm_group_order::name(result.value.status))
            <<",\"group\":"<<ecm_diagnostics::quote(result.value.group_order.str())<<",\"point\":"<<ecm_diagnostics::quote(result.value.point_order.str())
            <<",\"reason\":"<<ecm_diagnostics::quote(result.value.reason)<<",\"gp_failure\":"<<ecm_diagnostics::quote(result.gp_failure)<<"}\n";
    }
}
