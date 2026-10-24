#include "pari_native.h"
#include "pari_protocol.h"
#include <fstream>
#include <iostream>
#include <windows.h>

// One request per process: PARI's global stack/errors cannot escape into a
// worker or the GUI. The parent owns cancellation, memory and time limits.
int wmain(int argc,wchar_t** argv){
    namespace p=ecm_diagnostics::pari_protocol;using V=p::V;
    if(argc!=2)return 2;
    const auto publish=[](V value){value.set("protocol",V::number("2"));std::cout<<value.dump()<<std::endl;};
    try{
        if(std::filesystem::file_size(argv[1])>65536)throw std::runtime_error("PARI request exceeds limit");
        std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("Cannot read PARI request");std::string raw{std::istreambuf_iterator<char>(input),{}};const auto request=ecm_results::json::parse(raw);
        wchar_t path[32768];const auto size=GetModuleFileNameW(nullptr,path,32768);if(!size||size==32768)throw std::runtime_error("Cannot locate PARI host");
        ecm_diagnostics::NativeOrder engine(std::filesystem::path(path).parent_path());
        if(!engine.available()){auto error=p::object();error.set("error",V::string(engine.failure()));publish(error);return 3;}
        const auto op=request.get("operation").text();
        if(op=="factor"){
            const auto integer=p::number(request.get("integer"));
            auto result=engine.factor(integer,[&](const auto& v){publish(p::encode(v));});publish(p::encode(result));
        }else if(op=="group"){
            ecm_group_order::CurveRequest r;r.prime=p::number(request.get("integer"));r.sigma=std::stoull(request.get("sigma").text());
            const auto family=request.get("family").text();if(family!="edwards"&&family!="montgomery")throw std::invalid_argument("Unknown curve family");
            r.family=family=="edwards"?ecm_group_order::Family::Edwards:ecm_group_order::Family::Montgomery;r.parameter=std::stoi(request.get("parameter").text());
            auto result=engine.group(r,[&](const auto& v){publish(p::encode(v));});publish(p::encode(result));
        }else throw std::invalid_argument("Unknown PARI operation");
        return 0;
    }catch(const std::exception& e){auto error=p::object();error.set("error",V::string(e.what()));publish(error);return 4;}
}
