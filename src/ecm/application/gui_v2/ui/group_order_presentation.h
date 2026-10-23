#pragma once
#include "../../../storage/results/ecm_result_json.h"
#include <string>

namespace ecmgui_v2::ui {
// Presentation only: preserve the full proof metadata in the analysis result.
inline std::string order_factorization(const ecm_results::json::Value& result,bool point){
    std::string value;
    for(const auto& factor:result.get(point?"point_factors":"group_factors").items){
        if(!value.empty())value+=" × ";
        value+=factor.get("prime").text();
        const auto exponent=factor.get("exponent").text();
        if(exponent!="1")value+="^"+exponent;
    }
    const auto cofactor=result.get("cofactor").text();
    if(!point&&!cofactor.empty()&&cofactor!="1"){
        if(!value.empty())value+=" × ";
        value+="["+cofactor+"]"; // Brackets denote the unresolved part, not a prime.
    }
    if(value.empty()&&result.get(point?"point_order":"group_order").text()=="1")value="1";
    return value;
}
inline bool order_succeeded(const ecm_results::json::Value& result,const std::string& error){
    return error.empty()&&result.get("status").text()=="complete";
}
}
