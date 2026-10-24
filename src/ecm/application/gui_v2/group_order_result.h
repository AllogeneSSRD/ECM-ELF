#pragma once
#include "../../storage/results/ecm_result_json.h"

namespace ecmgui_v2 {
// Composite input is a collection of prime-field analyses, never a claimed
// group over F_n or F_(p^e). Keep each component's proof and failure metadata.
inline ecm_results::json::Value order_components(ecm_results::json::Value components,
        ecm_results::json::Value factors,const std::string& cofactor,
        const std::string& backend,bool input_proven){
    using V=ecm_results::json::Value;
    V out;out.type='o';out.set("protocol",V::number("1"));
    bool complete=cofactor=="1"&&!components.items.empty()&&components.items.size()==factors.items.size(),proven=input_proven;
    for(const auto& component:components.items){
        complete&=component.get("status").text()=="complete";
        proven&=component.get("all_proven").scalar=="true";
    }
    out.set("status",V::string(complete?"complete":"partial"));
    out.set("backend",V::string(backend));out.set("modulus_primality",V::string("composite"));
    V flag;flag.scalar=complete&&proven?"true":"false";out.set("all_proven",flag);
    out.set("input_factors",std::move(factors));out.set("input_cofactor",V::string(cofactor));
    out.set("components",std::move(components));return out;
}
}
