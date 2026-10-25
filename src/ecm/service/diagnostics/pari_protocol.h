#pragma once
#include "../../core/group_order/group_order.h"
#include "io/json.h"

// Private, versioned process protocol. Integers always remain decimal strings.
namespace ecm_diagnostics::pari_protocol {
namespace g = ecm_group_order;
using V = ecm_results::json::Value;
inline V flag(bool b){V v;v.scalar=b?"true":"false";return v;}
inline V object(){V v;v.type='o';return v;}
inline V parts(const std::vector<g::PrimePower>& values){
    V out;out.type='a';for(const auto& p:values){auto row=object();row.set("prime",V::string(p.prime.str()));row.set("exponent",V::number(std::to_string(p.exponent)));row.set("primality",V::string(g::name(p.primality)));out.items.push_back(row);}return out;
}
inline V encode(const g::Factorization& f){
    auto out=object();out.set("status",V::string(g::name(f.status)));out.set("reason",V::string(f.reason));out.set("factors",parts(f.factors));out.set("cofactor",V::string(f.cofactor.str()));out.set("all_proven",flag(f.all_proven));return out;
}
inline V encode(const g::Analysis& a){
    auto out=object();out.set("status",V::string(g::name(a.status)));out.set("reason",V::string(a.reason));out.set("modulus_primality",V::string(g::name(a.modulus_primality)));out.set("group_computed",flag(a.group_order_computed));out.set("group_order",V::string(a.group_order.str()));out.set("multiple",V::string(a.multiple.str()));out.set("point_order",V::string(a.point_order.str()));out.set("factorization",encode(a.multiple_factorization));out.set("point_factors",parts(a.point_factorization));out.set("point_minimality",flag(a.point_minimality_checked));out.set("all_proven",flag(a.all_proven));return out;
}
inline g::Status status(const V& v){const auto s=v.text();if(s=="complete")return g::Status::Complete;if(s=="partial")return g::Status::Partial;if(s=="resource_limit")return g::Status::ResourceLimit;if(s=="invalid_input")return g::Status::InvalidInput;throw std::runtime_error("Invalid PARI host status");}
inline g::Primality primality(const V& v){const auto s=v.text();if(s=="proven")return g::Primality::Proven;if(s=="probable")return g::Primality::Probable;if(s=="composite")return g::Primality::Composite;throw std::runtime_error("Invalid PARI host primality");}
inline g::Integer number(const V& v){const auto s=v.text();if(s.empty()||s.size()>2500||s.find_first_not_of("0123456789")!=s.npos)throw std::runtime_error("Invalid PARI host integer");return g::Integer(s);}
inline std::vector<g::PrimePower> read_parts(const V& v,const g::Integer& bound){
    if(v.type!='a'||v.items.size()>4096)throw std::runtime_error("Invalid PARI host factors");
    std::vector<g::PrimePower> out;g::Integer product(1),power;
    for(const auto& row:v.items){auto p=number(row.get("prime")),e=number(row.get("exponent"));
        if(mpz_cmp_ui(p.v,2)<0||mpz_cmp(p.v,bound.v)>0||!mpz_fits_ulong_p(e.v)||mpz_cmp_ui(e.v,1)<0||mpz_cmp_ui(e.v,4096)>0||(!out.empty()&&mpz_cmp(out.back().prime.v,p.v)>=0)||g::classify_prime(p)==g::Primality::Composite)throw std::runtime_error("Invalid PARI host factor power");
        const auto exponent=uint32_t(mpz_get_ui(e.v));if((mpz_sizeinbase(p.v,2)-1)*exponent>mpz_sizeinbase(bound.v,2))throw std::runtime_error("PARI host factor exceeds bound");
        mpz_pow_ui(power.v,p.v,exponent);mpz_mul(product.v,product.v,power.v);if(!mpz_divisible_p(bound.v,product.v))throw std::runtime_error("PARI host factor product mismatch");
        out.push_back({std::move(p),exponent,primality(row.get("primality"))});
    }return out;
}
inline g::Factorization factor(const V& v,const g::Integer& input){
    g::Factorization out;out.status=status(v.get("status"));out.reason=v.get("reason").text();out.cofactor=number(v.get("cofactor"));out.factors=read_parts(v.get("factors"),input);out.all_proven=v.get("all_proven").scalar=="true";
    g::Integer product=out.cofactor,power;for(const auto& p:out.factors){mpz_pow_ui(power.v,p.prime.v,p.exponent);mpz_mul(product.v,product.v,power.v);if(out.all_proven&&p.primality!=g::Primality::Proven)throw std::runtime_error("Unproven PARI host factor");}
    if(mpz_cmp(product.v,input.v)||(out.status==g::Status::Complete&&(mpz_cmp_ui(out.cofactor.v,1)||!out.all_proven)))throw std::runtime_error("Invalid PARI host factorization");return out;
}
inline g::Analysis group(const V& v,const g::CurveRequest& r){
    g::Analysis out;out.status=status(v.get("status"));out.reason=v.get("reason").text();out.modulus_primality=primality(v.get("modulus_primality"));out.group_order_computed=v.get("group_computed").scalar=="true";out.group_order=number(v.get("group_order"));out.multiple=number(v.get("multiple"));out.point_order=number(v.get("point_order"));out.point_minimality_checked=v.get("point_minimality").scalar=="true";out.all_proven=v.get("all_proven").scalar=="true";
    if(out.group_order_computed){
        g::Integer delta,bound;mpz_sub(delta.v,out.group_order.v,r.prime.v);mpz_sub_ui(delta.v,delta.v,1);mpz_mul(delta.v,delta.v,delta.v);mpz_mul_ui(bound.v,r.prime.v,4);
        if(mpz_cmp_ui(out.group_order.v,1)<0||mpz_cmp(delta.v,bound.v)>0||mpz_cmp(out.multiple.v,out.group_order.v))throw std::runtime_error("Invalid PARI host group order");
        out.multiple_factorization=factor(v.get("factorization"),out.group_order);
    }else if(out.status==g::Status::Complete)throw std::runtime_error("Missing PARI host group order");
    if(mpz_cmp_ui(out.point_order.v,0)>0){
        if(!out.group_order_computed||!mpz_divisible_p(out.group_order.v,out.point_order.v))throw std::runtime_error("Invalid PARI host point order");
        out.point_factorization=read_parts(v.get("point_factors"),out.point_order);
    }
    if(out.status==g::Status::Complete){
        if(!out.all_proven||out.modulus_primality!=g::Primality::Proven||!out.point_minimality_checked||out.multiple_factorization.status!=g::Status::Complete||mpz_cmp_ui(out.point_order.v,1)<0)throw std::runtime_error("Incomplete PARI host proof");
        auto curve=g::construct_curve(r);if(!g::multiply(curve,curve.point,out.point_order).infinity)throw std::runtime_error("Invalid PARI host point multiple");
        g::Integer product(1),power;
        for(const auto& p:out.point_factorization){if(p.primality!=g::Primality::Proven)throw std::runtime_error("Unproven PARI point factor");mpz_pow_ui(power.v,p.prime.v,p.exponent);mpz_mul(product.v,product.v,power.v);g::Integer reduced;mpz_divexact(reduced.v,out.point_order.v,p.prime.v);if(g::multiply(curve,curve.point,reduced).infinity)throw std::runtime_error("Invalid PARI host point minimality");}
        if(mpz_cmp(product.v,out.point_order.v))throw std::runtime_error("Incomplete PARI point factorization");
    }return out;
}
}
