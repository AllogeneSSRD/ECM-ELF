#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "group_order.h"
#include "ecm/application/configuration/runtime.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif
namespace ecm_stage1_app {
static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}
static std::string mpz_to_dec_string(const mpz_t v) {
    char *s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
}
struct PrimePowerBound {
    uint32_t p;
    uint32_t exp;
};

bool build_primes_up_to_B1(double B1, std::vector<uint32_t> &primes) {
    primes.clear();
    if (B1 < 2.0) {
        return true;
    }
    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }
    const uint32_t limit = (uint32_t)limit64;
    std::vector<char> sieve((size_t)limit + 1u, 1);
    sieve[0] = sieve[1] = 0;
    for (uint32_t p = 2; (uint64_t)p * (uint64_t)p <= limit; ++p) {
        if (!sieve[p]) continue;
        for (uint64_t q = (uint64_t)p * (uint64_t)p; q <= limit; q += p) {
            sieve[(size_t)q] = 0;
        }
    }

    for (uint32_t p = 2; p <= limit; ++p) {
        if (!sieve[p]) continue;
        primes.push_back(p);
    }
    return true;
}

static std::vector<PrimePowerBound> factor_by_small_primes(
    const mpz_t n, const std::vector<uint32_t> &primes) {
    std::vector<PrimePowerBound> out;
    mpz_t work;
    mpz_init_set(work, n);
    for (uint32_t p : primes) {
        uint32_t exp = 0;
        while (mpz_divisible_ui_p(work, p) != 0) {
            mpz_fdiv_q_ui(work, work, p);
            ++exp;
        }
        if (exp > 0) {
            out.push_back({p, exp});
        }
    }
    mpz_clear(work);
    return out;
}

static std::string format_group_order_smooth(const std::vector<PrimePowerBound> &parts) {
    std::ostringstream oss;
    oss << "[ ";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << "<" << parts[i].p << ", " << parts[i].exp << ">";
    }
    oss << " ]";
    return oss.str();
}

static std::string normalize_gp_path(std::string s) {
    trim(s);
    // Remove any outer single/double quotes repeatedly.
    while (s.size() >= 2 &&
           ((s.front() == '"' && s.back() == '"') ||
            (s.front() == '\'' && s.back() == '\''))) {
        s = s.substr(1, s.size() - 2);
        trim(s);
    }
    while (!s.empty() && (s.front() == '"' || s.front() == '\'')) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == '"' || s.back() == '\'')) {
        s.pop_back();
    }
    s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\''), s.end());
    return s;
}

bool gp_executable_exists(const std::string &exe_path) {
    const std::string cleaned = normalize_gp_path(exe_path);
    if (cleaned.empty()) return false;
#ifdef _WIN32
    char found[MAX_PATH];
    DWORD len = SearchPathA(nullptr, cleaned.c_str(), ".exe", MAX_PATH, found, nullptr);
    return len > 0;
#else
    if (cleaned.find('/') != std::string::npos) {
        return access(cleaned.c_str(), X_OK) == 0;
    }
    std::string cmd = "which \"" + cleaned + "\" > /dev/null 2>&1";
    return system(cmd.c_str()) == 0;
#endif
}

std::string resolve_gp_path(const std::string &exe_path) {
    const std::string cleaned = normalize_gp_path(exe_path);
    if (cleaned.empty()) return cleaned;
#ifdef _WIN32
    // If the path contains a directory separator, use it as-is.
    if (cleaned.find('\\') != std::string::npos ||
        cleaned.find('/') != std::string::npos) {
        return cleaned;
    }
    // Bare name → resolve via SearchPath.
    char found[MAX_PATH];
    DWORD len = SearchPathA(nullptr, cleaned.c_str(), ".exe", MAX_PATH, found, nullptr);
    if (len > 0 && len < MAX_PATH) {
        return std::string(found);
    }
#endif
    return cleaned;
}

std::string get_gp_executable(const std::string &explicit_path) {
    // 1. Explicit --gp argument takes highest priority.
    if (!explicit_path.empty()) {
        return normalize_gp_path(explicit_path);
    }
    // 2. Config (set from --gp). 3. Fall back to "gp" on PATH.
    if (!ecm_runtime_config().gp_bin.empty()) {
        return normalize_gp_path(ecm_runtime_config().gp_bin);
    }
    return "gp";
}


static void print_parts(const std::vector<ecm_group_order::PrimePower>&parts,double bound=-1) {
    std::cout<<"[ ";bool first=true;
    for(const auto &part:parts){
        if(bound>=0 && mpz_cmp_d(part.prime.v,bound)>0)continue;
        if(!first)std::cout<<", ";first=false;
        std::cout<<'<'<<part.prime.str()<<", "<<part.exponent<<'>';
    }
    std::cout<<" ]";
}
static void print_legacy_bounds(uint32_t i,int param,const std::vector<ecm_group_order::PrimePower>&parts) {
    using ecm_group_order::Integer;
    Integer best(0),rest(1),value;const ecm_group_order::PrimePower *selected=nullptr;
    for(const auto &part:parts){
        mpz_pow_ui(value.v,part.prime.v,part.exponent);
        if(mpz_cmp(value.v,best.v)>0){best=value;selected=&part;}
    }
    for(const auto &part:parts){
        const unsigned e=part.exponent-(&part==selected?1:0);
        mpz_pow_ui(value.v,part.prime.v,e);if(mpz_cmp(value.v,rest.v)>0)rest=value;
    }
    // Preserve historical GROUP-based fields; exact POINT bounds have a separate label.
    std::cout<<"  go_min_bounds["<<i<<"]: param="<<param<<" largest_prime="<<(selected?selected->prime.str():"0")
        <<" min_b1_stage1="<<best.str()<<" min_b1_stage2="<<rest.str()
        <<" min_b2_stage2="<<(selected?selected->prime.str():"0")<<" largest_exponent="<<(selected?selected->exponent:0)<<'\n';
}
static void print_point_bounds(uint32_t i,const std::vector<ecm_group_order::PrimePower>&parts,uint32_t multiplier) {
    using ecm_group_order::Integer;
    auto adjusted=parts;
    for(auto &part:adjusted){
        uint32_t extra=0,remaining=multiplier;
        if(mpz_cmp_ui(part.prime.v,multiplier)<=0){
            const auto q=mpz_get_ui(part.prime.v);
            while(remaining>1 && remaining%q==0){remaining/=uint32_t(q);++extra;}
        }
        part.exponent-=std::min(part.exponent,extra);
    }
    Integer stage1(2),value;
    for(const auto &part:adjusted){mpz_pow_ui(value.v,part.prime.v,part.exponent);if(mpz_cmp(value.v,stage1.v)>0)stage1=value;}
    std::vector<std::pair<Integer,Integer>> candidates;candidates.emplace_back(stage1,Integer(0));
    for(const auto &left:adjusted)if(left.exponent){
        Integer b1(2);
        for(const auto &part:adjusted){
            mpz_pow_ui(value.v,part.prime.v,part.exponent-(&part==&left?1:0));
            if(mpz_cmp(value.v,b1.v)>0)b1=value;
        }
        if(mpz_cmp(left.prime.v,b1.v)>0)candidates.emplace_back(b1,left.prime);
    }
    std::cout<<"  go_point_bounds["<<i<<"]: scalar="<<multiplier<<"LCM min_b1_stage1="<<stage1.str()<<" frontier=[ ";
    bool first=true;
    for(size_t j=0;j<candidates.size();++j){
        const auto &a=candidates[j];bool dominated=false;
        for(size_t k=0;k<candidates.size();++k)if(j!=k && mpz_cmp(candidates[k].first.v,a.first.v)<=0 && mpz_cmp(candidates[k].second.v,a.second.v)<=0)dominated=true;
        if(dominated)continue;if(!first)std::cout<<", ";first=false;
        std::cout<<'<'<<a.first.str()<<", "<<a.second.str()<<'>';
    }
    std::cout<<" ]\n";
}
void print_group_order_factor(const ecm_stage1::TaskResult &result,uint32_t i,int param,
                              ecm_diagnostics::Session &session,double B1,uint32_t multiplier,ecm_group_order::Family family) {
    try {
        ecm_group_order::CurveRequest request;
        mpz_set(request.prime.v,result.factor(i));request.family=family;
        request.parameter=family==ecm_group_order::Family::Montgomery?param:-1;
        request.sigma=result.has_sigmas()?result.sigma(i):uint64_t(result.first_sigma())+i;
        auto diagnosis=session.group(request);const auto &value=diagnosis.value;
        std::cout<<"  go_status["<<i<<"]: backend="<<diagnosis.backend<<" status="<<ecm_group_order::name(value.status)
            <<" all_proven="<<(value.all_proven?1:0)<<" reason="<<ecm_diagnostics::quote(value.reason)
            <<" pari_failure="<<ecm_diagnostics::quote(diagnosis.pari_failure)<<" gp_failure="<<ecm_diagnostics::quote(diagnosis.gp_failure)<<'\n';
        if(value.group_order_computed){
            std::cout<<"  go["<<i<<"]="<<value.group_order.str()<<'\n';
            std::cout<<"  go_factor["<<i<<"]=";print_parts(value.multiple_factorization.factors,B1);std::cout<<'\n';
            if(value.multiple_factorization.status==ecm_group_order::Status::Complete)
                print_legacy_bounds(i,request.parameter,value.multiple_factorization.factors);
        }
        if(value.status==ecm_group_order::Status::Complete && value.point_minimality_checked){
            std::cout<<"  go_point_order["<<i<<"]="<<value.point_order.str()<<'\n';
            std::cout<<"  go_point_factor["<<i<<"]=";print_parts(value.point_factorization);std::cout<<'\n';
            print_point_bounds(i,value.point_factorization,multiplier);
        }
    }catch(const std::exception &error){
        std::cerr<<"  go_status["<<i<<"]: status=unresolved reason="<<ecm_diagnostics::quote(error.what())<<'\n';
    }
}
} // namespace ecm_stage1_app
