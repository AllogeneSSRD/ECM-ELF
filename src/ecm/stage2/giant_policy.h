#pragma once
#include "giant_schedule.h"
#include <cstdlib>

namespace ecm_stage2 {
// Host scopes keep planning, execution and semantic replay on the same policy.
inline thread_local int giant_balance_override=-1;
inline thread_local Word giant_chain_override=0;
inline bool giant_tail_balance_requested() {
    if(giant_balance_override>=0)return giant_balance_override!=0;
    const char *v=std::getenv("NTT_GIANT_TAIL_BALANCE");return v && std::atoi(v)!=0;
}
inline Word giant_chain_points() {
    if(giant_chain_override)return giant_chain_override;
    const char *v=std::getenv("NTT_GIANT_CHAIN_BLOCK");
    return v && *v?std::max(Word(4),std::min(Word(1)<<20,std::strtoull(v,nullptr,10))):64;
}
struct GiantPolicyScope {
    int balance=giant_balance_override;Word chain=giant_chain_override;
    GiantPolicyScope(int b,Word c){if(b>=0)giant_balance_override=b;if(c)giant_chain_override=c;}
    ~GiantPolicyScope(){giant_balance_override=balance;giant_chain_override=chain;}
    GiantPolicyScope(const GiantPolicyScope&)=delete;
};
} // namespace ecm_stage2
