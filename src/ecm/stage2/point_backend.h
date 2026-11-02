#pragma once
#include "cuda/point_coop_plan.h"
#include <stdexcept>
#include <string>

namespace ecm_stage2 {
// Resolved execution identities, not the user's automatic-selection request.
// Unknown preserves old opt-in observations without relabelling their kernels.
constexpr unsigned point_backend_unknown=255;
inline bool cooperative_point_width(unsigned bits) {return bits>8192 && bits<=16384;}
inline int baby_backend(const std::string &value) {
    if(value=="auto")return -1;
    if(value=="scalar")return 0;
    if(value=="coop32")return 32;
    throw std::runtime_error("baby backend must be auto, scalar or coop32");
}
inline int giant_backend(const std::string &value) {
    if(value=="auto")return -1;
    if(value=="scalar")return 0;
    if(value=="coop16")return 16;
    if(value=="coop32")return 32;
    throw std::runtime_error("giant backend must be auto, scalar, coop16 or coop32");
}
inline const char *point_backend_name(unsigned value) {
    return value==32?"coop32":value==16?"coop16":value==point_backend_unknown?"legacy_unknown":"scalar";
}
inline unsigned point_backend_capacity(unsigned bits,unsigned mode) {
    return mode && mode!=point_backend_unknown && cooperative_point_width(bits)?
        unsigned(s2g_coop_capacity_words(int((bits+63)/64),int(mode))):0;
}
inline unsigned long long point_batch_bucket(unsigned long long points,unsigned long long cap=8192) {
    if(!cap)cap=8192;
    if(cap>(1ull<<24))cap=1ull<<24;
    const auto n=points<cap?points:cap;unsigned long long bucket=1;
    while(bucket<=n/2)bucket*=2;return bucket;
}
}
