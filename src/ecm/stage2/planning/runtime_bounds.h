#pragma once
#include "portable_model.h"

namespace ecm_stage2 { namespace tune { namespace portable {
// Performance identity is the executed bound. The request is only a minimum
// coverage contract; several requests may legitimately map to one actual B2.
inline bool runtime_bounds(Shape &shape,Word requested,bool rounding,Word maximum=Word(INT64_MAX)-8192) {
    Word actual=requested;
    if(rounding && (shape.force_ladder || !giant_round_b2(requested,shape.d,d_catalog::degree(shape.d),
        shape.giant_chunk,shape.chain_min,1,actual) || actual>maximum || actual>Word(INT64_MAX)-8192))return false;
    shape.b2=actual;return true;
}
inline bool runtime_bound_matches(const Shape &observed,Word requested,bool rounding,Word maximum=Word(INT64_MAX)-8192) {
    auto expected=observed;
    return runtime_bounds(expected,requested,rounding,maximum) && expected.b2==observed.b2;
}
} } }
