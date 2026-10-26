// Operator interface — ECM_STAGE1_*_IMPL macros injected by host before this header.

#ifndef ECM_STAGE1_MUL_IMPL
#error "ECM_STAGE1_MUL_IMPL not defined: host must inject selected montgomery mul symbol."
#endif
#ifndef ECM_STAGE1_SQR_IMPL
#error "ECM_STAGE1_SQR_IMPL not defined: host must inject selected montgomery sqr symbol."
#endif
#ifndef ECM_STAGE1_ADD_IMPL
#error "ECM_STAGE1_ADD_IMPL not defined: host must inject selected add_mod symbol."
#endif
#ifndef ECM_STAGE1_SUB_IMPL
#error "ECM_STAGE1_SUB_IMPL not defined: host must inject selected sub_mod symbol."
#endif
#ifndef ECM_STAGE1_SPECIAL_MULT_IMPL
#error "ECM_STAGE1_SPECIAL_MULT_IMPL not defined: host must inject selected special_mult symbol."
#endif

#define mont_mul(out, a, b, N, np0, limbs) ECM_STAGE1_MUL_IMPL((out), (a), (b), (N), (np0), (limbs))
#define mont_sqr(out, a, N, np0, limbs) ECM_STAGE1_SQR_IMPL((out), (a), (N), (np0), (limbs))
#define add_mod(r, a, b, N, limbs) ECM_STAGE1_ADD_IMPL((r), (a), (b), (N), (limbs))
#define sub_mod(r, a, b, N, limbs) ECM_STAGE1_SUB_IMPL((r), (a), (b), (N), (limbs))
#define special_mult(r, m, N, np0, limbs) ECM_STAGE1_SPECIAL_MULT_IMPL((r), (m), (N), (np0), (limbs))
