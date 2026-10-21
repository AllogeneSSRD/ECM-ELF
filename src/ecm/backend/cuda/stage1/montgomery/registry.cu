#include "../../arithmetic/cgbn/types.cuh"
#include "launch.h"
void cgbn_stage1_launch(cgbn_stage1_kernel_fn kernel,dim3 grid,dim3 block,
    cgbn_error_report_t *report,uint64_t scalar_bits,uint64_t start,uint64_t count,
    uint32_t *scalar,uint32_t *data,uint32_t curves,uint32_t sigma,uint32_t np0) {
    kernel<<<grid,block>>>(report,scalar_bits,start,count,scalar,data,curves,sigma,np0);
}
cgbn_stage1_kernel_fn ecm_mont_param3_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g6(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn ecm_mont_param0_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g6(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn ecm_mont_param2_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g6(bits,tpi))return fn;
    return nullptr;
}
