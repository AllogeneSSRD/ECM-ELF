#pragma once
#include "context.h"
#include "stage2_d_model.cuh"
#include <optional>

// Host-only planning storage. Retained by the curve until all its device owners
// are destroyed; the model caches keep their original lifetime.
struct Stage2PlanRuntime {
    size_t freeb=0,totalb=0,reserve=0,cap=0;
    unsigned long long D=0,P_baby=0;
    bool s4_on=false;
    std::optional<DPhaseModel> phase_model;
};
int prepare_stage2_plan(ecm_stage2::ModulusContext &L,size_t nw,
    unsigned long long sigma,unsigned long long B1,unsigned long long B2,
    unsigned long long D_in,bool choose_d,int curves,bool d_plan_only,
    ecm_stage2::Plan *plan_result,Stage2PlanRuntime &runtime)
;
