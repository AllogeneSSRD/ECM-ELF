#include "ecm_stage1_selection.h"

#include <cstdlib>
#include <iostream>

static void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int main() {
    using namespace ecm_stage1;
    Selection selection;
    require(parse_curve_model("ED", selection.curve) &&
            selection.curve == CurveModel::Edwards, "Edwards curve model must parse");
    require(parse_execution_target("cpu", selection.target) &&
            selection.target == ExecutionTarget::Cpu, "CPU target must parse");
    require(parse_curve_model("mont", selection.curve) &&
            selection.curve == CurveModel::Montgomery, "Montgomery curve model must parse");
    require(parse_execution_target("gpu", selection.target) &&
            selection.target == ExecutionTarget::Gpu, "GPU target must parse");
    require(!parse_curve_model("gpu", selection.curve) &&
            !parse_execution_target("ed", selection.target) &&
            !parse_curve_model("montgomery", selection.curve) &&
            !parse_curve_model("", selection.curve),
            "curve model and execution target must not accept mixed or empty values");
    selection.curve = CurveModel::Edwards;
    require(!valid_route(selection), "GPU Edwards must be rejected before execution");
    selection.target = ExecutionTarget::Cpu;
    require(valid_route(selection), "CPU Edwards must remain available");
    require(static_cast<int>(CpuImplementation::Auto) == 0 &&
            static_cast<int>(SimdReduction::Auto) == 0,
            "all automatic selection enums must use zero");

    CpuImplementation cpu = CpuImplementation::Auto;
    SimdReduction reduction = SimdReduction::Auto;
    ScalarConstruction scalar = ScalarConstruction::PrimePowerLcm;
    require(parse_cpu_implementation("scalar", cpu) && cpu == CpuImplementation::GmpScalar,
            "CPU implementation must parse");
    require(parse_simd_reduction("mersenne-fold", reduction) && reduction == SimdReduction::MersenneFold,
            "SIMD reduction must parse");
    require(parse_scalar_construction("1", scalar) &&
            scalar == ScalarConstruction::PrimePowerLcm, "scalar construction must parse");
    require(parse_scalar_construction("12", scalar) &&
            scalar == ScalarConstruction::PrimePowerLcmTimes12,
            "times12 scalar construction must parse");
    require(!parse_cpu_implementation("unknown", cpu) &&
            !parse_simd_reduction("unknown", reduction) &&
            !parse_scalar_construction("unknown", scalar) &&
            !parse_cpu_implementation("gmp_scalar", cpu) &&
            !parse_scalar_construction("choose12", scalar),
            "unknown and removed option values must be rejected");

    int param = -1;
    require(resolve_gpu_param(3, false, 0, param) && param == 0,
            "sigma prefix must select PARAM0");
    require(resolve_gpu_param(3, false, 2, param) && param == 2,
            "sigma prefix must select PARAM2");
    require(resolve_gpu_param(0, false, 3, param) && param == 3,
            "sigma prefix must select PARAM3");
    require(resolve_gpu_param(3, true, 3, param) && param == 3,
            "matching explicit parameter must be accepted");
    require(!resolve_gpu_param(3, true, 2, param),
            "contradictory sigma and explicit parameter must be rejected");
    require(!resolve_gpu_param(1, false, -1, param) &&
            !resolve_gpu_param(3, false, 1, param),
            "unknown parametrization must be rejected");
    return 0;
}
