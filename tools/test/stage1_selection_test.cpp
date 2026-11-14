#include "ecm/stage1/selection.h"

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
    require(valid_route(selection,true), "CUDA Edwards capability must enable the route");
    selection.target = ExecutionTarget::Cpu;
    require(valid_route(selection), "CPU Edwards must remain available");
    require(static_cast<int>(CpuImplementation::Auto) == 0 &&
            static_cast<int>(SimdReduction::Auto) == 0,
            "all automatic selection enums must use zero");

    CpuImplementation cpu = CpuImplementation::Auto;
    SimdReduction reduction = SimdReduction::Auto;
    CpuMontgomeryChain chain = CpuMontgomeryChain::Ladder;
    ScalarConstruction scalar = ScalarConstruction::PrimePowerLcm;
    require(parse_cpu_implementation("scalar", cpu) && cpu == CpuImplementation::GmpScalar,
            "CPU implementation must parse");
    require(parse_cpu_implementation("GWNUM", cpu) && cpu == CpuImplementation::Gwnum,
            "GWNUM implementation must parse case-insensitively");
    require(parse_simd_reduction("mersenne-fold", reduction) && reduction == SimdReduction::MersenneFold,
            "SIMD reduction must parse");
    require(parse_cpu_montgomery_chain("PRAC", chain) && chain == CpuMontgomeryChain::Prac &&
            parse_cpu_montgomery_chain("ladder", chain) && chain == CpuMontgomeryChain::Ladder &&
            !parse_cpu_montgomery_chain("unknown", chain), "CPU Montgomery chain selection must parse");
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

    for (const auto curve : {CurveModel::Edwards, CurveModel::Montgomery}) {
        selection.curve = curve;
        selection.cpu_implementation = CpuImplementation::Auto;
        const auto min_curves = curve == CurveModel::Edwards ? 8u : 2u;
        require(resolve_cpu_implementation(selection, 10240, min_curves - 1, true) ==
                    CpuImplementation::GmpScalar &&
                resolve_cpu_implementation(selection, 10240, min_curves, true) ==
                    CpuImplementation::IfmaSimd &&
                resolve_cpu_implementation(selection, 10240, min_curves, false) ==
                    CpuImplementation::GmpScalar,
                "narrow automatic CPU selection must preserve scalar and IFMA thresholds");
        require(resolve_cpu_implementation(selection, 10241, 1, true) ==
                    CpuImplementation::IfmaSimd &&
                !resolve_cpu_implementation(selection, 10241, 1, false) &&
                !resolve_cpu_implementation(selection, 10241, 8, false),
                "wide automatic CPU selection must require IFMA even for a single curve");
        selection.cpu_implementation = CpuImplementation::IfmaSimd;
        require(resolve_cpu_implementation(selection, 10240, 1, true) ==
                    CpuImplementation::IfmaSimd &&
                !resolve_cpu_implementation(selection, 10240, 1, false),
                "explicit SIMD must require IFMA for every batch size");
        selection.cpu_implementation = CpuImplementation::GmpScalar;
        const auto wide_scalar = resolve_cpu_implementation(selection, 10241, 1, false);
        require(resolve_cpu_implementation(selection, 10240, 8, true) ==
                    CpuImplementation::GmpScalar &&
                (curve == CurveModel::Edwards ? wide_scalar == CpuImplementation::GmpScalar
                                               : !wide_scalar),
                "only Edwards provides a wide GMP scalar implementation");
        selection.cpu_implementation = CpuImplementation::Gwnum;
        const auto gw = resolve_cpu_implementation(selection, 12511, 1, false);
        require(curve == CurveModel::Montgomery ? gw == CpuImplementation::Gwnum : !gw,
                "GWNUM must be explicit, independent of IFMA, and Montgomery-only");
    }

    selection.curve = CurveModel::Montgomery;
    require(parse_cpu_implementation_policy(
                "simd[:6144], scalar[ : 4096 ], gwnum[4096:], auto[\xEF\xBC\x9A]", selection),
            "CPU policy must accept overlapping ranges, whitespace and full-width colon");
    std::string decisions;
    CpuImplementationCapabilities caps{true, true, false, true, []() -> const char* { return nullptr; }};
    auto choose = [&](std::size_t bits) {
        return resolve_cpu_implementation_policy(selection, bits, 1, caps, decisions);
    };
    require(choose(4095) == CpuImplementation::IfmaSimd && choose(6143) == CpuImplementation::IfmaSimd &&
                choose(6144) == CpuImplementation::Gwnum,
            "policy must try SIMD first and exclude its stop boundary");
    caps.has_ifma = false;
    require(choose(4095) == CpuImplementation::GmpScalar && choose(4096) == CpuImplementation::Gwnum,
            "policy must continue past unavailable SIMD and honor the scalar stop boundary");
    caps.gwnum = []() -> const char* { return "missing test DLL"; };
    require(choose(6144) == CpuImplementation::GmpScalar && decisions.find("missing test DLL") != decisions.npos,
            "policy must continue to the explicit automatic fallback after a missing DLL");
    require(!choose(10241), "wide automatic fallback must still require IFMA");
    selection.curve = CurveModel::Edwards;
    require(choose(6144) == CpuImplementation::GmpScalar && decisions.find("PARAM0") != decisions.npos,
            "policy must skip GWNUM for Edwards");
    selection.curve = CurveModel::Montgomery;
    selection.simd_reduction = SimdReduction::MersenneFold;
    caps.has_ifma = true;
    require(!choose(4095) && decisions.find("full Mersenne") != decisions.npos,
            "policy must report unsatisfied explicit reduction rather than ignoring it");
    caps.mersenne_fold = true;
    require(choose(4095) == CpuImplementation::IfmaSimd, "policy must accept eligible SIMD reduction");
    selection.simd_reduction = SimdReduction::Auto;
    selection.cpu_montgomery_chain = CpuMontgomeryChain::Prac;
    caps.prac_bound = false;
    require(!choose(4095) && decisions.find("PRAC requires") != decisions.npos,
            "policy must validate the requested PRAC mode before execution");
    selection.cpu_montgomery_chain = CpuMontgomeryChain::Ladder;
    require(parse_cpu_implementation_policy("simd[:6144]", selection) && !choose(6144),
            "an exhausted rule list must fail without inventing an automatic fallback");
    for (const auto* invalid : {"simd[4096:4096]", "simd[:1]", "simd[0:]", "simd[1:2:3]",
                               "simd[],,auto[]", "simd[],", "unknown[]", "auto:[ ]",
                               "simd[18446744073709551616:]"})
        require(!parse_cpu_implementation_policy(invalid, selection), "invalid CPU range policy was accepted");
    require(parse_cpu_implementation_policy("", selection) && selection.cpu_implementation_rules.empty() &&
                selection.cpu_implementation == CpuImplementation::Auto,
            "an empty CPU implementation must restore the existing automatic mode");
    require(parse_cpu_implementation_policy("auto[]", selection) && choose(12511) == CpuImplementation::IfmaSimd,
            "empty brackets must select automatic execution at any width");

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
