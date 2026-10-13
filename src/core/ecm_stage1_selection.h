#pragma once

#include <string>
#include <string_view>

// Mathematical and execution choices are independent.
namespace ecm_stage1 {

enum class CurveModel { Montgomery, Edwards };
enum class ExecutionTarget { Cpu, Gpu };
enum class CpuImplementation { Auto = 0, IfmaSimd = 1, GmpScalar = 2 };
enum class SimdReduction { Auto = 0, Montgomery = 1, MersenneFold = 2 };
enum class ScalarConstruction { PrimePowerLcm, PrimePowerLcmTimes12 };

struct Selection {
    CurveModel curve = CurveModel::Montgomery;
    ExecutionTarget target = ExecutionTarget::Gpu;
    CpuImplementation cpu_implementation = CpuImplementation::Auto;
    SimdReduction simd_reduction = SimdReduction::Auto;
    ScalarConstruction scalar_construction = ScalarConstruction::PrimePowerLcm;
    int gpu_param = -1; // assigned from the generated config before GPU validation
};

inline std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char &ch : out) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return out;
}

inline bool parse_curve_model(std::string_view value, CurveModel &out) {
    const std::string v = lower_ascii(value);
    if (v == "mont") out = CurveModel::Montgomery;
    else if (v == "ed") out = CurveModel::Edwards;
    else return false;
    return true;
}

inline bool parse_execution_target(std::string_view value, ExecutionTarget &out) {
    const std::string v = lower_ascii(value);
    if (v == "cpu") out = ExecutionTarget::Cpu;
    else if (v == "gpu") out = ExecutionTarget::Gpu;
    else return false;
    return true;
}

inline bool parse_cpu_implementation(std::string_view value, CpuImplementation &out) {
    const std::string v = lower_ascii(value);
    if (v == "auto") out = CpuImplementation::Auto;
    else if (v == "simd") out = CpuImplementation::IfmaSimd;
    else if (v == "scalar") out = CpuImplementation::GmpScalar;
    else return false;
    return true;
}

inline bool parse_simd_reduction(std::string_view value, SimdReduction &out) {
    const std::string v = lower_ascii(value);
    if (v == "auto") out = SimdReduction::Auto;
    else if (v == "mersenne-fold")
        out = SimdReduction::MersenneFold;
    else if (v == "montgomery")
        out = SimdReduction::Montgomery;
    else return false;
    return true;
}

inline bool parse_scalar_construction(std::string_view value, ScalarConstruction &out) {
    const std::string v = lower_ascii(value);
    if (v == "1") out = ScalarConstruction::PrimePowerLcm;
    else if (v == "12")
        out = ScalarConstruction::PrimePowerLcmTimes12;
    else return false;
    return true;
}

inline bool valid_route(const Selection &selection) {
    return selection.target != ExecutionTarget::Gpu ||
           selection.curve == CurveModel::Montgomery;
}

inline bool valid_gpu_param(int value) { return value == 0 || value == 2 || value == 3; }

// -sigma i:s selects i unless --gpu-param explicitly selects another value.
inline bool resolve_gpu_param(int configured, bool explicit_param, int sigma_claim,
                              int &resolved) {
    if (!valid_gpu_param(configured) ||
        (sigma_claim >= 0 && !valid_gpu_param(sigma_claim))) return false;
    if (sigma_claim >= 0 && explicit_param && sigma_claim != configured) return false;
    resolved = sigma_claim >= 0 ? sigma_claim : configured;
    return true;
}

inline const char *curve_model_name(CurveModel curve) {
    return curve == CurveModel::Edwards ? "ed" : "mont";
}

inline const char *execution_target_name(ExecutionTarget target) {
    return target == ExecutionTarget::Gpu ? "gpu" : "cpu";
}

} // namespace ecm_stage1
