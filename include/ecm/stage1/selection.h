#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ecm_stage1 {

enum class CurveModel { Montgomery, Edwards };
enum class ExecutionTarget { Cpu, Gpu };
enum class CpuImplementation { Auto = 0, IfmaSimd = 1, GmpScalar = 2 };
enum class SimdReduction { Auto = 0, Montgomery = 1, MersenneFold = 2 };
enum class GpuAlgorithm { Cgbn = 1, Ntt = 2 };
enum class ScalarConstruction { PrimePowerLcm, PrimePowerLcmTimes12 };

struct Selection {
    CurveModel curve = CurveModel::Montgomery;
    ExecutionTarget target = ExecutionTarget::Gpu;
    CpuImplementation cpu_implementation = CpuImplementation::Auto;
    SimdReduction simd_reduction = SimdReduction::Auto;
    ScalarConstruction scalar_construction = ScalarConstruction::PrimePowerLcm;
    GpuAlgorithm gpu_algorithm = GpuAlgorithm::Cgbn;
    int gpu_param = -1; // assigned from the generated config before GPU validation
};

// Stable construction/execution vocabulary shared by application and workflows.
std::string lower_ascii(std::string_view);
bool parse_curve_model(std::string_view,CurveModel &);
bool parse_execution_target(std::string_view,ExecutionTarget &);
bool parse_cpu_implementation(std::string_view,CpuImplementation &);
bool parse_simd_reduction(std::string_view,SimdReduction &);
bool parse_gpu_algorithm(std::string_view,GpuAlgorithm &);
bool parse_scalar_construction(std::string_view,ScalarConstruction &);
std::optional<CpuImplementation> resolve_cpu_implementation(
    const Selection&, std::size_t modulus_bits, std::uint32_t curves, bool has_ifma);
bool valid_route(const Selection &,bool gpu_edwards=false);
bool valid_gpu_param(int);
bool resolve_gpu_param(int configured,bool explicit_param,int sigma_claim,int &resolved);
const char *curve_model_name(CurveModel);
const char *execution_target_name(ExecutionTarget);

} // namespace ecm_stage1
