#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ecm_stage1 {

enum class CurveModel { Montgomery, Edwards };
enum class ExecutionTarget { Cpu, Gpu };
enum class CpuImplementation { Auto = 0, IfmaSimd = 1, GmpScalar = 2, Gwnum = 3 };
enum class SimdReduction { Auto = 0, Montgomery = 1, MersenneFold = 2 };
enum class CpuMontgomeryChain { Ladder, Prac };
enum class GpuAlgorithm { Cgbn = 1, Ntt = 2 };
enum class ScalarConstruction { PrimePowerLcm, PrimePowerLcmTimes12 };

struct CpuImplementationRule {
    CpuImplementation implementation;
    std::size_t first_bit;
    std::optional<std::size_t> end_bit;
};

struct Selection {
    CurveModel curve = CurveModel::Montgomery;
    ExecutionTarget target = ExecutionTarget::Gpu;
    CpuImplementation cpu_implementation = CpuImplementation::Auto;
    bool strict_param = false;
    std::vector<CpuImplementationRule> cpu_implementation_rules;
    SimdReduction simd_reduction = SimdReduction::Auto;
    CpuMontgomeryChain cpu_montgomery_chain = CpuMontgomeryChain::Ladder;
    ScalarConstruction scalar_construction = ScalarConstruction::PrimePowerLcm;
    GpuAlgorithm gpu_algorithm = GpuAlgorithm::Cgbn;
    int gpu_param = -1; // assigned from the generated config before GPU validation
};

// Stable construction/execution vocabulary shared by application and workflows.
std::string lower_ascii(std::string_view);
bool parse_curve_model(std::string_view,CurveModel &);
bool parse_execution_target(std::string_view,ExecutionTarget &);
bool parse_cpu_implementation(std::string_view,CpuImplementation &);
bool parse_cpu_implementation_policy(std::string_view, Selection&);
bool parse_simd_reduction(std::string_view,SimdReduction &);
bool parse_cpu_montgomery_chain(std::string_view, CpuMontgomeryChain&);
bool parse_gpu_algorithm(std::string_view,GpuAlgorithm &);
bool parse_scalar_construction(std::string_view,ScalarConstruction &);
std::optional<CpuImplementation> resolve_cpu_implementation(
    const Selection&, std::size_t modulus_bits, std::uint32_t curves, bool has_ifma);
using GwnumProbe = const char* (*)(); // Null result means the DLL is usable.
struct CpuImplementationCapabilities {
    bool has_ifma;
    bool odd_modulus;
    bool mersenne_fold;
    bool prac_bound;
    GwnumProbe gwnum;
};
std::optional<CpuImplementation> resolve_cpu_implementation_policy(
    const Selection&, std::size_t modulus_bits, std::uint32_t curves,
    const CpuImplementationCapabilities&, std::string& decisions);
bool valid_route(const Selection &,bool gpu_edwards=false);
bool valid_gpu_param(int);
bool resolve_gpu_param(int configured,bool explicit_param,int sigma_claim,int &resolved);
const char *curve_model_name(CurveModel);
const char *execution_target_name(ExecutionTarget);

} // namespace ecm_stage1
