#pragma once

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

} // namespace ecm_stage1
