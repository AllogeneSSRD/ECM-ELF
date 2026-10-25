#pragma once
#include "ecm/service/stage1/task.h"
#include "../execution/ecm.h"
#include <vector>

// Host computation interface and private compatibility storage for old kernels.
namespace ecm_stage1_checkpoint { class Context; }

namespace ecm_stage1 {

// Private mutable execution state. The caller's resolved options remain immutable.
struct ExecutionOptions : Stage1Options {
    explicit ExecutionOptions(const Stage1Options& options);
    long long stage1_t0_ms = 0;
    uint32_t stage1_curves = 0;
    // Same owned selection is used by task identity and the CUDA workflow.
    std::string gpu_algorithm;
};

enum class CurveKind { Legacy, Factor, Stage2Point, CompletedWithoutPoint, Degenerate, Failed };

struct LegacyExecutionResult {
    using CurveKind = ecm_stage1::CurveKind;
    std::vector<CurveKind> curve_kinds;
    // Legacy CPU/OpenCL collaborators may still report through the old flags.
    // CUDA workflows report failure or stop explicitly, before returning.
    enum class Completion { Legacy, Completed, Stopped, SampleLimit, Failed };
    Completion completion = Completion::Legacy;
    TaskStatus status = TaskStatus::Failed;
    FailureSource failure_source = FailureSource::None;
    int task_return = ECM_ERROR; // Preserve the task/legacy return-code distinction.
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    bool publish_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    uint64_t firstsigma64 = 0; // full 64-bit sigma
    mpz_t* factors = nullptr;
    int* array_found = nullptr;
    LegacyExecutionResult() = default;
    LegacyExecutionResult(const LegacyExecutionResult&) = delete;
    LegacyExecutionResult& operator=(const LegacyExecutionResult&) = delete;
    LegacyExecutionResult(LegacyExecutionResult&& other) noexcept;
    LegacyExecutionResult& operator=(LegacyExecutionResult&& other) noexcept;
    ~LegacyExecutionResult();
    void allocate_curves(uint32_t count);
    void set_curve(uint32_t, CurveKind, mpz_srcptr, uint64_t sigma);
    void validate_curves(mpz_srcptr modulus) const;
    void fail(FailureSource source) noexcept;
    void stop() noexcept;
    void finish_task(int code, bool stopped) noexcept;
    void swap(LegacyExecutionResult& other) noexcept;
    // Per-curve sigma of THIS batch (malloc'd, `curves` entries, owned by this result; may be
    // null when a path does not know them). Needed because one batch does NOT always
    // share a sigma sequence: the Edwards path draws a random 64-bit sigma per curve,
    // so "firstsigma + i" would be wrong there. Consumers: the hit lines (D3, see
    // docs/usage/GUI.md) and therefore the GUI's results.txt sigma list.
    uint64_t* sigmas = nullptr;
};

// Computation owns one classified value per attempted curve. Raw arrays above
// are compatibility storage for old kernels/CPU workers and never reach task.
struct CurveOutcome {
    using Kind = CurveKind;
    Kind kind = Kind::Failed;
    uint64_t sigma = 0;
    mpz_t value;

    CurveOutcome() {
        mpz_init(value);
    }

    ~CurveOutcome() {
        mpz_clear(value);
    }

    CurveOutcome(CurveOutcome&& other) noexcept : kind(other.kind), sigma(other.sigma) {
        mpz_init(value);
        mpz_swap(value, other.value);
    }

    CurveOutcome& operator=(CurveOutcome&& other) noexcept {
        kind = other.kind;
        sigma = other.sigma;
        mpz_swap(value, other.value);
        return *this;
    }

    CurveOutcome(const CurveOutcome&) = delete;
    CurveOutcome& operator=(const CurveOutcome&) = delete;
};

struct ComputationResult {
    enum class Completion { Completed, Stopped, Failed };
    Completion completion = Completion::Failed;
    FailureSource failure = FailureSource::None;
    int legacy_code = ECM_ERROR;
    std::vector<CurveOutcome> curves;
    void set_curve(uint32_t, CurveKind, mpz_srcptr, uint64_t sigma);
    void fail(FailureSource) noexcept;
    void stop() noexcept;
    void validate(mpz_srcptr) const;
};

// A single compatibility conversion owns status mapping and moves GMP values.
ComputationResult take_computation_result(LegacyExecutionResult&&, int code, bool stopped);
using Stage1Entry = ComputationResult (*)(const Stage1Request&, ExecutionOptions&, ecm_stage1_checkpoint::Context&);

struct TaskDependencies {
    Stage1Entry execute;
    const std::string& (*build_identity)();
    int (*default_naf_window)();
    bool defer_gpu_checkpoint_validation = false;
};

bool validate_stage1_factor(mpz_srcptr modulus, mpz_srcptr factor) noexcept;
} // namespace ecm_stage1
