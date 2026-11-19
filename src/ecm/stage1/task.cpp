#include "execution.h"
#include "edwards/memory.h"
#include "edwards/cpu/ecm_edwards_cpu.h"
#include "../../io/stage1_checkpoint.h"
#include "ecm/stage1/selection.h"
#include "ecm/stage1/ntt_target.h"
#include "ecm/io/save.h"
#include "ecm/io/results.h"
#include "../platform/generated/configuration_defaults.h"
#include "ecm/platform/logging.h"
#if ECM_HAS_CUDA_STAGE1
#include "montgomery/cuda/ntt/batches.h"
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "../../io/exponent_cache.h"
#include <sstream>
#include <cstring>
#include <cmath>
#include "../../io/files.h"

namespace ecm_stage1 {
bool compute_batch_s(mpz_t s, double B1, unsigned torsion, const std::string &cache_dir) {
    if (B1 < 2.0) {
        mpz_set_ui(s, torsion);
        return true;
    }
    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }
    /* The exponent is cached on disk (validated on load, see ecm_stage1_exp_cache.h): at
       B1 = 260e6 building it costs ~10 s and every task in a worktodo would repeat that. */
    std::string status;
    if (!ecm_build_lcm_exponent_cached(s, limit64, torsion, cache_dir, &status)) {
        return false;
    }
    ecm_ts_fprintf(stdout, "stage1 exponent built: %s (torsion=%u)\n", status.c_str(), torsion);
    return true;
}

ecm_save::Naming save_naming(const Stage1Options &opt) {
    ecm_save::Naming naming;
    naming.curve = ecm_save::curve_prefix(opt.execution.selection.curve == CurveModel::Edwards,
                                          opt.execution.selection.target == ExecutionTarget::Gpu
                                              ? opt.execution.selection.gpu_param
                                              : 0);
    naming.target_kbnc = opt.identity.target_kbnc;
    naming.target_n = opt.identity.handoff_n;
    naming.mersenne = opt.identity.handoff_k_str == "1" && opt.identity.handoff_b == 2 &&
                      opt.identity.handoff_c == -1;
    return naming;
}


struct TaskResult::State {
    ComputationResult computation;
    TaskStatus status = TaskStatus::Failed;
    FailureSource failure_source = FailureSource::None;
    int task_return = ECM_ERROR;
    ecm_save::Range save_range;
    std::string completed_checkpoint;
};

Stage1ExecutionPreferences::Stage1ExecutionPreferences()
    : stage1_threads(ecm_config::defaults::stage1_stage1_threads),
      naf_w(ecm_config::defaults::stage1_naf_w), device_index(ecm_config::defaults::stage1_device) {
}

Stage1Persistence::Stage1Persistence()
    : save_name_pattern(ecm_config::defaults::stage1_save_name_pattern),
      ckpt_ms(ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS),
      ckpt_backups(ecm_config::defaults::stage1_ckpt_backups) {
}

TaskResult::TaskResult() : data_(std::make_unique<State>()) {
}

TaskResult::~TaskResult() = default;
TaskResult::TaskResult(TaskResult&&) noexcept = default;
TaskResult& TaskResult::operator=(TaskResult&&) noexcept = default;

TaskStatus TaskResult::status() const noexcept {
    return data_ ? data_->status : TaskStatus::Failed;
}

FailureSource TaskResult::failure_source() const noexcept {
    return data_ ? data_->failure_source : FailureSource::Execution;
}

int TaskResult::task_return() const noexcept {
    return data_ ? data_->task_return : ECM_ERROR;
}

int TaskResult::legacy_return() const noexcept {
    return data_ ? data_->computation.legacy_code : ECM_ERROR;
}

uint32_t TaskResult::curves() const noexcept {
    return data_ ? static_cast<uint32_t>(data_->computation.curves.size()) : 0;
}

uint32_t TaskResult::first_sigma() const noexcept {
    return curves() ? uint32_t(sigma(0)) : 0;
}

bool TaskResult::has_factors() const noexcept {
    return curves() != 0;
}

bool TaskResult::has_found() const noexcept {
    return curves() != 0;
}

bool TaskResult::has_sigmas() const noexcept {
    return curves() != 0;
}

mpz_srcptr TaskResult::factor(uint32_t i) const noexcept {
    return i < curves() ? data_->computation.curves[i].value : nullptr;
}

int TaskResult::found(uint32_t i) const noexcept {
    return i < curves() ? (data_->computation.curves[i].kind == CurveOutcome::Kind::Factor
                               ? ECM_FACTOR_FOUND_STEP1
                               : ECM_NO_FACTOR_FOUND)
                        : ECM_ERROR;
}

uint64_t TaskResult::sigma(uint32_t i) const noexcept {
    return i < curves() ? data_->computation.curves[i].sigma : 0;
}

const ecm_save::Range& TaskResult::save_range() const noexcept {
    return data_->save_range;
}

const std::string& TaskResult::completed_checkpoint() const noexcept {
    return data_->completed_checkpoint;
}

// Retained legacy sources must not replay curves already durably published.
static ecm_results::Submission stage1_submission(const Stage1Options& opt, const mpz_t N,
                                                 uint64_t b1) {
    ecm_results::Submission r;
    r.b1 = b1;
    r.b2 = b1;
    r.stage1_only = true;
    r.edwards = opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards;
    r.param = opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                  ? opt.execution.selection.gpu_param
                  : 0;
    r.user = opt.identity.result_user;
    r.computer = opt.identity.result_computer;
    r.aid = opt.identity.result_aid;
    if (opt.identity.target_kbnc) {
        r.kbnc = true;
        r.k = opt.identity.handoff_k_str;
        r.b = opt.identity.handoff_b;
        r.exponent = opt.identity.handoff_n;
        r.c = opt.identity.handoff_c;
        std::istringstream known(opt.identity.handoff_factors);
        std::string f;
        while (std::getline(known, f, ','))
            if (!f.empty())
                r.known.push_back(f);
    }
    // Validate complete metadata against the actual modulus before running.
    (void)ecm_results::target_exponent(r, N);
    return r;
}

static void validate_saved_batch(const ecm_save::Range& range, const Stage1Options& opt,
                                 mpz_srcptr expected, double bound,
                                 const ComputationResult& result) {
    if (!range.valid)
        return;
    const bool ed = opt.execution.selection.curve == CurveModel::Edwards;
    std::vector<uint64_t> sigmas;
    for (const auto& curve : result.curves)
        sigmas.push_back(curve.sigma);
    if (range.records.size() != sigmas.size())
        throw std::runtime_error("saved batch count differs from completed point count");
    ecm_save::validate_batch(range, expected, ed ? uint64_t(bound) : uint64_t(std::llround(bound)),
                             ed ? -1
                             : opt.execution.selection.target == ExecutionTarget::Cpu
                                 ? 0
                                 : opt.execution.selection.gpu_param,
                             sigmas);
}

static std::string save_destination(const Stage1Request& request) {
    if (!request.save_file.empty())
        return ecm_io::resolve_data_path(request.save_file.c_str());
    const auto& opt = request.options;
    if (opt.execution.selection.curve == CurveModel::Montgomery &&
        (opt.execution.selection.target == ExecutionTarget::Gpu || opt.persistence.tmp_dir.empty()))
        return {};
    const auto stem =
        ecm_save::format_name(opt.persistence.save_name_pattern, request.modulus, request.b1, save_naming(opt));
    return opt.persistence.tmp_dir.empty()
               ? stem
               : (std::filesystem::path(opt.persistence.tmp_dir) / stem).string();
}

static std::string encode_saved_batch(const Stage1Request& request,
                                      ComputationResult& result) {
    const auto& selection = request.options.execution.selection;
    if (selection.curve == CurveModel::Edwards) {
        std::string lines;
        ecm_edwards::Integer scalar;
        for (auto& curve : result.curves) {
            const bool factor = curve.kind == CurveOutcome::Kind::Factor;
            if (factor && curve.endpoint.x.empty()) {
                // Construction and dictionary hits have no final endpoint.
                // Replay only the surviving cofactor, with the original curve.
                if (!mpz_sgn(scalar.v) && !compute_batch_s(
                        scalar.v, request.b1, 48, request.options.persistence.exponent_cache_dir))
                    throw std::runtime_error("cannot rebuild Edwards scalar for cofactor endpoint");
                ecm_edwards::Integer cofactor, x, z, hit;
                mpz_divexact(cofactor.v, request.modulus, curve.value);
                const auto rc = edwards_stage1_curve(hit.v, x.v, z.v, cofactor.v, curve.sigma, scalar.v);
                if (rc == 0 || (rc == 1 && mpz_divisible_p(z.v, hit.v)))
                    curve.endpoint = ecm_edwards::save_endpoint(cofactor.v, x.v, z.v);
                if (curve.endpoint.x.empty())
                    ecm_ts_fprintf(stdout, "Stage1 cofactor has no recoverable endpoint: sigma=%llu\n",
                                   static_cast<unsigned long long>(curve.sigma));
            }
            lines += ecm_edwards::save_line(request.modulus, curve.value, curve.sigma, uint64_t(request.b1),
                                            factor, curve.endpoint,
                                            curve.kind == CurveOutcome::Kind::CompletedWithoutPoint);
        }
        return lines;
    }
    std::vector<EcmSavePoint> points;
    points.reserve(result.curves.size());
    for (const auto& curve : result.curves)
        points.push_back({curve.sigma, curve.value});
    return ecm_encode_mont_save(request.modulus, request.b1,
                                selection.target == ExecutionTarget::Cpu ? 0 : selection.gpu_param,
                                points,
                                ecm_save_modulus_expression(request.expression, request.modulus));
}

static std::string checkpoint_path(const Stage1Request& request) {
    if (request.options.persistence.tmp_dir.empty())
        return {};
    ecm_edwards::Integer plus;
    mpz_add_ui(plus.v, request.modulus, 1);
    const auto& id = request.options.identity;
    const auto exponent = id.target_kbnc ? id.handoff_n :
        (mpz_popcount(plus.v) == 1 ? mpz_scan1(plus.v, 0) : 0);
    const auto name = "." + save_naming(request.options).curve + "_s1_" +
        (exponent ? "m" + std::to_string(exponent)
                  : "bit" + std::to_string(mpz_sizeinbase(request.modulus, 2))) +
        "_" + std::to_string(uint64_t(request.b1)) + ".ckpt";
    return (std::filesystem::path(request.options.persistence.tmp_dir) / name)
        .lexically_normal().make_preferred().string();
}

struct CheckpointIdentity {
    std::string current;
    std::vector<std::string> compatible;
};

static CheckpointIdentity checkpoint_identity(const Stage1Request& request,
                                              const ExecutionOptions& options, int default_window) {
    const auto& selection = options.execution.selection;
    const bool edwards = selection.curve == CurveModel::Edwards;
    const auto param = selection.target == ExecutionTarget::Gpu ? selection.gpu_param : 0;
    CheckpointIdentity key;
    auto& identity = key.current;
    identity = ecm_edwards::hex(request.modulus) + "|" +
        (edwards ? "ed" : "mont" + std::to_string(param)) + "|" +
        std::to_string(uint64_t(request.b1)) + "|" +
        std::to_string(int(selection.scalar_construction)) + "|" +
        execution_target_name(selection.target);
    if (selection.target == ExecutionTarget::Cpu) {
        const auto implementation = "|" + std::to_string(int(selection.cpu_implementation));
        if (edwards) {
            // Older envelopes keyed the requested window. The arithmetic
            // payload already records the actual window used by its cursor.
            for (unsigned window = 3; window <= 12; ++window)
                key.compatible.push_back(identity + "|" + std::to_string(window) + implementation);
            identity += "|edwards-cpu-v2" + implementation;
        } else {
            identity += "|" + std::to_string(options.execution.naf_w ? options.execution.naf_w : default_window) +
                        implementation;
        }
        if (!edwards && selection.cpu_montgomery_chain == CpuMontgomeryChain::Prac)
            identity += "|prac-v1";
    } else if (selection.gpu_algorithm == GpuAlgorithm::Ntt) {
        const auto p = ntt_arithmetic_bits(request.modulus, request.options);
        const auto carrier = p && mpz_scan0(request.modulus, 0) != p
            ? "|ntt-carrier-v1|" + std::to_string(p) : std::string{};
        if (edwards) {
            // Legacy task identities included the requested encoding. The native payload
            // already records the encoding used for the active batch.
            key.compatible.push_back(identity + "|ntt-v1" + carrier);
            for (unsigned w = 2; w <= 12; ++w)
                key.compatible.push_back(identity + "|ntt-edwards-naf-v1|" + std::to_string(w) + carrier);
        }
        identity += (edwards ? "|ntt-edwards-v2" : "|ntt-v1") + carrier;
    } else if (selection.gpu_algorithm == GpuAlgorithm::Thread) {
        identity += "|cuda-edwards-thread-v1";
    }
#if !ECM_HAS_CUDA_STAGE1
    else if (edwards)
        identity += "|opencl-edwards-naf-v1";
#endif
    else if (!edwards)
        identity += "|" + options.cgbn_algorithm;
    return key;
}

#if ECM_HAS_CUDA_STAGE1
static CheckpointIdentity ntt_task_identity(CheckpointIdentity key, uint32_t curves) {
    const auto suffix = "|ntt-task-v1|" + std::to_string(curves);
    key.current += suffix;
    for (auto& legacy : key.compatible)
        legacy += suffix;
    return key;
}

std::optional<GpuAlgorithm> stage1_checkpoint_gpu_algorithm(const Stage1Request& request) {
    const auto path = checkpoint_path(request);
    if (path.empty()) return {};
    const auto backups = request.options.persistence.ckpt_backups;
    std::error_code error;
    bool has_file = std::filesystem::exists(path, error);
    for (unsigned i = 1; !has_file && i <= backups; ++i)
        has_file = std::filesystem::exists(ecm_stage1_backup_path(path, i), error);
    if (!has_file) return {};
    auto options = request.options;
    for (const auto algorithm : {GpuAlgorithm::Ntt, GpuAlgorithm::Cgbn, GpuAlgorithm::Thread}) {
        options.execution.selection.gpu_algorithm = algorithm;
        const auto id = checkpoint_identity(request, ExecutionOptions(options), 0);
        const auto count = !request.curves && algorithm == GpuAlgorithm::Ntt ? 1u : request.curves;
        if (algorithm == GpuAlgorithm::Ntt) {
            const auto split = ntt_task_identity(id, count);
            if (ecm_stage1_checkpoint::matches(path, split.current, 1, backups, split.compatible))
                return algorithm;
        }
        if (ecm_stage1_checkpoint::matches(path, id.current, count, backups, id.compatible))
            return algorithm;
    }
    return {};
}
#endif

static ecm_results::Stage1BatchRecord describe_result(
    const Stage1Request& request, ecm_results::Submission row, const CurveOutcome& curve,
    uint32_t index, const std::string& when, const std::string& publication_task,
    double seconds, const TaskDependencies& deps) {
    row.publication_task = publication_task;
    row.sigma = curve.sigma;
    row.when = when;
    if (curve.kind == CurveOutcome::Kind::Factor) {
        row.stage = 1;
        row.factors.push_back(ecm_results::decimal(curve.value));
    }
    std::ostringstream extra;
    extra << ",\"stage1-engine\":{\"curve\":" << index << ",\"target\":"
          << ecm_results::quoted(execution_target_name(request.options.execution.selection.target))
          << ",\"batch-seconds\":" << seconds << ",\"coverage\":\"B1\"}";
    auto internal = ecm_results::internal_json(row, request.modulus, 0, deps.build_identity(), extra.str());
    return {std::move(row), std::move(internal)};
}

TaskResult run_stage1_task(const Stage1Request& request, const TaskDependencies& deps) {
    const auto N = request.modulus;
    const double B1 = request.b1, B2 = request.b2;
    const uint32_t curves = request.curves;
    const auto &savefilename = request.save_file, &n_expr = request.expression;
    const bool saveappend = request.append_save;
    const Stage1Options& opt = request.options;
    TaskResult result;
    auto* out = result.data_.get(); // Private execution payload; caller receives an owner.
    ecm_stage1_checkpoint::Context session;
    auto phase = FailureSource::Preparation;
    try {
        ExecutionOptions execution(opt);
        auto& selection = execution.execution.selection;
        if (selection.target == ExecutionTarget::Cpu && !selection.cpu_implementation_rules.empty()) {
            if (!deps.resolve_cpu_policy)
                throw std::runtime_error("Stage1 CPU policy resolver is unavailable");
            if (!deps.resolve_cpu_policy(request, selection)) {
                out->failure_source = phase;
                return result;
            }
        }
        const auto base = stage1_submission(opt, N, uint64_t(B1));
        if (base.edwards && (selection.target == ExecutionTarget::Cpu || selection.gpu_algorithm != GpuAlgorithm::Ntt)) {
            using namespace ecm_edwards_memory;
            const auto available = host_free_bytes();
            const auto required = batch_host_bytes(curves, mpz_sizeinbase(N, 2));
            if (available && required > usable_bytes(*available, 64 * mib))
                throw std::runtime_error("Edwards Stage1: insufficient host memory for the requested curve batch (estimate " +
                                         std::to_string(required) + " bytes); use a smaller batch");
        }
        (void)ecm_results::submission_json(base, N);
        const auto same = ecm_io::same_file_resource;
        if (same(opt.persistence.results_file, opt.persistence.submission_file) ||
            same(save_destination(request), opt.persistence.results_file) ||
            same(save_destination(request), opt.persistence.submission_file))
            throw std::runtime_error(
                "Stage1 save, internal results and submission paths must differ");
        ecm_results::check_stage1_outputs(opt.persistence.results_file,
                                          opt.persistence.submission_file);
        const auto destination = save_destination(request);
        if (!destination.empty() &&
            !opencl_ecm_check_save_file_writable(destination, saveappend || savefilename.empty())) {
            phase = FailureSource::Publication;
            throw std::runtime_error("cannot prepare Stage1 save destination " + destination);
        }
        const auto path = checkpoint_path(request);
        if (!path.empty())
            std::filesystem::create_directories(opt.persistence.tmp_dir);
        session.set_backup_limit(opt.persistence.ckpt_backups);
        session.set_path(path);
        auto identity = checkpoint_identity(request, execution,
            selection.target == ExecutionTarget::Cpu && !opt.execution.naf_w ? deps.default_naf_window() : 0);
        bool split_ntt = false;
#if ECM_HAS_CUDA_STAGE1
        if (selection.target == ExecutionTarget::Gpu && selection.gpu_algorithm == GpuAlgorithm::Ntt) {
            split_ntt = curves > stage1_ntt_max_curves || opt.persistence.retain_completed_checkpoint;
            if (split_ntt && curves <= stage1_ntt_max_curves &&
                ecm_stage1_checkpoint::matches(path, identity.current, curves, opt.persistence.ckpt_backups,
                                               identity.compatible))
                split_ntt = false; // Resume an existing one-batch queue checkpoint.
        }
#endif
        if (!split_ntt)
            session.begin(identity.current, curves, opt.identity.sigma_fixed, opt.identity.fixed_sigma64,
                          deps.defer_gpu_checkpoint_validation &&
                              opt.execution.selection.target == ExecutionTarget::Gpu, identity.compatible);
        ecm_save::copy_history(opt.persistence.legacy_save_source, savefilename,
            opt.execution.selection.curve == CurveModel::Edwards ? -1 :
            (opt.execution.selection.target == ExecutionTarget::Gpu ? opt.execution.selection.gpu_param : 0));
        const auto begin = std::chrono::steady_clock::now();
        phase = FailureSource::Execution;
        const auto result_identity = ecm_edwards::hex(N) + "|" + std::to_string(base.b1) +
                                     (base.edwards ? "ed" : "mont" + std::to_string(base.param));
#if ECM_HAS_CUDA_STAGE1
        if (split_ntt) {
            const auto split = ntt_task_identity(std::move(identity), curves);
            out->computation = ecm_cuda_stage1::run_ntt_batches(
                request, deps, session, split.current, split.compatible,
                [&](const CurveOutcome& factor, uint32_t index) {
                    const auto key = result_identity + "|early-factor|" + std::to_string(factor.sigma);
                    const auto task = std::to_string(ecm_io::fingerprint(key));
                    const auto seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                    std::vector<ecm_results::Stage1BatchRecord> early;
                    early.push_back(describe_result(request, base, factor, index,
                        ecm_results::utc_timestamp(), task, seconds, deps));
                    ecm_results::publish_stage1_batch(opt.persistence.results_file,
                        opt.persistence.submission_file, std::move(early), N, key);
                });
        } else
#endif
            out->computation = deps.execute(request, execution, session);
        auto& computed = out->computation;
        if (computed.completion != ComputationResult::Completion::Completed) {
            out->status = computed.completion == ComputationResult::Completion::Stopped
                              ? TaskStatus::Stopped
                              : TaskStatus::Failed;
            out->failure_source = computed.failure;
            session.close(false);
            return result;
        }
        computed.validate(N);
        if (split_ntt ? computed.curves.size() < curves : computed.curves.size() != curves)
            throw std::runtime_error("Stage1 result count differs from requested curves");
        phase = FailureSource::Publication;
        if (!destination.empty()) {
            out->save_range =
                ecm_save::publish_batch(destination, encode_saved_batch(request, computed), true);
            ecm_ts_fprintf(stdout, "Stage1 save: %s (%zu saved curves)\n", destination.c_str(),
                           out->save_range.records.size());
            validate_saved_batch(out->save_range, opt, N, B1, computed);
        }
        const auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::vector<ecm_results::Stage1BatchRecord> records;
        auto task_identity = result_identity;
        for (const auto& curve : computed.curves)
            task_identity += "|" + std::to_string(curve.sigma);
        const auto publication_task = std::to_string(ecm_io::fingerprint(task_identity));
        const auto publication_when = ecm_results::utc_timestamp();
        for (uint32_t i = 0; i < computed.curves.size(); ++i)
            records.push_back(describe_result(request, base, computed.curves[i], i,
                publication_when, publication_task, seconds, deps));
        ecm_results::publish_stage1_batch(opt.persistence.results_file,
                                          opt.persistence.submission_file, std::move(records), N,
                                          task_identity);
        if (opt.persistence.retain_completed_checkpoint)
            out->completed_checkpoint = path;
        session.close(!opt.persistence.retain_completed_checkpoint);
        ecm_ts_fprintf(stdout, "Stage1 results: internal=%s submission=%s\n",
                       opt.persistence.results_file.c_str(),
                       opt.persistence.submission_file.c_str());
        out->status = TaskStatus::Completed;
        out->task_return = computed.legacy_code;
        return result;
    } catch (const std::exception& e) {
        if (const auto* checkpoint = dynamic_cast<const ecm_stage1_checkpoint::Error*>(&e))
            phase = checkpoint->state == ecm_stage1_checkpoint::RestoreState::IoFailure
                        ? FailureSource::Persistence
                        : FailureSource::Preparation;
        out->status = TaskStatus::Failed;
        out->failure_source = phase;
        session.close(false);
        ecm_ts_fprintf(stderr, "ERROR: Stage1: %s\n", e.what());
        out->task_return = ECM_ERROR;
        return result;
    }
}

} // namespace ecm_stage1
