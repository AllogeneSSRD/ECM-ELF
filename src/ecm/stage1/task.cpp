#include "execution_contract.h"
#include "../../io/stage1_checkpoint.h"
#include "ecm/stage1/selection.h"
#include "ecm/io/save.h"
#include "ecm/io/results.h"
#include "../platform/generated/configuration_defaults.h"
#include "ecm/platform/logging.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "preparation.h"
#include <sstream>
#include <cstring>
#include <cmath>
#include "../../io/files.h"

namespace ecm_stage1 {
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
        if (curve.kind != CurveOutcome::Kind::CompletedWithoutPoint)
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
                                      const ComputationResult& result) {
    const auto& selection = request.options.execution.selection;
    if (selection.curve == CurveModel::Edwards) {
        std::string lines;
        for (const auto& curve : result.curves)
            if (curve.kind != CurveOutcome::Kind::CompletedWithoutPoint)
                lines += ecm_edwards::save_line(request.modulus, curve.value, curve.sigma,
                                                uint64_t(request.b1),
                                                curve.kind == CurveOutcome::Kind::Factor);
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
        const auto base = stage1_submission(opt, N, uint64_t(B1));
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
        ecm_edwards::Integer plus;
        mpz_add_ui(plus.v, N, 1);
        const auto exponent = opt.identity.target_kbnc
                                  ? opt.identity.handoff_n
                                  : (mpz_popcount(plus.v) == 1 ? mpz_scan1(plus.v, 0) : 0);
        const auto name = std::string(base.edwards ? ".e" : ".m" + std::to_string(base.param)) +
                          "_s1_" +
                          (exponent ? "m" + std::to_string(exponent)
                                    : "bit" + std::to_string(mpz_sizeinbase(N, 2))) +
                          "_" + std::to_string(uint64_t(B1)) + ".ckpt";
        const auto path = opt.persistence.tmp_dir.empty()
                              ? std::string{}
                              : (std::filesystem::path(opt.persistence.tmp_dir) / name)
                                    .lexically_normal()
                                    .make_preferred()
                                    .string();
        if (!path.empty())
            std::filesystem::create_directories(opt.persistence.tmp_dir);
        session.set_backup_limit(opt.persistence.ckpt_backups);
        session.set_path(path);
        ExecutionOptions execution(opt);
        {
            auto identity = ecm_edwards::hex(N) + "|" +
                            (base.edwards ? "ed" : "mont" + std::to_string(base.param)) + "|" +
                            std::to_string(uint64_t(B1)) + "|" +
                            std::to_string(int(opt.execution.selection.scalar_construction)) + "|" +
                            ecm_stage1::execution_target_name(opt.execution.selection.target);
            if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Cpu)
                identity += "|" +
                            std::to_string(opt.execution.naf_w ? opt.execution.naf_w
                                                               : deps.default_naf_window()) +
                            "|" + std::to_string(int(opt.execution.selection.cpu_implementation));
            else if (!base.edwards) {
                identity += "|" + execution.gpu_algorithm;
            }
            session.begin(identity, curves, opt.identity.sigma_fixed, opt.identity.fixed_sigma64,
                          deps.defer_gpu_checkpoint_validation &&
                              opt.execution.selection.target == ExecutionTarget::Gpu);
        }
        ecm_save::copy_history(opt.persistence.legacy_save_source, savefilename,
            opt.execution.selection.curve == CurveModel::Edwards ? -1 :
            (opt.execution.selection.target == ExecutionTarget::Gpu ? opt.execution.selection.gpu_param : 0));
        const auto begin = std::chrono::steady_clock::now();
        phase = FailureSource::Execution;
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
        if (computed.curves.size() != curves)
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
        std::string task_identity = ecm_edwards::hex(N) + "|" + std::to_string(base.b1) +
                                    (base.edwards ? "ed" : "mont" + std::to_string(base.param));
        for (const auto& curve : computed.curves)
            task_identity += "|" + std::to_string(curve.sigma);
        const auto publication_task = std::to_string(ecm_io::fingerprint(task_identity));
        const auto publication_when = ecm_results::utc_timestamp();
        for (uint32_t i = 0; i < computed.curves.size(); ++i) {
            auto r = base;
            r.publication_task = publication_task;
            r.sigma = computed.curves[i].sigma;
            r.when = publication_when;
            if (computed.curves[i].kind == CurveOutcome::Kind::Factor) {
                r.stage = 1;
                r.factors.push_back(ecm_results::decimal(computed.curves[i].value));
            }
            std::ostringstream extra;
            extra << ",\"stage1-engine\":{\"curve\":" << i << ",\"target\":"
                  << ecm_results::quoted(
                         ecm_stage1::execution_target_name(opt.execution.selection.target))
                  << ",\"batch-seconds\":" << seconds << ",\"coverage\":\"B1\"}";
            records.push_back(
                {r, ecm_results::internal_json(r, N, 0, deps.build_identity(), extra.str())});
        }
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
