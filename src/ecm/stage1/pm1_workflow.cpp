#include "pm1_workflow.h"
#include "cuda/ntt/pm1.h"
#include "cuda/ntt/tune.h"
#include "cuda/ntt/ladder.h"
#include "ecm/integer.h"
#include "ecm/stage1/scalar.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/stop.h"
#include "io/exponent_cache.h"
#include "io/files.h"
#include "io/pm1_saves.h"
#include <algorithm>
#include <chrono>
#include <charconv>
#include <fstream>
#include <set>
#include <map>
#include <string_view>

namespace ecm_pm1 {
using ecm_edwards::Integer;
using ecm_mersenne_ntt::Pm1Progress;

std::vector<Pm1Task> batch_prefix(const std::string& path, size_t source_line, Pm1Task first,
                                  const ecm_stage1::Stage1NttOptions& options) {
    std::vector<Pm1Task> tasks{std::move(first)};
    if (options.max_batch<=1 || options.tune_file.empty()) return tasks;
    std::ifstream in(path);
    std::string raw;
    for (size_t n=0;n<source_line;++n)
        if (!std::getline(in,raw)) return tasks;
    if (ecm_queue::task_text(raw)!=tasks.front().raw_line) return tasks;
    std::set<unsigned> moduli{tasks.front().p};
    while (tasks.size()<options.max_batch && std::getline(in,raw)) {
        const auto kind=ecm_queue::classify_line(raw,nullptr,nullptr);
        if (kind==ecm_queue::LineKind::Comment) continue;
        if (kind!=ecm_queue::LineKind::Task) break;
        const auto text=ecm_queue::task_text(raw);
        Pm1Task next;
        std::string error;
        if (!ecm_is_pm1_line(text) || !ecm_parse_pm1_line(text,next,error)) break;
        if (!moduli.insert(next.p).second) break; // A later bound can depend on this lane's endpoint.
        tasks.push_back(std::move(next));
    }
    if (in.bad()) throw std::runtime_error("cannot read Pminus1 queue prefix: "+path);
    return tasks;
}

void BatchProgress::start(size_t index, uint64_t done, uint64_t total, uint64_t old_b1,
                          uint64_t target_b1, const char* phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (begin_==Clock::time_point{}) begin_=Clock::now();
    auto& lane = lanes_.at(index);
    const auto start_cursor = lane.phase ? lane.start_cursor : done;
    lane={done,total,old_b1,target_b1,start_cursor,phase,ecm_stage1_gpu_progress::SpeedWindow{}};
    report(true);
}

void BatchProgress::update(size_t index, uint64_t done, double block_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& lane=lanes_.at(index);
    lane.speed.add(double(done-lane.cursor),block_ms);
    lane.cursor=done;
    report(done==lane.total);
}

void BatchProgress::report(bool force) {
    const auto now=Clock::now();
    if (!force && now-last_<std::chrono::milliseconds(250)) return;
    uint64_t done=0, total=0;
    double eta=0, rate=0, task_equivalents=0;
    bool known=true, finishing_old=false;
    for (const auto& lane:lanes_) {
        if (!lane.phase) return;
        done+=lane.cursor;
        total+=lane.total;
        if (lane.total > lane.start_cursor)
            task_equivalents += double(lane.cursor - lane.start_cursor) / double(lane.total - lane.start_cursor);
        finishing_old=finishing_old || std::string_view(lane.phase)=="finish_old";
        if (lane.cursor<lane.total) {
            rate+=lane.speed.speed();
            known=known && lane.speed.speed()>0;
            eta=std::max(eta,lane.speed.remaining(lane.total,lane.cursor));
        }
    }
    // If K is 1, the old phase reaches the final cursor before the phase change.
    if (done==total && finishing_old) return;
    const double elapsed=std::chrono::duration<double>(now-begin_).count();
    const double seconds_per_task = task_equivalents > 0 ? elapsed / task_equivalents : 0;
    char remaining[64]="--";
    if (known) std::snprintf(remaining,sizeof(remaining),"%.1fs",eta);
    const auto& first=lanes_.front();
    ecm_ts_fprintf(stdout,"pm1_stage1_progress: iter=%llu/%llu %.2f%% iter_per_s=%.3f elapsed=%.1fs ETA=%s"
                   " first_iter=%llu old_B1=%llu target_B1=%llu phase=%s s_per_curve=%.3f\n",
                   (unsigned long long)done,(unsigned long long)total,total ? 100.*done/total : 100.,
                   1000.*rate,elapsed,remaining,(unsigned long long)first.total,
                   (unsigned long long)first.old_b1,(unsigned long long)first.target_b1,first.phase,seconds_per_task);
    last_=now;
}

namespace {
namespace fs = std::filesystem;

struct Phase {
    fs::path root;
    uint64_t bound;
    std::optional<OwlSave> owl;
    std::optional<ExtensionSave> extension;
    std::vector<uint32_t> exponent;
    uint64_t bits = 0;

    uint64_t cursor() const { return extension ? extension->k : owl ? owl->k : 1; }
};

bool load_phase(Phase& phase, unsigned p) {
    phase.extension = load_extension(phase.root, p, phase.bound);
    if (!phase.extension)
        phase.owl = load_owl(phase.root, p, phase.bound, 0);
    if (phase.owl && phase.owl->k == 1)
        phase.owl.reset(); // Only the initial base has been loaded.
    return phase.extension.has_value() || phase.owl.has_value();
}

std::map<uint64_t, fs::path, std::greater<uint64_t>> previous_roots(const fs::path& root,
                                                                    unsigned p, uint64_t bound) {
    std::map<uint64_t, fs::path, std::greater<uint64_t>> roots;
    const auto prefix = "M" + std::to_string(p) + "-B1-";
    for (const auto& entry : fs::directory_iterator(root.parent_path())) {
        const auto name = entry.path().filename().u8string();
        if (name.compare(0, prefix.size(), prefix) || !fs::is_directory(entry.symlink_status()))
            continue;
        uint64_t value = 0;
        const auto end = name.data() + name.size();
        const auto parsed = std::from_chars(name.data() + prefix.size(), end, value);
        if (parsed.ec == std::errc{} && parsed.ptr == end && value >= 2 && value < bound)
            roots.emplace(value, entry.path());
    }
    return roots;
}

void prepare(Phase& phase, const Pm1Task& task, const ecm_stage1::Stage1Options& options) {
    Integer exponent;
    if (phase.extension) {
        if (!ecm_build_lcm_extension(exponent.v, phase.extension->from, phase.bound))
            throw std::runtime_error("cannot build P-1 extension exponent");
    } else {
        std::string detail;
        if (!ecm_build_lcm_exponent_cached(exponent.v, phase.bound, 1,
                                           options.persistence.exponent_cache_dir, &detail))
            throw std::runtime_error("cannot build Pminus1 exponent: " + detail);
        mpz_mul_ui(exponent.v, exponent.v, 2UL * task.p);
    }
    phase.bits = mpz_sizeinbase(exponent.v, 2);
    if (phase.extension && phase.extension->bits && phase.extension->bits != phase.bits)
        throw std::runtime_error("P-1 extension checkpoint exponent mismatch");
    if (phase.owl && phase.owl->bits != phase.bits)
        phase.owl = load_owl(phase.root, task.p, phase.bound, phase.bits);
    phase.exponent.resize((phase.bits + 31) / 32);
    mpz_export(phase.exponent.data(), nullptr, -1, 4, 0, 0, exponent.v);
}

Pm1Progress advance(Phase& phase, const Pm1Task& task, uint64_t b2,
                    const ecm_stage1::Stage1Options& options, BatchProgress& progress, size_t lane,
                    uint64_t offset, uint64_t pending, uint64_t old_b1, uint64_t skipped = 1) {
    Pm1Progress state;
    double previous_seconds = 0;
    if (phase.extension) {
        auto& ext = *phase.extension;
        ext.bits = phase.bits;
        if (ext.residue.empty()) {
            ext.residue = ext.base;
            write_extension(phase.root, task.p, phase.bound, ext);
        }
        state = {ext.k, std::move(ext.residue), ext.errors};
        ecm_ts_fprintf(stdout,
                       "P-1 Stage1 extend M%u B1=%llu -> %llu K_bits=%llu k=%llu/%llu lane=%zu\n",
                       task.p, (unsigned long long)ext.from, (unsigned long long)phase.bound,
                       (unsigned long long)phase.bits, (unsigned long long)state.k,
                       (unsigned long long)phase.bits, lane + 1);
    } else if (phase.owl) {
        state = {phase.owl->k, std::move(phase.owl->residue), phase.owl->errors};
        previous_seconds = phase.owl->seconds;
    } else {
        state.residue.assign((task.p + 31) / 32, 0);
        state.residue.front() = 3;
    }
    ecm_ts_fprintf(stdout, "P-1 Stage1 M%u B1=%llu B2=%llu CUDA device=%d k=%llu/%llu lane=%zu\n",
                   task.p, (unsigned long long)phase.bound, (unsigned long long)b2,
                   options.execution.device_index, (unsigned long long)state.k,
                   (unsigned long long)phase.bits, lane + 1);
    const auto start_k = state.k;
    const bool extending = old_b1 != 0;
    const char* phase_name = !extending ? "stage1" : phase.bound < task.b1 ? "finish_old" : "extend";
    if (extending)
        ecm_ts_fprintf(stdout, "P-1 Stage1 phase M%u old_B1=%llu target_B1=%llu phase=%s remaining_iter=%llu lane=%zu\n",
                       task.p, (unsigned long long)old_b1, (unsigned long long)task.b1, phase_name,
                       (unsigned long long)(phase.bits - start_k), lane + 1);
    progress.start(lane, offset + start_k - skipped, offset + phase.bits - skipped + pending,
                   old_b1, task.b1, phase_name);
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    auto last_save = begin, block_start = begin;
    const auto ready = [&](const ecm_mersenne_ntt::Plan& plan) {
        if (lane == 0)
            ecm_ts_fprintf(stdout, "pm1_stage1_ntt: length=%u bpw=%u shape=%s graph=%u cached_powers=%u\n",
                           plan.length, plan.bpw, ecm_mersenne_ntt::tune_shape_descriptor(plan).c_str(),
                           plan.window, plan.pm1_cached_powers);
        block_start = Clock::now();
    };
    const auto verified = [&](const Pm1Progress& block) {
        const auto now = Clock::now();
        const bool stop = ecm_stage1_stop_requested();
        progress.update(lane, offset + block.k - skipped,
                        std::chrono::duration<double, std::milli>(now - block_start).count());
        if (stop || block.k == phase.bits ||
            (options.persistence.ckpt_ms &&
             now - last_save >= std::chrono::milliseconds(options.persistence.ckpt_ms))) {
            if (phase.extension) {
                auto& ext = *phase.extension;
                ext.k = block.k;
                ext.errors = block.errors;
                ext.residue = block.residue;
                write_extension(phase.root, task.p, phase.bound, ext);
            } else {
                const auto elapsed =
                    previous_seconds + std::chrono::duration<double>(now - begin).count();
                write_owl(phase.root, task.p, phase.bound, b2, phase.bits,
                          {block.k, block.errors, elapsed, block.residue},
                          options.persistence.ckpt_backups);
            }
            last_save = now;
            ecm_ts_fprintf(stdout, "P-1 verified k=%llu/%llu errors=%u M%u B1=%llu lane=%zu\n",
                           (unsigned long long)block.k, (unsigned long long)phase.bits,
                           block.errors, task.p, (unsigned long long)phase.bound, lane + 1);
        }
        block_start = Clock::now();
        return !stop;
    };
    if (state.k < phase.bits)
        state = ecm_mersenne_ntt::pm1_stage1(options.execution.device_index, task.p, phase.exponent,
                                             std::move(state), options.execution.ntt, verified,
                                             phase.extension ? &phase.extension->base : nullptr, ready);
    return state;
}
} // namespace

Outcome run(const Pm1Task& task, uint64_t b2, const ecm_stage1::Stage1Options& options,
            const std::filesystem::path& owl_root, const std::filesystem::path& prime95_dir,
            BatchProgress& progress, size_t lane) {
    if (options.execution.selection.target != ecm_stage1::ExecutionTarget::Gpu)
        throw std::runtime_error("Pminus1 requires the CUDA worker with target=gpu");
    Integer target;
    std::string detail;
    if (!ecm_compute_pm1_target(task, target.v, detail))
        throw std::runtime_error(detail);
    fs::create_directories(owl_root);
    const auto owner = "p1-" + std::to_string(task.p);
    ecm_queue::OwnerLease lease((owl_root / owner).u8string(), 0), prior_lease;
    const auto name = prime95_name(task.p);
    const auto destination = prime95_dir / name;
    const auto existing = load_prime95(destination, task.p);
    const auto local = ecm_io::same_file_resource(destination, owl_root / name)
                           ? existing : load_prime95(owl_root / name, task.p);
    if (existing && existing->b1 > task.b1)
        throw std::runtime_error("requested B1 is below the existing Prime95 endpoint: " +
                                 destination.u8string());

    Phase current{owl_root, task.b1};
    std::optional<Phase> prior;
    bool prior_owned = false;
    std::optional<Prime95Save> source;
    auto display = load_extension_progress(owl_root, task.p, task.b1);
    if (local && local->done && local->b1 == task.b1) {
        source = local;
    } else if (!load_phase(current, task.p)) {
        const auto roots = previous_roots(owl_root, task.p, task.b1);
        // Reuse the old phase chosen when this expansion began, even if its
        // endpoint was already saved before the new exponent could start.
        if (display) {
            if (const auto found = roots.find(display->from); found != roots.end()) {
                prior_lease.acquire((found->second / owner).u8string(), 0);
                Phase candidate{found->second, found->first};
                if (load_phase(candidate, task.p)) {
                    prior = std::move(candidate);
                    prior_owned = true;
                }
                else prior_lease.release();
            }
            if (!prior) display.reset();
        }
        // An unfinished local exponent must run to its own boundary first.
        if (!display) for (const auto& [bound, path] : roots) {
            Phase candidate{path, bound};
            if (!load_phase(candidate, task.p))
                continue;
            const bool unfinished = candidate.extension
                                        ? candidate.extension->k < candidate.extension->bits
                                        : candidate.owl->k < candidate.owl->bits;
            if (unfinished) {
                prior_lease.acquire((path / owner).u8string(), 0);
                candidate = {path, bound};
                if (!load_phase(candidate, task.p))
                    throw std::runtime_error("P-1 checkpoint disappeared");
                prior = std::move(candidate);
                prior_owned = true;
                break;
            }
            if (!prior)
                prior = std::move(candidate);
        }
        const bool unfinished =
            prior && (prior->extension ? prior->extension->k < prior->extension->bits
                                       : prior->owl->k < prior->owl->bits);
        if (!display && !unfinished) {
            source = existing;
            auto consider = [&](const fs::path& path) {
                if (path == destination)
                    return;
                auto candidate = load_prime95(path, task.p);
                if (candidate && candidate->b1 <= task.b1 &&
                    (!source || candidate->b1 > source->b1))
                    source = std::move(candidate);
            };
            consider(owl_root / name);
            consider(owl_root.parent_path().parent_path() /
                     name); // Imported m file next to ecm.toml.
            for (const auto& [bound, path] : roots) {
                if (source && bound <= source->b1)
                    break;
                consider(path / name);
            }
            if (source && (!prior || source->b1 >= prior->bound))
                prior.reset();
            else
                source.reset();
        }
        if (prior || (source && source->b1 < task.b1)) {
            current.extension.emplace();
            current.extension->from = prior ? prior->bound : source->b1;
            if (source)
                current.extension->base = std::move(source->residue);
        }
    }
    Pm1Progress state;
    if (source && source->b1 == task.b1 && !prior) {
        state.residue = std::move(source->residue);
        ecm_ts_fprintf(stdout, "P-1 Stage1 M%u B1=%llu using completed Prime95 endpoint\n", task.p,
                       (unsigned long long)task.b1);
        progress.start(lane, 0, 0, 0, task.b1, "completed");
    } else {
        prepare(current, task, options);
        const uint64_t old_b1 = current.extension ? current.extension->from : 0;
        uint64_t offset = display && display->from == old_b1 ? display->old_iterations : 0;
        if (prior) {
            if (!prior_owned) {
                prior_lease.acquire((prior->root / owner).u8string(), 0);
                if (!load_phase(*prior, task.p)) throw std::runtime_error("P-1 checkpoint disappeared");
            }
            prepare(*prior, task, options);
            const auto saved_b2 = prior->owl ? prior->owl->b2 : b2;
            const auto remaining = prior->bits - prior->cursor();
            if (offset >= prior->bits) offset = 0;
            if (remaining > offset) {
                offset = remaining;
                write_extension_progress(owl_root, task.p, task.b1, {old_b1, offset});
            }
            auto old = remaining ? advance(*prior, task, saved_b2, options, progress, lane, 0,
                                           current.bits - 1, old_b1, prior->bits - offset)
                                 : Pm1Progress{prior->bits, prior->extension ? std::move(prior->extension->residue)
                                                                            : std::move(prior->owl->residue), 0};
            if (old.k != prior->bits || ecm_stage1_stop_requested())
                return {};
            const auto old_archive = load_prime95(prior->root / name, task.p);
            write_prime95(prior->root, task.p, prior->bound, old.residue, nullptr,
                          old_archive ? &*old_archive : nullptr);
            current.extension->base = std::move(old.residue);
            current.extension->bits = current.bits;
            current.extension->residue = current.extension->base;
            write_extension(owl_root, task.p, task.b1, *current.extension);
            remove_pm1_checkpoints(prior->root, task.p);
            prior.reset();
        }
        state = advance(current, task, b2, options, progress, lane, offset, 0, old_b1);
        if (state.k != current.bits)
            return {};
    }
    Outcome result;
    auto path = write_prime95(prime95_dir, task.p, task.b1, state.residue, &result.save_hash,
                              existing ? &*existing : nullptr, owl_root);
    Integer g;
    mpz_import(g.v, state.residue.size(), -1, sizeof(uint32_t), 0, 0, state.residue.data());
    mpz_sub_ui(g.v, g.v, 1);
    mpz_gcd(g.v, g.v, target.v);
    result.completed = true;
    result.whole_target = mpz_cmp(g.v, target.v) == 0;
    if (!result.whole_target && mpz_cmp_ui(g.v, 1) > 0) {
        std::vector<char> digits(mpz_sizeinbase(g.v, 10) + 2);
        result.factor = mpz_get_str(digits.data(), 10, g.v);
    }
    remove_pm1_checkpoints(owl_root, task.p);
    ecm_ts_fprintf(stdout, "P-1 Stage1 complete M%u B1=%llu lane=%zu result=%s save=%s\n", task.p,
                   (unsigned long long)task.b1,lane+1,
                   result.whole_target ? "gcd=target" :
                   result.factor.empty() ? "no factor" : result.factor.c_str(), path.u8string().c_str());
    return result;
}
}
