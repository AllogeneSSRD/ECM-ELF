#include "pm1_workflow.h"
#include "cuda/ntt/pm1.h"
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

void BatchProgress::update(size_t index, uint64_t k, uint64_t bits, double block_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now=Clock::now();
    auto& lane=lanes_.at(index);
    if (!lane.bits) {
        lane.initial=k;
        lane.bits=bits;
        if (begin_==Clock::time_point{}) begin_=now;
    } else {
        lane.speed.add(double(k-lane.cursor),block_ms);
    }
    lane.cursor=k;
    if (k<bits && now-last_<std::chrono::milliseconds(250)) return;
    uint64_t done=0, total=0, advanced=0;
    double eta=0;
    bool known=true;
    for (const auto& item:lanes_) {
        if (!item.bits) return;
        // k includes the leading exponent bit, already represented by the seed 3.
        done+=item.cursor-1;
        total+=item.bits-1;
        advanced+=item.cursor-item.initial;
        if (item.cursor<item.bits) {
            known=known && item.speed.speed()>0;
            eta=std::max(eta,item.speed.remaining(item.bits,item.cursor));
        }
    }
    const double elapsed=std::chrono::duration<double>(now-begin_).count();
    if (!total) return; // An interval without a new prime power requires no iterations.
    char remaining[64]="--";
    if (known) std::snprintf(remaining,sizeof(remaining),"%.1fs",eta);
    ecm_ts_fprintf(stdout,"pm1_stage1_progress: iter=%llu/%llu %.2f%% iter_per_s=%.3f elapsed=%.1fs ETA=%s\n",
                   (unsigned long long)done,(unsigned long long)total,100.*done/total,
                   elapsed>0 ? advanced/elapsed : 0.,elapsed,remaining);
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
                    uint64_t offset, uint64_t total_bits) {
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
    progress.update(lane, offset + state.k, total_bits, 0);
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    auto last_save = begin, block_start = begin;
    const auto verified = [&](const Pm1Progress& block) {
        const auto now = Clock::now();
        const bool stop = ecm_stage1_stop_requested();
        progress.update(lane, offset + block.k, total_bits,
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
                                             phase.extension ? &phase.extension->base : nullptr);
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
    if (existing && existing->b1 > task.b1)
        throw std::runtime_error("requested B1 is below the existing Prime95 endpoint: " +
                                 destination.u8string());

    Phase current{owl_root, task.b1};
    std::optional<Phase> prior;
    std::optional<Prime95Save> source;
    if (!load_phase(current, task.p)) {
        const auto roots = previous_roots(owl_root, task.p, task.b1);
        // An unfinished local exponent must run to its own boundary first.
        for (const auto& [bound, path] : roots) {
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
                break;
            }
            if (!prior)
                prior = std::move(candidate);
        }
        const bool unfinished =
            prior && (prior->extension ? prior->extension->k < prior->extension->bits
                                       : prior->owl->k < prior->owl->bits);
        if (!unfinished) {
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
        progress.update(lane, 1, 1, 0);
    } else {
        prepare(current, task, options);
        uint64_t offset = 0;
        if (prior) {
            prepare(*prior, task, options);
            const auto saved_b2 = prior->owl ? prior->owl->b2 : b2;
            auto old = advance(*prior, task, saved_b2, options, progress, lane, 0,
                               prior->bits + current.bits - 1);
            if (old.k != prior->bits || ecm_stage1_stop_requested())
                return {};
            offset = prior->bits - 1;
            current.extension->base = std::move(old.residue);
            prior.reset();
        }
        state = advance(current, task, b2, options, progress, lane, offset, offset + current.bits);
        if (state.k != current.bits)
            return {};
    }
    Outcome result;
    auto path = write_prime95(prime95_dir, task.p, task.b1, state.residue, &result.save_hash, existing ? &*existing : nullptr);
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
    ecm_ts_fprintf(stdout, "P-1 Stage1 complete M%u B1=%llu lane=%zu result=%s save=%s\n", task.p,
                   (unsigned long long)task.b1,lane+1,
                   result.whole_target ? "gcd=target" :
                   result.factor.empty() ? "no factor" : result.factor.c_str(), path.u8string().c_str());
    return result;
}
}
