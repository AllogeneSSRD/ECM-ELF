#include "pm1_workflow.h"
#include "cuda/ntt/pm1.h"
#include "ecm/integer.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/stop.h"
#include "io/exponent_cache.h"
#include "io/files.h"
#include "io/pm1_saves.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>

namespace ecm_pm1 {
using ecm_edwards::Integer;
using ecm_mersenne_ntt::Pm1Progress;

std::vector<Pm1Task> batch_prefix(const std::string& path, size_t source_line, Pm1Task first,
                                  const ecm_stage1::Stage1NttOptions& options, bool shared_prime95_dir) {
    std::vector<Pm1Task> tasks{std::move(first)};
    if (options.max_batch<=1 || options.tune_file.empty()) return tasks;
    std::ifstream in(path);
    std::string raw;
    for (size_t n=0;n<source_line;++n)
        if (!std::getline(in,raw)) return tasks;
    if (ecm_queue::task_text(raw)!=tasks.front().raw_line) return tasks;
    std::set<std::pair<unsigned,uint64_t>> destinations{{tasks.front().p,shared_prime95_dir?0:tasks.front().b1}};
    while (tasks.size()<options.max_batch && std::getline(in,raw)) {
        const auto kind=ecm_queue::classify_line(raw,nullptr,nullptr);
        if (kind==ecm_queue::LineKind::Comment) continue;
        if (kind!=ecm_queue::LineKind::Task) break;
        const auto text=ecm_queue::task_text(raw);
        Pm1Task next;
        std::string error;
        if (!ecm_is_pm1_line(text) || !ecm_parse_pm1_line(text,next,error)) break;
        if (!destinations.emplace(next.p,shared_prime95_dir?0:next.b1).second) break;
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
    char remaining[64]="--";
    if (known) std::snprintf(remaining,sizeof(remaining),"%.1fs",eta);
    ecm_ts_fprintf(stdout,"pm1_stage1_progress: iter=%llu/%llu %.2f%% iter_per_s=%.3f elapsed=%.1fs ETA=%s\n",
                   (unsigned long long)done,(unsigned long long)total,100.*done/total,
                   elapsed>0 ? advanced/elapsed : 0.,elapsed,remaining);
    last_=now;
}

Outcome run(const Pm1Task& task, uint64_t b2, const ecm_stage1::Stage1Options& options,
            const std::filesystem::path& owl_root, const std::filesystem::path& prime95_dir,
            BatchProgress& progress, size_t lane) {
    if (options.execution.selection.target != ecm_stage1::ExecutionTarget::Gpu)
        throw std::runtime_error("Pminus1 requires the CUDA worker with target=gpu");
    Integer target;
    std::string detail;
    if (!ecm_compute_pm1_target(task, target.v, detail))
        throw std::runtime_error(detail);

    std::filesystem::create_directories(owl_root);
    ecm_queue::OwnerLease lease((owl_root / ("p1-" + std::to_string(task.p))).u8string(), 0);
    uint64_t bits;
    std::vector<uint32_t> words;
    {
        Integer exponent;
        if (!ecm_build_lcm_exponent_cached(exponent.v, task.b1, 1,
                                           options.persistence.exponent_cache_dir, &detail))
            throw std::runtime_error("cannot build Pminus1 exponent: " + detail);
        mpz_mul_ui(exponent.v, exponent.v, 2UL * task.p);
        bits = mpz_sizeinbase(exponent.v, 2);
        words.resize((bits + 31) / 32);
        mpz_export(words.data(), nullptr, -1, sizeof(uint32_t), 0, 0, exponent.v);
    }
    Pm1Progress state{};
    double previous_seconds = 0;
    if (auto save = load_owl(owl_root, task.p, task.b1, bits)) {
        state.k = save->k;
        state.errors = save->errors;
        state.residue = std::move(save->residue);
        previous_seconds = save->seconds;
    } else {
        state.k = 1;
        state.residue.assign((task.p + 31) / 32, 0);
        state.residue.front() = 3;
    }
    ecm_ts_fprintf(stdout, "P-1 Stage1 M%u B1=%llu B2=%llu CUDA device=%d k=%llu/%llu lane=%zu\n",
                   task.p, (unsigned long long)task.b1,(unsigned long long)b2,options.execution.device_index,
                   (unsigned long long)state.k, (unsigned long long)bits,lane+1);
    progress.update(lane,state.k,bits,0);
    const auto begin = std::chrono::steady_clock::now();
    auto last_save = begin;
    auto block_start = begin;
    const auto verified = [&](const Pm1Progress& block) {
        const auto now = std::chrono::steady_clock::now();
        const bool stop = ecm_stage1_stop_requested();
        progress.update(lane,block.k,bits,std::chrono::duration<double,std::milli>(now-block_start).count());
        if (stop || block.k == bits ||
            (options.persistence.ckpt_ms && now - last_save >= std::chrono::milliseconds(options.persistence.ckpt_ms))) {
            const double elapsed = previous_seconds + std::chrono::duration<double>(now - begin).count();
            write_owl(owl_root, task.p, task.b1, b2, bits,
                      {block.k, block.errors, elapsed, block.residue}, options.persistence.ckpt_backups);
            last_save = now;
            ecm_ts_fprintf(stdout, "P-1 verified k=%llu/%llu errors=%u M%u B1=%llu lane=%zu\n",
                           (unsigned long long)block.k, (unsigned long long)bits, block.errors,
                           task.p,(unsigned long long)task.b1,lane+1);
        }
        block_start=std::chrono::steady_clock::now();
        return !stop;
    };
    if (state.k < bits)
        state = ecm_mersenne_ntt::pm1_stage1(options.execution.device_index, task.p, words,
                                              std::move(state), options.execution.ntt, verified);
    if (state.k != bits)
        return {};
    Outcome result;
    auto path = write_prime95(prime95_dir, task.p, task.b1, state.residue, &result.save_hash);
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
