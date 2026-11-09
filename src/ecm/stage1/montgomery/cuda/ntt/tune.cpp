#include "tune.h"
#include "ladder.h"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../../../io/files.h"
#include "../../../../stage2/planning/tune_portable.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/stop.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <random>

namespace ecm_mersenne_ntt {
namespace {
using ecm_stage1::Stage1NttOptions;
namespace pt = ecm_stage2::tune::portable;
using ecm_stage2::tune::Fields;
constexpr unsigned max_batch = ecm_stage1::stage1_ntt_max_curves;
constexpr unsigned warmup_iterations = 100;
constexpr double plateau = 1.05, near_peak = 0.95;
constexpr const char* modes[] = {"linear", "weighted"};

struct Device {
    int index;
    std::string uuid;
    size_t budget;
};

Device device_info(int index, unsigned memory_mib) {
    CK(cudaSetDevice(index));
    CK(cudaFree(nullptr));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, index));
    size_t free = 0, total = 0;
    CK(cudaMemGetInfo(&free, &total));
    std::string uuid;
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char byte : prop.uuid.bytes) {
        uuid += hex[byte >> 4];
        uuid += hex[byte & 15];
    }
    return {index, uuid, memory_budget(free, memory_mib)};
}

pt::Profile load(const std::filesystem::path& path) {
    if (path.empty()) return {};
    if (std::filesystem::exists(path)) return pt::Profile::load(path);
    const auto legacy = path.parent_path() / "stage2_ecm_tune.toml";
    if (path.filename() == "tune.toml" && std::filesystem::is_regular_file(legacy))
        return pt::Profile::load(legacy);
    return {};
}

unsigned log_length(unsigned length) {
    unsigned log = 0;
    for (; length > 1; length >>= 1) ++log;
    return log;
}

struct Measurement {
    unsigned batch = 0;
    Plan plan{};
    std::vector<double> ms;
    size_t owned = 0, observed = 0, budget = 0;
    double time() const {
        auto ordered = ms;
        std::sort(ordered.begin(), ordered.end());
        const auto middle = ordered.size() / 2;
        return ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
    }
    double throughput() const { return 1000.0 * batch / time(); }
};

unsigned max_exponent(const Plan& plan) {
    constexpr unsigned supported_max = 262144;
    if (plan.weighted) return std::min(supported_max, plan.length * plan.bpw);
    unsigned lo = 0, hi = (plan.length - 1) / 2;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (exact_ok_terms(mid, plan.bpw)) lo = mid;
        else hi = mid - 1;
    }
    return std::min(supported_max, lo * plan.bpw);
}

void print_measurement(const char* prefix, const Measurement& m, unsigned iterations) {
    const auto& p = m.plan;
    auto forward = p.forward, inverse = p.inverse;
    std::replace(forward.begin(), forward.end(), '+', ',');
    std::replace(inverse.begin(), inverse.end(), '+', ',');
    ecm_ts_fprintf(stdout, "%s%u %u %u:%u:%u:%u:%u:%s:%s %uc %.0f iter/s\n", prefix,
        p.length, max_exponent(p), unsigned(p.weighted), p.tile, p.outer_max, p.window,
        unsigned(p.warp_tail), forward.c_str(), inverse.c_str(), m.batch, m.throughput()*iterations);
}

Stage1NttOptions fixed_plan(Stage1NttOptions o, const Plan& plan) {
    o.automatic = false;
    o.weighted = plan.weighted;
    o.length = plan.length;
    o.bpw = plan.bpw;
    o.tile = plan.tile;
    o.outer_max = plan.outer_max;
    o.window = int(plan.window);
    o.warp_tail = int(plan.warp_tail);
    return o;
}

Fields encode(const Measurement& m, unsigned iterations) {
    const auto& plan = m.plan;
    Fields row{
        {"length", std::to_string(plan.length)}, {"log2_length", std::to_string(log_length(plan.length))},
        {"iter_per_s", pt::numeric(m.throughput() * iterations)}, {"batch", std::to_string(m.batch)},
        {"ntt_type", pt::quoted(plan.weighted ? pt::weighted_ntt_type : pt::linear_ntt_type)},
        {"bpw", std::to_string(plan.bpw)}, {"tile", std::to_string(plan.tile)},
        {"max_radix_bits", std::to_string(plan.outer_max)}, {"passes_forward", std::to_string(plan.passes)},
        {"forward", pt::quoted(plan.forward)}, {"inverse", pt::quoted(plan.inverse)},
        {"graph_bits", std::to_string(plan.window)}, {"warp_tail", std::to_string(plan.warp_tail)},
        {"repeats", std::to_string(m.ms.size())}, {"owned_bytes", std::to_string(m.owned)},
        {"peak_bytes", std::to_string(std::max(m.owned, m.observed))},
        {"budget_bytes", std::to_string(m.budget)}
    };
    pt::stamp_provenance(row);
    return row;
}

void persist(const std::string& path, const Device& device, std::vector<Fields> rows) {
    if (path.empty() || rows.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    ecm_io::FileLock lock(path);
    auto db = load(path);
    const auto condition = std::to_string(db.condition({{"uuid_hex", pt::quoted(device.uuid)}}));
    for (auto& row : rows) {
        row["condition"] = condition;
        db.update_stage1_ntt(std::move(row));
    }
    ecm_io::write_atomic(path, db.serialize());
}

struct Cancelled : std::runtime_error { Cancelled() : std::runtime_error("Stage1 NTT tune stopped") {} };

class Calibration {
    const Device& device_;
    unsigned p_, bits_;
    std::vector<uint32_t> scalar_, initial_;
    std::map<unsigned, std::vector<uint32_t>> reference_;
public:
    Calibration(const Device& device, unsigned p, unsigned bits)
        : device_(device), p_(p), bits_(bits), scalar_((bits + 31) / 32),
          initial_(size_t(6) * max_batch * ((p + 31) / 32)) {
        std::mt19937 random(0x67ae128bu);
        for (auto& word : scalar_) word = random();
        const unsigned words = (p + 31) / 32;
        for (size_t i = 0; i < initial_.size(); i += words) {
            for (unsigned j = 0; j < words; ++j) initial_[i + j] = random();
            if (p % 32) initial_[i + words - 1] &= (1u << (p % 32)) - 1;
            // Leave a guaranteed zero bit in the first limb so M is never encoded.
            initial_[i] &= 0xfffffffeu;
        }
    }

    Measurement measure(unsigned batch, const Stage1NttOptions& options, unsigned samples) {
        if (ecm_stage1_stop_requested()) throw Cancelled();
        const unsigned words = (p_ + 31) / 32;
        const size_t stride = size_t(words) * batch, full_stride = size_t(words) * max_batch;
        std::vector<uint32_t> active(6 * stride);
        for (unsigned slot = 0; slot < 6; ++slot)
            std::copy_n(initial_.data() + slot * full_stride, stride, active.data() + slot * stride);
        Measurement m;
        m.batch = batch;
        Control control;
        control.warmup_iterations = warmup_iterations;
        control.describe = [&](const Plan& plan) { m.plan = plan; };
        control.poll = [](unsigned) { return Control::Action{false, ecm_stage1_stop_requested()}; };
        auto run = ladder(device_.index, p_, batch, active, scalar_, bits_, options, control, samples);
        if (run.stopped) throw Cancelled();
        const auto found = reference_.find(batch);
        if (found == reference_.end()) reference_.emplace(batch, std::move(run.points));
        else if (found->second != run.points) throw std::runtime_error("NTT tune endpoint mismatch");
        m.ms = std::move(run.device_ms);
        m.owned = run.owned_bytes;
        m.observed = run.observed_bytes;
        m.budget = run.budget_bytes;
        return m;
    }
};

std::optional<Measurement> measure_batch(Calibration& cal, unsigned batch,
                                         const Stage1NttOptions& options) {
    try {
        auto best = cal.measure(batch, options, 3);
        if (!options.tile && !options.profile && best.plan.tile >= 5) {
            auto alternate = options;
            alternate.tile = best.plan.tile - 2;
            try {
                auto challenger = cal.measure(batch, alternate, 3);
                if (challenger.time() < best.time()) best = std::move(challenger);
            } catch (const BudgetError&) {
            } catch (const std::invalid_argument&) {
            }
        }
        return best;
    } catch (const BudgetError&) {
        return {};
    } catch (const std::invalid_argument&) {
        return {};
    }
}

class BatchSearch {
    Calibration& cal_;
    Stage1NttOptions options_;
    std::map<unsigned, std::optional<Measurement>> rows_;

    Measurement* sample(unsigned n) {
        if (auto it = rows_.find(n); it != rows_.end())
            return it->second ? &*it->second : nullptr;
        auto inserted = rows_.emplace(n, measure_batch(cal_, n, options_)).first;
        return inserted->second ? &*inserted->second : nullptr;
    }

    void confirm(Measurement& a, Measurement& b) {
        // Recheck both incumbents in alternating order before ending the doubling sweep.
        for (auto* row : {&b, &a, &a, &b}) {
            auto extra = cal_.measure(row->batch, fixed_plan(options_, row->plan), 1);
            row->ms.push_back(extra.ms.front());
            row->owned = std::max(row->owned, extra.owned);
            row->observed = std::max(row->observed, extra.observed);
            row->budget = std::max(row->budget, extra.budget);
        }
    }

    const Measurement& peak() const {
        const Measurement* best = nullptr;
        for (const auto& item : rows_)
            if (item.second && (!best || item.second->throughput() > best->throughput()))
                best = &*item.second;
        return *best;
    }

public:
    BatchSearch(Calibration& cal, const Stage1NttOptions& options) : cal_(cal), options_(options) {}

    std::vector<Measurement> search(bool exhaustive) {
        auto* prior = sample(1);
        if (!prior) return {};
        if (exhaustive) {
            for (unsigned n = 2; n <= max_batch; ++n) sample(n);
        } else {
            unsigned stop = max_batch;
            for (unsigned n = 2; n <= max_batch; n *= 2) {
                auto* next = sample(n);
                if (next && next->throughput() < plateau * prior->throughput()) {
                    try { confirm(*prior, *next); }
                    catch (const BudgetError&) { next = nullptr; rows_[n].reset(); }
                }
                if (!next || next->throughput() < plateau * prior->throughput()) {
                    stop = n;
                    break;
                }
                prior = next;
            }
            // Probe the midpoint of both intervals adjacent to the stopping point.
            auto mid = [&](unsigned lo, unsigned hi) { if (hi > lo + 1) sample((lo + hi) / 2); };
            mid(std::max(1u, stop / 4), stop / 2);
            mid(stop / 2, stop);
            const double local_peak = peak().throughput();
            if (stop < max_batch) {
                if (auto* limit = sample(max_batch); limit && limit->throughput() > plateau * local_peak)
                    mid(stop, max_batch);
            }
            for (unsigned i = 0; i < 2; ++i) {
                const auto best = peak().batch;
                if (best > 1) sample(best - 1);
                if (best < max_batch) sample(best + 1);
            }
            // Refine the rising shoulder around the 95% target. The final
            // recommendation is selected from all measured rows.
            unsigned lo = 1, hi = peak().batch;
            while (hi > lo + 1) {
                const unsigned m = (lo + hi) / 2;
                auto* row = sample(m);
                if (row && row->throughput() >= near_peak * peak().throughput()) hi = m;
                else lo = m;
            }
        }
        std::vector<Measurement> result;
        for (auto& item : rows_) if (item.second) result.push_back(std::move(*item.second));
        return result;
    }
};

struct Cached {
    Stage1NttOptions options;
    double rate;
};

std::optional<Packing> eligible_packing(unsigned p, const Stage1NttOptions& options) {
    try {
        const auto packing = resolve_packing(p, options);
        if (options.tile <= log_length(packing.length)) return packing;
    } catch (const std::invalid_argument&) {}
    return {};
}

std::optional<Cached> decode(const Fields& row, const Packing& shape, const Device& device,
                             unsigned batch, const Stage1NttOptions& base) {
    using ecm_stage2::tune::uint;
    if (uint(row,"batch") != batch || uint(row,"length") != shape.length || uint(row,"bpw") != shape.bpw ||
        uint(row,"peak_bytes") > device.budget) return {};
    const unsigned tile = unsigned(uint(row,"tile")), outer = unsigned(uint(row,"max_radix_bits"));
    const auto window = int(uint(row,"graph_bits")), warp = int(uint(row,"warp_tail"));
    const unsigned fixed_tile = base.tile ? base.tile : base.profile
        ? std::min(log_length(shape.length), base.profile == 2 ? 10u : 12u) : 0;
    const unsigned fixed_outer = base.outer_max ? base.outer_max : base.profile ? 4 : 0;
    const int fixed_window = base.window >= 0 ? base.window : base.profile ? 16 : -1;
    const int fixed_warp = base.warp_tail >= 0 ? base.warp_tail : base.profile ? 1 : -1;
    if ((fixed_tile && tile != fixed_tile) || (fixed_outer && outer != fixed_outer) ||
        (fixed_window >= 0 && window != fixed_window) ||
        (fixed_warp >= 0 && warp != int(fixed_warp && tile >= 5))) return {};
    auto o = base;
    o.automatic = false;
    o.length = shape.length;
    o.bpw = shape.bpw;
    o.tile = tile;
    o.outer_max = outer;
    o.window = window;
    o.warp_tail = warp;
    return Cached{std::move(o), ecm_stage2::tune::real(row,"iter_per_s")};
}

} // namespace

int tune_command(int index, const Stage1NttOptions& options, unsigned p, bool exhaustive) {
    try {
        const auto device = device_info(index, options.memory_mib);
        std::vector<unsigned> widths{61,127,251,521,1279,2539,4003,8191,16001,32749,65521,131071,262139};
        if (p) widths = {p};
        constexpr unsigned iterations = 192;
        const auto start = now_s();
        std::array<Stage1NttOptions, 2> candidates{options, options};
        for (unsigned mode = 0; mode < candidates.size(); ++mode) {
            candidates[mode].automatic = false;
            candidates[mode].weighted = mode != 0;
        }
        // Resolve every packing first: the same input width can produce
        // different lengths, and a length can be shared by different inputs.
        struct Input {
            unsigned pending = 0;
            bool measured = false;
            std::optional<Calibration> calibration;
        };
        std::map<unsigned, std::array<std::vector<unsigned>, 2>> schedule;
        std::map<unsigned, Input> inputs;
        for (const unsigned width : widths) {
            if (width < 32 || width > 262144)
                throw std::invalid_argument("--tune ntt --bits requires 32..262144");
            auto& input = inputs[width];
            for (unsigned mode = 0; mode < candidates.size(); ++mode) {
                if (const auto packing = eligible_packing(width, candidates[mode])) {
                    schedule[packing->length][mode].push_back(width);
                    ++input.pending;
                }
            }
        }

        ecm_ts_fprintf(stdout, "ntt_type = convolution:tile_bits:max_radix_bits:graph_bits:"
                               "warp_tail:forward_pass_bits:inverse_pass_bits (0=linear, 1=weighted)\n");
        ecm_ts_fprintf(stdout, "length max_exp ntt_type batch iter/s (warmup=%u)\n", warmup_iterations);
        for (const auto& group : schedule) {
            std::vector<Fields> rows;
            for (unsigned mode = 0; mode < candidates.size(); ++mode) {
                for (const unsigned width : group.second[mode]) {
                    auto& input = inputs.at(width);
                    if (!input.calibration) input.calibration.emplace(device, width, iterations);
                    const auto result = BatchSearch(*input.calibration, candidates[mode]).search(exhaustive);
                    // Keep the reference endpoints until both modes have run for this input.
                    if (--input.pending == 0) input.calibration.reset();
                    if (result.empty()) continue;
                    input.measured = true;
                    const auto peak = std::max_element(result.begin(), result.end(), [](const auto& a, const auto& b) {
                        return a.throughput() < b.throughput();
                    });
                    const auto recommended = std::find_if(result.begin(), result.end(), [&](const auto& m) {
                        return m.throughput() >= near_peak * peak->throughput();
                    });
                    for (const auto& measurement : result) {
                        print_measurement("", measurement, iterations);
                        rows.push_back(encode(measurement, iterations));
                    }
                    print_measurement("Best ", *peak, iterations);
                    if (recommended != peak) print_measurement("95%  ", *recommended, iterations);
                }
            }
            if (rows.empty()) continue;
            persist(options.tune_file, device, std::move(rows));
        }
        if (std::any_of(inputs.begin(), inputs.end(), [](const auto& item) { return !item.second.measured; }))
            throw std::runtime_error("no safe NTT backend for at least one requested shape or VRAM budget");
        if (!options.tune_file.empty())
            ecm_ts_fprintf(stdout, "Saved %s (%.2f s)\n", options.tune_file.c_str(), now_s()-start);
        return 0;
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}

Stage1NttOptions select_tuned_plan(int index, unsigned p, unsigned batch,
                                   const Stage1NttOptions& options) {
    if (!options.automatic) return options;
    const auto device = device_info(index, options.memory_mib);
    std::array<Stage1NttOptions, 2> candidates{options, options};
    candidates[0].weighted = false;
    candidates[1].weighted = true;
    const std::array<std::optional<Packing>, 2> shapes{
        eligible_packing(p, candidates[0]), eligible_packing(p, candidates[1])};
    std::array<std::optional<Cached>, 2> choices;
    try {
        const auto db = load(options.tune_file);
        for (const auto& row : db.stage1_ntt) {
            const auto& condition = db.conditions.at(ecm_stage2::tune::uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;
            const unsigned mode = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            if (!shapes[mode]) continue;
            auto sample = decode(row, *shapes[mode], device, batch, candidates[mode]);
            if (sample && (!choices[mode] || sample->rate > choices[mode]->rate)) choices[mode] = std::move(sample);
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT tune read: %s\n", e.what());
    }
    const char* source = "tune";
    if (!choices[0] && !choices[1]) {
        source = "short_tune";
        const auto start = now_s();
        constexpr unsigned iterations = 128;
        Calibration cal(device, p, iterations);
        std::vector<Fields> rows;
        for (unsigned mode = 0; mode < 2; ++mode) {
            if (!shapes[mode]) continue;
            auto row = measure_batch(cal, batch, candidates[mode]);
            if (!row) continue;
            rows.push_back(encode(*row, iterations));
            choices[mode] = Cached{fixed_plan(candidates[mode], row->plan), row->throughput() * iterations};
        }
        try { persist(options.tune_file, device, std::move(rows)); }
        catch (const std::exception& e) { ecm_ts_fprintf(stderr, "Stage1 NTT tune write: %s\n", e.what()); }
        ecm_ts_fprintf(stdout, "stage1_ntt_short_tune: batch=%u warmup_iters=%u seconds=%.3f\n",
                       batch, warmup_iterations, now_s() - start);
    }
    if (!choices[0] && !choices[1]) throw BudgetError("no safe Stage1 NTT auto plan under the current VRAM budget");
    const unsigned winner = choices[1] && (!choices[0] || choices[1]->rate > choices[0]->rate);
    ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=%s p=%u batch=%u convolution=%s\n",
                   source, p, batch, modes[winner]);
    return choices[winner]->options;
}

} // namespace ecm_mersenne_ntt
