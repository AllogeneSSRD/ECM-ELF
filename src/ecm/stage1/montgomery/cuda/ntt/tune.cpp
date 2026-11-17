#include "tune.h"
#include "ladder.h"
#include "ecm/stage1/ntt_length.h"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../edwards/memory.h"
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
    unsigned sm_count;
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
    return {index, uuid, memory_budget(free, memory_mib), unsigned(prop.multiProcessorCount)};
}

pt::Profile load(const std::filesystem::path& path) {
    if (path.empty()) return {};
    if (std::filesystem::exists(path)) return pt::Profile::load(path);
    const auto legacy = path.parent_path() / "stage2_ecm_tune.toml";
    if (path.filename() == "tune.toml" && std::filesystem::is_regular_file(legacy))
        return pt::Profile::load(legacy);
    return {};
}

unsigned tile_log(unsigned length) {
    return ecm_stage1::stage1_ntt_length_log(ecm_stage1::stage1_ntt_radix2_length(length));
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
    const auto shape = tune_shape_descriptor(p);
    ecm_ts_fprintf(stdout, "%s%u %u %s %uc %.0f iter/s\n", prefix,
        p.length, max_exponent(p), shape.c_str(), m.batch, m.throughput()*iterations);
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
        {"length", std::to_string(plan.length)},
        {"log2_length", std::to_string(ecm_stage1::stage1_ntt_length_log(plan.length))},
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

void persist(const std::string& path, const Device& device, std::vector<Fields> rows, bool fresh = false) {
    if (path.empty() || rows.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    ecm_io::FileLock lock(path);
    auto db = load(path);
    const auto condition = std::to_string(db.condition({{"uuid_hex", pt::quoted(device.uuid)}}));
    for (auto& row : rows) {
        row["condition"] = condition;
        db.update_stage1_ntt(std::move(row), fresh);
    }
    ecm_io::write_atomic(path, db.serialize());
}

struct Cancelled : std::runtime_error { Cancelled() : std::runtime_error("Stage1 NTT tune stopped") {} };

class Calibration {
    const Device& device_;
    unsigned p_, bits_;
    std::vector<uint32_t> scalar_, initial_;
    unsigned reference_batch_ = 0;
    std::vector<uint32_t> reference_;
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
        // A larger batch appends lanes to each slot. Retain one reference and
        // check common lanes even when the plan or batch size changes.
        const size_t reference_stride = size_t(words) * reference_batch_;
        const size_t common = size_t(words) * std::min(batch, reference_batch_);
        if (common) {
            for (unsigned slot = 0; slot < 4; ++slot) {
                const auto* first = run.points.data() + slot * stride;
                if (!std::equal(first, first + common, reference_.data() + slot * reference_stride))
                    throw std::runtime_error("NTT tune endpoint mismatch");
            }
        }
        if (batch > reference_batch_) {
            reference_batch_ = batch;
            reference_ = std::move(run.points);
        }
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
            // A single 16384-point weighted ladder benefits from splitting
            // the four outer layers into two smaller kernels. Keep the
            // batch search and explicit radix choices on their own paths.
            if (batch == 1 && best.plan.weighted && best.plan.length == 16384 &&
                best.plan.tile == 12 && !options.outer_max)
                alternate.outer_max = 2;
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

    std::vector<Measurement> search(bool exhaustive, unsigned sms) {
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
            // The ladder launches groups of two and four transforms per curve.
            // Sample full waves that a power-of-two batch sweep can miss.
            for (unsigned batch : {(sms + 3) / 4, (sms + 1) / 2, sms, (3 * sms + 1) / 2, 2 * sms})
                if (batch && batch <= max_batch) sample(batch);
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

struct Candidate {
    Stage1NttOptions options;
    Packing shape;
};

enum : unsigned { Linear, Weighted, Mixed, CandidateCount };
using Candidates = std::array<std::optional<Candidate>, CandidateCount>;

std::optional<Candidate> candidate(unsigned p, Stage1NttOptions options, bool prefer_radix5) {
    try {
        const auto shape = resolve_packing(p, options, prefer_radix5);
        if (options.tile <= tile_log(shape.length)) {
            options.length = shape.length;
            options.automatic = false;
            return Candidate{std::move(options), shape};
        }
    } catch (const std::invalid_argument&) {}
    return {};
}

Candidates candidates_for(unsigned p, Stage1NttOptions options) {
    Candidates out;
    options.weighted = false;
    out[Linear] = candidate(p, options, false);
    options.weighted = true;
    out[Weighted] = candidate(p, options, false);
    if (!options.length) {
        auto mixed = candidate(p, options, true);
        if (mixed && ecm_stage1::stage1_ntt_supported_radix5(mixed->shape.length))
            out[Mixed] = std::move(mixed);
    }
    return out;
}

std::optional<Cached> decode(const Fields& row, const Packing& shape, const Device& device,
                             unsigned batch, const Stage1NttOptions& base) {
    using ecm_stage2::tune::uint;
    if (uint(row,"batch") != batch || uint(row,"length") != shape.length || uint(row,"bpw") != shape.bpw ||
        uint(row,"peak_bytes") > device.budget) return {};
    const unsigned tile = unsigned(uint(row,"tile")), outer = unsigned(uint(row,"max_radix_bits"));
    const auto window = int(uint(row,"graph_bits")), warp = int(uint(row,"warp_tail"));
    const unsigned fixed_tile = base.tile ? base.tile : base.profile
        ? std::min(tile_log(shape.length), base.profile == 2 ? 10u : 12u) : 0;
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

unsigned naf_window(const ecm_stage1::Stage1Request& request) {
    return request.options.execution.selection.curve == ecm_stage1::CurveModel::Edwards
        ? unsigned(request.options.execution.naf_w) : 0;
}

const char* curve_name(const ecm_stage1::Stage1Request& request) {
    return request.options.execution.selection.curve == ecm_stage1::CurveModel::Edwards ? "edwards" : "mont0";
}

unsigned lcm_multiplier(const ecm_stage1::Stage1Request& request) {
    const auto& selection = request.options.execution.selection;
    if (selection.curve == ecm_stage1::CurveModel::Edwards) return 48;
    return selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12 : 1;
}

Candidates production_shapes(unsigned p, Stage1NttOptions options, bool native_edwards) {
    if (options.automatic && !(native_edwards && options.tight_weighted_bound && !options.length))
        return candidates_for(p, options);
    if (options.automatic) {
        options.weighted = true;
        try {
            const auto shape = resolve_packing(p, options, true, WeightedBound::UnsignedProducts);
            if (shape.needs_unsigned_bound) {
                options.length = shape.length;
                Candidates shapes;
                shapes[Mixed] = Candidate{std::move(options), shape};
                return shapes;
            }
        } catch (const std::invalid_argument&) {
        }
        return candidates_for(p, options);
    }
    const auto shape = resolve_packing(p, options, true,
        native_edwards ? WeightedBound::UnsignedProducts : WeightedBound::Conservative);
    Candidates shapes;
    shapes[options.weighted ? Weighted : Linear] = Candidate{std::move(options), shape};
    return shapes;
}

unsigned production_capacity(const ecm_stage1::Stage1Request& request, const Device& device,
                             unsigned p, const Candidates& shapes) {
    const unsigned window = naf_window(request);
    if (window && (window < 2 || window > 12))
        throw std::invalid_argument("Edwards NTT NAF requires a window from 2 to 12");
    if (!std::isfinite(request.b1) || request.b1 < 0 || request.b1 > 5000000000.0)
        throw std::invalid_argument("invalid Stage1 NTT B1");
    const size_t words = (p + 31) / 32, entries = window ? 1u << (window - 2) : 0;
    const auto dictionary_budget = size_t(request.options.execution.naf_memory_mb) * ecm_edwards_memory::mb;
    unsigned cap = max_batch;
    bool any = false;
    for (const auto& candidate : shapes) {
        if (!candidate) continue;
        any = true;
        const auto length = size_t(candidate->shape.length);
        // Bound the planner's spectra and GPU dictionary workspace before a
        // batch is created. Runtime allocation remains authoritative.
        const auto dictionary = entries * (32 * length + 16 * words);
        const auto per_curve = 88 * length + 128 * words + dictionary;
        const auto fixed = 160 * length + 32 * words + size_t(request.b1 / 4 + 16);
        cap = unsigned(std::min<size_t>(cap, device.budget > fixed ? (device.budget - fixed) / per_curve : 0));
        if (dictionary_budget && dictionary)
            cap = unsigned(std::min<size_t>(cap, dictionary_budget / dictionary));
    }
    if (!any) throw std::invalid_argument("no legal Stage1 NTT packing for the requested parameters");
    if (entries)
        if (const auto free = ecm_edwards_memory::host_free_bytes()) {
            const auto limit = ecm_edwards_memory::usable_bytes(*free, 64 * ecm_edwards_memory::mib);
            // A padded host table temporarily has both its old and expanded
            // lane layouts. Leave room for scalar and dictionary preparation.
            const size_t fixed = (size_t(request.b1) * 5 + 128 * entries * words);
            const auto per_curve = (32 * entries + 64) * words;
            cap = unsigned(std::min<size_t>(cap, limit > fixed ? (limit - fixed) / per_curve : 0));
        }
    // The exact allocator may use less than the conservative workspace bound.
    // Let it diagnose the single-curve limit with its full measurements.
    return std::max(1u, cap);
}

struct BatchRate {
    double rate = 0;
    unsigned repeats = 0;
};

unsigned recommended_batch(const std::map<unsigned, BatchRate>& rows, unsigned min_repeats = 1) {
    double peak = 0;
    for (const auto& item : rows)
        if (item.second.repeats >= min_repeats)
            peak = std::max(peak, item.second.rate);
    if (peak)
        for (const auto& item : rows)
            if (item.second.repeats >= min_repeats && item.second.rate >= near_peak * peak)
                return item.first;
    return 0;
}

} // namespace

std::string tune_shape_descriptor(const Plan& plan) {
    auto forward = plan.forward, inverse = plan.inverse;
    std::replace(forward.begin(), forward.end(), '+', ',');
    std::replace(inverse.begin(), inverse.end(), '+', ',');
    return std::to_string(unsigned(plan.weighted)) + ':' + std::to_string(plan.tile) + ':' +
           std::to_string(plan.outer_max) + ':' + std::to_string(plan.window) + ':' +
           std::to_string(unsigned(plan.warp_tail)) + ':' + forward + ':' + inverse;
}

int tune_command(int index, const Stage1NttOptions& options, unsigned p, bool exhaustive) {
    try {
        const auto device = device_info(index, options.memory_mib);
        std::vector<unsigned> widths{61,127,251,521,1279,2539,4003,8191,16001,32749,
                                     58631,65521,115201,131071,225287,262139};
        if (p) widths = {p};
        constexpr unsigned iterations = 192;
        const auto start = now_s();
        // Resolve every packing first: the same input width can produce
        // different lengths, and a length can be shared by different inputs.
        struct Input {
            Candidates candidates;
            unsigned pending = 0;
            bool measured = false;
            std::optional<Calibration> calibration;
        };
        std::map<unsigned, std::array<std::vector<unsigned>, CandidateCount>> schedule;
        std::map<unsigned, Input> inputs;
        for (const unsigned width : widths) {
            if (width < 32 || width > 262144)
                throw std::invalid_argument("--tune ntt --bits requires 32..262144");
            auto& input = inputs[width];
            input.candidates = candidates_for(width, options);
            for (unsigned mode = 0; mode < CandidateCount; ++mode) {
                if (const auto& plan = input.candidates[mode]) {
                    schedule[plan->shape.length][mode].push_back(width);
                    ++input.pending;
                }
            }
        }

        ecm_ts_fprintf(stdout, "ntt_type = convolution:tile_bits:max_radix_bits:graph_bits:"
                               "warp_tail:forward_pass_bits:inverse_pass_bits (0=linear, 1=weighted)\n");
        ecm_ts_fprintf(stdout, "length max_exp ntt_type batch iter/s (warmup=%u)\n", warmup_iterations);
        for (const auto& group : schedule) {
            std::vector<Fields> rows;
            for (unsigned mode = 0; mode < CandidateCount; ++mode) {
                for (const unsigned width : group.second[mode]) {
                    auto& input = inputs.at(width);
                    if (!input.calibration) input.calibration.emplace(device, width, iterations);
                    const auto result = BatchSearch(*input.calibration, input.candidates[mode]->options)
                                            .search(exhaustive, device.sm_count);
                    // Keep the reference endpoints until all shapes have run for this input.
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
                                   const Stage1NttOptions& options, bool native_edwards) {
    if (!options.automatic) return options;
    if (native_edwards && options.tight_weighted_bound && !options.length) {
        auto weighted = options;
        weighted.automatic = false;
        weighted.weighted = true;
        try {
            const auto shape = resolve_packing(p, weighted, true, WeightedBound::UnsignedProducts);
            if (shape.needs_unsigned_bound) {
                // X-only calibration cannot measure this native-only shape.
                weighted.length = shape.length;
                ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=edwards_unsigned_bound "
                               "p=%u batch=%u convolution=weighted length=%u\n", p, batch, shape.length);
                return weighted;
            }
        } catch (const std::invalid_argument&) {
            // The requested constraints may still admit a linear plan.
        }
    }
    const auto device = device_info(index, options.memory_mib);
    const auto candidates = candidates_for(p, options);
    std::array<std::optional<Cached>, CandidateCount> choices;
    try {
        const auto db = load(options.tune_file);
        for (const auto& row : db.stage1_ntt) {
            if (pt::is_runtime_stage1_ntt(row)) continue;
            const auto& condition = db.conditions.at(ecm_stage2::tune::uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;
            const bool weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            for (unsigned mode = 0; mode < CandidateCount; ++mode) {
                const auto& plan = candidates[mode];
                if (!plan || plan->options.weighted != weighted) continue;
                auto sample = decode(row, plan->shape, device, batch, plan->options);
                if (sample && (!choices[mode] || sample->rate > choices[mode]->rate))
                    choices[mode] = std::move(sample);
            }
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT tune read: %s\n", e.what());
    }
    const char* source = "tune";
    bool measure = std::none_of(choices.begin(), choices.end(), [](const auto& choice) { return choice.has_value(); });
    if (candidates[Mixed]) {
        // Existing tune files can have valid binary rows but no mixed measurement.
        for (unsigned mode = 0; mode < CandidateCount; ++mode)
            measure |= candidates[mode].has_value() && !choices[mode];
    }
    if (measure) {
        source = "short_tune";
        const auto start = now_s();
        constexpr unsigned iterations = 128;
        Calibration cal(device, p, iterations);
        std::vector<Fields> rows;
        for (unsigned mode = 0; mode < CandidateCount; ++mode) {
            if (!candidates[mode] || choices[mode]) continue;
            const auto& plan = candidates[mode]->options;
            auto row = measure_batch(cal, batch, plan);
            if (!row) continue;
            rows.push_back(encode(*row, iterations));
            choices[mode] = Cached{fixed_plan(plan, row->plan), row->throughput() * iterations};
        }
        try { persist(options.tune_file, device, std::move(rows)); }
        catch (const std::exception& e) { ecm_ts_fprintf(stderr, "Stage1 NTT tune write: %s\n", e.what()); }
        ecm_ts_fprintf(stdout, "stage1_ntt_short_tune: batch=%u warmup_iters=%u seconds=%.3f\n",
                       batch, warmup_iterations, now_s() - start);
    }
    const Cached* winner = nullptr;
    for (const auto& choice : choices)
        if (choice && (!winner || choice->rate > winner->rate)) winner = &*choice;
    if (!winner) throw BudgetError("no safe Stage1 NTT auto plan under the current VRAM budget");
    ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=%s p=%u batch=%u convolution=%s length=%u\n",
                   source, p, batch, modes[winner->options.weighted], winner->options.length);
    return winner->options;
}

unsigned select_production_batch(const ecm_stage1::Stage1Request& request, unsigned remaining) {
    if (!remaining) return 0;
    const auto& exec = request.options.execution;
    const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
    const auto p = unsigned(mpz_sizeinbase(request.modulus, 2)), window = naf_window(request);
    const auto shapes = production_shapes(p, exec.ntt, window != 0);
    const auto cap = std::min(remaining, production_capacity(request, device, p, shapes));
    std::map<unsigned, BatchRate> production, synthetic;
    using ecm_stage2::tune::uint;
    using ecm_stage2::tune::real;
    try {
        const auto db = load(exec.ntt.tune_file);
        for (const auto& row : db.stage1_ntt) {
            const auto& condition = db.conditions.at(uint(row,"condition"));
            const auto batch = unsigned(uint(row,"batch"));
            if (batch > cap || !condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid)
                continue;
            const bool weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            const bool compatible = std::any_of(shapes.begin(), shapes.end(), [&](const auto& shape) {
                return shape && shape->options.weighted == weighted &&
                    decode(row, shape->shape, device, batch, shape->options).has_value();
            });
            if (!compatible) continue;
            const bool runtime = pt::is_runtime_stage1_ntt(row);
            if (runtime && (uint(row,"valid_curves") != batch || uint(row,"arithmetic_bits") != p ||
                            uint(row,"b1") != uint64_t(std::floor(request.b1 + 0.0001)) ||
                            uint(row,"lcm_multiplier") != lcm_multiplier(request) ||
                            uint(row,"naf_window") != window ||
                            pt::text(row,"curve") != curve_name(request)))
                continue;
            auto& value = (runtime ? production : synthetic)[batch];
            const auto rate = real(row,runtime ? "curves_per_s" : "iter_per_s");
            const auto repeats = unsigned(uint(row,"repeats"));
            if ((!value.repeats || rate > value.rate) &&
                !(runtime && value.repeats >= 2 && repeats < 2))
                value = {rate, repeats};
            else if (runtime && value.repeats < 2 && repeats >= 2)
                value = {rate, repeats};
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT batch tune read: %s\n", e.what());
    }
    const unsigned tuned = recommended_batch(synthetic);
    const unsigned seed = tuned ? tuned : std::min(cap, 64u);
    unsigned selected = recommended_batch(production, 2);
    if (!selected) selected = recommended_batch(production);
    const char* source = selected ? "production" : tuned ? "tune" : "default";
    if (!selected) selected = seed;
    // Search a bounded set with real, disjoint curve batches. Leave the final
    // two batches available for the incumbent and the requested tail policy.
    if (!exec.ntt.tune_file.empty() && remaining >= 2 * cap + selected) {
        const std::array<unsigned, 4> candidates{seed, std::max(1u, seed / 2),
                                               std::min(cap, seed + seed / 2), cap};
        for (unsigned repetitions : {0u, 1u}) {
            for (const auto candidate : candidates) {
                const auto it = production.find(candidate);
                if ((it == production.end() ? 0 : it->second.repeats) == repetitions) {
                    selected = candidate;
                    source = "production_trial";
                    break;
                }
            }
            if (std::string_view(source) == "production_trial") break;
        }
    }
    ecm_ts_fprintf(stdout, "stage1_ntt_batch_selection: source=%s p=%u remaining=%u batch=%u capacity=%u\n",
                   source, p, remaining, selected, cap);
    return selected;
}

void record_production_sample(const ecm_stage1::Stage1Request& request, const Plan& plan, const Run& run,
                              unsigned valid, unsigned batch, double compute_seconds) {
    const auto& exec = request.options.execution;
    if (!exec.ntt.observe_batch || exec.ntt.tune_file.empty() || !valid || !run.completed_bits || run.device_ms.empty() ||
        !(run.device_ms.front() > 0) || !(compute_seconds > 0) || request.b1 < 2)
        return;
    try {
        // Device identity is queried by the normal selector as well. Read the
        // recorded budget from the observed run rather than its teardown state.
        const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
        const double rate = double(valid) * run.completed_bits * 1000 / run.device_ms.front();
        const double compute_rate = valid / compute_seconds;
        auto row = encode(Measurement{batch, plan, run.device_ms, run.owned_bytes, run.observed_bytes,
                                      run.budget_bytes}, run.completed_bits);
        row["benchmark_kind"] = pt::quoted(pt::stage1_runtime_ntt_kind);
        row["curve"] = pt::quoted(curve_name(request));
        row["arithmetic_bits"] = std::to_string(mpz_sizeinbase(request.modulus, 2));
        row["b1"] = std::to_string(uint64_t(std::floor(request.b1 + 0.0001)));
        row["lcm_multiplier"] = std::to_string(lcm_multiplier(request));
        row["naf_window"] = std::to_string(naf_window(request));
        row["valid_curves"] = std::to_string(valid);
        row["iter_per_s"] = pt::numeric(rate);
        row["curves_per_s"] = pt::numeric(compute_rate);
        row["dataflow"] = std::to_string(plan.dataflow);
        persist(exec.ntt.tune_file, device, {std::move(row)}, true);
        ecm_ts_fprintf(stdout, "stage1_ntt_production_sample: batch=%u valid_curves=%u iter_per_s=%.3f "
            "curves_per_s=%.6f compute_seconds=%.6f\n", batch, valid, rate, compute_rate, compute_seconds);
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT production tune write: %s\n", e.what());
    }
}

} // namespace ecm_mersenne_ntt
