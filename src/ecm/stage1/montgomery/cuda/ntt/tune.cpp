#include "tune.h"
#include "ecm/stage1/ntt_target.h"
#include "ladder.h"
#include "ecm/stage1/scalar.h"
#include "../../../edwards/ntt_dictionary.h"
#include <set>
#include "ecm/stage1/ntt_length.h"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../edwards/memory.h"
#include "../../../../../io/files.h"
#include "../../../../stage2/planning/tune_portable.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/stop.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>

namespace ecm_mersenne_ntt {
namespace {
using ecm_stage1::Stage1NttOptions;
namespace pt = ecm_stage2::tune::portable;
using ecm_stage2::tune::Fields;
constexpr double near_peak = 0.95;
constexpr const char* modes[] = {"linear", "weighted"};

struct Device {
    int index;
    std::string uuid;
    size_t budget;
    size_t device_budget;
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
    return {index, uuid, memory_budget(free, memory_mib), memory_budget(free, 0)};
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
    unsigned steps = 0, blocks = 1;
    unsigned naf_columns = 0, sample_nonzero = 0;
    double total_ms() const { return std::accumulate(ms.begin(), ms.end(), 0.0); }
    double time() const {
        auto ordered = ms;
        std::sort(ordered.begin(), ordered.end());
        const auto middle = ordered.size() / 2;
        return ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
    }
    double group_ms() const { return naf_columns ? total_ms() / ms.size() : time(); }
    double throughput() const { return 1000.0 * batch / group_ms(); }
};

unsigned linear_exponent_limit(unsigned length, unsigned bpw) {
    unsigned lo = 0, hi = (length - 1) / 2;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (exact_ok_terms(mid, bpw)) lo = mid;
        else hi = mid - 1;
    }
    return lo * bpw;
}

unsigned max_exponent(const Plan& plan) {
    const unsigned limit = plan.weighted ? plan.length * plan.bpw : linear_exponent_limit(plan.length, plan.bpw);
    return std::min(ecm_stage1::stage1_ntt_max_exponent, limit);
}

void print_measurement(const Measurement& m, unsigned target) {
    const auto& p = m.plan;
    const auto shape = tune_shape_descriptor(p);
    ecm_ts_fprintf(stdout, "%u %u %s %uc %.0f iter/s dataflow=%u target_ms=%u timed_steps=%u\n",
        p.length, max_exponent(p), shape.c_str(), m.batch, m.throughput()*m.steps, p.dataflow,target,m.steps);
}

Stage1NttOptions fixed_plan(Stage1NttOptions o, const Plan& plan) {
    if (ecm_stage1::stage1_ntt_radix5(plan.length)) {
        using Layout = Stage1NttOptions::MixedLayout;
        o.mixed_layout = plan.dataflow & CoalescedRadix5 ? Layout::Coalesced :
                         plan.dataflow & PfaRadix5 ? Layout::Pfa : Layout::Twiddle;
    }
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

void persist(const std::string& path, const Device& device, std::vector<Fields> rows, bool fresh = false,
             const std::string& replace_scope = {}, bool partial = false) {
    if (path.empty() || rows.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    ecm_io::FileLock lock(path);
    auto db = load(path);
    const auto condition = std::to_string(db.condition({{"uuid_hex", pt::quoted(device.uuid)}}));
    if (!replace_scope.empty()) {
        const auto belongs = [&](const Fields& r) {
            return r.at("condition") == condition && r.count("batch_scope") &&
                   pt::text(r,"batch_scope") == replace_scope;
        };
        // A shorter interrupted run must not evict a completed search.
        if (partial && std::any_of(db.stage1_ntt.begin(),db.stage1_ntt.end(),
                [&](const Fields& r) { return belongs(r) && r.count("batch_search"); })) return;
        db.stage1_ntt.erase(std::remove_if(db.stage1_ntt.begin(),db.stage1_ntt.end(),belongs),db.stage1_ntt.end());
    }
    for (auto& row : rows) {
        row["condition"] = condition;
        db.update_stage1_ntt(std::move(row), fresh);
    }
    ecm_io::write_atomic(path, db.serialize());
}

struct Cancelled : std::runtime_error { Cancelled() : std::runtime_error("Stage1 NTT tune stopped") {} };
struct TimeLimit {};

struct NafSample {
    unsigned columns = 0, nonzero = 0;
    std::vector<int16_t> digits;
};

// One scalar per tuning command, one immutable sample per requested window.
class TuneNafSamples {
    uint64_t b1_;
    ecm_edwards::Integer scalar_;
    std::map<unsigned, NafSample> windows_;
public:
    explicit TuneNafSamples(uint64_t b1) : b1_(b1) {}

    const NafSample& get(unsigned window) {
        const auto found = windows_.find(window);
        if (found != windows_.end()) return found->second;
        if (!mpz_sgn(scalar_.v) && !ecm_build_lcm_exponent(scalar_.v, b1_, 48))
            throw std::runtime_error("cannot build tune scalar");
        EcmEdwardsNaf encoding(scalar_.v, window);
        using namespace ecm_stage1;
        NafSample sample;
        sample.columns = encoding.bits;
        const unsigned bits = std::min(sample.columns, naf_tune_span * naf_tune_strata);
        const unsigned strata = (bits + naf_tune_span - 1) / naf_tune_span;
        sample.digits.reserve(bits);
        for (unsigned i = 0; i < strata; ++i) {
            // MSB-first strata. Short scalars are covered in full without overlaps.
            const auto first = uint64_t(i) * sample.columns / strata;
            const auto end = uint64_t(i + 1) * sample.columns / strata;
            const unsigned size = unsigned(std::min<uint64_t>(end - first, naf_tune_span));
            const unsigned start = unsigned(first + (end - first - size) / 2);
            for (unsigned j = 0; j < size; ++j) {
                const auto digit = int16_t(encoding.digit(sample.columns - 1 - start - j));
                sample.digits.push_back(digit);
                sample.nonzero += digit != 0;
            }
        }
        ecm_ts_fprintf(stdout, "stage1_ntt_naf_sample: b1=%llu naf=%u columns=%u sampled=%u nonzero=%u\n",
                       static_cast<unsigned long long>(b1_), window, sample.columns, bits, sample.nonzero);
        return windows_.emplace(window, std::move(sample)).first->second;
    }
};

class Calibration {
    const Device& device_;
    unsigned p_, bits_;
    std::vector<uint32_t> scalar_;
    unsigned window_ = 0;
    const NafSample* naf_sample_ = nullptr;
    size_t dictionary_limit_ = 0;
    std::vector<uint32_t> dictionary_;
    unsigned reference_batch_ = 0;
    std::vector<uint32_t> reference_;
public:
    Calibration(const Device& device, unsigned p, unsigned bits)
        : device_(device), p_(p), bits_(bits), scalar_((bits + 31) / 32) {
        std::mt19937 random(0x67ae128bu);
        for (auto& word : scalar_) word = random();
    }

    void native(unsigned window, const NafSample& sample, size_t dictionary_limit) {
        window_ = window;
        dictionary_limit_ = dictionary_limit;
        naf_sample_ = &sample;
        bits_ = unsigned(sample.digits.size());
        const unsigned words = (p_ + 31) / 32;
        dictionary_.resize(size_t(4u << (window - 2)) * words);
        ecm_edwards::Integer n, d, x, y, a24, factor;
        mpz_setbit(n.v, p_);
        mpz_sub_ui(n.v, n.v, 1);
        ecm_edwards::MersenneConstruction context(n.v);
        for (unsigned sigma = 26; sigma < 42; ++sigma) {
            if (ecm_edwards::construct(d.v, x.v, y.v, a24.v, factor.v, sigma, n.v, &context)
                    == ecm_edwards::Setup::Ready &&
                ecm_edwards::ntt::affine_dictionary(dictionary_, 1, 0, words, window,
                    x.v, y.v, d.v, n.v, factor.v) == ecm_edwards::Setup::Ready) return;
        }
        // A composite carrier can factor during every construction. The order-four
        // point (1,0) still gives a valid affine odd-multiple table over this ring.
        ecm_ts_fprintf(stdout, "stage1_ntt_probe: p=%u using order-four calibration point\n", p_);
        mpz_set_ui(x.v, 1); mpz_set_ui(y.v, 0); mpz_set_ui(d.v, 2);
        if (ecm_edwards::ntt::affine_dictionary(dictionary_, 1, 0, words, window,
                x.v, y.v, d.v, n.v, factor.v) != ecm_edwards::Setup::Ready)
            throw std::runtime_error("cannot build Edwards tune dictionary");
    }

    Measurement measure(unsigned batch, const Stage1NttOptions& options, unsigned samples, unsigned target_ms = 0) {
        if (ecm_stage1_stop_requested()) throw Cancelled();
        const unsigned words = (p_ + 31) / 32;
        const size_t stride = size_t(words) * batch;
        std::vector<uint32_t> active((window_ ? 3 : 6) * stride);
        Measurement m;
        m.batch = batch;
        Control control;
        control.describe = [&](const Plan& plan) { m.plan = plan; };
        control.poll = [](unsigned) { return Control::Action{false, ecm_stage1_stop_requested()}; };
        control.sample_target_ms = target_ms;
        Run run;
        if (window_) {
            m.naf_columns = naf_sample_->columns;
            m.sample_nonzero = naf_sample_->nonzero;
            if (m.naf_columns > bits_) control.sample_span_steps = ecm_stage1::naf_tune_span;
            for (unsigned lane = 0; lane < batch; ++lane)
                active[stride + lane * words] = active[2 * stride + lane * words] = 1;
            const size_t slots = dictionary_.size() / words;
            std::vector<uint32_t> table(slots * stride);
            for (size_t i = 0; i < slots; ++i)
                for (unsigned lane = 0; lane < batch; ++lane)
                    std::copy_n(dictionary_.data() + i * words, words, table.data() + i * stride + lane * words);
            EdwardsNaf naf{window_, naf_sample_->digits, table, dictionary_limit_};
            run = edwards_naf(device_.index, p_, batch, active, naf, options, control, samples);
        } else {
            for (unsigned slot=0; slot<6; ++slot) {
                std::mt19937 random(0x67ae128bu + slot);
                auto first = active.begin() + slot*stride;
                for (unsigned lane=0; lane<batch; ++lane, first+=words) {
                    std::generate_n(first,words,[&]{return random();});
                    if (p_ % 32) first[words-1] &= (1u << (p_%32)) - 1;
                    *first &= 0xfffffffeu; // Keep every residue below M.
                }
            }
            run = ladder(device_.index, p_, batch, active, scalar_, bits_, options, control, samples);
        }
        m.blocks = run.blocks_per_sample;
        m.steps = bits_ * m.blocks;
        if (run.stopped) throw Cancelled();
        // A larger batch appends lanes to each slot. Retain one reference and
        // check common lanes even when the plan or batch size changes.
        const size_t reference_stride = size_t(words) * reference_batch_;
        const size_t common = size_t(words) * std::min(batch, reference_batch_);
        if (common) {
            for (unsigned slot = 0; slot < (window_ ? 3u : 4u); ++slot) {
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

std::optional<Candidate> candidate(unsigned p, Stage1NttOptions options, bool prefer_radix5,
                                   WeightedBound bound) {
    try {
        const auto shape = resolve_packing(p, options, prefer_radix5, bound);
        if (options.tile <= tile_log(shape.length)) {
            options.length = shape.length;
            options.automatic = false;
            return Candidate{std::move(options), shape};
        }
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
        ? std::min(tile_log(shape.length), base.profile == 2 ? 10u : 12u) : 0;
    const unsigned fixed_outer = base.outer_max ? base.outer_max : base.profile ? 4 : 0;
    const int fixed_window = base.window >= 0 ? base.window : base.profile ? 16 : -1;
    const int fixed_warp = base.warp_tail >= 0 ? base.warp_tail : base.profile ? 1 : -1;
    if ((fixed_tile && tile != fixed_tile) || (fixed_outer && outer != fixed_outer) ||
        (fixed_window >= 0 && window != fixed_window) ||
        (fixed_warp == 0 && warp) || (fixed_warp == 1 && tile >= 5 && !warp)) return {};
    auto o = base;
    o.automatic = false;
    o.length = shape.length;
    o.bpw = shape.bpw;
    o.tile = tile;
    o.outer_max = outer;
    o.window = window;
    o.warp_tail = warp;
    if (ecm_stage1::stage1_ntt_radix5(shape.length) && row.count("dataflow")) {
        using Layout = Stage1NttOptions::MixedLayout;
        const auto flow = uint(row, "dataflow");
        o.mixed_layout = flow & CoalescedRadix5 ? Layout::Coalesced : flow & PfaRadix5 ? Layout::Pfa : Layout::Twiddle;
    }
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
    const auto bound = native_edwards ? WeightedBound::UnsignedProducts : WeightedBound::Conservative;
    Candidates shapes;
    if (!options.automatic) {
        const auto packing = resolve_packing(p, options, true, bound);
        const unsigned mode = !options.weighted ? Linear :
            ecm_stage1::stage1_ntt_radix5(packing.length) ? Mixed : Weighted;
        options.length = packing.length;
        shapes[mode] = Candidate{std::move(options), packing};
        return shapes;
    }
    options.weighted = false;
    shapes[Linear] = candidate(p, options, false, bound);
    options.weighted = true;
    auto binary = candidate(p, options, false, bound);
    if (binary && !ecm_stage1::stage1_ntt_radix5(binary->shape.length)) shapes[Weighted] = binary;
    auto mixed = candidate(p, options, true, bound);
    if (mixed && ecm_stage1::stage1_ntt_radix5(mixed->shape.length)) shapes[Mixed] = mixed;
    return shapes;
}

Candidates tune_shapes(unsigned p, const Stage1NttOptions& options, bool native_edwards) {
    try { return production_shapes(p, options, native_edwards); }
    catch (const std::invalid_argument&) { return {}; }
}

std::map<unsigned, Candidates> plan_tune_lengths(const Stage1NttOptions& options,
        const ecm_stage1::NttTuneGrid& grid, bool native_edwards, double deadline) {
    using namespace ecm_stage1;
    const auto range = grid.exp.first ? grid.exp : TuneRange{stage1_ntt_min_exponent, stage1_ntt_max_exponent};
    if (range.first == range.last) return {{range.first, tune_shapes(range.first, options, native_edwards)}};

    std::map<unsigned, Candidates> plan;
    auto add_first = [&](unsigned mode, unsigned length, unsigned first, unsigned last) {
        if (!grid.length.contains(length)) return;
        for (unsigned p = std::max(first, range.first); p <= std::min(last, range.last); ++p) {
            if (ecm_stage1_stop_requested()) throw Cancelled();
            if (deadline && now_s() >= deadline) throw TimeLimit{};
            auto shapes = tune_shapes(p, options, native_edwards);
            if (shapes[mode] && shapes[mode]->shape.length == length) {
                plan[p][mode] = std::move(shapes[mode]);
                break;
            }
        }
    };

    std::set<unsigned> lengths;
    for (unsigned n = 2; n <= stage1_ntt_max_length; n *= 2) {
        lengths.insert(n);
        if (stage1_ntt_supported_radix5(5 * n)) lengths.insert(5 * n);
    }
    unsigned previous_linear = 0, previous_binary = 0, previous_weighted = 0;
    for (unsigned length : lengths) {
        if (options.length && options.length != length) continue;
        const bool mixed = stage1_ntt_supported_radix5(length);
        if (!mixed && (options.automatic || !options.weighted)) {
            unsigned limit = 0;
            for (unsigned b = 16; b <= 31; ++b)
                if (!options.bpw || options.bpw == b)
                    limit = std::max(limit, linear_exponent_limit(length, b));
            // Linear packing selects its shortest length before applying the tile constraint.
            const unsigned first = std::max(range.first, previous_linear + 1);
            if (first <= limit) add_first(Linear, length, first, first);
            previous_linear = limit;
        }
        if ((!options.automatic && !options.weighted) || options.tile > tile_log(length)) continue;

        // Below 'guaranteed' this length always fits. For native unsigned products,
        // even the narrow digits contribute at least L*(2^floor(p/L)-1)^2.
        // Only the band between these bounds can contain isolated admissible p.
        const unsigned max_bpw = options.bpw ? options.bpw : 31;
        unsigned safe_bits = max_bpw;
        while (safe_bits && !exact_ok_terms(2ull * length, safe_bits)) --safe_bits;
        const unsigned guaranteed = length * safe_bits;
        unsigned possible = guaranteed;
        if (native_edwards && options.tight_weighted_bound && safe_bits < max_bpw) {
            unsigned narrow_bits = safe_bits;
            while (narrow_bits < max_bpw && exact_ok_terms(length, narrow_bits + 1)) ++narrow_bits;
            possible = std::min(length * max_bpw, length * (narrow_bits + 1) - 1);
        }
        // Automatic binary plans compete only with shorter powers of two. Mixed
        // plans and forced weighted convolution compete with every shorter length.
        const unsigned previous = options.automatic && !mixed ? previous_binary : previous_weighted;
        add_first(mixed ? Mixed : Weighted, length, std::max(length, previous + 1), possible);
        previous_weighted = std::max(previous_weighted, guaranteed);
        if (!mixed) previous_binary = std::max(previous_binary, guaranteed);
    }
    return plan;
}

std::array<BatchMemory, CandidateCount> production_capacity(const ecm_stage1::Stage1Request& request,
        const Device& device, unsigned p, unsigned window, const Candidates& shapes,
        bool reserve_dictionary_workspace = true) {
    if (window && (window < 2 || window > 12))
        throw std::invalid_argument("Edwards NTT NAF requires a window from 2 to 12");
    if (!std::isfinite(request.b1) || request.b1 < 0 || request.b1 > 5000000000.0)
        throw std::invalid_argument("invalid Stage1 NTT B1");
    const size_t words = (p + 31) / 32, entries = window ? 1u << (window - 2) : 0;
    const auto dictionary_mb = request.options.execution.naf_memory_mb;
    const auto dictionary_budget = size_t(dictionary_mb) * ecm_edwards_memory::mb;
    size_t host_cap = request.options.execution.ntt.max_batch;
    if (entries)
        if (const auto free = ecm_edwards_memory::host_free_bytes()) {
            const auto limit = ecm_edwards_memory::usable_bytes(*free, 64 * ecm_edwards_memory::mib);
            // A padded host table temporarily has both its old and expanded
            // lane layouts. Leave room for scalar and dictionary preparation.
            const size_t fixed = (size_t(request.b1) * 5 + 128 * entries * words);
            const auto per_curve = (32 * entries + 64) * words;
            host_cap = std::min(host_cap, limit > fixed ? (limit - fixed) / per_curve : 0);
        }
    std::array<BatchMemory, CandidateCount> capacities{};
    for (unsigned mode = 0; mode < CandidateCount; ++mode) {
        if (!shapes[mode]) continue;
        const auto length = size_t(shapes[mode]->shape.length);
        // The estimate covers spectra and dictionary workspace. The exact
        // allocator checks actual allocation separately.
        // A direct batch may use the production host builder when only the spectra fit.
        const auto dictionary = entries * (32 * length + (reserve_dictionary_workspace ? 16 * words : 0));
        const auto per_curve = 88 * length + 128 * words + dictionary;
        const auto fixed = 160 * length + 32 * words + size_t(request.b1 / 4 + 16);
        size_t cap = std::min(host_cap, device.budget > fixed ? (device.budget - fixed) / per_curve : 0);
        if (dictionary_budget && dictionary)
            cap = std::min(cap, dictionary_budget / dictionary);
        capacities[mode] = {per_curve, device.device_budget, request.options.execution.ntt.memory_mib,
                            window ? dictionary_mb : 0u, unsigned(cap)};
    }
    return capacities;
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

// Ask the production dispatcher for its exact tile and arithmetic flags before
// reusing timings. This path allocates no device tables and runs no ladder.
Plan describe(const Device& device, unsigned p, unsigned batch, const Stage1NttOptions& options, unsigned window) {
    Plan plan{};
    Control control;
    control.describe_only = true;
    control.describe = [&](const Plan& actual) { plan = actual; };
    if (window) {
        const std::vector<int16_t> digits;
        const std::vector<uint32_t> table;
        edwards_naf(device.index, p, batch, {}, EdwardsNaf{window, digits, table}, options, control);
    } else ladder(device.index, p, batch, {}, {}, 0, options, control);
    return plan;
}

bool compatible_workload(const Fields& row, const ecm_stage1::Stage1Request& request, unsigned window) {
    using ecm_stage2::tune::uint;
    return row.count("sample_target_ms") && pt::text(row,"curve") == curve_name(request) &&
           uint(row,"naf_window") == window &&
           (!window || pt::is_stratified_naf(row) || uint(row,"b1") == uint64_t(request.b1));
}

// The 512-curve preview wrote 0.72.0 rows with the same timing protocol as 0.71.3.
constexpr std::string_view timing_version(std::string_view version) {
    if (std::string_view(ecm_program::version) == "0.71.5" &&
        (version == "0.71.4" || version == "0.71.5" || version == "0.72.0"))
        return "0.71.3";
    return version;
}

bool same_workload(const Fields& row, const ecm_stage1::Stage1Request& request, unsigned window) {
    return compatible_workload(row,request,window) && row.count("benchmark_kind") &&
           pt::text(row,"benchmark_kind") == (window ? pt::stratified_naf_kind : pt::stage1_graph_ntt_kind) &&
           timing_version(pt::text(row,"program_version")) == timing_version(ecm_program::version);
}

bool same_dispatch(const Fields& row, unsigned p, const Plan& plan) {
    using ecm_stage2::tune::uint;
    return uint(row,"bpw") == plan.bpw && (uint(row,"reference_bits")+4095)/4096 == (p+4095)/4096 &&
           uint(row,"dataflow") == plan.dataflow && uint(row,"carry_bits") == plan.carry_bits &&
           uint(row,"segments") == (p + 8191) / 8192 &&
           uint(row,"uniform_digits") == unsigned(plan.weighted && p % plan.length == 0) &&
           uint(row,"tile") == plan.tile && uint(row,"max_radix_bits") == plan.outer_max &&
           uint(row,"graph_bits") == plan.window && uint(row,"warp_tail") == unsigned(plan.warp_tail);
}

bool reusable_dispatch(const Fields& row, const Device& device, unsigned p, unsigned batch,
                       const Stage1NttOptions& options, unsigned window) {
    try {
        return same_dispatch(row,p,describe(device,p,batch,options,window));
    } catch (const BudgetError&) {
        return false;
    }
}

struct Score {
    unsigned batch = 0;
    Plan plan{};
    double rate = 0;
    double relative_sd = 0;
};

enum BatchRole : unsigned { Single = 1, Capacity = 2, Fastest = 4, Economical = 8 };

void table_header() {
    ecm_ts_fprintf(stdout,"length max_exp ntt_type batch iter/s\n");
}

void round_header(unsigned p, unsigned window, const char* phase) {
    ecm_ts_fprintf(stdout,"stage1_ntt_round: p=%u naf=%u phase=%s\n",p,window,phase);
    table_header();
}

void report_batches(const std::map<BatchRole,Score>& scores) {
    std::vector<std::array<std::string,7>> rows;
    std::array<int,6> widths{};
    for (const auto& [role,label] : {std::pair{Single,"Single"}, {Capacity,"Max"},
                                    {Fastest,"Best"}, {Economical,"95%"}}) {
        const auto found = scores.find(role);
        if (found == scores.end()) continue;
        const auto& s = found->second;
        std::ostringstream rate;
        rate << std::fixed << std::setprecision(0) << s.rate;
        rows.push_back({label,std::to_string(s.plan.length),std::to_string(max_exponent(s.plan)),
                        tune_shape_descriptor(s.plan),std::to_string(s.batch)+"c",rate.str(),
                        std::to_string(s.plan.dataflow)});
        for (size_t i=0; i<widths.size(); ++i)
            widths[i] = std::max(widths[i],int(rows.back()[i].size()));
    }
    for (const auto& row : rows)
        ecm_log_accent(EcmLogAccent::Report,"%-*s %*s %*s %-*s %*s %*s iter/s dataflow=%s\n",
            widths[0],row[0].c_str(),widths[1],row[1].c_str(),widths[2],row[2].c_str(),
            widths[3],row[3].c_str(),widths[4],row[4].c_str(),widths[5],row[5].c_str(),row[6].c_str());
}

enum class SamplePhase : unsigned { Coarse = 1, Fine = 2 };
SamplePhase sample_phase(const Fields& row) {
    const unsigned legacy = ecm_stage2::tune::uint(row,"repeats") == 5 ? 2 : 1;
    return static_cast<SamplePhase>(pt::optional_uint(row,"sample_phase",legacy));
}

std::vector<Fields> compact_batch_rows(std::vector<Fields> rows, const std::string& search) {
    using ecm_stage2::tune::uint;
    // Compare fine measurements when available. Coarse-only families still
    // establish a genuine upper admission endpoint even if pruned in refinement.
    const auto quality = [&](size_t j) {
        return std::make_pair(sample_phase(rows[j]),ecm_stage2::tune::real(rows[j],"iter_per_s"));
    };
    std::map<unsigned,size_t> winners;
    for (size_t i=0; i<rows.size(); ++i) {
        const auto b = unsigned(uint(rows[i],"batch"));
        if (!winners.count(b) || quality(i)>quality(winners.at(b))) winners[b]=i;
    }
    const auto best = std::max_element(winners.begin(),winners.end(),[&](const auto& a,const auto& b){
        return quality(a.second)<quality(b.second);
    });
    const auto peak = quality(best->second);
    const auto economic = std::find_if(winners.begin(),winners.end(),[&](const auto& a){
        const auto q=quality(a.second);
        return q.first==peak.first && q.second>=near_peak*peak.second;
    });
    std::map<unsigned,unsigned> roles;
    if (winners.count(1)) roles[1] = Single;
    roles[winners.rbegin()->first] |= Capacity;
    roles[best->first] |= Fastest;
    roles[economic->first] |= Economical;
    std::vector<Fields> retained;
    for (size_t i=0; i<rows.size(); ++i) {
        auto& row=rows[i];
        const auto b=unsigned(uint(row,"batch"));
        if (!roles.count(b)) continue;
        if (!search.empty()) row["batch_search"]=pt::quoted(search);
        row["batch_roles"]=std::to_string(winners.at(b)==i ? roles.at(b) : 0);
        retained.push_back(std::move(row));
    }
    return retained;
}

class TuneMatrix {
    const Device& device_;
    const ecm_stage1::Stage1Request& request_;
    unsigned p_, window_;
    ecm_stage1::NttTuneGrid grid_;
    double deadline_;
    TuneNafSamples& naf_samples_;
    std::optional<Calibration> calibration_;
    std::vector<Fields> history_, legacy_, pending_;
    std::set<size_t> hits_;
    std::string checkpoint_;
    std::map<unsigned,std::pair<SamplePhase,Score>> length_bests_;

    Score observe(SamplePhase phase, const Score& score) {
        auto& best = length_bests_[score.plan.length];
        if (phase > best.first || (phase == best.first && score.rate > best.second.rate))
            best = {phase,score};
        return score;
    }
public:
    unsigned measured = 0;
    TuneMatrix(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p,
               unsigned window, const pt::Profile& db, TuneNafSamples& naf_samples, const ecm_stage1::NttTuneGrid& grid = {}, double deadline = 0)
        : device_(device), request_(request), p_(p), window_(window), grid_(grid), deadline_(deadline),
          naf_samples_(naf_samples) {
        for (const auto& row : db.stage1_ntt) {
            const auto& condition = db.conditions.at(ecm_stage2::tune::uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid ||
                pt::is_runtime_stage1_ntt(row)) continue;
            if (same_workload(row, request, window)) history_.push_back(row);
            else if (!row.count("sample_target_ms")) legacy_.push_back(row);
        }
        std::stable_partition(history_.begin(),history_.end(),[](const Fields& r){ return !r.count("batch_scope"); });
    }
    void release_calibration() { calibration_.reset(); }
    size_t reused() const { return hits_.size(); }
    void report_lengths() const {
        for (const auto& [length,best] : length_bests_) {
            const auto& [phase,score] = best;
            ecm_log_accent(EcmLogAccent::Report,
                "stage1_ntt_length_done: p=%u naf=%u length=%u phase=%s batch=%u graph=%u iter_per_s=%.9g\n",
                p_,window_,length,phase==SamplePhase::Fine ? "fine" : "coarse",
                score.batch,score.plan.window,score.rate);
        }
    }
    void open_checkpoint(const std::string& scope, const std::string& search) {
        const auto& path = request_.options.execution.ntt.tune_file;
        if (path.empty()) return;
        const auto key = device_.uuid + "/" + scope + "/" + search + "/" + std::to_string(grid_.group_ms);
        checkpoint_ = path + "." + std::to_string(ecm_io::fingerprint(key)) + ".partial";
        if (!std::filesystem::exists(checkpoint_)) return;
        ecm_io::FileLock lock(checkpoint_);
        for (const auto& row : load(checkpoint_).stage1_ntt)
            if (row.count("batch_scope") && pt::text(row,"batch_scope")==scope && same_workload(row,request_,window_))
                history_.push_back(row);
    }
    void finish_checkpoint() {
        if (checkpoint_.empty() || !std::filesystem::exists(checkpoint_)) return;
        ecm_io::FileLock lock(checkpoint_);
        std::filesystem::remove(checkpoint_);
    }
    void recheck(Score& best, const std::string& scope, const std::string& search) {
        using ecm_stage2::tune::uint;
        using ecm_stage2::tune::real;
        const auto& path = request_.options.execution.ntt.tune_file;
        const auto matches = [&](const Fields& row) {
            return same_workload(row,request_,window_) && sample_phase(row)==SamplePhase::Fine &&
                   uint(row,"batch")==best.batch && uint(row,"length")==best.plan.length &&
                   (pt::text(row,"ntt_type")==pt::weighted_ntt_type)==best.plan.weighted &&
                   uint(row,"repeats")==grid_.fine_repeats && same_dispatch(row,p_,best.plan);
        };
        unsigned target = 0;
        for (auto i : hits_) {
            const auto& row = history_[i];
            if (matches(row) && real(row,"iter_per_s")==best.rate) {
                target = unsigned(uint(row,"sample_target_ms"));
                break;
            }
        }
        if (!target || path.empty()) return;
        round_header(p_,window_,"recheck");
        Fields fresh;
        auto options = fixed_plan(request_.options.execution.ntt,best.plan);
        const auto test = sample(options,best.batch,SamplePhase::Fine,&fresh,target);
        const double previous = best.rate;
        const char* status = test.rate ? "kept" : "unavailable";
        if (test.rate > previous) {
            // Another process may have raised the record or changed this search while the GPU ran.
            ecm_io::FileLock lock(path);
            auto db = load(path);
            const auto belongs = [&](const Fields& row) {
                const auto& condition = db.conditions.at(uint(row,"condition"));
                if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex")!=device_.uuid) return false;
                if (scope.empty()) return !row.count("batch_scope");
                return row.count("batch_scope") && pt::text(row,"batch_scope")==scope &&
                       row.count("batch_search") && pt::text(row,"batch_search")==search;
            };
            auto at = db.stage1_ntt.end();
            for (auto it = db.stage1_ntt.begin(); it != db.stage1_ntt.end(); ++it) {
                if (belongs(*it) && matches(*it) && uint(*it,"sample_target_ms")==target &&
                    (scope.empty() || (uint(*it,"batch_roles") & Fastest)) &&
                    (at == db.stage1_ntt.end() || real(*it,"iter_per_s") > real(*at,"iter_per_s"))) at = it;
            }
            if (at == db.stage1_ntt.end()) status = "superseded";
            if (at != db.stage1_ntt.end() && test.rate > real(*at,"iter_per_s")) {
                fresh["condition"] = at->at("condition");
                if (!scope.empty()) {
                    fresh["batch_scope"] = at->at("batch_scope");
                    fresh["batch_search"] = at->at("batch_search");
                }
                pt::validate_stage1_ntt(fresh);
                *at = std::move(fresh);
                if (!scope.empty()) {
                    std::vector<Fields> group;
                    for (const auto& row : db.stage1_ntt) if (belongs(row)) group.push_back(row);
                    auto retained = compact_batch_rows(std::move(group),search);
                    db.stage1_ntt.erase(std::remove_if(db.stage1_ntt.begin(),db.stage1_ntt.end(),belongs),db.stage1_ntt.end());
                    db.stage1_ntt.insert(db.stage1_ntt.end(),retained.begin(),retained.end());
                    history_ = std::move(retained);
                }
                ecm_io::write_atomic(path,db.serialize());
                best = test;
                status = "updated";
            }
        }
        const char* format = "stage1_ntt_recheck: p=%u naf=%u length=%u batch=%u graph=%u previous_iter_per_s=%.9g measured_iter_per_s=%.9g status=%s\n";
        if (best.rate > previous)
            ecm_log_accent(EcmLogAccent::Improved,format,p_,window_,best.plan.length,best.batch,best.plan.window,previous,test.rate,status);
        else
            ecm_ts_fprintf(stdout,format,p_,window_,best.plan.length,best.batch,best.plan.window,previous,test.rate,status);
    }
    std::optional<Stage1NttOptions> previous_plan(const Stage1NttOptions& options, unsigned batch,
                                                  const Plan& plan) const {
        std::optional<Cached> best;
        for (const auto& row : legacy_) {
            if ((pt::text(row,"ntt_type")==pt::weighted_ntt_type) != plan.weighted) continue;
            auto hint = decode(row,{plan.length,plan.bpw,plan.digits},device_,batch,options);
            if (hint && (!best || hint->rate>best->rate)) best=std::move(hint);
        }
        return best ? std::optional<Stage1NttOptions>(best->options) : std::nullopt;
    }
    unsigned interior(const Stage1NttOptions& options, unsigned cap) const {
        if (cap <= 2) return 1;
        const unsigned fallback = std::min(cap - 1, p_ >= 262144 ? 4u : 8u);
        std::map<unsigned, BatchRate> prior;
        using ecm_stage2::tune::uint;
        for (const auto& row : history_) {
            const auto b = unsigned(uint(row,"batch"));
            if (b <= 1 || b >= cap || sample_phase(row) != SamplePhase::Coarse || uint(row,"repeats") != grid_.coarse_repeats ||
                uint(row,"length") != options.length || uint(row,"peak_bytes") > device_.budget ||
                (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != options.weighted) continue;
            if (!reusable_dispatch(row,device_,p_,b,options,window_)) continue;
            if (b == fallback) return b;
            prior[b] = {ecm_stage2::tune::real(row,"iter_per_s"),1};
        }
        const unsigned previous = recommended_batch(prior);
        return previous ? previous : fallback;
    }
    void flush(bool explicit_batch = false) {
        if (explicit_batch) {
            for (auto i : hits_) {
                if (!history_[i].count("batch_scope")) continue;
                auto row = history_[i];
                row.erase("batch_scope");
                row.erase("batch_search");
                row.erase("batch_roles");
                pending_.push_back(std::move(row));
            }
        }
        persist(request_.options.execution.ntt.tune_file, device_, pending_);
        pending_.clear();
    }
    bool restore_batches(const std::string& scope, const std::string& search,
                         std::map<BatchRole,Score>& result, Score& endpoint) {
        Score coarse;
        using ecm_stage2::tune::uint;
        std::map<BatchRole,Score> selected;
        std::set<size_t> matches;
        for (size_t i=0; i<history_.size(); ++i) {
            const auto& row = history_[i];
            if (!row.count("batch_scope") || pt::text(row,"batch_scope") != scope ||
                !row.count("batch_search") || pt::text(row,"batch_search") != search) continue;
            if (uint(row,"sample_target_ms") < grid_.group_ms) return false;
            const unsigned batch = unsigned(uint(row,"batch"));
            auto options = request_.options.execution.ntt;
            options.weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            const Packing shape{unsigned(uint(row,"length")),unsigned(uint(row,"bpw"))};
            const auto decoded = decode(row,shape,device_,batch,options);
            if (!decoded) return false;
            try {
                const auto plan = describe(device_,p_,batch,decoded->options,window_);
                if (!same_dispatch(row,p_,plan)) return false;
                observe(sample_phase(row),{batch,plan,decoded->rate});
                if (sample_phase(row) == SamplePhase::Coarse &&
                    (batch > coarse.batch || (batch == coarse.batch && decoded->rate > coarse.rate)))
                    coarse = {batch,plan,decoded->rate};
                for (auto role : {Single, Capacity, Fastest, Economical})
                    if (pt::optional_uint(row,"batch_roles",0) & role)
                        selected[role] = {batch,plan,decoded->rate};
            } catch (const BudgetError&) { return false; }
            matches.insert(i);
        }
        if (!selected.count(Capacity) || !selected.count(Fastest) || !selected.count(Economical)) return false;
        if (!coarse.rate) return false;
        endpoint = coarse;
        result = std::move(selected);
        hits_ = std::move(matches);
        return true;
    }

    void save_batches(const std::string& scope, const std::string& search, bool complete) {
        using ecm_stage2::tune::uint;
        std::vector<Fields> rows = std::move(pending_);
        for (auto i : hits_) rows.push_back(history_[i]);
        if (rows.empty()) return;
        for (auto& row : rows) {
            row["batch_scope"]=pt::quoted(scope);
            row.erase("batch_search");
            row.erase("batch_roles");
        }
        if (!complete && !checkpoint_.empty()) persist(checkpoint_,device_,rows);
        auto retained = compact_batch_rows(std::move(rows),complete ? search : std::string{});
        persist(request_.options.execution.ntt.tune_file,device_,std::move(retained),false,scope,!complete);
        if (complete) finish_checkpoint();
    }
    Score sample(const Stage1NttOptions& options, unsigned batch, SamplePhase phase,
                 Fields* remeasurement = nullptr, unsigned saved_target = 0) {
        const unsigned groups = phase == SamplePhase::Fine ? grid_.fine_repeats : grid_.coarse_repeats;
        const unsigned target = remeasurement ? saved_target : grid_.group_ms;
        if (ecm_stage1_stop_requested()) throw Cancelled();
        try {
            auto plan = describe(device_, p_, batch, options, window_);
            using ecm_stage2::tune::uint;
            const auto lookup = [&](const std::vector<Fields>& rows, bool previous) -> Score {
                Score best;
                size_t selected = 0;
                for (size_t i=0; i<rows.size(); ++i) {
                    const auto& row = rows[i];
                    if (uint(row,"length") != plan.length || uint(row,"bpw") != plan.bpw ||
                        (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != plan.weighted ||
                        sample_phase(row) != phase || uint(row,"sample_target_ms") < target || uint(row,"repeats") != groups || uint(row,"batch") != batch ||
                        uint(row,"peak_bytes") > device_.budget || !same_dispatch(row,p_,plan)) continue;
                    // An explicit search owns its existing unscoped rows. Scoped
                    // samples can seed it when that configuration is missing.
                    if (best.rate && grid_.batch_explicit && !rows[selected].count("batch_scope") &&
                        row.count("batch_scope")) continue;
                    const auto rate = ecm_stage2::tune::real(row,"iter_per_s");
                    if (rate <= best.rate) continue;
                    selected = i;
                    best = {batch, plan, rate,
                        1000 * ecm_stage2::tune::real(row,"stddev_seconds") /
                            (window_ ? ecm_stage2::tune::real(row,"total_device_ms") / groups :
                                       ecm_stage2::tune::real(row,"median_ms"))};
                }
                if (previous && best.rate && hits_.insert(selected).second)
                    ecm_ts_fprintf(stdout,"stage1_ntt_reuse: length=%u batch=%u dataflow=%u tile=%u "
                        "outer=%u graph=%u target_ms=%u iter_per_s=%.0f\n",plan.length,batch,plan.dataflow,
                        plan.tile,plan.outer_max,plan.window,target,best.rate);
                return best;
            };
            if (!remeasurement) {
                if (auto hit = lookup(pending_, false); hit.rate) return observe(phase,hit);
                if (auto hit = lookup(history_, true); hit.rate) return observe(phase,hit);
            }
            if (deadline_ && now_s() >= deadline_) throw TimeLimit{};
            if (!calibration_) {
                Calibration prepared(device_, p_, 64);
                if (window_) prepared.native(window_, naf_samples_.get(window_),
                    size_t(request_.options.execution.naf_memory_mb) * ecm_edwards_memory::mb);
                calibration_.emplace(std::move(prepared));
            }
            const auto m = calibration_->measure(batch, options, groups, target);
            auto row = encode(m, m.steps);
            row["benchmark_kind"] = pt::quoted(window_ ? pt::stratified_naf_kind : pt::stage1_graph_ntt_kind);
            if (window_) {
                row["total_device_ms"] = pt::numeric(m.total_ms());
                row["naf_columns"] = std::to_string(m.naf_columns);
                row["sample_nonzero"] = std::to_string(m.sample_nonzero);
            }
            row["sample_phase"] = std::to_string(unsigned(phase));
            row["sample_target_ms"] = std::to_string(target);
            row["reference_bits"] = std::to_string(p_);
            row["timed_steps"] = std::to_string(m.steps);
            row["block_steps"] = std::to_string(m.steps / m.blocks);
            row["curve"] = pt::quoted(curve_name(request_));
            row["b1"] = std::to_string(uint64_t(request_.b1));
            row["naf_window"] = std::to_string(window_);
            row["segments"] = std::to_string((p_ + 8191) / 8192);
            row["uniform_digits"] = std::to_string(unsigned(m.plan.weighted && p_ % m.plan.length == 0));
            row["dataflow"] = std::to_string(m.plan.dataflow);
            row["carry_bits"] = std::to_string(m.plan.carry_bits);
            row["median_ms"] = pt::numeric(m.time());
            row["stddev_seconds"] = pt::numeric(pt::deviation(m.ms) / 1000);
            if (remeasurement) *remeasurement = std::move(row);
            else pending_.push_back(std::move(row));
            ++measured;
            print_measurement(m, target);
            return observe(phase,{batch,m.plan,m.throughput()*m.steps,pt::deviation(m.ms)/m.group_ms()});
        } catch (const BudgetError& e) {
            ecm_ts_fprintf(stdout, "stage1_ntt_skip: length=%u batch=%u reason=%s\n", options.length,batch,e.what());
            return {};
        } catch (const std::invalid_argument& e) {
            ecm_ts_fprintf(stdout, "stage1_ntt_skip: length=%u batch=%u reason=%s\n", options.length,batch,e.what());
            return {};
        } catch (const std::runtime_error& e) {
            if (std::string_view(e.what()) != cudaGetErrorString(cudaErrorMemoryAllocation)) throw;
            cudaGetLastError();
            ecm_ts_fprintf(stdout,"stage1_ntt_skip: length=%u batch=%u reason=cuda_allocation\n",options.length,batch);
            return {};
        }
    }
};

struct Family {
    Stage1NttOptions options;
    unsigned cap;
    Score single, peak;
    std::map<unsigned, Score> best;
    void keep(const Score& score) {
        if (!score.rate) return;
        if (score.batch == 1 && score.rate > single.rate) single = score;
        if (score.rate > peak.rate) peak = score;
        if (score.rate > best[score.batch].rate) best[score.batch] = score;
    }
};

void refine_batch(TuneMatrix& matrix, Family& f, unsigned batch, std::map<unsigned,Score>& finals) {
    if (f.best.count(batch)) return;
    auto best = matrix.sample(f.options,batch,SamplePhase::Fine);
    if (!best.rate) return;
    const auto base = best.plan;
    std::vector<Stage1NttOptions> trials;
    const auto& o = f.options;
    if (auto previous=matrix.previous_plan(o,batch,base)) trials.push_back(*previous);
    if (!o.profile && !o.tile && base.tile >= 5) {
        auto alt=o; alt.tile=base.tile-2; trials.push_back(alt);
        if (batch==1 && base.weighted && base.length==16384 && base.tile==12 && !o.outer_max) {
            alt.outer_max=2;
            trials.push_back(alt);
        }
    }
    if (!o.profile && !o.outer_max && tile_log(base.length)>base.tile) {
        auto alt=o; alt.outer_max=base.outer_max==2 ? 4 : 2; trials.push_back(alt);
    }
    for (const auto& alt : trials) {
        auto score = matrix.sample(alt,batch,SamplePhase::Fine);
        if (score.rate>best.rate) best = score;
    }
    f.keep(best);
    if (best.rate>finals[batch].rate) finals[batch]=best;
}

struct WindowSearch {
    unsigned w, p;
    const ecm_stage1::NttTuneGrid& grid;
    TuneMatrix matrix;
    std::vector<Family> families;
    std::ostringstream scope, search;
    Score capacity;
    const bool scan_graph;
    double single = 0, peak = 0;
    bool reached_caps = true, cached_fine = false, persisted = false;
    double seconds = 0;
    std::map<BatchRole,Score> cached;
    std::map<unsigned,Score> finals;

    WindowSearch(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p, unsigned w,
                 const Candidates& shapes, const ecm_stage1::NttTuneGrid& grid, TuneNafSamples& naf_samples,
                 const pt::Profile& db, bool exhaustive, double deadline)
        : w(w), p(p), grid(grid), matrix(device,request,p,w,db,naf_samples,grid,deadline),
          scan_graph(!request.options.execution.ntt.profile && request.options.execution.ntt.window < 0) {
        using Layout = Stage1NttOptions::MixedLayout;
        const auto& options = request.options.execution.ntt;
        const auto caps = production_capacity(request,device,p,w,shapes);
        for (unsigned mode=0; mode<CandidateCount; ++mode) {
            if (!shapes[mode]) continue;
            const auto& shape = *shapes[mode];
            const unsigned cap = std::min(caps[mode].max_curves, grid.batch.last);
            ecm_ts_fprintf(stdout,"stage1_ntt_plan: p=%u length=%u bpw=%u naf=%u capacity=%u%s\n",
                p,shape.shape.length,shape.shape.bpw,w,cap,cap ? "" : " reason=memory_budget");
            if (cap < grid.batch.first) continue;
            if (mode == Mixed) {
                for (auto layout : {Layout::Twiddle,Layout::Pfa,Layout::Coalesced}) {
                    auto o = shape.options;
                    o.mixed_layout = layout;
                    families.push_back({o, cap});
                }
            } else families.push_back({shape.options, cap});
        }
        // Hold Graph size constant while comparing windows, layouts and batches.
        if (scan_graph)
            for (auto& family : families) family.options.window = 16;
        if (families.empty() || grid.batch_explicit) return;
        scope << timing_version(ecm_program::version) << '/' << curve_name(request) << '/' << w << '/' << (p-1)/4096
              << '/' << options.profile << '/' << options.tile << '/' << options.outer_max
              << '/' << options.window << '/' << options.warp_tail;
        search << exhaustive << '/' << grid.batch.first << "/final-graph/repeat=" << grid.coarse_repeats << ':' << grid.fine_repeats;
        for (const auto& shape : shapes) {
            if (!shape) continue;
            auto identity = shape->options;
            identity.memory_mib = 0;
            scope << ':' << identity.weighted << '/' << identity.length << '/' << int(identity.mixed_layout);
            try {
                const auto base=describe(device,p,1,identity,w);
                scope << '/' << base.bpw << '/' << base.dataflow << '/' << base.carry_bits
                      << '/' << (base.weighted && p%base.length==0);
            } catch (const BudgetError&) { scope << "/unavailable/" << p; }
        }
        for (const auto& f : families)
            search << ':' << f.options.weighted << '/' << f.options.length << '/' << int(f.options.mixed_layout) << '/' << f.cap;
        matrix.open_checkpoint(scope.str(),search.str());
        cached_fine = matrix.restore_batches(scope.str(),search.str(),cached,capacity);
        if (!cached_fine)
            matrix.restore_batches(scope.str(),search.str()+"/screen",cached,capacity);
    }

    void coarse() {
        for (auto& f : families) {
            // If the estimated upper endpoint fails actual admission, bisect
            // against a measured lower endpoint before ranking the family.
            for (auto b : std::set<unsigned>{grid.batch.first, std::max(grid.batch.first, matrix.interior(f.options,f.cap)), f.cap}) {
                auto score = matrix.sample(f.options,b,SamplePhase::Coarse);
                f.keep(score);
                if (!score.rate) {
                    unsigned lo = f.best.empty() ? grid.batch.first-1 : f.best.rbegin()->first, hi = b;
                    while (hi > lo + 1) {
                        const unsigned mid = lo + (hi-lo)/2;
                        auto test = matrix.sample(f.options,mid,SamplePhase::Coarse);
                        f.keep(test);
                        if (test.rate) lo = mid;
                        else hi = mid;
                    }
                    f.cap = lo;
                    reached_caps = false;
                    break;
                }
            }
            if (f.best.count(grid.batch.first)) f.single = f.best.at(grid.batch.first);
            if (!f.best.empty()) {
                const auto& endpoint = f.best.rbegin()->second;
                if (endpoint.batch>capacity.batch || (endpoint.batch==capacity.batch && endpoint.rate>capacity.rate))
                    capacity=endpoint;
            }
            single = std::max(single,f.single.rate);
            peak = std::max(peak,f.peak.rate);
        }
    }

    bool refine(bool exhaustive) {
        if (cached_fine) return true;
        if (!cached.empty()) coarse(); // Rebuild per-family anchors if a screened window now advances.
        std::map<unsigned,std::vector<Family*>> rounds;
        for (auto& f : families) {
            if (!f.cap || !f.peak.rate) continue;
            const auto close = [](const Score& s, double top) {
                return s.rate >= top * (1 - std::clamp(2*s.relative_sd,0.05,0.20));
            };
            const auto endpoint = f.best.find(capacity.batch);
            const bool covers_capacity = endpoint != f.best.end() && endpoint->second.rate == capacity.rate;
            if (!exhaustive && !covers_capacity && !close(f.single,single) && !close(f.peak,peak)) {
                f.best.clear();
                continue;
            }
            std::set<unsigned> batches{grid.batch.first,f.peak.batch,f.cap};
            for (unsigned b=grid.batch.first; b<=f.cap; b=exhaustive ? b+1 : 2*b) batches.insert(b);
            for (auto batch : batches) rounds[batch].push_back(&f);
            f.best.clear();
            f.peak = {};
            f.single = {};
        }
        // Adjacent measurements compare competing layouts at the same batch.
        // Alternate their order across batches to distribute clock drift.
        bool reverse = false;
        for (auto& [batch, contenders] : rounds) {
            if (reverse) std::reverse(contenders.begin(),contenders.end());
            for (auto* f : contenders) refine_batch(matrix,*f,batch,finals);
            reverse = !reverse;
        }
        if (!exhaustive) {
            for (auto& f : families) {
                if (f.best.empty() || !f.peak.batch) continue;
                const unsigned center=f.peak.batch;
                for (unsigned b : {center-1,center+1})
                    if (b>=grid.batch.first && b<=f.cap) refine_batch(matrix,f,b,finals);
                unsigned low=grid.batch.first-1, high=f.peak.batch;
                for (const auto& item : f.best) {
                    if (item.second.rate >= near_peak*f.peak.rate) { high=item.first; break; }
                    low=item.first;
                }
                while (high>low+1) {
                    const unsigned mid=low+(high-low)/2;
                    refine_batch(matrix,f,mid,finals);
                    if (f.best.count(mid) && f.best.at(mid).rate>=near_peak*f.peak.rate) high=mid;
                    else low=mid;
                }
            }
        }
        if (finals.empty()) return false;
        const auto best = std::max_element(finals.begin(),finals.end(),[](const auto& a,const auto& b){
            return a.second.rate<b.second.rate;
        });
        if (scan_graph) {
            auto& winner = best->second;
            auto options = fixed_plan(families.front().options,winner.plan);
            ecm_ts_fprintf(stdout,"stage1_ntt_graph_scan: p=%u naf=%u batch=%u length=%u\n",
                           p,w,winner.batch,winner.plan.length);
            table_header();
            for (int steps : {4, 8, 32, 64}) {
                options.window = steps;
                const auto score = matrix.sample(options,winner.batch,SamplePhase::Fine);
                if (score.rate > winner.rate) winner = score;
            }
        }
        return true;
    }

    void report(bool repeat) {
        if (cached_fine) {
            if (repeat) {
                auto& best = cached.at(Fastest);
                const double previous = best.rate;
                matrix.recheck(best,scope.str(),search.str());
                if (best.rate > previous)
                    matrix.restore_batches(scope.str(),search.str(),cached,capacity);
            }
            report_batches(cached);
            return;
        }
        const auto best = std::max_element(finals.begin(),finals.end(),[](const auto& a,const auto& b){
            return a.second.rate<b.second.rate;
        });
        if (repeat && grid.batch_explicit) matrix.recheck(best->second,{},{});
        const auto economical = std::find_if(finals.begin(),finals.end(),[&](const auto& item){
            return item.second.rate>=near_peak*best->second.rate;
        });
        std::map<BatchRole,Score> scores{{Capacity,finals.count(capacity.batch) ? finals.at(capacity.batch) : capacity},
                                        {Fastest,best->second},{Economical,economical->second}};
        if (finals.count(1)) scores[Single] = finals.at(1);
        report_batches(scores);
    }

    void save(bool complete, bool refined) {
        if (grid.batch_explicit) matrix.flush(true);
        else {
            const auto id = search.str() + (reached_caps ? "" : "/recheck") + (refined ? "" : "/screen");
            matrix.save_batches(scope.str(),id,complete);
        }
    }
};

bool tune_windows(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p,
                  const Candidates& shapes, const ecm_stage1::NttTuneGrid& grid, TuneNafSamples& naf_samples,
                  const std::vector<unsigned>& windows, bool compete, bool exhaustive, double deadline,
                  bool recheck_cached = false) {
    const auto db = load(request.options.execution.ntt.tune_file);
    std::deque<WindowSearch> searches;
    std::map<unsigned,WindowSearch*> leaders;
    bool finished = false;
    try {
        for (unsigned w : windows) {
            if (ecm_stage1_stop_requested()) throw Cancelled();
            if (deadline && now_s() >= deadline) throw TimeLimit{};
            const double started = now_s();
            round_header(p,w,"coarse");
            auto& trial = searches.emplace_back(device,request,p,w,shapes,grid,naf_samples,db,exhaustive,deadline);
            if (trial.families.empty()) continue;
            if (trial.cached.empty()) trial.coarse();
            if (windows.size() > 1) trial.matrix.release_calibration();
            trial.seconds = now_s() - started;
            if (!trial.capacity.rate) continue;
            auto& leader = leaders[trial.capacity.batch];
            if (!leader || trial.capacity.rate > leader->capacity.rate) leader = &trial;
        }
        for (auto& trial : searches) {
            if (!trial.capacity.rate) continue;
            const double started = now_s();
            const bool advance = !compete || leaders.at(trial.capacity.batch) == &trial;
            if (advance)
                round_header(p,trial.w,"fine");
            if (compete)
                ecm_ts_fprintf(stdout,"stage1_ntt_naf_screen: p=%u naf=%u capacity=%u iter_per_s=%.0f refine=%u\n",
                               p,trial.w,trial.capacity.batch,trial.capacity.rate,unsigned(advance));
            const bool refined = advance && trial.refine(exhaustive);
            finished |= refined;
            if (trial.cached.empty() || (refined && !trial.cached_fine))
                trial.save(!advance || refined,refined);
            else trial.matrix.finish_checkpoint();
            trial.persisted = true;
            if (refined) trial.report(recheck_cached && !trial.matrix.measured);
            trial.matrix.release_calibration();
            trial.matrix.report_lengths();
            ecm_log_accent(EcmLogAccent::Report,"stage1_ntt_tune: p=%u measured=%u reused=%zu seconds=%.3f naf=%u\n",
                           p,trial.matrix.measured,trial.matrix.reused(),trial.seconds+now_s()-started,trial.w);
        }
    } catch (...) {
        for (auto& trial : searches)
            if (!trial.persisted && trial.matrix.measured) trial.save(false,false);
        throw;
    }
    return finished;
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

int tune_command(const ecm_stage1::Stage1Options& task, double b1, const ecm_stage1::NttTuneGrid& grid,
                 bool exhaustive, unsigned seconds) {
    using namespace ecm_stage1;
    try {
        if (!(b1 >= 2 && b1 <= 5000000000.0)) throw std::invalid_argument("invalid tune B1");
        if (grid.exp.last > stage1_ntt_max_exponent || (grid.exp.first && grid.exp.first < stage1_ntt_min_exponent))
            throw std::invalid_argument("tune exponent outside supported range");
        const bool edwards = task.execution.selection.curve == CurveModel::Edwards;
        if (grid.naf.first && !edwards) throw std::invalid_argument("naf requires Edwards (--param e)");
        if (grid.batch.first > task.execution.ntt.max_batch)
            throw std::invalid_argument("requested batch exceeds [stage1] ntt_max_batch");
        const auto device = device_info(task.execution.device_index, task.execution.ntt.memory_mib);
        ecm_ts_fprintf(stdout,"ntt_type = convolution:tile_bits:max_radix_bits:graph_bits:"
                               "warp_tail:forward_pass_bits:inverse_pass_bits (0=linear, 1=weighted)\n");
        std::vector<unsigned> windows;
        if (!edwards) windows = {0};
        else if (grid.naf.first) {
            for (unsigned w=grid.naf.first; w<=grid.naf.last; ++w) windows.push_back(w);
        } else windows = {6,8,10,12};

        const std::string empty;
        const double deadline = seconds ? now_s()+seconds : 0;
        bool measured = false;
        TuneNafSamples naf_samples{uint64_t(b1)};
        const auto plan = plan_tune_lengths(task.execution.ntt, grid, edwards, deadline);
        for (const auto& [width, shapes] : plan) {
            const Stage1Request request{nullptr,b1,b1,task.execution.ntt.max_batch,empty,false,empty,task};
            measured |= tune_windows(device,request,width,shapes,grid,naf_samples,windows,
                                     !grid.naf.first && windows.size()>1 && !exhaustive,exhaustive,deadline,true);
        }
        if (!measured) throw BudgetError("no matching Stage1 NTT length, width or batch fits the requested grid");
        return 0;
    } catch (const TimeLimit&) {
        ecm_ts_fprintf(stdout,"stage1_ntt_grid: status=time_limit\n");
        return 0;
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}

Stage1NttOptions select_tuned_plan(const ecm_stage1::Stage1Request& request, unsigned p, unsigned batch,
        const Stage1NttOptions& options, unsigned window, bool require_capacity, BatchMemory& memory) {
    const auto device = device_info(request.options.execution.device_index, options.memory_mib);
    auto candidates = production_shapes(p, options, window != 0);
    const auto capacities = production_capacity(request, device, p, window, candidates);
    const bool any_fits = std::any_of(capacities.begin(), capacities.end(),
        [batch](const BatchMemory& capacity) { return capacity.max_curves >= batch; });
    if (any_fits || (require_capacity && batch > 1)) {
        for (unsigned mode = 0; mode < CandidateCount; ++mode)
            if (capacities[mode].max_curves < batch) candidates[mode].reset();
    }
    if (std::none_of(candidates.begin(), candidates.end(), [](const auto& plan) { return plan.has_value(); }))
        throw BudgetError("no Stage1 NTT plan fits the batch memory budget");
    std::array<std::optional<Cached>, CandidateCount> choices;
    std::array<unsigned,CandidateCount> quality{};
    pt::Profile db;
    try { db = load(options.tune_file); }
    catch (const std::exception& e) { ecm_ts_fprintf(stderr,"Stage1 NTT tune read: %s\n",e.what()); }
    using ecm_stage2::tune::uint;
    for (const auto& row : db.stage1_ntt) {
        if (pt::is_runtime_stage1_ntt(row)) continue;
        const bool current = compatible_workload(row,request,window);
        if (!current && (row.count("sample_target_ms") || window)) continue;
        const auto& condition = db.conditions.at(uint(row,"condition"));
        if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;
        const bool weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
        for (unsigned mode=0; mode<CandidateCount; ++mode) {
            const auto& plan = candidates[mode];
            if (!plan || plan->options.weighted!=weighted) continue;
            auto sample = decode(row,plan->shape,device,batch,plan->options);
            if (!sample) continue;
            if (current && !reusable_dispatch(row,device,p,batch,sample->options,window)) continue;
            const unsigned q = current ? unsigned(sample_phase(row)) : 0;
            if (!choices[mode] || q>quality[mode] || (q==quality[mode] && sample->rate>choices[mode]->rate)) {
                choices[mode]=std::move(sample);
                quality[mode]=q;
            }
        }
    }
    const char* source = "tune";
    if (!options.automatic) {
        for (unsigned mode=0; mode<CandidateCount; ++mode) {
            if (!candidates[mode]) continue;
            memory=capacities[mode];
            return choices[mode] ? choices[mode]->options : options;
        }
    }
    TuneNafSamples naf_samples{uint64_t(request.b1)};
    TuneMatrix matrix(device,request,p,window,db,naf_samples);
    for (unsigned mode=0; mode<CandidateCount; ++mode) {
        if (!options.runtime_enabled || !candidates[mode] || quality[mode]) continue;
        const auto plan=choices[mode] ? choices[mode]->options : candidates[mode]->options;
        choices[mode].reset();
        auto row=matrix.sample(plan,batch,SamplePhase::Coarse);
        if (!row.rate) continue;
        choices[mode]=Cached{fixed_plan(plan,row.plan),row.rate};
        quality[mode]=unsigned(SamplePhase::Coarse);
        source="short_tune";
    }
    try { matrix.flush(); }
    catch (const std::exception& e) { ecm_ts_fprintf(stderr,"Stage1 NTT tune write: %s\n",e.what()); }
    unsigned winner_mode = CandidateCount;
    for (unsigned mode = 0; mode < CandidateCount; ++mode)
        if (choices[mode] && (winner_mode == CandidateCount || quality[mode] > quality[winner_mode] ||
            (quality[mode] == quality[winner_mode] && choices[mode]->rate > choices[winner_mode]->rate)))
            winner_mode = mode;
    if (winner_mode == CandidateCount && !options.runtime_enabled) {
        for (auto mode : {Mixed, Weighted, Linear}) {
            if (!candidates[mode]) continue;
            memory = capacities[mode];
            return candidates[mode]->options;
        }
    }
    if (winner_mode == CandidateCount) throw BudgetError("no safe Stage1 NTT auto plan under the current VRAM budget");
    const auto& winner = *choices[winner_mode];
    memory = capacities[winner_mode];
    ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=%s p=%u batch=%u convolution=%s length=%u\n",
                   source, p, batch, modes[winner.options.weighted], winner.options.length);
    return winner.options;
}

unsigned production_window(const ecm_stage1::Stage1Request& request, unsigned p, bool fixed_batch) {
    using namespace ecm_stage1;
    using ecm_stage2::tune::uint;
    const auto& exec = request.options.execution;
    const unsigned configured = naf_window(request);
    if (exec.ntt.tune_file.empty() || request.b1 < 2) return configured;
    const bool edwards = exec.selection.curve == CurveModel::Edwards;
    const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
    const unsigned limit = std::min(request.curves, exec.ntt.max_batch);
    auto best_window = [&](const pt::Profile& db, bool require_fine) -> std::optional<unsigned> {
        std::optional<unsigned> best;
        std::pair<bool,double> fastest{};
        std::map<unsigned,std::array<BatchMemory,CandidateCount>> capacities;
        std::map<unsigned,Candidates> shapes;
        for (const auto& row : db.stage1_ntt) {
            const bool runtime = pt::is_runtime_stage1_ntt(row);
            if (!runtime && !row.count("sample_target_ms")) continue;
            const auto w = unsigned(uint(row,"naf_window")), batch = unsigned(uint(row,"batch"));
            if ((configured && w != configured) || (!edwards && w) || (edwards && !w && !configured) ||
                batch > limit || (fixed_batch && batch != limit)) continue;
            if (runtime) {
                if (pt::text(row,"curve") != curve_name(request) || uint(row,"b1") != uint64_t(request.b1) ||
                    uint(row,"arithmetic_bits") != p || uint(row,"valid_curves") != batch ||
                    uint(row,"lcm_multiplier") != lcm_multiplier(request) ||
                    pt::optional_uint(row,"target_bits",p) != mpz_sizeinbase(request.modulus,2)) continue;
            } else if (!compatible_workload(row,request,w)) continue;
            const auto& condition = db.conditions.at(uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;
            if (!runtime && (uint(row,"reference_bits")-1)/4096 != (p-1)/4096) continue;
            const bool fine = runtime || sample_phase(row) == SamplePhase::Fine;
            const auto score = std::make_pair(fine,ecm_stage2::tune::real(row,"iter_per_s"));
            if ((require_fine && !fine) || score <= fastest) continue;
            if (!shapes.count(w)) {
                shapes[w] = production_shapes(p,exec.ntt,w != 0);
                capacities[w] = production_capacity(request,device,p,w,shapes.at(w));
            }
            for (unsigned mode=0; mode<CandidateCount; ++mode) {
                const auto& candidate = shapes.at(w)[mode];
                if (!candidate || batch > capacities.at(w)[mode].max_curves ||
                    candidate->options.weighted != (pt::text(row,"ntt_type") == pt::weighted_ntt_type)) continue;
                auto decoded = decode(row,candidate->shape,device,batch,candidate->options);
                if (!decoded || (!runtime && !reusable_dispatch(row,device,p,batch,decoded->options,w))) continue;
                fastest = score;
                best = w;
            }
        }
        return best;
    };
    pt::Profile db;
    try { db = load(exec.ntt.tune_file); }
    catch (const std::exception& e) {
        ecm_ts_fprintf(stderr,"Stage1 NTT tune read: %s\n",e.what());
        return configured;
    }
    auto best = best_window(db,!configured && exec.ntt.runtime_enabled);
    if (!best && exec.ntt.runtime_enabled) {
        TuneNafSamples naf_samples{uint64_t(request.b1)};
        NttTuneGrid grid;
        grid.batch = {fixed_batch ? limit : 1u, limit};
        grid.batch_explicit = fixed_batch;
        std::vector<unsigned> windows{configured};
        if (edwards && !configured) windows = {6,8,10,12};
        ecm_ts_fprintf(stdout,"stage1_ntt_runtime_tune: p=%u batch=%u-%u\n",p,grid.batch.first,limit);
        try {
            tune_windows(device,request,p,tune_shapes(p,exec.ntt,edwards),grid,naf_samples,windows,
                         windows.size()>1,false,0);
        } catch (const Cancelled&) { return configured; }
        best = best_window(load(exec.ntt.tune_file),false);
    }
    const unsigned selected = best.value_or(configured);
    if (edwards && !configured)
        ecm_ts_fprintf(stdout,"stage1_ntt_naf_selection: p=%u naf=%u source=%s\n",p,selected,best ? "tune" : "default");
    return selected;
}

bool production_batch_fits(const ecm_stage1::Stage1Request& request, unsigned p, unsigned window) {
    const auto& exec = request.options.execution;
    const auto shapes = tune_shapes(p, exec.ntt, window != 0);
    const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
    const auto capacities = production_capacity(request, device, p, window, shapes, false);
    return std::any_of(capacities.begin(), capacities.end(),
        [&](const BatchMemory& capacity) { return capacity.max_curves >= request.curves; });
}

unsigned select_production_batch(const ecm_stage1::Stage1Request& request, unsigned remaining) {
    if (!remaining) return 0;
    const auto& exec = request.options.execution;
    const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
    const auto p = ecm_stage1::ntt_arithmetic_bits(request.modulus, request.options);
    const auto target_bits = mpz_sizeinbase(request.modulus, 2);
    const auto window = naf_window(request);
    const auto shapes = production_shapes(p, exec.ntt, window != 0);
    if (std::none_of(shapes.begin(), shapes.end(), [](const auto& plan) { return plan.has_value(); }))
        throw std::invalid_argument("no legal Stage1 NTT packing for the requested parameters");
    const auto capacities = production_capacity(request, device, p, window, shapes);
    const auto linear_cap = capacities[Linear].max_curves;
    const auto weighted_cap = std::max(capacities[Weighted].max_curves, capacities[Mixed].max_curves);
    const auto cap = std::min(remaining, std::max({1u, linear_cap, weighted_cap}));
    std::map<unsigned, BatchRate> production, synthetic, refined;
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
            const bool modern = row.count("sample_target_ms") != 0;
            if (modern && !compatible_workload(row,request,window)) continue;
            if (!modern && window && !pt::is_runtime_stage1_ntt(row)) continue;
            bool compatible = false;
            for (unsigned mode = 0; mode < CandidateCount && !compatible; ++mode) {
                const auto& shape = shapes[mode];
                if (!shape || batch>capacities[mode].max_curves || shape->options.weighted!=weighted) continue;
                auto decoded=decode(row,shape->shape,device,batch,shape->options);
                compatible=decoded && (!modern || reusable_dispatch(row,device,p,batch,decoded->options,window));
            }
            if (!compatible) continue;
            const bool runtime = pt::is_runtime_stage1_ntt(row);
            if (runtime && (uint(row,"valid_curves") != batch || uint(row,"arithmetic_bits") != p ||
                            pt::optional_uint(row,"target_bits",p) != target_bits ||
                            uint(row,"b1") != uint64_t(std::floor(request.b1 + 0.0001)) ||
                            uint(row,"lcm_multiplier") != lcm_multiplier(request) ||
                            uint(row,"naf_window") != window ||
                            pt::text(row,"curve") != curve_name(request)))
                continue;
            auto& value = (runtime ? production : modern && sample_phase(row)==SamplePhase::Fine ? refined : synthetic)[batch];
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
    const unsigned tuned = recommended_batch(refined.empty() ? synthetic : refined);
    const unsigned seed = tuned ? tuned : std::min(cap, 64u);
    unsigned selected = recommended_batch(production, 2);
    if (!selected) selected = recommended_batch(production);
    const char* source = selected ? "production" : tuned ? "tune" : "default";
    if (!selected) selected = seed;
    // Try capacity first while there is enough work for a full batch and a tail.
    if (exec.ntt.runtime_enabled && !exec.ntt.tune_file.empty() && remaining > cap) {
        const std::array<unsigned, 4> candidates{cap, seed, std::max(1u, seed / 2),
                                               std::min(cap, seed + seed / 2)};
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
    ecm_ts_fprintf(stdout, "stage1_ntt_batch_selection: source=%s p=%u remaining=%u batch=%u capacity=%u "
                   "linear_capacity=%u weighted_capacity=%u\n",
                   source, p, remaining, selected, cap, linear_cap, weighted_cap);
    return selected;
}

void record_production_sample(const ecm_stage1::Stage1Request& request, const Plan& plan, const Run& run,
                              unsigned valid, unsigned batch, double compute_seconds) {
    const auto& exec = request.options.execution;
    if (!exec.ntt.runtime_enabled || !exec.ntt.observe_batch || exec.ntt.tune_file.empty() || !valid || !run.completed_bits || run.device_ms.empty() ||
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
        row["arithmetic_bits"] = std::to_string(ecm_stage1::ntt_arithmetic_bits(request.modulus, request.options));
        row["target_bits"] = std::to_string(mpz_sizeinbase(request.modulus, 2));
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
