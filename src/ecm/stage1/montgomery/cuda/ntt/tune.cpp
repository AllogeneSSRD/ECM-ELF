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
    double time() const {
        auto ordered = ms;
        std::sort(ordered.begin(), ordered.end());
        const auto middle = ordered.size() / 2;
        return ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
    }
    double throughput() const { return 1000.0 * batch / time(); }
};

unsigned max_exponent(const Plan& plan) {
    constexpr unsigned supported_max = ecm_stage1::stage1_ntt_max_exponent;
    if (plan.weighted) return std::min(supported_max, plan.length * plan.bpw);
    unsigned lo = 0, hi = (plan.length - 1) / 2;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (exact_ok_terms(mid, plan.bpw)) lo = mid;
        else hi = mid - 1;
    }
    return std::min(supported_max, lo * plan.bpw);
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
struct TimeLimit {};

class Calibration {
    const Device& device_;
    unsigned p_, bits_;
    std::vector<uint32_t> scalar_;
    unsigned window_ = 0;
    size_t dictionary_limit_ = 0;
    std::vector<int16_t> digits_;
    std::vector<uint32_t> dictionary_;
    unsigned reference_batch_ = 0;
    std::vector<uint32_t> reference_;
public:
    Calibration(const Device& device, unsigned p, unsigned bits)
        : device_(device), p_(p), bits_(bits), scalar_((bits + 31) / 32) {
        std::mt19937 random(0x67ae128bu);
        for (auto& word : scalar_) word = random();
    }

    void native(unsigned window, uint64_t b1, size_t dictionary_limit) {
        window_ = window;
        dictionary_limit_ = dictionary_limit;
        ecm_edwards::Integer k;
        if (!ecm_build_lcm_exponent(k.v, b1, 48)) throw std::runtime_error("cannot build tune scalar");
        EcmEdwardsNaf encoding(k.v, window);
        bits_ = std::min(bits_, encoding.bits);
        const unsigned end = (encoding.bits + uint64_t(bits_)) / 2;
        for (unsigned i = end; i > end - bits_; --i) digits_.push_back(int16_t(encoding.digit(i - 1)));
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
            for (unsigned lane = 0; lane < batch; ++lane)
                active[stride + lane * words] = active[2 * stride + lane * words] = 1;
            const size_t slots = dictionary_.size() / words;
            std::vector<uint32_t> table(slots * stride);
            for (size_t i = 0; i < slots; ++i)
                for (unsigned lane = 0; lane < batch; ++lane)
                    std::copy_n(dictionary_.data() + i * words, words, table.data() + i * stride + lane * words);
            EdwardsNaf naf{window_, digits_, table, dictionary_limit_};
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

std::array<BatchMemory, CandidateCount> production_capacity(const ecm_stage1::Stage1Request& request,
        const Device& device, unsigned p, unsigned window, const Candidates& shapes) {
    if (window && (window < 2 || window > 12))
        throw std::invalid_argument("Edwards NTT NAF requires a window from 2 to 12");
    if (!std::isfinite(request.b1) || request.b1 < 0 || request.b1 > 5000000000.0)
        throw std::invalid_argument("invalid Stage1 NTT B1");
    const size_t words = (p + 31) / 32, entries = window ? 1u << (window - 2) : 0;
    const auto dictionary_mb = request.options.execution.naf_memory_mb;
    const auto dictionary_budget = size_t(dictionary_mb) * ecm_edwards_memory::mb;
    size_t host_cap = max_batch;
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
        const auto dictionary = entries * (32 * length + 16 * words);
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

bool same_workload(const Fields& row, const ecm_stage1::Stage1Request& request, unsigned window) {
    using ecm_stage2::tune::uint;
    return row.count("sample_target_ms") && pt::text(row,"program_version") == ecm_program::version &&
           pt::text(row,"curve") == curve_name(request) && uint(row,"naf_window") == window &&
           (!window || uint(row,"b1") == uint64_t(request.b1));
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

class TuneMatrix {
    const Device& device_;
    const ecm_stage1::Stage1Request& request_;
    unsigned p_, window_;
    double deadline_;
    std::optional<Calibration> calibration_;
    std::vector<Fields> history_, legacy_, pending_;
    std::set<size_t> hits_;
public:
    unsigned measured = 0;
    TuneMatrix(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p,
               unsigned window, const pt::Profile& db, double deadline = 0)
        : device_(device), request_(request), p_(p), window_(window), deadline_(deadline) {
        for (const auto& row : db.stage1_ntt) {
            const auto& condition = db.conditions.at(ecm_stage2::tune::uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid ||
                pt::is_runtime_stage1_ntt(row)) continue;
            if (same_workload(row, request, window)) history_.push_back(row);
            else if (!row.count("sample_target_ms")) legacy_.push_back(row);
        }
    }
    size_t reused() const { return hits_.size(); }
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
            if (b <= 1 || b >= cap || uint(row,"sample_target_ms") != 20 ||
                uint(row,"length") != options.length || uint(row,"peak_bytes") > device_.budget ||
                (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != options.weighted) continue;
            if (!reusable_dispatch(row,device_,p_,b,options,window_)) continue;
            if (b == fallback) return b;
            prior[b] = {ecm_stage2::tune::real(row,"iter_per_s"),1};
        }
        const unsigned previous = recommended_batch(prior);
        return previous ? previous : fallback;
    }
    void flush() {
        persist(request_.options.execution.ntt.tune_file, device_, pending_);
        pending_.clear();
    }
    Score sample(const Stage1NttOptions& options, unsigned batch, unsigned target) {
        if (ecm_stage1_stop_requested()) throw Cancelled();
        try {
            auto plan = describe(device_, p_, batch, options, window_);
            using ecm_stage2::tune::uint;
            const auto lookup = [&](const std::vector<Fields>& rows, bool previous) -> Score {
                for (size_t i=0; i<rows.size(); ++i) {
                    const auto& row = rows[i];
                    if (uint(row,"length") != plan.length || uint(row,"bpw") != plan.bpw ||
                        (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != plan.weighted ||
                        uint(row,"sample_target_ms") != target || uint(row,"batch") != batch ||
                        uint(row,"peak_bytes") > device_.budget || !same_dispatch(row,p_,plan)) continue;
                    if (previous && hits_.insert(i).second)
                        ecm_ts_fprintf(stdout,"stage1_ntt_reuse: length=%u batch=%u dataflow=%u tile=%u "
                            "outer=%u graph=%u target_ms=%u iter_per_s=%.0f\n",plan.length,batch,plan.dataflow,
                            plan.tile,plan.outer_max,plan.window,target,ecm_stage2::tune::real(row,"iter_per_s"));
                    return {batch, plan, ecm_stage2::tune::real(row,"iter_per_s"),
                        1000 * ecm_stage2::tune::real(row,"stddev_seconds") / ecm_stage2::tune::real(row,"median_ms")};
                }
                return {};
            };
            if (auto hit = lookup(pending_, false); hit.rate) return hit;
            if (auto hit = lookup(history_, true); hit.rate) return hit;
            if (deadline_ && now_s() >= deadline_) throw TimeLimit{};
            if (!calibration_) {
                calibration_.emplace(device_, p_, 64);
                if (window_) calibration_->native(window_, uint64_t(request_.b1),
                    size_t(request_.options.execution.naf_memory_mb) * ecm_edwards_memory::mb);
            }
            const auto m = calibration_->measure(batch, options, target == 20 ? 3 : 5, target);
            auto row = encode(m, m.steps);
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
            pending_.push_back(std::move(row));
            ++measured;
            print_measurement(m, target);
            return {batch, m.plan, m.throughput() * m.steps, pt::deviation(m.ms) / m.time()};
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
    auto best = matrix.sample(f.options,batch,50);
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
    if (!o.profile && o.window<0) {
        auto alt=o; alt.window=base.window==16 ? 32 : 16; trials.push_back(alt);
    }
    for (const auto& alt : trials) {
        auto score = matrix.sample(alt,batch,50);
        if (score.rate>best.rate) best = score;
    }
    f.keep(best);
    if (best.rate>finals[batch].rate) finals[batch]=best;
}

void tune_width(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p, bool exhaustive, double deadline) {
    using Layout = Stage1NttOptions::MixedLayout;
    const auto& options = request.options.execution.ntt;
    const auto w = naf_window(request);
    const auto shapes = production_shapes(p, options, w != 0);
    const auto caps = production_capacity(request,device,p,w,shapes);
    std::vector<Family> families;
    for (unsigned mode=0; mode<CandidateCount; ++mode) {
        if (!shapes[mode]) continue;
        const auto& shape = *shapes[mode];
        const unsigned cap = caps[mode].max_curves;
        ecm_ts_fprintf(stdout,"stage1_ntt_plan: p=%u length=%u bpw=%u naf=%u capacity=%u%s\n",
            p,shape.shape.length,shape.shape.bpw,w,cap,cap ? "" : " reason=memory_budget");
        if (!cap) continue;
        if (mode == Mixed) {
            for (auto layout : {Layout::Twiddle,Layout::Pfa,Layout::Coalesced}) {
                auto o = shape.options;
                o.mixed_layout = layout;
                families.push_back({o, cap});
            }
        } else families.push_back({shape.options, cap});
    }
    TuneMatrix matrix(device, request, p, w, load(options.tune_file), deadline);
    const auto start = now_s();
    try {
        double single = 0, peak = 0;
        for (auto& f : families) {
            // If the estimated upper endpoint fails actual admission, bisect
            // against a measured lower endpoint before ranking the family.
            for (auto b : std::set<unsigned>{1, matrix.interior(f.options,f.cap), f.cap}) {
                auto score = matrix.sample(f.options,b,20);
                f.keep(score);
                if (!score.rate) {
                    unsigned lo = f.best.empty() ? 0 : f.best.rbegin()->first, hi = b;
                    while (hi > lo + 1) {
                        const unsigned mid = lo + (hi-lo)/2;
                        auto test = matrix.sample(f.options,mid,20);
                        f.keep(test);
                        if (test.rate) lo = mid;
                        else hi = mid;
                    }
                    f.cap = lo;
                    break;
                }
            }
            single = std::max(single,f.single.rate);
            peak = std::max(peak,f.peak.rate);
        }
        std::map<unsigned,Score> finals;
        std::map<unsigned,std::vector<Family*>> rounds;
        for (auto& f : families) {
            if (!f.cap || !f.peak.rate) continue;
            const auto close = [](const Score& s, double top) {
                return s.rate >= top * (1 - std::clamp(2*s.relative_sd,0.05,0.20));
            };
            if (!exhaustive && !close(f.single,single) && !close(f.peak,peak)) {
                f.best.clear();
                continue;
            }
            std::set<unsigned> grid{1,f.peak.batch,f.cap};
            for (unsigned b=2; b<=f.cap; b=exhaustive ? b+1 : 2*b) grid.insert(b);
            for (auto batch : grid) rounds[batch].push_back(&f);
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
                    if (b && b<=f.cap) refine_batch(matrix,f,b,finals);
                unsigned low=0, high=f.peak.batch;
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
        if (finals.empty()) throw BudgetError("no safe NTT backend for current memory budget");
        const auto best = std::max_element(finals.begin(),finals.end(),[](const auto& a,const auto& b){
            return a.second.rate<b.second.rate;
        });
        const auto economical = std::find_if(finals.begin(),finals.end(),[&](const auto& item){
            return item.second.rate>=near_peak*best->second.rate;
        });
        const auto report = [](const char* label,const Score& s){
            ecm_ts_fprintf(stdout,"%s %u %u %s %uc %.0f iter/s dataflow=%u\n",label,s.plan.length,
                max_exponent(s.plan),tune_shape_descriptor(s.plan).c_str(),s.batch,s.rate,s.plan.dataflow);
        };
        if (finals.count(1)) report("Single",finals.at(1));
        report("Best",best->second);
        report("95%",economical->second);
    } catch (const TimeLimit&) {
        matrix.flush();
        ecm_ts_fprintf(stdout,"stage1_ntt_tune: p=%u measured=%u reused=%zu status=time_limit\n",
                       p,matrix.measured,matrix.reused());
        throw;
    } catch (...) {
        matrix.flush();
        throw;
    }
    matrix.flush();
    ecm_ts_fprintf(stdout,"stage1_ntt_tune: p=%u measured=%u reused=%zu seconds=%.3f\n",
                   p,matrix.measured,matrix.reused(),now_s()-start);
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

int tune_command(const ecm_stage1::Stage1Options& task, double b1, unsigned p, bool exhaustive, unsigned seconds) {
    try {
        if (!(b1 >= 2 && b1 <= 5000000000.0)) throw std::invalid_argument("invalid tune B1");
        const auto device = device_info(task.execution.device_index, task.execution.ntt.memory_mib);
        ecm_ts_fprintf(stdout,"ntt_type = convolution:tile_bits:max_radix_bits:graph_bits:"
                               "warp_tail:forward_pass_bits:inverse_pass_bits (0=linear, 1=weighted)\n"
                               "length max_exp ntt_type batch iter/s\n");
        std::vector<unsigned> widths{61,127,251,521,1279,2539,4003,8191,16001,32749,
            58631,65521,115201,131071,225287,262139,524309,900001,1048573};
        if (p) widths = {p};
        const std::string empty;
        const double deadline = seconds ? now_s()+seconds : 0;
        const ecm_stage1::Stage1Request request{nullptr,b1,b1,max_batch,empty,false,empty,task};
        for (unsigned width : widths) {
            if (width < ecm_stage1::stage1_ntt_min_exponent || width > ecm_stage1::stage1_ntt_max_exponent)
                throw std::invalid_argument("--tune ntt --bits outside supported exponent range");
            tune_width(device,request,width,exhaustive,deadline);
        }
        return 0;
    } catch (const TimeLimit&) {
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
        const bool current = same_workload(row,request,window);
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
            const unsigned q = current ? unsigned(uint(row,"sample_target_ms")) : 0;
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
    TuneMatrix matrix(device,request,p,window,db);
    for (unsigned mode=0; mode<CandidateCount; ++mode) {
        if (!candidates[mode] || quality[mode]) continue;
        const auto plan=choices[mode] ? choices[mode]->options : candidates[mode]->options;
        choices[mode].reset();
        auto row=matrix.sample(plan,batch,20);
        if (!row.rate) continue;
        choices[mode]=Cached{fixed_plan(plan,row.plan),row.rate};
        quality[mode]=20;
        source="short_tune";
    }
    try { matrix.flush(); }
    catch (const std::exception& e) { ecm_ts_fprintf(stderr,"Stage1 NTT tune write: %s\n",e.what()); }
    unsigned winner_mode = CandidateCount;
    for (unsigned mode = 0; mode < CandidateCount; ++mode)
        if (choices[mode] && (winner_mode == CandidateCount || quality[mode] > quality[winner_mode] ||
            (quality[mode] == quality[winner_mode] && choices[mode]->rate > choices[winner_mode]->rate)))
            winner_mode = mode;
    if (winner_mode == CandidateCount) throw BudgetError("no safe Stage1 NTT auto plan under the current VRAM budget");
    const auto& winner = *choices[winner_mode];
    memory = capacities[winner_mode];
    ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=%s p=%u batch=%u convolution=%s length=%u\n",
                   source, p, batch, modes[winner.options.weighted], winner.options.length);
    return winner.options;
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
            if (modern && !same_workload(row,request,window)) continue;
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
            auto& value = (runtime ? production : modern && uint(row,"sample_target_ms")>=50 ? refined : synthetic)[batch];
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
    if (!exec.ntt.tune_file.empty() && remaining > cap) {
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
