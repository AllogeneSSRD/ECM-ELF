#include "tune.h"
#include "batch_timing.h"
#include "ecm/stage1/ntt_target.h"
#include "ladder.h"
#include "ecm/stage1/scalar.h"
#include "../../edwards/ntt_dictionary.h"
#include <set>
#include "ecm/stage1/ntt_sizes.h"
#include "ecm/stage1/ntt_shape_text.h"
#include "../../../cuda/ntt/runtime.h"
#include "../../edwards/memory.h"
#include "../../../../io/files.h"
#include "../../../stage2/planning/tune_portable.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/stop.h"
#include <algorithm>
#include <cstdio>
#include <array>
#include <climits>
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
    size_t owned = 0, observed = 0, budget = 0, naf_memory = 0;
    unsigned steps = 0, blocks = 1;
    unsigned naf_columns = 0, sample_nonzero = 0, pool_steps = 0;
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

ecm_stage1::NttPfaCode pfa_code(const Plan& plan) {
    using Code = ecm_stage1::NttPfaCode;
    if (plan.dataflow & Radix5ForwardOuter) return Code::FusedOuter;
    if (plan.dataflow & CoalescedRadix5) return Code::Coalesced;
    return plan.dataflow & PfaRadix5 ? Code::Direct : Code::None;
}

void print_measurement(const Measurement& m, unsigned target) {
    const auto& p = m.plan;
    const auto shape = tune_shape_descriptor(p);
    ecm_ts_fprintf(stdout, "%u %u %s %uc %.0f iter/s %u %u %.3f/%u ms %u %u\n",
        p.length, p.max_exp, shape.c_str(), m.batch, m.throughput()*m.steps, p.window, p.dataflow,
        m.group_ms(), target, m.steps, m.steps / m.blocks);
}

Stage1NttOptions fixed_plan(Stage1NttOptions o, const Plan& plan) {
    if (ecm_stage1::stage1_ntt_radix5(plan.length)) {
        using Layout = Stage1NttOptions::MixedLayout;
        o.mixed_layout = plan.dataflow & CoalescedRadix5 ? Layout::Coalesced :
                         plan.dataflow & PfaRadix5 ? Layout::Pfa : Layout::Twiddle;
        o.fused_radix5_inverse = (plan.dataflow & Radix5InverseLimbSums) != 0;
        o.fused_radix5_forward = (plan.dataflow & Radix5ForwardOuter) != 0;
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
        {"max_exp", std::to_string(plan.max_exp)},
        {"iter_per_s", pt::numeric(m.throughput() * iterations)}, {"batch", std::to_string(m.batch)},
        {"ntt_type", pt::quoted(plan.weighted ? pt::weighted_ntt_type : pt::linear_ntt_type)},
        {"bpw", std::to_string(plan.bpw)}, {"tile", std::to_string(plan.tile)},
        {"max_radix_bits", std::to_string(plan.outer_max)}, {"passes_forward", std::to_string(plan.passes)},
        {"forward", pt::quoted(plan.forward)}, {"inverse", pt::quoted(plan.inverse)},
        {"graph_bits", std::to_string(plan.window)}, {"warp_tail", std::to_string(plan.warp_tail)},
        {"repeats", std::to_string(m.ms.size())}, {"ntt_memory_bytes", std::to_string(m.owned)},
        {"naf_memory_bytes", std::to_string(m.naf_memory)},
        {"peak_bytes", std::to_string(std::max(m.owned, m.observed))},
        {"budget_bytes", std::to_string(m.budget)}
    };
    pt::stamp_provenance(row);
    return row;
}

std::vector<Fields> compact_batch_rows(std::vector<Fields> rows, const std::string& search, unsigned selected_batch = 0);
bool fits_recorded_memory(const Fields&, const Device&, const ecm_stage1::Stage1Request&);

void persist(const std::string& path, const Device& device, std::vector<Fields> rows,
             const std::string& replace_scope = {}, bool partial = false, bool observation = false,
             const std::string& forced_search = {}, const ecm_stage1::Stage1Request* request = nullptr, unsigned selected_batch = 0) {
    if (path.empty() || rows.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    ecm_io::FileLock lock(path);
    auto db = load(path);
    const auto condition = std::to_string(db.condition({{"uuid_hex", pt::quoted(device.uuid)}}));
    for (auto& row : rows) row["condition"] = condition;
    if (!replace_scope.empty()) {
        const auto belongs = [&](const Fields& r) {
            return r.at("condition") == condition && r.count("batch_scope") &&
                   pt::text(r,"batch_scope") == replace_scope;
        };
        // Force searches run cold. Compare with the latest completed search only
        // at publication, so a slower rerun cannot demote its best batch.
        if (!forced_search.empty()) {
            pt::Profile merged;
            merged.conditions = db.conditions;
            for (const auto& old : db.stage1_ntt)
                if (belongs(old) && old.count("batch_search") && pt::text(old,"batch_search")==forced_search &&
                    fits_recorded_memory(old,device,*request))
                    merged.update_stage1_ntt(old);
            for (auto& row : rows) merged.update_stage1_ntt(std::move(row));
            rows = compact_batch_rows(std::move(merged.stage1_ntt),forced_search,selected_batch);
        }
        // A shorter interrupted run must not evict a completed search.
        if (partial && std::any_of(db.stage1_ntt.begin(),db.stage1_ntt.end(),
                [&](const Fields& r) { return belongs(r) && r.count("batch_search"); })) return;
        // An unselected batch normally stays out of a compact search. Keep
        // measuring published decisions, and retain new memory/rate tradeoffs.
        const auto less=[](const Fields& a,const Fields& b) {
            return pt::matrix_row_less(a,b,pt::ntt_row_order);
        };
        pt::Profile alternatives;
        alternatives.conditions = db.conditions;
        for (const auto& row : rows)
            if (!row.count("batch_scope")) alternatives.update_stage1_ntt(row);
        std::map<Fields,unsigned,decltype(less)> counts(less);
        for (const auto& row : alternatives.stage1_ntt) ++counts[row];
        for (const auto& old : db.stage1_ntt) {
            const auto it = counts.find(old);
            if (it != counts.end() && (!belongs(old) || old.count("batch_search") || pt::optional_uint(old,"explicit_batch")))
                ++it->second;
        }
        rows.erase(std::remove_if(rows.begin(),rows.end(),[&](const Fields& row) {
            return !row.count("batch_scope") && counts.at(row) < 2;
        }),rows.end());
        // Superseded search roles expire, but their measurements can still
        // be the best choice under another memory budget.
        db.stage1_ntt.erase(std::remove_if(db.stage1_ntt.begin(),db.stage1_ntt.end(),[&](const Fields& old) {
            return belongs(old) && !old.count("batch_search") && !pt::optional_uint(old,"explicit_batch");
        }),db.stage1_ntt.end());
        for (auto& old : db.stage1_ntt)
            if (belongs(old))
                for (const char* key : {"batch_scope","batch_search","batch_roles"}) old.erase(key);
    }
    for (auto& row : rows) db.update_stage1_ntt(std::move(row),observation);
    ecm_io::write_atomic(path, db.serialize());
}

struct Cancelled : std::runtime_error { Cancelled() : std::runtime_error("Stage1 NTT tune stopped") {} };
struct TimeLimit {};

struct NafSample {
    unsigned columns = 0, nonzero = 0;
    std::vector<int16_t> digits;
    std::vector<TuneBlock> blocks;
};

// Use contiguous windows of real digits. Balance the cumulative number of adds
// against the full pool, then prefer the window closest to each band's centre.
void prepare_blocks(NafSample& sample) {
    const unsigned size = unsigned(sample.digits.size());
    const unsigned seam = sample.columns > size ? ecm_stage1::naf_tune_span : size;
    std::vector<unsigned> adds(size + 1);
    for (unsigned i = 0; i < size; ++i) adds[i+1] = adds[i] + (sample.digits[i] != 0);
    const unsigned cycle = sample.nonzero ? std::max(1u, (size + sample.nonzero/2) / sample.nonzero) : 1;
    for (unsigned steps = cycle; steps < size; steps *= 2) {
        unsigned bands = 1;
        while (bands < ecm_stage1::naf_tune_strata && steps / bands >= 4 * cycle) bands *= 2;
        const unsigned span = steps / bands;
        if (span > seam) continue;
        TuneBlock block{steps, span, 0, {}};
        block.digits.reserve(steps);
        for (unsigned band = 0; band < bands; ++band) {
            const unsigned lo = band*size/bands, hi = (band+1)*size/bands;
            const int wanted = int(std::lround(double((band+1)*span)*sample.nonzero/size));
            std::pair<int,int> best{INT_MAX, INT_MAX};
            unsigned start = lo;
            for (unsigned at = lo; at + span <= hi; ++at) {
                if (at/seam != (at+span-1)/seam) continue;
                const int n = int(block.nonzero + adds[at+span] - adds[at]);
                const auto score = std::make_pair(std::abs(n-wanted), std::abs(int(2*at+span)-int(lo+hi)));
                if (score < best) { best = score; start = at; }
            }
            if (best.first == INT_MAX) break;
            block.digits.insert(block.digits.end(), sample.digits.begin()+start, sample.digits.begin()+start+span);
            block.nonzero += adds[start+span] - adds[start];
        }
        if (block.digits.size() == steps && (!sample.nonzero || block.nonzero))
            sample.blocks.push_back(std::move(block));
    }
    sample.blocks.push_back({size, seam, sample.nonzero, {}});
}

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
        prepare_blocks(sample);
        ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_naf_sample: b1=%llu naf=%u columns=%u sampled=%u nonzero=%u\n",
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
    std::vector<uint32_t> projective_unit_;
    std::vector<TuneBlock> binary_blocks_;
    // One compact endpoint per coordinate and lane, keyed by the replayed block.
    std::map<unsigned, std::vector<std::array<size_t,4>>> references_;
public:
    Calibration(const Device& device, unsigned p, unsigned bits)
        : device_(device), p_(p), bits_(bits), scalar_((bits + 31) / 32) {
        std::mt19937 random(0x67ae128bu);
        for (auto& word : scalar_) word = random();
        for (unsigned steps = 4; steps < bits; steps *= 2) binary_blocks_.push_back({steps, 0, 0, {}});
        binary_blocks_.push_back({bits, 0, 0, {}});
    }

    void native(unsigned window, const NafSample& sample, size_t dictionary_limit,
                const Stage1NttOptions& options) {
        window_ = window;
        dictionary_limit_ = dictionary_limit;
        naf_sample_ = &sample;
        const unsigned words = (p_ + 31) / 32;
        bool use_gpu = ecm_edwards::ntt::prefer_device_dictionary(p_,1,window,options.naf_gpu_min_bits);
        const bool report_setup = use_gpu || options.naf_gpu_min_bits || p_ >= (1u << 20);
        const double started = now_s();
        if (report_setup)
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_dictionary: p=%u naf=%u entries=%u preparing\n",
                           p_,window,1u << (window-2));
        dictionary_.resize(size_t(4u << (window - 2)) * words);
        ecm_edwards::Integer n, d, x, y, a24, factor;
        mpz_setbit(n.v, p_);
        mpz_sub_ui(n.v, n.v, 1);
        // Represent the neutral point as (0:u:u). Dense coordinates exercise
        // normal reduction costs from the first doubling of a short sample.
        std::mt19937 random(0x1f83d9abu);
        projective_unit_.resize(words);
        do {
            std::generate(projective_unit_.begin(), projective_unit_.end(), [&]{ return random(); });
            if (p_ % 32) projective_unit_.back() &= (1u << (p_ % 32)) - 1;
            mpz_import(y.v, words, -1, sizeof(uint32_t), 0, 0, projective_unit_.data());
            mpz_gcd(factor.v, y.v, n.v);
        } while (mpz_cmp_ui(factor.v, 1) != 0);
        ecm_edwards::MersenneConstruction context(n.v);
        const auto build = [&] {
            if (use_gpu) {
                std::vector<uint32_t> seed(3ull * words);
                auto* dst = seed.data();
                for (mpz_srcptr coordinate : {x.v,y.v,d.v}) {
                    mpz_export(dst,nullptr,-1,sizeof(uint32_t),0,0,coordinate);
                    dst += words;
                }
                EdwardsNaf naf{window,sample.digits,seed,dictionary_limit,true};
                naf.affine_output = &dictionary_;
                try {
                    const auto prep = edwards_naf(device_.index,p_,1,seed,naf,options,Control{});
                    return prep.dictionary_status.at(0) == 0;
                } catch (const BudgetError&) {
                    use_gpu = false;
                }
            }
            return ecm_edwards::ntt::affine_dictionary(dictionary_,1,0,words,window,x.v,y.v,d.v,n.v,factor.v)
                == ecm_edwards::Setup::Ready;
        };
        for (unsigned sigma = 26; sigma < 42; ++sigma) {
            if (ecm_stage1_stop_requested()) throw Cancelled();
            if (ecm_edwards::construct(d.v, x.v, y.v, a24.v, factor.v, sigma, n.v, &context)
                    == ecm_edwards::Setup::Ready && build()) break;
            if (sigma != 41) continue;
            // An order-four point also has a valid affine table over this ring.
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_probe: p=%u using order-four calibration point\n", p_);
            use_gpu = false;
            ecm_edwards::ntt::order_four_calibration_dictionary(dictionary_,words,n.v);
        }
        if (report_setup)
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_dictionary: p=%u naf=%u builder=%s setup_ms=%.3f\n",
                           p_,window,use_gpu ? "gpu" : "host",1000*(now_s()-started));
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
        const auto& blocks = window_ ? naf_sample_->blocks : binary_blocks_;
        control.sample_blocks = &blocks;
        Run run;
        if (window_) {
            m.naf_columns = naf_sample_->columns;
            for (unsigned slot : {1u, 2u})
                for (unsigned lane = 0; lane < batch; ++lane)
                    std::copy(projective_unit_.begin(), projective_unit_.end(),
                              active.begin() + slot*stride + size_t(lane)*words);
            const size_t slots = dictionary_.size() / words;
            std::vector<uint32_t> table(batch > 1 ? slots * stride : 0);
            for (size_t i = 0; batch > 1 && i < slots; ++i)
                for (unsigned lane = 0; lane < batch; ++lane)
                    std::copy_n(dictionary_.data() + i * words, words, table.data() + i * stride + lane * words);
            EdwardsNaf naf{window_, naf_sample_->digits, batch == 1 ? dictionary_ : table, dictionary_limit_};
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
        if (run.stopped) throw Cancelled();
        const auto& block = blocks[run.sample_block_index];
        m.pool_steps = blocks.back().steps;
        m.steps = block.steps * m.blocks;
        m.sample_nonzero = block.nonzero;
        auto& reference = references_[block.steps];
        const auto common = reference.size();
        if (common < batch) reference.resize(batch);
        for (unsigned slot = 0; slot < (window_ ? 3u : 4u); ++slot) {
            for (unsigned lane = 0; lane < batch; ++lane) {
                const auto* coordinate = run.points.data() + slot*stride + size_t(lane)*words;
                const std::string_view bytes(reinterpret_cast<const char*>(coordinate), words*sizeof(*coordinate));
                const auto hash = std::hash<std::string_view>{}(bytes);
                if (lane < common && hash != reference[lane][slot]) throw std::runtime_error("NTT tune endpoint mismatch");
                reference[lane][slot] = hash;
            }
        }
        m.ms = std::move(run.device_ms);
        m.owned = run.owned_bytes;
        m.observed = run.observed_bytes;
        m.budget = run.budget_bytes;
        m.naf_memory = run.dictionary_spectra_bytes + run.dictionary_workspace_bytes;
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
                             unsigned p, unsigned batch, const Stage1NttOptions& base, bool length_hint = false) {
    using ecm_stage2::tune::uint;
    if (uint(row,"batch") != batch || uint(row,"length") != shape.length || (!length_hint && uint(row,"bpw") != shape.bpw)) return {};
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
        o.fused_radix5_inverse = (flow & Radix5InverseLimbSums) != 0;
        o.fused_radix5_forward = (flow & Radix5ForwardOuter) != 0;
        if (o.fused_radix5_inverse && p / shape.length < radix5_limb_min_digit_bits) return {};
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

// One search per table entry. Competing NAF windows and radix-5 layouts
// remain local to this transform, including compaction of the four batch roles.
struct TuneSize {
    unsigned max_exp, reference_bits;
    Candidates shapes;
};

std::vector<TuneSize> plan_tune_lengths(const Stage1NttOptions& options,
        const ecm_stage1::NttTuneGrid& grid, bool native_edwards) {
    using namespace ecm_stage1;
    const bool tight = native_edwards && options.tight_weighted_bound;
    std::vector<TuneSize> plan;
    for (const auto& size : ntt_sizes) {
        if (!grid.length.contains(size.length) || (options.length && options.length != size.length)) continue;
        for (const bool weighted : {false,true}) {
            if (!options.automatic && options.weighted != weighted) continue;
            const unsigned capacity = weighted ? (tight ? size.edwards : size.weighted) : size.linear;
            if (!capacity) continue;
            unsigned limit = capacity;
            if (options.bpw)
                limit = std::min(limit,weighted ? size.length * options.bpw :
                                 linear_exponent_limit(size.length,options.bpw));
            if (limit < stage1_ntt_min_exponent) continue;
            auto fixed = options;
            fixed.length = size.length;
            fixed.weighted = weighted;
            const auto mode = !weighted ? Linear : stage1_ntt_radix5(size.length) ? Mixed : Weighted;
            auto shape = candidate(limit,fixed,mode==Mixed,
                tight ? WeightedBound::UnsignedProducts : WeightedBound::Conservative);
            if (!shape) continue;
            TuneSize entry{capacity,limit,{}};
            entry.shapes[mode] = std::move(shape);
            plan.push_back(std::move(entry));
        }
    }
    std::stable_sort(plan.begin(),plan.end(),[](const TuneSize& a,const TuneSize& b){
        return a.max_exp < b.max_exp;
    });
    if (grid.exp.first && !plan.empty()) {
        if (5 * plan.back().max_exp <= 4 * grid.exp.first) return {};
        const auto first_above = [&](unsigned scaled) {
            const auto it = std::find_if(plan.begin(),plan.end(),[&](const TuneSize& entry){
                return 5 * entry.max_exp > scaled;
            });
            return it == plan.end() ? plan.back().max_exp : it->max_exp;
        };
        const unsigned first = first_above(4 * grid.exp.first), last = first_above(6 * grid.exp.last);
        plan.erase(std::remove_if(plan.begin(),plan.end(),[&](const TuneSize& entry){
            return entry.max_exp < first || entry.max_exp > last;
        }),plan.end());
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
    const unsigned batch_limit = request.options.execution.ntt.max_batch;
    const size_t host_budget = entries ? ecm_edwards_memory::ntt_host_budget() : SIZE_MAX;
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
        size_t cap = std::min<size_t>(batch_limit, device.budget > fixed ? (device.budget - fixed) / per_curve : 0);
        if (dictionary_budget && dictionary)
            cap = std::min(cap, dictionary_budget / dictionary);
        while (entries && cap) {
            const bool gpu = ecm_edwards::ntt::production_device_dictionary(
                p, unsigned(cap), window, request.options.execution.ntt.naf_gpu_min_bits);
            const size_t prep = ecm_edwards::ntt::preparation_bytes(gpu, entries, words, cap);
            if (prep + size_t(request.b1) * 5 <= host_budget) break;
            --cap;
        }
        capacities[mode] = {per_curve, device.device_budget, request.options.execution.ntt.memory_mib,
                            window ? dictionary_mb : 0u, unsigned(cap)};
    }
    return capacities;
}

struct BatchRate {
    double rate = 0;
    unsigned repeats = 0;
};

unsigned recommended_batch(const std::map<unsigned, BatchRate>& rows, unsigned percent = 95) {
    double peak = 0;
    for (const auto& item : rows)
        if (item.second.repeats)
            peak = std::max(peak, item.second.rate);
    if (peak)
        for (const auto& item : rows)
            if (item.second.repeats && item.second.rate >= (percent / 100.0) * peak)
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
           uint(row,"naf_window") == window;
}

bool fits_recorded_memory(const Fields& row, const Device& device, const ecm_stage1::Stage1Request& request) {
    const size_t naf_budget = size_t(request.options.execution.naf_memory_mb) * ecm_edwards_memory::mb;
    return ecm_stage2::tune::uint(row,"ntt_memory_bytes") <= device.budget &&
           (!naf_budget || pt::optional_uint(row,"naf_memory_bytes") <= naf_budget);
}

bool same_workload(const Fields& row, const ecm_stage1::Stage1Request& request, unsigned window) {
    return compatible_workload(row,request,window) && row.count("benchmark_kind") &&
           pt::text(row,"benchmark_kind") == (window ? pt::stratified_naf_kind : pt::stage1_graph_ntt_kind);
}

bool same_dispatch(const Fields& row, const Plan& plan) {
    using ecm_stage2::tune::uint;
    return uint(row,"bpw") == plan.bpw &&
           uint(row,"dataflow") == plan.dataflow &&
           uint(row,"tile") == plan.tile && uint(row,"max_radix_bits") == plan.outer_max &&
           uint(row,"graph_bits") == plan.window && uint(row,"warp_tail") == unsigned(plan.warp_tail);
}

bool reusable_dispatch(const Fields& row, const Device& device, unsigned p, unsigned batch,
                       const Stage1NttOptions& options, unsigned window) {
    try {
        return same_dispatch(row,describe(device,p,batch,options,window));
    } catch (const BudgetError&) {
        return false;
    }
}

// Validate the candidate at the current exponent before comparing its saved rate.
unsigned production_quality(const Fields& row, const Device& device, unsigned p, unsigned batch,
                            const Stage1NttOptions& options, unsigned window, bool allow_hint) {
    try {
        const auto plan=describe(device,p,batch,options,window);
        if (p <= pt::optional_uint(row,"max_exp") && (allow_hint || same_dispatch(row,plan))) return 3;
    } catch (const BudgetError&) {}
    return 0;
}

struct Score {
    unsigned batch = 0;
    Plan plan{};
    double rate = 0;
    double relative_sd = 0;
};

enum BatchRole : unsigned { Single = 1, Capacity = 2, Fastest = 4, Economical = 8 };

void table_header() {
    ecm_ts_fprintf(stdout,"length max_exp ntt_shape batch iter/s graph_steps dataflow actual/target steps/group steps/block\n");
}

void round_header(unsigned p, unsigned window, const char* phase) {
    ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_round: p=%u naf=%u phase=%s\n",p,window,phase);
}

std::string throughput_change(double current, double previous) {
    if (current <= 0 || previous <= 0) return "n/a";
    const double percent = 100 * ((current - previous) / previous);
    char change[64];
    std::snprintf(change,sizeof(change),percent == 0 ? "%.2f%%" : "%+.2f%%",percent);
    return change;
}

void report_batches(const std::map<BatchRole,Score>& scores, const Score& old95, const Score& measured95, unsigned p, unsigned w, unsigned percent = 95) {
    if (old95.rate > 0 && measured95.rate > 0) {
        const auto& published = scores.at(Economical);
        const bool changed = published.batch != old95.batch || published.rate != old95.rate;
        const auto accent = changed ? EcmLogAccent::Improved : EcmLogAccent::Retained;
        ecm_log_accent(accent,"stage1_ntt_95_compare: p=%u naf=%u length=%u batch=%u graph=%u\n",
                       p,w,measured95.plan.length,measured95.batch,measured95.plan.window);
        ecm_log_accent(accent,"%uc, %.9g vs %uc, %.9g (old)  %s, status=%s\n",
                       measured95.batch,measured95.rate,old95.batch,old95.rate,
                       throughput_change(measured95.rate,old95.rate).c_str(),changed ? "updated" : "kept");
    }
    std::vector<std::array<std::string,8>> rows;
    std::array<int,6> widths{};
    const auto target_label = std::to_string(percent) + "%";
    for (const auto& [role,label] : {std::pair{Single,"Single"}, {Capacity,"Max"},
                                    {Fastest,"Best"}, {Economical,target_label.c_str()}}) {
        const auto found = scores.find(role);
        if (found == scores.end()) continue;
        const auto& s = found->second;
        std::ostringstream rate;
        rate << std::fixed << std::setprecision(0) << s.rate;
        rows.push_back({label,std::to_string(s.plan.length),std::to_string(s.plan.max_exp),
                        tune_shape_descriptor(s.plan),std::to_string(s.batch)+"c",rate.str(),
                        std::to_string(s.plan.window),std::to_string(s.plan.dataflow)});
        for (size_t i=0; i<widths.size(); ++i)
            widths[i] = std::max(widths[i],int(rows.back()[i].size()));
    }
    for (const auto& row : rows)
        ecm_log_accent(EcmLogAccent::BatchResult,"%-*s %*s %*s %-*s %*s %*s iter/s graph_steps=%s dataflow=%s\n",
            widths[0],row[0].c_str(),widths[1],row[1].c_str(),widths[2],row[2].c_str(),
            widths[3],row[3].c_str(),widths[4],row[4].c_str(),widths[5],row[5].c_str(),row[6].c_str(),row[7].c_str());
}

enum class SamplePhase : unsigned { Coarse = 1, Fine = 2 };
SamplePhase sample_phase(const Fields& row) {
    const unsigned legacy = ecm_stage2::tune::uint(row,"repeats") == 5 ? 2 : 1;
    return static_cast<SamplePhase>(pt::optional_uint(row,"sample_phase",legacy));
}

std::vector<Fields> compact_batch_rows(std::vector<Fields> rows, const std::string& search, unsigned selected_batch) {
    using ecm_stage2::tune::uint;
    // Compare fine measurements when available. Coarse-only families still
    // establish a genuine upper admission endpoint even if pruned in refinement.
    const auto quality = [&](size_t j) {
        return std::make_pair(pt::tested_fine(rows[j]),ecm_stage2::tune::real(rows[j],"iter_per_s"));
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
    // Keep the requested target available to production alongside the
    // standard cached roles, even when fresh anchor rates differ from history.
    if (selected_batch) roles.try_emplace(selected_batch,0);
    std::vector<Fields> retained;
    for (size_t i=0; i<rows.size(); ++i) {
        auto& row=rows[i];
        const auto b=unsigned(uint(row,"batch"));
        if (!roles.count(b)) {
            for (const char* key : {"batch_scope","batch_search","batch_roles"}) row.erase(key);
            retained.push_back(std::move(row));
            continue;
        }
        if (!search.empty()) row["batch_search"]=pt::quoted(search);
        row["batch_roles"]=std::to_string(winners.at(b)==i ? roles.at(b) : 0);
        retained.push_back(std::move(row));
    }
    return retained;
}

enum class FineHistory { All, Hits, Snapshot };

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
    std::map<unsigned,Score> measured_fine_;
    std::string checkpoint_;
    std::map<unsigned,std::pair<SamplePhase,Score>> length_bests_;

    Score observe(SamplePhase phase, const Score& score) {
        auto& best = length_bests_[score.plan.length];
        if (phase > best.first || (phase == best.first && score.rate > best.second.rate))
            best = {phase,score};
        return score;
    }
public:
    bool header_pending = true;
    unsigned measured = 0;
    Score measured95() const {
        const auto peak = std::max_element(measured_fine_.begin(),measured_fine_.end(),[](const auto& a,const auto& b){
            return a.second.rate < b.second.rate;
        });
        if (peak == measured_fine_.end()) return {};
        return std::find_if(measured_fine_.begin(),measured_fine_.end(),[&](const auto& item){
            return item.second.rate >= near_peak*peak->second.rate;
        })->second;
    }
    TuneMatrix(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p,
               unsigned window, const pt::Profile& db, TuneNafSamples& naf_samples, const ecm_stage1::NttTuneGrid& grid = {}, double deadline = 0)
        : device_(device), request_(request), p_(p), window_(window), grid_(grid), deadline_(deadline),
          naf_samples_(naf_samples) {
        for (const auto& row : db.stage1_ntt) {
            const auto& condition = db.conditions.at(ecm_stage2::tune::uint(row,"condition"));
            if (!fits_recorded_memory(row,device,request) || !condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid ||
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
        if (grid_.force || !std::filesystem::exists(checkpoint_)) return;
        ecm_io::FileLock lock(checkpoint_);
        for (const auto& row : load(checkpoint_).stage1_ntt)
            if (fits_recorded_memory(row,device_,request_) && row.count("batch_scope") && pt::text(row,"batch_scope")==scope && same_workload(row,request_,window_))
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
            return fits_recorded_memory(row,device_,request_) && same_workload(row,request_,window_) &&
                   uint(row,"batch")==best.batch && uint(row,"length")==best.plan.length &&
                   (pt::text(row,"ntt_type")==pt::weighted_ntt_type)==best.plan.weighted &&
                   pt::tested_ms(row,2,grid_.fine_repeats) && same_dispatch(row,best.plan);
        };
        unsigned target = 0;
        for (auto i : hits_) {
            const auto& row = history_[i];
            if (matches(row) && real(row,"iter_per_s")==best.rate) {
                target = pt::tested_ms(row,2,grid_.fine_repeats);
                break;
            }
        }
        if (!target || path.empty()) return;
        round_header(p_,window_,"recheck");
        header_pending = true;
        Fields fresh;
        auto options = fixed_plan(request_.options.execution.ntt,best.plan);
        const auto test = sample(options,best.batch,SamplePhase::Fine,&fresh,target);
        double previous = best.rate;
        auto accent = EcmLogAccent::Retained;
        const char* status = test.rate ? "kept" : "unavailable";
        if (test.rate > 0) {
            // Another process may have raised the record or changed this search while the GPU ran.
            ecm_io::FileLock lock(path);
            auto db = load(path);
            const auto belongs = [&](const Fields& row) {
                const auto& condition = db.conditions.at(uint(row,"condition"));
                if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex")!=device_.uuid) return false;
                if (!fits_recorded_memory(row,device_,request_)) return false;
                if (scope.empty()) return true;
                return row.count("batch_scope") && pt::text(row,"batch_scope")==scope &&
                       row.count("batch_search") && pt::text(row,"batch_search")==search;
            };
            auto at = db.stage1_ntt.end();
            for (auto it = db.stage1_ntt.begin(); it != db.stage1_ntt.end(); ++it) {
                if (belongs(*it) && matches(*it) && pt::tested_ms(*it,2,grid_.fine_repeats)==target &&
                    (scope.empty() || (uint(*it,"batch_roles") & Fastest)) &&
                    (at == db.stage1_ntt.end() || real(*it,"iter_per_s") > real(*at,"iter_per_s"))) at = it;
            }
            if (at == db.stage1_ntt.end()) status = "superseded";
            else previous = real(*at,"iter_per_s");
            if (at != db.stage1_ntt.end() && !pt::stage1_supersedes(*at,fresh)) {
                const bool newer = pt::stage1_release_order(fresh,*at)>0;
                fresh["condition"] = at->at("condition");
                if (!scope.empty()) {
                    fresh["batch_scope"] = at->at("batch_scope");
                    fresh["batch_search"] = at->at("batch_search");
                }
                pt::validate_stage1_ntt(fresh);
                db.update_stage1_ntt(std::move(fresh));
                if (!scope.empty()) {
                    std::vector<Fields> group;
                    for (const auto& row : db.stage1_ntt) if (belongs(row)) group.push_back(row);
                    auto retained = compact_batch_rows(std::move(group),search);
                    db.stage1_ntt.erase(std::remove_if(db.stage1_ntt.begin(),db.stage1_ntt.end(),belongs),db.stage1_ntt.end());
                    db.stage1_ntt.insert(db.stage1_ntt.end(),retained.begin(),retained.end());
                    history_ = std::move(retained);
                }
                ecm_io::write_atomic(path,db.serialize());
                if (newer || test.rate > previous) best = test;
                status = newer || test.rate > previous ? "updated" : "retained";
                accent = EcmLogAccent::Improved;
            }
        }
        ecm_log_accent(accent,"stage1_ntt_recheck: p=%u naf=%u length=%u batch=%u graph=%u\n",
            p_,window_,best.plan.length,best.batch,best.plan.window);
        ecm_log_accent(accent,"%.9g vs %.9g (old)  %s, status=%s\n",test.rate,previous,throughput_change(test.rate,previous).c_str(),status);
    }
    std::optional<Stage1NttOptions> previous_plan(const Stage1NttOptions& options, unsigned batch,
                                                  const Plan& plan) const {
        std::optional<Cached> best;
        for (const auto& row : legacy_) {
            if ((pt::text(row,"ntt_type")==pt::weighted_ntt_type) != plan.weighted) continue;
            auto hint = decode(row,{plan.length,plan.bpw,plan.digits},device_,p_,batch,options);
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
                uint(row,"length") != options.length ||
                (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != options.weighted) continue;
            if (!reusable_dispatch(row,device_,p_,b,options,window_)) continue;
            if (b == fallback) return b;
            const auto rate = ecm_stage2::tune::real(row,"iter_per_s");
            if (rate > prior[b].rate) prior[b] = {rate,1};
        }
        const unsigned previous = recommended_batch(prior);
        return previous ? previous : fallback;
    }
    // Explicit batch ranges have no compact role rows. Recover their finer
    // measurements, including midpoints absent from the initial shortlist.
    void restore_finals(const Stage1NttOptions& base, unsigned cap, std::map<unsigned,Score>& finals, FineHistory mode) {
        using ecm_stage2::tune::uint;
        const auto shape = candidate(p_,base,ecm_stage1::stage1_ntt_radix5(base.length),
            window_ ? WeightedBound::UnsignedProducts : WeightedBound::Conservative);
        if (!shape) return;
        for (size_t i=0; i<history_.size(); ++i) {
            if (mode == FineHistory::Hits && !hits_.count(i)) continue;
            const auto& row = history_[i];
            const unsigned batch = unsigned(uint(row,"batch"));
            if (batch < grid_.batch.first || batch > cap || uint(row,"length") != base.length ||
                (pt::text(row,"ntt_type")==pt::weighted_ntt_type) != base.weighted ||
                pt::tested_ms(row,2,grid_.fine_repeats) < grid_.group_ms) continue;
            auto hit = decode(row,shape->shape,device_,p_,batch,base);
            if (!hit || (base.mixed_layout != Stage1NttOptions::MixedLayout::Automatic &&
                         hit->options.mixed_layout != base.mixed_layout)) continue;
            try {
                auto plan = describe(device_,p_,batch,hit->options,window_);
                if (!same_dispatch(row,plan)) continue;
                const Score score{batch,plan,hit->rate};
                if (mode != FineHistory::Snapshot) hits_.insert(i);
                if (score.rate > finals[batch].rate)
                    finals[batch] = mode == FineHistory::Snapshot ? score : observe(SamplePhase::Fine,score);
            } catch (const BudgetError&) {}
        }
    }
    void flush(bool explicit_batch = false) {
        if (explicit_batch) {
            for (auto i : hits_) {
                if (pt::optional_uint(history_[i],"explicit_batch")) continue;
                auto row = history_[i];
                row.erase("batch_scope");
                row.erase("batch_search");
                row.erase("batch_roles");
                pending_.push_back(std::move(row));
            }
        }
        if (explicit_batch) for (auto& row : pending_) row["explicit_batch"]="1";
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
            if (std::max(pt::tested_ms(row,1,grid_.coarse_repeats),pt::tested_ms(row,2,grid_.fine_repeats)) < grid_.group_ms) return false;
            const unsigned batch = unsigned(uint(row,"batch"));
            auto options = request_.options.execution.ntt;
            options.weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            const Packing shape{unsigned(uint(row,"length")),unsigned(uint(row,"bpw"))};
            const auto decoded = decode(row,shape,device_,p_,batch,options);
            if (!decoded) return false;
            try {
                const auto plan = describe(device_,p_,batch,decoded->options,window_);
                if (!same_dispatch(row,plan)) return false;
                observe(sample_phase(row),{batch,plan,decoded->rate});
                if (pt::tested_ms(row,1,grid_.coarse_repeats) >= grid_.group_ms &&
                    (batch > coarse.batch || (batch == coarse.batch && decoded->rate > coarse.rate)))
                    coarse = {batch,plan,decoded->rate};
                for (auto role : {Single, Capacity, Fastest, Economical})
                    if ((pt::optional_uint(row,"batch_roles",0) & role) && decoded->rate > selected[role].rate)
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

    void save_batches(const std::string& scope, const std::string& search, bool complete, unsigned selected_batch) {
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
        if (grid_.force && complete)
            persist(request_.options.execution.ntt.tune_file,device_,std::move(rows),scope,false,false,search,&request_,selected_batch);
        else {
            auto retained = compact_batch_rows(std::move(rows),complete ? search : std::string{},selected_batch);
            persist(request_.options.execution.ntt.tune_file,device_,std::move(retained),scope,!complete);
        }
        if (complete) finish_checkpoint();
    }
    Score sample(const Stage1NttOptions& options, unsigned batch, SamplePhase phase,
                 Fields* remeasurement = nullptr, unsigned saved_target = 0, bool fresh = false) {
        const unsigned groups = phase == SamplePhase::Fine ? grid_.fine_repeats : grid_.coarse_repeats;
        const unsigned target = remeasurement ? saved_target : grid_.group_ms;
        if (ecm_stage1_stop_requested()) throw Cancelled();
        try {
            auto plan = describe(device_, p_, batch, options, window_);
            if (options.fused_radix5_forward && !(plan.dataflow & Radix5ForwardOuter)) return {};
            using ecm_stage2::tune::uint;
            const auto lookup = [&](const std::vector<Fields>& rows, bool previous) -> Score {
                Score best;
                size_t selected = 0;
                for (size_t i=0; i<rows.size(); ++i) {
                    const auto& row = rows[i];
                    if (uint(row,"length") != plan.length || uint(row,"bpw") != plan.bpw ||
                        (pt::text(row,"ntt_type") == pt::weighted_ntt_type) != plan.weighted ||
                        pt::tested_ms(row,unsigned(phase),groups) < target || uint(row,"batch") != batch ||
                        !same_dispatch(row,plan)) continue;
                    // An explicit search owns its existing unscoped rows. Scoped
                    // samples can seed it when that configuration is missing.
                    if (best.rate && grid_.batch_explicit && !rows[selected].count("batch_scope") &&
                        row.count("batch_scope")) continue;
                    const auto rate = ecm_stage2::tune::real(row,"iter_per_s");
                    if (rate <= best.rate) continue;
                    selected = i;
                    best = {batch, plan, rate,
                        1000 * ecm_stage2::tune::real(row,"stddev_seconds") /
                            (window_ ? ecm_stage2::tune::real(row,"total_device_ms") / uint(row,"repeats") :
                                       ecm_stage2::tune::real(row,"median_ms"))};
                }
                if (previous && best.rate && hits_.insert(selected).second)
                    ecm_ts_fprintf(stdout,"stage1_ntt_reuse: length=%u batch=%u ntt_shape=%s "
                        "graph_steps=%u dataflow=%u target_ms=%u iter_per_s=%.0f\n",plan.length,batch,
                        tune_shape_descriptor(plan).c_str(),plan.window,plan.dataflow,target,best.rate);
                return best;
            };
            if (!remeasurement && !fresh) {
                if (auto hit = lookup(pending_, false); hit.rate) return observe(phase,hit);
                if (auto hit = lookup(history_, true); hit.rate) return observe(phase,hit);
            }
            if (deadline_ && now_s() >= deadline_) throw TimeLimit{};
            if (!calibration_) {
                Calibration prepared(device_, p_, 64);
                if (window_) prepared.native(window_, naf_samples_.get(window_),
                    size_t(request_.options.execution.naf_memory_mb) * ecm_edwards_memory::mb,options);
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
            row["sample_pool_steps"] = std::to_string(m.pool_steps);
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
            if (header_pending) {
                table_header();
                header_pending = false;
            }
            print_measurement(m, target);
            Score score{batch,m.plan,m.throughput()*m.steps,pt::deviation(m.ms)/m.group_ms()};
            if (phase == SamplePhase::Fine && score.rate > measured_fine_[batch].rate)
                measured_fine_[batch] = score;
            return observe(phase,score);
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
    if (auto previous=matrix.previous_plan(o,batch,base)) {
        previous->mixed_layout = o.mixed_layout;
        previous->window = o.window;
        trials.push_back(*previous);
    }
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
    Score capacity, old95;
    const bool scan_graph;
    double single = 0, peak = 0;
    bool reached_caps = true, cached_fine = false, persisted = false;
    double seconds = 0;
    std::map<BatchRole,Score> cached, found;
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
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_plan: p=%u length=%u bpw=%u naf=%u capacity=%u%s\n",
                p,shape.shape.length,shape.shape.bpw,w,cap,cap ? "" : " reason=memory_budget");
            if (cap < grid.batch.first) continue;
            auto base = shape.options;
            if (mode == Mixed) base.mixed_layout = Layout::Coalesced;
            families.push_back({base, cap});
        }
        // Hold Graph size constant while comparing windows, layouts and batches.
        if (scan_graph)
            for (auto& family : families) family.options.window = 16;
        if (families.empty()) return;
        if (grid.batch_explicit) {
            std::map<unsigned,Score> historical;
            restore_fine_scores(historical,FineHistory::Snapshot);
            if (!historical.empty()) {
                const auto best = std::max_element(historical.begin(),historical.end(),[](const auto& a,const auto& b){
                    return a.second.rate < b.second.rate;
                });
                old95 = std::find_if(historical.begin(),historical.end(),[&](const auto& item){
                    return item.second.rate >= near_peak*best->second.rate;
                })->second;
            }
            return;
        }
        scope << curve_name(request) << '/' << w
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
                scope << '/' << base.bpw << '/' << base.dataflow;
            } catch (const BudgetError&) { scope << "/unavailable/" << p; }
        }
        for (const auto& f : families)
            search << ':' << f.options.weighted << '/' << f.options.length << '/' << int(f.options.mixed_layout) << '/' << f.cap;
        matrix.open_checkpoint(scope.str(),search.str());
        cached_fine = matrix.restore_batches(scope.str(),search.str(),cached,capacity);
        if (cached_fine) old95 = cached.at(Economical);
        else matrix.restore_batches(scope.str(),search.str()+"/screen",cached,capacity);
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
            }
        }
        if (finals.empty()) return false;
        const auto best = std::max_element(finals.begin(),finals.end(),[](const auto& a,const auto& b){
            return a.second.rate<b.second.rate;
        });
        auto& winner = best->second;
        if (ecm_stage1::stage1_ntt_radix5(winner.plan.length)) {
            using Layout = Stage1NttOptions::MixedLayout;
            auto options = fixed_plan(families.front().options,winner.plan);
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_pfa_scan: p=%u naf=%u batch=%u length=%u graph=%u\n",
                           p,w,winner.batch,winner.plan.length,winner.plan.window);
            matrix.header_pending = true;
            options.fused_radix5_forward = false;
            for (const auto layout : {Layout::Twiddle,Layout::Pfa,Layout::Coalesced}) {
                options.mixed_layout = layout;
                const auto score = matrix.sample(options,winner.batch,SamplePhase::Fine);
                if (score.rate > winner.rate) winner = score;
            }
            if (w && winner.plan.passes > 2 && (winner.plan.dataflow & ForwardTileInput)) {
                options.mixed_layout = Layout::Coalesced;
                options.fused_radix5_forward = true;
                const auto score = matrix.sample(options,winner.batch,SamplePhase::Fine);
                if (score.rate > winner.rate) winner = score;
            }
            if (p / winner.plan.length >= radix5_limb_min_digit_bits) {
                options = fixed_plan(families.front().options, winner.plan);
                options.fused_radix5_inverse = !options.fused_radix5_inverse;
                const auto score = matrix.sample(options,winner.batch,SamplePhase::Fine);
                if (score.rate > winner.rate) winner = score;
            }
        }
        if (scan_graph) {
            auto options = fixed_plan(families.front().options,winner.plan);
            ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_graph_scan: p=%u naf=%u batch=%u length=%u\n",
                           p,w,winner.batch,winner.plan.length);
            matrix.header_pending = true;
            const int measured_window = int(winner.plan.window);
            for (int steps : {4, 8, 16, 32, 64}) {
                if (steps == measured_window) continue;
                options.window = steps;
                const auto score = matrix.sample(options,winner.batch,SamplePhase::Fine);
                if (score.rate > winner.rate) winner = score;
            }
        }
        return true;
    }

    void restore_fine_scores(std::map<unsigned,Score>& scores, FineHistory mode) {
        for (const auto& f : families) {
            auto base = f.options;
            if (scan_graph) base.window = -1;
            base.mixed_layout = Stage1NttOptions::MixedLayout::Automatic;
            matrix.restore_finals(base,f.cap,scores,mode);
        }
    }

    void find_target() {
        if (grid.batch_explicit) restore_fine_scores(finals,FineHistory::All);
        const auto peak = cached_fine ? cached.at(Fastest) :
            std::max_element(finals.begin(),finals.end(),[](const auto& a,const auto& b){
                return a.second.rate < b.second.rate;
            })->second;
        const auto single_plan = cached_fine && cached.count(Single) ? cached.at(Single).plan :
                                 finals.count(1) ? finals.at(1).plan : peak.plan;
        const auto fine_capacity = finals.find(capacity.batch);
        const auto max_plan = cached_fine ? cached.at(Capacity) :
            fine_capacity != finals.end() ? fine_capacity->second : capacity;
        const auto measure_anchor = [&](unsigned batch, const Plan& plan) {
            return matrix.sample(fixed_plan(families.front().options,plan),batch,SamplePhase::Fine,nullptr,0,true);
        };
        round_header(p,w,"find_anchors");
        matrix.header_pending = true;
        const auto single = measure_anchor(1,peak.batch == 1 ? peak.plan : single_plan);
        const auto remeasured_peak = peak.batch == 1 ? single : measure_anchor(peak.batch,peak.plan);
        const auto maximum = max_plan.batch == peak.batch ? remeasured_peak :
            measure_anchor(max_plan.batch,max_plan.plan);
        if (!single.rate || !remeasured_peak.rate || !maximum.rate)
            throw BudgetError("could not remeasure Stage1 NTT find anchors");
        auto best = maximum.rate > remeasured_peak.rate ? maximum : remeasured_peak;
        auto first = grid.batch.first;
        if (first == 1 && single.rate > best.rate) best = single;
        ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_find: p=%u naf=%u target=%u best_batch=%u graph=%u\n",
                       p,w,grid.find_percent,best.batch,best.plan.window);
        matrix.header_pending = true;
        const auto options = fixed_plan(families.front().options,best.plan);
        const double fraction = grid.find_percent / 100.0;
        auto selected = remeasured_peak.batch < best.batch && remeasured_peak.rate > fraction * best.rate
            ? remeasured_peak : best;
        if (first == 1) {
            if (single.rate > fraction * best.rate) selected = single;
            else ++first;
        }
        // The selected endpoint qualifies. If a midpoint raises the peak,
        // it also becomes the new qualifying endpoint. Earlier failures
        // remain failures because the threshold can only increase.
        while (first < selected.batch) {
            const auto batch = first + (selected.batch - first) / 2;
            const auto score = matrix.sample(options,batch,SamplePhase::Fine);
            if (score.rate > best.rate) best = score;
            if (score.rate > fraction * best.rate) selected = score;
            else first = batch + 1;
        }
        found = {{Single,single},{Capacity,maximum},{Fastest,best},{Economical,selected}};
    }

    void report(bool repeat) {
        if (!found.empty()) {
            report_batches(found,{},{},p,w,grid.find_percent);
            return;
        }
        if (cached_fine) {
            if (repeat) {
                auto& best = cached.at(Fastest);
                const double previous = best.rate;
                matrix.recheck(best,scope.str(),search.str());
                if (best.rate != previous)
                    matrix.restore_batches(scope.str(),search.str(),cached,capacity);
            }
            report_batches(cached,old95,matrix.measured95(),p,w);
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
        report_batches(scores,old95,matrix.measured95(),p,w);
    }

    void save(bool complete, bool refined) {
        if (grid.batch_explicit) matrix.flush(true);
        else {
            const auto id = search.str() + (reached_caps ? "" : "/recheck") + (refined ? "" : "/screen");
            matrix.save_batches(scope.str(),id,complete,found.empty() ? 0 : found.at(Economical).batch);
        }
    }
};

bool tune_windows(const Device& device, const ecm_stage1::Stage1Request& request, unsigned p,
                  const Candidates& shapes, const ecm_stage1::NttTuneGrid& grid, TuneNafSamples& naf_samples,
                  const std::vector<unsigned>& windows, bool compete, bool exhaustive, double deadline,
                  bool recheck_cached = false) {
    const auto db = grid.force ? pt::Profile{} : load(request.options.execution.ntt.tune_file);
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
            if (advance) {
                round_header(p,trial.w,"fine");
                trial.matrix.header_pending = true;
            }
            if (compete)
                ecm_log_accent(EcmLogAccent::Subtitle,"stage1_ntt_naf_screen: p=%u naf=%u capacity=%u iter_per_s=%.0f refine=%u\n",
                               p,trial.w,trial.capacity.batch,trial.capacity.rate,unsigned(advance));
            const bool refined = advance && trial.refine(exhaustive);
            finished |= refined;
            if (refined && grid.find_percent) trial.find_target();
            if (trial.cached.empty() || (refined && (!trial.cached_fine || grid.find_percent)))
                trial.save(!advance || refined,refined);
            else trial.matrix.finish_checkpoint();
            trial.persisted = true;
            if (refined) trial.report(recheck_cached && !grid.find_percent && !trial.matrix.measured);
            trial.matrix.release_calibration();
            trial.matrix.report_lengths();
            ecm_log_accent(EcmLogAccent::Report,"stage1_ntt_tune: p=%u measured=%u reused=%zu seconds=%.3f naf=%u\n",
                           p,trial.matrix.measured,trial.matrix.reused(),trial.seconds+now_s()-started,trial.w);
            ecm_ts_fprintf(stdout,"============================================================\n");
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
    return ecm_stage1::ntt_shape_text(plan.weighted, std::to_string(plan.tile),
                                     plan.forward, plan.inverse, pfa_code(plan));
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
        const auto& path=task.execution.ntt.tune_file;
        if (!path.empty() && std::filesystem::is_regular_file(path)) {
            ecm_io::FileLock lock(path);
            auto profile=load(path);
            const bool graph_compacted=pt::compact_stage1_graphs(profile.stage1_ntt);
            if (!profile.stage1_ntt.empty() && (graph_compacted || profile.stage1_compacted ||
                    pt::optional_uint(profile.metadata,"format")<pt::format))
                ecm_io::write_atomic(path,profile.serialize());
        }
        const auto device = device_info(task.execution.device_index, task.execution.ntt.memory_mib);
        ecm_ts_fprintf(stdout,"ntt_shape = convolution:tile_layers:forward:inverse:pfa\n");
        ecm_ts_fprintf(stdout,"convolution 0=linear 1=weighted; pfa 0=none 1=direct 2=coalesced 3=coalesced+outer\n");
        std::vector<unsigned> windows;
        if (!edwards) windows = {0};
        else if (grid.naf.first) {
            for (unsigned w=grid.naf.first; w<=grid.naf.last; ++w) windows.push_back(w);
        } else windows = {6,8,10,12};

        const std::string empty;
        const double deadline = seconds ? now_s()+seconds : 0;
        bool measured = false;
        TuneNafSamples naf_samples{uint64_t(b1)};
        const auto plan = plan_tune_lengths(task.execution.ntt, grid, edwards);
        for (const auto& entry : plan) {
            const Stage1Request request{nullptr,b1,b1,task.execution.ntt.max_batch,empty,false,empty,task};
            measured |= tune_windows(device,request,entry.reference_bits,entry.shapes,grid,naf_samples,windows,
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
        if (!fits_recorded_memory(row,device,request)) continue;
        if (p > pt::optional_uint(row,"max_exp")) continue;
        const bool current = compatible_workload(row,request,window) || (pt::is_runtime_stage1_ntt(row) &&
            pt::text(row,"curve")==curve_name(request) && uint(row,"naf_window")==window);
        if (!current && (row.count("benchmark_kind") || window)) continue;
        const auto& condition = db.conditions.at(uint(row,"condition"));
        if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;
        const bool weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
        for (unsigned mode=0; mode<CandidateCount; ++mode) {
            const auto& plan = candidates[mode];
            if (!plan || plan->options.weighted!=weighted) continue;
            auto sample = decode(row,plan->shape,device,p,batch,plan->options,current);
            if (!sample) continue;
            const unsigned possible_quality = current ? 3 : 0;
            if (choices[mode] && (quality[mode] > possible_quality ||
                (quality[mode] == possible_quality && choices[mode]->rate >= sample->rate))) continue;
            const unsigned q = current ? production_quality(row,device,p,batch,sample->options,window,true) : 0;
            if (current && !q) continue;
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
    const bool covered = std::any_of(quality.begin(),quality.end(),[](unsigned q){return q != 0;});
    for (unsigned mode=0; mode<CandidateCount; ++mode) {
        if (covered || !options.runtime_enabled || !candidates[mode]) continue;
        const auto plan=choices[mode] ? choices[mode]->options : candidates[mode]->options;
        choices[mode].reset();
        auto row=matrix.sample(plan,batch,SamplePhase::Coarse);
        if (!row.rate) continue;
        choices[mode]=Cached{fixed_plan(plan,row.plan),row.rate};
        quality[mode]=3;
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

bool reuse_tuned_plan(const ecm_stage1::Stage1Request& request, unsigned p, unsigned batch, unsigned window,
                      const Stage1NttOptions& options, BatchMemory& memory) {
    const auto device = device_info(request.options.execution.device_index, options.memory_mib);
    // The fixed packing was already checked at this exponent. Only recheck
    // the live budgets of its one transform before entering the allocator.
    Candidates shapes;
    shapes[0] = Candidate{options,Packing{options.length}};
    memory = production_capacity(request,device,p,window,shapes)[0];
    return memory.max_curves >= batch;
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
    auto best_window = [&](const pt::Profile& db) -> std::optional<unsigned> {
        std::optional<unsigned> best;
        std::pair<unsigned,double> fastest{};
        std::map<unsigned,std::array<BatchMemory,CandidateCount>> capacities;
        std::map<unsigned,Candidates> shapes;
        for (const auto& row : db.stage1_ntt) {
            if (!fits_recorded_memory(row,device,request)) continue;
            const bool runtime = pt::is_runtime_stage1_ntt(row);
            if (!runtime && !row.count("sample_target_ms")) continue;
            const auto w = unsigned(uint(row,"naf_window")), batch = unsigned(uint(row,"batch"));
            if ((configured && w != configured) || (!edwards && w) || (edwards && !w && !configured) ||
                batch > limit || (fixed_batch && batch != limit)) continue;
            if (p > pt::optional_uint(row,"max_exp") || pt::text(row,"curve") != curve_name(request)) continue;
            const auto& condition = db.conditions.at(uint(row,"condition"));
            if (!condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid) continue;

            if (!shapes.count(w)) {
                shapes[w] = production_shapes(p,exec.ntt,w != 0);
                capacities[w] = production_capacity(request,device,p,w,shapes.at(w));
            }
            for (unsigned mode=0; mode<CandidateCount; ++mode) {
                const auto& candidate = shapes.at(w)[mode];
                if (!candidate || batch > capacities.at(w)[mode].max_curves ||
                    candidate->options.weighted != (pt::text(row,"ntt_type") == pt::weighted_ntt_type)) continue;
                const bool hint=true;
                auto decoded = decode(row,candidate->shape,device,p,batch,candidate->options,hint);
                if (!decoded) continue;
                const unsigned quality=production_quality(row,device,p,batch,decoded->options,w,hint);
                const auto score=std::make_pair(quality,ecm_stage2::tune::real(row,"iter_per_s"));
                if (!quality || score<=fastest) continue;
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
    auto best = best_window(db);
    if (!best && exec.ntt.runtime_enabled) {
        TuneNafSamples naf_samples{uint64_t(request.b1)};
        NttTuneGrid grid;
        grid.batch = {fixed_batch ? limit : 1u, limit};
        grid.batch_explicit = fixed_batch;
        std::vector<unsigned> windows{configured};
        if (edwards && !configured) windows = {6,8,10,12};
        ecm_ts_fprintf(stdout,"stage1_ntt_runtime_tune: p=%u batch=%u-%u\n",p,grid.batch.first,limit);
        try {
            for (const auto& shape : tune_shapes(p,exec.ntt,edwards)) {
                if (!shape) continue;
                Candidates one{};
                const unsigned mode = !shape->options.weighted ? Linear :
                    ecm_stage1::stage1_ntt_radix5(shape->shape.length) ? Mixed : Weighted;
                one[mode] = shape;
                tune_windows(device,request,p,one,grid,naf_samples,windows,windows.size()>1,false,0);
            }
        } catch (const Cancelled&) { return configured; }
        best = best_window(load(exec.ntt.tune_file));
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
    const auto span_started = std::chrono::steady_clock::now();
    const double started = now_s();
    if (!remaining) return 0;
    const auto& exec = request.options.execution;
    const auto device = device_info(exec.device_index, exec.ntt.memory_mib);
    const auto p = ecm_stage1::ntt_arithmetic_bits(request.modulus, request.options);
    const auto window = naf_window(request);
    const auto shapes = production_shapes(p, exec.ntt, window != 0);
    if (std::none_of(shapes.begin(), shapes.end(), [](const auto& plan) { return plan.has_value(); }))
        throw std::invalid_argument("no legal Stage1 NTT packing for the requested parameters");
    const auto capacities = production_capacity(request, device, p, window, shapes);
    const auto linear_cap = capacities[Linear].max_curves;
    const auto weighted_cap = std::max(capacities[Weighted].max_curves, capacities[Mixed].max_curves);
    const auto cap = std::min(remaining, std::max({1u, linear_cap, weighted_cap}));
    std::map<unsigned, BatchRate> eligible;
    std::map<unsigned, bool> runtime_winner;
    using ecm_stage2::tune::uint;
    using ecm_stage2::tune::real;
    try {
        const auto db = load(exec.ntt.tune_file);
        for (const auto& row : db.stage1_ntt) {
            if (!fits_recorded_memory(row,device,request)) continue;
            const auto& condition = db.conditions.at(uint(row,"condition"));
            const auto batch = unsigned(uint(row,"batch"));
            if (batch > cap || !condition.count("uuid_hex") || pt::text(condition,"uuid_hex") != device.uuid)
                continue;
            const bool weighted = pt::text(row,"ntt_type") == pt::weighted_ntt_type;
            if (p > pt::optional_uint(row,"max_exp")) continue;
            const bool modern = row.count("sample_target_ms") != 0;
            if (modern && !compatible_workload(row,request,window)) continue;
            if (!modern && window && !pt::is_runtime_stage1_ntt(row)) continue;
            const bool runtime = pt::is_runtime_stage1_ntt(row);
            if (runtime && (uint(row,"naf_window") != window ||
                            pt::text(row,"curve") != curve_name(request)))
                continue;
            const auto rate = real(row,"iter_per_s");
            const auto incumbent = eligible.find(batch);
            if (incumbent != eligible.end() && rate <= incumbent->second.rate) continue;
            unsigned quality = 0;
            for (unsigned mode = 0; mode < CandidateCount; ++mode) {
                const auto& shape = shapes[mode];
                if (!shape || batch>capacities[mode].max_curves || shape->options.weighted!=weighted) continue;
                const bool hint = modern || pt::is_runtime_stage1_ntt(row);
                auto decoded = decode(row,shape->shape,device,p,batch,shape->options,hint);
                if (decoded) quality = std::max(quality, modern ?
                    production_quality(row,device,p,batch,decoded->options,window,hint) : 3u);
            }
            if (!quality) continue;
            auto& value = eligible[batch];
            if (rate > value.rate) {
                value = {rate,1};
                runtime_winner[batch] = runtime;
            }
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT batch tune read: %s\n", e.what());
    }
    const unsigned tuned = recommended_batch(eligible,exec.ntt.batch_target_percent);
    unsigned selected = tuned ? tuned : std::min(cap, 64u);
    const char* source = tuned ? (runtime_winner.at(tuned) ? "production" : "tune") : "default";
    if (!tuned && exec.ntt.runtime_enabled && !exec.ntt.tune_file.empty() && remaining > cap) {
        selected = cap;
        source = "production_trial";
    }
    ecm_ts_fprintf(stdout, "stage1_ntt_batch_selection: source=%s p=%u remaining=%u batch=%u capacity=%u "
                   "linear_capacity=%u weighted_capacity=%u ms=%.3f\n",
                   source, p, remaining, selected, cap, linear_cap, weighted_cap, 1000 * (now_s() - started));
    log_batch_span("select_batch",span_started);
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
                                      run.budget_bytes, run.dictionary_spectra_bytes + run.dictionary_workspace_bytes}, run.completed_bits);
        row["benchmark_kind"] = pt::quoted(pt::stage1_runtime_ntt_kind);
        row["curve"] = pt::quoted(curve_name(request));
        row["reference_bits"] = std::to_string(ecm_stage1::ntt_arithmetic_bits(request.modulus, request.options));
        row["b1"] = std::to_string(uint64_t(std::floor(request.b1 + 0.0001)));
        row["naf_window"] = std::to_string(naf_window(request));
        row["iter_per_s"] = pt::numeric(rate);
        row["curves_per_s"] = pt::numeric(compute_rate);
        row["dataflow"] = std::to_string(plan.dataflow);
        const auto store_span = std::chrono::steady_clock::now();
        const double store_started = now_s();
        persist(exec.ntt.tune_file, device, {std::move(row)}, {}, false, true);
        log_batch_span("tune_write",store_span);
        ecm_ts_fprintf(stdout, "stage1_ntt_production_sample: batch=%u valid_curves=%u iter_per_s=%.3f "
            "curves_per_s=%.6f compute_seconds=%.6f persist_ms=%.3f\n", batch, valid, rate, compute_rate, compute_seconds,
            1000 * (now_s() - store_started));
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT production tune write: %s\n", e.what());
    }
}

} // namespace ecm_mersenne_ntt
