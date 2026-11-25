#include "workflow.h"
#include "batch_timing.h"
#include "ecm/stage1/progress.h"
#include "prefetch.h"
#include "ladder.h"
#include "host_lanes.h"
#include "tune.h"
#include "../../montgomery/cuda/host_math.h"
#include "../../execution_details.h"
#include "../../sigma.h"
#include "../../../curve/edwards.h"
#include "../../edwards/ntt_dictionary.h"
#include "../../edwards/cpu/ecm_edwards_cpu.h"
#include "../../edwards/memory.h"
#include "../../../../io/stage1_binary.h"
#include "../../../../io/stage1_checkpoint.h"
#include "ecm/stage1/ntt_target.h"
#include "ecm/integer.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>

namespace ecm_cuda_stage1 {
namespace {
using namespace ecm_stage1;
using Clock = std::chrono::steady_clock;
using ActiveLanes = std::bitset<stage1_ntt_max_curves>;

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void log_ntt_execution(const ExecutionOptions& opt, size_t modulus_bits, size_t batch,
                       unsigned naf_window, unsigned resume, const ecm_mersenne_ntt::Plan& plan,
                       const ecm_mersenne_ntt::BatchMemory& memory,
                       size_t valid = 0, size_t target_bits = 0, uint64_t b1 = 0, unsigned iterations = 0) {
    const bool edwards = opt.execution.selection.curve == CurveModel::Edwards;
    const char* chain = !edwards ? "ladder" : naf_window ? "naf" : "x-only";
    std::ostringstream details;
    details << "B1=" << b1 << " iter_total=" << iterations << " device=" << opt.execution.device_index << " chain=" << chain
            << " ntt_bits=" << modulus_bits << " ntt_type=" << (plan.weighted ? "weighted" : "linear")
            << " ntt_length=" << plan.length << " ntt_bpw=" << plan.bpw
            << " naf=" << naf_window << " ntt_batch=" << batch << " ntt_digits=" << plan.digits
            << " ntt_resume=" << resume << " ntt_shape=" << ecm_mersenne_ntt::tune_shape_descriptor(plan)
            << " ntt_tile_bits=" << plan.tile
            << " ntt_graph_bits=" << plan.window << " ntt_warp_tail=" << unsigned(plan.warp_tail)
            << " ntt_forward_pass_bits=" << plan.forward << " ntt_inverse_pass_bits=" << plan.inverse
            << " ntt_bpw_avg=" << std::fixed << std::setprecision(6) << double(modulus_bits) / plan.digits
            << " ntt_bpw_max=" << plan.bpw << " ntt_u=" << plan.coefficient_bound
            << " ntt_q=" << plan.field_modulus
            << " ntt_memory_per_curve_bytes=" << memory.per_curve_bytes
            << " ntt_memory_config_mib=" << memory.configured_mib
            << " ntt_memory_device_bytes=" << memory.device_budget_bytes
            << " ntt_naf_memory_mb=" << memory.naf_budget_mb
            << " max_curves_in_memory=" << memory.max_curves;
    if (valid && valid != batch) details << " ntt_valid_curves=" << valid;
    if (target_bits && target_bits != modulus_bits)
        details << " target_bits=" << target_bits << " carrier_exponent=" << modulus_bits;
    log_execution_details(opt, "ntt", details.str());
}

struct Snapshot {
    unsigned count = 0;
    unsigned cursor = 0;
    unsigned naf_window = 0;
    ActiveLanes lanes;
    std::vector<uint32_t> points;
};

constexpr std::string_view single_mask_snapshot = "ECMNTTS1";
constexpr std::string_view multi_mask_snapshot = "ECMNTTS2";
constexpr std::string_view edwards_naf_snapshot = "ECMNTTE1";

bool read_snapshot(std::string_view bytes, unsigned p, unsigned curves, bool edwards, Snapshot& out) {
    using ecm_stage1_binary::Reader;
    Reader reader(bytes);
    const unsigned mask_words = (curves + 31) / 32;
    if (reader.left() < single_mask_snapshot.size() + (2 + mask_words) * 4)
        return false;
    const auto magic = reader.take(single_mask_snapshot.size());
    Snapshot decoded;
    decoded.count = reader.u32();
    if (magic == edwards_naf_snapshot) {
        if (!edwards || reader.left() < (2 + mask_words) * 4)
            return false;
        decoded.naf_window = reader.u32();
        if (decoded.naf_window < 2 || decoded.naf_window > 12)
            return false;
    } else if (magic != (curves <= 32 ? single_mask_snapshot : multi_mask_snapshot)) {
        return false;
    }
    decoded.cursor = reader.u32();
    for (unsigned word = 0; word < mask_words; ++word) {
        const auto mask = reader.u32();
        const unsigned base = 32 * word;
        const unsigned width = std::min(32u, curves - base);
        if (width < 32 && (mask >> width)) return false;
        for (unsigned bit = 0; bit < width; ++bit)
            decoded.lanes[base + bit] = (mask >> bit) & 1u;
    }
    if (decoded.cursor > decoded.count || decoded.lanes.none())
        return false;
    const unsigned words = (p + 31) / 32;
    const size_t size = size_t(decoded.naf_window ? 3 : 4) * words * decoded.lanes.count();
    if (reader.left() != size * sizeof(uint32_t))
        return false;
    decoded.points.resize(size);
    for (auto& word : decoded.points) word = reader.u32();
    const unsigned top = p % 32 ? (1u << (p % 32)) - 1 : UINT32_MAX;
    for (size_t i = 0; i < decoded.points.size(); i += words) {
        const auto hi = decoded.points.begin() + i + words - 1;
        if (*hi > top || (*hi == top && std::all_of(decoded.points.begin() + i, hi,
                                                    [](auto v) { return v == UINT32_MAX; })))
            return false;
    }
    out = std::move(decoded);
    return true;
}

std::string encode_snapshot(unsigned count, unsigned cursor, unsigned curves, const ActiveLanes& lanes,
                            unsigned naf_window, const std::vector<uint32_t>& points) {
    using ecm_stage1_binary::u32;
    std::string payload(naf_window ? edwards_naf_snapshot :
                        curves <= 32 ? single_mask_snapshot : multi_mask_snapshot);
    payload.reserve(payload.size() + (2 + (curves + 31) / 32 + points.size()) * 4);
    u32(payload, count);
    if (naf_window) u32(payload, naf_window);
    u32(payload, cursor);
    for (unsigned base = 0; base < curves; base += 32) {
        uint32_t mask = 0;
        for (unsigned bit = 0; bit < std::min(32u, curves - base); ++bit)
            if (lanes[base + bit]) mask |= uint32_t(1) << bit;
        u32(payload, mask);
    }
    for (auto word : points) u32(payload, word);
    return payload;
}

// Edwards saves already use the Montgomery x coordinate (1+y)/(1-y).
// Construct the Edwards curve and point before mapping into the shared ladder.
CurveKind prepare_edwards_lane(mpz_srcptr n, uint64_t sigma, mpz_ptr factor, uint32_t* initial,
                               unsigned words, mpz_ptr d, mpz_ptr px, mpz_ptr py,
                               ecm_edwards::MersenneConstruction* construction,
                               const ecm_edwards::MersenneConstruction::Jacobian* auxiliary) {
    using ecm_edwards::Integer;
    using ecm_edwards::Setup;
    Integer a24, x, inv, den;
    mpz_set_si(x.v, -1);
    auto setup = ecm_edwards::construct(d, px, py, a24.v, factor, sigma, n, construction, auxiliary,
                                        auxiliary && construction ? x.v : nullptr);
    if (auxiliary && setup != Setup::Ready) {
        mpz_gcd(den.v, auxiliary->z.v, n);
        ecm_ts_fprintf(stdout,"stage1_ntt_auxiliary: sigma=%llu %s replay=cpu\n",
                       static_cast<unsigned long long>(sigma),
                       mpz_cmp_ui(den.v, 1) ? "nonunit_z=1" : "exceptional_map=1");
        mpz_set_si(x.v, -1);
        setup = ecm_edwards::construct(d, px, py, a24.v, factor, sigma, n, construction);
    }
    if (setup == Setup::Factor) return CurveKind::Factor;
    if (setup != Setup::Ready)
        return ecm_edwards::construction_is_degenerate(setup, factor, n)
            ? CurveKind::Degenerate : CurveKind::Failed;

    if (mpz_sgn(x.v) < 0) {
        mpz_ui_sub(den.v, 1, py);
        const auto chart = ecm_edwards::inverse(inv.v, den.v, n, factor);
        if (chart == Setup::Factor) return CurveKind::Factor;
        if (chart == Setup::Degenerate) return CurveKind::CompletedWithoutPoint;
        mpz_add_ui(x.v, py, 1);
        mpz_mul(x.v, x.v, inv.v);
        mpz_mod(x.v, x.v, n);
    }
    // The point with x=0 has order two and the Edwards scalar is even.
    if (mpz_sgn(x.v) == 0) return CurveKind::CompletedWithoutPoint;

    Integer one, aa, bb, e, x2, z2;
    const auto reduce = [&](mpz_ptr value) {
        if (construction) construction->reduce(value);
        else mpz_mod(value, value, n);
    };
    mpz_set_ui(one.v, 1);
    mpz_add_ui(aa.v, x.v, 1);
    mpz_mul(aa.v, aa.v, aa.v);
    reduce(aa.v);
    mpz_sub_ui(bb.v, x.v, 1);
    mpz_mul(bb.v, bb.v, bb.v);
    reduce(bb.v);
    mpz_sub(e.v, aa.v, bb.v);
    mpz_mul(x2.v, aa.v, bb.v);
    reduce(x2.v);
    mpz_mul(z2.v, a24.v, e.v);
    mpz_add(z2.v, z2.v, bb.v);
    mpz_mul(z2.v, z2.v, e.v);
    reduce(z2.v);
    const mpz_srcptr slots[] = {x.v, one.v, x2.v, z2.v, a24.v, x.v};
    for (unsigned slot = 0; slot < 6; ++slot)
        mpz_export(initial + slot * words, nullptr, -1, sizeof(uint32_t), 0, 0, slots[slot]);
    return CurveKind::Stage2Point;
}

void prepare_edwards_batch(mpz_srcptr n, const std::vector<uint64_t>& sigmas, unsigned words,
        const std::vector<uint32_t>& auxiliary, unsigned threads, bool use_prefix, unsigned carrier_bits, std::vector<CurveKind>& kinds, ecm_edwards::Array& factors,
        std::vector<uint32_t>& initial, uint32_t* seeds, bool speculative = false) {
    const unsigned curves = unsigned(sigmas.size());
    std::vector<unsigned> order(curves);
    std::iota(order.begin(), order.end(), 0u);
    if (use_prefix) {
        // Only traversal order changes. Every result is stored in its original lane.
        std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return sigmas[a] < sigmas[b]; });
    }
    const auto prepare = [&](unsigned worker) {
        std::optional<ecm_edwards::MersenneConstruction> scratch;
        if (use_prefix) scratch.emplace(n,carrier_bits);
        ecm_edwards::Integer d, px, py;
        for (unsigned j = curves * worker / threads; j < curves * (worker + 1) / threads; ++j) {
            if (speculative && ecm_stage1_stop_requested()) return;
            const unsigned i = order[j];
            ecm_edwards::MersenneConstruction::Jacobian point;
            const bool use_device_point = !auxiliary.empty();
            if (use_device_point) {
                mpz_ptr coords[] = {point.x.v,point.y.v,point.z.v};
                for (size_t slot=0; slot<3; ++slot) {
                    const auto* src = auxiliary.data()+(slot*curves+i)*words;
                    mpz_import(coords[slot],words,-1,sizeof(uint32_t),0,0,src);
                    mpz_mod(coords[slot],coords[slot],n);
                }

            }
            kinds[i] = prepare_edwards_lane(n, sigmas[i], factors[i],
                initial.data() + size_t(i) * 6 * words, words, d.v, px.v, py.v, scratch ? &*scratch : nullptr,
                use_device_point ? &point : nullptr);
            if (!seeds || kinds[i] != CurveKind::Stage2Point) continue;
            const mpz_srcptr seed_coords[] = {px.v, py.v, d.v};
            for (unsigned slot = 0; slot < 3; ++slot)
                mpz_export(seeds + (size_t(slot) * curves + i) * words,
                           nullptr, -1, sizeof(uint32_t), 0, 0, seed_coords[slot]);
        }
    };
    std::vector<std::future<void>> jobs;
    jobs.reserve(threads - 1);
    for (unsigned i = 1; i < threads; ++i)
        jobs.emplace_back(std::async(std::launch::async, prepare, i));
    prepare(0);
    for (auto& job : jobs) job.get();
}

// Run at most one host worker beside the active scalar loop. Only compact
// curve outputs survive this job; NAF encoding and table construction stay
// on the foreground path after the scheduler commits its next batch.
void start_curve_prefetch(CurvePrefetch& buffer, mpz_srcptr modulus, unsigned p, bool carrier,
                         unsigned words, Stage1NttOptions options, int device, size_t device_headroom) {
    const auto next = buffer.next;
    const size_t host_bytes = (240 + 18 * next.size()) * words * sizeof(uint32_t);
    if (ecm_stage1_stop_requested() || host_bytes > ecm_edwards_memory::ntt_host_budget()) return;
    buffer.predicted = next;
    try {
        buffer.pending = std::async(std::launch::async, [=]() mutable -> std::unique_ptr<PreparedCurves> {
            try {
                const auto started = Clock::now();
                const unsigned count = unsigned(next.size());
                size_t device_bytes = 0;
                std::vector<uint32_t> auxiliary;
                if (p >= (1u << 20) && *std::max_element(next.begin(), next.end()) >= (1ull << 20)) {
                    const auto mib = device_headroom / ecm_edwards_memory::mib;
                    if (!mib) return {};
                    options.memory_mib = unsigned(mib);
                    if (options.automatic) {
                        options.weighted = true;
                        options.length = options.tile = options.bpw = 0;
                    }
                    auxiliary = ecm_mersenne_ntt::auxiliary_points(device, p, next, options, nullptr, &device_bytes);
                }
                if (ecm_stage1_stop_requested()) return {};
                auto out = std::make_unique<PreparedCurves>(count, words);
                const bool fold = !carrier || (p >= 8192 && p - mpz_sizeinbase(modulus, 2) <= 256);
                prepare_edwards_batch(modulus, next, words, auxiliary, 1, fold, carrier ? p : 0,
                                      out->kinds, out->factors, out->initial, out->seeds.data(), true);
                if (ecm_stage1_stop_requested()) return {};
                ecm_mersenne_ntt::log_batch_span("prefetch_ready",started);
                ecm_ts_fprintf(stdout, "stage1_ntt_prefetch: ready curves=%u seed_and_ladder_bytes=%zu "
                               "host_estimate_bytes=%zu auxiliary_device_bytes=%zu elapsed_ms=%.3f\n", count,
                               (out->initial.size() + out->seeds.size()) * sizeof(uint32_t), host_bytes, device_bytes,milliseconds_since(started));
                return out;
            } catch (const std::exception& e) {
                ecm_ts_fprintf(stdout, "stage1_ntt_prefetch: skipped: %s\n", e.what());
                return {};
            }
        });
    } catch (const std::system_error& e) {
        ecm_ts_fprintf(stdout, "stage1_ntt_prefetch: skipped: %s\n", e.what());
    }
}

std::vector<uint32_t> build_host_naf_fallback(ecm_mersenne_ntt::Run& run, mpz_srcptr modulus,
        const std::vector<uint32_t>& seeds, unsigned curves, unsigned words, unsigned window, size_t dictionary_budget) {
    const size_t entries = size_t(1) << (window - 2);
    const size_t stride = size_t(curves) * words;
    const size_t table_words = 4 * entries * stride;
    const size_t scratch_bytes = (table_words + 8 * entries * words + stride) * sizeof(uint32_t);
    if (dictionary_budget && table_words * sizeof(uint32_t) > dictionary_budget)
        throw ecm_mersenne_ntt::BudgetError("Edwards NTT host fallback exceeds naf_memory_mb");
    if (scratch_bytes > ecm_edwards_memory::ntt_host_budget())
        throw ecm_mersenne_ntt::BudgetError("Edwards NTT host fallback exceeds available host memory");
    ecm_ts_fprintf(stdout, "Stage1 NTT: GPU dictionary workspace exceeds the VRAM budget, building on the host\n");

    std::vector<uint32_t> table(table_words);
    run.dictionary_status.resize(curves);
    ecm_edwards::Integer px, py, d, factor;
    for (unsigned i = 0; i < curves; ++i) {
        const auto* ptr = seeds.data() + size_t(i) * words;
        mpz_import(px.v, words, -1, sizeof(uint32_t), 0, 0, ptr);
        mpz_import(py.v, words, -1, sizeof(uint32_t), 0, 0, ptr + stride);
        mpz_import(d.v, words, -1, sizeof(uint32_t), 0, 0, ptr + 2 * stride);
        const auto status = ecm_edwards::ntt::affine_dictionary(table, curves, i, words, window,
                                                               px.v, py.v, d.v, modulus, factor.v);
        if (status == ecm_edwards::Setup::Ready) continue;
        if (status == ecm_edwards::Setup::Factor) {
            run.dictionary_status[i] = 1;
            run.dictionary_factors.resize(stride);
            mpz_export(run.dictionary_factors.data() + size_t(i) * words, nullptr,
                       -1, sizeof(uint32_t), 0, 0, factor.v);
        } else {
            run.dictionary_status[i] = 2;
        }
    }
    return table;
}
} // namespace

ComputationResult run_ntt_stage1(const Stage1Request& request, ExecutionOptions& opt,
                                 ecm_stage1_checkpoint::Context& checkpoint) {
    ComputationResult result;
    auto phase = FailureSource::Preparation;
    try {
        const auto compute_started = Clock::now();
        const auto N = request.modulus;
        const size_t p = ntt_arithmetic_bits(N, request.options);
        const auto target_bits = mpz_sizeinbase(N, 2);
        const bool carrier = p && mpz_scan0(N, 0) != p;
        const unsigned curves = request.curves;
        const bool edwards = opt.execution.selection.curve == CurveModel::Edwards;
        const unsigned requested_window = edwards ? unsigned(opt.execution.naf_w) : 0;
        if (requested_window && (requested_window < 2 || requested_window > 12))
            throw std::invalid_argument("Edwards NTT NAF requires a window from 2 to 12");
        if (!curves || curves > stage1_ntt_max_curves || (!edwards && opt.execution.selection.gpu_param != 0) ||
            !p)
            throw std::invalid_argument(
                "NTT requires Edwards or Montgomery param0, a legal Mersenne carrier, and 1-" +
                std::to_string(stage1_ntt_max_curves) + " curves");
        if (carrier)
            ecm_ts_fprintf(stdout, "stage1_ntt_carrier: target_bits=%zu arithmetic_bits=%zu exponent=%zu\n",
                           target_bits, p, p);
        ecm_edwards::Integer scalar;
        const unsigned torsion = edwards ? 48u :
            opt.execution.selection.scalar_construction == ScalarConstruction::PrimePowerLcmTimes12
                ? 12u : 1u;
        if (!compute_batch_s(scalar.v, request.b1, torsion, opt.persistence.exponent_cache_dir))
            throw std::runtime_error("cannot compute the Stage1 scalar");
        const size_t scalar_bits = mpz_sizeinbase(scalar.v, 2);
        if (scalar_bits > UINT32_MAX - 63)
            throw std::invalid_argument("Stage1 scalar exceeds the NTT cursor range");
        const unsigned binary_count = unsigned(scalar_bits - 1);
        const unsigned words = unsigned((p + 31) / 32);
        const size_t host_limit = edwards ? ecm_edwards_memory::ntt_host_budget() : SIZE_MAX;
        const size_t encoding_bytes = (scalar_bits + 1) * sizeof(int16_t) + edwards_naf_bytes(scalar_bits, true);
        const auto prepare_naf = [&](unsigned window) {
            if (encoding_bytes > host_limit)
                throw ecm_mersenne_ntt::BudgetError("Edwards NTT NAF encoding exceeds available host memory");
            return edwards_prepare_naf(scalar.v, uint64_t(request.b1), int(window), true);
        };

        Snapshot snapshot;
        std::shared_ptr<const EdwardsNafPlan> naf_plan;
        double restore_encoding_ms = 0;
        phase = FailureSource::Persistence;
        checkpoint.validate_pending([&](const auto& records, auto& sigmas) {
            if (records.size() != curves || sigmas.size() != curves ||
                (!edwards && !montgomery_sigma_batch_fits(sigmas.front(), curves, 0)))
                return false;
            for (unsigned i = 0; i < curves; ++i)
                if ((edwards ? sigmas[i] == 0 : sigmas[i] != sigmas.front() + i) ||
                    (i && !records[i].empty()))
                    return false;
            Snapshot candidate;
            if (!read_snapshot(records.front(), unsigned(p), curves, edwards, candidate) ||
                candidate.count > scalar_bits + 1)
                return false;
            const auto encode_started = Clock::now();
            auto plan = candidate.naf_window ? prepare_naf(candidate.naf_window) : nullptr;
            if (candidate.count != (plan ? plan->size() : binary_count))
                return false;
            if (plan) restore_encoding_ms = milliseconds_since(encode_started);
            naf_plan = std::move(plan);
            snapshot = std::move(candidate);
            return true;
        });
        bool has_checkpoint = snapshot.lanes.any();
        const unsigned naf_window = has_checkpoint ? snapshot.naf_window : requested_window;
        if (naf_window != requested_window)
            ecm_ts_fprintf(stdout, "stage1_ntt_naf_resume: saved_window=%u configured_window=%u cursor=%u\n",
                           naf_window, requested_window, snapshot.cursor);
        phase = FailureSource::Preparation;

        const size_t entries = naf_window ? size_t(1) << (naf_window - 2) : 0;
        const bool gpu_dictionary = ecm_edwards::ntt::production_device_dictionary(
            p, curves, naf_window, opt.execution.ntt.naf_gpu_min_bits);
        const bool automatic_dictionary = !std::getenv("ECM_EDWARDS_NTT_DICTIONARY");
        const size_t budget = size_t(opt.execution.naf_memory_mb) * ecm_edwards_memory::mb;
        const size_t dictionary_slots = !naf_window ? 0 : gpu_dictionary ? 3 : 4 * entries;
        const size_t host_dictionary_words = dictionary_slots * curves * words;
        const size_t host_seed_words = naf_window && !gpu_dictionary ? 3ull * curves * words : 0;
        const size_t host_required = naf_window ? encoding_bytes +
            ecm_edwards::ntt::preparation_bytes(gpu_dictionary, entries, words, curves) :
            6ull * curves * words * sizeof(uint32_t);
        std::vector<int16_t> digits;
        double encoding_ms = restore_encoding_ms, curve_setup_ms = 0, host_points_ms = 0;
        unsigned host_dictionary_curves = 0;
        const auto naf_started = Clock::now();
        if (naf_window) {
            const size_t table_bytes = host_dictionary_words * sizeof(uint32_t);
            if (!gpu_dictionary && budget && table_bytes > budget)
                throw ecm_mersenne_ntt::BudgetError("Edwards NTT host dictionary exceeds naf_memory_mb");
            if (host_required > host_limit)
                throw ecm_mersenne_ntt::BudgetError("Edwards NTT NAF preparation exceeds available host memory");
            const auto encode_started = Clock::now();
            if (!naf_plan) naf_plan = prepare_naf(naf_window);
            digits.resize(naf_plan->size());
            for (size_t i = 0; i < digits.size(); ++i)
                digits[i] = int16_t((*naf_plan)[digits.size() - 1 - i]);
            naf_plan.reset();
            encoding_ms += milliseconds_since(encode_started);
        }
        const unsigned count = naf_window ? unsigned(digits.size()) : binary_count;
        std::vector<uint32_t> packed((scalar_bits + 31) / 32);
        mpz_export(packed.data(), nullptr, -1, sizeof(uint32_t), 0, 0, scalar.v);

        phase = FailureSource::Persistence;
        uint64_t saved_sigma = 0;
        checkpoint.saved_sigma(0, saved_sigma);
        std::vector<uint64_t> sigmas(curves);
        if (edwards) {
            CurveSigmas selection(opt.identity, CurveModel::Edwards);
            for (unsigned i = 0; i < curves; ++i) {
                uint64_t saved = 0;
                checkpoint.saved_sigma(i, saved);
                sigmas[i] = selection.next(saved, i);
            }
        } else {
            const uint64_t first = select_batch_sigma(saved_sigma, opt.identity.sigma_fixed,
                                                        opt.identity.fixed_sigma64, curves, 0);
            for (unsigned i = 0; i < curves; ++i) sigmas[i] = first + i;
        }
        phase = FailureSource::Preparation;
        result.curves.resize(curves);
        ecm_edwards::Array factors(curves);
        std::vector<uint32_t> lanes_initial(size_t(curves) * 6 * words);
        std::vector<CurveKind> kinds(curves, CurveKind::Stage2Point);
        std::vector<uint32_t> dictionary(host_dictionary_words);
        std::vector<unsigned> fallback;
        ecm_mersenne_ntt::TransformWorkspace workspace;
        const auto dict_started = Clock::now();
        if (edwards) {
            const unsigned available = opt.execution.worker_cpu_count ? opt.execution.worker_cpu_count :
                std::max(1u, std::thread::hardware_concurrency());
            const unsigned automatic = (p >= 4096 && curves >= 8) || (carrier && p >= (1u << 20)) ? 4 : 1;
            const unsigned requested = opt.execution.gpu_init_threads ? opt.execution.gpu_init_threads : automatic;
            // Reserve complete prefix snapshots plus ordinary GMP temporaries per worker.
            const size_t scratch_bytes = size_t(3 * 64 + 48) * words * sizeof(uint32_t);
            const size_t affordable = host_required < host_limit ? (host_limit - host_required) / scratch_bytes : 0;
            const unsigned workers = unsigned(std::max<size_t>(1, std::min<size_t>(
                std::min({curves, available, requested}), affordable)));
            std::vector<uint32_t> host_seeds(host_seed_words);
            auto* seeds = !naf_window ? nullptr : gpu_dictionary ? dictionary.data() : host_seeds.data();
            const auto curve_started = Clock::now();
            const bool prefix = affordable != 0 && (!carrier || (p >= 8192 && p - target_bits <= 256));
            std::unique_ptr<PreparedCurves> ready;
            auto* prefetch = opt.curve_prefetch;
            if (prefetch && prefetch->pending.valid()) {
                auto prepared = prefetch->pending.get();
                if (prefetch->predicted == sigmas) ready = std::move(prepared);
            }
            if (ready) {
                lanes_initial.swap(ready->initial);
                kinds.swap(ready->kinds);
                for (unsigned i = 0; i < curves; ++i) mpz_swap(factors[i], ready->factors[i]);
                if (seeds) std::copy(ready->seeds.begin(), ready->seeds.end(), seeds);
                ecm_ts_fprintf(stdout, "stage1_ntt_prefetch: reused curves=%u\n", curves);
            } else {
                std::vector<uint32_t> auxiliary;
                const size_t aux_bytes = 9ull*curves*words*sizeof(uint32_t);
                if (affordable && aux_bytes <= (host_limit-host_required) - workers*scratch_bytes &&
                    p >= (1u << 20) && std::any_of(sigmas.begin(),sigmas.end(),
                        [](uint64_t sigma){return sigma >= (1ull << 20);})) {
                    auto options = opt.execution.ntt;
                    if (options.automatic) {
                        // The scalar-loop plan may not have been selected yet.
                        options.weighted = true;
                        options.length = options.tile = options.bpw = 0;
                    }
                    try {
                        auxiliary = ecm_mersenne_ntt::auxiliary_points(opt.execution.device_index,unsigned(p),sigmas,options,&workspace);
                        ecm_ts_fprintf(stdout,"stage1_ntt_auxiliary: backend=gpu curves=%u ms=%.3f\n",
                                       curves,milliseconds_since(curve_started));
                    } catch (const ecm_mersenne_ntt::BudgetError&) {
                        workspace.reset();
                        ecm_ts_fprintf(stdout,"stage1_ntt_auxiliary: workspace exceeds budget, replay=cpu\n");
                    }
                }
                prepare_edwards_batch(N, sigmas, words, auxiliary, workers, prefix, carrier ? unsigned(p) : 0,
                                      kinds, factors, lanes_initial, seeds);
            }
            curve_setup_ms = milliseconds_since(curve_started);
            ecm_mersenne_ntt::log_batch_span(ready ? "curve_reuse" : "curve_setup",curve_started);
            ecm_ts_fprintf(stdout, "stage1_ntt_edwards_setup: threads=%u ms=%.3f prefix=%u fold=%u\n",
                workers, curve_setup_ms, unsigned(prefix), unsigned(prefix && p >= 8192));
            if (ecm_stage1_stop_requested()) {
                result.stop();
                return result;
            }
            ecm_edwards::Integer d, px, py;
            for (unsigned i = 0; i < curves; ++i) {
                if (!naf_window || kinds[i] != CurveKind::Stage2Point) continue;
                if (gpu_dictionary && (!has_checkpoint || snapshot.lanes[i])) continue;
                // Earlier snapshots could exclude lanes classified by the
                // host dictionary before entering the device loop.
                const auto points_started = Clock::now();
                const auto* seed = seeds + size_t(i) * words;
                const size_t stride = size_t(curves) * words;
                mpz_import(px.v, words, -1, sizeof(uint32_t), 0, 0, seed);
                mpz_import(py.v, words, -1, sizeof(uint32_t), 0, 0, seed + stride);
                mpz_import(d.v, words, -1, sizeof(uint32_t), 0, 0, seed + 2 * stride);
                std::vector<uint32_t> legacy_table;
                if (gpu_dictionary) {
                    const size_t table_words = 4 * entries * words;
                    if ((budget && table_words * sizeof(uint32_t) > budget) ||
                        3 * table_words * sizeof(uint32_t) > ecm_edwards_memory::ntt_host_budget())
                        throw ecm_mersenne_ntt::BudgetError("Edwards checkpoint dictionary exceeds host budget");
                    legacy_table.resize(table_words);
                }
                auto& table = gpu_dictionary ? legacy_table : dictionary;
                const auto status = ecm_edwards::ntt::affine_dictionary(table, gpu_dictionary ? 1 : curves,
                    gpu_dictionary ? 0 : i, words,
                    naf_window, px.v, py.v, d.v, N, factors[i]);
                host_points_ms += milliseconds_since(points_started);
                ++host_dictionary_curves;
                if (gpu_dictionary && status == ecm_edwards::Setup::Ready)
                    throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
                        "NTT checkpoint excludes a usable Edwards dictionary lane");
                if (status == ecm_edwards::Setup::Factor) kinds[i] = CurveKind::Factor;
                if (status == ecm_edwards::Setup::Degenerate) fallback.push_back(i);
            }
        } else {
            std::vector<int> hits(curves, ECM_NO_FACTOR_FOUND);
            uint32_t degenerate = curves;
            ecm_cuda_mont_host::ConstructionResults construction{
                factors.data(), hits.data(), nullptr, nullptr, &degenerate};
            size_t bytes = 0;
            const std::unique_ptr<uint32_t, decltype(&std::free)> initial_curves(
                ecm_cuda_mont_host::set_p_2p_suyama(N, curves, sigmas.front(), words * 32,
                                                    &bytes, &construction), std::free);
            if (degenerate < curves)
                throw std::runtime_error("degenerate param0 construction");
            for (unsigned i = 0; i < curves; ++i) {
                if (hits[i] == ECM_FACTOR_FOUND_STEP1) kinds[i] = CurveKind::Factor;
                constexpr unsigned slot_map[] = {3, 4, 5, 6, 1, 2};
                for (unsigned slot = 0; slot < 6; ++slot)
                    std::copy_n(initial_curves.get() + (7 * i + slot_map[slot]) * words,
                                words, lanes_initial.data() + (6 * i + slot) * words);
            }
        }
        std::vector<unsigned> active;
        ActiveLanes lanes;
        int legacy = ECM_NO_FACTOR_FOUND;
        auto edwards_identity = [&](unsigned lane) {
            ecm_ts_fprintf(stdout, "Edwards identity: sigma=%llu, no Stage2 point\n",
                           static_cast<unsigned long long>(sigmas[lane]));
            result.set_curve(lane, CurveKind::CompletedWithoutPoint, nullptr, sigmas[lane]);
        };
        for (unsigned i = 0; i < curves; ++i) {
            if (kinds[i] == CurveKind::Failed)
                throw std::runtime_error("invalid Atkin-Morain construction on sigma " +
                                         std::to_string(sigmas[i]));
            if (kinds[i] == CurveKind::Degenerate) {
                ecm_ts_fprintf(stdout, "NTT Edwards skipped degenerate construction: curve=%u sigma=%llu gcd=N\n",
                               i, static_cast<unsigned long long>(sigmas[i]));
                result.set_curve(i, CurveKind::Degenerate, nullptr, sigmas[i]);
            } else if (kinds[i] == CurveKind::Factor) {
                result.set_curve(i, CurveKind::Factor, factors.data()[i], sigmas[i]);
                legacy = ECM_FACTOR_FOUND_STEP1;
            } else if (kinds[i] == CurveKind::CompletedWithoutPoint) {
                edwards_identity(i);
            } else if (std::find(fallback.begin(), fallback.end(), i) == fallback.end()) {
                active.push_back(i);
                lanes.set(i);
            }
        }
        // The host builder can now prove a factor for a lane present in a
        // GPU-built snapshot. Removing that lane is safe; adding one is not.
        const bool compact_snapshot = has_checkpoint && naf_window && !gpu_dictionary &&
            snapshot.lanes != lanes && (lanes & ~snapshot.lanes).none();
        if (has_checkpoint && snapshot.lanes != lanes && !compact_snapshot)
            throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
                                               "NTT checkpoint active curve mask differs");

        struct Observation {
            ecm_mersenne_ntt::Plan plan;
            ecm_mersenne_ntt::Run run;
            unsigned valid, batch;
            double seconds;
        };
        std::optional<Observation> observation;
        if (!active.empty()) {
            const auto valid = unsigned(active.size());
            const size_t stride = size_t(words) * valid;
            std::vector<uint32_t> initial((naf_window ? 3 : 6) * stride);
            if (naf_window) {
                for (size_t i = 0; i < active.size(); ++i) {
                    initial[stride + i * words] = 1;
                    initial[2 * stride + i * words] = 1;
                }
                if (active.size() != curves) {
                    for (size_t slot = 0; slot < dictionary_slots; ++slot)
                        for (size_t i = 0; i < active.size(); ++i) {
                            const size_t src = slot * curves + active[i], dst = slot * active.size() + i;
                            if (src != dst)
                                std::copy_n(dictionary.data() + src * words, words, dictionary.data() + dst * words);
                        }
                    dictionary.resize(dictionary_slots * stride);
                }
                ecm_ts_fprintf(stdout, "stage1_ntt_edwards_naf: window=%u columns=%u additions=%zu host_dictionary_ms=%.3f\n",
                    naf_window, count, size_t(std::count_if(digits.begin(), digits.end(), [](int d) { return d != 0; })),
                    1000 * std::chrono::duration<double>(Clock::now() - dict_started).count());
            } else {
                for (unsigned slot = 0; slot < 6; ++slot)
                    for (size_t i = 0; i < active.size(); ++i)
                        std::copy_n(lanes_initial.data() + (6 * active[i] + slot) * words,
                                    words, initial.data() + slot * stride + i * words);
            }
            if (compact_snapshot) {
                const size_t saved_stride = snapshot.lanes.count() * words;
                for (size_t lane = 0, saved_index = 0, active_index = 0; lane < curves; ++lane) {
                    if (!snapshot.lanes[lane]) continue;
                    if (lanes[lane]) {
                        for (unsigned slot = 0; slot < 3; ++slot)
                            std::copy_n(snapshot.points.data() + slot * saved_stride + saved_index * words,
                                        words, initial.data() + slot * stride + active_index * words);
                        ++active_index;
                    }
                    ++saved_index;
                }
            } else if (has_checkpoint) {
                std::copy(snapshot.points.begin(), snapshot.points.end(), initial.begin());
            }

            const unsigned batch = valid;

            ecm_mersenne_ntt::Control control;
            ecm_mersenne_ntt::Plan executed_plan{};
            ecm_mersenne_ntt::BatchMemory memory;
            control.resume = snapshot.cursor;
            control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
                executed_plan = plan;
                log_ntt_execution(opt, p, batch, naf_window, control.resume, plan, memory, valid, target_bits, uint64_t(request.b1), count);
                ecm_ts_fprintf(stdout, "stage1_ntt_plan: device=%d p=%zu curves=%zu length=%u bpw=%u digits=%u "
                    "forward_passes=%u inverse_passes=%u forward_layers=%s inverse_layers=%s "
                    "tile_layers=%u tile_elements=%u tile_threads=%u tile_smem_bytes=%u "
                    "inverse_tile_threads=%u inverse_tile_smem_bytes=%u "
                    "outer_max_layers=%u outer_threads=%u warp_tail=%u graph_bits=%u "
                    "profile=%u source=%s resume=%u bits=%u convolution=%s curve=%s dataflow=%u "
                    "bpw_avg=%.6f bpw_max=%u U=%llu q=%llu\n",
                    opt.execution.device_index, p, size_t(batch), plan.length, plan.bpw, plan.digits,
                    plan.passes, plan.passes, plan.forward.c_str(), plan.inverse.c_str(),
                    plan.tile, 1u << plan.tile, plan.tile_threads, plan.tile_smem_bytes,
                    plan.inverse_tile_threads, plan.inverse_tile_smem_bytes,
                    plan.outer_max, plan.outer_threads, unsigned(plan.warp_tail),
                    plan.window, opt.execution.ntt.profile, plan.source, control.resume, count,
                    plan.weighted ? "weighted" : "linear", edwards ? "edwards" : "mont0", plan.dataflow,
                    double(p) / plan.digits, plan.bpw, plan.coefficient_bound, plan.field_modulus);
            };
            auto last_save = Clock::now(), last_progress = last_save;
            auto progress_started = last_save;
            bool first_poll = true;
            auto rate_clock=last_save;unsigned rate_cursor=control.resume;
            ecm_stage1_gpu_progress::SpeedWindow iter_speed;
            if (workspace.size) control.workspace = &workspace;
            Clock::time_point loop_started;
            control.iterations_started = [&](size_t headroom) {
                loop_started = Clock::now();
                if (opt.curve_prefetch && !opt.curve_prefetch->next.empty())
                    start_curve_prefetch(*opt.curve_prefetch, N, unsigned(p), carrier, words,
                                         opt.execution.ntt, opt.execution.device_index, headroom);
            };
            const auto report_progress = [&](unsigned cursor, Clock::time_point now, bool stop) {
                if (first_poll)progress_started=now;
                else if(cursor>=rate_cursor)iter_speed.add(cursor-rate_cursor, std::chrono::duration<double,std::milli>(now-rate_clock).count());
                rate_clock=now;rate_cursor=cursor;
                if (first_poll || cursor == count || stop || now - last_progress >= std::chrono::seconds(2)) {
                    const double eta = cursor == count ? 0 : iter_speed.speed() > 0 ? iter_speed.remaining(count, cursor) : -1;
                    ecm_ts_fprintf(stdout, "stage1_ntt_progress: %s=%u/%u (%.1f%%) elapsed=%.3f iter_per_s=%.3f eta=%.3f\n",
                                   naf_window ? "columns" : "bits", cursor, count, 100.0 * cursor / count,
                                   std::chrono::duration<double>(now-progress_started).count(), iter_speed.speed()*1000, eta);
                    last_progress = now;
                }
                first_poll=false;
            };
            control.poll = [&](unsigned cursor) {
                const auto now = Clock::now();
                const bool stop = ecm_stage1_stop_requested();
                report_progress(cursor, now, stop);
                const auto elapsed = std::chrono::duration<double, std::milli>(now - last_save).count();
                const bool due = opt.persistence.ckpt_ms && elapsed >= opt.persistence.ckpt_ms;
                return ecm_mersenne_ntt::Control::Action{!checkpoint.path().empty() && (stop || due), stop};
            };
            control.checkpoint = [&](unsigned cursor, const auto& points) {
                const auto payload = encode_snapshot(count, cursor, curves, lanes, naf_window, points);
                const bool ok = has_checkpoint ? checkpoint.write_blob(ecm_stage1_curve_record(0), payload)
                                               : checkpoint.import_payload(payload, sigmas);
                if (!ok)
                    throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::IoFailure,
                                                       "cannot persist NTT Stage1 checkpoint");
                has_checkpoint = true;
                last_save = Clock::now();rate_clock=last_save;
            };

            phase = FailureSource::Device;
            const auto plan_started = Clock::now();
            auto* plan_cache = opt.ntt_plan_cache;
            const bool reuse = plan_cache && !has_checkpoint && plan_cache->p == p &&
                plan_cache->batch >= batch && plan_cache->window == naf_window &&
                ecm_mersenne_ntt::reuse_tuned_plan(request,unsigned(p),batch,naf_window,plan_cache->options,memory);
            if (plan_cache) plan_cache->reused = reuse;
            const auto selected = reuse ? plan_cache->options :
                ecm_mersenne_ntt::select_tuned_plan(request,unsigned(p),batch,opt.execution.ntt,
                    naf_window,opt.execution.ntt.observe_batch && !has_checkpoint,memory,
                    plan_cache && !has_checkpoint ? plan_cache->preferred_batch : 0,
                    has_checkpoint && naf_window ? &digits : nullptr);
            const auto plan_ms = milliseconds_since(plan_started);
            ecm_mersenne_ntt::log_batch_span(reuse ? "plan_reuse" : "select_plan",plan_started);
            ecm_ts_fprintf(stdout,"stage1_ntt_plan_timing: ms=%.3f reused=%u\n",plan_ms,unsigned(reuse));
            if (ecm_stage1_stop_requested()) {
                result.stop();
                return result;
            }
            ecm_mersenne_ntt::EdwardsNaf naf{naf_window, digits, dictionary, budget, gpu_dictionary, {}};
            if (carrier) naf.check_modulus = N;
            if (gpu_dictionary && automatic_dictionary)
                naf.host_fallback = [&](ecm_mersenne_ntt::Run& run) {
                    return build_host_naf_fallback(run, N, dictionary, batch, words, naf_window, budget);
                };
            const double host_init_ms = milliseconds_since(naf_started) + restore_encoding_ms;
            const auto cuda_started = Clock::now();
            auto run = naf_window ? ecm_mersenne_ntt::edwards_naf(opt.execution.device_index, unsigned(p),
                 batch, initial, naf, selected, control)
                : ecm_mersenne_ntt::ladder(opt.execution.device_index, unsigned(p),
                    batch, initial, packed, count, selected, control);
            const auto cuda_finished = Clock::now();
            if (loop_started != Clock::time_point{}) {
                ecm_mersenne_ntt::log_batch_span("cuda_init",cuda_started,loop_started);
                ecm_mersenne_ntt::log_batch_span("scalar_loop",loop_started,cuda_finished);
            }
            // The ladder polls before launching a slice, so it never polls the final cursor.
            if (!run.stopped && run.completed_bits == count)
                report_progress(count, Clock::now(), false);
            workspace.reset();
            ecm_ts_fprintf(stdout, "stage1_ntt_memory: owned_bytes=%zu observed_bytes=%zu budget_bytes=%zu\n",
                run.owned_bytes, run.observed_bytes, run.budget_bytes);
            if (naf_window) {
                const double init_ms = host_init_ms + 1000 * run.setup_seconds;
                ecm_ts_fprintf(stdout, "stage1_ntt_edwards_dictionary: gpu_ms=%.3f backend=%s root_inverse_ms=%.3f\n",
                    run.dictionary_gpu_ms, run.dictionary_built_on_gpu ? "gpu" : "host", run.dictionary_inverse_ms);
                ecm_ts_fprintf(stdout, "stage1_ntt_naf_init: ms=%.3f encoding_bytes=%zu host_dictionary_bytes=%zu "
                    "dictionary_spectra_bytes=%zu dictionary_workspace_bytes=%zu\n",
                    init_ms,
                    edwards_naf_bytes(scalar_bits, true) + digits.size() * sizeof(int16_t),
                    (host_dictionary_words + host_seed_words) * sizeof(uint32_t) + run.dictionary_fallback_host_bytes,
                    run.dictionary_spectra_bytes, run.dictionary_workspace_bytes);
                const double all_host_points_ms = host_points_ms + run.dictionary_host_ms;
                ecm_ts_fprintf(stdout, "stage1_ntt_naf_init_parts: encoding_ms=%.3f curve_setup_ms=%.3f "
                    "host_points_ms=%.3f device_dictionary_ms=%.3f other_ms=%.3f "
                    "entries_per_curve=%zu host_dictionary_curves=%zu device_dictionary_curves=%zu\n",
                    encoding_ms, curve_setup_ms, all_host_points_ms, run.dictionary_device_wall_ms,
                    init_ms - encoding_ms - curve_setup_ms - all_host_points_ms - run.dictionary_device_wall_ms,
                    entries, size_t(host_dictionary_curves) +
                        (run.dictionary_fallback_host_bytes ? batch : 0), size_t(batch));
            }
            ecm_ts_fprintf(stdout, "stage1_ntt_timing: %s=%u device_ms=%.3f wall_ms=%.3f "
                "setup_ms=%.3f readback_ms=%.3f stopped=%u\n",
                naf_window ? "columns" : "bits", run.completed_bits - control.resume, run.device_ms.front(), run.wall_ms.front(),
                1000 * run.setup_seconds, 1000 * run.readback_seconds, unsigned(run.stopped));
            if (run.stopped) {
                result.stop();
                return result;
            }

            phase = FailureSource::Execution;
            ecm_edwards::Array numerators(active.size()), denominators(active.size());
            std::vector<unsigned char> invertible(active.size());
            const auto normalize = [&] {
                const auto started = Clock::now();
                ecm_mersenne_ntt::for_host_lanes(unsigned(p),active.size(),[&](size_t i) {
                    if (!run.dictionary_status.empty() && run.dictionary_status[i]) return;
                    const auto x = numerators[i], z = denominators[i];
                    ecm_edwards::Integer y, inverse;
                    mpz_import(x, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + i * words);
                    mpz_import(z, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + stride + i * words);
                    if (naf_window) {
                        mpz_swap(y.v, z);
                        mpz_import(z, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + 2 * stride + i * words);
                        mpz_add(x, z, y.v);
                        mpz_sub(z, z, y.v);
                    }
                    if (!mpz_invert(inverse.v,z,N)) return;
                    mpz_mul(x,x,inverse.v);
                    mpz_mod(x,x,N);
                    invertible[i] = 1;
                });
                ecm_mersenne_ntt::log_batch_span("normalize",started);
                return milliseconds_since(started);
            };
            const bool save_final = opt.persistence.ckpt_ms && !checkpoint.path().empty() && control.resume != count;
            const bool overlap = save_final && p >= (1u << 20) && active.size() > 1;
            std::future<double> normalization;
            if (overlap)
                normalization = std::async(std::launch::async,normalize);
            const auto checkpoint_started = Clock::now();
            if (overlap) {
                control.checkpoint(count, run.points);
                ecm_mersenne_ntt::log_batch_span("final_checkpoint",checkpoint_started);
            }
            double checkpoint_ms = milliseconds_since(checkpoint_started);
            const double normalize_ms = normalization.valid() ? normalization.get() : normalize();
            const auto classify_started = Clock::now();
            ecm_edwards::Integer inv;
            for (size_t i = 0; i < active.size(); ++i) {
                const unsigned lane = active[i];
                if (!run.dictionary_status.empty() && run.dictionary_status[i]) {
                    if (run.dictionary_status[i]==1) {
                        mpz_import(inv.v, words, -1, sizeof(uint32_t), 0, 0,
                                   run.dictionary_factors.data()+i*words);
                        result.set_curve(lane, CurveKind::Factor, inv.v, sigmas[lane]);
                        legacy = ECM_FACTOR_FOUND_STEP1;
                    } else {
                        fallback.push_back(lane);
                    }
                    continue;
                }
                const auto x = numerators[i], z = denominators[i];
                if (!invertible[i]) {
                    mpz_gcd(inv.v, z, N);
                    if (mpz_cmp(inv.v, N) == 0) {
                        if (edwards) {
                            if (naf_window) fallback.push_back(lane);
                            else edwards_identity(lane);
                            continue;
                        }
                        result.set_curve(lane, CurveKind::Degenerate, nullptr, sigmas[lane]);
                        throw std::runtime_error("degenerate NTT endpoint on sigma " + std::to_string(sigmas[lane]));
                    }
                    result.set_curve(lane, CurveKind::Factor, inv.v, sigmas[lane]);
                    if (edwards) result.curves[lane].save_factor_endpoint(N, x, z);
                    legacy = ECM_FACTOR_FOUND_STEP1;
                } else {
                    result.set_curve(lane, CurveKind::Stage2Point, x, sigmas[lane]);
                }
            }
            const double classify_ms = milliseconds_since(classify_started);
            if (save_final && !overlap) {
                const auto started = Clock::now();
                control.checkpoint(count, run.points);
                checkpoint_ms = milliseconds_since(started);
                ecm_mersenne_ntt::log_batch_span("final_checkpoint",started);
            }
            ecm_ts_fprintf(stdout,"stage1_ntt_finish: normalize_ms=%.3f classify_ms=%.3f checkpoint_ms=%.3f overlap=%u\n",
                           normalize_ms,classify_ms,checkpoint_ms,
                           unsigned(overlap));
            if (plan_cache && naf_window == requested_window && valid == curves && fallback.empty() &&
                std::none_of(run.dictionary_status.begin(),run.dictionary_status.end(),
                             [](unsigned status) { return status != 0; })) {
                plan_cache->p = unsigned(p);
                plan_cache->batch = batch;
                plan_cache->window = naf_window;
                plan_cache->options = selected;
                plan_cache->options.automatic = false;
                plan_cache->options.length = executed_plan.length;
                plan_cache->options.bpw = executed_plan.bpw;
            }
            if (opt.execution.ntt.observe_batch && naf_window == requested_window && !control.resume &&
                fallback.empty() && valid == curves &&
                std::none_of(run.dictionary_status.begin(), run.dictionary_status.end(),
                             [](unsigned status) { return status != 0; }))
                observation = Observation{std::move(executed_plan), std::move(run), valid, batch,
                                          (milliseconds_since(compute_started) - plan_ms) / 1000};
        }
        auto replay_ntt = opt.execution.ntt;
        if (!fallback.empty() && replay_ntt.weighted && replay_ntt.length &&
            ecm_mersenne_ntt::resolve_packing(unsigned(p), replay_ntt, true,
                ecm_mersenne_ntt::WeightedBound::UnsignedProducts).needs_unsigned_bound) {
            // An explicit native Edwards plan can exceed the x-only packing bound.
            replay_ntt.length = 0;
        }
        for (unsigned lane : fallback) {
            ecm_ts_fprintf(stdout, "stage1_ntt_edwards_naf: x-only replay sigma=%llu\n",
                           static_cast<unsigned long long>(sigmas[lane]));
            const auto* begin = lanes_initial.data() + size_t(6) * lane * words;
            const std::vector<uint32_t> initial(begin, begin + 6 * words);
            ecm_mersenne_ntt::Control control;
            ecm_mersenne_ntt::BatchMemory memory;
            control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
                log_ntt_execution(opt, p, 1, 0, control.resume, plan, memory, 0, target_bits, uint64_t(request.b1), binary_count);
            };
            control.poll = [](unsigned) {
                return ecm_mersenne_ntt::Control::Action{false, ecm_stage1_stop_requested()};
            };
            const auto selected = ecm_mersenne_ntt::select_tuned_plan(request, unsigned(p), 1,
                replay_ntt, 0, false, memory);
            const auto run = ecm_mersenne_ntt::ladder(opt.execution.device_index, unsigned(p), 1,
                initial, packed, binary_count, selected, control);
            if (run.stopped) {
                result.stop();
                return result;
            }
            ecm_edwards::Integer x, z, inv;
            mpz_import(x.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data());
            mpz_import(z.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + words);
            const auto status = ecm_edwards::inverse(inv.v, z.v, N, factors[lane]);
            if (status == ecm_edwards::Setup::Ready) {
                mpz_mul(x.v, x.v, inv.v);
                mpz_mod(x.v, x.v, N);
                result.set_curve(lane, CurveKind::Stage2Point, x.v, sigmas[lane]);
            } else if (status == ecm_edwards::Setup::Factor) {
                result.set_curve(lane, CurveKind::Factor, factors[lane], sigmas[lane]);
                result.curves[lane].save_factor_endpoint(N, x.v, z.v);
                legacy = ECM_FACTOR_FOUND_STEP1;
            } else {
                edwards_identity(lane);
            }
        }
        result.legacy_code = legacy;
        result.completion = ComputationResult::Completion::Completed;
        result.validate(N);
        if (observation)
            ecm_mersenne_ntt::record_production_sample(request, observation->plan, observation->run,
                                                      observation->valid, observation->batch, observation->seconds);
    } catch (const ecm_stage1_checkpoint::Error& error) {
        result.fail(FailureSource::Persistence);
        ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT checkpoint: %s\n", error.what());
    } catch (const ecm_mersenne_ntt::BudgetError& error) {
        if (ecm_stage1_stop_requested()) {
            result.stop();
        } else {
            result.fail(phase);
            result.retry_smaller_ntt_batch = true;
            ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT: %s\n", error.what());
        }
    } catch (const std::exception& error) {
        if (ecm_stage1_stop_requested()) {
            result.stop();
        } else {
            result.fail(phase);
            ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT: %s\n", error.what());
        }
    }
    return result;
}
} // namespace ecm_cuda_stage1
