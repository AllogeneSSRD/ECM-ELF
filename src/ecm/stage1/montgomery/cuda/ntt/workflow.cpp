#include "workflow.h"
#include "ladder.h"
#include "tune.h"
#include "../host_math.h"
#include "../../../execution_details.h"
#include "../../../sigma.h"
#include "../../../../curve/edwards.h"
#include "../../../edwards/ntt_dictionary.h"
#include "../../../edwards/cpu/ecm_edwards_cpu.h"
#include "../../../edwards/memory.h"
#include "../../../../../io/stage1_binary.h"
#include "../../../../../io/stage1_checkpoint.h"
#include "ecm/integer.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace ecm_cuda_stage1 {
namespace {
using namespace ecm_stage1;
using Clock = std::chrono::steady_clock;
using ActiveLanes = std::bitset<stage1_ntt_max_curves>;

void log_ntt_execution(const ExecutionOptions& opt, size_t modulus_bits, size_t batch,
                       unsigned naf_window, unsigned resume, const ecm_mersenne_ntt::Plan& plan) {
    const bool edwards = opt.execution.selection.curve == CurveModel::Edwards;
    const char* chain = !edwards ? "ladder" : naf_window ? "naf" : "x-only";
    std::ostringstream details;
    details << "device=" << opt.execution.device_index << " chain=" << chain
            << " ntt_bits=" << modulus_bits << " ntt_type=" << (plan.weighted ? "weighted" : "linear")
            << " ntt_length=" << plan.length << " ntt_bpw=" << plan.bpw
            << " naf=" << naf_window << " ntt_batch=" << batch << " ntt_digits=" << plan.digits
            << " ntt_resume=" << resume << " ntt_shape=" << ecm_mersenne_ntt::tune_shape_descriptor(plan)
            << " ntt_tile_bits=" << plan.tile << " ntt_max_radix_bits=" << plan.outer_max
            << " ntt_graph_bits=" << plan.window << " ntt_warp_tail=" << unsigned(plan.warp_tail)
            << " ntt_forward_pass_bits=" << plan.forward << " ntt_inverse_pass_bits=" << plan.inverse
            << " ntt_bpw_avg=" << std::fixed << std::setprecision(6) << double(modulus_bits) / plan.digits
            << " ntt_bpw_max=" << plan.bpw << " ntt_u=" << plan.coefficient_bound
            << " ntt_q=" << plan.field_modulus;
    log_execution_details(opt, "ntt", details.str());
}

struct Snapshot {
    unsigned cursor = 0;
    ActiveLanes lanes;
    std::vector<uint32_t> points;
};

constexpr std::string_view single_mask_snapshot = "ECMNTTS1";
constexpr std::string_view multi_mask_snapshot = "ECMNTTS2";

bool read_snapshot(std::string_view bytes, unsigned count, unsigned p, unsigned curves,
                   unsigned naf_window, Snapshot& out) {
    using ecm_stage1_binary::Reader;
    Reader reader(bytes);
    const auto magic = naf_window ? std::string_view("ECMNTTE1") :
                       curves <= 32 ? single_mask_snapshot : multi_mask_snapshot;
    const unsigned mask_words = (curves + 31) / 32;
    if (reader.left() < magic.size() + (2 + mask_words) * 4 ||
        reader.take(magic.size()) != magic || reader.u32() != count)
        return false;
    if (naf_window && (reader.left() < (2 + mask_words) * 4 || reader.u32() != naf_window))
        return false;
    Snapshot decoded;
    decoded.cursor = reader.u32();
    for (unsigned word = 0; word < mask_words; ++word) {
        const auto mask = reader.u32();
        const unsigned base = 32 * word;
        const unsigned width = std::min(32u, curves - base);
        if (width < 32 && (mask >> width)) return false;
        for (unsigned bit = 0; bit < width; ++bit)
            decoded.lanes[base + bit] = (mask >> bit) & 1u;
    }
    if (decoded.cursor > count || decoded.lanes.none())
        return false;
    const unsigned words = (p + 31) / 32;
    const size_t size = size_t(naf_window ? 3 : 4) * words * decoded.lanes.count();
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
    std::string payload(naf_window ? std::string_view("ECMNTTE1") :
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
                               unsigned words, mpz_ptr d, mpz_ptr px, mpz_ptr py) {
    using ecm_edwards::Integer;
    using ecm_edwards::Setup;
    Integer a24, x, inv, den;
    const auto setup = ecm_edwards::construct(d, px, py, a24.v, factor, sigma, n);
    if (setup == Setup::Factor) return CurveKind::Factor;
    if (setup == Setup::Degenerate) return CurveKind::Degenerate;

    mpz_ui_sub(den.v, 1, py);
    const auto chart = ecm_edwards::inverse(inv.v, den.v, n, factor);
    if (chart == Setup::Factor) return CurveKind::Factor;
    if (chart == Setup::Degenerate) return CurveKind::CompletedWithoutPoint;
    mpz_add_ui(x.v, py, 1);
    mpz_mul(x.v, x.v, inv.v);
    mpz_mod(x.v, x.v, n);
    // The point with x=0 has order two and the Edwards scalar is even.
    if (mpz_sgn(x.v) == 0) return CurveKind::CompletedWithoutPoint;

    Integer one, aa, bb, e, x2, z2;
    mpz_set_ui(one.v, 1);
    mpz_add_ui(aa.v, x.v, 1);
    mpz_mul(aa.v, aa.v, aa.v);
    mpz_mod(aa.v, aa.v, n);
    mpz_sub_ui(bb.v, x.v, 1);
    mpz_mul(bb.v, bb.v, bb.v);
    mpz_mod(bb.v, bb.v, n);
    mpz_sub(e.v, aa.v, bb.v);
    mpz_mul(x2.v, aa.v, bb.v);
    mpz_mod(x2.v, x2.v, n);
    mpz_mul(z2.v, a24.v, e.v);
    mpz_add(z2.v, z2.v, bb.v);
    mpz_mul(z2.v, z2.v, e.v);
    mpz_mod(z2.v, z2.v, n);
    const mpz_srcptr slots[] = {x.v, one.v, x2.v, z2.v, a24.v, x.v};
    for (unsigned slot = 0; slot < 6; ++slot)
        mpz_export(initial + slot * words, nullptr, -1, sizeof(uint32_t), 0, 0, slots[slot]);
    return CurveKind::Stage2Point;
}
} // namespace

ComputationResult run_ntt_stage1(const Stage1Request& request, ExecutionOptions& opt,
                                 ecm_stage1_checkpoint::Context& checkpoint) {
    ComputationResult result;
    auto phase = FailureSource::Preparation;
    try {
        const auto N = request.modulus;
        const size_t p = mpz_sizeinbase(N, 2);
        const unsigned curves = request.curves;
        const bool edwards = opt.execution.selection.curve == CurveModel::Edwards;
        const unsigned naf_window = edwards ? unsigned(opt.execution.naf_w) : 0;
        if (naf_window && (naf_window < 2 || naf_window > 12))
            throw std::invalid_argument("Edwards NTT NAF requires a window from 2 to 12");
        if (!curves || curves > stage1_ntt_max_curves || (!edwards && opt.execution.selection.gpu_param != 0) ||
            p < 32 || p > 262144 || mpz_sgn(N) <= 0 || mpz_scan0(N, 0) != p)
            throw std::invalid_argument(
                "NTT requires Edwards or Montgomery param0, a full 2^p-1 (32 <= p <= 262144), and 1-" +
                std::to_string(stage1_ntt_max_curves) + " curves");
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
        const size_t entries = naf_window ? size_t(1) << (naf_window - 2) : 0;
        bool gpu_dictionary = naf_window && p >= 32768;
        if (naf_window) {
            if (const char* override = std::getenv("ECM_EDWARDS_NTT_DICTIONARY")) {
                const std::string_view mode(override);
                if (mode == "gpu") gpu_dictionary = true;
                else if (mode == "host") gpu_dictionary = false;
                else throw std::invalid_argument("ECM_EDWARDS_NTT_DICTIONARY must be host or gpu");
            }
        }
        const size_t budget = size_t(opt.execution.naf_memory_mb) * ecm_edwards_memory::mb;
        const size_t dictionary_slots = !naf_window ? 0 : gpu_dictionary ? 3 : 4 * entries;
        const size_t host_dictionary_words = dictionary_slots * curves * words;
        std::vector<int16_t> digits;
        if (naf_window) {
            const size_t table_bytes = host_dictionary_words * sizeof(uint32_t);
            if (!gpu_dictionary && budget && table_bytes > budget)
                throw ecm_mersenne_ntt::BudgetError("Edwards NTT host dictionary exceeds naf_memory_mb");
            const auto free = ecm_edwards_memory::host_free_bytes();
            const size_t host_bytes = table_bytes + (scalar_bits + 1) * sizeof(int16_t) +
                edwards_naf_bytes(scalar_bits, true) +
                (gpu_dictionary ? entries + 2 * curves : 8 * entries) * words * sizeof(uint32_t);
            if (free && host_bytes > ecm_edwards_memory::usable_bytes(*free, 64 * ecm_edwards_memory::mib))
                throw ecm_mersenne_ntt::BudgetError("Edwards NTT NAF preparation exceeds available host memory");
            const auto plan = edwards_prepare_naf(scalar.v, uint64_t(request.b1), int(naf_window), true);
            digits.resize(plan->size());
            for (size_t i = 0; i < digits.size(); ++i)
                digits[i] = int16_t((*plan)[digits.size() - 1 - i]);
        }
        const unsigned count = naf_window ? unsigned(digits.size()) : binary_count;
        std::vector<uint32_t> packed((scalar_bits + 31) / 32);
        mpz_export(packed.data(), nullptr, -1, sizeof(uint32_t), 0, 0, scalar.v);

        Snapshot snapshot;
        bool has_checkpoint = false;
        phase = FailureSource::Persistence;
        checkpoint.validate_pending([&](const auto& records, auto& sigmas) {
            if (records.size() != curves || sigmas.size() != curves ||
                (!edwards && !montgomery_sigma_batch_fits(sigmas.front(), curves, 0)))
                return false;
            for (unsigned i = 0; i < curves; ++i)
                if ((edwards ? sigmas[i] == 0 : sigmas[i] != sigmas.front() + i) ||
                    (i && !records[i].empty()))
                    return false;
            has_checkpoint = read_snapshot(records.front(), count, unsigned(p), curves, naf_window, snapshot);
            return has_checkpoint;
        });
        uint64_t saved_sigma = 0;
        checkpoint.saved_sigma(0, saved_sigma);
        std::vector<uint64_t> sigmas(curves);
        if (edwards) {
            for (unsigned i = 0; i < curves; ++i) {
                uint64_t saved = 0;
                checkpoint.saved_sigma(i, saved);
                sigmas[i] = select_curve_sigma(saved, opt.identity.sigma_fixed, opt.identity.fixed_sigma64, i);
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
        const auto dict_started = Clock::now();
        if (edwards) {
            ecm_edwards::Integer d, px, py;
            for (unsigned i = 0; i < curves; ++i) {
                kinds[i] = prepare_edwards_lane(N, sigmas[i], factors.data()[i],
                    lanes_initial.data() + size_t(i) * 6 * words, words, d.v, px.v, py.v);
                if (!naf_window || kinds[i] != CurveKind::Stage2Point) continue;
                if (gpu_dictionary && (!has_checkpoint || snapshot.lanes[i])) {
                    const mpz_srcptr seed[] = {px.v, py.v, d.v};
                    for (unsigned j=0; j<3; ++j)
                        mpz_export(dictionary.data()+(size_t(j)*curves+i)*words,nullptr,-1,sizeof(uint32_t),0,0,seed[j]);
                    continue;
                }
                // Earlier snapshots could exclude lanes classified by the
                // host dictionary before entering the device loop.
                std::vector<uint32_t> legacy_table;
                if (gpu_dictionary) legacy_table.resize(4 * entries * words);
                auto& table = gpu_dictionary ? legacy_table : dictionary;
                const auto status = ecm_edwards::ntt::affine_dictionary(table, gpu_dictionary ? 1 : curves,
                    gpu_dictionary ? 0 : i, words,
                    naf_window, px.v, py.v, d.v, N, factors[i]);
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
            if (kinds[i] == CurveKind::Degenerate)
                throw std::runtime_error("degenerate Atkin-Morain construction on sigma " +
                                         std::to_string(sigmas[i]));
            if (kinds[i] == CurveKind::Factor) {
                result.set_curve(i, CurveKind::Factor, factors.data()[i], sigmas[i]);
                legacy = ECM_FACTOR_FOUND_STEP1;
            } else if (kinds[i] == CurveKind::CompletedWithoutPoint) {
                edwards_identity(i);
            } else if (std::find(fallback.begin(), fallback.end(), i) == fallback.end()) {
                active.push_back(i);
                lanes.set(i);
            }
        }
        if (has_checkpoint && snapshot.lanes != lanes)
            throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
                                               "NTT checkpoint active curve mask differs");

        if (!active.empty()) {
            const size_t stride = words * active.size();
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
            if (has_checkpoint) std::copy(snapshot.points.begin(), snapshot.points.end(), initial.begin());

            ecm_mersenne_ntt::Control control;
            control.resume = snapshot.cursor;
            control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
                log_ntt_execution(opt, p, active.size(), naf_window, control.resume, plan);
                ecm_ts_fprintf(stdout, "stage1_ntt_plan: device=%d p=%zu curves=%zu length=%u bpw=%u digits=%u "
                    "forward_passes=%u inverse_passes=%u forward_layers=%s inverse_layers=%s "
                    "tile_layers=%u tile_elements=%u tile_threads=%u tile_smem_bytes=%u "
                    "inverse_tile_threads=%u inverse_tile_smem_bytes=%u "
                    "outer_max_layers=%u outer_threads=%u warp_tail=%u graph_bits=%u "
                    "profile=%u source=%s resume=%u bits=%u convolution=%s curve=%s dataflow=%u "
                    "bpw_avg=%.6f bpw_max=%u U=%llu q=%llu\n",
                    opt.execution.device_index, p, active.size(), plan.length, plan.bpw, plan.digits,
                    plan.passes, plan.passes, plan.forward.c_str(), plan.inverse.c_str(),
                    plan.tile, 1u << plan.tile, plan.tile_threads, plan.tile_smem_bytes,
                    plan.inverse_tile_threads, plan.inverse_tile_smem_bytes,
                    plan.outer_max, plan.outer_threads, unsigned(plan.warp_tail),
                    plan.window, opt.execution.ntt.profile, plan.source, control.resume, count,
                    plan.weighted ? "weighted" : "linear", edwards ? "edwards" : "mont0", plan.dataflow,
                    double(p) / plan.digits, plan.bpw, plan.coefficient_bound, plan.field_modulus);
            };
            auto last_save = Clock::now(), last_progress = last_save;
            control.poll = [&](unsigned cursor) {
                const auto now = Clock::now();
                if (now - last_progress >= std::chrono::seconds(2)) {
                    ecm_ts_fprintf(stdout, "stage1_ntt_progress: %s=%u/%u (%.1f%%)\n",
                                   naf_window ? "columns" : "bits", cursor, count, 100.0 * cursor / count);
                    last_progress = now;
                }
                const bool stop = ecm_stage1_stop_requested();
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
                last_save = Clock::now();
            };

            phase = FailureSource::Device;
            const auto selected = ecm_mersenne_ntt::select_tuned_plan(opt.execution.device_index,
                unsigned(p), unsigned(active.size()), opt.execution.ntt, naf_window != 0);
            if (ecm_stage1_stop_requested()) {
                result.stop();
                return result;
            }
            const auto run = naf_window ? ecm_mersenne_ntt::edwards_naf(opt.execution.device_index, unsigned(p),
                unsigned(active.size()), initial, {naf_window, digits, dictionary, budget, gpu_dictionary}, selected, control)
                : ecm_mersenne_ntt::ladder(opt.execution.device_index, unsigned(p),
                    unsigned(active.size()), initial, packed, count, selected, control);
            ecm_ts_fprintf(stdout, "stage1_ntt_memory: owned_bytes=%zu observed_bytes=%zu budget_bytes=%zu\n",
                run.owned_bytes, run.observed_bytes, run.budget_bytes);
            if (naf_window)
                ecm_ts_fprintf(stdout, "stage1_ntt_edwards_dictionary: gpu_ms=%.3f backend=%s root_inverse_ms=%.3f\n",
                    run.dictionary_gpu_ms, gpu_dictionary ? "gpu" : "host", run.dictionary_inverse_ms);
            ecm_ts_fprintf(stdout, "stage1_ntt_timing: %s=%u device_ms=%.3f wall_ms=%.3f "
                "setup_ms=%.3f readback_ms=%.3f stopped=%u\n",
                naf_window ? "columns" : "bits", run.completed_bits - control.resume, run.device_ms.front(), run.wall_ms.front(),
                1000 * run.setup_seconds, 1000 * run.readback_seconds, unsigned(run.stopped));
            if (run.stopped) {
                result.stop();
                return result;
            }

            phase = FailureSource::Execution;
            ecm_edwards::Integer x, z, inv, y;
            for (size_t i = 0; i < active.size(); ++i) {
                const unsigned lane = active[i];
                if (gpu_dictionary && run.dictionary_status[i]) {
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
                mpz_import(x.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + i * words);
                mpz_import(z.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + stride + i * words);
                if (naf_window) {
                    mpz_swap(y.v, z.v);
                    mpz_import(z.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + 2 * stride + i * words);
                    mpz_add(x.v, z.v, y.v);
                    mpz_sub(z.v, z.v, y.v);
                }
                if (!mpz_invert(inv.v, z.v, N)) {
                    mpz_gcd(inv.v, z.v, N);
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
                    legacy = ECM_FACTOR_FOUND_STEP1;
                } else {
                    mpz_mul(x.v, x.v, inv.v);
                    mpz_mod(x.v, x.v, N);
                    result.set_curve(lane, CurveKind::Stage2Point, x.v, sigmas[lane]);
                }
            }
            if (opt.persistence.ckpt_ms && !checkpoint.path().empty() && control.resume != count)
                control.checkpoint(count, run.points);
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
            control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
                log_ntt_execution(opt, p, 1, 0, control.resume, plan);
            };
            control.poll = [](unsigned) {
                return ecm_mersenne_ntt::Control::Action{false, ecm_stage1_stop_requested()};
            };
            const auto selected = ecm_mersenne_ntt::select_tuned_plan(opt.execution.device_index,
                unsigned(p), 1, replay_ntt);
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
                legacy = ECM_FACTOR_FOUND_STEP1;
            } else {
                edwards_identity(lane);
            }
        }
        result.legacy_code = legacy;
        result.completion = ComputationResult::Completion::Completed;
        result.validate(N);
    } catch (const ecm_stage1_checkpoint::Error& error) {
        result.fail(FailureSource::Persistence);
        ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT checkpoint: %s\n", error.what());
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
