#pragma once
#include "../execution.h"
#include "memory.h"
#include "../execution_details.h"
#include "../sigma.h"
#include "../../curve/edwards.h"
#include "../../platform/generated/configuration_defaults.h"
#include "../../../io/stage1_checkpoint.h"
#include "ecm/stage1/progress.h"
#include "ecm/stage1/scalar.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace ecm_edwards_thread {
using namespace ecm_stage1;
using ecm_edwards::Integer;
inline constexpr unsigned max_bits = 1024;

namespace {
using Clock = std::chrono::steady_clock;


struct Cursor {
    uint32_t done = 0, window = 6, two_xy = 1;
};

// The batch envelope already binds N, B1, sigma list and the execution path,
// and checksums the payload. Geometry is fixed by N, so the device layout can
// be restored on another little-endian device.
inline bool decode(std::string_view format, const std::string& bytes, Cursor& cursor, std::vector<uint32_t>& state) {
    const size_t header = format.size();
    std::array<uint32_t, 3> fields{};
    if (bytes.size() != header + sizeof(fields) + state.size() * sizeof(uint32_t) ||
        bytes.compare(0, header, format))
        return false;
    std::memcpy(fields.data(), bytes.data() + header, sizeof(fields));
    cursor = {fields[0], fields[1], fields[2]};
    if (cursor.window < 2 || cursor.window > 12 || cursor.two_xy > 1)
        return false;
    std::memcpy(state.data(), bytes.data() + header + sizeof(fields), state.size() * sizeof(uint32_t));
    return true;
}

inline void persist(std::string_view format, ecm_stage1_checkpoint::Context& checkpoint, const Cursor& cursor,
             const std::vector<uint32_t>& state) {
    const std::array<uint32_t, 3> fields{cursor.done, cursor.window, cursor.two_xy};
    std::string bytes(format);
    bytes.append(reinterpret_cast<const char*>(fields.data()), sizeof(fields));
    bytes.append(reinterpret_cast<const char*>(state.data()), state.size() * sizeof(uint32_t));
    if (!checkpoint.write_blob(ecm_stage1_curve_record(0), bytes))
        throw std::runtime_error("cannot commit Thread Edwards checkpoint");
}

inline uint32_t montgomery_inverse(uint32_t n) {
    uint32_t inverse = 1;
    for (unsigned i = 0; i < 5; ++i)
        inverse *= 2 - n * inverse;
    return 0 - inverse;
}

struct FieldLayout {
    mpz_srcptr n;
    unsigned limbs;
    uint32_t count;
    Integer radix, inverse, value;
    std::vector<uint32_t> words;

    FieldLayout(mpz_srcptr modulus, unsigned width, uint32_t curves)
        : n(modulus), limbs(width), count(curves), words(limbs) {
        mpz_setbit(radix.v, 32 * limbs);
        mpz_mod(radix.v, radix.v, n);
        if (!mpz_invert(inverse.v, radix.v, n))
            throw std::runtime_error("Thread Edwards requires an odd modulus");
    }

    void pack(uint32_t* data, uint32_t lane, unsigned field, mpz_srcptr a) {
        mpz_mul(value.v, a, radix.v);
        mpz_mod(value.v, value.v, n);
        std::fill(words.begin(), words.end(), 0);
        mpz_export(words.data(), nullptr, -1, sizeof(uint32_t), 0, 0, value.v);
        for (unsigned j = 0; j < limbs; ++j)
            data[(size_t(field) * limbs + j) * count + lane] = words[j];
    }

    void unpack(mpz_ptr a, const uint32_t* data, uint32_t lane, unsigned field) {
        for (unsigned j = 0; j < limbs; ++j)
            words[j] = data[(size_t(field) * limbs + j) * count + lane];
        mpz_import(a, limbs, -1, sizeof(uint32_t), 0, 0, words.data());
        if (mpz_cmp(a, n) >= 0)
            throw std::runtime_error("noncanonical Thread Edwards residue");
        mpz_mul(a, a, inverse.v);
        mpz_mod(a, a, n);
    }
};

inline void finish_curves(ComputationResult& result, FieldLayout& layout,
                   const std::vector<uint32_t>& base, const std::vector<uint32_t>& state,
                   const std::vector<uint64_t>& sigmas) {
    ecm_edwards::Array point(4);
    Integer d, x, z, factor, inverse;
    bool found = false;
    for (uint32_t i = 0; i < layout.count; ++i) {
        if (result.curves[i].kind == CurveKind::Factor) {
            found = true;
            continue;
        }
        if (result.curves[i].kind == CurveKind::Degenerate)
            continue;
        for (unsigned field = 0; field < 4; ++field)
            layout.unpack(point[field], state.data(), i, field);
        layout.unpack(d.v, base.data(), i, 0);
        if (!ecm_edwards::point_valid(layout.n, d.v, point.data()))
            throw std::runtime_error("invalid Thread Edwards final point");
        // Birational map u=(Z+Y)/(Z-Y), giving the normalized Stage2 X/Z.
        mpz_sub(z.v, point[2], point[1]);
        mpz_gcd(factor.v, z.v, layout.n);
        if (mpz_cmp(factor.v, layout.n) == 0) {
            result.set_curve(i, CurveKind::CompletedWithoutPoint, nullptr, sigmas[i]);
        } else if (mpz_cmp_ui(factor.v, 1) > 0) {
            result.set_curve(i, CurveKind::Factor, factor.v, sigmas[i]);
            mpz_add(x.v, point[2], point[1]);
            result.curves[i].save_factor_endpoint(layout.n, x.v, z.v);
            found = true;
        } else {
            if (!mpz_invert(inverse.v, z.v, layout.n))
                throw std::runtime_error("could not normalize Thread Edwards point");
            mpz_add(x.v, point[2], point[1]);
            mpz_mul(x.v, x.v, inverse.v);
            mpz_mod(x.v, x.v, layout.n);
            result.set_curve(i, CurveKind::Stage2Point, x.v, sigmas[i]);
        }
    }
    result.legacy_code = found ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
    result.completion = ComputationResult::Completion::Completed;
    result.validate(layout.n);
}

inline uint32_t entries(unsigned window) { return 1u << (window - 2); }

inline void prepare_curves(ComputationResult& result, mpz_srcptr n, unsigned limbs,
                           const std::vector<uint64_t>& sigmas, const std::vector<uint32_t>& constants,
                           std::vector<uint32_t>& base, unsigned workers) {
    const uint32_t count = static_cast<uint32_t>(sigmas.size());
    const bool full_mersenne = mpz_scan0(n, 0) == mpz_sizeinbase(n, 2);
    const auto prepare = [&](unsigned worker) {
        FieldLayout layout(n, limbs, count);
        std::unique_ptr<ecm_edwards::MersenneConstruction> mersenne;
        if (full_mersenne)
            mersenne = std::make_unique<ecm_edwards::MersenneConstruction>(n);
        Integer d, px, py, a24, factor;
        const auto first = uint32_t(uint64_t(count) * worker / workers);
        const auto last = uint32_t(uint64_t(count) * (worker + 1) / workers);
        for (uint32_t i = first; i < last; ++i) {
            const auto rc = ecm_edwards::construct(d.v, px.v, py.v, a24.v, factor.v, sigmas[i], n, mersenne.get());
            if (rc != ecm_edwards::Setup::Ready) {
                if (rc == ecm_edwards::Setup::Factor) {
                    result.set_curve(i, CurveKind::Factor, factor.v, sigmas[i]);
                } else if (ecm_edwards::construction_is_degenerate(rc, factor.v, n)) {
                    ecm_ts_fprintf(stdout, "Thread Edwards skipped degenerate construction: curve=%u sigma=%llu gcd=N\n",
                                   i, static_cast<unsigned long long>(sigmas[i]));
                    result.set_curve(i, CurveKind::Degenerate, nullptr, sigmas[i]);
                } else {
                    throw std::runtime_error("invalid Atkin-Morain construction sigma=" + std::to_string(sigmas[i]));
                }
                for (unsigned j = 0; j < limbs; ++j)
                    base[(2 * size_t(limbs) + j) * count + i] = constants[limbs + j];
            } else {
                layout.pack(base.data(), i, 0, d.v);
                layout.pack(base.data(), i, 1, px.v);
                layout.pack(base.data(), i, 2, py.v);
            }
        }
    };
    std::vector<std::future<void>> jobs;
    jobs.reserve(workers - 1);
    for (unsigned i = 1; i < workers; ++i)
        jobs.emplace_back(std::async(std::launch::async, prepare, i));
    prepare(0);
    for (auto& job : jobs)
        job.get();
}

} // namespace

template<class Device>
ComputationResult run_stage1(const Stage1Request& request, ExecutionOptions& opt,
                                    ecm_stage1_checkpoint::Context& checkpoint) {
    ComputationResult result;
    auto failure = FailureSource::Preparation;
    try {
        const auto init = Clock::now();
        const auto n = request.modulus;
        const uint32_t count = request.curves;
        const auto bits = mpz_sizeinbase(n, 2);
        const auto b1 = std::floor(request.b1 + 0.0001);
        if (!count || !std::isfinite(b1) || b1 < 2 || b1 > 5000000000.0 ||
            bits > max_bits || mpz_cmp_ui(n, 3) <= 0 || !mpz_odd_p(n))
            throw std::invalid_argument("Thread Edwards requires odd N > 3 up to 1024 bits, B1 in 2..5000000000 and a nonempty batch");
        Cursor cursor{0, opt.execution.naf_w ? unsigned(opt.execution.naf_w) : 6u,
            unsigned(opt.execution.edwards_doubling.value_or(ecm_config::defaults::stage1_edwards_doubling))};
        if (cursor.window < 2 || cursor.window > 12 || cursor.two_xy > 1)
            throw std::invalid_argument("Thread Edwards requires NAF w=2..12 and doubling=0 or 1");
        const uint32_t endian_test = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian_test) != 1)
            throw std::runtime_error("Thread Edwards requires a little-endian host");
        constexpr unsigned tiers[] = {4, 6, 8, 12, 16, 24, 32};
        const unsigned limbs = *std::lower_bound(std::begin(tiers), std::end(tiers), (bits + 31) / 32);
        const uint64_t plane = uint64_t(count) * limbs;
        const uint64_t vector_bytes = plane * sizeof(uint32_t);
        const auto host_free = ecm_edwards_memory::host_free_bytes();
        const auto host_need = ecm_edwards_memory::scalar_preparation_bytes(uint64_t(b1)) +
            ecm_edwards_memory::batch_host_bytes(count, bits) + 20 * vector_bytes;
        if ((host_free && host_need > ecm_edwards_memory::usable_bytes(*host_free, 64 * ecm_edwards_memory::mib)) ||
            11 * vector_bytes > SIZE_MAX)
            throw std::runtime_error("insufficient Thread Edwards host memory; reduce the batch or B1");
        std::vector<uint32_t> state(4 * size_t(plane), 0);
        const auto path = ecm_stage1_curve_record(0);
        const bool checkpoint_enabled = !checkpoint.path().empty();
        // Restore the original formula/window before rebuilding the shared NAF.
        Integer scalar;
        if (!compute_batch_s(scalar.v, b1, 48, opt.persistence.exponent_cache_dir))
            throw std::runtime_error("cannot build Thread Edwards exponent");
        const auto scalar_bits = mpz_sizeinbase(scalar.v, 2);
        if (scalar_bits >= UINT32_MAX)
            throw std::runtime_error("Thread Edwards exponent exceeds NAF column limit");
        const uint64_t naf_bytes = ((uint64_t(scalar_bits) + 32) / 32 + 1) * sizeof(uint32_t);
        if (host_free && host_need + 3 * naf_bytes >
            ecm_edwards_memory::usable_bytes(*host_free, 64 * ecm_edwards_memory::mib))
            throw std::runtime_error("insufficient host memory for the Thread Edwards NAF plan");

        checkpoint.validate_pending([&](const std::vector<std::string>& records,
                                          std::vector<uint64_t>& seeds) {
            if (records.size() != count || seeds.size() != count ||
                std::any_of(records.begin() + 1, records.end(), [](const auto& r) { return !r.empty(); }))
                return false;
            if (records[0].empty())
                return true;
            Cursor candidate;
            std::vector<uint32_t> candidate_state(state.size());
            if (!decode(Device::checkpoint_format, records[0], candidate, candidate_state) ||
                candidate.done > scalar_bits + 1)
                return false;
            const EcmEdwardsNaf candidate_naf(scalar.v, candidate.window);
            if (candidate.done > candidate_naf.bits)
                return false;
            FieldLayout candidate_layout(n, limbs, count);
            ecm_edwards::Array point(4);
            Integer d, x, y, a24, factor;
            std::unique_ptr<ecm_edwards::MersenneConstruction> mersenne;
            if (mpz_scan0(n, 0) == bits)
                mersenne = std::make_unique<ecm_edwards::MersenneConstruction>(n);
            try {
                for (uint32_t i = 0; i < count; ++i) {
                    if (!seeds[i])
                        return false;
                    const auto code = ecm_edwards::construct(d.v, x.v, y.v, a24.v, factor.v,
                                                            seeds[i], n, mersenne.get());
                    if (code == ecm_edwards::Setup::Factor)
                        continue;
                    const bool degenerate = ecm_edwards::construction_is_degenerate(code, factor.v, n);
                    if (code != ecm_edwards::Setup::Ready && !degenerate)
                        return false;
                    if (degenerate)
                        mpz_set_ui(d.v, 0);
                    for (unsigned k = 0; k < 4; ++k)
                        candidate_layout.unpack(point[k], candidate_state.data(), i, k);
                    if (!ecm_edwards::point_valid(n, d.v, point.data()) ||
                        (degenerate && !ecm_edwards::point_is_identity(point.data())))
                        return false;
                }
            } catch (const std::runtime_error&) {
                return false;
            }
            return true;
        });
        std::string saved;
        const bool resumed = checkpoint_enabled && checkpoint.read_blob(path, saved);
        if (resumed && !decode(Device::checkpoint_format, saved, cursor, state))
            throw std::runtime_error("incompatible thread-per-curve Edwards checkpoint");
        saved.clear();

        Device device(opt, limbs);
        const auto max_buffer = device.max_buffer();
        const auto budget = ecm_edwards_memory::configured_budget(device.memory_budget(), opt.execution.naf_memory_mb);
        const auto fits = [&](unsigned window) {
            const auto dictionary = vector_bytes * 4 * entries(window);
            return dictionary <= max_buffer && dictionary <= SIZE_MAX &&
                naf_bytes <= max_buffer && dictionary + 11 * vector_bytes + 2 * naf_bytes +
                    2 * limbs * sizeof(uint32_t) <= budget;
        };
        const auto requested = cursor.window;
        while (!resumed && cursor.window > 2 && !fits(cursor.window))
            --cursor.window;
        if (!fits(cursor.window))
            throw std::runtime_error("Thread Edwards NAF exceeds device memory budget; reduce the batch or B1");
        if (requested != cursor.window)
            ecm_ts_fprintf(stderr, "WARNING: Thread Edwards NAF window %u -> %u (device memory budget)\n",
                           requested, cursor.window);
        const EcmEdwardsNaf naf(scalar.v, cursor.window);
        if (cursor.done > naf.bits)
            throw std::runtime_error("Thread Edwards checkpoint cursor exceeds the NAF plan");
        const auto weights = entries(cursor.window);

        FieldLayout layout(n, limbs, count);
        std::vector<uint32_t> base(3 * size_t(plane), 0), constants(2 * limbs, 0);
        mpz_export(constants.data(), nullptr, -1, sizeof(uint32_t), 0, 0, n);
        mpz_export(constants.data() + limbs, nullptr, -1, sizeof(uint32_t), 0, 0, layout.radix.v);
        const uint32_t np0 = montgomery_inverse(constants.front());
        result.curves.resize(count);
        std::vector<uint64_t> sigmas(count);
        const unsigned available = opt.execution.worker_cpu_count ? opt.execution.worker_cpu_count :
            std::max(1u, std::thread::hardware_concurrency());
        const unsigned requested_workers = opt.execution.gpu_init_threads ? opt.execution.gpu_init_threads :
            count >= 1024 ? 4 : 1;
        const unsigned workers = std::min({available, requested_workers, count});
        const auto setup_start = Clock::now();
        CurveSigmas selection(opt.identity, CurveModel::Edwards);
        for (uint32_t i = 0; i < count; ++i) {
            uint64_t sigma = 0;
            checkpoint.saved_sigma(i, sigma);
            sigmas[i] = checkpoint.sigma(i, selection.next(sigma, i));
        }
        if (!resumed) {
            for (size_t field : {1u, 2u}) {
                for (unsigned j = 0; j < limbs; ++j) {
                    auto* target = state.data() + (field * limbs + j) * count;
                    std::fill_n(target, count, constants[limbs + j]);
                }
            }
        }
        prepare_curves(result, n, limbs, sigmas, constants, base, workers);
        const auto setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
        if (ecm_stage1_stop_requested()) {
            result.stop();
            return result;
        }
        if (cursor.done < naf.bits)
            device.prepare(cursor.two_xy != 0);
        log_execution_details(opt, Device::name, "algorithm=naf kernel_weight=" + std::to_string(32 * limbs) +
            " tpi=1 naf=" + std::to_string(cursor.window) + " doubling=" +
            std::to_string(cursor.two_xy ? 5 : 4) + " " + device.details() +
            " resume=" + std::to_string(cursor.done));
        double dictionary_ms = 0, loop_ms = 0, init_seconds = 0;
        uint32_t launches = 0;
        if (cursor.done < naf.bits) {
            const auto params = device.buffer(constants.size() * sizeof(uint32_t), true);
            const auto points = device.buffer(4 * vector_bytes, false);
            const auto table = device.buffer(4 * vector_bytes * weights, false);
            const auto nonzero = device.buffer(naf_bytes, true);
            const auto codes = device.buffer(naf_bytes, true);
            device.upload(params, constants.data(), constants.size() * sizeof(uint32_t));
            device.upload(points, state.data(), 4 * vector_bytes);
            device.upload(nonzero, naf.nonzero.data(), naf_bytes);
            device.upload(codes, naf.codes.data(), naf_bytes);
            failure = FailureSource::Device;
            {
                const auto seeds = device.buffer(3 * vector_bytes, true);
                const auto scratch = device.buffer(4 * vector_bytes, false);
                device.upload(seeds, base.data(), 3 * vector_bytes);
                for (uint32_t first = 0; first < weights; first += 16) {
                    if (ecm_stage1_stop_requested()) {
                        result.stop();
                        return result;
                    }
                    const uint32_t stop = std::min(first + 16, weights);
                    dictionary_ms += device.dictionary(params, seeds, table, scratch, count, np0, first, stop, weights);
                }
            }
            const auto loop = Clock::now();
            init_seconds = std::chrono::duration<double>(loop - init).count();
            auto last_checkpoint = loop;
            auto last_progress = loop;
            ecm_stage1_gpu_progress::SpeedWindow speed;
            uint32_t columns = 32;
            while (cursor.done < naf.bits && !ecm_stage1_stop_requested()) {
                const uint32_t first = cursor.done, delta = std::min(columns, naf.bits - first);
                const uint32_t stop = first + delta;
                const double elapsed_ms = device.slice(params, points, table, nonzero, codes, count, np0,
                                                       naf.bits, first, stop, weights);
                loop_ms += elapsed_ms;
                ++launches;
                cursor.done = stop;
                if (elapsed_ms > 0) {
                    speed.add(delta, elapsed_ms);
                    columns = uint32_t(std::clamp(delta * 50.0 / elapsed_ms, 1.0, 4096.0));
                }
                const auto now = Clock::now();
                if (now - last_progress >= std::chrono::seconds(1)) {
                    const auto elapsed = std::chrono::duration<double>(now - loop).count();
                    const auto line = ecm_stage1_gpu_progress::line(100.0 * stop / naf.bits, stop, delta,
                        "cols", speed.per_curve(naf.bits, count), elapsed, speed.remaining(naf.bits, stop), speed.speed()*1000);
                    ecm_ts_fprintf(stdout, "%s\n", line.c_str());
                    last_progress = now;
                }
                if (checkpoint_enabled && opt.persistence.ckpt_ms &&
                    now - last_checkpoint >= std::chrono::milliseconds(opt.persistence.ckpt_ms)) {
                    device.download(points, state.data(), 4 * vector_bytes);
                    failure = FailureSource::Persistence;
                    persist(Device::checkpoint_format, checkpoint, cursor, state);
                    failure = FailureSource::Device;
                    last_checkpoint = Clock::now();
                }
            }
            device.download(points, state.data(), 4 * vector_bytes);
        }
        if (checkpoint_enabled) {
            failure = FailureSource::Persistence;
            persist(Device::checkpoint_format, checkpoint, cursor, state);
        }
        if (cursor.done < naf.bits) {
            result.stop();
            return result;
        }
        failure = FailureSource::Execution;
        finish_curves(result, layout, base, state, sigmas);
        ecm_ts_fprintf(stdout, "%s Edwards NAF: columns=%u additions=%u dictionary_bytes=%llu "
            "construction_s=%.6f init_s=%.6f dictionary_gpu_s=%.6f loop_gpu_s=%.6f launches=%u construction_threads=%u\n",
            Device::label, naf.bits, naf.additions, static_cast<unsigned long long>(4 * vector_bytes * weights),
            setup_seconds, init_seconds, dictionary_ms / 1000, loop_ms / 1000, launches, workers);
    } catch (const std::exception& error) {
        result.fail(failure);
        ecm_ts_fprintf(stderr, "ERROR: %s Edwards Stage1: %s\n", Device::label, error.what());
    }
    return result;
}
} // namespace ecm_edwards_thread
