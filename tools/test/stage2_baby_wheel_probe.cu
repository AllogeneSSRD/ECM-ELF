// Standalone CUDA comparison of the production baby ladder and sparse xADD walks.
#include <cuda_runtime.h>
#include "ecm/integer.h"
#include <cgbn/cgbn.h>
#include "../../kernels/cuda/stage2_baby_wheel.cuh"
#include "../../src/ecm/stage2/cuda/device/point_coop_ladder.cuh"
#include "../../src/ecm/stage2/cuda/point_coop_plan.h"
#include "../../src/ecm/stage2/cuda/stage2_baby_wheel_plan.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ecm_stage2::baby_wheel;
using ecm_edwards::Integer;
using Clock = std::chrono::steady_clock;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template<class T>
struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { if (count) check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& input) {
        check(cudaMemcpy(data, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void get(T* out, size_t count, size_t offset = 0) const {
        check(cudaMemcpy(out, data + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

void pack(Word* dest, int nw, mpz_srcptr value) {
    std::fill_n(dest, nw, 0);
    size_t used;
    mpz_export(dest, &used, -1, sizeof(Word), 0, 0, value);
}

void unpack(mpz_ptr dest, const Word* input, int nw) {
    mpz_import(dest, nw, -1, sizeof(Word), 0, 0, input);
}

struct Coordinates {
    Buffer<Word> x, z;
    explicit Coordinates(size_t words) : x(words), z(words) {}
};

// For the harness use the same widths and chunk limit as ordinary Stage2.
template<int NW, int TPI>
void ladder(const Word* hw, Word ninv, int w, const Word* j, unsigned count,
            int mersenne, Coordinates& out) {
    constexpr unsigned threads = 64, cap = 8192;
    for (unsigned first = 0; first < count; first += cap) {
        const unsigned size = std::min(cap, count - first);
        if constexpr (TPI) {
            stage2_coop_ladder<NW, TPI><<<(size * TPI + threads - 1) / threads, threads>>>(
                hw, ninv, w, hw + w, hw + 2 * w, hw + 3 * w, hw + 4 * w, j + first, size, mersenne,
                false, out.x.data + size_t(first) * w, out.z.data + size_t(first) * w);
        } else {
            s2g_ladder_kernel<NW, true><<<(size + threads - 1) / threads, threads>>>(
                hw, ninv, w, hw + w, hw + 2 * w, hw + 3 * w, hw + 4 * w, j + first, size,
                out.x.data + size_t(first) * w, out.z.data + size_t(first) * w, false);
        }
    }
    check(cudaGetLastError());
}

bool unit(mpz_srcptr value, mpz_srcptr target) {
    Integer gcd;
    mpz_gcd(gcd.v, value, target);
    return mpz_cmp_ui(gcd.v, 1) == 0;
}

struct Recovery {
    size_t bad_segments = 0, replayed_points = 0;
    bool common_guard_failed = false;
};

void double_point(mpz_ptr x, mpz_ptr z, mpz_srcptr a24, mpz_srcptr n) {
    Integer a, b, delta;
    mpz_add(a.v, x, z); mpz_powm_ui(a.v, a.v, 2, n);
    mpz_sub(b.v, x, z); mpz_powm_ui(b.v, b.v, 2, n);
    mpz_sub(delta.v, a.v, b.v);
    mpz_mul(x, a.v, b.v); mpz_mod(x, x, n);
    mpz_mul(z, a24, delta.v); mpz_add(z, z, b.v);
    mpz_mul(z, z, delta.v); mpz_mod(z, z, n);
}

void sum_points(mpz_ptr out_x, mpz_ptr out_z, mpz_srcptr x, mpz_srcptr z,
                mpz_srcptr y, mpz_srcptr w, mpz_srcptr diff_x, mpz_srcptr n) {
    Integer a, b, c;
    mpz_mul(a.v, x, y); mpz_mul(c.v, z, w);
    mpz_sub(a.v, a.v, c.v); mpz_powm_ui(a.v, a.v, 2, n);
    mpz_mul(b.v, x, w); mpz_mul(c.v, z, y);
    mpz_sub(b.v, b.v, c.v); mpz_powm_ui(b.v, b.v, 2, n);
    mpz_set(out_x, a.v); // Qz is one in the standalone fixture.
    mpz_mul(out_z, b.v, diff_x); mpz_mod(out_z, out_z, n);
}

void reference(Word index, mpz_ptr x, mpz_ptr z, mpz_srcptr qx, mpz_srcptr a24, mpz_srcptr n) {
    Integer next_x, next_z;
    mpz_mod(x, qx, n); mpz_set_ui(z, 1);
    mpz_set(next_x.v, x); mpz_set(next_z.v, z);
    double_point(next_x.v, next_z.v, a24, n);
    Word leading = 1;
    while (leading <= index / 2) leading *= 2;
    for (Word bit = leading / 2; bit; bit /= 2) {
        if (index & bit) {
            sum_points(x, z, x, z, next_x.v, next_z.v, qx, n);
            double_point(next_x.v, next_z.v, a24, n);
        } else {
            sum_points(next_x.v, next_z.v, x, z, next_x.v, next_z.v, qx, n);
            double_point(x, z, a24, n);
        }
    }
}

void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << ']';
}

template<int NW, int TPI = 0>
void run(const std::vector<Word>& indices, const Plan& plan, int repetitions, unsigned d, unsigned wheel,
         unsigned length, int mersenne,
         mpz_srcptr target, mpz_srcptr arithmetic, mpz_srcptr qx, mpz_srcptr a24, double plan_ms) {
    const auto setup = Clock::now();
    const int w = int((mpz_sizeinbase(arithmetic, 2) + 63) / 64);
    const auto rows = unsigned(indices.size());
    const size_t words = size_t(rows) * w;
    Integer r, value;
    mpz_set_ui(r.v, 1); mpz_mul_2exp(r.v, r.v, 64 * w); mpz_mod(r.v, r.v, arithmetic);
    std::vector<Word> inputs(5 * w);
    pack(inputs.data(), w, arithmetic);
    mpz_mul(value.v, qx, r.v); mpz_mod(value.v, value.v, arithmetic);
    pack(inputs.data() + w, w, value.v);
    pack(inputs.data() + 2 * w, w, r.v);
    mpz_mul(value.v, a24, r.v); mpz_mod(value.v, value.v, arithmetic);
    pack(inputs.data() + 3 * w, w, value.v);
    pack(inputs.data() + 4 * w, w, r.v);
    Word inv = 1;
    for (unsigned i = 0; i < 6; ++i) inv *= 2 - inputs[0] * inv;
    const Word ninv = 0 - inv;
    mpz_sub_ui(value.v, a24, 1);
    const bool curve_unit = unit(qx, target) && unit(a24, target) && unit(value.v, target);

    Buffer<Word> hw(inputs.size()), device_indices(rows), seeds(plan.seeds.size());
    Buffer<unsigned> slots(plan.slots.size());
    Buffer<Segment> segments(plan.segments.size());
    Coordinates base(words), trial(words), seed_points(plan.seeds.size() * w);
    Buffer<Word> guards(plan.segments.size() * w);
    std::vector<Word> products(plan.segments.size() * w), delta(2 * w);
    hw.put(inputs); device_indices.put(indices); seeds.put(plan.seeds);
    slots.put(plan.slots); segments.put(plan.segments);
    const double setup_ms = elapsed(setup);

    const auto generate_base = [&](Coordinates& out) {
        ladder<NW, TPI>(hw.data, ninv, w, device_indices.data, rows, mersenne, out);
        check(cudaDeviceSynchronize());
    };
    const auto generate_walk = [&]() {
        Recovery recovery;
        if (curve_unit) {
            ladder<NW, TPI>(hw.data, ninv, w, seeds.data, unsigned(plan.seeds.size()), mersenne, seed_points);
            seed_points.x.get(delta.data(), w);
            seed_points.z.get(delta.data() + w, w);
            unpack(value.v, delta.data(), w);
            const bool step_x_unit = unit(value.v, target);
            unpack(value.v, delta.data() + w, w);
            recovery.common_guard_failed = !step_x_unit || !unit(value.v, target);
        } else {
            recovery.common_guard_failed = true;
        }
        if (recovery.common_guard_failed) {
            generate_base(trial);
            recovery.bad_segments = plan.segments.size();
            recovery.replayed_points = rows;
            return recovery;
        }
        constexpr unsigned threads = 64, sharing = TPI ? TPI : 1;
        const auto count = unsigned(plan.segments.size());
        const unsigned grid = (count * sharing + threads - 1) / threads;
        if constexpr (TPI) {
            coop_chain<NW, TPI><<<grid, threads>>>(hw.data, ninv, w, hw.data + 4 * w, segments.data,
                count, slots.data, seed_points.x.data, seed_points.z.data, mersenne,
                trial.x.data, trial.z.data, guards.data);
        } else {
            scalar_chain<NW><<<grid, threads>>>(hw.data, ninv, w, hw.data + 4 * w, segments.data,
                count, slots.data, seed_points.x.data, seed_points.z.data,
                trial.x.data, trial.z.data, guards.data);
        }
        check(cudaGetLastError());
        guards.get(products.data(), products.size());
        std::vector<unsigned> replay_slots;
        std::vector<Word> replay_indices;
        for (unsigned i = 0; i < count; ++i) {
            unpack(value.v, products.data() + size_t(i) * w, w);
            if (unit(value.v, target)) continue;
            ++recovery.bad_segments;
            const auto seg = plan.segments[i];
            for (unsigned j = 0; j < seg.span; ++j) {
                const auto slot = plan.slots[seg.walk_begin + j];
                if (slot == skip) continue;
                replay_slots.push_back(slot);
                replay_indices.push_back(indices[slot]);
            }
        }
        recovery.replayed_points = replay_slots.size();
        if (!replay_indices.empty()) {
            const size_t replay_words = replay_slots.size() * w;
            Buffer<unsigned> map(replay_slots.size());
            Buffer<Word> indices_to_replay(replay_indices.size());
            Coordinates replay(replay_words);
            map.put(replay_slots); indices_to_replay.put(replay_indices);
            ladder<NW, TPI>(hw.data, ninv, w, indices_to_replay.data, unsigned(replay_indices.size()),
                            mersenne, replay);
            scatter_replayed<unsigned><<<unsigned((replay_words + 127) / 128), 128>>>(
                map.data, replay_words, w, replay.x.data, replay.z.data, trial.x.data, trial.z.data);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
        }
        return recovery;
    };

    generate_base(base);
    generate_walk();
    std::vector<double> base_ms, wheel_ms;
    Recovery recovery;
    const auto time_base = [&]() {
        const auto start = Clock::now();
        generate_base(base); base_ms.push_back(elapsed(start));
    };
    const auto time_wheel = [&]() {
        const auto start = Clock::now();
        recovery = generate_walk(); wheel_ms.push_back(elapsed(start));
    };
    for (int i = 0; i < repetitions; ++i) {
        if (i % 2 == 0) { time_base(); time_wheel(); }
        else { time_wheel(); time_base(); }
    }

    std::vector<Word> bx(words), bz(words), wx(words), wz(words);
    base.x.get(bx.data(), words); base.z.get(bz.data(), words);
    trial.x.get(wx.data(), words); trial.z.get(wz.data(), words);
    Integer x0, z0, x1, z1, gcd0, gcd1, a, b;
    size_t nonunit = 0, zero = 0, gmp_samples = 0;
    const unsigned sample_count = std::min(48u, rows);
    unsigned next_sample = 0;
    for (unsigned i = 0; i < rows; ++i) {
        const size_t offset = size_t(i) * w;
        unpack(x0.v, bx.data() + offset, w); unpack(z0.v, bz.data() + offset, w);
        unpack(x1.v, wx.data() + offset, w); unpack(z1.v, wz.data() + offset, w);
        mpz_gcd(gcd0.v, z0.v, target); mpz_gcd(gcd1.v, z1.v, target);
        bool equal = mpz_cmp(gcd0.v, gcd1.v) == 0;
        if (mpz_cmp_ui(gcd0.v, 1) == 0) {
            mpz_mul(a.v, x0.v, z1.v); mpz_mul(b.v, x1.v, z0.v);
            mpz_sub(a.v, a.v, b.v);
            equal = equal && mpz_divisible_p(a.v, target);
        } else {
            if (mpz_cmp(gcd0.v, target) == 0) ++zero;
            else ++nonunit;
            // Preserve the original projective representative for factor/cache fallback.
            equal = equal && mpz_cmp(x0.v, x1.v) == 0 && mpz_cmp(z0.v, z1.v) == 0;
        }
        if (!equal) throw std::runtime_error("wheel mismatch at scalar " + std::to_string(indices[i]));
        if (i == next_sample && gmp_samples < sample_count) {
            reference(indices[i], a.v, b.v, qx, a24, target);
            mpz_mul(a.v, a.v, r.v); mpz_sub(a.v, a.v, x0.v);
            mpz_mul(b.v, b.v, r.v); mpz_sub(b.v, b.v, z0.v);
            if (!mpz_divisible_p(a.v, target) || !mpz_divisible_p(b.v, target))
                throw std::runtime_error("GMP baseline mismatch at scalar " + std::to_string(indices[i]));
            ++gmp_samples;
            next_sample = unsigned(gmp_samples * rows / sample_count);
        }
    }
    std::cout << "{\"status\":\"PASS\",\"target_bits\":" << mpz_sizeinbase(target, 2)
              << ",\"arithmetic_bits\":" << mpz_sizeinbase(arithmetic, 2)
              << ",\"d\":" << d << ",\"wheel\":" << wheel << ",\"max_walk\":" << length
              << ",\"points\":" << rows << ",\"segments\":" << plan.segments.size()
              << ",\"seeds\":" << plan.seeds.size() << ",\"walked\":" << plan.slots.size()
              << ",\"z_nonunit\":" << nonunit << ",\"z_zero\":" << zero
              << ",\"bad_segments\":" << recovery.bad_segments << ",\"replayed_points\":" << recovery.replayed_points
              << ",\"common_guard_failed\":" << (recovery.common_guard_failed ? "true" : "false")
              << ",\"verified_points\":" << rows << ",\"gmp_samples\":" << gmp_samples << ",\"tpi\":" << TPI
              << ",\"plan_ms\":" << plan_ms << ",\"setup_ms\":" << setup_ms << ",\"base_ms\":";
    print_samples(base_ms);
    std::cout << ",\"wheel_ms\":";
    print_samples(wheel_ms);
    std::cout << "}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("DEVICE D WHEEL MAX_WALK REPEATS MERSENNE_BITS; stdin: target, arithmetic, Qx, A24 hex");
        const int device = std::stoi(argv[1]);
        const unsigned d = std::stoul(argv[2]), wheel = std::stoul(argv[3]), length = std::stoul(argv[4]);
        const int repetitions = std::stoi(argv[5]), mersenne = std::stoi(argv[6]);
        if (d < 4 || d > 1200000 || wheel < 2 || wheel % 2 || wheel > 30030 || d % wheel ||
            length < 2 || length > 64 || repetitions < 1 || repetitions > 11 || mersenne < 0)
            throw std::runtime_error("invalid probe shape");
        Integer target, arithmetic, qx, a24;
        for (const auto p : {target.v, arithmetic.v, qx.v, a24.v}) {
            std::string line;
            if (!(std::cin >> line) || mpz_set_str(p, line.c_str(), 16) != 0)
                throw std::runtime_error("expected four hexadecimal inputs");
        }
        const int bits = int(mpz_sizeinbase(arithmetic.v, 2));
        if (mpz_cmp_ui(target.v, 3) < 0 || !mpz_odd_p(target.v) || !mpz_divisible_p(arithmetic.v, target.v) ||
            mpz_cmp(arithmetic.v, target.v) < 0 || bits > 16384 ||
            (mersenne && (mersenne != bits || mpz_popcount(arithmetic.v) != unsigned(bits))))
            throw std::runtime_error("invalid target or arithmetic modulus");
        const auto start = Clock::now();
        std::vector<Word> indices;
        for (unsigned j=1;j<=d/2;++j)
            if (std::gcd(j,d)==1) indices.push_back(j);
        const auto plan = make_plan(indices, wheel, length);
        const double plan_ms = elapsed(start);
        check(cudaSetDevice(device));
        S2GDeviceConstants constants{};
        constants.point_square = 1;
        constants.point_mersenne_bits = mersenne;
        check(cudaMemcpyToSymbol(g_s2g_device_constants, &constants, sizeof(constants)));
        const int words = (bits + 63) / 64;
#define RUN(NW, TPI) run<NW, TPI>(indices, plan, repetitions, d, wheel, length, mersenne, target.v, arithmetic.v, qx.v, a24.v, plan_ms)
        if (words <= 4) RUN(4, 0);
        else if (words <= 8) RUN(8, 0);
        else if (words <= 16) RUN(16, 0);
        else if (words <= 32) RUN(32, 0);
        else if (words <= 64) RUN(64, 0);
        else if (words <= 128) RUN(128, 0);
        else switch (s2g_coop_capacity_words(words, 32)) {
            case 160: RUN(160, 32); break;
            case 192: RUN(192, 32); break;
            case 224: RUN(224, 32); break;
            default: RUN(256, 32); break;
        }
#undef RUN
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage2_baby_wheel_probe: " << error.what() << '\n';
        return 2;
    }
}
