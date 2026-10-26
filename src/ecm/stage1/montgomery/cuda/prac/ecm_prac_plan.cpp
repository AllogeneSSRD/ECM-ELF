// Prime95 ecm.cpp:2645/2857 simplified PRAC search, priced for normalized CGBN.
#include "ecm_prac_plan.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

uint64_t ecm_prac_hash(const void *data, size_t bytes, uint64_t hash) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}
uint64_t ecm_prac_hash_mpz(mpz_srcptr n) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < mpz_size(n); ++i) {
        mp_limb_t limb = mpz_getlimbn(n, i);
        hash = ecm_prac_hash(&limb, sizeof(limb), hash);
    }
    return hash;
}
namespace {
struct Header {
    uint64_t magic, b1, scalar, count, work, checksum;
    uint32_t version, search, torsion, endian;
};
constexpr uint64_t magic = 0x31504e414c504345ull;
constexpr double ratios[] = {0.6180339887498948, 0.7236067977499790,
    0.5801787282954641, 0.6328398060887063, 0.6124299495094950,
    0.6201819808074158, 0.6172146165344039, 0.6183471196562281,
    0.6179144065288179, 0.6180796684698958};

uint32_t cost(uint32_t p, uint32_t initial) {
    uint32_t e = p - initial, d = initial - e, work = 11;
    while (d != e) {
        if (d < e) std::swap(d, e);
        work += 6;
        if (uint64_t(d) * 100 <= uint64_t(e) * 296) d -= e;
        else if ((d & 1) == (e & 1)) { d = (d - e) / 2; work += 5; }
        else if (!(d & 1)) { d /= 2; work += 5; }
        else { e /= 2; work += 5; }
    }
    if (d != 1) throw std::runtime_error("non-coprime PRAC seed");
    return work;
}
void choose(EcmPracPrime &entry) {
    if (entry.p == 2) { entry.d = 0; entry.work = 5; return; }
    uint32_t best = UINT32_MAX, chosen = 0;
    uint32_t seen[70]; size_t used = 0;
    for (double ratio : ratios) {
        int64_t center = int64_t(std::ceil(double(entry.p) * ratio));
        for (int64_t candidate = center - 3; candidate <= center + 3; ++candidate) {
            if (candidate <= entry.p / 2 || candidate >= entry.p) continue;
            uint32_t d = uint32_t(candidate);
            if (std::find(seen, seen + used, d) != seen + used) continue;
            seen[used++] = d;
            uint32_t work = cost(entry.p, d);
            if (work < best) { best = work; chosen = d; }
        }
    }
    if (!chosen) throw std::runtime_error("no PRAC seed");
    entry.d = chosen; entry.work = best;
}
std::vector<EcmPracPrime> sieve(uint32_t limit, uint32_t torsion) {
    uint32_t root = uint32_t(std::sqrt(double(limit)));
    std::vector<uint8_t> base(root + 1, 1);
    std::vector<uint32_t> small;
    for (uint32_t p = 2; p <= root; ++p) if (base[p]) {
        small.push_back(p);
        for (uint64_t j = uint64_t(p) * p; j <= root; j += p) base[size_t(j)] = 0;
    }
    std::vector<EcmPracPrime> result;
    auto append = [&](uint32_t p) {
        uint32_t repeats = 1;
        if (p <= root) {
            uint64_t power = p;
            while (power <= limit / p) { power *= p; ++repeats; }
        }
        if (torsion == 12) repeats += p == 2 ? 2 : p == 3 ? 1 : 0;
        result.push_back({p, 0, repeats, 0});
    };
    append(2);
    constexpr uint32_t width = 1u << 19;
    std::vector<uint8_t> segment(width);
    for (uint64_t low = 3; low <= limit; low += 2ull * width) {
        uint64_t high = std::min<uint64_t>(limit, low + 2ull * width - 2);
        size_t count = size_t((high - low) / 2 + 1);
        std::fill(segment.begin(), segment.begin() + count, 1);
        for (uint32_t p : small) if (p != 2) {
            uint64_t start = std::max<uint64_t>(uint64_t(p) * p, ((low + p - 1) / p) * p);
            if (!(start & 1)) start += p;
            for (uint64_t j = start; j <= high; j += 2ull * p) segment[size_t((j - low) / 2)] = 0;
        }
        for (size_t i = 0; i < count; ++i) if (segment[i]) append(uint32_t(low + 2 * i));
    }
    // B1=2 still needs an extra factor 3 in choose12.
    if (limit == 2 && torsion == 12) result.push_back({3, 0, 1, 0});
    return result;
}
}

EcmPracPlan ecm_prac_build(uint32_t B1, uint32_t torsion, mpz_srcptr s,
                          const std::string &cache_dir) {
    if (B1 < 2 || (torsion != 1 && torsion != 12)) throw std::runtime_error("invalid PRAC B1/torsion");
    auto start = std::chrono::steady_clock::now();
    EcmPracPlan plan;
    Header want{magic, B1, ecm_prac_hash_mpz(s), 0, 0, 0, 1, 7, torsion, 0x12345678};
    std::filesystem::path file;
    if (!cache_dir.empty()) file = std::filesystem::path(cache_dir) /
        ("prac_v1_b" + std::to_string(B1) + "_t" + std::to_string(torsion) + "_s7.bin");
    bool hit = false;
    if (!file.empty()) {
        std::ifstream in(file, std::ios::binary); Header h{};
        if (in.read(reinterpret_cast<char *>(&h), sizeof(h)) && h.magic == want.magic &&
            h.version == want.version && h.search == want.search && h.b1 == want.b1 &&
            h.scalar == want.scalar && h.torsion == want.torsion && h.endian == want.endian &&
            h.count > 0 && h.count <= uint64_t(B1) / 2 + 2) {
            std::error_code ec;
            uint64_t bytes = h.count * sizeof(EcmPracPrime);
            if (std::filesystem::file_size(file, ec) == sizeof(h) + bytes && !ec) {
                plan.primes.resize(size_t(h.count));
                if (in.read(reinterpret_cast<char *>(plan.primes.data()), bytes) &&
                    ecm_prac_hash(plan.primes.data(), size_t(bytes)) == h.checksum) {
                    uint64_t sum = 0; bool valid = true;
                    for (const auto &p : plan.primes) {
                        valid &= p.p >= 2 && (p.p <= B1 || (B1 == 2 && torsion == 12 && p.p == 3)) &&
                                 p.repetitions > 0 && p.repetitions <= 34 && p.work > 0 &&
                                 (p.p == 2 ? p.d == 0 : p.d > p.p / 2 && p.d < p.p);
                        sum += uint64_t(p.work) * p.repetitions;
                    }
                    hit = valid && sum == h.work;
                    if (hit) want = h;
                }
            }
        }
    }
    if (!hit) {
        plan.primes = sieve(B1, torsion);
        std::atomic<size_t> next{0};
        unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        if (const char *env = std::getenv("ECM_PRAC_THREADS")) {
            int value = std::atoi(env);
            if (value < 1 || value > 32) throw std::runtime_error("ECM_PRAC_THREADS must be 1..32");
            threads = unsigned(value);
        }
        // Join all started workers even if thread creation or a worker fails.
        std::vector<std::thread> workers;
        workers.reserve(threads);
        std::exception_ptr failure;
        std::mutex failure_mutex;
        auto run = [&] {
            try {
                for (;;) {
                    size_t first = next.fetch_add(1024);
                    if (first >= plan.primes.size()) break;
                    size_t end = std::min(first + 1024, plan.primes.size());
                    for (size_t i = first; i < end; ++i) choose(plan.primes[i]);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                next.store(plan.primes.size());
            }
        };
        try {
            for (unsigned t = 0; t < threads; ++t) workers.emplace_back(run);
        } catch (...) {
            next.store(plan.primes.size());
            for (auto &worker : workers) worker.join();
            throw;
        }
        for (auto &worker : workers) worker.join();
        if (failure) std::rethrow_exception(failure);
        want.count = plan.primes.size();
        for (const auto &p : plan.primes) want.work += uint64_t(p.work) * p.repetitions;
        want.checksum = ecm_prac_hash(plan.primes.data(), plan.primes.size() * sizeof(EcmPracPrime));
        if (!file.empty()) {
            std::error_code ec; std::filesystem::create_directories(file.parent_path(), ec);
            // Cache is optional. A truncated/concurrent file is rejected on the next load.
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(&want), sizeof(want));
            out.write(reinterpret_cast<const char *>(plan.primes.data()), plan.primes.size() * sizeof(EcmPracPrime));
        }
    }
    plan.work = want.work;
    plan.identity = ecm_prac_hash(&want, sizeof(want));
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    plan.status = std::string(hit ? "cache hit" : "built") + ", " + std::to_string(plan.primes.size()) +
        " primes, " + std::to_string(plan.primes.size() * sizeof(EcmPracPrime)) + " bytes, " +
        std::to_string(elapsed) + " s";
    return plan;
}
