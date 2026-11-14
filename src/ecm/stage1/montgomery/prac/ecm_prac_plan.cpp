// Prime95 ecm.cpp:2645/2857 simplified PRAC search, priced for normalized CGBN.
#include "ecm_prac_plan.h"
#include "io/files.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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
// This layout remains the input to the checkpoint identity hash.
struct PlanIdentity {
    uint64_t magic, b1, scalar, count, work, checksum;
    uint32_t version, search, torsion, endian;
};
static_assert(sizeof(PlanIdentity) == 64);

struct SeedHeader {
    PlanIdentity plan;
    EcmPracCost ladder, prac;
    uint64_t checksum;
};
static_assert(sizeof(SeedHeader) == 120);

template<class Value>
bool read(std::istream &in, Value &value) {
    return bool(in.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

template<class Value>
void write(std::ostream &out, const Value &value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint64_t header_checksum(const SeedHeader &header) {
    return ecm_prac_hash(&header, offsetof(SeedHeader, checksum));
}

bool same_cost(const EcmPracCost &a, const EcmPracCost &b) {
    return a.dbl == b.dbl && a.add == b.add && a.work == b.work;
}

std::string costs(const EcmPracCost &ladder, const EcmPracCost &prac) {
    auto describe = [](const EcmPracCost &value) {
        return "D=" + std::to_string(value.dbl) + " A=" + std::to_string(value.add) +
               " W=" + std::to_string(value.work);
    };
    return "ladder " + describe(ladder) + "; PRAC " + describe(prac);
}

bool same_key(const PlanIdentity &a, const PlanIdentity &b) {
    return a.magic == b.magic && a.version == b.version && a.search == b.search &&
           a.b1 == b.b1 && a.scalar == b.scalar && a.torsion == b.torsion && a.endian == b.endian;
}

constexpr uint64_t magic = 0x31504e414c504345ull;
constexpr uint32_t endian = 0x12345678;
bool read_seed_header(std::istream &in, SeedHeader &header) {
    if (!read(in, header) || header.checksum != header_checksum(header)) return false;
    const auto &p = header.plan;
    const auto &l = header.ladder;
    const auto &c = header.prac;
    return p.magic == magic && p.version == 2 && p.endian == endian &&
           p.b1 >= 2 && p.b1 <= UINT32_MAX && (p.torsion == 1 || p.torsion == 12) &&
           p.search >= 1 && p.search <= 50 && p.count > 0 && p.count <= p.b1 / 2 + 2 &&
           l.dbl > 0 && l.dbl <= 32 * p.b1 + 8 && l.add == l.dbl - 1 &&
           l.work == 5 * (l.dbl + l.add) &&
           c.dbl <= 256 * p.count && c.add <= 256 * p.count &&
           c.work == 5 * c.dbl + 6 * c.add && c.work == p.work;
}

bool has_size(const std::filesystem::path &file, uint64_t size) {
    std::error_code error;
    return std::filesystem::file_size(file, error) == size && !error;
}

bool write_plan(const std::filesystem::path& file, const SeedHeader& header,
                const std::vector<EcmPracPrime>& primes) {
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    if (error)
        return false;
#ifdef _WIN32
    const auto process = _getpid();
#else
    const auto process = getpid();
#endif
    static std::atomic<uint64_t> sequence{0};
    auto temporary = file;
    temporary += ".tmp." + std::to_string(process) + "." + std::to_string(sequence.fetch_add(1));
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    write(out, header);
    uint32_t seeds[4096];
    for (size_t first = 0; first < primes.size(); first += std::size(seeds)) {
        const size_t count = std::min(std::size(seeds), primes.size() - first);
        for (size_t i = 0; i < count; ++i) seeds[i] = primes[first + i].d;
        out.write(reinterpret_cast<const char*>(seeds), count * sizeof(seeds[0]));
        if (!out) break;
    }
    out.close();
    if (out && ecm_replace_file(temporary, file))
        return true;
    std::filesystem::remove(temporary, error);
    return false;
}
constexpr double ratios[] = {0.6180339887498948, 0.7236067977499790,
    0.5801787282954641, 0.6328398060887063, 0.6124299495094950,
    0.6201819808074158, 0.6172146165344039, 0.6183471196562281,
    0.6179144065288179, 0.6180796684698958};

EcmPracCost chain_cost(uint32_t p, uint32_t initial) {
    if (p == 2) {
        if (initial != 0) throw std::runtime_error("invalid seed for PRAC doubling");
        return {1, 0, 5};
    }
    if (initial <= p / 2 || initial >= p) throw std::runtime_error("invalid PRAC seed");
    uint32_t e = p - initial, d = initial - e, dbl = 1, add = 1;
    while (d != e) {
        if (d < e) std::swap(d, e);
        ++add;
        if (uint64_t(d) * 100 <= uint64_t(e) * 296) {
            d -= e;
        } else {
            ++dbl;
            if ((d & 1) == (e & 1)) d = (d - e) / 2;
            else if (!(d & 1)) d /= 2;
            else e /= 2;
        }
    }
    if (d != 1) throw std::runtime_error("non-coprime PRAC seed");
    return {dbl, add, 5 * dbl + 6 * add};
}
EcmPracCost choose(EcmPracPrime &entry, unsigned search) {
    if (entry.p == 2) {
        entry.work = 5;
        return {1, 0, 5};
    }
    EcmPracCost best{0, 0, UINT64_MAX};
    struct Window { int64_t low, high; };
    Window previous[std::size(ratios)];
    size_t used = 0;
    for (double ratio : ratios) {
        const auto first = int64_t(std::ceil(double(entry.p) * ratio)) - search / 2;
        const Window window{std::max(first, int64_t(entry.p / 2) + 1),
                            std::min(first + search - 1, int64_t(entry.p) - 1)};
        for (int64_t r = window.low; r <= window.high; ++r) {
            int64_t skip = r - 1;
            for (size_t i = 0; i < used; ++i)
                if (previous[i].low <= r && r <= previous[i].high)
                    skip = std::max(skip, previous[i].high);
            if (skip >= r) {
                r = skip;
                continue;
            }
            const auto cost = chain_cost(entry.p, uint32_t(r));
            if (cost.work < best.work) {
                best = cost;
                entry.d = uint32_t(r);
            }
        }
        if (window.low <= window.high) previous[used++] = window;
    }
    if (best.work == UINT64_MAX) throw std::runtime_error("no PRAC seed");
    entry.work = uint32_t(best.work);
    return best;
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
void accumulate(EcmPracCost &total, const EcmPracCost &cost, uint32_t repetitions) {
    total.dbl += cost.dbl * repetitions;
    total.add += cost.add * repetitions;
    total.work += cost.work * repetitions;
}

template<class Operation>
EcmPracCost process(std::vector<EcmPracPrime> &primes, unsigned threads, Operation operation) {
    EcmPracCost total;
    if (threads == 1) {
        for (auto &p : primes) accumulate(total, operation(p), p.repetitions);
        return total;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    std::exception_ptr failure;
    std::mutex mutex;
    auto run = [&] {
        EcmPracCost local;
        try {
            for (;;) {
                const size_t first = next.fetch_add(1024);
                if (first >= primes.size()) break;
                const size_t end = std::min(first + 1024, primes.size());
                for (size_t i = first; i < end; ++i)
                    accumulate(local, operation(primes[i]), primes[i].repetitions);
            }
            const std::lock_guard<std::mutex> lock(mutex);
            accumulate(total, local, 1);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
            next.store(primes.size());
        }
    };
    try {
        for (unsigned i = 0; i < threads; ++i) workers.emplace_back(run);
    } catch (...) {
        next.store(primes.size());
        for (auto &worker : workers) worker.join();
        throw;
    }
    for (auto &worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    return total;
}

bool read_seeds(std::istream &in, std::vector<EcmPracPrime> &primes) {
    uint32_t seeds[4096];
    for (size_t first = 0; first < primes.size(); first += std::size(seeds)) {
        const size_t count = std::min(std::size(seeds), primes.size() - first);
        if (!in.read(reinterpret_cast<char*>(seeds), count * sizeof(seeds[0]))) return false;
        for (size_t i = 0; i < count; ++i) {
            auto &p = primes[first + i];
            if (p.p == 2 ? seeds[i] != 0 : (seeds[i] <= p.p / 2 || seeds[i] >= p.p))
                return false;
            p.d = seeds[i];
        }
    }
    return true;
}

uint64_t descriptor_hash(const std::vector<EcmPracPrime> &primes) {
    return ecm_prac_hash(primes.data(), primes.size() * sizeof(EcmPracPrime));
}

bool read_legacy(const std::filesystem::path &file, const PlanIdentity &key,
                 std::vector<EcmPracPrime> &primes, uint64_t &expected_hash) {
    std::ifstream in(file, std::ios::binary);
    PlanIdentity header{};
    if (!read(in, header) || !same_key(header, key) || header.count != primes.size() ||
        !has_size(file, sizeof(header) + header.count * sizeof(EcmPracPrime))) return false;
    EcmPracPrime records[4096];
    uint64_t sum = 0, hash = 14695981039346656037ull;
    for (size_t first = 0; first < primes.size(); first += std::size(records)) {
        const size_t count = std::min(std::size(records), primes.size() - first);
        const size_t bytes = count * sizeof(records[0]);
        if (!in.read(reinterpret_cast<char*>(records), bytes)) return false;
        hash = ecm_prac_hash(records, bytes, hash);
        for (size_t i = 0; i < count; ++i) {
            auto &p = primes[first + i];
            const auto &r = records[i];
            if (r.p != p.p || r.repetitions != p.repetitions ||
                (r.p == 2 ? r.d != 0 : (r.d <= r.p / 2 || r.d >= r.p))) return false;
            p.d = r.d;
            p.work = r.work;
            sum += uint64_t(r.work) * r.repetitions;
        }
    }
    expected_hash = header.checksum;
    return sum == header.work && hash == expected_hash;
}
}

unsigned ecm_prac_search_width(const char *option) {
    if (!option) return 7;
    const std::string_view input(option);
    unsigned value = 0;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size() || value < 1 || value > 50)
        throw std::runtime_error("PRAC search width must be 1..50");
    return value;
}

std::string ecm_prac_cache_summary(const std::string &file) {
    const auto path = std::filesystem::u8path(file);
    std::ifstream in(path, std::ios::binary);
    SeedHeader header{};
    if (!read_seed_header(in, header) ||
        !has_size(path, sizeof(header) + header.plan.count * sizeof(uint32_t)))
        throw std::runtime_error("invalid PRAC v2 header or file size");
    return "B1=" + std::to_string(header.plan.b1) + " torsion=" + std::to_string(header.plan.torsion) +
           " search=" + std::to_string(header.plan.search) + " seeds=" + std::to_string(header.plan.count) +
           " bytes=" + std::to_string(sizeof(header) + header.plan.count * sizeof(uint32_t)) + "\n" +
           costs(header.ladder, header.prac);
}

EcmPracPlan ecm_prac_build(uint32_t B1, uint32_t torsion, mpz_srcptr s,
                          const std::string &cache_dir, unsigned max_threads, unsigned search) {
    if (B1 < 2 || (torsion != 1 && torsion != 12) || search < 1 || search > 50)
        throw std::runtime_error("invalid PRAC B1/torsion/search");
    const auto start = std::chrono::steady_clock::now();
    EcmPracPlan plan;
    const uint64_t bits = mpz_sizeinbase(s, 2);
    plan.ladder = {bits, bits - 1, 10 * bits - 5};
    PlanIdentity key{magic, B1, ecm_prac_hash_mpz(s), 0, 0, 0, 2, search, torsion, endian};
    std::filesystem::path file, legacy;
    if (!cache_dir.empty()) {
        const auto root = std::filesystem::u8path(cache_dir);
        const auto suffix = "_b" + std::to_string(B1) + "_t" + std::to_string(torsion) +
                            "_s" + std::to_string(search) + ".bin";
        file = root / ("prac_v2" + suffix);
        if (search == 7) legacy = root / ("prac_v1" + suffix);
    }
    unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    if (max_threads != 0) {
        threads = std::min(threads, max_threads);
    } else if (const char *env = std::getenv("ECM_PRAC_THREADS")) {
        const int value = std::atoi(env);
        if (value < 1 || value > 32) throw std::runtime_error("ECM_PRAC_THREADS must be 1..32");
        threads = unsigned(value);
    }
    plan.primes = sieve(B1, torsion);
    key.count = plan.primes.size();
    auto replay = [](EcmPracPrime &p) {
        const auto cost = chain_cost(p.p, p.d);
        p.work = uint32_t(cost.work);
        return cost;
    };

    uint64_t checksum = 0;
    if (!file.empty()) {
        std::ifstream in(file, std::ios::binary);
        SeedHeader header{};
        if (read_seed_header(in, header) && same_key(header.plan, key) &&
            header.plan.count == key.count && same_cost(plan.ladder, header.ladder) &&
            has_size(file, sizeof(header) + key.count * sizeof(uint32_t)) && read_seeds(in, plan.primes)) {
            plan.prac = process(plan.primes, threads, replay);
            checksum = descriptor_hash(plan.primes);
            plan.cache_hit = checksum == header.plan.checksum && same_cost(plan.prac, header.prac);
        }
    }
    bool migrated = false;
    if (!plan.cache_hit && !legacy.empty()) {
        auto old_key = key;
        old_key.version = 1;
        uint64_t old_hash = 0;
        if (read_legacy(legacy, old_key, plan.primes, old_hash)) {
            plan.prac = process(plan.primes, threads, replay);
            checksum = descriptor_hash(plan.primes);
            migrated = checksum == old_hash;
            plan.cache_hit = migrated;
        }
    }
    if (!plan.cache_hit) {
        plan.prac = process(plan.primes, threads, [search](EcmPracPrime &p) { return choose(p, search); });
        checksum = descriptor_hash(plan.primes);
    }
    key.work = plan.work = plan.prac.work;
    key.checksum = checksum;
    if (!file.empty() && (!plan.cache_hit || migrated)) {
        SeedHeader header{key, plan.ladder, plan.prac, 0};
        header.checksum = header_checksum(header);
        plan.cache_written = write_plan(file, header, plan.primes);
    }

    // Hash the legacy logical descriptor identity for search 7, regardless of storage layout.
    if (search == 7) key.version = 1;
    plan.identity = ecm_prac_hash(&key, sizeof(key));
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    plan.status = std::string(plan.cache_hit ? "cache hit" : "built") +
        (migrated ? " (v1)" : "") + ", search=" + std::to_string(search) + ", " +
        std::to_string(plan.primes.size()) + " primes, " +
        std::to_string(plan.primes.size() * sizeof(EcmPracPrime)) + " descriptor bytes, " +
        std::to_string(elapsed) + " s, " + costs(plan.ladder, plan.prac);
    return plan;
}
