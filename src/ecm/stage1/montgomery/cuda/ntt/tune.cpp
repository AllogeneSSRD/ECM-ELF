#include "tune.h"
#include "ladder.h"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../../../io/files.h"
#include "../../../../../io/json.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/stop.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <map>
#include <optional>
#include <random>
#include <sstream>

namespace ecm_mersenne_ntt {
namespace {
using ecm_stage1::Stage1NttOptions;
using ecm_results::json::Value;
constexpr unsigned max_batch = ecm_stage1::stage1_ntt_max_curves;
constexpr double plateau = 1.05, near_peak = 0.95;
constexpr const char* modes[] = {"linear", "weighted"};

struct Device {
    int index;
    std::string uuid, name;
    size_t budget;
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
    return {index, uuid, prop.name, memory_budget(free, memory_mib)};
}

std::string cache_key(const Device& device, unsigned p, const Stage1NttOptions& o) {
    std::ostringstream key;
    key << device.uuid << ':' << p << ':' << o.length << ':' << o.bpw << ':' << o.tile << ':'
        << o.outer_max << ':' << o.profile << ':' << o.window << ':' << o.warp_tail << ':' << o.memory_mib;
    return key.str();
}

Value object() { Value v; v.type = 'o'; return v; }
Value number(uint64_t n) { return Value::number(std::to_string(n)); }
Value decimal(double n) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(12) << n;
    return Value::number(stream.str());
}

Value load(const std::string& path) {
    if (path.empty() || !std::filesystem::exists(path))
        return object();
    auto db = ecm_results::json::parse(ecm_io::read_file(path));
    if (db.type != 'o' || db.get("schema").text() != "1" || db.get("profiles").type != 'o')
        throw std::runtime_error("invalid Stage1 NTT tune file schema: " + path);
    return db;
}

void persist(const std::string& path, const std::string& key, const Value& entry,
             unsigned short_batch = 0) {
    if (path.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    ecm_io::FileLock lock(path);
    auto db = load(path);
    auto profiles = db.get("profiles");
    if (profiles.type != 'o') profiles = object();
    auto profile = entry;
    const auto& existing = profiles.get(key);
    auto short_rows = existing.get("short");
    if (short_batch) {
        profile = existing;
        if (profile.type != 'o') profile = object();
        if (short_rows.type != 'o') short_rows = object();
        short_rows.set(std::to_string(short_batch), entry);
    }
    if (short_rows.type == 'o')
        profile.set("short", std::move(short_rows));
    db.set("schema", number(1));
    profiles.set(key, std::move(profile));
    db.set("profiles", std::move(profiles));
    ecm_io::write_atomic(path, db.dump() + "\n");
}

struct Measurement {
    unsigned batch = 0;
    Plan plan{};
    std::vector<double> ms;
    size_t owned = 0, observed = 0;
    double time() const {
        auto ordered = ms;
        std::sort(ordered.begin(), ordered.end());
        const auto middle = ordered.size() / 2;
        return ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
    }
    double throughput() const { return 1000.0 * batch / time(); }
};

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

Value encode(const Measurement& m) {
    auto out = object(), plan = object();
    out.set("batch", number(m.batch));
    out.set("ms", decimal(m.time()));
    out.set("curves_per_second", decimal(m.throughput()));
    out.set("repeats", number(m.ms.size()));
    out.set("owned_bytes", number(m.owned));
    out.set("observed_bytes", number(m.observed));
    out.set("peak_bytes", number(std::max(m.owned, m.observed)));
    plan.set("length", number(m.plan.length));
    plan.set("bpw", number(m.plan.bpw));
    plan.set("tile", number(m.plan.tile));
    plan.set("outer_max", number(m.plan.outer_max));
    plan.set("graph_bits", number(m.plan.window));
    plan.set("warp_tail", number(m.plan.warp_tail));
    plan.set("forward", Value::string(m.plan.forward));
    plan.set("inverse", Value::string(m.plan.inverse));
    out.set("plan", std::move(plan));
    return out;
}

struct Cancelled : std::runtime_error { Cancelled() : std::runtime_error("Stage1 NTT tune stopped") {} };

class Calibration {
    const Device& device_;
    unsigned p_, bits_;
    std::vector<uint32_t> scalar_, initial_;
    std::map<unsigned, std::vector<uint32_t>> reference_;
public:
    unsigned runs = 0;
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
        control.describe = [&](const Plan& plan) { m.plan = plan; };
        control.poll = [](unsigned) { return Control::Action{false, ecm_stage1_stop_requested()}; };
        ++runs;
        auto run = ladder(device_.index, p_, batch, active, scalar_, bits_, options, control, samples);
        if (run.stopped) throw Cancelled();
        const auto found = reference_.find(batch);
        if (found == reference_.end()) reference_.emplace(batch, std::move(run.points));
        else if (found->second != run.points) throw std::runtime_error("NTT tune endpoint mismatch");
        m.ms = std::move(run.device_ms);
        m.owned = run.owned_bytes;
        m.observed = run.observed_bytes;
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

    Value search(bool exhaustive) {
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
        const auto& best = peak();
        const Measurement* recommended = &best;
        auto result = object(), samples = object();
        for (const auto& [batch, row] : rows_) {
            if (!row) continue;
            if (row->throughput() >= near_peak * best.throughput() && batch < recommended->batch)
                recommended = &*row;
            samples.set(std::to_string(batch), encode(*row));
        }
        result.set("peak_batch", number(best.batch));
        result.set("recommended_batch", number(recommended->batch));
        result.set("single_curve", encode(*rows_.at(1)));
        result.set("peak", encode(best));
        result.set("recommended", encode(*recommended));
        result.set("samples", std::move(samples));
        return result;
    }
};

Value context(const Device& device, unsigned p, unsigned bits) {
    auto row = object();
    row.set("gpu", Value::string(device.name));
    row.set("uuid", Value::string(device.uuid));
    row.set("p", number(p));
    row.set("bits", number(bits));
    row.set("version", Value::string(ecm_program::version));
    row.set("budget_bytes", number(device.budget));
    return row;
}

struct Cached {
    Stage1NttOptions options;
    double ms;
};

std::optional<Cached> decode(const Value& sample, bool weighted, size_t budget,
                             const Stage1NttOptions& base) {
    if (sample.type != 'o') return {};
    try {
        if (std::stoull(sample.get("peak_bytes").text()) > budget) return {};
        const auto& plan = sample.get("plan");
        auto bounded = [&](const char* key, unsigned lo, unsigned hi) {
            const auto text = plan.get(key).text();
            size_t end = 0;
            const auto value = std::stoul(text, &end);
            if (end != text.size() || value < lo || value > hi)
                throw std::invalid_argument(key);
            return unsigned(value);
        };
        auto o = base;
        o.automatic = false;
        o.weighted = weighted;
        o.length = bounded("length", 2, 65536);
        o.bpw = bounded("bpw", 1, 31);
        o.tile = bounded("tile", 1, 12);
        o.outer_max = bounded("outer_max", 1, 4);
        o.window = int(bounded("graph_bits", 0, 64));
        o.warp_tail = int(bounded("warp_tail", 0, 1));
        const double ms = std::stod(sample.get("ms").text());
        if (!std::isfinite(ms) || ms <= 0) return {};
        return Cached{std::move(o), ms};
    } catch (const std::exception&) {
        return {};
    }
}

} // namespace

int tune_command(int index, const Stage1NttOptions& options, unsigned p, bool exhaustive) {
    try {
        const auto device = device_info(index, options.memory_mib);
        // One representative exponent per default transform width. An exact
        // width can always be added with --bits.
        std::vector<unsigned> widths{61,127,251,521,1279,2539,4003,8191,16001,32749,65521,131071,262139};
        if (p) widths = {p};
        constexpr unsigned bits = 192;
        for (const unsigned width : widths) {
            if (width < 32 || width > 262144)
                throw std::invalid_argument("--tune ntt --bits requires 32..262144");
            const auto start = now_s();
            Calibration cal(device, width, bits);
            auto row = context(device, width, bits);
            row.set("search", Value::string(exhaustive ? "exhaustive" : "adaptive"));
            std::array<double, 2> score{};
            for (unsigned mode = 0; mode < 2; ++mode) {
                auto candidate = options;
                candidate.automatic = false;
                candidate.weighted = mode != 0;
                auto result = BatchSearch(cal, candidate).search(exhaustive);
                if (result.type == 'o')
                    score[mode] = std::stod(result.get("peak").get("curves_per_second").text());
                row.set(modes[mode], std::move(result));
            }
            if (score[0] == 0 && score[1] == 0)
                throw std::runtime_error("no safe NTT backend for the requested shape or VRAM budget");
            row.set("peak_backend", Value::string(modes[score[1] > score[0]]));
            row.set("runs", number(cal.runs));
            row.set("elapsed_seconds", decimal(now_s() - start));
            persist(options.tune_file, cache_key(device, width, options), row);
            ecm_ts_fprintf(stdout, "stage1_ntt_tune: %s\n", row.dump().c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}

Stage1NttOptions select_tuned_plan(int index, unsigned p, unsigned batch,
                                   const Stage1NttOptions& options) {
    if (!options.automatic) return options;
    const auto device = device_info(index, options.memory_mib);
    const auto key = cache_key(device, p, options), b = std::to_string(batch);
    std::array<std::optional<Cached>, 2> choices;
    auto read = [&](const Value& row) {
        for (unsigned i = 0; i < 2; ++i) choices[i] = decode(row.get(modes[i]), i != 0, device.budget, options);
    };
    const char* source = "tune";
    try {
        const auto db = load(options.tune_file);
        const auto& group = db.get("profiles").get(key);
        for (unsigned i = 0; i < 2; ++i)
            choices[i] = decode(group.get(modes[i]).get("samples").get(b), i != 0, device.budget, options);
        // A mode absent from the full search may simply have skipped this batch.
        // A short row has already tried both modes at the exact batch.
        if (!choices[0] || !choices[1]) read(group.get("short").get(b));
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "Stage1 NTT tune read: %s\n", e.what());
    }
    if (!choices[0] && !choices[1]) {
        source = "short_tune";
        const auto start = now_s();
        constexpr unsigned bits = 96;
        Calibration cal(device, p, bits);
        auto exact = context(device, p, bits);
        for (unsigned i = 0; i < 2; ++i) {
            auto candidate = options;
            candidate.automatic = false;
            candidate.weighted = i != 0;
            auto row = measure_batch(cal, batch, candidate);
            exact.set(modes[i], row ? encode(*row) : Value{});
        }
        read(exact);
        if (choices[0] || choices[1]) {
            try { persist(options.tune_file, key, exact, batch); }
            catch (const std::exception& e) { ecm_ts_fprintf(stderr, "Stage1 NTT tune write: %s\n", e.what()); }
        }
        ecm_ts_fprintf(stdout, "stage1_ntt_short_tune: p=%u batch=%u seconds=%.3f\n", p, batch, now_s() - start);
    }
    if (!choices[0] && !choices[1]) throw BudgetError("no safe Stage1 NTT auto plan under the current VRAM budget");
    const auto winner = choices[1] && (!choices[0] || choices[1]->ms < choices[0]->ms) ? 1 : 0;
    ecm_ts_fprintf(stdout, "stage1_ntt_selection: source=%s p=%u batch=%u convolution=%s\n",
                   source, p, batch, modes[winner]);
    return choices[winner]->options;
}

} // namespace ecm_mersenne_ntt
