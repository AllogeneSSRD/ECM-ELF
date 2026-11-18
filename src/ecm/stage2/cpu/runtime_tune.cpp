#include "runtime_tune.h"

#include "io/config_format.h"
#include "io/files.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace ecm_cpu_tune {
namespace {
namespace fs = std::filesystem;
constexpr std::size_t max_profile_bytes = 64 * 1024 * 1024;
constexpr std::size_t max_row_bytes = 16 * 1024;
constexpr uint64_t max_repeats = 1000;
constexpr std::string_view table_header = "[[cpu_stage2]]";

constexpr struct {
    const char* name;
    double Sample::*value;
} means[] = {
    {"mean_seconds", &Sample::seconds},
    {"init_seconds", &Sample::init_seconds},
    {"main_seconds", &Sample::main_seconds},
    {"baby_seconds", &Sample::baby_seconds},
    {"ftree_seconds", &Sample::ftree_seconds},
    {"reciprocal_seconds", &Sample::reciprocal_seconds},
    {"giant_seconds", &Sample::giant_seconds},
    {"gtree_seconds", &Sample::gtree_seconds},
    {"fold_seconds", &Sample::fold_seconds},
    {"scale_seconds", &Sample::scale_seconds},
    {"descent_seconds", &Sample::descent_seconds},
    {"leaf_gcd_seconds", &Sample::leaf_gcd_seconds},
};

struct Row {
    Key key;
    Candidate candidate;
    Sample mean;
    uint64_t repeats = 0;
};

auto scope(const Key& key) {
    return std::tie(key.cpu, key.build, key.bits, key.exponent, key.b1, key.b2,
                    key.memory_mib, key.threads, key.affinity);
}

auto geometry(const Candidate& candidate) {
    return std::tie(candidate.d, candidate.poly_size, candidate.fft_length);
}

bool valid(const Row& row) {
    for (const auto& field : means)
        if (!std::isfinite(row.mean.*field.value) || row.mean.*field.value < 0)
            return false;
    return !row.key.cpu.empty() && !row.key.build.empty() &&
           row.key.bits >= 2 && row.key.b1 > 0 && row.key.b2 > row.key.b1 &&
           row.key.threads > 0 && row.key.memory_mib > 0 &&
           row.candidate.d >= 6 && row.candidate.d <= 20000000 &&
           row.candidate.d % 2 == 0 && row.candidate.poly_size > 0 &&
           row.candidate.fft_length > 0 && row.repeats > 0 && row.repeats <= max_repeats &&
           row.mean.seconds > 0 && std::isfinite(row.mean.estimated_memory_mib) &&
           row.mean.estimated_memory_mib >= 0 &&
           !row.mean.fft_description.empty() && !row.mean.isa.empty();
}

std::string read_profile(const fs::path& path) {
    if (path.empty() || !fs::exists(path)) return {};
    if (fs::file_size(path) > max_profile_bytes)
        throw std::runtime_error("CPU Stage2 tune profile exceeds 64 MiB");
    auto bytes = ecm_io::read_file(path);
    if (bytes.size() > max_profile_bytes)
        throw std::runtime_error("CPU Stage2 tune profile exceeds 64 MiB");
    return bytes;
}

struct Field {
    std::string value;
    char type = 0;
};

using Fields = std::map<std::string, Field>;

template<class Number>
Number number(const Fields& fields, const char* name, bool integer = false) {
    const auto& field = fields.at(name);
    if (field.type != 'i' && (integer || field.type != 'f'))
        throw std::runtime_error("invalid CPU Stage2 tune number");
    auto begin = field.value.data();
    const auto end = begin + field.value.size();
    if (begin != end && *begin == '+') ++begin;
    Number value{};
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end)
        throw std::runtime_error("invalid CPU Stage2 tune number");
    return value;
}

std::string string(const Fields& fields, const char* name) {
    const auto& field = fields.at(name);
    if (field.type != 's') throw std::runtime_error("invalid CPU Stage2 tune string");
    return field.value;
}

// The shared profile stores single-line table names and scalar CPU fields.
// Track byte ranges so updating one CPU row leaves every other section intact.
template<class Visit>
void visit_rows(std::string_view bytes, Visit visit) {
    const auto missing = std::string_view::npos;
    const auto header = bytes.find(table_header);
    if (header == missing) return;
    const auto previous_newline = bytes.rfind('\n', header);
    const auto first = previous_newline == missing ?
        (bytes.substr(0, 3) == "\xef\xbb\xbf" ? std::size_t{3} : std::size_t{0}) :
        previous_newline + 1;
    std::size_t row_begin = missing, row_end = 0;
    Fields fields;
    bool good = true;
    const auto finish = [&] {
        if (row_begin == missing || !good) return;
        Row row;
        const auto u = [&](const char* key) { return number<uint64_t>(fields, key, true); };
        const auto s = [&](const char* key) { return string(fields, key); };
        try {
            row.key = {s("cpu"), s("build"), u("bits"), u("exponent"), u("b1"), u("b2"),
                       u("memory_mib"), u("threads"), fields.count("affinity") ? s("affinity") : ""};
            row.candidate = {u("d"), u("p"), u("fft_length")};
            row.repeats = u("repeats");
            for (const auto& field : means)
                row.mean.*field.value = number<double>(fields, field.name);
            row.mean.estimated_memory_mib = number<double>(fields, "estimated_memory_mib");
            row.mean.blocks = u("blocks");
            row.mean.giants = u("giants");
            row.mean.fft_description = s("fft_description");
            row.mean.isa = s("isa");
        } catch (const std::runtime_error&) {
            return;
        } catch (const std::out_of_range&) {
            return;
        }
        if (valid(row)) visit(row, row_begin, row_end);
    };

    for (std::size_t at = first; at < bytes.size();) {
        const auto newline = bytes.find('\n', at);
        const auto end = newline == missing ? bytes.size() : newline + 1;
        const auto text = ecm_config::trim(std::string(bytes.substr(at, end - at)));
        if (!text.empty() && text.front() == '[') {
            finish();
            fields.clear();
            good = true;
            auto suffix = std::string_view(text);
            const bool prefix = suffix.substr(0, table_header.size()) == table_header;
            if (prefix) suffix.remove_prefix(table_header.size());
            const auto tail = ecm_config::trim(std::string(suffix));
            row_begin = prefix && (tail.empty() || tail.front() == '#') ? at : missing;
            row_end = end;
        } else if (row_begin != missing && good && !text.empty() && text.front() != '#') {
            row_end = end;
            try {
                const auto equals = text.find('=');
                if (end - row_begin > max_row_bytes || equals == text.npos) {
                    good = false;
                } else {
                    Field field;
                    field.value = ecm_config::toml_value(text.substr(equals + 1), field.type);
                    good = fields.emplace(ecm_config::trim(text.substr(0, equals)),
                                          std::move(field)).second;
                }
            } catch (const std::runtime_error&) {
                good = false;
            }
        }
        at = end;
    }
    finish();
}

std::string encode(const Row& row) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    const auto text = [](const std::string& value) { return ecm_results::json::quote(value); };
    out << table_header << '\n'
        << "cpu = " << text(row.key.cpu) << '\n'
        << "build = " << text(row.key.build) << '\n'
        << "bits = " << row.key.bits << '\n'
        << "exponent = " << row.key.exponent << '\n'
        << "b1 = " << row.key.b1 << '\n'
        << "b2 = " << row.key.b2 << '\n'
        << "memory_mib = " << row.key.memory_mib << '\n'
        << "threads = " << row.key.threads << '\n'
        << "affinity = " << text(row.key.affinity) << '\n'
        << "d = " << row.candidate.d << '\n'
        << "p = " << row.candidate.poly_size << '\n'
        << "fft_length = " << row.candidate.fft_length << '\n'
        << "repeats = " << row.repeats << '\n'
        << "blocks = " << row.mean.blocks << '\n'
        << "giants = " << row.mean.giants << '\n'
        << "fft_description = " << text(row.mean.fft_description) << '\n'
        << "isa = " << text(row.mean.isa) << '\n'
        << std::setprecision(std::numeric_limits<double>::max_digits10) << std::showpoint
        << "estimated_memory_mib = " << row.mean.estimated_memory_mib << '\n';
    for (const auto& field : means)
        out << field.name << " = " << row.mean.*field.value << '\n';
    return out.str();
}
} // namespace

uint64_t select(const fs::path& path, const Key& key, const std::vector<Candidate>& candidates,
                uint64_t min_repeats, bool explore) {
    if (candidates.empty()) return 0;
    if (path.empty() || (explore && candidates.size() == 1))
        return explore ? candidates.front().d : 0;
    struct PlanResult {
        uint64_t repeats = 0;
        double seconds = 0;
    };
    std::vector<PlanResult> results(candidates.size());
    visit_rows(read_profile(path), [&](const Row& row, std::size_t, std::size_t) {
        if (scope(row.key) != scope(key)) return;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (geometry(row.candidate) == geometry(candidates[index]) &&
                row.repeats >= results[index].repeats)
                results[index] = {row.repeats, row.mean.seconds};
        }
    });
    if (explore) {
        const auto next = std::min_element(results.begin(), results.end(),
            [](const PlanResult& a, const PlanResult& b) { return a.repeats < b.repeats; });
        if (next->repeats < min_repeats) return candidates[next - results.begin()].d;
    }
    double fastest = std::numeric_limits<double>::infinity();
    uint64_t winner = explore ? candidates.front().d : 0;
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& measured = results[index];
        if (measured.repeats && measured.repeats >= min_repeats && measured.seconds < fastest) {
            fastest = measured.seconds;
            winner = candidates[index].d;
        }
    }
    return winner;
}

void observe(const fs::path& path, const Key& key, const Candidate& candidate, const Sample& sample,
             uint64_t sample_limit) {
    if (path.empty()) return;
    Row update{key, candidate, sample, 1};
    if (!valid(update) || sample_limit < 1 || sample_limit > max_repeats)
        throw std::runtime_error("invalid CPU Stage2 tune sample");
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string());
    auto bytes = read_profile(path);
    const auto missing = std::string::npos;
    std::size_t begin = missing, end = 0;
    Row old;
    visit_rows(bytes, [&](const Row& row, std::size_t first, std::size_t last) {
        if (scope(row.key) == scope(key) && geometry(row.candidate) == geometry(candidate) &&
            row.repeats >= old.repeats) {
            old = row;
            begin = first;
            end = last;
        }
    });
    if (begin != missing) {
        if (old.repeats >= sample_limit) return;
        update.repeats = old.repeats + 1;
        for (const auto& field : means) {
            const auto previous = old.mean.*field.value;
            update.mean.*field.value = previous + (sample.*field.value - previous) / update.repeats;
        }
        update.mean.estimated_memory_mib = std::max(old.mean.estimated_memory_mib,
                                                   sample.estimated_memory_mib);
        bytes.replace(begin, end - begin, encode(update));
    } else {
        if (!bytes.empty() && bytes.back() != '\n') bytes += '\n';
        if (!bytes.empty()) bytes += '\n';
        bytes += encode(update);
    }
    if (bytes.size() > max_profile_bytes)
        throw std::runtime_error("CPU Stage2 tune profile exceeds 64 MiB");
    ecm_io::write_atomic(path, bytes);
}
} // namespace ecm_cpu_tune
