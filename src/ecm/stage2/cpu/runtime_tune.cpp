#include "runtime_tune.h"

#include "../planning/cpu_tune_section.h"
#include "../planning/tune_matrix_text.h"
#include "ecm/io/results.h"
#include "ecm/platform/program_version.h"
#include "io/config_format.h"
#include "io/files.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ecm_cpu_tune {
CpuIdentity cpu_identity(std::string_view description) {
    constexpr std::string_view marker = " supported:";
    const auto at = description.find(marker);
    if (at == description.npos) return {std::string(description), {}};
    return {std::string(description.substr(0, at)),
            ecm_config::trim(std::string(description.substr(at + marker.size())))};
}

namespace {
namespace fs = std::filesystem;
namespace json = ecm_results::json;
using namespace ecm_stage2::tune;
constexpr std::size_t max_profile_bytes = 64 * 1024 * 1024;
constexpr std::size_t max_row_bytes = 16 * 1024;
constexpr uint64_t max_repeats = 1000;
constexpr unsigned row_cpu_format = 2;
constexpr unsigned format = 3;
constexpr const char* columns[] = {
    "threads", "gw_version", "modulus_mode", "polymult_isa", "fft_length",
    "max_exponent", "b1", "b2", "d", "p", "f_tree_splits", "planned_mib",
    "repeats", "mean_seconds", "stddev_seconds", "tested_min", "tested_max", "version", "updated_at",
};

struct Row {
    Key key;
    Candidate plan;
    uint64_t repeats = 0, tested_min = 0, tested_max = 0;
    double mean_seconds = 0, stddev_seconds = 0;
    std::string version, updated_at;
};

auto cpu_scope(const CpuIdentity& cpu) {
    return std::tie(cpu.name, cpu.supported);
}

auto scope(const Key& key, const Candidate& plan) {
    return std::tuple_cat(cpu_scope(key.cpu), std::tie(key.layout.threads, key.layout.smt_extra, key.gw_version,
                    key.mersenne, plan.isa, plan.fft_length, key.b1, key.b2, plan.d,
                    plan.poly_size, plan.f_tree_splits));
}

bool matches(const Row& row, const Key& key, const Candidate& plan) {
    // The preflight memory estimate also distinguishes different allocation
    // geometries at one FFT length. Allow roundoff from the stored decimal.
    return scope(row.key, row.plan) == scope(key, plan) &&
           std::abs(row.plan.planned_mib - plan.planned_mib) <= plan.planned_mib * 1e-11;
}

bool valid(const Row& row) {
    const auto& key = row.key;
    const auto& plan = row.plan;
    const auto& threads = key.layout.threads;
    const auto& extra = key.layout.smt_extra;
    return !key.cpu.name.empty() && !key.gw_version.empty() && !plan.isa.empty() &&
           (threads[0] || threads[1]) && threads[0] <= 1024 && threads[1] <= 1024 &&
           (threads[0] ? extra[0] < threads[0] : !extra[0]) &&
           (threads[1] ? extra[1] < threads[1] : !extra[1]) &&
           key.b1 && key.b2 > key.b1 && plan.d >= 6 && plan.d <= 20000000 && plan.d % 2 == 0 &&
           plan.poly_size && plan.fft_length && plan.f_tree_splits < 64 &&
           std::isfinite(plan.planned_mib) && plan.planned_mib > 0 &&
           row.tested_min >= 2 && row.tested_min <= row.tested_max && row.tested_max <= plan.max_exponent &&
           row.repeats && row.repeats <= max_repeats && std::isfinite(row.mean_seconds) &&
           row.mean_seconds > 0 && std::isfinite(row.stddev_seconds) && row.stddev_seconds >= 0;
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

template<class Number>
Number number(const json::Value& cell) {
    Number value{};
    const auto begin = cell.scalar.data();
    const auto end = begin + cell.scalar.size();
    if (cell.type != 'd') throw std::runtime_error("invalid CPU Stage2 tune number");
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end)
        throw std::runtime_error("invalid CPU Stage2 tune number");
    return value;
}

std::string string(const json::Value& cell) {
    if (cell.type != 's') throw std::runtime_error("invalid CPU Stage2 tune string");
    return cell.scalar;
}

ThreadCounts counts(const json::Value& cell) {
    if (cell.type != 'a' || cell.items.size() != 2)
        throw std::runtime_error("invalid CPU Stage2 thread pair");
    return {number<unsigned>(cell.items[0]), number<unsigned>(cell.items[1])};
}

std::vector<std::string_view> column_names(bool cpu_in_row) {
    if (!cpu_in_row) return {std::begin(columns), std::end(columns)};
    std::vector<std::string_view> names{"cpu", "threads", "gw_version", "modulus_mode", "fft_length",
                                       "fft_layout", "polymult_isa"};
    names.insert(names.end(), std::begin(columns) + 5, std::end(columns));
    return names;
}

Row decode(const std::vector<json::Value>& cells, bool with_smt, bool cpu_in_row, const CpuIdentity& cpu) {
    if (cells.size() != std::size(columns) + with_smt + (cpu_in_row ? 2 : 0))
        throw std::runtime_error("invalid CPU Stage2 tune row size");
    Row row;
    std::size_t i = 0;
    const auto u = [&] { return number<uint64_t>(cells[i++]); };
    const auto s = [&] { return string(cells[i++]); };
    const auto real = [&] { return number<double>(cells[i++]); };
    row.key.cpu = cpu_in_row ? cpu_identity(s()) : cpu;
    row.key.layout.threads = counts(cells[i++]);
    row.key.gw_version = s();
    const auto mode = s();
    if (mode != "mersenne" && mode != "general") throw std::runtime_error("invalid CPU Stage2 modulus mode");
    row.key.mersenne = mode == "mersenne";
    if (cpu_in_row) {
        row.plan.fft_length = u();
        (void)s();  // V2 stored the human-readable coefficient FFT description.
        row.plan.isa = s();
    } else {
        row.plan.isa = s();
        row.plan.fft_length = u();
    }
    row.plan.max_exponent = u();
    row.key.b1 = u();
    row.key.b2 = u();
    row.plan.d = u();
    row.plan.poly_size = u();
    row.plan.f_tree_splits = number<unsigned>(cells[i++]);
    row.plan.planned_mib = real();
    row.repeats = u();
    row.mean_seconds = real();
    row.stddev_seconds = real();
    row.tested_min = u();
    row.tested_max = u();
    row.version = s();
    row.updated_at = s();
    if (with_smt) row.key.layout.smt_extra = counts(cells[i]);
    return row;
}

struct Profile {
    struct Range { std::size_t begin, end; };
    std::vector<Range> sections;
    std::vector<Row> rows;
    unsigned source_format = 0;
};

Profile parse_profile(const std::string& bytes) {
    Profile result;
    bool cpu_root = false, cpu = false, legacy = false, matrix = false, header = false, closed = false, with_smt = false;
    CpuIdentity cpu_model;
    for (std::size_t at = bytes.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0; at < bytes.size();) {
        const auto newline = bytes.find('\n', at);
        const auto end = newline == bytes.npos ? bytes.size() : newline + 1;
        auto line = bytes.substr(at, end - at);
        line = ecm_config::trim(line.substr(0, ecm_config::toml_comment_offset(line)));
        if (is_tune_section(line)) {
            cpu_root = has_tune_header(line, cpu_section);
            cpu = cpu_root || is_cpu_host_section(line);
            legacy = has_tune_header(line, legacy_cpu_section);
            if (cpu || legacy) result.sections.push_back({at, end});
            matrix = header = closed = with_smt = false;
            cpu_model = {};
        } else if (!line.empty() && (cpu || legacy)) {
            auto& section = result.sections.back();
            section.end = end;
            if (cpu && !closed) {
                const auto equals = line.find('=');
                const auto key = ecm_config::trim(line.substr(0, equals));
                const auto value = equals == line.npos ? std::string{} : ecm_config::trim(line.substr(equals + 1));
                if (!matrix && cpu_root && key == "format") {
                    result.source_format = number<unsigned>(json::parse(value));
                    if (result.source_format != format && result.source_format != row_cpu_format)
                        throw std::runtime_error("unsupported CPU Stage2 tune matrix format");
                } else if (!matrix && key == "cpu") {
                    cpu_model.name = string(json::parse(value));
                } else if (!matrix && key == "supported") {
                    cpu_model.supported = string(json::parse(value));
                } else if (!matrix && key == "matrix") {
                    if (!result.source_format || (result.source_format == format && cpu_model.name.empty()) ||
                        (result.source_format == row_cpu_format && !cpu_root) || value != "[")
                        throw std::runtime_error("invalid CPU Stage2 tune table");
                    matrix = true;
                } else if (matrix && line == "]") {
                    closed = true;
                } else if (matrix && line.size() <= max_row_bytes) {
                    if (line.back() == ',') line.pop_back();
                    try {
                        const auto values = json::parse(line);
                        if (values.type != 'a') throw std::runtime_error("invalid CPU Stage2 tune array");
                        const bool cpu_in_row = result.source_format == row_cpu_format;
                        if (!header) {
                            const auto names = column_names(cpu_in_row);
                            with_smt = values.items.size() == names.size() + 1 &&
                                       string(values.items.back()) == "smt_extra";
                            if (values.items.size() != names.size() + with_smt)
                                throw std::runtime_error("invalid tune columns");
                            for (std::size_t i = 0; i < names.size(); ++i)
                                if (string(values.items[i]) != names[i])
                                    throw std::runtime_error("invalid tune columns");
                            header = true;
                        } else {
                            auto row = decode(values.items, with_smt, cpu_in_row, cpu_model);
                            if (valid(row)) result.rows.push_back(std::move(row));
                        }
                    } catch (const std::runtime_error&) {
                        if (!header) throw;
                        // A malformed observation does not invalidate the other rows.
                    }
                }
            }
        }
        at = end;
    }
    return result;
}

std::string real_literal(double value) {
    char buffer[64];
    const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value, std::chars_format::general, 12);
    if (result.ec != std::errc()) throw std::runtime_error("cannot format CPU Stage2 tune number");
    return {buffer, result.ptr};
}

std::string thread_pair(const ThreadCounts& count) {
    return "[" + std::to_string(count[0]) + ", " + std::to_string(count[1]) + "]";
}

void encode_host(std::ostream& out, std::vector<Row>::const_iterator first, std::vector<Row>::const_iterator last) {
    const bool with_smt = std::any_of(first, last, [](const Row& row) {
        return row.key.layout.smt_extra != ThreadCounts{};
    });
    out << "cpu = " << json::quote(first->key.cpu.name) << '\n'
        << "supported = " << json::quote(first->key.cpu.supported) << '\n';
    std::vector<std::vector<std::string>> matrix(1);
    matrix.reserve(static_cast<std::size_t>(last - first) + 1);
    for (const auto* column : columns) matrix.front().push_back(json::quote(column));
    if (with_smt) matrix.front().push_back("\"smt_extra\"");
    for (auto it = first; it != last; ++it) {
        const auto& row = *it;
        const auto& k = row.key;
        const auto& p = row.plan;
        using std::to_string;
        std::vector<std::string> cells{
            thread_pair(k.layout.threads), json::quote(k.gw_version), json::quote(k.mersenne ? "mersenne" : "general"),
            json::quote(p.isa), to_string(p.fft_length), to_string(p.max_exponent),
            to_string(k.b1), to_string(k.b2), to_string(p.d), to_string(p.poly_size), to_string(p.f_tree_splits),
            real_literal(p.planned_mib), to_string(row.repeats), real_literal(row.mean_seconds),
            real_literal(row.stddev_seconds), to_string(row.tested_min), to_string(row.tested_max),
            json::quote(row.version), json::quote(row.updated_at)
        };
        if (with_smt) cells.push_back(thread_pair(k.layout.smt_extra));
        matrix.push_back(std::move(cells));
    }
    write_aligned_matrix(out, matrix);
}

std::string encode(const std::vector<Row>& rows) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << cpu_section << "\nformat = " << format << '\n';
    std::size_t host = 0;
    for (auto first = rows.begin(); first != rows.end();) {
        const auto last = std::find_if(first, rows.end(), [&](const Row& row) {
            return cpu_scope(row.key.cpu) != cpu_scope(first->key.cpu);
        });
        if (host) out << '\n' << cpu_host_prefix << host << "]\n";
        encode_host(out, first, last);
        first = last;
        ++host;
    }
    return out.str();
}
} // namespace

uint64_t select(const fs::path& path, const Key& key, const std::vector<Candidate>& candidates,
                uint64_t min_repeats, bool explore) {
    if (candidates.empty()) return 0;
    if (path.empty() || (explore && candidates.size() == 1))
        return explore ? candidates.front().d : 0;
    const auto profile = parse_profile(read_profile(path));
    std::vector<const Row*> results(candidates.size(), nullptr);
    for (const auto& row : profile.rows) {
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].fft_exponent <= row.plan.max_exponent &&
                matches(row, key, candidates[i]) &&
                (!results[i] || row.repeats > results[i]->repeats))
                results[i] = &row;
        }
    }
    if (explore) {
        const auto next = std::min_element(results.begin(), results.end(), [](const Row* a, const Row* b) {
            return (a ? a->repeats : 0) < (b ? b->repeats : 0);
        });
        if (!*next || (*next)->repeats < min_repeats) return candidates[next - results.begin()].d;
    }
    double fastest = std::numeric_limits<double>::infinity();
    uint64_t winner = explore ? candidates.front().d : 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto measured = results[i];
        if (measured && measured->repeats >= min_repeats && measured->mean_seconds < fastest) {
            fastest = measured->mean_seconds;
            winner = candidates[i].d;
        }
    }
    return winner;
}

void observe(const fs::path& path, const Key& key, const Candidate& plan, double seconds,
             uint64_t sample_limit) {
    if (path.empty()) return;
    Row update{key, plan, 1, key.width, key.width, seconds, 0, ecm_program::version,
               ecm_results::utc_timestamp() + " UTC"};
    if (!valid(update) || plan.fft_exponent > plan.max_exponent || sample_limit < 1 || sample_limit > max_repeats)
        throw std::runtime_error("invalid CPU Stage2 tune sample");
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string());
    auto bytes = read_profile(path);
    auto profile = parse_profile(bytes);
    auto found = std::find_if(profile.rows.begin(), profile.rows.end(), [&](const auto& row) {
        return matches(row, key, plan);
    });
    if (found == profile.rows.end()) {
        profile.rows.push_back(std::move(update));
    } else if (found->repeats < sample_limit) {
        auto& old = *found;
        update.repeats = old.repeats + 1;
        const double delta = seconds - old.mean_seconds;
        update.mean_seconds = old.mean_seconds + delta / update.repeats;
        const double m2 = old.stddev_seconds * old.stddev_seconds * (old.repeats - 1) +
                          delta * (seconds - update.mean_seconds);
        update.stddev_seconds = std::sqrt(std::max(0.0, m2) / old.repeats);
        update.tested_min = std::min(old.tested_min, key.width);
        update.tested_max = std::max(old.tested_max, key.width);
        update.plan.max_exponent = std::max(old.plan.max_exponent, plan.max_exponent);
        update.plan.planned_mib = std::max(old.plan.planned_mib, plan.planned_mib);
        old = std::move(update);
    } else if (profile.source_format != row_cpu_format) return;
    std::sort(profile.rows.begin(), profile.rows.end(), [](const auto& a, const auto& b) {
        return scope(a.key, a.plan) < scope(b.key, b.plan);
    });
    const auto insertion = profile.sections.empty() ? bytes.size() : profile.sections.front().begin;
    for (auto it = profile.sections.rbegin(); it != profile.sections.rend(); ++it)
        bytes.erase(it->begin, it->end - it->begin);
    auto matrix = encode(profile.rows);
    if (insertion && !(insertion == 3 && bytes.compare(0, 3, "\xef\xbb\xbf") == 0)) {
        if (bytes[insertion - 1] != '\n') matrix.insert(matrix.begin(), '\n');
        if (insertion < 2 || bytes[insertion - 2] != '\n') matrix.insert(matrix.begin(), '\n');
    }
    bytes.insert(insertion, matrix);
    if (bytes.size() > max_profile_bytes)
        throw std::runtime_error("CPU Stage2 tune profile exceeds 64 MiB");
    ecm_io::write_atomic(path, bytes);
}
} // namespace ecm_cpu_tune
