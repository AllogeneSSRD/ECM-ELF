#include "runtime_tune.h"

#include "../planning/cpu_tune_section.h"
#include "ecm/io/results.h"
#include "ecm/platform/program_version.h"
#include "io/config_format.h"
#include "io/files.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ecm_cpu_tune {
namespace {
namespace fs = std::filesystem;
namespace json = ecm_results::json;
using namespace ecm_stage2::tune;
constexpr std::size_t max_profile_bytes = 64 * 1024 * 1024;
constexpr std::size_t max_row_bytes = 16 * 1024;
constexpr uint64_t max_repeats = 1000;
constexpr unsigned format = 2;
constexpr const char* columns[] = {
    "cpu", "threads", "gw_version", "modulus_mode", "fft_length",
    "fft_layout", "polymult_isa", "max_exponent", "b1", "b2", "d", "p", "f_tree_splits", "planned_mib",
    "repeats", "mean_seconds", "stddev_seconds", "tested_min", "tested_max", "version", "updated_at",
};

struct Row {
    Key key;
    Candidate plan;
    uint64_t repeats = 0, tested_min = 0, tested_max = 0;
    double mean_seconds = 0, stddev_seconds = 0;
    std::string version, updated_at;
};

auto scope(const Key& key, const Candidate& plan) {
    return std::tie(key.cpu, key.layout.threads, key.layout.smt_extra, key.gw_version, key.mersenne,
                    plan.fft_length, plan.fft_layout, plan.isa, key.b1, key.b2, plan.d,
                    plan.poly_size, plan.f_tree_splits);
}

bool valid(const Row& row) {
    const auto& key = row.key;
    const auto& plan = row.plan;
    const auto& threads = key.layout.threads;
    const auto& extra = key.layout.smt_extra;
    return !key.cpu.empty() && !key.gw_version.empty() && !plan.fft_layout.empty() && !plan.isa.empty() &&
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

Row decode(const std::vector<json::Value>& cells, bool with_smt) {
    if (cells.size() != std::size(columns) + with_smt) throw std::runtime_error("invalid CPU Stage2 tune row size");
    Row row;
    std::size_t i = 0;
    const auto u = [&] { return number<uint64_t>(cells[i++]); };
    const auto s = [&] { return string(cells[i++]); };
    const auto real = [&] { return number<double>(cells[i++]); };
    row.key.cpu = s();
    row.key.layout.threads = counts(cells[i++]);
    row.key.gw_version = s();
    const auto mode = s();
    if (mode != "mersenne" && mode != "general") throw std::runtime_error("invalid CPU Stage2 modulus mode");
    row.key.mersenne = mode == "mersenne";
    row.plan.fft_length = u();
    row.plan.fft_layout = s();
    row.plan.isa = s();
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
};

Profile parse_profile(const std::string& bytes) {
    Profile result;
    bool cpu = false, legacy = false, format_ok = false, matrix = false, header = false, closed = false, with_smt = false;
    for (std::size_t at = bytes.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0; at < bytes.size();) {
        const auto newline = bytes.find('\n', at);
        const auto end = newline == bytes.npos ? bytes.size() : newline + 1;
        auto line = bytes.substr(at, end - at);
        line = ecm_config::trim(line.substr(0, ecm_config::toml_comment_offset(line)));
        if (is_tune_section(line)) {
            cpu = has_tune_header(line, cpu_section);
            legacy = has_tune_header(line, legacy_cpu_section);
            if (cpu || legacy) result.sections.push_back({at, end});
            format_ok = matrix = header = closed = with_smt = false;
        } else if (!line.empty() && (cpu || legacy)) {
            auto& section = result.sections.back();
            section.end = end;
            if (cpu && !closed) {
                const auto equals = line.find('=');
                const auto key = ecm_config::trim(line.substr(0, equals));
                if (!matrix && equals != line.npos && key == "format") {
                    if (ecm_config::trim(line.substr(equals + 1)) != std::to_string(format))
                        throw std::runtime_error("unsupported CPU Stage2 tune matrix format");
                    format_ok = true;
                } else if (!matrix && format_ok && equals != line.npos && key == "matrix") {
                    matrix = ecm_config::trim(line.substr(equals + 1)) == "[";
                } else if (matrix && line == "]") {
                    closed = true;
                } else if (matrix && line.size() <= max_row_bytes) {
                    if (line.back() == ',') line.pop_back();
                    try {
                        const auto values = json::parse(line);
                        if (values.type != 'a') throw std::runtime_error("invalid CPU Stage2 tune array");
                        if (!header) {
                            with_smt = values.items.size() == std::size(columns) + 1 &&
                                       string(values.items.back()) == "smt_extra";
                            if (values.items.size() != std::size(columns) + with_smt)
                                throw std::runtime_error("invalid tune columns");
                            for (std::size_t i = 0; i < std::size(columns); ++i)
                                if (string(values.items[i]) != columns[i]) throw std::runtime_error("invalid tune columns");
                            header = true;
                        } else {
                            auto row = decode(values.items, with_smt);
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

std::string encode(const std::vector<Row>& rows) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    const auto pair = [&](const ThreadCounts& count) {
        out << '[' << count[0] << ", " << count[1] << ']';
    };
    const bool with_smt = std::any_of(rows.begin(), rows.end(), [](const Row& row) {
        return row.key.layout.smt_extra != ThreadCounts{};
    });
    out << cpu_section << "\nformat = " << format << "\nmatrix = [\n    [";
    for (std::size_t i = 0; i < std::size(columns); ++i)
        out << (i ? ", " : "") << json::quote(columns[i]);
    if (with_smt) out << ", \"smt_extra\"";
    out << "],\n" << std::setprecision(12);
    for (const auto& row : rows) {
        const auto& k = row.key;
        const auto& p = row.plan;
        out << "    [" << json::quote(k.cpu) << ", ";
        pair(k.layout.threads);
        out << ", " << json::quote(k.gw_version) << ", " << json::quote(k.mersenne ? "mersenne" : "general")
            << ", " << p.fft_length << ", " << json::quote(p.fft_layout) << ", " << json::quote(p.isa)
            << ", " << p.max_exponent << ", " << k.b1 << ", " << k.b2 << ", " << p.d << ", " << p.poly_size
            << ", " << p.f_tree_splits << ", " << p.planned_mib << ", " << row.repeats << ", " << row.mean_seconds
            << ", " << row.stddev_seconds << ", " << row.tested_min << ", " << row.tested_max
            << ", " << json::quote(row.version) << ", " << json::quote(row.updated_at);
        if (with_smt) {
            out << ", ";
            pair(k.layout.smt_extra);
        }
        out << "],\n";
    }
    out << "]\n";
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
                scope(row.key, row.plan) == scope(key, candidates[i]) &&
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
        return scope(row.key, row.plan) == scope(key, plan);
    });
    if (found != profile.rows.end()) {
        auto& old = *found;
        if (old.repeats >= sample_limit) return;
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
    } else {
        profile.rows.push_back(std::move(update));
    }
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
