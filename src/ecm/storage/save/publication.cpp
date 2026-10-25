#include "publication.h"
#include "ecm_edwards_io.h"
#include "../../application/configuration/ecm_expr.h"
#include "sync/files.h"
#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ecm_save {
static std::string key(std::string line) {
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    return line.substr(0, line.find("; PROGRAM="));
}

template <class Visitor> static uint64_t visit_records(const std::string& bytes, Visitor visitor) {
    const std::string_view text(bytes);
    size_t at = 0;
    uint64_t count = 0;
    while (at < text.size()) {
        const auto newline = text.find('\n', at);
        auto line = text.substr(at, newline == text.npos ? text.size() - at : newline - at);
        at = newline == text.npos ? text.size() : newline + 1;
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        const auto first = line.find_first_not_of(" \t\r");
        if (first == line.npos || line[first] == '#' || line[first] == ';')
            continue;
        line.remove_prefix(first);
        if (line.substr(0, 7) != "METHOD=")
            throw std::runtime_error("invalid save record; expected METHOD= at record " +
                                     std::to_string(count));
        visitor(count++, line.substr(0, line.find("; PROGRAM=")));
    }
    return count;
}

static std::vector<std::string> records(const std::string& bytes) {
    std::vector<std::string> out;
    visit_records(bytes, [&](uint64_t, std::string_view record) { out.emplace_back(record); });
    return out;
}

uint64_t record_count(const std::string& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(std::filesystem::u8path(path), error);
    if (error)
        throw std::runtime_error("cannot inspect save: " + path);
    return exists ? visit_records(ecm_sync::read_file(std::filesystem::u8path(path)),
                                  [](uint64_t, std::string_view) {})
                  : 0;
}

std::string verified_snapshot(const Range& range, uint64_t* count) {
    if (!range.valid)
        throw std::runtime_error("Stage1 save range was not established by publication");
    ecm_sync::FileLock lock(range.path);
    auto bytes = ecm_sync::read_file(std::filesystem::u8path(range.path));
    const auto total = visit_records(bytes, [&](uint64_t index, std::string_view record) {
        if (index >= range.skip && index - range.skip < range.records.size() &&
            record != std::string_view(range.records[size_t(index - range.skip)]))
            throw std::runtime_error("published Stage1 save range changed: " + range.path);
    });
    if (range.skip > total || range.records.size() > total - range.skip)
        throw std::runtime_error("published Stage1 save range changed: " + range.path);
    if (count)
        *count = total;
    return bytes;
}

Range inspect_range(const std::string& path, uint64_t skip, uint32_t count) {
    ecm_sync::FileLock lock(path);
    Range range;
    range.path =
        std::filesystem::absolute(std::filesystem::u8path(path)).lexically_normal().u8string();
    range.skip = skip;
    const auto total = visit_records(ecm_sync::read_file(std::filesystem::u8path(path)),
                                     [&](uint64_t index, std::string_view record) {
                                         if (index >= skip && index - skip < count)
                                             range.records.emplace_back(record);
                                     });
    if (!count || skip > total || count > total - skip)
        throw std::runtime_error("save does not contain the requested curve range: " + path);
    range.valid = true;
    return range;
}

Range publish_batch(const std::string& path, const std::string& lines, bool append) {
    const auto file = std::filesystem::u8path(path);
    if (!file.parent_path().empty())
        std::filesystem::create_directories(file.parent_path());
    ecm_sync::FileLock lock(path);
    std::string bytes;
    if (append && std::filesystem::exists(file))
        bytes = ecm_sync::read_file(file);
    // This shared writer also preserves opaque JSONL receipts used outside
    // saved ECM points. Only METHOD batches establish a Stage1 save range.
    const auto first = lines.find_first_not_of(" \t\r\n");
    if (first != lines.npos && lines.compare(first, 7, "METHOD=") != 0) {
        std::set<std::string> seen;
        std::string line;
        std::istringstream old(bytes);
        while (std::getline(old, line))
            seen.insert(key(line));
        std::istringstream incoming(lines);
        while (std::getline(incoming, line))
            if (seen.insert(key(line)).second) {
                if (!bytes.empty() && bytes.back() != '\n')
                    bytes += '\n';
                bytes += line + '\n';
            }
        ecm_sync::write_atomic(file, bytes);
        return {};
    }
    auto old = records(bytes);
    std::map<std::string, uint64_t> positions;
    for (uint64_t i = 0; i < old.size(); ++i)
        positions.emplace(old[i], i);
    std::istringstream input(lines);
    std::string line;
    std::vector<std::pair<uint64_t, std::string>> batch;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const auto at = line.find("SIGMA=");
        size_t used = 0;
        if (line.compare(0, 7, "METHOD=") != 0 || at == line.npos)
            throw std::runtime_error("invalid incoming save record");
        const auto sigma = std::stoull(line.substr(at + 6), &used);
        if (!used || at + 6 + used >= line.size() || line[at + 6 + used] != ';')
            throw std::runtime_error("invalid save sigma");
        batch.emplace_back(sigma, std::move(line));
    }
    std::stable_sort(batch.begin(), batch.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    Range published;
    published.path = std::filesystem::absolute(file).lexically_normal().u8string();
    published.skip = old.size();
    uint64_t next_position = old.size();
    std::set<std::string> incoming;
    for (const auto& record : batch) {
        const auto identity = key(record.second);
        if (!incoming.insert(identity).second)
            continue;
        auto inserted = positions.emplace(identity, next_position);
        if (inserted.second) {
            ++next_position;
            if (!bytes.empty() && bytes.back() != '\n')
                bytes += '\n';
            bytes += record.second + '\n';
        }
        const auto index = inserted.first->second;
        if (published.records.empty())
            published.skip = index;
        if (index != published.skip + published.records.size())
            throw std::runtime_error("batch save records are not contiguous after deduplication");
        published.records.push_back(identity);
    }
    published.valid = true; // An empty Edwards publication is a valid completed batch.
    ecm_sync::write_atomic(file, bytes);
    return published;
}

bool publish(const std::string& path, const std::string& lines, bool append) {
    try {
        (void)publish_batch(path, lines, append);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void validate_batch(const Range& range, mpz_srcptr expected, uint64_t expected_bound,
                    int expected_param, const std::vector<uint64_t>& expected_sigmas,
                    bool allow_missing_points) {
    const bool ed = expected_param < 0;
    const int param = ed ? 0 : expected_param;
    std::set<uint64_t> sigmas(expected_sigmas.begin(), expected_sigmas.end());
    const bool check_sigmas = !expected_sigmas.empty();
    if (check_sigmas && (sigmas.size() != expected_sigmas.size() ||
                         (ed && allow_missing_points ? range.records.size() > sigmas.size()
                                                     : range.records.size() != sigmas.size())))
        throw std::runtime_error("saved batch curve count differs");
    for (const auto& record : range.records) {
        std::map<std::string, std::string> fields;
        std::istringstream in(record);
        std::string token;
        while (std::getline(in, token, ';')) {
            const auto equals = token.find('=');
            if (equals == token.npos)
                continue;
            const auto trim = [](std::string s) {
                const auto first = s.find_first_not_of(" \t\r"), last = s.find_last_not_of(" \t\r");
                return first == s.npos ? std::string{} : s.substr(first, last - first + 1);
            };
            if (!fields.emplace(trim(token.substr(0, equals)), trim(token.substr(equals + 1)))
                     .second)
                throw std::runtime_error("duplicate saved point field");
        }
        const auto integer = [&](const char* key) {
            const auto& value = fields.at(key);
            size_t used = 0;
            const auto n = std::stoull(value, &used);
            if (used != value.size() || value.empty() || value[0] == '-')
                throw std::runtime_error(std::string("invalid save ") + key);
            return uint64_t(n);
        };
        const auto sigma = integer("SIGMA"), b1 = integer("B1");
        if ((check_sigmas && !sigmas.erase(sigma)) || b1 != expected_bound)
            throw std::runtime_error("save SIGMA/B1 differs from completed batch");
        ecm_edwards::Integer n, x;
        std::string error;
        if (!ecm_parse_expression(fields.at("N"), n.v, &error) || mpz_cmp(n.v, expected) ||
            mpz_set_str(x.v, fields.at("X").c_str(), 0) || mpz_sgn(x.v) < 0 ||
            mpz_cmp(x.v, expected) >= 0)
            throw std::runtime_error("save N/X does not match completed batch");
        uint64_t checksum = 0;
        if (ed) {
            if (fields.at("METHOD") != "ECM-EDWARDS-AM" || fields.at("SCALAR") != "48LCM" ||
                integer("VERSION") != 1 ||
                (fields.at("STATUS") != "POINT" && fields.at("STATUS") != "FACTOR"))
                throw std::runtime_error("invalid Edwards saved point format");
            checksum = ecm_edwards::save_checksum(ecm_edwards::hex(n.v), ecm_edwards::hex(x.v),
                                                  sigma, b1, fields.at("STATUS") == "FACTOR");
        } else {
            const auto actual_param = fields.count("PARAM") ? integer("PARAM") : 0;
            if (fields.at("METHOD") != "ECM" || actual_param != uint64_t(param))
                throw std::runtime_error("save curve/param differs from completed batch");
            constexpr uint64_t prime = 4294967291ull;
            checksum = (b1 % prime) * (sigma % prime) % prime;
            checksum = checksum * mpz_fdiv_ui(n.v, static_cast<unsigned long>(prime)) % prime;
            checksum = checksum * mpz_fdiv_ui(x.v, static_cast<unsigned long>(prime)) % prime;
            checksum = checksum * (uint64_t(param) + 1) % prime;
        }
        const auto actual_checksum = integer("CHECKSUM");
        if (actual_checksum != checksum)
            throw std::runtime_error("save=" + range.path + " sigma=" + std::to_string(sigma) +
                                     " key=CHECKSUM value='" + std::to_string(actual_checksum) +
                                     "'; expected " + std::to_string(checksum));
    }
}

void validate_p95_batch(const Range& range, mpz_srcptr modulus) {
    if (range.records.empty())
        throw std::runtime_error("empty P95 curve range");
    const auto field = [](const std::string& line, const std::string& name,
                          const std::string& fallback) {
        const auto at = line.find(name + "=");
        if (at == line.npos)
            return fallback;
        const auto start = at + name.size() + 1, end = line.find(';', start);
        return line.substr(start, end - start);
    };
    const auto& first = range.records.front();
    if (first.compare(0, 11, "METHOD=ECM;") != 0)
        throw std::runtime_error("Prime95 requires Montgomery METHOD=ECM records");
    const int param = std::stoi(field(first, "PARAM", "0"));
    if (param != 0 && param != 3)
        throw std::runtime_error("Prime95 handoff supports PARAM0/3 only");
    const auto bound = std::stoull(field(first, "B1", ""));
    validate_batch(range, modulus, bound, param);
    std::set<uint64_t> unique;
    for (const auto& record : range.records) {
        const auto sigma = std::stoull(field(record, "SIGMA", ""));
        if (!sigma || sigma > uint64_t(INT64_MAX))
            throw std::runtime_error("INPUT[handoff] key=sigma value='" + std::to_string(sigma) +
                                     "'; Prime95 save reader requires [1,9223372036854775807]; "
                                     "disable handoff for wider sigmas");
        if (!unique.insert(sigma).second)
            throw std::runtime_error("duplicate sigma in P95 curve range");
    }
}

} // namespace ecm_save
