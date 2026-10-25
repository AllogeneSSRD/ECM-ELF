#include "ecm/io/save.h"

#include "ecm/platform/program_version.h"
#include "ecm/platform/logging.h"

#include <algorithm>
#include <charconv>
#include <map>
#include <set>
#include <stdexcept>
#include "io/files.h"
#include "ecm/application/configuration/ecm_expr.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr unsigned long CHKSUMMOD = 4294967291UL;

std::string mpz_to_dec_string(const mpz_t v) {
    char* s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
}

std::string build_who_field() {
    const char* uname = std::getenv("LOGNAME");
    if (!uname || !*uname) {
        uname = std::getenv("USERNAME");
    }
    std::string user = (uname && *uname) ? uname : "";

    std::string host;
#ifdef _WIN32
    char hbuf[MAX_COMPUTERNAME_LENGTH + 2] = {0};
    DWORD sz = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(hbuf, &sz) && sz > 0) {
        host.assign(hbuf, hbuf + sz);
    }
#else
    char hbuf[64] = {0};
    if (gethostname(hbuf, sizeof(hbuf) - 1) == 0) {
        host = hbuf;
    }
#endif

    if (user.empty() && host.empty()) {
        return "";
    }
    return user + "@" + host;
}

} // namespace

bool opencl_ecm_check_save_file_writable(const std::string& savefilename, bool saveappend) {
    if (savefilename.empty()) {
        return false;
    }
    if (!saveappend && access(savefilename.c_str(), 0) == 0) {
        ecm_ts_fprintf(stderr, "Save file %s already exists, will not overwrite\n",
                       savefilename.c_str());
        return false;
    }
    if (!ecm_io::ensure_parent_dir(savefilename.c_str())) {
        ecm_ts_fprintf(stderr, "Could not create parent directory for %s\n", savefilename.c_str());
        return false;
    }
    FILE* savefile = fopen(savefilename.c_str(), "a");
    if (savefile == nullptr) {
        ecm_ts_fprintf(stderr, "Could not open file %s for writing\n", savefilename.c_str());
        return false;
    }
    fclose(savefile);
    if (!saveappend) {
        struct stat st{};
        if (stat(savefilename.c_str(), &st) != 0 || st.st_size != 0) {
            ecm_ts_fprintf(stderr, "Save file %s initialization failed\n", savefilename.c_str());
            return false;
        }
        if (remove(savefilename.c_str()) != 0) {
            ecm_ts_fprintf(stderr, "Save file %s could not be cleaned up\n", savefilename.c_str());
            return false;
        }
    }
    return true;
}

std::string ecm_save_modulus_expression(const std::string& original_expr, const mpz_t N) {
    // X and CHECKSUM were computed modulo N. The published expression must
    // describe that same modulus even when this batch found a factor.
    return original_expr.empty() ? mpz_to_dec_string(N) : original_expr;
}

std::string ecm_encode_mont_save(mpz_srcptr n, double b1, int param,
                                 const std::vector<EcmSavePoint>& points,
                                 const std::string& expression) {
    std::ostringstream out;
    ecm_edwards::Integer sigma, checksum;
    const auto now = std::time(nullptr);
    char timebuf[128] = {0};
    if (const auto* local = std::localtime(&now))
        std::strftime(timebuf, sizeof(timebuf), "%a %b %d %H:%M:%S %Y", local);
    const auto who = build_who_field();
    for (const auto& point : points) {
        // unsigned long is only 32 bits on Windows; retain the full sigma.
        mpz_set_ui(sigma.v, static_cast<unsigned long>(point.sigma >> 32));
        mpz_mul_2exp(sigma.v, sigma.v, 32);
        mpz_add_ui(sigma.v, sigma.v, static_cast<unsigned long>(point.sigma & 0xffffffffull));
        mpz_set_d(checksum.v, b1);
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(sigma.v, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(n, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(point.value, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, static_cast<unsigned long>(param + 1) % CHKSUMMOD);
        out << "METHOD=ECM";
        if (param)
            out << "; PARAM=" << param;
        out << "; SIGMA=" << mpz_to_dec_string(sigma.v) << "; B1=" << std::llround(b1)
            << "; N=" << expression << "; X=0x" << ecm_edwards::hex(point.value)
            << "; CHECKSUM=" << mpz_fdiv_ui(checksum.v, CHKSUMMOD)
            << "; PROGRAM=" << ecm_program::name << "; X0=0x0; Y0=0x0;"
            << (who.empty() ? "" : " WHO=" + who + ";") << " TIME=" << timebuf << ";\n";
    }
    return out.str();
}

bool opencl_ecm_append_save_lines(const std::string& path, const mpz_t n, double b1, uint32_t first,
                                  uint32_t count, mpz_t* values, const std::string& expression,
                                  int param) {
    std::vector<EcmSavePoint> points;
    points.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        points.push_back({uint64_t(first) + i, values[i]});
    if (!ecm_save::publish(path, ecm_encode_mont_save(n, b1, param, points, expression), true))
        return false;
    ecm_ts_fprintf(stdout, "Saved %u curve line(s) to %s\n", count, path.c_str());
    return true;
}

bool ecm_append_save_lines_mont(const std::string& path, const mpz_t n, double b1,
                                const uint64_t* sigmas, uint32_t count, const mpz_t* values,
                                const int* hit, const std::string& expression) {
    std::vector<EcmSavePoint> points;
    points.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        points.push_back({sigmas[i], values[i]});
    (void)hit;
    if (!ecm_save::publish(path, ecm_encode_mont_save(n, b1, 0, points, expression), true))
        return false;
    ecm_ts_fprintf(stdout, "Saved %u Montgomery curve line(s) to %s\n", count, path.c_str());
    return true;
}

// Edwards final text codec.
namespace ecm_edwards {
uint64_t save_checksum(const std::string& n, const std::string& x, uint64_t sigma, uint64_t b1,
                       bool factor) {
    return ecm_io::fingerprint("ECM-EDWARDS-AM|1|48LCM|" + n + "|" + x + "|" +
                               std::to_string(sigma) + "|" + std::to_string(b1) + "|" +
                               (factor ? "FACTOR" : "POINT"));
}

std::string save_line(const mpz_t n, const mpz_t x, uint64_t sigma, uint64_t b1, bool factor) {
    const auto nh = hex(n), xh = hex(x);
    return "METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA=" + std::to_string(sigma) +
           "; B1=" + std::to_string(b1) + "; N=0x" + nh + "; X=0x" + xh +
           "; STATUS=" + (factor ? "FACTOR" : "POINT") +
           "; CHECKSUM=" + std::to_string(save_checksum(nh, xh, sigma, b1, factor)) + ";\n";
}

static std::string trim(std::string s) {
    const auto a = s.find_first_not_of(" \r\n\t"), b = s.find_last_not_of(" \r\n\t");
    return a == s.npos ? "" : s.substr(a, b - a + 1);
}

static uint64_t decimal(const std::string& s) {
    uint64_t n = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    if (s.empty() || r.ec != std::errc() || r.ptr != s.data() + s.size())
        throw std::runtime_error("expected unsigned decimal integer: " + s);
    return n;
}

SavedCurve parse_saved_curve(std::string text) {
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    if (text.find('\0') != text.npos)
        throw std::runtime_error("binary saves are unsupported");
    std::map<std::string, std::string> fields;
    size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(';', start);
        auto token = trim(text.substr(start, end == text.npos ? end : end - start));
        start = end == text.npos ? text.size() : end + 1;
        if (token.empty())
            continue;
        auto eq = token.find('=');
        if (eq == token.npos)
            throw std::runtime_error("missing '=' in save field");
        auto key = trim(token.substr(0, eq));
        if (!fields.emplace(key, trim(token.substr(eq + 1))).second)
            throw std::runtime_error("duplicate save field: " + key);
    }
    for (const auto* key :
         {"METHOD", "VERSION", "SCALAR", "SIGMA", "B1", "N", "X", "STATUS", "CHECKSUM"})
        if (!fields.count(key) || fields[key].empty())
            throw std::runtime_error(std::string("missing save field: ") + key);
    if (fields["METHOD"] != "ECM-EDWARDS-AM" || fields["VERSION"] != "1" ||
        fields["SCALAR"] != "48LCM" || fields.count("PARAM"))
        throw std::runtime_error("expected Edwards AM VERSION=1 SCALAR=48LCM, without PARAM");
    SavedCurve r;
    r.sigma = decimal(fields["SIGMA"]);
    r.b1 = decimal(fields["B1"]);
    if (!r.sigma || r.b1 < 2 || r.b1 > (1ull << 53))
        throw std::runtime_error(
            "invalid SIGMA or B1 (B1 must be exactly representable by libecm)");
    if (fields["STATUS"] != "POINT" && fields["STATUS"] != "FACTOR")
        throw std::runtime_error("invalid save STATUS");
    r.factor = fields["STATUS"] == "FACTOR";
    Integer n, x, z;
    if (mpz_set_str(n.v, fields["N"].c_str(), 0) || mpz_cmp_ui(n.v, 3) <= 0 || !mpz_odd_p(n.v))
        throw std::runtime_error("invalid odd N > 3");
    if (mpz_set_str(x.v, fields["X"].c_str(), 0) || mpz_sgn(x.v) < 0 || mpz_cmp(x.v, n.v) >= 0)
        throw std::runtime_error("X outside [0,N)");
    if (fields.count("Z") && (mpz_set_str(z.v, fields["Z"].c_str(), 0) || mpz_cmp_ui(z.v, 1)))
        throw std::runtime_error("save must be normalized (Z=1)");
    r.n = hex(n.v);
    r.x = hex(x.v);
    if (decimal(fields["CHECKSUM"]) != save_checksum(r.n, r.x, r.sigma, r.b1, r.factor))
        throw std::runtime_error("save CHECKSUM mismatch");
    if (r.factor && (mpz_cmp_ui(x.v, 1) <= 0 || !mpz_divisible_p(n.v, x.v)))
        throw std::runtime_error("invalid saved factor");
    return r;
}

std::vector<SavedCurve> read_saved_curves(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open save: " + path);
    std::vector<SavedCurve> result;
    std::string line;
    uint64_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        const auto t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';')
            continue;
        try {
            auto r = parse_saved_curve(line);
            r.line = number;
            result.push_back(std::move(r));
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(number) + ": " + e.what());
        }
    }
    if (in.bad())
        throw std::runtime_error("save read failed");
    if (result.empty())
        throw std::runtime_error("save contains no curves");
    return result;
}
} // namespace ecm_edwards

// Save publication and contiguous record range validation.
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
    return exists ? visit_records(ecm_io::read_file(std::filesystem::u8path(path)),
                                  [](uint64_t, std::string_view) {})
                  : 0;
}

std::string verified_snapshot(const Range& range, uint64_t* count) {
    if (!range.valid)
        throw std::runtime_error("Stage1 save range was not established by publication");
    ecm_io::FileLock lock(range.path);
    auto bytes = ecm_io::read_file(std::filesystem::u8path(range.path));
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
    ecm_io::FileLock lock(path);
    Range range;
    range.path =
        std::filesystem::absolute(std::filesystem::u8path(path)).lexically_normal().u8string();
    range.skip = skip;
    const auto total = visit_records(ecm_io::read_file(std::filesystem::u8path(path)),
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
    ecm_io::FileLock lock(path);
    std::string bytes;
    if (append && std::filesystem::exists(file))
        bytes = ecm_io::read_file(file);
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
    ecm_io::write_atomic(file, bytes);
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
