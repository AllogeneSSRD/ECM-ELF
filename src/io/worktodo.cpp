#include "io/worktodo.h"
#include "io/files.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

void trim(std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
}

void strip_bom(std::string& s) {
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB && static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
}

bool is_int_like(const std::string& s) {
    std::string t = s;
    trim(t);
    if (t.empty()) {
        return false;
    }
    if (t[0] == '-' || t[0] == '+') {
        t.erase(t.begin());
    }
    if (t.empty()) {
        return false;
    }
    for (char c : t) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool assignment_id(const std::string& value) {
    return value.size() == 32 && value.find_first_not_of("0123456789abcdefABCDEF") == value.npos;
}

bool no_assignment_id(const std::string& value) {
    return value.empty() || value == "N/A" || value == "n/a";
}

// Split a CSV row, honouring double quotes and "" escapes. Each field is trimmed.
std::vector<std::string> split_csv(const std::string& s, std::vector<bool>* quotes = nullptr) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false, had_quote = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        if (in_quotes) {
            if (ch == '"') {
                if (i + 1 < s.size() && s[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(ch);
            }
        } else {
            if (ch == '"') {
                in_quotes = had_quote = true;
            } else if (ch == ',') {
                out.push_back(cur);
                cur.clear();
                if (quotes)
                    quotes->push_back(had_quote);
                had_quote = false;
            } else {
                cur.push_back(ch);
            }
        }
    }
    out.push_back(cur);
    if (quotes)
        quotes->push_back(had_quote);
    for (std::string& f : out) {
        trim(f);
    }
    return out;
}

bool csv_quotes_valid(const std::string& s) {
    bool quoted = false;
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == '"') {
            if (quoted && i + 1 < s.size() && s[i + 1] == '"') {
                ++i;
                continue;
            }
            quoted = !quoted;
        }
    return !quoted;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) {
        return std::wstring();
    }
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    if (!w.empty() && w.back() == L'\0') {
        w.pop_back();
    }
    return w;
}

long long filetime_to_epoch(const FILETIME& ft) {
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return static_cast<long long>(ull.QuadPart / 10000000ULL - 11644473600ULL);
}
#endif

bool replace_file(const std::string& tmp, const std::string& dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

} // namespace

static size_t save_column(const std::vector<std::string>& cols, const std::vector<bool>& quotes,
                          bool modulus_from_save) {
    const auto& first = cols.front();
    if (modulus_from_save) {
        // Quote a filename that would otherwise be read as an AID.
        return !quotes.front() && (no_assignment_id(first) || assignment_id(first)) ? 1 : 0;
    }
    return (!is_int_like(first) || assignment_id(first)) ? 5 : 4;
}

bool ecm_rewrite_stage2_save(const std::string& line, const std::string& save,
                             std::string& rewritten, std::string& err) {
    const bool stage1 = line.compare(0, 10, "ECMSTAGE1=") == 0;
    EcmStage2Task task;
    if (!(stage1 ? ecm_parse_stage1_line(line, task, err) : ecm_parse_stage2_line(line, task, err)))
        return false;
    const auto begin = line.find('=') + 1;
    std::vector<bool> quotes;
    const auto fields = split_csv(line.substr(begin), &quotes);
    const size_t wanted = save_column(fields, quotes, task.modulus_from_save);
    size_t column = 0, start = begin;
    bool quoted = false;
    for (size_t i = begin; i <= line.size(); ++i) {
        if (i < line.size() && line[i] == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                ++i;
                continue;
            }
            quoted = !quoted;
            continue;
        }
        if (i != line.size() && (quoted || line[i] != ','))
            continue;
        if (column == wanted) {
            std::string replacement = "\"";
            for (const auto c : save) {
                replacement += c;
                if (c == '"')
                    replacement += '"';
            }
            replacement += '"';
            size_t left = start, right = i;
            while (left < right && std::isspace(static_cast<unsigned char>(line[left])))
                ++left;
            while (right > left && std::isspace(static_cast<unsigned char>(line[right - 1])))
                --right;
            rewritten = line.substr(0, left) + replacement + line.substr(right);
            EcmStage2Task check;
            if ((stage1 ? ecm_parse_stage1_line(rewritten, check, err)
                        : ecm_parse_stage2_line(rewritten, check, err)) &&
                check.save_name == save)
                return true;
            err = "cannot rewrite save name";
            return false;
        }
        ++column;
        start = i + 1;
    }
    err = "missing save-name field";
    return false;
}

static bool parse_save_line(const std::string& line, std::string_view prefix, EcmStage2Task& task,
                            std::string& err) {
    std::string s = line;
    strip_bom(s);
    trim(s);
    task = EcmStage2Task{};
    task.raw_line = s;
    const bool stage1 = prefix == "ECMSTAGE1=";
    task.modulus_from_save = prefix == ecm_queue::stage2_n_prefix;

    if (!csv_quotes_valid(s)) {
        err = "key=CSV value='" + s +
              "'; expected a closing double quote (escape quotes as two double quotes)";
        return false;
    }

    if (s.size() <= prefix.size() || s.compare(0, prefix.size(), prefix) != 0) {
        err = "line does not start with " + std::string(prefix);
        return false;
    }

    std::vector<bool> quotes;
    const auto cols = split_csv(s.substr(prefix.size()), &quotes);
    const auto save = save_column(cols, quotes, task.modulus_from_save);
    const auto core = task.modulus_from_save ? size_t(0) : size_t(4);
    if (save != core && !no_assignment_id(cols.front()))
        task.aid = cols.front();
    if (save >= cols.size() || cols[save].empty()) {
        err = task.modulus_from_save ? "ECMSTAGE2N needs a save filename" :
                                      "worktodo needs k,b,n,c and a save filename";
        return false;
    }
    task.save_name = cols[save];
    size_t tail = save + 1;
    size_t end = cols.size();
    if (end > tail && quotes.back() && (!stage1 || end == tail + 4)) {
        for (const auto& factor : split_csv(cols[--end]))
            if (!factor.empty()) {
                if (task.modulus_from_save) {
                    const auto first = factor.find_first_not_of('0');
                    if (factor.find_first_not_of("0123456789") != factor.npos ||
                        first == factor.npos || (first == factor.size() - 1 && factor[first] == '1')) {
                        err = "known factors must be decimal integers greater than 1";
                        return false;
                    }
                }
                task.factors.push_back(factor);
            }
    }
    if (end > tail + 3 || (stage1 && end != tail + 3)) {
        err = "expected B2, skip_curves, num_curves and an optional quoted known-factors field";
        return false;
    }
    if (!stage1 && std::any_of(quotes.begin() + tail, quotes.begin() + end, [](bool quoted) {
            return quoted;
        })) {
        err = "known factors must be the final quoted field";
        return false;
    }
    try {
        if (!task.modulus_from_save) {
            const auto idx = save - 4;
            task.k = cols[idx];
            task.b = cols[idx + 1];
            task.c = cols[idx + 3];
            const auto n = ecm_config::unsigned_integer(cols[idx + 2], "exponent n");
            if (!is_int_like(cols[idx + 2]) || n > ULONG_MAX)
                throw std::runtime_error("invalid exponent n: " + cols[idx + 2]);
            task.n = static_cast<unsigned long>(n);
        }
        if (tail < end && !cols[tail].empty()) {
            task.b2_text = cols[tail];
            task.has_b2 = true;
            (void)ecm_config::unsigned_integer(task.b2_text, "B2");
        }
        if (tail + 1 < end && !cols[tail + 1].empty()) {
            task.has_skip = true;
            task.skip_curves = ecm_config::unsigned_integer(cols[tail + 1], "skip_curves");
        }
        const auto count = tail + 2 < end && !cols[tail + 2].empty() ?
            ecm_config::unsigned_integer(cols[tail + 2], "curves_to_run") : 0;
        if (count > UINT32_MAX || (stage1 && !count) ||
            (stage1 && !is_int_like(cols[tail + 2])))
            throw std::runtime_error("key=curves_to_run value='" + cols[tail + 2] +
                                     "'; expected integer " + (stage1 ? "[1,4294967295]" : "[0,4294967295]"));
        task.curves_to_run = static_cast<uint32_t>(count);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    return true;
}

bool ecm_parse_stage1_line(const std::string& line, EcmStage2Task& task, std::string& err) {
    return parse_save_line(line, "ECMSTAGE1=", task, err);
}

bool ecm_parse_stage2_line(const std::string& line, EcmStage2Task& task, std::string& err) {
    return parse_save_line(line, ecm_queue::stage2_from_save(ecm_queue::task_text(line)) ?
                          ecm_queue::stage2_n_prefix : ecm_queue::stage2_prefix, task, err);
}

bool ecm_rewrite_stage2_range(const std::string& line, uint64_t skip, uint32_t count,
                              std::string& rewritten, std::string& err) {
    EcmStage2Task task;
    const bool stage1 = line.compare(0, 10, "ECMSTAGE1=") == 0;
    if (!(stage1 ? ecm_parse_stage1_line(line, task, err) : ecm_parse_stage2_line(line, task, err)))
        return false;
    const auto begin = line.find('=') + 1;
    std::vector<bool> quotes;
    const auto fields = split_csv(line.substr(begin), &quotes);
    const size_t save = save_column(fields, quotes, task.modulus_from_save);
    // Preserve every unmodified CSV token, including quoting and whitespace.
    std::vector<std::pair<size_t, size_t>> spans;
    size_t start = begin;
    bool quoted = false;
    for (size_t i = begin; i <= line.size(); ++i) {
        if (i < line.size() && line[i] == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                ++i;
                continue;
            }
            quoted = !quoted;
            continue;
        }
        if (i != line.size() && (quoted || line[i] != ','))
            continue;
        spans.push_back({start, i});
        start = i + 1;
    }
    const bool factors = spans.size() > save + 1 && quotes.back() &&
                         (!stage1 || spans.size() == save + 5);
    const auto end = spans.size() - size_t(factors);
    if (end <= save + 3) {
        const auto suffix = factors ? spans[end].first - 1 : line.size();
        const bool has_b2_field = end > save + 1;
        rewritten = line.substr(0, spans[save + size_t(has_b2_field)].second);
        if (!has_b2_field)
            rewritten += ',';
        rewritten += ',' + std::to_string(skip) + ',' + std::to_string(count) + line.substr(suffix);
        return true;
    }
    const auto s = spans[save + 2], c = spans[save + 3];
    rewritten = line.substr(0, s.first) + std::to_string(skip) +
                line.substr(s.second, c.first - s.second) + std::to_string(count) +
                line.substr(c.second);
    return true;
}

bool ecm_parse_ecm2_line(const std::string& line, Ecm2Task& task, std::string& err) {
    std::string s = line;
    strip_bom(s);
    trim(s);
    task = Ecm2Task{};
    task.raw_line = s;
    if (!csv_quotes_valid(s)) {
        err = "key=CSV value='" + s + "'; expected a closing double quote";
        return false;
    }

    // Prime95 将 ECM= 与 ECM2= 视为等价 (commonc.c:2934)。
    // 格式:
    // [<AID>,|N/A,|<nul>][FFT2=<fftl>,|<nul>]<k>,<b>,<n>,<c>,<B1>[,<B2>][,<curves>][,<sigma>][,"factors"]
    std::string body;
    if (s.size() > 5 && s.compare(0, 5, "ECM2=") == 0) {
        body = s.substr(5);
    } else if (s.size() > 4 && s.compare(0, 4, "ECM=") == 0) {
        body = s.substr(4);
    } else {
        err = "line does not start with ECM= or ECM2=";
        return false;
    }

    // 分离 quoted known-factors 尾字段 (ECM= 里唯一带引号的字段)。
    std::string unquoted = body;
    std::string factors_raw;
    {
        const std::size_t q0 = body.find('"');
        if (q0 != std::string::npos) {
            unquoted = body.substr(0, q0);
            const std::size_t q1 = body.rfind('"');
            if (q1 != std::string::npos && q1 > q0) {
                factors_raw = body.substr(q0 + 1, q1 - q0 - 1);
                if (!ecm_config::trim(body.substr(q1 + 1)).empty()) {
                    err = "key=known-factors; expected end of task after closing quote";
                    return false;
                }
            }
        }
    }

    const std::vector<std::string> cols = split_csv(unquoted);
    if (cols.empty() || (cols.size() == 1 && cols[0].empty())) {
        err = "no fields after ECM=/ECM2=";
        return false;
    }

    std::size_t idx = 0;

    // Optional AID: first field is an AID iff it is not integer-like and not "FFT2=...".
    if ((!is_int_like(cols[idx]) || assignment_id(cols[idx])) && !starts_with(cols[idx], "FFT2=")) {
        if (!no_assignment_id(cols[idx]))
            task.aid = cols[idx];
        ++idx;
    }
    // Optional FFT2 length (informational; e.g. "FFT2=192K").
    if (idx < cols.size() && starts_with(cols[idx], "FFT2=")) {
        task.fft2 = cols[idx].substr(5);
        ++idx;
    }

    // Required: k, b, n, c, B1.
    if (cols.size() < idx + 5) {
        err = "not enough fields (need k,b,n,c,B1)";
        return false;
    }
    task.k = cols[idx];
    task.b = cols[idx + 1];
    task.c = cols[idx + 3];

    { // n (exponent)
        const std::string& nstr = cols[idx + 2];
        if (!is_int_like(nstr) || nstr.empty() || nstr[0] == '-') {
            err = "invalid exponent n: '" + nstr + "'";
            return false;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long v = std::strtoul(nstr.c_str(), &end, 10);
        if (errno == ERANGE || end == nstr.c_str() || *end != '\0') {
            err = "invalid exponent n: '" + nstr + "'";
            return false;
        }
        task.n = v;
    }
    { // B1 (required)
        const std::string& b1str = cols[idx + 4];
        char* end = nullptr;
        const double v = std::strtod(b1str.c_str(), &end);
        if (end == b1str.c_str() || *end != '\0' || !std::isfinite(v) || !(v > 0.0)) {
            err = "key=B1 value='" + b1str + "'; expected finite positive bound";
            return false;
        }
        task.B1 = v;
    }

    // Optional positional fields (匹配 Prime95): B2 (默认 0), curves_to_run (默认 100), sigma,
    // factors。
    std::size_t p = idx + 5;
    task.B2 = 0.0;
    if (cols.size() > p) {
        const std::string& b2str = cols[p];
        if (!b2str.empty()) {
            char* end = nullptr;
            const double v = std::strtod(b2str.c_str(), &end);
            if (end != b2str.c_str() && *end == '\0' && std::isfinite(v) && v >= 0.0) {
                task.B2 = v;
                task.has_b2 = true;
                task.b2_text = b2str;
            } else {
                err = "key=B2 value='" + b2str + "'; expected finite nonnegative bound";
                return false;
            }
        }
        ++p;
    }
    // Omitted fields remain absent; io/normalize supplies Prime95's defaults.
    if (cols.size() > p) {
        const std::string& cstr = cols[p];
        if (!cstr.empty()) {
            char* end = nullptr;
            errno = 0;
            const unsigned long v = std::strtoul(cstr.c_str(), &end, 10);
            if (is_int_like(cstr) && cstr[0] != '-' && errno != ERANGE && end != cstr.c_str() &&
                *end == '\0' && v > 0 && v <= 0xFFFFFFFFul) {
                task.curves_to_run = static_cast<uint32_t>(v);
                task.has_curves = true;
            } else {
                err = "key=curves_to_run value='" + cstr + "'; expected integer [1,4294967295]";
                return false;
            }
        }
        ++p;
    }
    if (cols.size() > p) {
        const std::string& sstr = cols[p];
        if (!sstr.empty()) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = std::strtoull(sstr.c_str(), &end, 10);
            if (is_int_like(sstr) && sstr[0] != '-' && errno != ERANGE && end != sstr.c_str() &&
                *end == '\0') {
                task.has_sigma = v > 0;
                task.has_sigma_field = true;
                task.sigma = v;
            } else {
                err = "key=sigma value='" + sstr +
                      "'; expected integer [0,18446744073709551615], 0=unspecified";
                return false;
            }
        }
        ++p;
    }
    // known factors (来自 quoted 尾字段)
    for (size_t i = p; i < cols.size(); ++i)
        if (!cols[i].empty()) {
            err = "key=CSV value='" + cols[i] +
                  "'; unexpected field; expected quoted known factors or end of task";
            return false;
        }
    if (!factors_raw.empty()) {
        const std::vector<std::string> facs = split_csv(factors_raw);
        for (const std::string& f : facs) {
            if (!f.empty()) {
                task.factors.push_back(f);
            }
        }
    }
    return true;
}

bool ecm_extract_b1_from_save_name(const std::string& save_name, double* b1_out, std::string& err) {
    const std::string suffix = ".save";
    if (save_name.size() <= suffix.size() ||
        save_name.compare(save_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        err = "save_name does not end with .save: '" + save_name + "'";
        return false;
    }
    const std::string stem = save_name.substr(0, save_name.size() - suffix.size());
    const std::size_t us = stem.rfind('_');
    if (us == std::string::npos || us + 1 >= stem.size()) {
        err = "save_name has no '_<B1>' token: '" + save_name + "'";
        return false;
    }
    const std::string token = stem.substr(us + 1);
    char* end = nullptr;
    const double b1 = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || !std::isfinite(b1) || b1 <= 0.0) {
        err = "cannot parse B1 from save_name token '" + token + "'";
        return false;
    }
    *b1_out = b1;
    return true;
}

// Shared N computation: N = (k*b^n + c) / (f1*f2*...), exact integer arithmetic.
static bool compute_n_from_fields(const std::string& k_str, const std::string& b_str,
                                  unsigned long n, const std::string& c_str,
                                  const std::vector<std::string>& factors, mpz_t N,
                                  std::string& err) {
    mpz_t k, b, c;
    mpz_init(k);
    mpz_init(b);
    mpz_init(c);

    bool ok = true;
    if (mpz_set_str(k, k_str.c_str(), 10) != 0) {
        err = "invalid k: '" + k_str + "'";
        ok = false;
    } else if (mpz_set_str(b, b_str.c_str(), 10) != 0) {
        err = "invalid b: '" + b_str + "'";
        ok = false;
    } else if (mpz_set_str(c, c_str.c_str(), 10) != 0) {
        err = "invalid c: '" + c_str + "'";
        ok = false;
    }

    mpz_t bn;
    mpz_init(bn);
    if (ok) {
        mpz_pow_ui(bn, b, n); // b^n
        mpz_mul(N, k, bn);    // k*b^n
        mpz_add(N, N, c);     // k*b^n + c
        if (mpz_sgn(N) <= 0) {
            err = "computed N is not positive";
            ok = false;
        }
    }

    if (ok) {
        mpz_t f;
        mpz_init(f);
        for (const std::string& fs : factors) {
            if (mpz_set_str(f, fs.c_str(), 10) != 0) {
                err = "invalid known factor: '" + fs + "'";
                ok = false;
                break;
            }
            if (mpz_sgn(f) <= 0) {
                err = "non-positive known factor: '" + fs + "'";
                ok = false;
                break;
            }
            if (!mpz_divisible_p(N, f)) {
                err = "known factor " + fs + " does not divide (k*b^n+c) exactly";
                ok = false;
                break;
            }
            mpz_divexact(N, N, f);
        }
        mpz_clear(f);
    }

    mpz_clear(bn);
    mpz_clear(k);
    mpz_clear(b);
    mpz_clear(c);
    return ok;
}

bool ecm_compute_stage2_n(const EcmStage2Task& task, mpz_t N, std::string& err) {
    if (task.modulus_from_save) {
        err = "ECMSTAGE2N reads N from the selected save records";
        return false;
    }
    return compute_n_from_fields(task.k, task.b, task.n, task.c, task.factors, N, err);
}

bool ecm_compute_ecm2_n(const Ecm2Task& task, mpz_t N, std::string& err) {
    return compute_n_from_fields(task.k, task.b, task.n, task.c, task.factors, N, err);
}

int ecm_worktodo_parse_worker_header(const std::string& line, bool* is_bracket_line) {
    return ecm_config::worker_header(line, is_bracket_line);
}

namespace {

using ecm_queue::classify_line;
using ecm_queue::LineKind;
using ecm_queue::worker_matches;

// Preserve the legacy bool-returning queue interface; the platform gate is
// shared with GUI editing, including the complete read-modify-write lifetime.
struct QueueEditLock {
    std::unique_ptr<ecm_io::FileLock> lock;
    std::string error;
    explicit QueueEditLock(const std::string& path) {
        try {
            lock = std::make_unique<ecm_io::FileLock>(path, ecm_io::FileLockDomain::QueueEdit);
        } catch (const std::exception& failure) {
            error = failure.what();
            std::fprintf(stderr, "ERROR: %s\n", error.c_str());
        }
    }
    bool valid() const { return bool(lock); }
};

} // namespace

bool ecm_worktodo_first_line(const std::string& path, std::string& line) {
    return ecm_worktodo_first_line(path, 1, line);
}

bool ecm_worktodo_first_line(const std::string& path, int worker, std::string& line) {
    std::string err;
    return ecm_worktodo_read_first_line(path, worker, line, err) == WorktodoReadResult::Task;
}

WorktodoStage ecm_worktodo_stage(const std::string& line) {
    return ecm_queue::task_stage(ecm_queue::task_text(line));
}

WorktodoReadResult ecm_worktodo_read_next_line(const std::string& path, int worker,
                                               WorktodoSelection selection, WorktodoStage consumer,
                                               std::string& line, std::string& err,
                                               size_t* source_line) {
    line.clear();
    err.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        err = "cannot read worktodo: " + path;
        return WorktodoReadResult::Error;
    }
    const auto result = ecm_queue::read_next_task(in, worker, selection, consumer, line, source_line);
    if (result == WorktodoReadResult::Error)
        err = "worktodo read failed: " + path;
    return result;
}

WorktodoReadResult ecm_worktodo_read_first_line(const std::string& path, int worker,
                                                std::string& line, std::string& err) {
    return ecm_worktodo_read_next_line(path, worker, {}, WorktodoStage::Any, line, err);
}

bool ecm_worktodo_stage1_lines(const std::string& path, int worker, std::vector<std::string>& lines,
                               std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot read worktodo: " + path;
        return false;
    }
    lines.clear();
    std::string raw;
    int section = 1;
    while (std::getline(in, raw))
        if (classify_line(raw, &section, nullptr) == LineKind::Task &&
            worker_matches(section, worker)) {
            strip_bom(raw);
            trim(raw);
            if (starts_with(raw, "ECMSTAGE1="))
                lines.push_back(raw);
        }
    if (in.bad()) {
        err = "incomplete worktodo read: " + path;
        return false;
    }
    return true;
}

bool ecm_worktodo_advance(const std::string& path, const std::string& first_line,
                          WorktodoAction action) {
    return ecm_worktodo_advance(path, 1, first_line, action);
}

static void append_to_worker(std::vector<std::string>& lines, int worker, const std::string& line) {
    // The unlabelled leading segment belongs to worker 1.
    int section = 1;
    bool found_section = worker == 1;
    size_t insertion = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        classify_line(lines[i], &section, nullptr);
        if (section == worker) {
            found_section = true;
            insertion = i + 1;
        }
    }
    if (!found_section) {
        lines.push_back("[Worker #" + std::to_string(worker) + "]");
        insertion = lines.size();
    }
    lines.insert(lines.begin() + static_cast<std::vector<std::string>::difference_type>(insertion), line);
}

bool ecm_worktodo_advance(const std::string& path, int worker, const std::string& first_line,
                          WorktodoAction action, const std::string& replacement,
                          const std::string& receipt, const std::string& receipt_payload) {
    if ((action == WorktodoAction::Replace || action == WorktodoAction::AppendToWorker) &&
        (replacement.empty() || replacement.find_first_of("\r\n") != std::string::npos))
        return false;
    QueueEditLock lock(path);
    if (!lock.valid())
        return false;
    std::string before;
    try {
        before = ecm_io::read_file(std::filesystem::u8path(path));
    } catch (...) {
        return false;
    }
    std::istringstream in(before);
    std::vector<std::string> lines;
    std::string l;
    while (std::getline(in, l)) {
        if (!l.empty() && l.back() == '\r')
            l.pop_back();
        lines.push_back(l);
    }
    int section = 1;
    bool found = false;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const LineKind kind = classify_line(lines[i], &section, nullptr);
        if (kind != LineKind::Task) {
            continue; // headers/comments stay untouched
        }
        if (!worker_matches(section, worker)) {
            continue; // another worker's line
        }
        // The selected task can follow another stage when queue priority is set.
        // Match its exact line in this worker section before changing it.
        std::string current = lines[i], expected = first_line;
        strip_bom(current);
        strip_bom(expected);
        trim(current);
        trim(expected);
        if (current != expected)
            continue;
        if (action == WorktodoAction::Remove || action == WorktodoAction::AppendToWorker) {
            lines.erase(lines.begin() + static_cast<std::vector<std::string>::difference_type>(i));
            if (action == WorktodoAction::AppendToWorker)
                append_to_worker(lines, section, replacement);
        } else if (action == WorktodoAction::MarkError) {
            std::string t = lines[i];
            strip_bom(t);
            trim(t);
            lines[i] = "# ERROR " + t;
        } else {
            lines[i] = replacement;
        }
        found = true;
        break;
    }
    if (!found) {
        return false;
    }

    std::string after;
    for (const auto& ln : lines)
        after += ln + "\n";
    try {
        if (!receipt.empty())
            ecm_io::write_atomic(std::filesystem::u8path(receipt),
                                 receipt_payload + "\nECM-ADVANCE\n" +
                                     std::to_string(ecm_io::fingerprint(before)) + " " +
                                     std::to_string(ecm_io::fingerprint(after)) + "\n");
        ecm_io::write_atomic(std::filesystem::u8path(path), after);
        return true;
    } catch (...) {
        return false;
    }
}

bool ecm_worktodo_list_workers(const std::string& path, std::vector<uint32_t>& workers,
                               std::string& err) {
    err.clear();
    workers.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        err = "cannot open " + path;
        return false;
    }
    int section = 1;
    std::vector<uint32_t> seen;
    std::string l;
    while (std::getline(in, l)) {
        const LineKind kind = classify_line(l, &section, nullptr);
        if (kind != LineKind::Task) {
            continue;
        }
        const uint32_t w = static_cast<uint32_t>(section);
        if (std::find(seen.begin(), seen.end(), w) == seen.end()) {
            seen.push_back(w);
        }
    }
    std::sort(seen.begin(), seen.end());
    workers = seen;
    return true;
}

bool ecm_worktodo_worker_loads(const std::string& path, std::map<int, size_t>& loads,
                               std::string& err) {
    loads.clear();
    err.clear();
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::u8path(path), ec)) {
        if (ec)
            err = "cannot inspect destination queue: " + ec.message();
        return !ec;
    }
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) {
        err = "cannot read destination queue: " + path;
        return false;
    }
    std::string raw;
    int section = 1;
    while (std::getline(in, raw)) {
        const auto kind = classify_line(raw, &section, nullptr);
        if (kind == LineKind::Header)
            loads.emplace(section, 0);
        else if (kind == LineKind::Task && ecm_worktodo_stage(raw) != WorktodoStage::Any)
            ++loads[section];
    }
    if (in.bad()) {
        err = "cannot finish reading destination queue: " + path;
        return false;
    }
    return true;
}

bool ecm_append_text_line(const std::string& path, const std::string& line) {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        return false;
    }
    out << line << "\n";
    return !out.fail();
}

bool ecm_worktodo_append_unique(const std::string& path, const std::string& finished, int worker,
                                const std::string& line, std::string& err) {
    err.clear();
    if (worker < 1 || worker > 1000000 || line.empty() ||
        line.find_first_of("\r\n") != std::string::npos) {
        err = "invalid Stage2 transfer worker or line";
        return false;
    }
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent, ec);
    if (ec) {
        err = "cannot create Stage2 queue directory: " + ec.message();
        return false;
    }
    QueueEditLock lock(path);
    if (!lock.valid()) {
        err = lock.error;
        return false;
    }

    auto read_lines = [&](const std::string& name, std::vector<std::string>& lines) -> bool {
        ec.clear();
        const bool exists = std::filesystem::exists(name, ec);
        if (ec)
            return false;
        if (!exists)
            return true;
        std::ifstream in(name, std::ios::binary);
        if (!in)
            return false;
        std::string raw;
        while (std::getline(in, raw))
            lines.push_back(raw);
        return !in.bad();
    };
    std::vector<std::string> lines, done;
    if (!read_lines(path, lines)) {
        err = "cannot read Stage2 queue";
        return false;
    }
    try {
        if (!finished.empty()) {
            const auto finished_path = std::filesystem::u8path(finished);
            if (ecm_io::same_file_resource(std::filesystem::u8path(path), finished_path)) {
                err = "Stage2 queue and finished record must use distinct paths";
                return false;
            }
            if (finished_path.has_parent_path())
                std::filesystem::create_directories(finished_path.parent_path());
            const ecm_io::FileLock finished_lock(finished, ecm_io::FileLockDomain::QueueEdit);
            if (!read_lines(finished, done)) {
                err = "cannot read Stage2 finished record";
                return false;
            }
        }
    } catch (const std::exception& failure) {
        err = failure.what();
        return false;
    }
    auto same = [&](const std::string& raw) {
        std::string value = raw;
        strip_bom(value);
        trim(value);
        return value == line;
    };
    auto same_done = [&](const std::string& raw) {
        if (same(raw))
            return true;
        std::string value = raw;
        strip_bom(value);
        trim(value);
        return value == "# " + line; // Stage2 invalid-input receipt
    };
    if (std::any_of(lines.begin(), lines.end(), same) ||
        std::any_of(done.begin(), done.end(), same_done))
        return true;

    // Repeated headers without a worker transition do not start a new queue.
    // Preserve explicit boundaries around foreign bracket lines and all tasks,
    // comments and blanks; only redundant worker labels are removed.
    std::vector<std::string> compact;
    compact.reserve(lines.size());
    int section = 1;
    bool explicit_header = false;
    for (auto& raw : lines) {
        const int previous = section;
        const auto kind = classify_line(raw, &section, nullptr);
        if (kind == LineKind::Header) {
            if (explicit_header && section == previous)
                continue;
            explicit_header = true;
        } else if (kind == LineKind::ForeignBracket) {
            explicit_header = false;
        }
        compact.push_back(std::move(raw));
    }
    lines = std::move(compact);

    append_to_worker(lines, worker, line);
    const std::string tmp = path + ".transfer.tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "cannot write Stage2 queue temp file";
        return false;
    }
    for (const auto& raw : lines)
        out << raw << '\n';
    out.close();
    if (!out || !replace_file(tmp, path)) {
        err = "cannot publish Stage2 queue";
        return false;
    }
    return true;
}

bool ecm_append_text_line_unique(const std::string& path, const std::string& line) {
    try {
        const auto file = std::filesystem::u8path(path);
        if (!file.parent_path().empty())
            std::filesystem::create_directories(file.parent_path());
        QueueEditLock lock(path);
        if (!lock.valid())
            return false;
        auto bytes = std::filesystem::exists(file) ? ecm_io::read_file(file) : std::string{};
        std::istringstream in(bytes);
        std::string old;
        while (std::getline(in, old)) {
            if (!old.empty() && old.back() == '\r')
                old.pop_back();
            if (old == line)
                return true;
        }
        if (!bytes.empty() && bytes.back() != '\n')
            bytes += '\n';
        ecm_io::write_atomic(file, bytes + line + "\n");
        return true;
    } catch (...) {
        return false;
    }
}

bool ecm_worktodo_store_completion(const std::string& receipt, const CompletedWorktodo& c,
                                   std::string& err) {
    try {
        std::ostringstream out;
        out << (c.append_replacement ? "ECM-STAGE1-COMPLETED 2\n" : "ECM-STAGE1-COMPLETED 1\n")
            << std::quoted(c.ticket) << '\n'
            << std::quoted(c.record) << '\n'
            << std::quoted(c.replacement) << '\n'
            << std::quoted(c.finished) << '\n'
            << std::quoted(c.checkpoint) << '\n';
        if (std::filesystem::exists(std::filesystem::u8path(receipt)))
            throw std::runtime_error("previous Stage1 completion has not been acknowledged");
        ecm_io::write_atomic(std::filesystem::u8path(receipt), out.str());
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool ecm_worktodo_commit_completion(const std::string& receipt, const std::string& queue,
                                    int worker, CompletedWorktodo& c, std::string& err) {
    try {
        const auto bytes = ecm_io::read_file(std::filesystem::u8path(receipt));
        std::istringstream in(bytes);
        std::string magic;
        std::getline(in, magic);
        c.append_replacement = magic == "ECM-STAGE1-COMPLETED 2";
        if ((magic != "ECM-STAGE1-COMPLETED 1" && !c.append_replacement) ||
            !(in >> std::quoted(c.ticket) >> std::quoted(c.record) >> std::quoted(c.replacement) >>
              std::quoted(c.finished) >> std::quoted(c.checkpoint)))
            throw std::runtime_error("invalid Stage1 completion receipt: " + receipt);
        const auto evidence = bytes.find("\nECM-ADVANCE\n");
        if (evidence != bytes.npos) {
            std::istringstream versions(bytes.substr(evidence + 13));
            uint64_t before = 0, after = 0;
            if (!(versions >> before >> after))
                throw std::runtime_error("invalid queue commit evidence: " + receipt);
            const auto current =
                ecm_io::fingerprint(ecm_io::read_file(std::filesystem::u8path(queue)));
            if (current == after)
                return true;
            if (current != before)
                throw std::runtime_error("queue changed across an interrupted completion; receipt "
                                         "retained for inspection: " +
                                         receipt);
        }
        if (!ecm_append_text_line_unique(c.finished, c.record))
            throw std::runtime_error("cannot publish finished record: " + c.finished);
        const auto payload = evidence == bytes.npos ? bytes : bytes.substr(0, evidence);
        const auto action = c.replacement.empty() ? WorktodoAction::Remove :
                            c.append_replacement ? WorktodoAction::AppendToWorker : WorktodoAction::Replace;
        if (!ecm_worktodo_advance(queue, worker, c.ticket, action, c.replacement, receipt, payload))
            throw std::runtime_error(
                "cannot match/advance completed task; result and completion receipt retained: " +
                queue);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
