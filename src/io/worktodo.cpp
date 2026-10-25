#include "io/worktodo.h"
#include "io/files.h"
#include "io/config_format.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <utility>
#include <vector>
#include <cerrno>
#include <climits>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

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

// Split a CSV row, honouring double quotes and "" escapes. Each field is trimmed.
std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
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
                in_quotes = true;
            } else if (ch == ',') {
                out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(ch);
            }
        }
    }
    out.push_back(cur);
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

bool ecm_rewrite_stage2_save(const std::string& line, const std::string& save,
                             std::string& rewritten, std::string& err) {
    const bool stage1 = line.compare(0, 10, "ECMSTAGE1=") == 0;
    EcmStage2Task task;
    if (!(stage1 ? ecm_parse_stage1_line(line, task, err) : ecm_parse_stage2_line(line, task, err)))
        return false;
    const auto begin = line.find('=') + 1;
    const auto fields = split_csv(line.substr(begin));
    const size_t wanted = is_int_like(fields[0]) ? 4 : 5;
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
            err = "cannot rewrite Stage1 save name";
            return false;
        }
        ++column;
        start = i + 1;
    }
    err = "missing Stage1 save-name field";
    return false;
}

static bool parse_save_line(const std::string& line, const std::string& prefix, EcmStage2Task& task,
                            std::string& err) {
    std::string s = line;
    strip_bom(s);
    trim(s);
    task = EcmStage2Task{};
    task.raw_line = s;

    if (!csv_quotes_valid(s)) {
        err = "key=CSV value='" + s +
              "'; expected a closing double quote (escape quotes as two double quotes)";
        return false;
    }

    if (s.size() <= prefix.size() || s.compare(0, prefix.size(), prefix) != 0) {
        err = "line does not start with " + prefix;
        return false;
    }

    const std::vector<std::string> cols = split_csv(s.substr(prefix.size()));
    std::size_t idx = 0;
    if (!is_int_like(cols[0])) {
        task.aid = cols[0]; // optional AID ("N/A" or empty also lands here)
        idx = 1;
    }

    const std::size_t need = idx + 8;
    if (cols.size() < need) {
        err = "not enough fields (need at least " + std::to_string(8) + " core columns)";
        return false;
    }
    if (cols.size() > need + 1) {
        err = "key=CSV; expected 8 core fields and at most one quoted known-factors field";
        return false;
    }

    task.k = cols[idx];
    task.b = cols[idx + 1];
    task.c = cols[idx + 3];
    task.save_name = cols[idx + 4];
    try {
        task.b2_text = cols[idx + 5].empty() ? "0" : cols[idx + 5];
        task.has_b2 = !cols[idx + 5].empty();
        task.has_skip = !cols[idx + 6].empty();
        (void)ecm_config::unsigned_integer(task.b2_text, "B2 (integer >=0)");
        task.skip_curves =
            cols[idx + 6].empty()
                ? 0
                : ecm_config::unsigned_integer(cols[idx + 6], "skip_curves (integer >=0)");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    // n (exponent) must fit in unsigned long.
    {
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

    // curves_to_run.
    {
        const std::string& cstr = cols[idx + 7];
        char* end = nullptr;
        errno = 0;
        const unsigned long v = std::strtoul(cstr.c_str(), &end, 10);
        if (errno == ERANGE || cstr.empty() || cstr[0] == '-' || end == cstr.c_str() ||
            *end != '\0' || v == 0 || v > 0xFFFFFFFFul) {
            err = "key=curves_to_run value='" + cstr + "'; expected integer [1,4294967295]";
            return false;
        }
        task.curves_to_run = static_cast<uint32_t>(v);
    }

    // Optional known factors: a single quoted field "f1,f2,...".
    if (cols.size() > need) {
        const std::vector<std::string> facs = split_csv(cols[need]);
        for (const std::string& f : facs) {
            if (!f.empty()) {
                task.factors.push_back(f);
            }
        }
    }
    return true;
}

bool ecm_parse_stage1_line(const std::string& line, EcmStage2Task& task, std::string& err) {
    return parse_save_line(line, "ECMSTAGE1=", task, err);
}

bool ecm_parse_stage2_line(const std::string& line, EcmStage2Task& task, std::string& err) {
    return parse_save_line(line, "ECMSTAGE2=", task, err);
}

bool ecm_rewrite_stage2_range(const std::string& line, uint64_t skip, uint32_t count,
                              std::string& rewritten, std::string& err) {
    EcmStage2Task task;
    const bool stage1 = line.compare(0, 10, "ECMSTAGE1=") == 0;
    if (!(stage1 ? ecm_parse_stage1_line(line, task, err) : ecm_parse_stage2_line(line, task, err)))
        return false;
    const auto begin = line.find('=') + 1;
    const auto fields = split_csv(line.substr(begin));
    const size_t offset = is_int_like(fields[0]) ? 0 : 1;
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
    const auto s = spans.at(offset + 6), c = spans.at(offset + 7);
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
    // 格式: [<AID>,|N/A,|<nul>][FFT2=<fftl>,|<nul>]<k>,<b>,<n>,<c>,<B1>[,<B2>][,<curves>][,<sigma>][,"factors"]
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
    if (!is_int_like(cols[idx]) && !starts_with(cols[idx], "FFT2=")) {
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

    // Optional positional fields (匹配 Prime95): B2 (默认 0), curves_to_run (默认 100), sigma, factors。
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
    return compute_n_from_fields(task.k, task.b, task.n, task.c, task.factors, N, err);
}

bool ecm_compute_ecm2_n(const Ecm2Task& task, mpz_t N, std::string& err) {
    return compute_n_from_fields(task.k, task.b, task.n, task.c, task.factors, N, err);
}

int ecm_worktodo_parse_worker_header(const std::string& line, bool* is_bracket_line) {
    return ecm_config::worker_header(line, is_bracket_line);
}

namespace {

// Classification of one raw line for the section-aware queue walk. Keeping the
// original (untrimmed) text is what lets the rewrite preserve the file verbatim.
enum class LineKind { Task, Comment, Header, ForeignBracket };

LineKind classify_line(const std::string& raw, int* section_out, int* header_out) {
    std::string t = raw;
    strip_bom(t);
    trim(t);
    bool bracket = false;
    const int w = ecm_worktodo_parse_worker_header(t, &bracket);
    if (w > 0) {
        if (header_out)
            *header_out = w;
        if (section_out)
            *section_out = w;
        return LineKind::Header;
    }
    if (bracket) {
        return LineKind::ForeignBracket;
    }
    if (t.empty() || t[0] == '#') {
        return LineKind::Comment;
    }
    return LineKind::Task;
}

bool worker_matches(int section, int worker) {
    return worker <= 0 || section == worker;
}

struct QueueEditLock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    explicit QueueEditLock(const std::string& path) {
        const std::string name = path + ".edit.lock";
        for (int attempt = 0; attempt < 100; ++attempt) {
#ifdef _WIN32
            const std::wstring wide = utf8_to_wide(name);
            handle = CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                 OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle != INVALID_HANDLE_VALUE)
                break;
#else
            fd = open(name.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0)
                break;
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    bool valid() const {
#ifdef _WIN32
        return handle != INVALID_HANDLE_VALUE;
#else
        return fd >= 0;
#endif
    }

    ~QueueEditLock() {
#ifdef _WIN32
        if (valid())
            CloseHandle(handle);
#else
        if (valid()) {
            flock(fd, LOCK_UN);
            close(fd);
        }
#endif
    }
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
    std::string text = line;
    strip_bom(text);
    trim(text);
    if (starts_with(text, "ECMSTAGE2="))
        return WorktodoStage::Stage2;
    // ECM/ECM2 remain native Prime95 Stage1 assignments. An unknown task is
    // diagnosed by Stage1 rather than silently hidden from the queue.
    return WorktodoStage::Stage1;
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
    int section = 1;
    std::string first, preferred, raw;
    size_t number = 0, first_number = 0, preferred_number = 0;
    while (std::getline(in, raw)) {
        ++number;
        if (classify_line(raw, &section, nullptr) != LineKind::Task ||
            !worker_matches(section, worker))
            continue;
        std::string task = raw;
        strip_bom(task);
        trim(task);
        const auto stage = ecm_worktodo_stage(task);
        if (selection.skip == static_cast<int>(stage))
            continue;
        if (first.empty()) {
            first = task;
            first_number = number;
        }
        if (selection.priority == static_cast<int>(stage) && preferred.empty()) {
            preferred = task;
            preferred_number = number;
        }
    }
    if (in.bad()) {
        err = "worktodo read failed: " + path;
        return WorktodoReadResult::Error;
    }
    const std::string& selected = preferred.empty() ? first : preferred;
    if (selected.empty() ||
        (consumer != WorktodoStage::Any && ecm_worktodo_stage(selected) != consumer))
        return WorktodoReadResult::Empty;
    line = selected;
    if (source_line)
        *source_line = preferred.empty() ? first_number : preferred_number;
    return WorktodoReadResult::Task;
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

bool ecm_worktodo_advance(const std::string& path, int worker, const std::string& first_line,
                          WorktodoAction action, const std::string& replacement,
                          const std::string& receipt, const std::string& receipt_payload) {
    if (action == WorktodoAction::Replace &&
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
        if (action == WorktodoAction::Remove) {
            lines.erase(lines.begin() + static_cast<std::vector<std::string>::difference_type>(i));
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
        err = "cannot lock Stage2 queue: " + path;
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
    if (!read_lines(path, lines) || !read_lines(finished, done)) {
        err = "cannot read Stage2 queue or finished record";
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

    // Append to the last existing segment of this worker, before the next
    // worker's header. The leading unlabelled segment belongs to worker 1.
    section = 1;
    bool found_section = worker == 1;
    std::size_t insertion = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
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
    lines.insert(lines.begin() + static_cast<std::vector<std::string>::difference_type>(insertion),
                 line);
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
        out << "ECM-STAGE1-COMPLETED 1\n"
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
        if (magic != "ECM-STAGE1-COMPLETED 1" ||
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
        if (!ecm_worktodo_advance(queue, worker, c.ticket,
                                  c.replacement.empty() ? WorktodoAction::Remove
                                                        : WorktodoAction::Replace,
                                  c.replacement, receipt, payload))
            throw std::runtime_error(
                "cannot match/advance completed task; result and completion receipt retained: " +
                queue);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

namespace ecm_workers {
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == s.npos ? std::string{}
                           : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

// Shared list/range syntax. Callers decide whether out-of-range IDs are fatal.
bool parse_list(const std::string& spec, int maximum, std::vector<int>& workers, int& ignored) {
    workers.clear();
    ignored = 0;
    if (maximum < 1)
        return false;
    std::vector<unsigned char> seen(size_t(maximum) + 1);
    const auto number = [](const std::string& text, long long& value) {
        const auto s = trim(text);
        if (s.empty())
            return false;
        char* end = nullptr;
        errno = 0;
        value = std::strtoll(s.c_str(), &end, 10);
        return errno != ERANGE && end != s.c_str() && !*end;
    };
    std::istringstream input(spec);
    std::string item;
    while (std::getline(input, item, ',')) {
        item = trim(item);
        if (item.empty())
            continue;
        const auto dash = item.find('-');
        long long low = 0, high = 0;
        if (dash == item.npos) {
            if (!number(item, low))
                return false;
            high = low;
        } else if (!number(item.substr(0, dash), low) || !number(item.substr(dash + 1), high))
            return false;
        if (high < low)
            std::swap(low, high);
        // Bound iteration even for a very large user-supplied range.
        long double outside = 0;
        if (low < 1)
            outside += static_cast<long double>((std::min)(high, 0LL)) - low + 1;
        if (high > maximum)
            outside += static_cast<long double>(high) -
                       (std::max)(low, static_cast<long long>(maximum) + 1) + 1;
        ignored = int((std::min)(static_cast<long double>(INT_MAX), ignored + outside));
        low = (std::max)(1LL, low);
        high = (std::min)(static_cast<long long>(maximum), high);
        for (auto id = low; id <= high; ++id)
            if (!seen[size_t(id)]) {
                seen[size_t(id)] = 1;
                workers.push_back(int(id));
            }
    }
    return true;
}
} // namespace ecm_workers
