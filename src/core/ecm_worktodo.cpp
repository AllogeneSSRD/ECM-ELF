#include "ecm_worktodo.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

void trim(std::string &s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
}

void strip_bom(std::string &s) {
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
}

bool is_int_like(const std::string &s) {
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

// Split a CSV row, honouring double quotes and "" escapes. Each field is trimmed.
std::vector<std::string> split_csv(const std::string &s) {
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
    for (std::string &f : out) {
        trim(f);
    }
    return out;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string &s) {
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

long long filetime_to_epoch(const FILETIME &ft) {
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return static_cast<long long>(ull.QuadPart / 10000000ULL - 11644473600ULL);
}
#endif

bool replace_file(const std::string &tmp, const std::string &dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

} // namespace

bool ecm_parse_stage2_line(const std::string &line, EcmStage2Task &task, std::string &err) {
    std::string s = line;
    strip_bom(s);
    trim(s);
    task = EcmStage2Task{};
    task.raw_line = s;

    const std::string prefix = "ECMSTAGE2=";
    if (s.size() <= prefix.size() ||
        s.compare(0, prefix.size(), prefix) != 0) {
        err = "line does not start with ECMSTAGE2=";
        return false;
    }

    const std::vector<std::string> cols = split_csv(s.substr(prefix.size()));
    std::size_t idx = 0;
    if (!is_int_like(cols[0])) {
        task.aid = cols[0];      // optional AID ("N/A" or empty also lands here)
        idx = 1;
    }

    const std::size_t need = idx + 8;
    if (cols.size() < need) {
        err = "not enough fields (need at least " + std::to_string(8) + " core columns)";
        return false;
    }

    task.k = cols[idx];
    task.b = cols[idx + 1];
    task.c = cols[idx + 3];
    task.save_name = cols[idx + 4];
    // cols[idx+5] = B2 (ignored), cols[idx+6] = skip_curves (ignored).

    // n (exponent) must fit in unsigned long.
    {
        const std::string &nstr = cols[idx + 2];
        if (!is_int_like(nstr) || nstr.empty() || nstr[0] == '-') {
            err = "invalid exponent n: '" + nstr + "'";
            return false;
        }
        char *end = nullptr;
        const unsigned long v = std::strtoul(nstr.c_str(), &end, 10);
        if (end == nstr.c_str() || *end != '\0') {
            err = "invalid exponent n: '" + nstr + "'";
            return false;
        }
        task.n = v;
    }

    // curves_to_run.
    {
        const std::string &cstr = cols[idx + 7];
        char *end = nullptr;
        const unsigned long v = std::strtoul(cstr.c_str(), &end, 10);
        if (end == cstr.c_str() || *end != '\0' || v == 0 || v > 0xFFFFFFFFul) {
            err = "invalid curves_to_run: '" + cstr + "'";
            return false;
        }
        task.curves_to_run = static_cast<uint32_t>(v);
    }

    // Optional known factors: a single quoted field "f1,f2,...".
    if (cols.size() > need) {
        const std::vector<std::string> facs = split_csv(cols[need]);
        for (const std::string &f : facs) {
            if (!f.empty()) {
                task.factors.push_back(f);
            }
        }
    }
    return true;
}

bool ecm_extract_b1_from_save_name(const std::string &save_name, double *b1_out, std::string &err) {
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
    char *end = nullptr;
    const double b1 = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || b1 <= 0.0) {
        err = "cannot parse B1 from save_name token '" + token + "'";
        return false;
    }
    *b1_out = b1;
    return true;
}

bool ecm_compute_stage2_n(const EcmStage2Task &task, mpz_t N, std::string &err) {
    mpz_t k, b, c;
    mpz_init(k);
    mpz_init(b);
    mpz_init(c);

    bool ok = true;
    if (mpz_set_str(k, task.k.c_str(), 10) != 0) {
        err = "invalid k: '" + task.k + "'";
        ok = false;
    } else if (mpz_set_str(b, task.b.c_str(), 10) != 0) {
        err = "invalid b: '" + task.b + "'";
        ok = false;
    } else if (mpz_set_str(c, task.c.c_str(), 10) != 0) {
        err = "invalid c: '" + task.c + "'";
        ok = false;
    }

    mpz_t bn;
    mpz_init(bn);
    if (ok) {
        mpz_pow_ui(bn, b, task.n);       // b^n
        mpz_mul(N, k, bn);               // k*b^n
        mpz_add(N, N, c);                // k*b^n + c
        if (mpz_sgn(N) <= 0) {
            err = "computed N is not positive";
            ok = false;
        }
    }

    if (ok) {
        mpz_t f;
        mpz_init(f);
        for (const std::string &fs : task.factors) {
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

bool ecm_worktodo_first_line(const std::string &path, std::string &line) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string l;
    while (std::getline(in, l)) {
        strip_bom(l);
        trim(l);
        if (l.empty() || l[0] == '#') {
            continue;
        }
        line = l;
        return true;
    }
    return false;
}

bool ecm_worktodo_advance(const std::string &path, const std::string &first_line,
                          WorktodoAction action) {
    (void)first_line;  // We re-scan for the first task line instead of trusting the caller.
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::vector<std::string> lines;
    std::string l;
    while (std::getline(in, l)) {
        lines.push_back(l);
    }
    in.close();

    bool found = false;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::string t = lines[i];
        strip_bom(t);
        trim(t);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        // This is the first task line. `t` should equal `first_line`; if the
        // file changed under us, still act on the line we actually found.
        if (action == WorktodoAction::Remove) {
            lines.erase(lines.begin() + static_cast<std::vector<std::string>::difference_type>(i));
        } else {
            lines[i] = "# ERROR " + t;
        }
        found = true;
        break;
    }
    if (!found) {
        return false;
    }

    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    for (const std::string &ln : lines) {
        out << ln << "\n";
    }
    out.close();
    if (out.fail()) {
        return false;
    }
    return replace_file(tmp, path);
}

bool ecm_append_text_line(const std::string &path, const std::string &line) {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        return false;
    }
    out << line << "\n";
    return !out.fail();
}

void ecm_sync_save_files(const std::string &dir, const std::string &sync_dir_1,
                         const std::string &sync_dir_2, bool full,
                         long long since_epoch_seconds) {
    if (sync_dir_1.empty() && sync_dir_2.empty()) {
        return;
    }
#ifdef _WIN32
    // Use wide APIs so CJK paths (e.g. GIMPS_同步) survive the UTF-8 ini → Win32
    // conversion correctly.
    const std::wstring wdir = utf8_to_wide(dir);
    const std::wstring wpat = wdir + L"\\*.save";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wpat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    const std::wstring wsync1 = utf8_to_wide(sync_dir_1);
    const std::wstring wsync2 = utf8_to_wide(sync_dir_2);
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }
        if (!full && filetime_to_epoch(fd.ftLastWriteTime) <= since_epoch_seconds) {
            continue;
        }
        const std::wstring wsrc = wdir + L"\\" + fd.cFileName;
        if (!wsync1.empty()) {
            CopyFileW(wsrc.c_str(), (wsync1 + L"\\" + fd.cFileName).c_str(), FALSE);
        }
        if (!wsync2.empty()) {
            CopyFileW(wsrc.c_str(), (wsync2 + L"\\" + fd.cFileName).c_str(), FALSE);
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) {
        return;
    }
    const auto copy_narrow = [](const std::string &src, const std::string &dst) {
        std::ifstream in(src, std::ios::binary);
        if (!in.is_open()) {
            return;
        }
        std::ofstream out(dst, std::ios::binary | std::ios::trunc);
        if (out.is_open()) {
            out << in.rdbuf();
        }
    };
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        const std::string name = e->d_name;
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".save") != 0) {
            continue;
        }
        const std::string fullpath = dir + "/" + name;
        struct stat st;
        if (stat(fullpath.c_str(), &st) != 0) {
            continue;
        }
        if (!full && static_cast<long long>(st.st_mtime) <= since_epoch_seconds) {
            continue;
        }
        if (!sync_dir_1.empty()) {
            copy_narrow(fullpath, sync_dir_1 + "/" + name);
        }
        if (!sync_dir_2.empty()) {
            copy_narrow(fullpath, sync_dir_2 + "/" + name);
        }
    }
    closedir(d);
#endif
}
