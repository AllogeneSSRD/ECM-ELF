#include "p95_worktodo.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <sys/stat.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

namespace {

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::string upper(std::string s) {
    for (char &c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

// Split on ',' honouring a double-quoted tail (the known-factors field).
std::vector<std::string> split_fields(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') { in_quotes = !in_quotes; cur += c; continue; }
        if (c == ',' && !in_quotes) { out.push_back(trim(cur)); cur.clear(); continue; }
        cur += c;
    }
    out.push_back(trim(cur));
    return out;
}

// The known-factor list is the only quoted field, and it is always last:
//   ECM=k,b,n,c,B1[,B2][,curves][,sigma][,"f1,f2,..."]
// Its presence shifts the optional positional fields, so peel it off first and
// return the inner text (quotes removed). `body` is left without the tail.
std::string peel_quoted_tail(std::string &body) {
    const size_t q1 = body.find('"');
    if (q1 == std::string::npos) return std::string();
    const size_t q2 = body.rfind('"');
    if (q2 <= q1) return std::string();
    const std::string inner = body.substr(q1 + 1, q2 - q1 - 1);
    std::string head = body.substr(0, q1);
    head = trim(head);
    if (!head.empty() && head.back() == ',') head.pop_back();
    body = head;
    return inner;
}

// Index of the first numeric field, skipping an optional AID / N/A / FFT2=.
size_t numeric_field_start(const std::vector<std::string> &f) {
    if (f.empty()) return 0;
    if (f[0] == "N/A" || f[0].empty()) return 1;
    if (upper(f[0]).compare(0, 5, "FFT2=") == 0) return 1;
    bool all_hex = true;
    for (char c : f[0]) if (!std::isxdigit((unsigned char)c)) { all_hex = false; break; }
    if (all_hex && f[0].size() >= 16) return 1;
    return 0;
}

// "[Worker #7]" (optionally prefixed with Prime95's ";;MOVED;;" marker).
// Returns the worker number, or -1 when the line is not a section header.
int parse_worker_header(const std::string &line) {
    std::string s = trim(line);
    const std::string moved = ";;MOVED;;";
    if (s.compare(0, moved.size(), moved) == 0) s = trim(s.substr(moved.size()));
    if (s.size() < 10) return -1;
    if (s[0] != '[') return -1;
    if (upper(s).compare(0, 9, "[WORKER #") != 0) return -1;
    const size_t close = s.find(']');
    if (close == std::string::npos || close < 9) return -1;
    const std::string num = trim(s.substr(9, close - 9));
    if (num.empty()) return -1;
    for (char c : num) if (!std::isdigit((unsigned char)c)) return -1;
    return std::atoi(num.c_str());
}

} // namespace

bool p95_read_worktodo(const std::string &path, std::vector<P95WorkerSection> &sections,
                       std::string &err) {
    sections.clear();
    std::ifstream in(path);
    if (!in) {
        struct stat st;
        if (stat(path.c_str(), &st) != 0) return true;   // missing file: no work
        err = "cannot open " + path;
        return false;
    }

    P95WorkerSection head;
    head.worker = 0;
    sections.push_back(head);

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const int w = parse_worker_header(line);
        if (w >= 0) {
            P95WorkerSection s;
            s.worker = w;
            sections.push_back(s);
            continue;
        }
        sections.back().lines.push_back(line);
    }

    // Drop the synthetic header-less section when it stayed empty.
    if (!sections.empty() && sections.front().worker == 0 && sections.front().lines.empty()) {
        sections.erase(sections.begin());
    }
    return true;
}

bool p95_line_is_active(const std::string &line) {
    const std::string s = trim(line);
    if (s.empty()) return false;
    if (s[0] == '#') return false;
    if (s.compare(0, 9, ";;MOVED;;") == 0) return false;
    return true;
}

bool p95_line_is_handoff(const std::string &line) {
    const std::string s = trim(line);
    if (s.empty() || s[0] == '#') return false;

    std::string body = s;
    const std::string kw = upper(body);
    if (kw.compare(0, 4, "ECM=") == 0) {
        body = body.substr(4);
    } else if (kw.compare(0, 5, "ECM2=") == 0) {
        body = body.substr(5);
    } else {
        return false;
    }

    peel_quoted_tail(body);                 // 已知因子串不是位置字段
    std::vector<std::string> f = split_fields(body);
    const size_t i = numeric_field_start(f);

    // Expect: k,b,n,c,B1,B2,curves,sigma
    if (f.size() < i + 8) return false;
    const long curves = std::atol(f[i + 6].c_str());
    const unsigned long long sigma = std::strtoull(f[i + 7].c_str(), nullptr, 10);
    return curves == 1 && sigma != 0;
}

size_t p95_count_active(const P95WorkerSection &s) {
    size_t n = 0;
    for (const std::string &l : s.lines) if (p95_line_is_active(l)) n++;
    return n;
}

size_t p95_count_handoff(const P95WorkerSection &s) {
    size_t n = 0;
    for (const std::string &l : s.lines) if (p95_line_is_handoff(l)) n++;
    return n;
}

size_t p95_total_handoff(const std::vector<P95WorkerSection> &sections) {
    size_t n = 0;
    for (const P95WorkerSection &s : sections) n += p95_count_handoff(s);
    return n;
}

bool p95_write_worktodo_add(const std::string &path,
                            const std::vector<std::pair<int, std::string>> &assignments,
                            bool append, std::string &err) {
    std::vector<P95WorkerSection> sections;
    if (append && !p95_read_worktodo(path, sections, err)) return false;
    if (!append) sections.clear();

    for (const std::pair<int, std::string> &a : assignments) {
        P95WorkerSection *target = nullptr;
        for (P95WorkerSection &s : sections) {
            if (s.worker == a.first) { target = &s; break; }
        }
        if (!target) {
            P95WorkerSection s;
            s.worker = a.first;
            sections.push_back(s);
            target = &sections.back();
        }
        target->lines.push_back(a.second);
    }

    // Keep a deterministic, ascending worker order.
    std::stable_sort(sections.begin(), sections.end(),
                     [](const P95WorkerSection &a, const P95WorkerSection &b) {
                         return a.worker < b.worker;
                     });

    const std::string tmp = path + ".feeder.tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) { err = "cannot write " + tmp; return false; }
        bool first = true;
        for (const P95WorkerSection &s : sections) {
            if (!first) out << "\n";
            first = false;
            if (s.worker > 0) out << "[Worker #" << s.worker << "]\n";
            for (const std::string &l : s.lines) out << l << "\n";
        }
        out.close();
        if (out.fail()) { err = "write failed: " + tmp; return false; }
    }

    // Atomic-ish replace (Prime95 may be reading the file concurrently).
    remove(path.c_str());
    if (rename(tmp.c_str(), path.c_str()) != 0) {
#ifdef _WIN32
        // Windows cannot rename over an existing file; remove() above should have
        // handled it, so retry once after a tiny delay.
        Sleep(50);
        remove(path.c_str());
        if (rename(tmp.c_str(), path.c_str()) != 0) {
            err = "cannot replace " + path;
            remove(tmp.c_str());
            return false;
        }
#else
        err = "cannot replace " + path;
        remove(tmp.c_str());
        return false;
#endif
    }
    return true;
}

bool p95_read_prime_int(const std::string &prime_txt_path, const std::string &key,
                        long long &value) {
    std::ifstream in(prime_txt_path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string s = trim(line);
        if (s.empty() || s[0] == '#' || s[0] == '[') continue;
        const size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        if (trim(s.substr(0, eq)) != key) continue;
        const std::string v = trim(s.substr(eq + 1));
        if (v.empty()) continue;
        char *end = nullptr;
        const long long n = std::strtoll(v.c_str(), &end, 10);
        if (end == v.c_str()) continue;
        value = n;
        return true;
    }
    return false;
}

std::string p95_ecm_save_name(uint32_t n) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "e%07u", n);
    return std::string(buf);
}

uint32_t p95_read_save_state(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0xFFFFFFFFu;
    in.seekg(0x40, std::ios::beg);
    unsigned char b[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char *>(b), 4);
    if (!in) return 0xFFFFFFFFu;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

bool p95_list_tmp_files(const std::string &dir, std::vector<std::string> &names) {
    names.clear();
    std::string d = dir;
    if (d.empty()) d = ".";
    struct Entry { std::string name; long long mtime; };
    std::vector<Entry> found;

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    const std::string pattern = d + "\\*.tmp";
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        // An empty directory is not an error.
        return true;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ULARGE_INTEGER t;
        t.LowPart = fd.ftLastWriteTime.dwLowDateTime;
        t.HighPart = fd.ftLastWriteTime.dwHighDateTime;
        found.push_back({std::string(fd.cFileName), (long long)t.QuadPart});
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *dp = opendir(d.c_str());
    if (!dp) return false;
    struct dirent *de;
    while ((de = readdir(dp)) != nullptr) {
        const std::string name = de->d_name;
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".tmp") != 0) continue;
        struct stat st;
        if (stat((d + "/" + name).c_str(), &st) != 0) continue;
        found.push_back({name, (long long)st.st_mtime});
    }
    closedir(dp);
#endif

    std::sort(found.begin(), found.end(), [](const Entry &a, const Entry &b) {
        if (a.mtime != b.mtime) return a.mtime < b.mtime;
        return a.name < b.name;
    });
    for (const Entry &e : found) names.push_back(e.name);
    return true;
}
