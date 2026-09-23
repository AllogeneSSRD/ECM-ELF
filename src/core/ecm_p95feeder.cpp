// ecm_p95feeder.cpp — standalone Prime95 handoff feeder.
//
// Split out of ecm.exe: ecm.exe now only writes local stage-1 saves
// (<tmp_dir>/e{n:07d}_c{curve:06d}.tmp, MIDSTAGE state=2). This program watches
// that local directory and moves each finished curve into a running Prime95
// instance, so Prime95 can run stage 2 on our stage-1 output.
//
// Each delivery:
//   1. copy <tmp_dir>/e{n:07d}_c{k}.tmp  ->  <p95_dir>/e{n:07d}
//   2. append an ECM= line (curves=1 + our specific sigma, plus the known-factor
//      list taken from the matching prime95 worktodo entry) into
//      <p95_dir>/worktodo.add under a free "[Worker #N]" section.
//
// Prime95 then appends the entries of each "[Worker #]" section into
// worktodo.txt and deletes worktodo.add (undoc.txt), which is why we deliver via
// worktodo.add instead of rewriting worktodo.txt behind a running Prime95.
//
// Safety rules:
//   * a save is only accepted when its N and B1 match an entry of the configured
//     local worktodo file(s)  ("保存文件需要N和B1与worktodo对应上");
//   * at most `max_in_flight` handoffs are outstanding (default: Prime95's
//     MaxHighMemWorkers, read from prime.txt) — stage 2 is the high-memory phase;
//   * at most one handoff per N is outstanding, because Prime95 names the ECM
//     save e{n:07d} per exponent (two in flight would clobber each other);
//   * tasks are only placed into worker sections that are currently empty.
//
// Usage:
//   ecm_p95feeder [--ini <file>] [--once] [--dry-run] [-v] [--help]

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gmp.h>

#include "ecm_edwards_save.h"
#include "ecm_worktodo.h"
#include "p95_worktodo.h"

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#define FEEDER_ACCESS _access
#else
#include <unistd.h>
#define FEEDER_ACCESS access
#endif

namespace {

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct FeederConfig {
    std::string tmp_dir = ".";            // 本地 stage-1 落盘目录 (待转发)
    std::string p95_dir;                  // Prime95 工作目录
    std::string worktodo = "worktodo.txt,worktodo.finished.txt";  // 校验用 (逗号分隔)
    std::string exe_dir = ".";            // exe 目录 (worktodo 路径回退查找用)
    int poll_seconds = 5;                 // 轮询间隔
    int max_in_flight = 0;                // 0 = 用 p95 prime.txt 的 MaxHighMemWorkers
    std::string worker_allow;             // "" = 所有 Worker; "1,2" = 仅这些
    std::string log_file;                 // 空 = 只输出到屏幕
    int verbose = 1;
    int dry_run = 0;                      // 1 = 只打印, 不改任何文件
    int keep_tmp = 0;                     // 1 = 转发后保留本地 .tmp
};

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { out.push_back(trim(cur)); cur.clear(); continue; }
        cur += c;
    }
    out.push_back(trim(cur));
    return out;
}

bool load_ini(const std::string &path, FeederConfig &cfg, std::string &err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string s = trim(line);
        if (s.empty() || s[0] == '#' || s[0] == '[') continue;
        const size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(s.substr(0, eq));
        const std::string val = trim(s.substr(eq + 1));
        if (key == "tmp_dir") cfg.tmp_dir = val;
        else if (key == "p95_dir") cfg.p95_dir = val;
        else if (key == "worktodo") cfg.worktodo = val;
        else if (key == "poll_seconds") cfg.poll_seconds = std::atoi(val.c_str());
        else if (key == "max_in_flight") cfg.max_in_flight = std::atoi(val.c_str());
        else if (key == "worker_allow") cfg.worker_allow = val;
        else if (key == "log_file") cfg.log_file = val;
        else if (key == "verbose") cfg.verbose = std::atoi(val.c_str());
        else if (key == "dry_run") cfg.dry_run = std::atoi(val.c_str());
        else if (key == "keep_tmp") cfg.keep_tmp = std::atoi(val.c_str());
    }
    if (cfg.poll_seconds < 1) cfg.poll_seconds = 1;
    if (cfg.max_in_flight < 0) cfg.max_in_flight = 0;
    return true;
}

std::string with_sep(std::string dir) {
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
    return dir;
}

FILE *g_log = nullptr;

void log_line(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    // 时间戳 (与 driver 的日志风格一致)
    std::time_t t = std::time(nullptr);
    std::tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);

    std::printf("[%s] %s\n", ts, buf);
    std::fflush(stdout);
    if (g_log) {
        std::fprintf(g_log, "[%s] %s\n", ts, buf);
        std::fflush(g_log);
    }
}

// ---------------------------------------------------------------------------
// worktodo validation ("N 和 B1 要与 worktodo 对应上")
// ---------------------------------------------------------------------------

struct WorkExpectation {
    unsigned long n = 0;
    uint64_t B1 = 0;
    unsigned long b = 2;
    long c = -1;
    std::string factors;      // 原文的已知因子串 (含引号内内容), 用于转发
    std::string raw_line;
};

// 从一行 ECM=/ECM2= worktodo 里取出 (b, n, c, B1, factors); 非 ECM 行返回 false.
bool parse_expectation(const std::string &line, WorkExpectation &out) {
    std::string s = trim(line);
    if (s.empty() || s[0] == '#') return false;
    std::string body;
    if (s.size() > 4 && (s.compare(0, 4, "ECM=") == 0)) body = s.substr(4);
    else if (s.size() > 5 && (s.compare(0, 5, "ECM2=") == 0)) body = s.substr(5);
    else return false;

    // 已知因子串是唯一的引号字段, 且总在末尾 —— 先剥离它, 否则可选字段会错位.
    std::string factors;
    {
        const size_t q1 = body.find('"');
        if (q1 != std::string::npos) {
            const size_t q2 = body.rfind('"');
            if (q2 > q1) {
                factors = body.substr(q1 + 1, q2 - q1 - 1);
                std::string head = trim(body.substr(0, q1));
                if (!head.empty() && head.back() == ',') head.pop_back();
                body = head;
            }
        }
    }

    std::vector<std::string> f;
    std::string cur;
    for (size_t i = 0; i <= body.size(); ++i) {
        if (i == body.size() || body[i] == ',') { f.push_back(trim(cur)); cur.clear(); continue; }
        cur += body[i];
    }

    size_t i = 0;
    if (i < f.size() && (f[i] == "N/A" || f[i].empty())) i++;
    else if (i < f.size() && f[i].compare(0, 5, "FFT2=") == 0) i++;
    else if (i < f.size()) {
        bool all_hex = !f[i].empty();
        for (char ch : f[i]) if (!std::isxdigit((unsigned char)ch)) { all_hex = false; break; }
        if (all_hex && f[i].size() >= 16) i++;
    }
    if (f.size() < i + 5) return false;

    char *end = nullptr;
    out.b = std::strtoul(f[i + 1].c_str(), &end, 10);
    out.n = std::strtoul(f[i + 2].c_str(), &end, 10);
    out.c = std::strtol(f[i + 3].c_str(), &end, 10);
    out.B1 = std::strtoull(f[i + 4].c_str(), &end, 10);
    out.factors = factors;
    out.raw_line = s;
    return out.n != 0 && out.B1 != 0;
}

// 解析一个 worktodo 路径: 绝对路径直接使用; 否则依次尝试 tmp_dir、exe 目录、CWD.
// 返回第一个存在的路径 (都不存在则返回空串 + 填 tried 供日志).
std::string resolve_worktodo_path(const FeederConfig &cfg, const std::string &name,
                                 std::vector<std::string> *tried) {
    const bool absolute = (name.size() >= 2 && name[1] == ':') ||
                          (!name.empty() && (name[0] == '/' || name[0] == '\\'));
    std::vector<std::string> candidates;
    if (absolute) {
        candidates.push_back(name);
    } else {
        candidates.push_back(with_sep(cfg.tmp_dir) + name);   // 与存档同目录 (最常见)
        candidates.push_back(with_sep(cfg.exe_dir) + name);   // 队列目录 (worktodo/finished)
        candidates.push_back(name);                           // 当前工作目录
    }
    for (const std::string &c : candidates) {
        if (tried) tried->push_back(c);
        if (FEEDER_ACCESS(c.c_str(), 0) == 0) return c;
    }
    return std::string();
}

// 收集所有配置的 worktodo 文件里的 ECM 期望 (支持 [Worker #N] 段与普通行).
void load_expectations(const FeederConfig &cfg, std::vector<WorkExpectation> &out,
                       std::vector<std::string> &scanned) {
    out.clear();
    scanned.clear();
    for (const std::string &raw : split_csv(cfg.worktodo)) {
        if (raw.empty()) continue;
        const std::string path = resolve_worktodo_path(cfg, raw, nullptr);
        if (path.empty()) continue;
        scanned.push_back(path);

        std::vector<P95WorkerSection> sections;
        std::string err;
        if (!p95_read_worktodo(path, sections, err)) continue;
        for (const P95WorkerSection &s : sections) {
            for (const std::string &l : s.lines) {
                WorkExpectation e;
                if (parse_expectation(l, e)) out.push_back(e);
            }
        }
    }
}

const WorkExpectation *find_expectation(const std::vector<WorkExpectation> &exps,
                                        const ecm_save_common &cm) {
    for (const WorkExpectation &e : exps) {
        if (e.n != cm.n) continue;
        if (e.b != cm.b) continue;
        if (e.c != cm.c) continue;
        if (e.B1 != cm.B1) continue;
        return &e;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// 单次轮询
// ---------------------------------------------------------------------------

struct Deliverable {
    std::string tmp_name;      // 本地文件名
    uint32_t n = 0;
    uint64_t sigma = 0;
    uint64_t B1 = 0;
    std::string factors;       // 来自 worktodo (可空)
    std::string line;          // 要写进 worktodo.add 的 ECM= 行
    std::string save_name;     // e{n:07d}
};

std::string build_ecm_line(const ecm_save_common &cm, uint64_t sigma,
                           const std::string &factors) {
    char kbuf[64];
    std::snprintf(kbuf, sizeof(kbuf), "%.0f", cm.k);
    std::string line = "ECM=";
    line += kbuf;
    line += "," + std::to_string(cm.b);
    line += "," + std::to_string(cm.n);
    line += "," + std::to_string(cm.c);
    line += "," + std::to_string(cm.B1);
    line += "," + std::to_string(cm.B2);
    line += ",1," + std::to_string(sigma);
    if (!factors.empty()) line += ",\"" + factors + "\"";
    return line;
}

bool copy_file(const std::string &src, const std::string &dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << in.rdbuf();
    out.close();
    return !out.fail();
}

int run_cycle(const FeederConfig &cfg) {
    if (cfg.p95_dir.empty()) {
        log_line("ERROR: p95_dir is not configured");
        return -1;
    }
    const std::string p95 = with_sep(cfg.p95_dir);
    const std::string tmp = with_sep(cfg.tmp_dir);

    // --- 1. Prime95 侧状态 ---
    std::vector<P95WorkerSection> sections;
    std::string err;
    if (!p95_read_worktodo(p95 + "worktodo.txt", sections, err)) {
        log_line("ERROR: %s", err.c_str());
        return -1;
    }

    long long max_high_mem = 1;
    const bool prime_has_key =
        p95_read_prime_int(p95 + "prime.txt", "MaxHighMemWorkers", max_high_mem);
    if (!prime_has_key || max_high_mem < 1) max_high_mem = 1;

    long long num_workers = 0;
    p95_read_prime_int(p95 + "prime.txt", "NumWorkers", num_workers);

    size_t in_flight = p95_total_handoff(sections);
    {
        std::vector<P95WorkerSection> add_sections;
        std::string aerr;
        if (p95_read_worktodo(p95 + "worktodo.add", add_sections, aerr)) {
            in_flight += p95_total_handoff(add_sections);
        }
    }

    const long long cap = (cfg.max_in_flight > 0) ? cfg.max_in_flight : max_high_mem;
    long long slots = cap - (long long)in_flight;

    // 已经在飞的 N (同一 N 只能有一个 handoff: p95 的存档名 e{n:07d} 按指数命名)
    std::set<unsigned long> ns_in_flight;
    for (const P95WorkerSection &s : sections) {
        for (const std::string &l : s.lines) {
            if (!p95_line_is_handoff(l)) continue;
            WorkExpectation e;
            if (parse_expectation(l, e)) ns_in_flight.insert(e.n);
        }
    }

    // --- 2. 空闲 Worker ---
    std::vector<int> free_workers;
    for (const P95WorkerSection &s : sections) {
        if (s.worker <= 0) continue;
        if (p95_count_active(s) != 0) continue;
        if (!cfg.worker_allow.empty()) {
            bool allowed = false;
            for (const std::string &w : split_csv(cfg.worker_allow)) {
                if (std::atoi(w.c_str()) == s.worker) { allowed = true; break; }
            }
            if (!allowed) continue;
        }
        free_workers.push_back(s.worker);
    }
    std::sort(free_workers.begin(), free_workers.end());
    if (free_workers.empty() && sections.empty()) free_workers.push_back(1);  // 无段文件

    // --- 3. 本地待转发 .tmp ---
    std::vector<std::string> tmps;
    if (!p95_list_tmp_files(cfg.tmp_dir, tmps)) {
        log_line("ERROR: cannot list %s", cfg.tmp_dir.c_str());
        return -1;
    }

    std::vector<WorkExpectation> exps;
    std::vector<std::string> scanned;
    load_expectations(cfg, exps, scanned);

    if (cfg.verbose) {
        std::string files;
        for (const std::string &f : scanned) {
            if (!files.empty()) files += ", ";
            files += f;
        }
        log_line("poll: workers=%zu free=%zu in_flight=%zu/%lld pending_tmp=%zu "
                 "expectations=%zu [%s] MaxHighMemWorkers=%lld",
                 sections.size(), free_workers.size(), in_flight, cap, tmps.size(),
                 exps.size(), files.empty() ? "no worktodo file found" : files.c_str(),
                 max_high_mem);
    }

    if (tmps.empty()) return 0;

    int delivered = 0;
    for (const std::string &name : tmps) {
        if (slots <= 0) {
            if (cfg.verbose) log_line("hold %s: in-flight cap reached (%lld)", name.c_str(), cap);
            break;
        }
        if (free_workers.empty()) {
            if (cfg.verbose) log_line("hold %s: no empty [Worker #N] section", name.c_str());
            break;
        }

        const std::string tmp_path = tmp + name;
        ecm_save_common cm;
        mpz_t Qx, Qz;
        mpz_inits(Qx, Qz, NULL);
        const bool ok = ecm_edwards_read_midstage(tmp_path, cm, Qx, Qz);
        mpz_clears(Qx, Qz, NULL);
        if (!ok) {
            log_line("skip %s: not a valid MIDSTAGE save", name.c_str());
            continue;
        }
        if (ns_in_flight.count(cm.n) != 0) {
            if (cfg.verbose) log_line("hold %s: N=%u already in flight", name.c_str(), cm.n);
            continue;
        }

        // N 与 B1 必须与 worktodo 对得上
        const WorkExpectation *exp = nullptr;
        if (!exps.empty()) {
            exp = find_expectation(exps, cm);
            if (!exp) {
                log_line("skip %s: no worktodo entry matches N=%u B1=%llu (k=%.0f b=%u c=%d)",
                         name.c_str(), cm.n, (unsigned long long)cm.B1, cm.k, cm.b, cm.c);
                continue;
            }
        } else if (cfg.verbose) {
            log_line("note: no worktodo file found; accepting %s without N/B1 check",
                     name.c_str());
        }

        Deliverable d;
        d.tmp_name = name;
        d.n = cm.n;
        d.sigma = cm.sigma;
        d.B1 = cm.B1;
        d.factors = exp ? exp->factors : std::string();
        d.line = build_ecm_line(cm, cm.sigma, d.factors);
        d.save_name = p95_ecm_save_name(cm.n);

        const int worker = free_workers.front();
        free_workers.erase(free_workers.begin());

        if (cfg.dry_run) {
            log_line("DRY-RUN would deliver %s -> %s%s as [Worker #%d] %s",
                     name.c_str(), p95.c_str(), d.save_name.c_str(), worker, d.line.c_str());
            delivered++;
            slots--;
            ns_in_flight.insert(cm.n);
            continue;
        }

        // 先落存档, 再写 worktodo.add: p95 捡到 worktodo.add 时会立刻去读存档.
        if (!copy_file(tmp_path, p95 + d.save_name)) {
            log_line("ERROR: cannot copy %s -> %s", tmp_path.c_str(),
                     (p95 + d.save_name).c_str());
            continue;
        }
        std::vector<std::pair<int, std::string>> asg;
        asg.push_back(std::make_pair(worker, d.line));
        std::string werr;
        if (!p95_write_worktodo_add(p95 + "worktodo.add", asg, /*append=*/true, werr)) {
            log_line("ERROR: cannot update worktodo.add: %s", werr.c_str());
            continue;
        }

        log_line("delivered %s -> %s%s [Worker #%d] %s", name.c_str(), p95.c_str(),
                 d.save_name.c_str(), worker, d.line.c_str());
        delivered++;
        slots--;
        ns_in_flight.insert(cm.n);

        if (!cfg.keep_tmp) {
            remove(tmp_path.c_str());
        }
    }

    if (delivered == 0 && cfg.verbose && !tmps.empty()) {
        log_line("idle: %zu local save(s) waiting (in_flight=%zu/%lld, free_workers=%zu)",
                 tmps.size(), in_flight, cap, free_workers.size());
    }
    return delivered;
}

// Write a commented default config when none exists yet.
bool write_default_ini(const std::string &path) {
    std::ofstream o(path, std::ios::trunc);
    if (!o) return false;
    o <<
        "# ecm_p95feeder configuration\n"
        "#\n"
        "# ecm.exe writes stage-1 results locally as e{n:07d}_c{curve:06d}.tmp;\n"
        "# this program moves each finished curve into Prime95 so it can run stage 2.\n"
        "#\n"
        "# Local directory holding the .tmp saves (usually the same dir as ecm.ini).\n"
        "# 本地存放 e{n:07d}_c{curve:06d}.tmp 的目录（通常与 ecm.ini 同目录）。\n"
        "tmp_dir = .\n"
        "\n"
        "# Prime95 working directory (must contain prime.txt and worktodo.txt).\n"
        "# Prime95 工作目录（需含 prime.txt 与 worktodo.txt）。\n"
        "p95_dir =\n"
        "\n"
        "# worktodo file(s) used to check that a save's N and B1 match real work\n"
        "# (CSV; relative names are looked up in tmp_dir, then the exe dir, then the\n"
        "# current dir — the first hit wins and is printed at startup).\n"
        "# Empty = skip the check.\n"
        "# 用于校验存档 N/B1 是否对得上任务的 worktodo 文件（逗号分隔；相对路径依次\n"
        "# 在 tmp_dir、exe 目录、当前目录里找，取第一个命中的，启动时会打印）。\n"
        "# 留空 = 跳过校验。\n"
        "worktodo = worktodo.txt,worktodo.finished.txt\n"
        "\n"
        "# Poll interval in seconds.\n"
        "# 轮询间隔（秒）。\n"
        "poll_seconds = 5\n"
        "\n"
        "# Max handoffs outstanding at once. 0 = read MaxHighMemWorkers from the\n"
        "# Prime95 prime.txt (stage 2 is the high-memory phase).\n"
        "# 同时在飞的最大交接数。0 = 取 Prime95 prime.txt 的 MaxHighMemWorkers。\n"
        "max_in_flight = 0\n"
        "\n"
        "# Restrict deliveries to these worker numbers, e.g. \"1,2\". Empty = any\n"
        "# worker whose [Worker #N] section is currently empty.\n"
        "# 限定分配到这些 Worker 号（如 \"1,2\"）。留空 = 任何当前空闲的 Worker。\n"
        "worker_allow =\n"
        "\n"
        "# Log file (relative to the exe dir). Empty = exe dir / feeder.log.\n"
        "# 日志文件。留空 = exe 目录下的 feeder.log。\n"
        "log_file =\n"
        "\n"
        "# 1 = keep the local .tmp after a successful delivery.\n"
        "# 1 = 转发成功后保留本地 .tmp。\n"
        "keep_tmp = 0\n"
        "\n"
        "verbose = 1\n"
        "\n"
        "# 1 = only report what would be delivered (also available as --dry-run).\n"
        "# 1 = 只报告不落地。\n"
        "dry_run = 0\n";
    o.close();
    return !o.fail();
}

void print_usage(const char *prog) {    std::printf(
        "ecm_p95feeder — move local Edwards stage-1 saves into a Prime95 instance\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --ini <file>     config file (default: feeder.ini next to the exe)\n"
        "  --once           run a single poll cycle and exit (for testing)\n"
        "  --dry-run        report what would be delivered, change nothing\n"
        "  -v               verbose (default on)\n"
        "  -h, --help       this help\n"
        "\n"
        "Config keys (feeder.ini):\n"
        "  tmp_dir          local dir holding e{n:07d}_c{k}.tmp        (default .)\n"
        "  p95_dir          Prime95 working directory                 (required)\n"
        "  worktodo         worktodo file(s) used to validate N/B1    (CSV, relative\n"
        "                   to tmp_dir; default worktodo.txt,worktodo.finished.txt)\n"
        "  poll_seconds     poll interval                             (default 5)\n"
        "  max_in_flight    outstanding handoffs; 0 = MaxHighMemWorkers from\n"
        "                   prime.txt                                (default 0)\n"
        "  worker_allow     restrict to these worker numbers, e.g. 1,2 (default all)\n"
        "  log_file         also append to this file                  (default none)\n"
        "  verbose          1/0                                       (default 1)\n"
        "  dry_run          1/0                                       (default 0)\n"
        "  keep_tmp         1 = keep the local .tmp after delivery     (default 0)\n",
        prog);
}

} // namespace

int main(int argc, char **argv) {
    std::string ini;
    bool once = false;

    // exe 目录 (默认 ini 位置)
    std::string exe_dir = ".";
    {
        const std::string a0 = argv[0] ? argv[0] : "";
        const size_t s1 = a0.find_last_of("\\/");
        if (s1 != std::string::npos) exe_dir = a0.substr(0, s1);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        if (a == "--ini" && i + 1 < argc) { ini = argv[++i]; continue; }
        if (a == "--once") { once = true; continue; }
        if (a == "--dry-run") { continue; }   // ini 里的 dry_run 决定行为
    }

    FeederConfig cfg;
    FeederConfig cli_overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dry-run") cli_overrides.dry_run = 1;
        if (a == "-v") cli_overrides.verbose = 1;
    }

    if (ini.empty()) ini = exe_dir + "/feeder.ini";
    {
        std::string err;
        FeederConfig file_cfg;
        if (load_ini(ini, file_cfg, err)) {
            cfg = file_cfg;
        } else {
            if (write_default_ini(ini)) {
                log_line("no config at %s — wrote a default template; "
                         "set p95_dir and rerun", ini.c_str());
            } else {
                log_line("no config at %s (%s); using defaults", ini.c_str(), err.c_str());
            }
        }
    }
    if (cli_overrides.dry_run) cfg.dry_run = 1;
    cfg.exe_dir = exe_dir;

    // 启动时把实际使用的 worktodo 校验文件打出来 (路径解析有多个回退位置).
    if (cfg.verbose) {
        std::vector<std::string> shown;
        for (const std::string &raw : split_csv(cfg.worktodo)) {
            if (raw.empty()) continue;
            std::vector<std::string> tried;
            const std::string p = resolve_worktodo_path(cfg, raw, &tried);
            if (!p.empty()) {
                shown.push_back(p);
            } else {
                std::string t;
                for (const std::string &c : tried) { if (!t.empty()) t += ", "; t += c; }
                log_line("warn: worktodo '%s' not found (tried: %s)", raw.c_str(), t.c_str());
            }
        }
        for (const std::string &p : shown) log_line("worktodo check file: %s", p.c_str());
        if (shown.empty()) {
            log_line("warn: no worktodo check file — saves will be delivered WITHOUT "
                     "N/B1 validation; set 'worktodo' in feeder.ini");
        }
    }

    if (cfg.log_file.empty()) {
        cfg.log_file = exe_dir + "/feeder.log";
    }
    {
        std::string lp = cfg.log_file;
        if (lp.size() < 2 || (lp[0] != '/' && lp[1] != ':')) lp = with_sep(exe_dir) + lp;
        g_log = std::fopen(lp.c_str(), "a");
    }

    log_line("ecm_p95feeder starting: tmp_dir=%s p95_dir=%s poll=%ds%s",
             cfg.tmp_dir.c_str(), cfg.p95_dir.c_str(), cfg.poll_seconds,
             cfg.dry_run ? " [DRY-RUN]" : "");

    int rc = 0;
    for (;;) {
        rc = run_cycle(cfg);
        if (once) break;
        std::this_thread::sleep_for(std::chrono::seconds(cfg.poll_seconds));
    }
    if (g_log) std::fclose(g_log);
    return rc < 0 ? 1 : 0;
}
