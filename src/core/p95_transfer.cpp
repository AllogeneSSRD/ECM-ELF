#include "p95_transfer.h"

#include "p95_worktodo.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

namespace {

// Prime95's own ceiling (MAX_NUM_WORKERS in commonc.h).
const int kMaxP95Workers = 1024;

// worktodo.add lock: wait up to this long for another writer, then park the line.
const int kLockWaitMs = 3000;
const int kLockRetryMs = 100;
// A lock file older than this was left behind by a killed process: preempt it.
const long long kLockStaleMs = 60000;

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::string lower(std::string s) {
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string dir_of(const std::string &path) {
    const size_t slash = path.find_last_of("\\/");
    if (slash == std::string::npos) return std::string(".");
    if (slash == 0) return path.substr(0, 1);
    return path.substr(0, slash);
}

std::string join_path(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    const char last = dir[dir.size() - 1];
    if (last == '\\' || last == '/') return dir + name;
    return dir + "\\" + name;
}

void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    usleep((useconds_t)ms * 1000);
#endif
}

bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Lines of a text file, CR stripped, blanks dropped. A missing file is empty.
std::vector<std::string> read_lines(const std::string &path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        out.push_back(line);
    }
    return out;
}

// Replace a small text file atomically (temp + move): a crash must never leave a
// half-written pending list behind.
bool write_lines_atomic(const std::string &path, const std::vector<std::string> &lines,
                        std::string &err) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        if (!out) { err = "cannot write " + tmp; return false; }
        for (const std::string &l : lines) out << l << "\n";
        out.close();
        if (out.fail()) { err = "write failed: " + tmp; remove(tmp.c_str()); return false; }
    }
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        err = "cannot replace " + path;
        remove(tmp.c_str());
        return false;
    }
#else
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        err = "cannot replace " + path;
        remove(tmp.c_str());
        return false;
    }
#endif
    return true;
}

// ── worktodo.add lock ────────────────────────────────────────────────────────────────
// Our own writers are the only ones taking this lock; Prime95 never does. It exists so
// two workers of this driver cannot interleave a read-modify-write of worktodo.add.

bool lock_create(const std::string &path, std::string &err) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        err = (e == ERROR_FILE_EXISTS || e == ERROR_ALREADY_EXISTS)
                  ? "held by another writer"
                  : "cannot create the lock file";
        return false;
    }
    DWORD pid = GetCurrentProcessId();
    DWORD written = 0;
    (void)WriteFile(h, &pid, (DWORD)sizeof(pid), &written, nullptr);
    CloseHandle(h);
    return true;
#else
    const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) {
        err = (errno == EEXIST) ? "held by another writer" : "cannot create the lock file";
        return false;
    }
    const long pid = (long)getpid();
    if (write(fd, &pid, sizeof(pid)) < 0) { /* the pid is only informational */ }
    close(fd);
    return true;
#endif
}

bool lock_age_ms(const std::string &path, long long &age_ms) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fad)) return false;
    ULARGE_INTEGER t, n;
    t.LowPart = fad.ftLastWriteTime.dwLowDateTime;
    t.HighPart = fad.ftLastWriteTime.dwHighDateTime;
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    n.LowPart = now.dwLowDateTime;
    n.HighPart = now.dwHighDateTime;
    age_ms = (n.QuadPart > t.QuadPart) ? (long long)((n.QuadPart - t.QuadPart) / 10000ull) : 0;
    return true;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    const long long now_ms = (long long)time(nullptr) * 1000;
    const long long mt_ms = (long long)st.st_mtime * 1000;
    age_ms = (now_ms > mt_ms) ? (now_ms - mt_ms) : 0;
    return true;
#endif
}

bool lock_acquire(const std::string &path, std::string &err) {
    int waited = 0;
    for (;;) {
        std::string why;
        if (lock_create(path, why)) return true;

        long long age = 0;
        if (lock_age_ms(path, age) && age > kLockStaleMs) {
            /* Left behind by a process that died mid-write. */
            remove(path.c_str());
            continue;
        }
        if (waited >= kLockWaitMs) {
            err = why + " (waited " + std::to_string(kLockWaitMs) + " ms)";
            return false;
        }
        sleep_ms(kLockRetryMs);
        waited += kLockRetryMs;
    }
}

void lock_release(const std::string &path) { remove(path.c_str()); }

// ── routing ─────────────────────────────────────────────────────────────────────────

size_t active_for_worker(const std::vector<P95WorkerSection> &sections, int worker) {
    size_t n = 0;
    for (const P95WorkerSection &s : sections) {
        if (s.worker == worker) n += p95_count_active(s);
        /* Lines before the first [Worker #N] header belong to worker 1 in Prime95
           (incorporateWorkToDoAddFile starts at tnum 0). */
        else if (worker == 1 && s.worker == 0) n += p95_count_active(s);
    }
    return n;
}

bool has_section(const std::vector<P95WorkerSection> &sections, int worker) {
    for (const P95WorkerSection &s : sections) if (s.worker == worker) return true;
    return false;
}

// 0 = append without a section header.
int choose_worker(const P95TransferConfig &cfg, const std::string &todo_dir,
                  const std::string &add_path, std::string &note) {
    const std::string spec = trim(cfg.add_workers);
    if (spec.empty()) return 0;

    std::vector<int> candidates;
    if (lower(spec) == "auto") {
        long long workers = 0;
        const std::string prime_txt = join_path(todo_dir, "prime.txt");
        if (!p95_read_prime_int(prime_txt, "NumWorkers", workers) || workers < 1) {
            note = "prime.txt has no usable NumWorkers; appending without a section header";
            return 0;
        }
        if (workers > kMaxP95Workers) workers = kMaxP95Workers;
        for (int w = 1; w <= (int)workers; ++w) candidates.push_back(w);
    } else {
        int ignored = 0;
        if (!p95_parse_add_workers(spec, candidates, ignored)) {
            note = "p95_add_workers = '" + spec + "' is not a worker list; "
                   "appending without a section header";
            return 0;
        }
        if (ignored > 0) {
            note = "p95_add_workers: ignored " + std::to_string(ignored) +
                   " value(s) outside 1.." + std::to_string(kMaxP95Workers);
        }
    }
    if (candidates.empty()) {
        note = "p95_add_workers lists no usable worker; appending without a section header";
        return 0;
    }

    /* A [Worker #N] section that Prime95's worktodo.txt does not have would leave the
       line in limbo (Prime95 routes an unknown header to another thread or drops it), so
       the header has to exist there -- worktodo.add itself is not evidence. */
    std::vector<P95WorkerSection> todo_sections;
    std::string err;
    if (!p95_read_worktodo(cfg.worktodo_path, todo_sections, err)) {
        note = "cannot read " + cfg.worktodo_path + " (" + err +
               "); appending without a section header";
        return 0;
    }
    std::vector<int> valid;
    for (int w : candidates) if (has_section(todo_sections, w)) valid.push_back(w);
    if (valid.empty()) {
        note = "worktodo.txt has none of the [Worker #" +
               std::to_string(candidates.front()) + "...] sections p95_add_workers asks for; "
               "appending without a section header";
        return 0;
    }
    if (valid.size() == 1) return valid.front();

    /* Several workers are valid: hand the line to the least loaded one. Count what
       Prime95 still has to do plus what is waiting in worktodo.add (that file is about to
       be incorporated). Ties go to the lowest worker number. */
    std::vector<P95WorkerSection> add_sections;
    std::string add_err;
    (void)p95_read_worktodo(add_path, add_sections, add_err);

    int best = valid.front();
    size_t best_count = active_for_worker(todo_sections, best) + active_for_worker(add_sections, best);
    for (size_t i = 1; i < valid.size(); ++i) {
        const size_t n = active_for_worker(todo_sections, valid[i]) +
                         active_for_worker(add_sections, valid[i]);
        if (n < best_count) { best = valid[i]; best_count = n; }
    }
    return best;
}

void park_lines(const P95TransferConfig &cfg, const std::vector<std::string> &lines,
                std::string &err) {
    if (cfg.pending_path.empty()) {
        err = "no pending file path is configured";
        return;
    }
    std::vector<std::string> all = read_lines(cfg.pending_path);
    for (const std::string &l : lines) all.push_back(l);
    write_lines_atomic(cfg.pending_path, all, err);
}

} // namespace

bool p95_parse_add_workers(const std::string &spec, std::vector<int> &workers, int &ignored) {
    workers.clear();
    ignored = 0;
    const std::string s = trim(spec);
    if (s.empty()) return true;

    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const std::string t = trim(item);
        if (t.empty()) continue;
        const size_t dash = t.find('-');
        long lo = 0, hi = 0;
        if (dash == std::string::npos) {
            char *end = nullptr;
            lo = std::strtol(t.c_str(), &end, 10);
            if (end == t.c_str() || *end != '\0') return false;
            hi = lo;
        } else {
            const std::string a = trim(t.substr(0, dash));
            const std::string b = trim(t.substr(dash + 1));
            char *ea = nullptr, *eb = nullptr;
            lo = std::strtol(a.c_str(), &ea, 10);
            if (ea == a.c_str() || *ea != '\0') return false;
            hi = std::strtol(b.c_str(), &eb, 10);
            if (eb == b.c_str() || *eb != '\0') return false;
            if (hi < lo) std::swap(lo, hi);
        }
        for (long w = lo; w <= hi; ++w) {
            if (w < 1 || w > kMaxP95Workers) { ++ignored; continue; }
            if (std::find(workers.begin(), workers.end(), (int)w) == workers.end()) {
                workers.push_back((int)w);
            }
        }
    }
    return true;
}

std::string p95_transfer_pending_path(const std::string &exe_dir) {
    return join_path(exe_dir.empty() ? std::string(".") : exe_dir, "p95_add_pending.txt");
}

std::string p95_transfer_add_path(const std::string &worktodo_path) {
    if (trim(worktodo_path).empty()) return std::string();
    return join_path(dir_of(worktodo_path), "worktodo.add");
}

size_t p95_transfer_pending_count(const std::string &pending_path) {
    if (pending_path.empty()) return 0;
    return read_lines(pending_path).size();
}

P95TransferResult p95_transfer_deliver(const P95TransferConfig &cfg, const std::string &line) {
    P95TransferResult r;
    if (trim(cfg.worktodo_path).empty()) {
        r.enabled = false;
        r.ok = true;
        r.note = "p95 handoff is disabled (p95_worktodo_path is empty)";
        return r;
    }
    r.enabled = true;

    std::string task = line;
    if (!task.empty() && task.back() == '\r') task.pop_back();
    if (trim(task).empty()) {
        r.ok = true;
        r.note = "empty line, nothing to hand over";
        return r;
    }

    const std::string todo_dir = dir_of(cfg.worktodo_path);
    const std::string add_path = join_path(todo_dir, "worktodo.add");
    r.add_path = add_path;

    /* Lines parked by an earlier failure go out first, in order: a task whose handoff
       failed must not be forgotten just because its queue line is already finished. */
    std::vector<std::string> queued = read_lines(cfg.pending_path);
    r.pending_delivered = queued.size();
    queued.push_back(task);

    std::string note;
    r.worker = choose_worker(cfg, todo_dir, add_path, note);
    r.note = note;

    const std::string lock = add_path + ".lock";
    std::string err;
    if (!lock_acquire(lock, err)) {
        std::string park_err;
        park_lines(cfg, queued, park_err);
        r.ok = false;
        r.error = "worktodo.add is locked: " + err;
        if (!park_err.empty()) r.error += "; parking failed too: " + park_err;
        r.pending_left = p95_transfer_pending_count(cfg.pending_path);
        return r;
    }

    std::vector<std::pair<int, std::string>> assignments;
    assignments.reserve(queued.size());
    for (const std::string &l : queued) assignments.push_back(std::make_pair(r.worker, l));

    /* Our own temp suffix: the feeder may be writing the same directory at the same
       moment, and two writers must not share one temp file. */
    const bool wrote = p95_write_worktodo_add(add_path, assignments, true, err, ".ecm.tmp");
    lock_release(lock);

    if (!wrote) {
        std::string park_err;
        park_lines(cfg, queued, park_err);
        r.ok = false;
        r.error = "cannot append to " + add_path + ": " + err;
        if (!park_err.empty()) r.error += "; parking failed too: " + park_err;
        r.pending_left = p95_transfer_pending_count(cfg.pending_path);
        return r;
    }

    if (!queued.empty() && !cfg.pending_path.empty()) remove(cfg.pending_path.c_str());
    r.ok = true;
    r.added = queued.size();
    r.pending_left = 0;
    return r;
}
