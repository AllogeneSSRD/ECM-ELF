#include "sync/p95.h"
#include "ecm/platform/log_read.h"
#include "io/files.h"
#include "io/worktodo.h"
#include <memory>

#include "io/prime95.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
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

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b]))
        ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

std::string lower(std::string s) {
    for (char& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string dir_of(const std::string& path) {
    const size_t slash = path.find_last_of("\\/");
    if (slash == std::string::npos)
        return std::string(".");
    if (slash == 0)
        return path.substr(0, 1);
    return path.substr(0, slash);
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty())
        return name;
    const char last = dir[dir.size() - 1];
    if (last == '\\' || last == '/')
        return dir + name;
    return dir + "\\" + name;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Lines of a text file, CR stripped, blanks dropped. A missing file is empty.
std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty())
        return out;
    std::error_code error;
    const bool exists = std::filesystem::exists(std::filesystem::u8path(path), error);
    if (error)
        throw std::runtime_error("cannot inspect P95 responsibility file: " + path + ": " +
                                 error.message());
    if (!exists)
        return out;
    std::ifstream in(std::filesystem::u8path(path));
    if (!in)
        throw std::runtime_error("cannot read P95 responsibility file: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (trim(line).empty())
            continue;
        out.push_back(line);
    }
    if (in.bad())
        throw std::runtime_error("incomplete P95 responsibility file: " + path);
    return out;
}

// Replace a small text file atomically (temp + move): a crash must never leave a
// half-written pending list behind.
bool write_lines_atomic(const std::string& path, const std::vector<std::string>& lines,
                        std::string& err) {
    try {
        std::string bytes;
        for (const auto& line : lines)
            bytes += line + '\n';
        ecm_io::write_atomic(std::filesystem::u8path(path), bytes);
        return true;
    } catch (const std::exception& failure) {
        err = failure.what();
        return false;
    }
}

// ── routing ─────────────────────────────────────────────────────────────────────────

size_t active_for_worker(const std::vector<P95WorkerSection>& sections, int worker) {
    size_t n = 0;
    for (const P95WorkerSection& s : sections) {
        if (s.worker == worker)
            n += p95_count_active(s);
        /* Lines before the first [Worker #N] header belong to worker 1 in Prime95
           (incorporateWorkToDoAddFile starts at tnum 0). */
        else if (worker == 1 && s.worker == 0)
            n += p95_count_active(s);
    }
    return n;
}

bool has_section(const std::vector<P95WorkerSection>& sections, int worker) {
    for (const P95WorkerSection& s : sections)
        if (s.worker == worker)
            return true;
    return false;
}

// 0 = append without a section header.
int choose_worker(const P95TransferConfig& cfg, const std::string& todo_dir,
                  const std::string& add_path, std::string& note) {
    const std::string spec = trim(cfg.add_workers);
    if (spec.empty())
        return 0;

    std::vector<int> candidates;
    if (lower(spec) == "auto") {
        long long workers = 0;
        const std::string prime_txt = join_path(todo_dir, "prime.txt");
        if (!p95_read_prime_int(prime_txt, "NumWorkers", workers) || workers < 1) {
            note = "prime.txt has no usable NumWorkers; appending without a section header";
            return 0;
        }
        if (workers > kMaxP95Workers)
            workers = kMaxP95Workers;
        for (int w = 1; w <= (int)workers; ++w)
            candidates.push_back(w);
    } else {
        int ignored = 0;
        if (!p95_parse_add_workers(spec, candidates, ignored)) {
            note = "p95_add_workers = '" + spec +
                   "' is not a worker list; "
                   "appending without a section header";
            return 0;
        }
        if (ignored > 0) {
            note = "p95_add_workers: ignored " + std::to_string(ignored) + " value(s) outside 1.." +
                   std::to_string(kMaxP95Workers);
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
    for (int w : candidates)
        if (has_section(todo_sections, w))
            valid.push_back(w);
    if (valid.empty()) {
        note = "worktodo.txt has none of the [Worker #" + std::to_string(candidates.front()) +
               "...] sections p95_add_workers asks for; "
               "appending without a section header";
        return 0;
    }
    if (valid.size() == 1)
        return valid.front();

    /* Several workers are valid: hand the line to the least loaded one. Count what
       Prime95 still has to do plus what is waiting in worktodo.add (that file is about to
       be incorporated). Ties go to the lowest worker number. */
    std::vector<P95WorkerSection> add_sections;
    std::string add_err;
    (void)p95_read_worktodo(add_path, add_sections, add_err);

    int best = valid.front();
    size_t best_count =
        active_for_worker(todo_sections, best) + active_for_worker(add_sections, best);
    for (size_t i = 1; i < valid.size(); ++i) {
        const size_t n =
            active_for_worker(todo_sections, valid[i]) + active_for_worker(add_sections, valid[i]);
        if (n < best_count) {
            best = valid[i];
            best_count = n;
        }
    }
    return best;
}

void park_lines(const P95TransferConfig& cfg, const std::vector<std::string>& lines,
                std::string& err) {
    if (cfg.pending_path.empty()) {
        err = "no pending file path is configured";
        return;
    }
    std::vector<std::string> all = read_lines(cfg.pending_path);
    for (const std::string& l : lines)
        if (std::find(all.begin(), all.end(), l) == all.end())
            all.push_back(l);
    write_lines_atomic(cfg.pending_path, all, err);
}

// ── ledger of delivered lines ────────────────────────────────────────────────────────
// One record per line:  "sent <line>"  /  "recovered <line>".
// It exists so the driver can notice that Prime95 threw a delivered line away (rejected AID)
// and hand it over again without the AID, instead of losing the stage-1 work silently.
std::vector<std::string> read_ledger(const std::string& path) { return read_lines(path); }

void append_ledger(const std::string& path, const std::string& record) {
    if (path.empty() || record.empty())
        return;
    const auto file = std::filesystem::u8path(path);
    if (!file.parent_path().empty())
        std::filesystem::create_directories(file.parent_path());
    ecm_io::FileLock lock(path);
    auto records = read_lines(path);
    if (std::find(records.begin(), records.end(), record) != records.end())
        return;
    records.push_back(record);
    std::string error;
    if (!write_lines_atomic(path, records, error))
        throw std::runtime_error(error);
}

// "sent"/"recovered" + the line (the line itself may contain spaces? no, but paths do).
std::string ledger_line(const std::string& record) {
    const std::size_t sp = record.find(' ');
    return (sp == std::string::npos) ? std::string() : trim(record.substr(sp + 1));
}

std::string ledger_kind(const std::string& record) {
    const std::size_t sp = record.find(' ');
    return (sp == std::string::npos) ? record : record.substr(0, sp);
}

// Is `line` (AID-stripped comparison) already in one of Prime95's worktodo lists?
bool line_known_to_p95(const std::vector<P95WorkerSection>& sections, const std::string& line) {
    const std::string want = trim(p95_strip_aid(line));
    for (const P95WorkerSection& s : sections) {
        for (const std::string& l : s.lines) {
            if (trim(p95_strip_aid(l)) == want)
                return true;
        }
    }
    return false;
}

} // namespace

bool p95_parse_add_workers(const std::string& spec, std::vector<int>& workers, int& ignored) {
    return ecm_workers::parse_list(spec, kMaxP95Workers, workers, ignored);
}

std::string p95_transfer_pending_path(const std::string& exe_dir) {
    return join_path(exe_dir.empty() ? std::string(".") : exe_dir, "p95_add_pending.txt");
}

std::string p95_transfer_add_path(const std::string& worktodo_path) {
    if (trim(worktodo_path).empty())
        return std::string();
    return join_path(dir_of(worktodo_path), "worktodo.add");
}

size_t p95_transfer_pending_count(const std::string& pending_path) {
    if (pending_path.empty())
        return 0;
    return read_lines(pending_path).size();
}

std::string p95_transfer_ledger_path(const std::string& exe_dir) {
    return join_path(exe_dir.empty() ? std::string(".") : exe_dir, "p95_add_sent.txt");
}

// ── assignment keys ─────────────────────────────────────────────────────────────────

std::string p95_line_aid(const std::string& line) {
    // ECMSTAGE2=[<aid>,]<k>,<b>,…  /  ECM2=[<aid>,][FFT2=..,]<k>,…: the AID is the first field
    // when it is not integer-like. Our own reader uses the same rule (ecm_worktodo.cpp), so a
    // line the queue accepted is read the same way here.
    std::string s = trim(line);
    std::size_t eq = s.find('=');
    if (eq == std::string::npos)
        return std::string();
    s = s.substr(eq + 1);
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos)
        return std::string();
    const std::string first = trim(s.substr(0, comma));
    if (first.empty())
        return std::string();
    bool integer_like = true;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const char c = first[i];
        if (i == 0 && (c == '+' || c == '-'))
            continue;
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            integer_like = false;
            break;
        }
    }
    if (integer_like)
        return std::string(); // that field is k, not an AID
    if (lower(first) == "n/a")
        return std::string(); // explicit "no AID"
    return first;
}

std::string p95_strip_aid(const std::string& line) {
    const std::string aid = p95_line_aid(line);
    if (aid.empty())
        return line;
    std::string s = trim(line);
    const std::size_t eq = s.find('=');
    if (eq == std::string::npos)
        return line;
    const std::string head = s.substr(0, eq + 1);
    const std::string rest = s.substr(eq + 1);
    const std::size_t comma = rest.find(',');
    if (comma == std::string::npos)
        return line;
    return head + rest.substr(comma + 1);
}

std::vector<std::string> p95_rejected_aids(const std::string& prime_log,
                                           const std::string& results_txt, std::string* err) {
    std::vector<std::string> out;
    bool read_any = false;
    const std::string files[2] = {prime_log, results_txt};
    for (const std::string& path : files) {
        if (path.empty())
            continue;
        const bool read = ecm_visit_log_lines(path, [&](const std::string& line) {
            /* Prime95 writes, on the line after "PrimeNet error 43: Invalid assignment key":
                   ap: no such assignment key, GUID: <guid>, key: <32 hex>
               Only that shape is accepted, so an unrelated "key:" elsewhere cannot poison the
               set and make us drop a perfectly good AID. */
            if (line.find("no such assignment key") == std::string::npos)
                return;
            const std::size_t k = line.rfind("key:");
            if (k == std::string::npos)
                return;
            std::string key = trim(line.substr(k + 4));
            for (char& c : key)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if (key.size() < 8 || key.size() > 64)
                return;
            bool hex = true;
            for (char c : key) {
                if (!std::isxdigit(static_cast<unsigned char>(c))) {
                    hex = false;
                    break;
                }
            }
            if (!hex)
                return;
            if (std::find(out.begin(), out.end(), key) == out.end())
                out.push_back(key);
        });
        read_any = read_any || read;
    }
    if (!read_any && err != nullptr) {
        *err = "no Prime95 log could be read (prime.log / results.txt missing)";
    }
    return out;
}

std::vector<std::string> p95_transfer_recover_lines(const P95TransferConfig& cfg,
                                                    size_t* rejected_count) {
    std::vector<std::string> out;
    const std::vector<std::string> rejected =
        p95_rejected_aids(cfg.prime_log_path, cfg.results_txt_path);
    if (rejected_count != nullptr)
        *rejected_count = rejected.size();
    if (!cfg.recover_lost || rejected.empty() || cfg.sent_ledger_path.empty())
        return out;

    std::vector<std::string> ledger = read_ledger(cfg.sent_ledger_path);
    if (ledger.empty())
        return out;

    /* What Prime95 holds right now: a line that is still in worktodo.txt or waiting in
       worktodo.add is fine and must not be sent twice. */
    std::vector<P95WorkerSection> todo_sections, add_sections;
    std::string err;
    (void)p95_read_worktodo(cfg.worktodo_path, todo_sections, err);
    (void)p95_read_worktodo(p95_transfer_add_path(cfg.worktodo_path), add_sections, err);

    /* A line only needs recovery when Prime95 no longer has it: with a rejected AID Prime95
       deletes the work unit (commonc.c:6567), so it vanished from worktodo.txt without ever
       running stage 2. */
    std::vector<std::string> done; // lines already recovered once
    for (const std::string& rec : ledger) {
        if (ledger_kind(rec) == "recovered") {
            const std::string l = ledger_line(rec);
            if (!l.empty())
                done.push_back(trim(p95_strip_aid(l)));
        }
    }

    for (const std::string& rec : ledger) {
        if (ledger_kind(rec) != "sent")
            continue;
        const std::string line = trim(ledger_line(rec));
        if (line.empty())
            continue;
        const std::string aid = p95_line_aid(line);
        if (aid.empty())
            continue; // no AID: Prime95 cannot reject it
        if (std::find(rejected.begin(), rejected.end(), aid) == rejected.end())
            continue;
        const std::string stripped = trim(p95_strip_aid(line));
        if (std::find(done.begin(), done.end(), stripped) != done.end())
            continue;
        if (line_known_to_p95(todo_sections, line) || line_known_to_p95(add_sections, line)) {
            continue; // still queued: nothing to recover
        }
        if (std::find(out.begin(), out.end(), stripped) == out.end())
            out.push_back(stripped);
    }
    return out;
}

P95TransferResult p95_transfer_deliver(const P95TransferConfig& cfg, const std::string& line) {
    P95TransferResult r;
    bool published = false;
    if (trim(cfg.worktodo_path).empty()) {
        r.enabled = false;
        r.ok = true;
        r.note = "p95 handoff is disabled (p95_worktodo_path is empty)";
        return r;
    }
    r.enabled = true;
    try {
        // Serializes the complete pending snapshot/ack with other delivery attempts.
        // The add writer has its own shared gate inside io/prime95 (also used by feeder).
        std::unique_ptr<ecm_io::FileLock> pending;
        if (!cfg.pending_path.empty()) {
            const auto parent = std::filesystem::u8path(cfg.pending_path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
            pending = std::make_unique<ecm_io::FileLock>(cfg.pending_path,
                                                         ecm_io::FileLockDomain::Publication, 0);
        }

        std::string task = line;
        if (!task.empty() && task.back() == '\r')
            task.pop_back();
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

        /* Work Prime95 already dropped (rejected AID) is re-queued WITHOUT its AID: the stage-1
           result is on disk, and Prime95 registers a fresh assignment when there is no key. */
        std::vector<std::string> recovered = p95_transfer_recover_lines(cfg);
        r.recovered = recovered.size();
        for (const std::string& l : recovered)
            queued.push_back(l);

        /* An AID PrimeNet has already rejected must NOT be attached: Prime95 would report it and
           delete the task (commonc.c:6550-6569). Everything else keeps its AID. */
        const std::vector<std::string> rejected =
            p95_rejected_aids(cfg.prime_log_path, cfg.results_txt_path);
        const std::string task_aid = p95_line_aid(task);
        if (!task_aid.empty() &&
            std::find(rejected.begin(), rejected.end(), task_aid) != rejected.end()) {
            r.aid_dropped = true;
            task = p95_strip_aid(task);
        } else if (!cfg.keep_aid && !task_aid.empty()) {
            r.aid_dropped = true; // ini p95_keep_aid = 0
            task = p95_strip_aid(task);
        }
        if (std::find(queued.begin(), queued.end(), task) == queued.end())
            queued.push_back(task);

        std::string note;
        r.worker = choose_worker(cfg, todo_dir, add_path, note);
        r.note = note;
        if (r.aid_dropped) {
            const std::string why =
                (!task_aid.empty() &&
                 std::find(rejected.begin(), rejected.end(), task_aid) != rejected.end())
                    ? std::string("PrimeNet rejected AID ") + task_aid
                    : std::string("p95_keep_aid = 0");
            const std::string extra =
                why + ": delivered without the AID so Prime95 registers the work itself "
                      "(a rejected key makes Prime95 delete the task)";
            r.note = r.note.empty() ? extra : (r.note + "; " + extra);
        }
        if (r.recovered > 0) {
            const std::string extra =
                std::to_string(r.recovered) +
                " line(s) Prime95 had dropped (rejected AID) were re-delivered "
                "without their AID";
            r.note = r.note.empty() ? extra : (r.note + "; " + extra);
        }

        std::string err;
        // Keep the exact set of attempted lines until all acknowledgements are
        // durable. A failed ledger write must not lose recovered assignments.
        if (!cfg.pending_path.empty() && !write_lines_atomic(cfg.pending_path, queued, err))
            throw std::runtime_error(err);
        std::vector<std::pair<int, std::string>> assignments;
        assignments.reserve(queued.size());
        for (const std::string& l : queued)
            assignments.push_back(std::make_pair(r.worker, l));

        /* Our own temp suffix: the feeder may be writing the same directory at the same
           moment, and two writers must not share one temp file. */
        const bool wrote =
            p95_write_worktodo_add(add_path, assignments, true, err, ".ecm.tmp", &published);
        if (wrote) {
            published = true;
            /* Ledger: what went out (so a line Prime95 later throws away can be recovered) and
               which lines were recoveries themselves (so they are not recovered twice). */
            if (!cfg.sent_ledger_path.empty()) {
                for (const std::string& l : recovered)
                    append_ledger(cfg.sent_ledger_path, "recovered " + l);
                if (!r.aid_dropped)
                    append_ledger(cfg.sent_ledger_path, "sent " + task);
                else
                    append_ledger(cfg.sent_ledger_path,
                                  "sent " + task); // AID-free: nothing to recover
            }
        }

        if (!wrote) {
            std::string park_err;
            if (!published)
                park_lines(cfg, queued, park_err);
            r.ok = false;
            r.publication_uncertain = published;
            r.error = "cannot append to " + add_path + ": " + err;
            if (!park_err.empty())
                r.error += "; parking failed too: " + park_err;
            r.pending_left = p95_transfer_pending_count(cfg.pending_path);
            return r;
        }

        if (!queued.empty() && !cfg.pending_path.empty())
            remove(cfg.pending_path.c_str());
        r.ok = true;
        r.added = queued.size();
        r.pending_left = 0;
        return r;
    } catch (const std::exception& failure) {
        r.ok = false;
        r.publication_uncertain = published;
        r.error = failure.what();
        return r;
    }
}

std::string p95_transfer_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"') {
            out += '\\';
            out += c;
        } else if (c == '\r' || c == '\n')
            out += ' ';
        else
            out += c;
    }
    if (!out.empty() && out.back() == '\\')
        out += '\\';
    return out;
}

// Durable handoff intent, file publication and restart reconciliation.
#include "ecm/io/save.h"
#include "ecm/platform/logging.h"
#include "io/files.h"
#include "io/prime95.h"
#include "io/worktodo.h"
#include "p95.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ecm_sync {
namespace {
uint64_t checksum(const std::string& bytes, size_t prefix_size) {
    uint64_t value = 14695981039346656037ull;
    for (size_t i = 0; i < prefix_size; ++i) {
        value ^= static_cast<unsigned char>(bytes[i]);
        value *= 1099511628211ull;
    }
    return value;
}

std::filesystem::path pending_dir(const P95TransferConfig& cfg) {
    return std::filesystem::u8path(cfg.pending_path + ".handoff");
}

struct Intent {
    std::string source;
    std::string target;
    std::string line;
    std::string source_assignment;
    uint64_t size = 0;
    uint64_t sum = 0;
    uint64_t range_hash = 0;
    P95TransferConfig delivery;
};

// Field order is the on-disk version 3 protocol. Keep the frozen destination and
// routing policy when replaying an intent under a newer worker configuration.
std::string encode(const Intent& intent) {
    std::ostringstream out;
    out << "ECM-P95-HANDOFF 3\n"
        << intent.size << ' ' << intent.sum << ' ' << intent.range_hash << '\n'
        << std::quoted(intent.source) << '\n'
        << std::quoted(intent.target) << '\n'
        << std::quoted(intent.line) << '\n'
        << std::quoted(intent.source_assignment) << '\n'
        << std::quoted(intent.delivery.worktodo_path) << '\n'
        << std::quoted(intent.delivery.add_workers) << '\n'
        << std::quoted(intent.delivery.prime_log_path) << '\n'
        << std::quoted(intent.delivery.results_txt_path) << '\n'
        << std::quoted(intent.delivery.sent_ledger_path) << '\n'
        << intent.delivery.keep_aid << ' ' << intent.delivery.recover_lost << '\n';
    return out.str();
}

Intent decode(const std::string& bytes) {
    std::istringstream in(bytes);
    std::string magic;
    std::getline(in, magic);
    Intent intent;
    if (magic != "ECM-P95-HANDOFF 3" ||
        !(in >> intent.size >> intent.sum >> intent.range_hash >> std::quoted(intent.source) >>
          std::quoted(intent.target) >> std::quoted(intent.line) >>
          std::quoted(intent.source_assignment) >> std::quoted(intent.delivery.worktodo_path) >>
          std::quoted(intent.delivery.add_workers) >> std::quoted(intent.delivery.prime_log_path) >>
          std::quoted(intent.delivery.results_txt_path) >>
          std::quoted(intent.delivery.sent_ledger_path) >> intent.delivery.keep_aid >>
          intent.delivery.recover_lost))
        throw std::runtime_error("invalid P95 handoff intent");
    return intent;
}

bool contains_line(const std::string& file, const std::string& line,
                   const std::string& prefix = {}) {
    if (file.empty() || !std::filesystem::exists(std::filesystem::u8path(file)))
        return false;
    std::istringstream in(ecm_io::read_file(std::filesystem::u8path(file)));
    std::string actual;
    while (std::getline(in, actual)) {
        if (!actual.empty() && actual.back() == '\r')
            actual.pop_back();
        if (actual == prefix + line)
            return true;
    }
    return false;
}

void validate_ready(const std::string& line) {
    EcmStage2Task task;
    std::string error;
    if (!ecm_parse_stage2_line(line, task, error))
        throw std::runtime_error("pending P95 task is not a saved Stage2 task: " + error);
    if (!std::filesystem::u8path(task.save_name).is_absolute())
        throw std::runtime_error("pending P95 save must use an absolute path: " + task.save_name);
    const auto count = ecm_save::record_count(task.save_name);
    if (task.skip_curves > count || task.curves_to_run > count - task.skip_curves)
        throw std::runtime_error("pending P95 save does not contain its referenced curve range: " +
                                 task.save_name);
    const auto range =
        ecm_save::inspect_range(task.save_name, task.skip_curves, task.curves_to_run);
    mpz_t modulus;
    mpz_init(modulus);
    try {
        if (!ecm_compute_stage2_n(task, modulus, error))
            throw std::runtime_error(error);
        ecm_save::validate_p95_batch(range, modulus);
    } catch (...) {
        mpz_clear(modulus);
        throw;
    }
    mpz_clear(modulus);
}
} // namespace

bool park_p95(const P95TransferConfig& cfg, const std::string& line, const ecm_save::Range& range,
              std::string& error) {
    if (cfg.worktodo_path.empty())
        return true;
    try {
        if (cfg.sync_dir.empty())
            throw std::runtime_error("p95_sync_dir must name the save destination");
        const auto bytes = ecm_save::verified_snapshot(range);
        // Validate the local saved-point assignment before taking durable
        // delivery responsibility or exposing any external work line.
        validate_ready(line);

        Intent intent;
        intent.source = range.path;
        intent.delivery = cfg;
        intent.source_assignment = line;
        std::string identities;
        for (const auto& record : range.records)
            identities += record + '\n';
        intent.range_hash = checksum(identities, identities.size());
        intent.target = std::filesystem::absolute(std::filesystem::u8path(cfg.sync_dir) /
                                                  std::filesystem::u8path(range.path).filename())
                            .lexically_normal()
                            .u8string();
        std::string rewritten;
        if (!ecm_rewrite_stage2_save(line, intent.target, rewritten, error))
            return false;
        intent.line = rewritten;
        intent.size = bytes.size();
        intent.sum = checksum(bytes, bytes.size());
        const auto encoded = encode(intent);

        const auto folder = pending_dir(cfg);
        std::filesystem::create_directories(folder);
        ecm_io::FileLock lock(cfg.pending_path + ".handoff");
        // Identity excludes later source suffixes and delivery configuration:
        // retrying this batch must reuse its first persisted destination.
        const auto logical = intent.source + '\n' + intent.source_assignment + '\n' +
                             std::to_string(intent.range_hash);
        const auto path = folder / (std::to_string(checksum(logical, logical.size())) + ".intent");
        auto done = path;
        done += ".done";
        for (const auto& existing : {done, path}) {
            if (!std::filesystem::exists(existing))
                continue;

            const auto previous = decode(ecm_io::read_file(existing));
            if (previous.source != intent.source ||
                previous.source_assignment != intent.source_assignment ||
                previous.range_hash != intent.range_hash || bytes.size() < previous.size ||
                checksum(bytes, size_t(previous.size)) != previous.sum)
                throw std::runtime_error("P95 receipt/source identity conflict");
            return true; // Keep the original destination and responsibility frozen.
        }

        ecm_io::write_atomic(path, encoded);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool flush_p95(const P95TransferConfig& cfg, std::string& error) {
    if (cfg.worktodo_path.empty())
        return true;
    try {
        // Import old ready-line pending files without dropping them. Native ECM
        // lines cannot prove a saved range and require operator correction.
        if (std::filesystem::exists(std::filesystem::u8path(cfg.pending_path))) {
            const auto legacy = ecm_io::read_file(std::filesystem::u8path(cfg.pending_path));
            std::istringstream in(legacy);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty())
                    continue;

                validate_ready(line);
                EcmStage2Task task;
                std::string ignored;
                (void)ecm_parse_stage2_line(line, task, ignored);
                const auto range =
                    ecm_save::inspect_range(task.save_name, task.skip_curves, task.curves_to_run);
                if (!park_p95(cfg, line, range, error))
                    throw std::runtime_error(error);
            }
            if (!legacy.empty())
                ecm_io::write_atomic(std::filesystem::u8path(cfg.pending_path), "");
        }
        const auto folder = pending_dir(cfg);
        if (!std::filesystem::exists(folder))
            return true;

        ecm_io::FileLock delivery(cfg.pending_path + ".handoff");
        std::vector<std::filesystem::path> intents;
        for (const auto& entry : std::filesystem::directory_iterator(folder))
            if (entry.path().extension() == ".intent")
                intents.push_back(entry.path());
        std::sort(intents.begin(), intents.end());

        for (const auto& path : intents) {
            auto done = path;
            done += ".done";
            if (std::filesystem::exists(done)) {
                std::filesystem::remove(path);
                continue;
            }
            const auto encoded = ecm_io::read_file(path);
            const auto intent = decode(encoded);
            const auto source = std::filesystem::u8path(intent.source),
                       target = std::filesystem::u8path(intent.target);
            std::string bytes;
            {
                ecm_io::FileLock lock(intent.source);
                bytes = ecm_io::read_file(source);
            }
            // A source may grow after Stage1 completes; its recorded prefix must
            // remain unchanged. Release its lock before locking the destination.
            if (bytes.size() < intent.size || checksum(bytes, size_t(intent.size)) != intent.sum)
                throw std::runtime_error("P95 source save changed before delivery: " +
                                         intent.source);
            std::filesystem::create_directories(target.parent_path());
            {
                ecm_io::FileLock lock(intent.target);
                const auto existing =
                    std::filesystem::exists(target) ? ecm_io::read_file(target) : std::string{};
                const auto common = (std::min)(existing.size(), bytes.size());
                if (existing.compare(0, common, bytes, 0, common) != 0)
                    throw std::runtime_error(
                        "P95 destination save prefix conflicts; refusing to replace history: " +
                        intent.target);
                if (existing.size() < bytes.size())
                    ecm_io::write_atomic(target, bytes);
            }
            validate_ready(intent.line);

            // Legacy pending/recovery must also name already usable saves. Never
            // release a new add file while an older line references missing data.
            if (std::filesystem::exists(std::filesystem::u8path(cfg.pending_path))) {
                std::istringstream in(ecm_io::read_file(std::filesystem::u8path(cfg.pending_path)));
                std::string line;
                while (std::getline(in, line))
                    if (!line.empty())
                        validate_ready(line);
            }
            for (const auto& line : p95_transfer_recover_lines(intent.delivery))
                validate_ready(line);
            auto inflight = path;
            inflight += ".sending";
            const auto add = p95_transfer_add_path(intent.delivery.worktodo_path);
            const auto ready = path.u8string() + ".ready";
            for (const auto& line : read_lines(ready))
                validate_ready(line);
            if (std::filesystem::exists(inflight)) {
                // A previous process may have sent the task before crashing.
                // Without evidence of delivery, stop instead of sending twice.
                const bool known =
                    contains_line(add, intent.line) ||
                    contains_line(add, p95_strip_aid(intent.line)) ||
                    contains_line(intent.delivery.worktodo_path, intent.line) ||
                    contains_line(intent.delivery.worktodo_path, p95_strip_aid(intent.line)) ||
                    contains_line(intent.delivery.sent_ledger_path, intent.line, "sent ") ||
                    contains_line(intent.delivery.sent_ledger_path, p95_strip_aid(intent.line),
                                  "sent ");
                if (!known)
                    throw std::runtime_error("P95 delivery acknowledgement is ambiguous; retained "
                                             "intent requires inspection: " +
                                             path.u8string());
                ecm_io::FileLock pending_ack(ready);
                auto attempted = read_lines(ready);
                if (attempted.empty() && std::filesystem::exists(std::filesystem::u8path(ready)))
                    throw std::runtime_error("empty P95 attempted-delivery record: " + ready);
                if (attempted.empty())
                    attempted.push_back(intent.line); // Old sending records.
                for (const auto& line : attempted) {
                    const auto stripped = p95_strip_aid(line);
                    const auto observed = [&](const std::string& actual) {
                        return contains_line(add, actual) ||
                               contains_line(intent.delivery.worktodo_path, actual) ||
                               contains_line(intent.delivery.sent_ledger_path, actual, "sent ") ||
                               contains_line(intent.delivery.sent_ledger_path, actual,
                                             "recovered ");
                    };
                    const auto actual = observed(line) ? line : stripped;
                    if (!observed(actual))
                        throw std::runtime_error(
                            "P95 pending delivery acknowledgement is ambiguous: " + ready);
                    append_ledger(intent.delivery.sent_ledger_path, "sent " + actual);
                    if (p95_line_aid(actual).empty())
                        append_ledger(intent.delivery.sent_ledger_path, "recovered " + actual);
                }
            } else {
                ecm_io::write_atomic(inflight, encoded);
                // Each intent owns its low-level retry file; another batch must
                // never flush an assignment whose save it has not validated.
                auto delivery_config = intent.delivery;
                delivery_config.pending_path = path.u8string() + ".ready";
                const auto result = p95_transfer_deliver(delivery_config, intent.line);
                if (!result.ok) {
                    if (!result.publication_uncertain)
                        std::filesystem::remove(inflight);
                    throw std::runtime_error(result.error);
                }
                ecm_ts_fprintf(stdout,
                               "p95_add: %s worker=%d added=%zu pending_delivered=%zu "
                               "recovered=%zu aid=%s file=\"%s\" note=\"%s\"\n",
                               result.note.empty() ? "ok" : "warn", result.worker, result.added,
                               result.pending_delivered, result.recovered,
                               result.aid_dropped ? "dropped" : "kept", result.add_path.c_str(),
                               p95_transfer_quote(result.note).c_str());
            }
            // Retain the durable acknowledgement for deduplication before
            // removing responsibility and the ambiguous-send marker.
            ecm_io::write_atomic(done, encoded);
            std::filesystem::remove(path);
            std::filesystem::remove(inflight);
            std::filesystem::remove(std::filesystem::u8path(ready));
        }
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

size_t pending_p95_count(const P95TransferConfig& cfg) {
    size_t count = p95_transfer_pending_count(cfg.pending_path);
    const auto folder = pending_dir(cfg);
    if (!std::filesystem::exists(folder))
        return count;
    for (const auto& entry : std::filesystem::directory_iterator(folder))
        if (entry.path().extension() == ".intent")
            ++count;
    return count;
}
} // namespace ecm_sync
