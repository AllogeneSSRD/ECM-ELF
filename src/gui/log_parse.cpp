#include "log_parse.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ecmgui {
namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Reads the number at `p` (digits, optional decimal part) and advances `p` past it.
bool read_double(const char *&p, double &out) {
    const char *start = p;
    while (is_digit(*p)) ++p;
    if (*p == '.') {
        ++p;
        while (is_digit(*p)) ++p;
    }
    if (p == start) return false;
    out = std::strtod(std::string(start, static_cast<std::size_t>(p - start)).c_str(), nullptr);
    return true;
}

bool read_u64(const char *&p, unsigned long long &out) {
    const char *start = p;
    unsigned long long v = 0;
    while (is_digit(*p)) {
        v = v * 10ull + static_cast<unsigned long long>(*p - '0');
        ++p;
    }
    if (p == start) return false;
    out = v;
    return true;
}

const char *find_str(const char *hay, const char *needle) {
    return std::strstr(hay, needle);
}

// "…( 42.3 s/curve )" style: "(~%.2f s/curve)"
bool read_s_per_curve(const std::string &plain, double &out) {
    const std::size_t p = plain.find("s/curve");
    if (p == std::string::npos) return false;
    std::size_t q = plain.rfind('(', p);
    if (q == std::string::npos) return false;
    const char *c = plain.c_str() + q + 1;
    while (*c == ' ' || *c == '~') ++c;
    return read_double(c, out);
}

// "elapsed 51.0s" and "ETA 69.0s" / "remaining 69.0s"
bool read_after(const std::string &plain, const char *key, double &out) {
    const std::size_t p = plain.find(key);
    if (p == std::string::npos) return false;
    const char *c = plain.c_str() + p + std::strlen(key);
    while (*c == ' ' || *c == ':' || *c == '=') ++c;
    return read_double(c, out);
}

bool starts_with(const std::string &s, const char *prefix) {
    const std::size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

bool contains(const std::string &s, const char *needle) {
    return s.find(needle) != std::string::npos;
}

} // namespace

std::string strip_ansi(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0x1B) {
            const unsigned char n = (i + 1 < s.size()) ? static_cast<unsigned char>(s[i + 1]) : 0;
            if (n == '[') {
                // CSI: parameter bytes then a final byte in 0x40..0x7E.
                std::size_t j = i + 2;
                while (j < s.size()) {
                    const unsigned char b = static_cast<unsigned char>(s[j]);
                    ++j;
                    if (b >= 0x40 && b <= 0x7E) break;
                }
                i = j;
                continue;
            }
            if (n == ']') {
                // OSC: terminated by BEL or ST.
                std::size_t j = i + 2;
                while (j < s.size() && !(s[j] == 0x07 ||
                                         (s[j] == 0x1B && j + 1 < s.size() && s[j + 1] == '\\'))) {
                    ++j;
                }
                i = (j < s.size() && s[j] == 0x1B) ? j + 2 : j + 1;
                continue;
            }
            i += 2;                       // two-byte escape
            continue;
        }
        out.push_back(s[i]);
        ++i;
    }
    return out;
}

void split_timestamp(const std::string &s, std::string &timestamp, std::string &plain) {
    timestamp.clear();
    plain = s;
    // "[YYYY-MM-DD HH:MM:SS] "
    if (s.size() < 21 || s[0] != '[') return;
    for (int i = 1; i <= 4; ++i) {
        if (!is_digit(s[static_cast<std::size_t>(i)])) return;
    }
    if (s[5] != '-' || s[8] != '-' || s[11] != ' ' || s[14] != ':' || s[17] != ':' ||
        s[20] != ']') {
        return;
    }
    for (int i : {6, 7, 9, 10, 12, 13, 15, 16, 18, 19}) {
        if (!is_digit(s[static_cast<std::size_t>(i)])) return;
    }
    timestamp = s.substr(1, 19);
    std::size_t p = 21;
    if (p < s.size() && s[p] == ' ') ++p;
    plain = s.substr(p);
}

ParsedLine parse_line(const std::string &line) {
    ParsedLine out;
    out.text = strip_ansi(line);
    split_timestamp(out.text, out.timestamp, out.plain);
    const std::string &p = out.plain;

    // ---- hit ---------------------------------------------------------------
    if (starts_with(p, "factor[")) {
        const std::size_t eq = p.find('=');
        if (eq != std::string::npos) {
            const std::string v = p.substr(eq + 1);
            // "factor[i]=<decimal>" possibly followed by the D3 fields
            // " curve=… sigma=… param=… method=… save=…".
            std::size_t end = 0;
            while (end < v.size() && is_digit(v[end])) ++end;
            if (end > 0) {
                out.factor = v.substr(0, end);
                out.is_hit = true;
                out.hit.factor = out.factor;
                const char *c = p.c_str();
                if (const char *q = find_str(c, " curve=")) {
                    const char *r = q + 7;
                    unsigned long long n = 0;
                    const char *r2 = r;
                    if (read_u64(r2, n)) out.hit.curve = static_cast<int>(n);
                }
                if (const char *q = find_str(c, " sigma=")) {
                    const char *r = q + 7;
                    unsigned long long n = 0;
                    const char *r2 = r;
                    if (read_u64(r2, n)) {
                        out.hit.sigma = n;
                        out.hit.has_sigma = true;
                    }
                }
                if (const char *q = find_str(c, " param=")) {
                    const char *r = q + 7;
                    unsigned long long n = 0;
                    const char *r2 = r;
                    if (read_u64(r2, n)) out.hit.param = static_cast<int>(n);
                }
                if (const char *q = find_str(c, " method=")) {
                    const char *r = q + 8;
                    while (*r != '\0' && *r != ' ') out.hit.method.push_back(*r++);
                }
                if (const char *q = find_str(c, " save=")) {
                    const char *r = q + 6;
                    while (*r != '\0' && *r != ' ' && *r != '\r') out.hit.save.push_back(*r++);
                }
            }
        }
    }
    if (contains(p, "FACTOR FOUND") || contains(p, "-> factor found") ||
        contains(p, "found in Step 1")) {
        out.is_hit = true;
    }

    // ---- queue finished ----------------------------------------------------
    if (contains(p, "queue done")) {
        out.queue_done = true;
        const std::size_t comma = p.find(',');
        if (comma != std::string::npos) {
            const char *c = p.c_str() + comma + 1;
            while (*c == ' ') ++c;
            unsigned long long n = 0;
            if (read_u64(c, n)) out.tasks_processed = static_cast<long long>(n);
        }
    }

    // ---- "this driver does not know --worker" ------------------------------
    // A driver built before D1/D2 pushes the unknown `--worker N` into its positional
    // list, so it never reaches run_queue_manager(): it prints the single-run banner
    // ("mode: cpu-stub, gpucurves=0") and then "No input number on stdin", because the
    // queue-mode task line is not there either. The GUI restarts it, the breaker trips
    // after 3 tries, and without this flag the user only sees "worker keeps restarting"
    // (measured on a release build, 2026-09-28).
    // NOTE: recorded here and turned into an Error at the very end -- the classification
    // chain below ends with `out.kind = LogKind::Raw`, so setting it here would be lost.
    if (contains(p, "No input number on stdin")) out.old_driver = true;

    // ---- progress ----------------------------------------------------------
    const bool gpu_progress = starts_with(p, "GPU: [");
    const bool cpu_progress = starts_with(p, "stage1: [");
    if (gpu_progress || cpu_progress) {
        ProgressInfo pr;
        pr.valid = true;
        pr.gpu = gpu_progress;
        const char *c = p.c_str();
        const char *pct = find_str(c, "%");
        if (pct != nullptr) {
            // walk back to the number before '%'
            const char *q = pct;
            while (q > c && (is_digit(*(q - 1)) || *(q - 1) == '.')) --q;
            double v = 0.0;
            const char *r = q;
            if (read_double(r, v)) pr.pct = v;

            // The counts follow the percentage: "42.3/100" on the CPU form and
            // "123456, +789 bits" on the GPU form (both after two spaces).
            q = pct + 1;
            while (*q == ' ') ++q;
            double first = 0.0;
            const char *r2 = q;
            if (read_double(r2, first)) {
                pr.curves_done = static_cast<unsigned long long>(first);
                if (*r2 == '/') {
                    ++r2;
                    unsigned long long total = 0;
                    const char *r3 = r2;
                    if (read_u64(r3, total)) pr.curves_total = total;
                }
            }
        }
        const char *bits = find_str(c, "bits");
        if (bits != nullptr) {
            const char *q = bits;
            while (q > c && *(q - 1) == ' ') --q;      // "+789 bits": skip the space
            while (q > c && is_digit(*(q - 1))) --q;
            unsigned long long b = 0;
            const char *r = q;
            if (read_u64(r, b)) pr.bits = b;
        }
        read_s_per_curve(p, pr.s_per_curve);
        read_after(p, "elapsed", pr.elapsed_s);
        if (!read_after(p, "ETA", pr.eta_s)) {
            read_after(p, "remaining", pr.eta_s);
        }
        out.progress = pr;
        out.kind = LogKind::Progress;
        return out;
    }

    // ---- events / errors ---------------------------------------------------
    const bool is_error = starts_with(p, "ERROR") || starts_with(p, "# ERROR") ||
                          starts_with(p, "FATAL") || contains(p, "error:") ||
                          contains(p, "warning:");
    if (is_error) {
        out.kind = LogKind::Error;
        return out;
    }
    if (starts_with(p, "START:")) {
        out.kind = LogKind::Event;
        out.start_line = p.substr(6);
        while (!out.start_line.empty() && out.start_line.front() == ' ') {
            out.start_line.erase(out.start_line.begin());
        }
        return out;
    }
    if (out.is_hit || starts_with(p, "Checkpoint") || starts_with(p, "Resuming") ||
        contains(p, "===== ECM queue manager") || starts_with(p, "worker :") ||
        // queue-manager startup banner (ecm_driver.cpp:3140-3166): one line per knob,
        // worth keeping in the log pane when a run is started or restarted.
        starts_with(p, "config :") || starts_with(p, "worktodo :") ||
        starts_with(p, "finished :") || starts_with(p, "log_file :") ||
        starts_with(p, "saves :") || starts_with(p, "method :") ||
        starts_with(p, "gpu backend :") || starts_with(p, "note:") ||
        contains(p, "raise -gpucurves") || contains(p, "needs >= 2 resident")) {
        out.kind = LogKind::Event;
        return out;
    }
    if (out.queue_done) {
        out.kind = LogKind::QueueDone;
        return out;
    }
    out.kind = LogKind::Raw;
    if (out.old_driver) out.kind = LogKind::Error;   // "old driver" must reach the log pane
    return out;
}

// ---------------------------------------------------------------------------
// LineSplitter
// ---------------------------------------------------------------------------

void LineSplitter::push(const char *data, std::size_t len) {
    if (data != nullptr && len > 0) buffer_.append(data, len);
}

bool LineSplitter::next(std::string &line) {
    for (std::size_t i = 0; i < buffer_.size(); ++i) {
        const char c = buffer_[i];
        if (c == '\n' || c == '\r') {
            line = buffer_.substr(0, i);
            std::size_t skip = i + 1;
            // "\r\n" counts as one break
            if (c == '\r' && skip < buffer_.size() && buffer_[skip] == '\n') ++skip;
            buffer_.erase(0, skip);
            return true;
        }
    }
    return false;
}

// Shown by the GUI when `ParsedLine::old_driver` is set: the fix is on the driver side,
// and saying so saves the user from hunting through worker/GPU configuration.
const char *const kOldDriverHint =
    "this worker executable does not understand '--worker' (it prints "
    "\"No input number on stdin\"): it is older than the D1/D2 driver changes -- rebuild "
    "ecm_cuda from the current source, or point [GUI] exe= at a current build";

} // namespace ecmgui
