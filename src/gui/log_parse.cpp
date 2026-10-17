#include "log_parse.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

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

// ---- `p95_add:` notice fields -------------------------------------------------------
// "… worker=3 added=1 file=\"D:\\p95 dir\\worktodo.add\" note=\"free text\"" -- every
// value except a number is quoted, so a path with spaces or a free-text reason cannot be
// mistaken for another key (the driver escapes `"` and `\` inside the value).
bool kv_pos(const std::string &s, const char *key, std::size_t &val_start) {
    const std::string pat = std::string(" ") + key + "=";
    const std::size_t pos = s.find(pat);
    if (pos == std::string::npos) return false;
    val_start = pos + pat.size();
    return true;
}

bool read_kv_int(const std::string &s, const char *key, long long &out) {
    std::size_t v = 0;
    if (!kv_pos(s, key, v)) return false;
    const char *p = s.c_str() + v;
    if (*p == '"') return false;             // quoted -> it is a string, not a number
    char *end = nullptr;
    const long long n = std::strtoll(p, &end, 10);
    if (end == p) return false;
    out = n;
    return true;
}

bool read_kv_str(const std::string &s, const char *key, std::string &out) {
    std::size_t v = 0;
    if (!kv_pos(s, key, v)) return false;
    if (v >= s.size() || s[v] != '"') return false;
    std::string text;
    for (std::size_t i = v + 1; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '\\' && i + 1 < s.size() && (s[i + 1] == '"' || s[i + 1] == '\\')) {
            // \" -> a quote in the text, \\ -> one backslash (see p95_quote in
            // ecm_driver.cpp). Any other \x stays as it is, so a Windows path with single
            // backslashes needs no doubling and reads normally in the log pane.
            text += s[i + 1];
            ++i;
            continue;
        }
        if (c == '"') { out = text; return true; }
        text += c;
    }
    return false;                            // unterminated: keep the caller's default
}

// One `p95_add:` line -> a notice. Returns false when the line is not one.
bool parse_p95_notice(const std::string &p, P95Notice &out) {
    if (!starts_with(p, "p95_add:")) return false;
    const std::string rest = p.substr(8);

    out.valid = true;
    if (starts_with(rest, " ok")) out.level = P95Notice::Level::Ok;
    else if (starts_with(rest, " warn")) out.level = P95Notice::Level::Warn;
    else if (starts_with(rest, " pending")) out.level = P95Notice::Level::Pending;
    else out.level = P95Notice::Level::Ready;

    long long n = 0;
    if (read_kv_int(rest, "worker", n)) out.worker = static_cast<int>(n);
    if (read_kv_int(rest, "added", n)) out.added = n;
    if (read_kv_int(rest, "pending_delivered", n)) out.pending_delivered = n;
    if (read_kv_int(rest, "lines", n)) out.lines = n;
    if (read_kv_int(rest, "pending", n)) out.pending_at_start = n;
    read_kv_str(rest, "file", out.file);
    read_kv_str(rest, "note", out.note);
    read_kv_str(rest, "error", out.error);
    return true;
}

bool parse_stage2(const std::string &line,Stage2Event &e) {
    using Kind=Stage2Event::Kind;
    auto integer=[&](const char *key,unsigned long long &value) {
        long long v=0;if(read_kv_int(line,key,v) && v>=0)value=static_cast<unsigned long long>(v);
    };
    if(starts_with(line,"stage2_short_calibration:")) {
        e.kind=Kind::Phase;e.phase="calibration";
    } else if(starts_with(line,"stage2_short_calibration_done:")) {
        e.kind=Kind::Phase;e.phase="starting";
    } else if(starts_with(line,"stage2_plan:")) {
        e.kind=Kind::Plan;integer("available",e.total);integer("completed",e.done);integer("B2",e.b2);integer("D",e.d);
        const auto a=line.find("save="),b=line.find(" requested=");
        if(a!=line.npos && b!=line.npos && b>a+5)e.save=line.substr(a+5,b-a-5);
    } else if(starts_with(line,"curve_start:")) {
        if(std::sscanf(line.c_str(),"curve_start: %llu/%llu record=%llu sigma=%llu B1=%llu",
            &e.done,&e.total,&e.record,&e.sigma,&e.b1)<2)return false;
        e.kind=Kind::CurveStart;
    } else if(starts_with(line,"real_shape:") || starts_with(line,"stage2_real:")) {
        e.kind=Kind::Shape;integer("B1",e.b1);integer("B2",e.b2);integer("D",e.d);integer("P",e.p);
        if(!e.p){const char *key=" P=phi(D)/2=";const auto a=line.find(key);if(a!=line.npos){const char *value=line.c_str()+a+std::strlen(key);read_u64(value,e.p);}}
        integer("S_bits",e.bits);if(!e.bits)integer("N_bits",e.bits);
    } else if(starts_with(line,"stage2_phase:")) {
        e.kind=Kind::Phase;const auto a=line.find(':')+1,b=line.find(" previous=");
        e.phase=line.substr(a,(b==line.npos?line.size():b)-a);
        while(!e.phase.empty() && e.phase.front()==' ')e.phase.erase(e.phase.begin());
        read_after(line,"elapsed=",e.elapsed);
    } else if(starts_with(line,"batched_progress:")) {
        if(std::sscanf(line.c_str(),"batched_progress: batch=%llu/%llu",&e.batch,&e.batches)!=2)return false;
        e.kind=Kind::Batch;read_after(line,"t=",e.elapsed);read_after(line,"left=",e.left);
    } else if(starts_with(line,"stage2_full_wall:")) {
        e.kind=Kind::FullWall;read_after(line,"total=",e.seconds);
    } else if(starts_with(line,"stage2_result:")) {
        e.kind=Kind::Result;integer("record",e.record);read_after(line,"seconds=",e.seconds);
        integer("B1",e.b1);read_kv_str(line,"save",e.save);
        if(const char *at=find_str(line.c_str()," sigma=")){at+=7;read_u64(at,e.sigma);}
        const auto a=line.find("\"factors\":[");
        if(a!=line.npos) {
            const auto end=line.find(']',a+11);size_t at=a+11;
            while(end!=line.npos && at<end) {
                const auto q=line.find('"',at);if(q==line.npos || q>=end)break;
                const auto r=line.find('"',q+1);if(r==line.npos || r>=end)break;
                const auto factor=line.substr(q+1,r-q-1);
                if(!factor.empty() && factor.find_first_not_of("0123456789")==factor.npos)e.factors.push_back(factor);
                at=r+1;
            }
        }
    } else if(starts_with(line,"curve_done:")) {
        if(std::sscanf(line.c_str(),"curve_done: %llu/%llu record=%llu",&e.done,&e.total,&e.record)<2)return false;
        e.kind=Kind::CurveDone;read_after(line,"wall=",e.seconds);
    } else if(starts_with(line,"task_done:")) {
        e.kind=Kind::TaskDone;integer("completed",e.done);
    } else if(starts_with(line,"stage2_complete:") || starts_with(line,"stage2_stopped:"))e.kind=Kind::Complete;
    if(e.kind==Kind::CurveStart || e.kind==Kind::Result) {
        std::size_t start=0;
        if(kv_pos(line,"curve_model",start)) {
            const auto end=line.find_first_of(" \t\r\n",start);
            e.curve_model=line.substr(start,end==line.npos?line.npos:end-start);
        }
        if(e.kind==Kind::CurveStart && e.curve_model.empty())e.curve_model="mont";
        if(e.curve_model=="mont") {
            e.param=0;long long value=0;
            if(read_kv_int(line,"param",value) && value>=0 && value<=3)e.param=static_cast<int>(value);
        }
    }
    return e.kind!=Kind::None;
}

} // namespace

void Stage2Status::apply(const Stage2Event &e) {
    using Kind=Stage2Event::Kind;valid=true;
    if(e.kind==Kind::Plan) {
        *this=Stage2Status{};valid=true;total=e.total;done=std::min(e.done,total);current=done;
        b2=e.b2;d=e.d;save=e.save;phase="starting";
    } else if(e.kind==Kind::CurveStart) {
        total=e.total;current=std::min(e.done,total);done=current?current-1:0;
        record=e.record;sigma=e.sigma;b1=e.b1;engine_seconds=0;batch=batches=0;batch_left=0;elapsed=0;phase="read";
        curve_model=e.curve_model.empty()?"mont":e.curve_model;param=curve_model=="mont"?e.param:-1;
    } else if(e.kind==Kind::Shape) {
        if(e.b1)b1=e.b1;if(e.b2)b2=e.b2;if(e.d)d=e.d;if(e.p)p=e.p;if(e.bits)bits=e.bits;
    } else if(e.kind==Kind::Phase) {
        phase=e.phase;elapsed=e.elapsed;batch=batches=0;batch_left=0;
    } else if(e.kind==Kind::Batch) {
        batch=e.batch;batches=e.batches;batch_left=e.left;
    } else if(e.kind==Kind::Result) {
        if(e.record)record=e.record;if(e.sigma)sigma=e.sigma;if(e.b1)b1=e.b1;if(!e.save.empty())save=e.save;
        if(!e.curve_model.empty()){curve_model=e.curve_model;param=curve_model=="mont"?e.param:-1;}
    } else if(e.kind==Kind::FullWall)engine_seconds=e.seconds;
    else if(e.kind==Kind::CurveDone) {
        total=e.total;done=std::min(e.done,total);current=done;record=e.record;
        const auto seconds=engine_seconds>0?engine_seconds:e.seconds;
        if(seconds>0 && std::isfinite(seconds)){seconds_sum+=seconds;++measured_curves;}
        batch=batches=0;batch_left=0;phase="curve_done";
    } else if(e.kind==Kind::TaskDone){done=total;phase="task_done";}
    else if(e.kind==Kind::Complete)phase="complete";
}
ProgressInfo Stage2Status::progress()const {
    ProgressInfo out;out.valid=valid && total>0;out.gpu=true;
    out.curves_done=done;out.curves_total=total;out.pct=total?100.*done/total:0;out.elapsed_s=elapsed;
    out.bits=bits;
    out.s_per_curve=measured_curves?seconds_sum/measured_curves:0;
    out.eta_s=(total>done?total-done:0)*out.s_per_curve;return out;
}
HitInfo Stage2Status::factor_hit(const std::string &factor)const {
    HitInfo hit;hit.factor=factor;hit.curve=static_cast<int>(record);
    hit.sigma=sigma;hit.has_sigma=sigma!=0;hit.param=param;
    hit.save=save;hit.curve_model=curve_model;hit.execution_target="gpu";
    return hit;
}

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
    if(parse_stage2(p,out.stage2)) {
        out.queue_done=starts_with(p,"stage2_complete:");
        const auto kind=out.stage2.kind;
        const bool detail=kind==Stage2Event::Kind::Shape || kind==Stage2Event::Kind::Batch || kind==Stage2Event::Kind::FullWall;
        out.kind=out.queue_done?LogKind::QueueDone:detail?LogKind::Raw:LogKind::Event;
        return out;
    }

    // ---- hit ---------------------------------------------------------------
    if (starts_with(p, "factor[")) {
        const std::size_t eq = p.find('=');
        if (eq != std::string::npos) {
            const std::string v = p.substr(eq + 1);
            // "factor[i]=<decimal>" possibly followed by the D3 fields
            // " curve=… sigma=… param=… curve_model=… execution_target=… save=…".
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
                if (const char *q = find_str(c, " curve_model=")) {
                    const char *r = q + 13;
                    while (*r != '\0' && *r != ' ') out.hit.curve_model.push_back(*r++);
                }
                if (const char *q = find_str(c, " execution_target=")) {
                    const char *r = q + 18;
                    while (*r != '\0' && *r != ' ') out.hit.execution_target.push_back(*r++);
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

    // ---- p95_add: notices (Prime95 handoff, docs/usage/GUI.md 18) ----------
    // Classified as an Event so the notice also reaches the log pane; the notice strip
    // (App::draw_p95_notice) aggregates the fields.
    {
        P95Notice notice;
        if (parse_p95_notice(p, notice)) {
            out.p95 = notice;
            out.kind = LogKind::Event;
            return out;
        }
    }

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
        const char *columns = find_str(c, "columns");
        if(columns != nullptr) {
            const char *q=columns;
            while(q>c && *(q-1)==' ')--q;
            while(q>c && is_digit(*(q-1)))--q;
            read_u64(q,pr.columns);
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
                          contains(p, "warning:") || starts_with(p,"WARNING:") || starts_with(p,"ecm_cuda_stage2:");
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
        starts_with(p, "saves :") || starts_with(p, "curve model :") ||
        starts_with(p, "execution target :") ||
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
