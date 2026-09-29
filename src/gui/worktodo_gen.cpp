#include "worktodo_gen.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#ifdef _WIN32
#include <sys/stat.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace ecmgui {
namespace {

std::string trim(const std::string &s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool is_int_like(const std::string &raw) {
    // Same rule as ecm.py's is_int_like(): optional sign, then digits only.
    const std::string s = trim(raw);
    if (s.empty()) return false;
    std::size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

// ecm.py's _NUM_RE: optional sign, digits with an optional decimal part, optional exponent.
bool is_number_like(const std::string &raw) {
    const std::string s = trim(raw);
    if (s.empty()) return false;
    std::size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    bool digits = false, dot = false;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (std::isdigit(static_cast<unsigned char>(c))) { digits = true; continue; }
        if (c == '.' && !dot) { dot = true; continue; }
        if ((c == 'e' || c == 'E') && digits) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if (i >= s.size()) return false;
            for (; i < s.size(); ++i) {
                if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
            }
            return true;
        }
        return false;
    }
    return digits;
}

bool strtod_value(const std::string &raw, double &out) {
    if (!is_number_like(raw)) return false;
    out = std::strtod(trim(raw).c_str(), nullptr);
    return true;
}

// Splits on ','; the only quoted field (the known-factor list) is stripped BEFORE the
// split, exactly like ecm.py and our C++ driver parser.
std::vector<std::string> split_csv(const std::string &text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ',') { out.push_back(trim(cur)); cur.clear(); continue; }
        cur += c;
    }
    out.push_back(trim(cur));
    return out;
}

std::string csv_quote(const std::string &value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

std::string to_string_ll(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    return std::string(buf);
}

// bits of a positive integer given as a decimal string: floor((L-1)*log2(10) + log2(d))+1.
int decimal_bits(const std::string &digits) {
    std::string d = trim(digits);
    while (d.size() > 1 && d[0] == '0') d.erase(d.begin());
    if (d.empty() || !is_int_like(d)) return 0;
    const double lead = static_cast<double>(d[0] - '0');
    if (lead <= 0.0) return 0;
    const double log2v = static_cast<double>(d.size() - 1) * 3.321928094887362 + std::log2(lead);
    return static_cast<int>(std::floor(log2v)) + 1;
}

long long int_or(const std::string &s, long long fallback) {
    try {
        return std::stoll(trim(s));
    } catch (...) {
        return fallback;
    }
}

// ecm.py's Task::number_key(): numerical identity of k*b^n+c.
struct NumberKey {
    long long k = 0, b = 0, n = 0, c = 0;
    bool operator<(const NumberKey &o) const {
        if (k != o.k) return k < o.k;
        if (b != o.b) return b < o.b;
        if (n != o.n) return n < o.n;
        return c < o.c;
    }
};

NumberKey number_key(const GenTask &t) {
    NumberKey key;
    key.k = int_or(t.k, 0);
    key.b = int_or(t.b, 0);
    key.n = t.n;
    key.c = int_or(t.c, 0);
    return key;
}

double b1_value(const GenTask &t) {
    double v = 0.0;
    return strtod_value(t.b1, v) ? v : 0.0;
}
double b2_value(const GenTask &t) {
    double v = 0.0;
    return strtod_value(t.b2, v) ? v : 0.0;
}
bool has_real_aid(const GenTask &t) {
    if (t.aid.empty()) return false;
    std::string up = t.aid;
    for (char &c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return up != "N/A";
}

// ── emitting (mirrors ecm.py's emit_ecmstage2) ───────────────────────────────────────

std::string emit_stage2(const GenTask &t, const std::string &save, long long skip_curves,
                        long long curves) {
    std::vector<std::string> parts;
    if (!t.aid.empty()) parts.push_back(t.aid);
    parts.push_back(t.k);
    parts.push_back(t.b);
    parts.push_back(to_string_ll(t.n));
    parts.push_back(t.c);
    parts.push_back(csv_quote(save));
    parts.push_back(t.b2);
    parts.push_back(to_string_ll(skip_curves));
    parts.push_back(to_string_ll(curves));
    if (!t.factors.empty()) {
        std::string joined;
        for (std::size_t i = 0; i < t.factors.size(); ++i) {
            if (i) joined += ",";
            joined += t.factors[i];
        }
        parts.push_back(csv_quote(joined));
    }
    std::string out = "ECMSTAGE2=";
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) out += ",";
        out += parts[i];
    }
    return out;
}

} // namespace

// ── GpuProfile ───────────────────────────────────────────────────────────────────────

const GpuTier *GpuProfile::pick(int bits) const {
    for (const GpuTier &t : tiers) {
        if (t.bits >= bits + carry_bits) return &t;
    }
    return nullptr;
}

bool parse_gpu_info(const std::string &text, GpuProfile &out, std::string &err) {
    out = GpuProfile();
    std::istringstream in(text);
    std::string line;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;

        if (line.compare(0, 5, "tier ") == 0 || line.compare(0, 5, "tier\t") == 0) {
            GpuTier tier;
            std::istringstream ls(line);
            std::string tok;
            while (ls >> tok) {
                const std::size_t eq = tok.find('=');
                if (eq == std::string::npos) continue;      // the leading `tier` token
                const std::string key = tok.substr(0, eq);
                const long long v = int_or(tok.substr(eq + 1), 0);
                if (key == "bits") tier.bits = static_cast<int>(v);
                else if (key == "tpb") tier.tpb = static_cast<int>(v);
                else if (key == "tpi") tier.tpi = static_cast<int>(v);
                else if (key == "ipb") tier.ipb = static_cast<int>(v);
                else if (key == "blocks_per_sm") tier.blocks_per_sm = static_cast<int>(v);
                else if (key == "blocks_min") tier.blocks_min = static_cast<int>(v);
                else if (key == "curves_min") tier.curves_min = v;
                else if (key == "blocks_wave") tier.blocks_wave = static_cast<int>(v);
                else if (key == "curves_wave") tier.curves_wave = v;
            }
            if (tier.bits > 0) out.tiers.push_back(tier);
            continue;
        }

        // "key=value" (or "key=\"value\""): the shape --gpu-info prints for the rest.
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            val = val.substr(1, val.size() - 2);
        }
        if (key == "gpu_info") {
            if (val == "not_applicable") {
                out.not_applicable = true;
                out.error = "the linked backend does not support --gpu-info";
                continue;
            }
            saw_header = (val == "1");
        } else if (key == "device") out.device = static_cast<int>(int_or(val, 0));
        else if (key == "name") out.name = val;
        else if (key == "sm_count") out.sm_count = static_cast<int>(int_or(val, 0));
        else if (key == "carry_bits") out.carry_bits = static_cast<int>(int_or(val, 6));
        else if (key == "gpu_param") out.gpu_param = static_cast<int>(int_or(val, 3));
        else if (key == "fold") out.fold = static_cast<int>(int_or(val, 0));
        else if (key == "reason" && out.not_applicable) out.error = val;
    }

    if (out.not_applicable) {
        err = out.error;
        return false;
    }
    if (!saw_header) {
        out.error = "no gpu_info=1 line (is this the output of `--gpu-info`?)";
        err = out.error;
        return false;
    }
    if (out.sm_count <= 0 || out.tiers.empty()) {
        out.error = "the report has no SM count or no kernel tiers";
        err = out.error;
        return false;
    }
    std::stable_sort(out.tiers.begin(), out.tiers.end(),
                     [](const GpuTier &a, const GpuTier &b) { return a.bits < b.bits; });
    out.valid = true;
    err.clear();
    return true;
}

// ── parsing ─────────────────────────────────────────────────────────────────────────

bool effective_bits(const std::string &k, const std::string &b, long long n,
                    const std::string &c, const std::vector<std::string> &factors,
                    int &bits_out, std::string &err) {
    const long long ki = int_or(trim(k), 0);
    const long long bi = int_or(trim(b), 0);
    if (ki <= 0 || bi < 2 || n < 0) {
        err = "k*b^n+c needs k > 0, b >= 2 and n >= 0 to size the kernel tier";
        return false;
    }
    // log2(k) + n*log2(b), then round up: an integer v > 0 has floor(log2 v)+1 bits.
    const double log2v = std::log2(static_cast<double>(ki)) +
                         static_cast<double>(n) * std::log2(static_cast<double>(bi));
    double int_part = 0.0;
    const double frac = std::modf(log2v, &int_part);
    long long bits = static_cast<long long>(std::floor(log2v + 1.0e-9)) + 1;

    // k*b^n + c with a negative c: when k*b^n is an EXACT power of two (k=1, b a power of
    // two -- the Mersenne case, and the common one), subtracting |c| clears the top bit, so
    // 2^521-1 has 521 bits and not 522. Without this correction every Mersenne task would be
    // reported one bit too large and could land a tier up.
    double c_value = 0.0;
    const bool have_c = strtod_value(trim(c), c_value);
    if (have_c && c_value < 0.0 && std::fabs(frac) < 1.0e-9) {
        bits -= 1;
    }
    if (bits < 1) bits = 1;

    for (const std::string &f : factors) {
        const std::string digits = trim(f);
        if (!is_int_like(digits)) {
            err = "known factor '" + f + "' is not a plain integer";
            return false;
        }
        const int fb = decimal_bits(digits);
        if (fb <= 0) {
            err = "known factor '" + f + "' is not usable";
            return false;
        }
        bits -= fb;
        // Subtracting the divisor's bit count is a LOWER bound for the quotient's bit count
        // (the true value can need one bit more), so with known factors the estimate may be
        // one bit low. That is the safe direction for a recommendation: a smaller tier means
        // a larger `ipb`, i.e. slightly MORE curves than strictly needed.
        if (bits < 1) bits = 1;
    }
    bits_out = static_cast<int>(bits);
    err.clear();
    return true;
}

bool parse_assignment(const std::string &line, int idx, GenTask &out, std::string &err) {
    const std::string s = trim(line);
    std::string body;
    if (s.compare(0, 5, "ECM2=") == 0) { body = s.substr(5); out.keyword = "ECM2"; }
    else if (s.compare(0, 4, "ECM=") == 0) { body = s.substr(4); out.keyword = "ECM"; }
    else {
        err = "line does not start with ECM= or ECM2=";
        return false;
    }

    std::string factors_raw;
    std::string unquoted = body;
    const std::size_t q0 = body.find('"');
    if (q0 != std::string::npos) {
        unquoted = body.substr(0, q0);
        const std::size_t q1 = body.rfind('"');
        if (q1 > q0) factors_raw = body.substr(q0 + 1, q1 - q0 - 1);
    }

    const std::vector<std::string> cols = split_csv(unquoted);
    if (cols.empty() || (cols.size() == 1 && cols[0].empty())) {
        err = "no fields after ECM=/ECM2=";
        return false;
    }

    std::size_t i = 0;
    if (!is_int_like(cols[i]) && cols[i].compare(0, 5, "FFT2=") != 0) {
        out.aid = cols[i];
        i++;
    }
    if (i < cols.size() && cols[i].compare(0, 5, "FFT2=") == 0) {
        out.fft2 = cols[i].substr(5);
        i++;
    }
    if (cols.size() < i + 5) {
        err = "not enough fields (need k,b,n,c,B1)";
        return false;
    }

    out.k = cols[i];
    out.b = cols[i + 1];
    const std::string n_raw = cols[i + 2];
    out.c = cols[i + 3];
    out.b1 = cols[i + 4];

    if (!is_int_like(n_raw) || (!n_raw.empty() && n_raw[0] == '-')) {
        err = "invalid exponent n: '" + n_raw + "'";
        return false;
    }
    out.n = int_or(n_raw, 0);
    if (out.n < 0) {
        err = "invalid exponent n: '" + n_raw + "'";
        return false;
    }
    double b1v = 0.0;
    if (!strtod_value(out.b1, b1v) || b1v <= 0.0) {
        err = "invalid B1: '" + out.b1 + "'";
        return false;
    }

    std::size_t p = i + 5;
    out.b2 = "0";
    if (cols.size() > p) {
        if (!cols[p].empty()) {
            double v = 0.0;
            if (!strtod_value(cols[p], v) || v < 0.0) {
                err = "invalid B2: '" + cols[p] + "'";
                return false;
            }
            out.b2 = cols[p];
        }
        p++;
    }

    out.curves = 100;
    if (cols.size() > p) {
        if (!cols[p].empty() && is_int_like(cols[p])) {
            const long long v = int_or(cols[p], 0);
            if (v > 0 && v <= 0xFFFFFFFFll) out.curves = v;
        }
        p++;
    }

    out.sigma.clear();
    if (cols.size() > p) {
        if (!cols[p].empty() && is_int_like(cols[p]) && int_or(cols[p], 0) > 0) out.sigma = cols[p];
        p++;
    }

    if (!factors_raw.empty()) {
        for (const std::string &f : split_csv(factors_raw)) {
            if (!f.empty()) out.factors.push_back(f);
        }
    }
    out.idx = idx;
    err.clear();
    return true;
}

// ── save names ──────────────────────────────────────────────────────────────────────

std::string render_save_name(const std::string &pattern, const GenTask &t) {
    std::string out = pattern;
    const std::pair<const char *, std::string> subs[] = {
        {"{k}", t.k}, {"{b}", t.b}, {"{c}", t.c},
        {"{n}", to_string_ll(t.n)}, {"{b1}", t.b1}, {"{b2}", t.b2},
    };
    for (const auto &s : subs) {
        std::string key = s.first;
        std::size_t pos = 0;
        while ((pos = out.find(key, pos)) != std::string::npos) {
            out.replace(pos, key.size(), s.second);
            pos += s.second.size();
        }
    }
    return out;
}

const char *check_save_name(const std::string &name) {
    static thread_local std::string why;
    if (name.size() < 6 || name.compare(name.size() - 5, 5, ".save") != 0) {
        why = "save name does not match <...>_<B1>.save: '" + name + "'";
        return why.c_str();
    }
    const std::string stem = name.substr(0, name.size() - 5);
    const std::size_t us = stem.rfind('_');
    if (us == std::string::npos || us + 1 >= stem.size()) {
        why = "save name does not match <...>_<B1>.save: '" + name + "'";
        return why.c_str();
    }
    const std::string token = stem.substr(us + 1);
    double v = 0.0;
    if (!strtod_value(token, v) || v <= 0.0) {
        why = "cannot parse B1 from save-name token '" + token + "' (in '" + name + "')";
        return why.c_str();
    }
    return nullptr;
}

// ── pipeline ────────────────────────────────────────────────────────────────────────

GenResult generate(const std::string &input, const GpuProfile &gpu, const GenOptions &opt) {
    GenResult r;
    if (!opt.valid) {
        r.error = opt.error.empty() ? "the generator options are invalid" : opt.error;
        return r;
    }
    const bool need_recommendation = opt.curves_fixed < 0 && opt.use_recommended;
    if (need_recommendation && !gpu.valid) {
        r.error = "the recommended curves need a working `--gpu-info` report: " +
                  (gpu.error.empty() ? std::string("no GPU profile") : gpu.error);
        return r;
    }

    std::vector<GenTask> tasks;
    std::istringstream in(input);
    std::string line;
    int idx = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string s = trim(line);
        if (s.empty() || s[0] == '#') { r.skipped_comment++; continue; }
        if (s.compare(0, 5, "ECM2=") != 0 && s.compare(0, 4, "ECM=") != 0) {
            // ecm.py's split: ECMSTAGE2= is our OUTPUT format (the generator only takes
            // assignments), everything else is an unknown line.
            if (s.compare(0, 10, "ECMSTAGE2=") == 0) r.skipped_other++;
            else r.skipped_unknown++;
            continue;
        }
        GenTask t;
        std::string err;
        if (!parse_assignment(s, idx, t, err)) {
            // ecm.py aborts here; the GUI reports the line and generates the rest, so one
            // typo in a paste does not throw away the other 200 assignments.
            r.skipped_unknown++;
            r.parse_errors.push_back(err + " -- " + s);
            continue;
        }
        r.read++;
        tasks.push_back(t);
        idx++;
    }

    // ---- filters (ecm.py's apply_filters) ----
    {
        std::vector<GenTask> kept;
        for (const GenTask &t : tasks) {
            if (opt.min_n >= 0 && t.n < opt.min_n) { r.filtered++; continue; }
            if (opt.max_n >= 0 && t.n > opt.max_n) { r.filtered++; continue; }
            if (opt.min_curves >= 0 && t.curves < opt.min_curves) { r.filtered++; continue; }
            if (opt.max_curves >= 0 && t.curves > opt.max_curves) { r.filtered++; continue; }
            kept.push_back(t);
        }
        tasks.swap(kept);
    }

    // ---- dedup (winner: B1, curves, real AID, first seen; factors are merged) ----
    if (opt.dedup) {
        std::map<NumberKey, std::vector<std::size_t>> groups;
        std::vector<NumberKey> order;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            const NumberKey key = number_key(tasks[i]);
            if (groups.find(key) == groups.end()) order.push_back(key);
            groups[key].push_back(i);
        }
        std::vector<GenTask> out;
        for (const NumberKey &key : order) {
            const std::vector<std::size_t> &rows = groups[key];
            if (rows.size() == 1) { out.push_back(tasks[rows[0]]); continue; }
            r.duplicates += static_cast<int>(rows.size() - 1);
            std::size_t best = rows[0];
            for (std::size_t k = 1; k < rows.size(); ++k) {
                const GenTask &a = tasks[rows[k]];
                const GenTask &b = tasks[best];
                const double ab1 = b1_value(a), bb1 = b1_value(b);
                bool better = false;
                if (ab1 != bb1) better = ab1 > bb1;
                else if (a.curves != b.curves) better = a.curves > b.curves;
                else if (has_real_aid(a) != has_real_aid(b)) better = has_real_aid(a);
                else better = a.idx < b.idx;
                if (better) best = rows[k];
            }
            GenTask winner = tasks[best];
            std::vector<std::string> merged;
            for (std::size_t i : rows) {
                for (const std::string &f : tasks[i].factors) {
                    if (std::find(merged.begin(), merged.end(), f) == merged.end()) merged.push_back(f);
                }
            }
            winner.factors = merged;
            out.push_back(winner);
        }
        tasks.swap(out);
    } else {
        for (GenTask &t : tasks) {
            std::vector<std::string> seen;
            for (const std::string &f : t.factors) {
                if (std::find(seen.begin(), seen.end(), f) == seen.end()) seen.push_back(f);
            }
            t.factors = seen;
        }
    }

    // ---- rewrites ----
    for (GenTask &t : tasks) {
        if (!opt.set_b1.empty()) t.b1 = opt.set_b1;
        if (!opt.set_b2.empty()) t.b2 = opt.set_b2;
        if (opt.set_has_na && t.aid.empty()) t.aid = "N/A";
        if (opt.sort_factors) {
            std::stable_sort(t.factors.begin(), t.factors.end(),
                             [](const std::string &a, const std::string &b) {
                                 return int_or(a, 0) < int_or(b, 0);
                             });
        }
    }

    // ---- sort (ecm.py's _SORT_KEYS, stable with idx as the final tie-break) ----
    if (!trim(opt.sort_by).empty()) {
        std::vector<std::string> keys;
        std::stringstream ss(opt.sort_by);
        std::string k;
        while (std::getline(ss, k, ',')) {
            const std::string kk = trim(k);
            if (kk.empty()) continue;
            if (kk != "n" && kk != "k" && kk != "b" && kk != "c" && kk != "b1" && kk != "b2" &&
                kk != "curves" && kk != "aid") {
                r.error = "unsupported sort field: '" + kk + "' (n,k,b,c,b1,b2,curves,aid)";
                return r;
            }
            keys.push_back(kk);
        }
        std::stable_sort(tasks.begin(), tasks.end(), [&](const GenTask &a, const GenTask &b) {
            for (const std::string &key : keys) {
                int cmp = 0;
                if (key == "n") cmp = (a.n < b.n) ? -1 : (a.n > b.n ? 1 : 0);
                else if (key == "k") cmp = (int_or(a.k, 0) < int_or(b.k, 0)) ? -1 : (int_or(a.k, 0) > int_or(b.k, 0) ? 1 : 0);
                else if (key == "b") cmp = (int_or(a.b, 0) < int_or(b.b, 0)) ? -1 : (int_or(a.b, 0) > int_or(b.b, 0) ? 1 : 0);
                else if (key == "c") cmp = (int_or(a.c, 0) < int_or(b.c, 0)) ? -1 : (int_or(a.c, 0) > int_or(b.c, 0) ? 1 : 0);
                else if (key == "b1") { const double x = b1_value(a), y = b1_value(b); cmp = (x < y) ? -1 : (x > y ? 1 : 0); }
                else if (key == "b2") { const double x = b2_value(a), y = b2_value(b); cmp = (x < y) ? -1 : (x > y ? 1 : 0); }
                else if (key == "curves") cmp = (a.curves < b.curves) ? -1 : (a.curves > b.curves ? 1 : 0);
                else if (key == "aid") cmp = a.aid.compare(b.aid);
                if (cmp != 0) return opt.sort_desc ? (cmp > 0) : (cmp < 0);
            }
            return opt.sort_desc ? (a.idx > b.idx) : (a.idx < b.idx);
        });
    }

    // ---- per-line tier + curves, save name, emit ----
    struct Emitted {
        int worker = 0;
        std::string line;
    };
    std::vector<Emitted> emitted;

    // Workers of the target device, ascending: lines are distributed round-robin so a card
    // driven by several workers gets an even split, in sorted order.
    std::vector<int> target_workers;
    for (const auto &wd : opt.worker_devices) {
        if (wd.second == opt.target_device) target_workers.push_back(wd.first);
    }
    std::sort(target_workers.begin(), target_workers.end());

    std::size_t rr = 0;
    for (GenTask &t : tasks) {
        std::string err;
        if (!effective_bits(t.k, t.b, t.n, t.c, t.factors, t.bits, err)) {
            r.error = "n=" + to_string_ll(t.n) + ": " + err;
            return r;
        }
        long long curves = t.curves;
        if (opt.curves_fixed >= 0) {
            curves = opt.curves_fixed;
        } else if (opt.use_recommended) {
            const GpuTier *tier = gpu.pick(t.bits);
            if (tier == nullptr) {
                r.error = "n=" + to_string_ll(t.n) + ": no kernel tier of this build covers " +
                          std::to_string(t.bits) + " bits";
                return r;
            }
            const int per_sm = (opt.blocks_per_sm > 0) ? opt.blocks_per_sm : 1;
            curves = static_cast<long long>(per_sm) * gpu.sm_count * tier->ipb;
            if (curves < tier->curves_min) curves = tier->curves_min;
        }
        t.curves_out = curves;

        const std::string save = render_save_name(opt.save_pattern, t);
        if (const char *why = check_save_name(save)) {
            r.error = std::string(why) +
                      " (the driver takes B1 from the save name; change the save pattern)";
            return r;
        }

        Emitted e;
        e.worker = target_workers.empty() ? 0 : target_workers[rr % target_workers.size()];
        e.line = emit_stage2(t, save, opt.skip_curves, curves);
        emitted.push_back(e);
        rr++;
    }

    // ---- segments + text (one section per worker that got work, ascending) ----
    std::vector<int> workers;
    for (const Emitted &e : emitted) {
        if (std::find(workers.begin(), workers.end(), e.worker) == workers.end()) {
            workers.push_back(e.worker);
        }
    }
    std::sort(workers.begin(), workers.end());
    std::string text;
    for (int w : workers) {
        GenSegment seg;
        seg.worker = w;
        for (const Emitted &e : emitted) {
            if (e.worker == w) seg.lines.push_back(e.line);
        }
        if (w > 0) text += "[Worker #" + std::to_string(w) + "]\r\n";
        for (const std::string &l : seg.lines) text += l + "\r\n";
        // A [Worker #N] section ends with a blank line (ecm.py's write_section); a
        // header-less file has no blank line at all (ecm.py's write_lines), which is what
        // the byte-for-byte comparison against --out-ecmstage2 pins down.
        if (w > 0) text += "\r\n";
        r.segments.push_back(seg);
    }

    r.tasks = tasks;
    r.text = text;
    r.ok = true;
    return r;
}

// ── applying ────────────────────────────────────────────────────────────────────────

bool stamp_file(const std::string &path, FileStamp &out) {
    out = FileStamp();
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    out.exists = true;
    out.size = static_cast<long long>(st.st_size);
    out.mtime = static_cast<long long>(st.st_mtime);
    return true;
}

bool apply_append(const std::string &path, const std::string &text, const FileStamp &expected,
                  std::string &err, size_t *appended) {
    if (text.empty()) {
        err = "nothing to append (generate a preview first)";
        return false;
    }
    FileStamp now;
    const bool exists_now = stamp_file(path, now);
    if (expected.exists != exists_now || (exists_now && (now.size != expected.size ||
                                                         now.mtime != expected.mtime))) {
        err = "the target changed since the preview was generated (" + path +
              "); generate the preview again";
        return false;
    }

    std::string payload = text;
    if (exists_now && now.size > 0) {
        // ecm.py's rule: add a separating newline when the file does not end with one.
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            err = "cannot read " + path;
            return false;
        }
        in.seekg(-1, std::ios::end);
        char last = '\0';
        in.get(last);
        if (last != '\n' && last != '\r') payload = "\r\n" + payload;
    }

    // Append-only, and no truncation of what is already there: a running queue owns the
    // file's existing lines.
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        err = "cannot open " + path + " for append";
        return false;
    }
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    out.close();
    if (out.fail()) {
        err = "write failed: " + path;
        return false;
    }
    if (appended != nullptr) *appended = payload.size();
    return true;
}

} // namespace ecmgui
