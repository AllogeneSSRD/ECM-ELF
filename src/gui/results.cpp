#include "results.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {
namespace {

std::string utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

bool replace_file(const std::string &tmp, const std::string &dst, std::string &err) {
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        err = "MoveFileEx failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    return true;
#else
    if (std::rename(tmp.c_str(), dst.c_str()) != 0) {
        err = "rename failed";
        return false;
    }
    return true;
#endif
}

// JSON helpers: the JSONL is hand-written (no dependency), so the escaping rules that
// matter here are just the ones for the strings we emit (filenames and task lines).
std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string json_unescape(const std::string &s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        switch (s[++i]) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case 'u': {
                // Only the \u00XX escapes our writer produces need to be readable.
                if (i + 4 < s.size()) {
                    const std::string hex = s.substr(i + 1, 4);
                    out.push_back(static_cast<char>(std::strtol(hex.c_str(), nullptr, 16)));
                    i += 4;
                }
                break;
            }
            default: out.push_back(s[i]);
        }
    }
    return out;
}

// Extracts "key":<int> / "key":"<string>" from one JSONL object (the reader only has
// to understand files this program wrote).
bool json_int(const std::string &obj, const char *key, long long *out) {
    const std::string needle = std::string("\"") + key + "\":";
    const std::size_t p = obj.find(needle);
    if (p == std::string::npos) return false;
    *out = std::strtoll(obj.c_str() + p + needle.size(), nullptr, 10);
    return true;
}

std::string json_str(const std::string &obj, const char *key) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const std::size_t p = obj.find(needle);
    if (p == std::string::npos) return std::string();
    std::size_t end = p + needle.size();
    std::string raw;
    while (end < obj.size()) {
        if (obj[end] == '\\' && end + 1 < obj.size()) {
            raw.push_back(obj[end]);
            raw.push_back(obj[end + 1]);
            end += 2;
            continue;
        }
        if (obj[end] == '"') break;
        raw.push_back(obj[end]);
        ++end;
    }
    return json_unescape(raw);
}

// "factors":["123"] -> first element.
std::string json_first_factor(const std::string &obj) {
    const std::string needle = "\"factors\":[\"";
    const std::size_t p = obj.find(needle);
    if (p == std::string::npos) return std::string();
    const std::size_t start = p + needle.size();
    const std::size_t end = obj.find('"', start);
    if (end == std::string::npos) return std::string();
    return obj.substr(start, end - start);
}

std::string join_ints(const std::vector<int> &v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(v[i]);
    }
    return out;
}

std::string join_u64(const std::vector<unsigned long long> &v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(v[i]);
    }
    return out;
}

int bit_length(const std::string &decimal) {
    // Enough for the table: a decimal digit contributes 3.32 bits.
    const std::size_t digits = decimal.size();
    if (digits == 0) return 0;
    int bits = static_cast<int>(digits * 3.321928);
    const char lead = decimal[0];
    if (lead == '1') bits -= 1;
    return bits > 0 ? bits : 1;
}

} // namespace

bool ResultsStore::parse_save_name(const std::string &save, int *exponent, std::string *b1_text) {
    // Contract (src/core/ecm_worktodo.h): m{n}_{b1}.save, with B1 in the LAST "_" token.
    if (save.size() < 4) return false;
    if (save.compare(0, 1, "m") != 0) return false;
    const std::size_t dot = save.rfind(".save");
    if (dot == std::string::npos) return false;
    const std::size_t us = save.rfind('_', dot);
    if (us == std::string::npos || us < 2) return false;
    const std::string n_text = save.substr(1, us - 1);
    const std::string b1 = save.substr(us + 1, dot - us - 1);
    if (n_text.empty() || b1.empty()) return false;
    for (char c : n_text) {
        if (c < '0' || c > '9') return false;
    }
    char *endp = nullptr;
    const double v = std::strtod(b1.c_str(), &endp);
    if (endp == nullptr || *endp != '\0') return false;
    (void)v;
    if (exponent != nullptr) *exponent = std::atoi(n_text.c_str());
    if (b1_text != nullptr) *b1_text = b1;
    return true;
}

int ResultsStore::exponent_from_n_expr(const std::string &expr) {
    // "(1*2^677-1)" / "(2^991-1)/(...)" -> 677 / 991
    const std::size_t p = expr.find("2^");
    if (p == std::string::npos) return 0;
    std::size_t q = p + 2;
    const std::size_t start = q;
    while (q < expr.size() && expr[q] >= '0' && expr[q] <= '9') ++q;
    if (q == start) return 0;
    return std::atoi(expr.substr(start, q - start).c_str());
}

bool ResultsStore::init(const std::string &json_path, const std::string &txt_path,
                        std::string &err) {
    json_path_ = json_path;
    txt_path_ = txt_path;
    merged_.clear();
    hits_ = 0;
    dirty_ = false;
    last_error_.clear();
    err.clear();
    if (json_path_.empty()) return true;      // results disabled

    // Replay an existing JSONL so a restart keeps merging into the same table.
    std::ifstream in(json_path_, std::ios::in | std::ios::binary);
    if (!in.is_open()) return true;           // first run
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        HitRecord r;
        r.factor = json_first_factor(line);
        if (r.factor.empty()) continue;
        long long v = 0;
        if (json_int(line, "exponent", &v)) r.exponent = static_cast<int>(v);
        if (json_int(line, "param", &v)) r.param = static_cast<int>(v);
        if (json_int(line, "curve", &v)) r.curve = static_cast<int>(v);
        if (json_int(line, "sigma", &v)) {
            r.sigma = static_cast<unsigned long long>(v);
            r.has_sigma = true;
        }
        if (json_int(line, "worker", &v)) r.worker = static_cast<int>(v);
        if (json_int(line, "device", &v)) r.device = static_cast<int>(v);
        long long b1i = 0;
        if (json_int(line, "b1", &b1i)) r.b1 = static_cast<double>(b1i);
        r.b1_text = json_str(line, "b1_text");
        r.n_expr = json_str(line, "n");
        r.method = json_str(line, "method");
        r.save = json_str(line, "save");
        r.task = json_str(line, "task");
        r.timestamp = json_str(line, "timestamp");
        merge(r);
    }
    return true;
}

void ResultsStore::merge(const HitRecord &r) {
    ++hits_;
    const std::string ts = r.timestamp.empty() ? utc_now() : r.timestamp;
    for (MergedFactor &m : merged_) {
        if (m.factor != r.factor) continue;
        m.hits++;
        m.last_seen = ts;
        if (r.curve >= 0 &&
            std::find(m.curves.begin(), m.curves.end(), r.curve) == m.curves.end()) {
            m.curves.push_back(r.curve);
        }
        if (r.has_sigma &&
            std::find(m.sigmas.begin(), m.sigmas.end(), r.sigma) == m.sigmas.end()) {
            m.sigmas.push_back(r.sigma);
        }
        if (m.n_expr.empty()) m.n_expr = r.n_expr;
        if (m.exponent == 0) m.exponent = r.exponent;
        if (m.b1_text.empty()) {
            m.b1_text = r.b1_text;
            m.b1 = r.b1;
        }
        if (m.param < 0) m.param = r.param;
        if (m.method.empty()) m.method = r.method;
        if (m.save.empty()) m.save = r.save;
        m.worker = r.worker;
        m.device = r.device;
        return;
    }
    MergedFactor m;
    m.factor = r.factor;
    m.bits = bit_length(r.factor);
    m.exponent = r.exponent;
    m.n_expr = r.n_expr;
    m.b1_text = r.b1_text;
    m.b1 = r.b1;
    m.param = r.param;
    m.method = r.method;
    if (r.curve >= 0) m.curves.push_back(r.curve);
    if (r.has_sigma) m.sigmas.push_back(r.sigma);
    m.hits = 1;
    m.first_seen = ts;
    m.last_seen = ts;
    m.worker = r.worker;
    m.device = r.device;
    m.save = r.save;
    merged_.push_back(std::move(m));
}

bool ResultsStore::add(const HitRecord &r, std::string &err) {
    err.clear();
    if (json_path_.empty()) return true;
    if (r.factor.empty()) {
        err = "empty factor";
        return false;
    }
    HitRecord rec = r;
    if (rec.timestamp.empty()) rec.timestamp = utc_now();

    // 1. Append the JSONL line (the durable record).
    {
        std::ofstream out(json_path_, std::ios::out | std::ios::app | std::ios::binary);
        if (!out.is_open()) {
            err = "cannot append to " + json_path_;
            last_error_ = err;
            return false;
        }
        std::ostringstream js;
        js << "{\"status\":\"F\", \"worktype\":\"ECM\"";
        if (rec.exponent > 0) js << ", \"exponent\":" << rec.exponent;
        js << ", \"factors\":[\"" << json_escape(rec.factor) << "\"]";
        if (rec.b1 > 0.0) js << ", \"b1\":" << static_cast<long long>(rec.b1);
        if (!rec.b1_text.empty()) js << ", \"b1_text\":\"" << json_escape(rec.b1_text) << "\"";
        if (!rec.n_expr.empty()) js << ", \"n\":\"" << json_escape(rec.n_expr) << "\"";
        if (rec.param >= 0) js << ", \"param\":" << rec.param;
        if (!rec.method.empty()) js << ", \"method\":\"" << json_escape(rec.method) << "\"";
        if (rec.curve >= 0) js << ", \"curve\":" << rec.curve;
        if (rec.has_sigma) js << ", \"sigma\":" << rec.sigma;
        if (!rec.save.empty()) js << ", \"save\":\"" << json_escape(rec.save) << "\"";
        js << ", \"stage\":1";
        js << ", \"worker\":" << rec.worker << ", \"device\":" << rec.device;
        if (!rec.task.empty()) js << ", \"task\":\"" << json_escape(rec.task) << "\"";
        js << ", \"timestamp\":\"" << json_escape(rec.timestamp) << "\"}\n";
        const std::string text = js.str();
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.close();
        if (out.fail()) {
            err = "write failed: " + json_path_;
            last_error_ = err;
            return false;
        }
    }

    // 2. Fold it into the merged table (written by flush()).
    merge(rec);
    dirty_ = true;
    return true;
}

std::string ResultsStore::format_line(const MergedFactor &f) {
    std::ostringstream os;
    const std::string name = f.exponent > 0 ? ("M" + std::to_string(f.exponent))
                                            : std::string("N");
    os << name << " has a factor: " << f.factor << " (ECM";
    if (!f.curves.empty()) os << " curves " << join_ints(f.curves);
    if (!f.b1_text.empty()) os << ", B1=" << f.b1_text;
    if (f.param >= 0) os << ", param " << f.param;
    if (!f.method.empty()) os << ", " << f.method;
    if (!f.sigmas.empty()) os << ", Sigmas=[" << join_u64(f.sigmas) << "]";
    os << ", hits=" << f.hits << ")";
    return os.str();
}

bool ResultsStore::flush(std::string &err) {
    err.clear();
    if (txt_path_.empty()) return true;
    if (!dirty_) return true;

    const std::string tmp = txt_path_ + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out.is_open()) {
            err = "cannot write " + tmp;
            last_error_ = err;
            return false;
        }
        out << "# ecm_gui results - one line per distinct factor, merged over every hit;\n"
            << "# generated from " << json_path_ << " (rebuildable at any time)\n"
            << "# updated " << utc_now() << "   factors: " << merged_.size()
            << "   hits: " << hits_ << "\r\n";
        for (const MergedFactor &f : merged_) {
            out << format_line(f) << "\r\n";
        }
        out.close();
        if (out.fail()) {
            err = "write failed: " + tmp;
            last_error_ = err;
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (!replace_file(tmp, txt_path_, err)) {
        last_error_ = err;
        std::remove(tmp.c_str());
        return false;
    }
    dirty_ = false;
    return true;
}

bool ResultsStore::rebuild_from_jsonl(std::string &err) {
    const std::string json = json_path_;
    const std::string txt = txt_path_;
    if (json.empty()) {
        err = "no json path";
        return false;
    }
    if (!init(json, txt, err)) return false;
    dirty_ = true;
    return flush(err);
}

} // namespace ecmgui
