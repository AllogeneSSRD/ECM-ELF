// Minimal save -> CUDA Stage2 driver. Queue edits happen only after successful
// child exit; a fresh child isolates the experimental engine's per-curve globals.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include "ecm_cuda_stage2.h"
#include "ecm_expr.h"
#include "ecm_queue_config.h"
#include "ecm_worktodo.h"
#include "ecm_stage2_fingerprint.h"
#include "ecm_stage2_factorize.h"
#include "ecm_stage2_cost_profile.h"
#include "ecm_stage2_geometry.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace s2prod {
namespace fs = std::filesystem;
struct Big {
    mpz_t z;
    Big() { mpz_init(z); }
    ~Big() { mpz_clear(z); }
    Big(const Big &) = delete;
    Big &operator=(const Big &) = delete;
};
std::string trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == s.npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string upper(std::string s) {
    for (char &c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}
std::string number(const mpz_t z, int base = 16) {
    std::vector<char> buf(mpz_sizeinbase(z, base) + 3);
    mpz_get_str(buf.data(), base, z);
    return buf.data();
}
void set64(mpz_t z, uint64_t n) {
    mpz_set_ui(z, static_cast<unsigned long>(n >> 32));
    mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, static_cast<unsigned long>(n & 0xffffffffu));
}
// Exact decimal/scientific input: no double rounding for sigma or B2.
uint64_t u64(std::string s, const char *label) {
    s = trim(s);
    auto bad = [&]() { return std::runtime_error(std::string("invalid ") + label + ": " + s); };
    if (s.empty() || s.size() > 128 || s[0] == '-') throw bad();
    if (s[0] == '+') s.erase(0, 1);
    int exp = 0;
    const auto e = s.find_first_of("eE");
    if (e != s.npos) {
        const std::string es = s.substr(e + 1);
        size_t pos = 0;
        try { exp = std::stoi(es, &pos); } catch (...) { throw bad(); }
        if (pos != es.size() || exp < -1000 || exp > 1000) throw bad();
        s.resize(e);
    }
    std::string digits;
    bool dot = false;
    int fractional = 0;
    for (char c : s) {
        if (c == '.' && !dot) { dot = true; continue; }
        if (c < '0' || c > '9') throw bad();
        digits += c;
        if (dot) ++fractional;
    }
    if (digits.empty()) throw bad();
    Big v, scale, maximum;
    if (mpz_set_str(v.z, digits.c_str(), 10)) throw bad();
    const int power = exp - fractional;
    mpz_ui_pow_ui(scale.z, 10, static_cast<unsigned long>(std::abs(power)));
    if (power < 0) {
        if (!mpz_divisible_p(v.z, scale.z)) throw bad();
        mpz_divexact(v.z, v.z, scale.z);
    } else mpz_mul(v.z, v.z, scale.z);
    set64(maximum.z, UINT64_MAX);
    if (mpz_cmp(v.z, maximum.z) > 0) throw bad();
    return std::stoull(number(v.z, 10));
}
std::string json_string(const std::string &s) {
    std::string out = "\"";
    const char *hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += c;
    }
    return out + '"';
}
uint64_t fingerprint(const std::string &s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}
struct Record {
    std::string n, x;
    uint64_t sigma = 0, b1 = 0, offset = 0, hash = 0, index = 0;
    bool checksum = false;
};
Record parse_record(std::string line) {
    if (line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    if (line.find('\0') != line.npos)
        throw std::runtime_error("binary checkpoint is not a METHOD=ECM text save");
    std::map<std::string, std::string> fields;
    size_t a = 0;
    while (a < line.size()) {
        const auto end = line.find(';', a);
        std::string token = trim(line.substr(a, end == line.npos ? end : end - a));
        a = end == line.npos ? line.size() : end + 1;
        if (token.empty()) continue;
        const auto eq = token.find('=');
        if (eq == token.npos) throw std::runtime_error("save field has no '='");
        const auto key = upper(trim(token.substr(0, eq)));
        if (!fields.emplace(key, trim(token.substr(eq + 1))).second)
            throw std::runtime_error("duplicate save field: " + key);
    }
    for (const char *key : {"METHOD", "SIGMA", "B1", "N", "X"})
        if (!fields.count(key) || fields[key].empty())
            throw std::runtime_error(std::string("save is missing ") + key);
    if (upper(fields["METHOD"]) != "ECM") throw std::runtime_error("save METHOD must be ECM");
    if (fields.count("PARAM") && u64(fields["PARAM"], "PARAM") != 0)
        throw std::runtime_error("only Suyama PARAM=0 text saves are supported");
    std::string sigma = fields["SIGMA"];
    if (sigma.compare(0, 2, "0:") == 0) sigma.erase(0, 2);
    Record r;
    r.sigma = u64(sigma, "SIGMA");
    r.b1 = u64(fields["B1"], "B1");
    if (r.sigma < 6 || r.b1 < 2) throw std::runtime_error("SIGMA must be >=6 and B1 >=2");
    Big n, x, z, chk, factor;
    std::string err;
    if (!ecm_parse_expression(fields["N"], n.z, &err)) throw std::runtime_error("N: " + err);
    if (mpz_cmp_ui(n.z, 3) <= 0 || !mpz_odd_p(n.z) || mpz_sizeinbase(n.z, 2) > ecm_stage2::max_input_bits)
        throw std::runtime_error("N must be odd, >3 and at most 16384 bits");
    if (mpz_set_str(x.z, fields["X"].c_str(), 0) || mpz_sgn(x.z) < 0 || mpz_cmp(x.z, n.z) >= 0)
        throw std::runtime_error("X must be an integer in [0,N), decimal or 0x hex");
    if (fields.count("Z") && (mpz_set_str(z.z, fields["Z"].c_str(), 0) || mpz_cmp_ui(z.z, 1)))
        throw std::runtime_error("only normalized affine saves (Z=1 or absent) are supported");
    if (fields.count("CHECKSUM")) {
        set64(chk.z, r.b1);
        set64(factor.z, r.sigma);
        mpz_mul(chk.z, chk.z, factor.z);
        mpz_mul(chk.z, chk.z, n.z);
        mpz_mul(chk.z, chk.z, x.z);
        const uint64_t expected = mpz_fdiv_ui(chk.z, 4294967291UL);
        if (u64(fields["CHECKSUM"], "CHECKSUM") != expected)
            throw std::runtime_error("save CHECKSUM mismatch");
        r.checksum = true;
    }
    r.n = number(n.z); r.x = number(x.z);
    return r;
}
bool comment(const std::string &line) {
    const auto t = trim(line);
    return t.empty() || t[0] == '#' || t[0] == ';';
}
std::vector<Record> records(const fs::path &path, uint64_t skip, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open save: " + path.string());
    std::vector<Record> out;
    std::string line;
    uint64_t index = 0;
    while (in) {
        const auto offset = in.tellg();
        if (!std::getline(in, line)) break;
        if (comment(line)) continue;
        ++index;
        if (index <= skip) continue;
        try {
            auto r = parse_record(line);
            r.offset = static_cast<uint64_t>(offset); r.index = index; r.hash = fingerprint(line);
            out.push_back(std::move(r));
        } catch (const std::exception &e) {
            throw std::runtime_error("save record " + std::to_string(index) + ": " + e.what());
        }
        if (count && out.size() == count) break;
    }
    if (in.bad()) throw std::runtime_error("save read failed");
    if (out.empty() || (count && out.size() != count))
        throw std::runtime_error("save does not contain the requested curves");
    return out;
}
fs::path absolute_from(const fs::path &base, const std::string &p) {
    fs::path v(p);
    return fs::absolute(v.is_absolute() ? v : base / v).lexically_normal();
}
fs::path executable() {
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (!n || n == buf.size()) throw std::runtime_error("cannot locate executable");
    return fs::path(std::wstring(buf.data(), n));
}
struct Options {
    std::string ini, save, worktodo, results, log;
    uint64_t b2 = 0, d = 0, skip = 0, curves = 0, batch = 0, arena = 0;
    int device = -1, worker = 1;
    bool dry = false, once = false, selection = false, help = false;
    bool has_d = false, has_batch = false, has_arena = false;
    bool child = false;
    uint64_t offset = 0, hash = 0, index = 0;
    bool plan_only = false, tune_options = false;
    std::string tune, tune_file;
    int tune_first = 16, tune_last = 27, tune_repeats = 5;
    uint64_t tune_memory_mb = 1024;
    bool factorize_hits = false;
    bool factor_only = false;
    bool has_gp = false, has_factor_timeout = false;
    std::string gp = "gp.exe";
    unsigned factor_timeout = 30;
    bool auto_b2=false,cost_device_info=false;
    std::string cost_profile;
    uint64_t auto_min=0,auto_max=0,owner_mb=640,stage1_batch=1;
    double stage1_seconds=0,ratio_adjust=1;
    bool has_owner=false,has_stage1_seconds=false,has_stage1_batch=false,has_ratio=false;
};
double positive(const std::string &s,const char *name) {
    size_t used=0;const double value=std::stod(s,&used);
    if(used!=s.size()||!std::isfinite(value)||value<=0)throw std::runtime_error(std::string(name)+" must be finite and positive");return value;
}
Options arguments(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + a);
            return argv[i];
        };
        auto num = [&]() { return u64(value(), a.c_str()); };
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--ini" || a == "-ini") o.ini = value();
        else if (a == "--save") o.save = value();
        else if (a == "--worktodo") o.worktodo = value();
        else if (a == "--results") o.results = value();
        else if (a == "--log") o.log = value();
        else if (a == "--b2") o.b2 = num();
        else if (a == "--d") { o.d = num(); o.has_d = true; }
        else if (a == "--skip-curves") { o.skip = num(); o.selection = true; }
        else if (a == "--curves") { o.curves = num(); o.selection = true; }
        else if (a == "--batch-mb") { o.batch = num(); o.has_batch = true; }
        else if (a == "--arena-mb") { o.arena = num(); o.has_arena = true; }
        else if (a == "--device" || a == "--worker") {
            const auto n = num();
            if (n > INT_MAX || (a == "--worker" && !n)) throw std::runtime_error("invalid " + a);
            if (a == "--device") o.device = static_cast<int>(n); else o.worker = static_cast<int>(n);
        }
        else if (a == "--dry-run") o.dry = true;
        else if (a == "--plan-only") o.plan_only = true;
        else if (a == "--factorize-hits") o.factorize_hits = true;
        else if (a == "--factor-only") o.factor_only = true;
        else if (a == "--auto-b2") o.auto_b2=true;
        else if (a == "--cost-profile") o.cost_profile=value();
        else if (a == "--cost-device-info") o.cost_device_info=true;
        else if (a == "--auto-min-b2") o.auto_min=num();
        else if (a == "--auto-max-b2") o.auto_max=num();
        else if (a == "--owner-budget-mb") {o.owner_mb=num();o.has_owner=true;if(o.owner_mb>1048576)throw std::runtime_error("invalid owner budget");}
        else if (a == "--stage1-batch") {o.stage1_batch=num();o.has_stage1_batch=true;if(!o.stage1_batch||o.stage1_batch>1048576)throw std::runtime_error("invalid Stage1 batch");}
        else if (a == "--stage1-seconds-per-curve") {o.stage1_seconds=positive(value(),"Stage1 seconds");o.has_stage1_seconds=true;}
        else if (a == "--stage2-ratio-adjust") {o.ratio_adjust=positive(value(),"Stage2 ratio adjust");o.has_ratio=true;}
        else if (a == "--gp") { o.gp = value(); o.has_gp = true; }
        else if (a == "--factor-timeout") {
            const auto seconds = num();
            if (!seconds || seconds > 600) throw std::runtime_error("factor-timeout must be 1..600 seconds");
            o.factor_timeout = static_cast<unsigned>(seconds);
            o.has_factor_timeout = true;
        }
        else if (a == "--tune") o.tune = value();
        else if (a == "--tune-file") { o.tune_file = value(); o.tune_options = true; }
        else if (a == "--length-log2") {
            o.tune_options = true;
            const auto range = value();
            const auto colon = range.find(':');
            const auto first = u64(range.substr(0, colon), "length-log2");
            const auto last = colon == range.npos ? first : u64(range.substr(colon + 1), "length-log2");
            if (first < 16 || last > 27 || first > last)
                throw std::runtime_error("length-log2 must be 16..27 or FIRST:LAST");
            o.tune_first = static_cast<int>(first); o.tune_last = static_cast<int>(last);
        }
        else if (a == "--tune-repeats") {
            o.tune_options = true; const auto n = num();
            if (!n || n > 1000) throw std::runtime_error("tune-repeats must be 1..1000");
            o.tune_repeats = static_cast<int>(n);
        }
        else if (a == "--tune-memory-mb") {
            o.tune_options = true; o.tune_memory_mb = num();
            if (!o.tune_memory_mb || o.tune_memory_mb > 1048576)
                throw std::runtime_error("invalid tune memory budget in MiB");
        }
        else if (a == "--once") o.once = true;
        else if (a == "--curve-worker") o.child = true;
        else if (a == "--record-offset") o.offset = num();
        else if (a == "--record-hash") o.hash = num();
        else if (a == "--record-index") o.index = num();
        else throw std::runtime_error("unknown option: " + a);
    }
    return o;
}
struct Settings {
    uint64_t b2 = 0, d = 0, batch = 64, arena = 0;
    std::string results;
    bool factorize_hits = false;
    bool factor_only = false;
    std::string gp;
    uint64_t factor_timeout = 30;
    bool auto_b2=false,has_owner=false;
    std::string cost_profile;
    uint64_t auto_min=0,auto_max=0,owner_mb=640,stage1_batch=1;
    double stage1_seconds=0,ratio_adjust=1;
};
Settings stage2_ini(const fs::path &path, int worker) {
    std::ifstream in(path);
    std::map<std::string, std::string> global, local;
    std::string line;
    int scope = 0;
    while (std::getline(in, line)) {
        line = trim(line);
        if (comment(line)) continue;
        if (line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
        bool bracket = false;
        const int section = ecm_worktodo_parse_worker_header(line, &bracket);
        if (section) { scope = section; continue; }
        if (bracket) continue;
        const auto eq = line.find('=');
        if (eq == line.npos || (scope && scope != worker)) continue;
        const auto key = upper(trim(line.substr(0, eq)));
        (scope ? local : global)[key] = trim(line.substr(eq + 1));
    }
    for (const auto &kv : local) global[kv.first] = kv.second;
    Settings s;
    auto get = [&](const char *key, uint64_t &target) {
        if (global.count(key)) target = u64(global[key], key);
    };
    get("STAGE2_B2", s.b2); get("STAGE2_D", s.d);
    get("STAGE2_BATCH_MB", s.batch); get("STAGE2_ARENA_MB", s.arena);
    uint64_t factorize=0; get("STAGE2_FACTORIZE_HITS",factorize);
    if(factorize>1)throw std::runtime_error("stage2_factorize_hits must be 0 or 1");
    s.factorize_hits=factorize!=0;
    uint64_t factor_only=0; get("STAGE2_FACTOR_ONLY",factor_only);
    if(factor_only>1)throw std::runtime_error("stage2_factor_only must be 0 or 1");
    s.factor_only=factor_only!=0;
    uint64_t auto_b2=0;get("STAGE2_AUTO_B2",auto_b2);
    if(auto_b2>1)throw std::runtime_error("stage2_auto_b2 must be 0 or 1");s.auto_b2=auto_b2!=0;
    if(global.count("STAGE2_COST_PROFILE"))s.cost_profile=global["STAGE2_COST_PROFILE"];
    get("STAGE2_AUTO_MIN_B2",s.auto_min);get("STAGE2_AUTO_MAX_B2",s.auto_max);
    if(global.count("STAGE2_FOLD_MB")){get("STAGE2_FOLD_MB",s.owner_mb);s.has_owner=true;}
    if(s.owner_mb>1048576)throw std::runtime_error("invalid stage2_fold_mb");
    get("STAGE1_BATCH",s.stage1_batch);if(!s.stage1_batch||s.stage1_batch>1048576)throw std::runtime_error("invalid stage1_batch");
    if(global.count("STAGE1_SECONDS_PER_CURVE"))s.stage1_seconds=positive(global["STAGE1_SECONDS_PER_CURVE"],"Stage1 seconds");
    if(global.count("STAGE2_RATIO_ADJUST"))s.ratio_adjust=positive(global["STAGE2_RATIO_ADJUST"],"Stage2 ratio adjust");
    get("STAGE2_FACTOR_TIMEOUT",s.factor_timeout);
    if(!s.factor_timeout || s.factor_timeout>600)throw std::runtime_error("stage2_factor_timeout must be 1..600");
    if(global.count("STAGE2_GP"))s.gp=global["STAGE2_GP"];
    if (global.count("STAGE2_RESULTS_FILE")) s.results = global["STAGE2_RESULTS_FILE"];
    return s;
}
std::vector<std::string> csv(const std::string &s, std::vector<bool> *quotes = nullptr) {
    std::vector<std::string> out;
    std::string field;
    bool quoted = false, had_quote = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '"') {
            had_quote = true;
            if (quoted && i + 1 < s.size() && s[i + 1] == '"') { field += c; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) {
            out.push_back(trim(field)); field.clear();
            if (quotes) quotes->push_back(had_quote);
            had_quote = false;
        }
        else field += c;
    }
    if (quoted) throw std::runtime_error("unclosed worktodo quote");
    out.push_back(trim(field));
    if (quotes) quotes->push_back(had_quote);
    return out;
}
struct QueueSelection {
    EcmStage2Task task;
    uint64_t b2 = 0, skip = 0, count = 0;
};
QueueSelection queue_fields(std::string line) {
    if (line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    line = trim(line);
    const std::string prefix = "ECMSTAGE2=";
    if (line.compare(0, prefix.size(), prefix))
        throw std::runtime_error("worktodo must contain ECMSTAGE2 save tasks");
    std::vector<bool> quotes;
    const auto cols = csv(line.substr(prefix.size()), &quotes);
    auto integer_text = [](const std::string &s) {
        size_t i = (!s.empty() && (s[0] == '+' || s[0] == '-')) ? 1 : 0;
        if (i == s.size()) return false;
        for (; i < s.size(); ++i) if (s[i] < '0' || s[i] > '9') return false;
        return true;
    };
    QueueSelection q;
    size_t idx = integer_text(cols.front()) ? 0 : 1;
    if (idx) q.task.aid = cols.front();
    if (cols.size() < idx + 5) throw std::runtime_error("ECMSTAGE2 needs k,b,n,c,filename");
    q.task.raw_line = line;
    q.task.k = cols[idx]; q.task.b = cols[idx + 1]; q.task.c = cols[idx + 3];
    const uint64_t exponent = u64(cols[idx + 2], "worktodo exponent");
    if (exponent > ecm_stage2::max_input_bits) throw std::runtime_error("worktodo exponent exceeds supported input range");
    q.task.n = static_cast<unsigned long>(exponent);
    q.task.save_name = cols[idx + 4];
    if (q.task.save_name.empty()) throw std::runtime_error("worktodo save filename is empty");
    idx += 5;
    uint64_t *targets[] = {&q.b2, &q.skip, &q.count};
    const char *labels[] = {"worktodo B2", "worktodo skip_curves", "worktodo num_curves"};
    for (size_t i = 0; i < 3 && idx < cols.size() && !quotes[idx]; ++i, ++idx)
        *targets[i] = cols[idx].empty() ? 0 : u64(cols[idx], labels[i]);
    if (q.count > UINT32_MAX) throw std::runtime_error("worktodo num_curves exceeds uint32");
    q.task.curves_to_run = static_cast<uint32_t>(q.count);
    if (idx < cols.size()) {
        if (!quotes[idx] || idx + 1 != cols.size())
            throw std::runtime_error("extra worktodo fields; known factors must be the final quoted field");
        for (const auto &factor : csv(cols[idx])) if (!factor.empty()) q.task.factors.push_back(factor);
    }
    return q;
}
void append(const fs::path &path, const std::string &line) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out << line << '\n'; out.flush();
    if (!out) throw std::runtime_error("cannot append: " + path.string());
}
std::wstring quote(const std::wstring &s) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : s) {
        if (c == L'\\') { ++slashes; continue; }
        out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0; out += c;
    }
    out.append(2 * slashes, L'\\');
    return out + L'"';
}
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    ~Handle() { if (value != INVALID_HANDLE_VALUE && value) CloseHandle(value); }
};
// Optional diagnostic for the calling process's CUDA context. Use the same
// linked runtime as the CUDA translation unit.
// The schedule bits affect CPU waits, not kernel ordering or arithmetic.
void configure_cuda_wait(int device) {
    const char *value=std::getenv("NTT_CUDA_WAIT_MODE");
    if(!value)return;
    const uint64_t mode=ecm_stage2::cost::integer(value);
    if(mode!=0&&mode!=1&&mode!=2&&mode!=4)
        throw std::runtime_error("NTT_CUDA_WAIT_MODE must be 0(auto), 1(spin), 2(yield), or 4(blocking)");
    auto check=[](cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string("CUDA wait configuration failed: ")+cudaGetErrorString(error));};
    check(cudaSetDevice(device));
    unsigned before=0,after=0;check(cudaGetDeviceFlags(&before));
    // MapHost (bit 3) is implicit in CUDA 13's returned flags and must not be
    // passed to cudaSetDeviceFlags. Preserve all other non-schedule flags.
    check(cudaSetDeviceFlags((before&~15u)|static_cast<unsigned>(mode)));check(cudaGetDeviceFlags(&after));
    if((after&7u)!=mode||(after&~7u)!=(before&~7u))
        throw std::runtime_error("CUDA wait flags did not match requested context state");
    std::cout<<"stage2_cuda_wait: device="<<device<<" requested="<<mode
             <<" before="<<before<<" after="<<after<<std::endl;
}
int child_run(const Options &o, const fs::path &save, const Record &r,
              uint64_t b2, uint64_t d, int device, const fs::path &results, const fs::path &log) {
    const fs::path exe = executable();
    std::wstring cmd = quote(exe.wstring()) + L" --curve-worker --save " + quote(save.wstring());
    auto arg = [&](const wchar_t *key, uint64_t n) { cmd += L" "; cmd += key; cmd += L" "; cmd += std::to_wstring(n); };
    arg(L"--record-offset", r.offset); arg(L"--record-hash", r.hash); arg(L"--record-index", r.index);
    arg(L"--b2", b2); arg(L"--d", d); arg(L"--device", device); arg(L"--worker", o.worker);
    cmd += L" --results " + quote(results.wstring());
    if (o.factorize_hits) {
        cmd += L" --factorize-hits --gp " + quote(fs::path(o.gp).wstring());
        arg(L"--factor-timeout", o.factor_timeout);
    }
    if (o.factor_only) cmd += L" --factor-only";
    if (o.auto_b2 && !b2) {
        auto real=[](double value){std::wostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)<<value;return out.str();};
        cmd += L" --auto-b2 --cost-profile " + quote(fs::path(o.cost_profile).wstring());
        arg(L"--arena-mb",o.arena);arg(L"--owner-budget-mb",o.owner_mb);arg(L"--stage1-batch",o.stage1_batch);
        if(o.auto_min)arg(L"--auto-min-b2",o.auto_min);if(o.auto_max)arg(L"--auto-max-b2",o.auto_max);
        if(o.stage1_seconds)cmd+=L" --stage1-seconds-per-curve "+real(o.stage1_seconds);
        cmd+=L" --stage2-ratio-adjust "+real(o.ratio_adjust);
    }
    STARTUPINFOW si{}; si.cb = sizeof(si);
    Handle logfile;
    if (!log.empty()) {
        fs::create_directories(log.parent_path());
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        logfile.value = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (logfile.value == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open engine log");
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = si.hStdError = logfile.value;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, !log.empty(), 0, nullptr, nullptr, &si, &pi))
        throw std::runtime_error("cannot launch curve process: " + std::to_string(GetLastError()));
    Handle process, thread; process.value = pi.hProcess; thread.value = pi.hThread;
    if (WaitForSingleObject(pi.hProcess, INFINITE) != WAIT_OBJECT_0)
        throw std::runtime_error("curve process wait failed");
    DWORD code = 1;
    if (!GetExitCodeProcess(pi.hProcess, &code)) throw std::runtime_error("cannot read curve exit status");
    return code == 0 ? 0 : 1;
}
std::string select_auto(Options &o,const Record &r,bool apply=true) {
    namespace c=ecm_stage2::cost;
    if(o.cost_profile.empty())throw std::runtime_error("Auto B2 requires --cost-profile FILE");
    Handle guard;guard.value=CreateFileW(fs::path(o.cost_profile).c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(guard.value==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot lock cost profile for reading");
    const std::string hash=ecm_stage2::sha256_file(fs::path(o.cost_profile));
    const auto profile=c::Profile::load(fs::path(o.cost_profile));
    if(profile.binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("cost profile binary fingerprint mismatch; recalibration required");
    // Bind only the measured configuration; explicit conflicting environment fails.
    auto require=[](const char *key,uint64_t expected,uint64_t fallback) {
        const char *value=std::getenv(key);const uint64_t actual=value?ecm_stage2::cost::integer(value):fallback;
        if(actual!=expected)throw std::runtime_error(std::string("cost profile configuration mismatch: ")+key);
    };
    require("NTT_CUDA_WAIT_MODE",0,0);
    for(const char *key:{"NTT_XADD6","NTT_BABY_DEVICE","NTT_POINT_MERSENNE","NTT_SMALL_PRIME_REUSE","NTT_GIANT_SEED_DEVICE",
        "NTT_GFINV_SEG_EXACT","NTT_GFINV_BATCH","NTT_FOLD_FLAT","NTT_FOLD_DEVICE","NTT_GROOT_DEVICE","NTT_SCALED_DESCENT",
        "NTT_S4_OUTPUT_WINDOW","NTT_S4_CHUNK_OUTPUT","NTT_DEVICE_GLEAF","NTT_GROOT_TO_FOLD","NTT_S4_ORACLE_ASYNC",
        "NTT_S4_CARRY_BATCH","NTT_FUSE_WARP_TAIL"})require(key,1,1);
    for(const char *key:{"NTT_GIANT_LADDER","NTT_GL_SHIFT_SCALE","NTT_CARRY_CHECK_FUSED","NTT_S4_HOSTPACK","NTT_S4_FINAL_READBACK"})require(key,0,0);
    require("NTT_FUSE_COOP_OUTER",2,2);require("NTT_FUSE_T",12,12);require("NTT_FUSE_M",4,4);
    require("NTT_S4_BATCH_MB",64,64);require("NTT_DEVICE_GLEAF_MAX_MB",512,512);
    require("NTT_GIANT_CHAIN_BLOCK",64,64);require("NTT_GIANT_CHAIN_MIN",profile.chain_min,profile.chain_min);
    require("NTT_GIANT_CHAIN_SMALL_BLOCK",0,0); // Not covered by current v2 cost profiles.
    require("NTT_GIANT_SEED_PAIR",0,0); // Paired/rebased seeds require new measured costs.
    require("NTT_GIANT_BASE_CPU",0,0);
    require("NTT_LADDER_CAP",8192,8192);
    require("CUDA_LAUNCH_BLOCKING",0,0);
    require("NTT_S4_SAMPLE",96,96);require("NTT_S4_CHECK_EVERY",8,8);
    require("NTT_ARENA_WORKSPACE_POOL",1,1);require("NTT_FUSE_COMPACT_SCRATCH",1,1);
    require("NTT_S4_FLAT_DIRECT",1,1);require("NTT_GROOT_COMPACT_RAW",1,1);
    if(o.factor_only)_putenv_s("NTT_NAME_HITS","0");
    const char *naming=std::getenv("NTT_NAME_HITS");
    if(naming)require("NTT_NAME_HITS",profile.naming,profile.naming);
    else _putenv_s("NTT_NAME_HITS",profile.naming?"1":"0");
    if(!profile.naming)o.factor_only=true;
    Big n;mpz_set_str(n.z,r.n.c_str(),16);const int bits=(int)mpz_sizeinbase(n.z,2);
    if(bits<2||bits>8192||mpz_popcount(n.z)!=(mp_bitcnt_t)bits)throw std::runtime_error("cost profile requires exact Mersenne input");
    uint64_t arena=o.arena;
    if(!arena)if(const char *v=std::getenv("NTT_ARENA_CAP_KB")) {
        const auto kb=c::integer(v);if(!kb||kb%1024)throw std::runtime_error("cost profile requires an explicit whole-MiB arena scope");arena=kb/1024;
    }
    if(std::none_of(profile.scopes.begin(),profile.scopes.end(),[&](const c::Scope &s){return s.bits==(uint64_t)bits&&s.b1==r.b1&&(!arena||s.arena_mb==arena);}))
        throw std::runtime_error("no measured bit-width/B1/arena scope");
    EcmStage2DeviceInfo device;
    if(ecm_cuda_stage2_device_info(o.device,&device,profile.uuid.c_str()))throw std::runtime_error("cannot query matching Auto B2 device");
    c::Request request;request.bits=bits;request.b1=r.b1;request.d=o.d;request.arena_mb=arena;request.owner_mb=o.owner_mb;
    request.b2min=o.auto_min;request.b2max=o.auto_max;request.batch=o.stage1_batch;request.t1=o.stage1_seconds;request.adjust=o.ratio_adjust;
    const auto plan=c::choose(profile,request,device);o.b2=plan.b2;o.d=plan.d;o.arena=plan.arena_mb;o.owner_mb=plan.owner_mb;
    if(apply){
        _putenv_s("NTT_GIANT_CHAIN_MIN",std::to_string(profile.chain_min).c_str());
        _putenv_s("NTT_ARENA_CAP_KB",std::to_string(o.arena*1024).c_str());
        _putenv_s("NTT_FOLD_DEVICE_MAX_MB",std::to_string(o.owner_mb).c_str());
        _putenv_s("NTT_D_MODEL","0");
    }
    return c::json(plan,hash);
}
int curve_worker(Options o) {
    if (o.factor_only && _putenv_s("NTT_NAME_HITS", "0"))
        throw std::runtime_error("cannot disable optional prime-witness naming");
    if (o.save.empty() || o.results.empty() || !o.index || o.device < 0 || (!o.b2 && !o.auto_b2))
        throw std::runtime_error("incomplete internal curve-worker arguments");
    std::ifstream in(o.save, std::ios::binary);
    if (!in) throw std::runtime_error("cannot reread save");
    in.seekg(static_cast<std::streamoff>(o.offset));
    std::string line;
    if (!std::getline(in, line) || fingerprint(line) != o.hash)
        throw std::runtime_error("save changed after planning");
    const Record r = parse_record(line);
    std::string result;
    const uint64_t requested_d=o.d,requested_b2=o.b2;
    const auto start = std::chrono::steady_clock::now();
    std::string auto_json;
    if(!o.b2&&o.auto_b2){auto_json=select_auto(o,r);std::cout<<auto_json<<std::endl;}
    const double auto_seconds=auto_json.empty()?0:std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (o.b2 <= r.b1) throw std::runtime_error("B2 must be greater than saved B1");
    Big n, x, gcd;
    mpz_set_str(n.z, r.n.c_str(), 16); mpz_set_str(x.z, r.x.c_str(), 16);
    mpz_gcd(gcd.z, x.z, n.z);
    std::string status = "stage2_completed";
    int code = 0;
    if (mpz_cmp_ui(gcd.z, 1) > 0 && mpz_cmp(gcd.z, n.z) < 0) {
        status = "factor_in_saved_X";
        result = "\"hits\":1,\"bad_factors\":0,\"factors\":[" + json_string(number(gcd.z, 10)) + "]";
        std::cout << "saved_X_factor: " << number(gcd.z, 10) << '\n';
    } else {
        if (!mpz_cmp(gcd.z, n.z)) throw std::runtime_error("saved X=0 gives no usable Stage1 point");
        configure_cuda_wait(o.device);
        code = ecm_cuda_stage2_run(r.n.c_str(), r.x.c_str(), r.sigma, r.b1, o.b2, o.d, o.device,
            [](const char *text, void *ctx) { *static_cast<std::string *>(ctx) = text; }, &result);
    }
    if (code || result.empty()) return code ? code : 1;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double factor_seconds = 0;
    if (o.factorize_hits) {
        const auto begin = std::chrono::steady_clock::now();
        result += ',' + ecm_stage2::factor_details(result, n.z, fs::path(o.gp), o.factor_timeout,
                                                   fs::path(o.results).parent_path() / "factor_details");
        factor_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    append(o.results, "{\"status\":" + json_string(status) + ",\"save\":" + json_string(o.save) +
        ",\"record\":" + std::to_string(o.index) + ",\"record_hash\":" + json_string(std::to_string(o.hash)) +
        ",\"worker\":" + std::to_string(o.worker) + ",\"device\":" + std::to_string(o.device) +
        ",\"N_hex\":" + json_string(r.n) + ",\"sigma\":" + std::to_string(r.sigma) +
        ",\"B1\":" + std::to_string(r.b1) + ",\"B2\":" + std::to_string(o.b2) +
        ",\"requested_D\":" + std::to_string(requested_d) + ",\"seconds\":" + std::to_string(seconds) +
        ",\"factorization_seconds\":" + std::to_string(factor_seconds) +
        ",\"param\":0" +
        ",\"requested_factor_only\":" + std::string(o.factor_only ? "true" : "false") +
        ",\"auto_b2\":" + std::string(auto_json.empty() ? "false" : "true") +
        ",\"requested_B2\":" + std::to_string(requested_b2) + ",\"auto_planning_seconds\":" + std::to_string(auto_seconds) +
        (auto_json.empty() ? "" : ",\"auto_plan\":" + auto_json) +
        ",\"timestamp_ms\":" + std::to_string(timestamp) + "," + result + "}");
    return 0;
}
void help() {
    std::cout << "ecm_cuda_stage2 - resume normalized Suyama PARAM=0 Stage1 text saves\n"
        "  ecm_cuda_stage2 --save FILE --b2 B2 [--curves N] [--skip-curves N]\n"
        "  ecm_cuda_stage2 [--ini ecm.ini] [--worktodo FILE] [--worker N] [--once]\n"
        "Options: --device N --d D --batch-mb MB --arena-mb MB --results FILE\n"
        "         --log FILE --dry-run --help\n"
        "         --factorize-hits [--gp gp.exe] [--factor-timeout 30]\n"
        "         --factor-only (skip optional prime-witness naming; raw factors may be composite)\n"
        "Auto B2: --auto-b2 --cost-profile FILE [--stage1-batch N]\n"
        "         [--stage1-seconds-per-curve S] [--stage2-ratio-adjust R]\n"
        "         [--auto-min-b2 B2 --auto-max-b2 B2] [--owner-budget-mb MB]\n"
        "         Requires a matching measured runtime profile; no extrapolation.\n"
        "         --plan-only (queries device memory and D; runs no curve)\n"
        "Tune: --tune ntt --device N [--length-log2 16:27] [--tune-repeats 5]\n"
        "      [--tune-memory-mb 1024] [--tune-file stage2_tune.jsonl]\n"
        "      Requires a fixed Goldilocks backend; measures field convolution only.\n"
        "Queue: ECMSTAGE2=[AID,]k,b,n,c,save[,B2-or-zero][,skip][,count][,\"factors\"]\n"
        "INI: worktodo, finished, tmp_dir, log_file, device; stage2_b2, stage2_d,\n"
        "     stage2_batch_mb, stage2_arena_mb, stage2_results_file; [Worker #N].\n"
        "D=0 chooses automatically; count/curves=0 means all remaining.\n"
        "Missing implicit INI uses defaults. No Stage1 computation or checkpointing.\n";
}
int driver(Options o) {
    if (o.help) { help(); return 0; }
    if (o.child) return curve_worker(o);
    if(o.auto_b2&&o.b2)throw std::runtime_error("explicit --auto-b2 conflicts with nonzero --b2");
    if(o.cost_device_info) {
        if(!o.save.empty()||!o.worktodo.empty()||!o.tune.empty()||o.auto_b2||o.plan_only||o.dry||o.b2)
            throw std::runtime_error("--cost-device-info is independent of curve/queue options");
        EcmStage2DeviceInfo info;if(ecm_cuda_stage2_device_info(o.device<0?0:o.device,&info))throw std::runtime_error("cannot query cost-profile device");
        std::cout<<"{\"type\":\"cost_device\",\"uuid_hex\":\""<<info.uuid_hex<<"\",\"major\":"<<info.major<<",\"minor\":"<<info.minor
            <<",\"runtime\":"<<info.runtime<<",\"driver\":"<<info.driver<<",\"fixed_mode\":"<<info.fixed_mode<<",\"outer_unroll_u\":"<<info.outer_unroll_u
            <<",\"free_bytes\":"<<info.free_bytes<<",\"total_bytes\":"<<info.total_bytes<<"}"<<std::endl;return 0;
    }
    if (o.tune.empty() && o.tune_options) throw std::runtime_error("tune options require --tune ntt");
    if (!o.tune.empty() && (o.tune != "ntt" || !o.save.empty() || !o.worktodo.empty() ||
        o.b2 || o.has_d || o.selection || o.dry || o.plan_only || o.once || o.auto_b2 || !o.cost_profile.empty()))
        throw std::runtime_error("--tune ntt is independent of save/queue/curve planning options");
    if (o.plan_only && o.dry) throw std::runtime_error("choose --plan-only or --dry-run");
    const fs::path cwd = fs::current_path();
    const fs::path ini = o.ini.empty() ? executable().parent_path() / "ecm.ini" : absolute_from(cwd, o.ini);
    EcmQueueConfig cfg;
    if (!ecm_queue_config_load(ini.string(), o.worker, cfg) && !o.ini.empty())
        throw std::runtime_error("cannot read explicit ini: " + ini.string());
    Settings s = stage2_ini(ini, o.worker);
    if(s.factorize_hits)o.factorize_hits=true;
    if(s.factor_only)o.factor_only=true;
    o.auto_b2=o.auto_b2||s.auto_b2;
    if(!o.auto_min)o.auto_min=s.auto_min;if(!o.auto_max)o.auto_max=s.auto_max;
    if(!o.has_owner){if(s.has_owner)o.owner_mb=s.owner_mb;else if(const char *v=std::getenv("NTT_FOLD_DEVICE_MAX_MB"))o.owner_mb=ecm_stage2::cost::integer(v);}
    if(o.owner_mb>1048576)throw std::runtime_error("invalid owner budget");
    if(!o.has_stage1_batch)o.stage1_batch=s.stage1_batch;
    if(!o.has_stage1_seconds)o.stage1_seconds=s.stage1_seconds;
    if(!o.has_ratio)o.ratio_adjust=s.ratio_adjust;
    if(!o.has_gp && !s.gp.empty())o.gp=s.gp;
    if(!o.has_factor_timeout)o.factor_timeout=static_cast<unsigned>(s.factor_timeout);
    const fs::path base = ini.parent_path();
    if(!o.cost_profile.empty())o.cost_profile=absolute_from(cwd,o.cost_profile).string();
    else if(!s.cost_profile.empty())o.cost_profile=absolute_from(base,s.cost_profile).string();
    const fs::path worktodo = o.worktodo.empty() ? absolute_from(base, cfg.worktodo) : absolute_from(cwd, o.worktodo);
    const fs::path finished = absolute_from(base, cfg.finished);
    const fs::path tmp = absolute_from(base, cfg.tmp_dir);
    const std::string suffix = o.worker == 1 ? "" : "_" + std::to_string(o.worker);
    const fs::path results = !o.results.empty() ? absolute_from(cwd, o.results) :
        absolute_from(base, s.results.empty() ? "stage2_results" + suffix + ".jsonl" : s.results);
    fs::path log;
    if (!o.log.empty()) log = absolute_from(cwd, o.log);
    else if (!cfg.log_file_explicit) log = absolute_from(base, "stage2_screen" + suffix + ".log");
    else if (!cfg.log_file.empty()) log = absolute_from(base, cfg.log_file);
    const int device = o.device < 0 ? cfg.device : o.device;
    const uint64_t d = o.has_d ? o.d : s.d;
    o.device=device;o.d=d;
    if (device < 0 || (d && (d < 6 || d % 2))) throw std::runtime_error("device must be >=0; D must be even and >=6");
    const uint64_t batch = o.has_batch ? o.batch : s.batch, arena = o.has_arena ? o.arena : s.arena;
    o.arena=arena;
    if (!batch || batch > 1048576 || arena > 1048576) throw std::runtime_error("invalid Stage2 memory budget in MB");
    _putenv_s("NTT_S4_BATCH_MB", std::to_string(batch).c_str());
    if (arena) _putenv_s("NTT_ARENA_CAP_KB", std::to_string(arena * 1024).c_str());
    if (!o.tune.empty()) {
        const fs::path destination = absolute_from(cwd, o.tune_file.empty() ? "stage2_tune.jsonl" : o.tune_file);
        auto same_path = [&](const fs::path &other) {
            std::error_code error;
            if (fs::equivalent(destination, other, error) && !error) return true;
            return upper(destination.lexically_normal().string()) == upper(other.lexically_normal().string());
        };
        if (upper(destination.extension().string()) != ".JSONL")
            throw std::runtime_error("tune-file must use the .jsonl extension");
        if (same_path(executable()) || same_path(ini) || same_path(worktodo) || same_path(finished) ||
            same_path(results) || (!log.empty() && same_path(log)))
            throw std::runtime_error("tune-file must differ from executable, config, queue and result files");
        if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
        const fs::path partial(destination.string() + ".partial." + std::to_string(GetCurrentProcessId()));
        std::ofstream output(partial, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write tune-file: " + partial.string());
        const auto binary_hash = ecm_stage2::sha256_file(executable());
        const auto manifest = executable().parent_path() / "build_manifest.json";
        const auto manifest_hash = fs::is_regular_file(manifest) ? ecm_stage2::sha256_file(manifest) : "";
        output << "{\"type\":\"profile\",\"schema\":1,\"unit\":\"field_convolution\",\"binary_sha256\":"
               << json_string(binary_hash) << ",\"build_manifest_sha256\":" << json_string(manifest_hash)
               << ",\"min_log2\":" << o.tune_first << ",\"max_log2\":" << o.tune_last
               << ",\"repeats\":" << o.tune_repeats << "}\n";
        auto sink = [](const char *json, void *ctx) {
            auto &out = *static_cast<std::ofstream *>(ctx);
            out << json << '\n'; out.flush();
            if (!out) throw std::runtime_error("tune profile write failed");
            std::cout << json << std::endl;
        };
        const int code = ecm_cuda_stage2_tune_ntt(device, o.tune_first, o.tune_last, o.tune_repeats,
                                                 o.tune_memory_mb * 1048576, sink, &output);
        if (code) throw std::runtime_error("tune failed or measured no shapes; partial profile retained: " + partial.string());
        if (binary_hash != ecm_stage2::sha256_file(executable()) ||
            (!manifest_hash.empty() && manifest_hash != ecm_stage2::sha256_file(manifest)))
            throw std::runtime_error("tune binary/build manifest changed; partial profile retained");
        output.flush(); if (!output) throw std::runtime_error("tune profile flush failed");
        output.close(); if (!output) throw std::runtime_error("tune profile close failed");
        if (!MoveFileExW(partial.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot publish tune profile; partial profile retained: " + partial.string());
        std::cout << "tune_complete: profile=" << destination.string() << std::endl;
        return 0;
    }
    const bool queue = o.save.empty();
    if (queue && o.selection) throw std::runtime_error("queue curve selection comes from ECMSTAGE2; use --save for overrides");
    Handle lock;
    if (queue) {
        if (!fs::exists(worktodo)) throw std::runtime_error("worktodo not found: " + worktodo.string());
        if (!o.dry && !o.plan_only) {
            // One production consumer per queue file also serializes the shared
            // atomic rewrite across worker sections. Use separate queues for
            // concurrent processes in this minimal driver.
            fs::path lockpath(worktodo.string() + ".stage2.lock");
            lock.value = CreateFileW(lockpath.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
            if (lock.value == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot lock worktodo worker (another process may own it)");
        }
    }
    uint64_t completed = 0;
    for (;;) {
        fs::path save;
        std::string task_line, expected_n;
        uint64_t skip = o.skip, count = o.curves, b2 = o.b2 ? o.b2 : s.b2;
        if (queue) {
            if (!ecm_worktodo_first_line(worktodo.string(), o.worker, task_line)) break;
            const auto fields = queue_fields(task_line);
            const auto &task = fields.task;
            std::string err;
            skip = fields.skip; count = fields.count;
            if (!o.b2 && fields.b2) b2 = fields.b2;
            std::cout << "queue_fields: filename=" << task.save_name << " B2=" << fields.b2
                      << " skip_curves=" << skip << " num_curves=" << count << std::endl;
            Big n;
            if (!ecm_compute_stage2_n(task, n.z, err)) throw std::runtime_error("worktodo N: " + err);
            expected_n = number(n.z);
            save = absolute_from(tmp, task.save_name);
        } else save = absolute_from(cwd, o.save);
        if(o.auto_b2&&!b2&&o.cost_profile.empty())throw std::runtime_error("Auto B2 requires --cost-profile FILE");
        const auto plan = records(save, skip, count);
        for (const auto &r : plan) {
            if ((!(o.auto_b2&&!b2)&&b2 <= r.b1) || b2 > static_cast<uint64_t>(INT64_MAX) - 8192)
                throw std::runtime_error("B2 must exceed every saved B1 and fit the engine's signed index range");
            if (!expected_n.empty() && r.n != expected_n) throw std::runtime_error("worktodo N differs from save N");
            if (queue && r.b1 != plan.front().b1) throw std::runtime_error("queue saves must have the same B1");
        }
        std::cout << "stage2_plan: save=" << save.string() << " curves=" << plan.size()
                  << " B2=" << b2 << " D=" << d << " device=" << device << " worker=" << o.worker
                  << " auto_b2=" << (o.auto_b2&&!b2 ? 1 : 0)
                  << " log=" << log.string() << " results=" << results.string() << '\n';
        for (const auto &r : plan) {
            std::cout << "curve_start: record=" << r.index << " sigma=" << r.sigma << " B1=" << r.b1
                      << " checksum=" << (r.checksum ? "verified" : "absent") << std::endl;
            if (o.dry) continue;
            if (o.plan_only) {
                if(o.auto_b2&&!b2){Options local=o;local.b2=0;std::cout<<select_auto(local,r,false)<<std::endl;continue;}
                std::string result;
                const int code = ecm_cuda_stage2_plan(r.n.c_str(), r.sigma, r.b1, b2, d, device,
                    [](const char *json, void *ctx) { *static_cast<std::string *>(ctx) = json; }, &result);
                if (code || result.empty()) throw std::runtime_error("Stage2 planning failed");
                std::cout << result << std::endl;
                continue;
            }
            if (child_run(o, save, r, b2, d, device, results, log))
                throw std::runtime_error("curve failed; queue retained; inspect " + log.string());
            ++completed;
            std::cout << "curve_done: record=" << r.index << " sigma=" << r.sigma << std::endl;
        }
        if (!queue || o.dry || o.plan_only) break;
        std::string current;
        if (!ecm_worktodo_first_line(worktodo.string(), o.worker, current) || current != task_line)
            throw std::runtime_error("task completed but worktodo changed; queue retained");
        append(finished, task_line);
        if (!ecm_worktodo_advance(worktodo.string(), o.worker, task_line, WorktodoAction::Remove))
            throw std::runtime_error("task completed but worktodo changed or could not be advanced");
        if (o.once) break;
    }
    std::cout << (o.plan_only ? "plan_complete" : o.dry ? "dry_run_complete" : "stage2_complete")
              << ": curves=" << completed << '\n';
    return 0;
}
} // namespace s2prod

int main(int argc, char **argv) {
    try { return s2prod::driver(s2prod::arguments(argc, argv)); }
    catch (const std::exception &e) { std::cerr << "ecm_cuda_stage2: " << e.what() << '\n'; return 2; }
}
