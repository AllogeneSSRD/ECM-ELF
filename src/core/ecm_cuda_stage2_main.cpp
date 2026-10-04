// Minimal save -> CUDA Stage2 driver. Queue edits happen only after successful
// child exit; a fresh child isolates the experimental engine's per-curve globals.
#define NOMINMAX
#include <windows.h>
#include "ecm_cuda_stage2.h"
#include "ecm_expr.h"
#include "ecm_queue_config.h"
#include "ecm_worktodo.h"
#include <algorithm>
#include <chrono>
#include <cctype>
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
    if (mpz_cmp_ui(n.z, 3) <= 0 || !mpz_odd_p(n.z) || mpz_sizeinbase(n.z, 2) > 8192)
        throw std::runtime_error("N must be odd, >3 and at most 8192 bits");
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
};
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
    if (exponent > 8192) throw std::runtime_error("worktodo exponent exceeds supported input range");
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
int child_run(const Options &o, const fs::path &save, const Record &r,
              uint64_t b2, uint64_t d, int device, const fs::path &results, const fs::path &log) {
    const fs::path exe = executable();
    std::wstring cmd = quote(exe.wstring()) + L" --curve-worker --save " + quote(save.wstring());
    auto arg = [&](const wchar_t *key, uint64_t n) { cmd += L" "; cmd += key; cmd += L" "; cmd += std::to_wstring(n); };
    arg(L"--record-offset", r.offset); arg(L"--record-hash", r.hash); arg(L"--record-index", r.index);
    arg(L"--b2", b2); arg(L"--d", d); arg(L"--device", device); arg(L"--worker", o.worker);
    cmd += L" --results " + quote(results.wstring());
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
int curve_worker(const Options &o) {
    if (o.save.empty() || o.results.empty() || !o.index || o.device < 0 || !o.b2)
        throw std::runtime_error("incomplete internal curve-worker arguments");
    std::ifstream in(o.save, std::ios::binary);
    if (!in) throw std::runtime_error("cannot reread save");
    in.seekg(static_cast<std::streamoff>(o.offset));
    std::string line;
    if (!std::getline(in, line) || fingerprint(line) != o.hash)
        throw std::runtime_error("save changed after planning");
    const Record r = parse_record(line);
    if (o.b2 <= r.b1) throw std::runtime_error("B2 must be greater than saved B1");
    std::string result;
    const auto start = std::chrono::steady_clock::now();
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
        code = ecm_cuda_stage2_run(r.n.c_str(), r.x.c_str(), r.sigma, r.b1, o.b2, o.d, o.device,
            [](const char *text, void *ctx) { *static_cast<std::string *>(ctx) = text; }, &result);
    }
    if (code || result.empty()) return code ? code : 1;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    append(o.results, "{\"status\":" + json_string(status) + ",\"save\":" + json_string(o.save) +
        ",\"record\":" + std::to_string(o.index) + ",\"record_hash\":" + json_string(std::to_string(o.hash)) +
        ",\"worker\":" + std::to_string(o.worker) + ",\"device\":" + std::to_string(o.device) +
        ",\"N_hex\":" + json_string(r.n) + ",\"sigma\":" + std::to_string(r.sigma) +
        ",\"B1\":" + std::to_string(r.b1) + ",\"B2\":" + std::to_string(o.b2) +
        ",\"requested_D\":" + std::to_string(o.d) + ",\"seconds\":" + std::to_string(seconds) +
        ",\"timestamp_ms\":" + std::to_string(timestamp) + "," + result + "}");
    return 0;
}
void help() {
    std::cout << "ecm_cuda_stage2 - resume normalized Suyama PARAM=0 Stage1 text saves\n"
        "  ecm_cuda_stage2 --save FILE --b2 B2 [--curves N] [--skip-curves N]\n"
        "  ecm_cuda_stage2 [--ini ecm.ini] [--worktodo FILE] [--worker N] [--once]\n"
        "Options: --device N --d D --batch-mb MB --arena-mb MB --results FILE\n"
        "         --log FILE --dry-run --help\n"
        "Queue: ECMSTAGE2=[AID,]k,b,n,c,save[,B2-or-zero][,skip][,count][,\"factors\"]\n"
        "INI: worktodo, finished, tmp_dir, log_file, device; stage2_b2, stage2_d,\n"
        "     stage2_batch_mb, stage2_arena_mb, stage2_results_file; [Worker #N].\n"
        "D=0 chooses automatically; count/curves=0 means all remaining.\n"
        "Missing implicit INI uses defaults. No Stage1 computation or checkpointing.\n";
}
int driver(Options o) {
    if (o.help) { help(); return 0; }
    if (o.child) return curve_worker(o);
    const fs::path cwd = fs::current_path();
    const fs::path ini = o.ini.empty() ? executable().parent_path() / "ecm.ini" : absolute_from(cwd, o.ini);
    EcmQueueConfig cfg;
    if (!ecm_queue_config_load(ini.string(), o.worker, cfg) && !o.ini.empty())
        throw std::runtime_error("cannot read explicit ini: " + ini.string());
    Settings s = stage2_ini(ini, o.worker);
    const fs::path base = ini.parent_path();
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
    if (device < 0 || (d && (d < 6 || d % 2))) throw std::runtime_error("device must be >=0; D must be even and >=6");
    const uint64_t batch = o.has_batch ? o.batch : s.batch, arena = o.has_arena ? o.arena : s.arena;
    if (!batch || batch > 1048576 || arena > 1048576) throw std::runtime_error("invalid Stage2 memory budget in MB");
    _putenv_s("NTT_S4_BATCH_MB", std::to_string(batch).c_str());
    if (arena) _putenv_s("NTT_ARENA_CAP_KB", std::to_string(arena * 1024).c_str());
    const bool queue = o.save.empty();
    if (queue && o.selection) throw std::runtime_error("queue curve selection comes from ECMSTAGE2; use --save for overrides");
    Handle lock;
    if (queue) {
        if (!fs::exists(worktodo)) throw std::runtime_error("worktodo not found: " + worktodo.string());
        if (!o.dry) {
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
        const auto plan = records(save, skip, count);
        for (const auto &r : plan) {
            if (b2 <= r.b1 || b2 > static_cast<uint64_t>(INT64_MAX) - 8192)
                throw std::runtime_error("B2 must exceed every saved B1 and fit the engine's signed index range");
            if (!expected_n.empty() && r.n != expected_n) throw std::runtime_error("worktodo N differs from save N");
            if (queue && r.b1 != plan.front().b1) throw std::runtime_error("queue saves must have the same B1");
        }
        std::cout << "stage2_plan: save=" << save.string() << " curves=" << plan.size()
                  << " B2=" << b2 << " D=" << d << " device=" << device << " worker=" << o.worker
                  << " log=" << log.string() << " results=" << results.string() << '\n';
        for (const auto &r : plan) {
            std::cout << "curve_start: record=" << r.index << " sigma=" << r.sigma << " B1=" << r.b1
                      << " checksum=" << (r.checksum ? "verified" : "absent") << std::endl;
            if (o.dry) continue;
            if (child_run(o, save, r, b2, d, device, results, log))
                throw std::runtime_error("curve failed; queue retained; inspect " + log.string());
            ++completed;
            std::cout << "curve_done: record=" << r.index << " sigma=" << r.sigma << std::endl;
        }
        if (!queue || o.dry) break;
        std::string current;
        if (!ecm_worktodo_first_line(worktodo.string(), o.worker, current) || current != task_line)
            throw std::runtime_error("task completed but worktodo changed; queue retained");
        append(finished, task_line);
        if (!ecm_worktodo_advance(worktodo.string(), o.worker, task_line, WorktodoAction::Remove))
            throw std::runtime_error("task completed but worktodo changed or could not be advanced");
        if (o.once) break;
    }
    std::cout << (o.dry ? "dry_run_complete" : "stage2_complete") << ": curves=" << completed << '\n';
    return 0;
}
} // namespace s2prod

int main(int argc, char **argv) {
    try { return s2prod::driver(s2prod::arguments(argc, argv)); }
    catch (const std::exception &e) { std::cerr << "ecm_cuda_stage2: " << e.what() << '\n'; return 2; }
}
