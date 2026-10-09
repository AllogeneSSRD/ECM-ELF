// Standalone save -> CUDA Stage2 driver. Completed curve receipts commit queue
// progress; invalid input tasks are commented out. Each child owns engine state.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include "ecm_cuda_stage2.h"
#include "ecm_stage2_modulus.h"
#include "ecm_expr.h"
#include "ecm_queue_config.h"
#include "ecm_worktodo.h"
#include "ecm_stage2_fingerprint.h"
#include "ecm_stage2_factorize.h"
#include "ecm_stage2_cost_profile.h"
#include "ecm_stage2_geometry.h"
#include "ecm_stage2_logging.h"
#include "ecm_stage2_console.h"
#include "ecm_stage2_queue_state.h"
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
#include <iomanip>
#include <sstream>
#include <random>

#ifndef NTT_GL_ADD_SUB_MASK
#define NTT_GL_ADD_SUB_MASK 0
#endif

namespace s2prod {
namespace fs = std::filesystem;
struct TaskInputError : std::runtime_error { using std::runtime_error::runtime_error; };
volatile LONG stop_requests = 0;
fs::path fatal_log;
BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    InterlockedIncrement(&stop_requests);
    return TRUE;
}
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
uint64_t u64(std::string s,const char *label) {
    return ecm_config::unsigned_integer(std::move(s),label);
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
// Unlike the legacy helper, a read failure must not look like an empty queue.
bool queue_first(const fs::path &path,int worker,std::string &line) {
    std::ifstream input(path);
    if(!input)throw std::runtime_error("cannot read worktodo: "+path.string());
    int scope=1;
    std::string text;
    while(std::getline(input,text)) {
        if(text.compare(0,3,"\xef\xbb\xbf")==0)text.erase(0,3);
        text=trim(text);
        bool bracket=false;
        const int section=ecm_worktodo_parse_worker_header(text,&bracket);
        if(section){scope=section;continue;}
        if(bracket||text.empty()||text[0]=='#'||scope!=worker)continue;
        line=text;return true;
    }
    if(input.bad())throw std::runtime_error("worktodo read failed: "+path.string());
    return false;
}
std::vector<Record> records(const fs::path &path, uint64_t skip, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (!fs::exists(path)) throw TaskInputError("save not found: " + path.string());
        throw std::runtime_error("cannot open save: " + path.string());
    }
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
        } catch (const std::runtime_error &e) {
            throw TaskInputError("save record " + std::to_string(index) + ": " + e.what());
        }
        if (count && out.size() == count) break;
    }
    if (in.bad()) throw std::runtime_error("save read failed");
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
    std::string ini, save, worktodo, results, log, debug_file, receipt;
    bool debug_log = false;
    uint64_t b2 = 0, d = 0, skip = 0, curves = 0, batch = 0, arena = 0;
    unsigned carrier_exponent=0;
    int device = -1, worker = 1;
    int log_level = -1;
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
    std::string gp = ecm_config::defaults::stage2_stage2_gp;
    unsigned factor_timeout = static_cast<unsigned>(ecm_config::defaults::stage2_stage2_factor_timeout);
    bool auto_b2=false,cost_device_info=false;
    std::string cost_profile;
    uint64_t auto_min=0,auto_max=0,owner_mb=ecm_config::defaults::stage2_stage2_fold_mb,stage1_batch=ecm_config::defaults::stage2_stage1_batch;
    double stage1_seconds=ecm_config::defaults::stage2_stage1_seconds_per_curve,ratio_adjust=ecm_config::defaults::stage2_stage2_ratio_adjust;
    bool has_owner=false,has_stage1_seconds=false,has_stage1_batch=false,has_ratio=false;
};
double positive(const std::string &s,const char *name) {
    return ecm_config::positive(s,name);
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
        else if (a == "--log-level") o.log_level = stage2_log::parse(value());
        else if (a == "--debug-log-file") { o.debug_file=value(); o.debug_log=true; }
        else if (a == "--queue-receipt") o.receipt=value();
        else if (a == "--b2") o.b2 = num();
        else if (a == "--d") { o.d = num(); o.has_d = true; }
        else if (a == "--carrier-exponent") {
            const auto p=num();
            if(p && (p<2 || p>ecm_stage2::max_input_bits))throw std::runtime_error("carrier-exponent must be 0 or 2..16384");
            o.carrier_exponent=static_cast<unsigned>(p);
        }
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
        else if (a == "--owner-budget-mb") {o.owner_mb=num();o.has_owner=true;if(o.owner_mb>ecm_config::limits::stage2_stage2_fold_mb_maximum)throw std::runtime_error("invalid owner budget");}
        else if (a == "--stage1-batch") {o.stage1_batch=num();o.has_stage1_batch=true;if(!o.stage1_batch||o.stage1_batch>ecm_config::limits::stage2_stage1_batch_maximum)throw std::runtime_error("invalid Stage1 batch");}
        else if (a == "--stage1-seconds-per-curve") {o.stage1_seconds=positive(value(),"Stage1 seconds");o.has_stage1_seconds=true;}
        else if (a == "--stage2-ratio-adjust") {o.ratio_adjust=positive(value(),"Stage2 ratio adjust");o.has_ratio=true;}
        else if (a == "--gp") { o.gp = value(); o.has_gp = true; }
        else if (a == "--factor-timeout") {
            const auto seconds = num();
            if (!seconds || seconds > ecm_config::limits::stage2_stage2_factor_timeout_maximum) throw std::runtime_error("factor-timeout must be 1..600 seconds");
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
using Settings=ecm_config::Stage2Values;
Settings stage2_ini(const fs::path &path,int worker) {
    std::vector<ecm_config::IniLine> lines;
    ecm_config::read_ini(path.string(),lines);
    return ecm_config::read_stage2(ecm_config::cli_entries(lines,worker,true));
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
    bool separator=false;
    if(fs::exists(path) && fs::file_size(path)>0) {
        std::ifstream tail(path,std::ios::binary);tail.seekg(-1,std::ios::end);char last=0;
        if(!tail.get(last))throw std::runtime_error("cannot inspect append tail: "+path.string());
        separator=last!='\n';
    }
    HANDLE h=CreateFileW(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot append: "+path.string());
    LARGE_INTEGER end{};
    if(!SetFilePointerEx(h,end,nullptr,FILE_END)){CloseHandle(h);throw std::runtime_error("cannot seek append file");}
    const std::string data=(separator?"\n":"")+line+'\n';DWORD written=0;
    const bool ok=data.size()<=MAXDWORD && WriteFile(h,data.data(),static_cast<DWORD>(data.size()),&written,nullptr) && written==data.size() && FlushFileBuffers(h);
    CloseHandle(h);
    if(!ok)throw std::runtime_error("cannot flush append: "+path.string());
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
    if(stage2_log::enabled(stage2_log::debug))std::cout<<"stage2_cuda_wait: device="<<device<<" requested="<<mode
             <<" before="<<before<<" after="<<after<<std::endl;
}
int child_run(const Options &o, const fs::path &save, const Record &r,
              uint64_t b2, uint64_t d, int device, const fs::path &results, const fs::path &log,
              uint64_t batch, const Settings &ini_settings) {
    const fs::path exe = executable();
    std::wstring cmd = quote(exe.wstring()) + L" --curve-worker --save " + quote(save.wstring());
    auto arg = [&](const wchar_t *key, uint64_t n) { cmd += L" "; cmd += key; cmd += L" "; cmd += std::to_wstring(n); };
    arg(L"--record-offset", r.offset); arg(L"--record-hash", r.hash); arg(L"--record-index", r.index);
    arg(L"--b2", b2); arg(L"--d", d); arg(L"--device", device); arg(L"--worker", o.worker);
    if(o.carrier_exponent)arg(L"--carrier-exponent",o.carrier_exponent);
    arg(L"--log-level", static_cast<uint64_t>(o.log_level));
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
    if(!o.receipt.empty())cmd+=L" --queue-receipt "+quote(fs::path(o.receipt).wstring());
    if(o.debug_log)cmd+=L" --debug-log-file "+quote(fs::path(o.debug_file).wstring());
    // Preserve readable batch statistics in the file even with a concise console.
    const int engine_level=ecm_cuda_stage2_default_log_level()==stage2_log::debug?stage2_log::debug:stage2_log::batches;
    arg(L"--log-level",static_cast<uint64_t>(engine_level));
    std::ofstream output;
    if(!log.empty()) {
        fs::create_directories(log.parent_path());
        output.open(log,std::ios::binary|std::ios::app);
        if(!output)throw std::runtime_error("cannot open Stage2 log: "+log.string());
    }
    if(o.debug_log)fs::create_directories(fs::path(o.debug_file).parent_path());
    SECURITY_ATTRIBUTES security{sizeof(security),nullptr,TRUE};
    Handle out_read,out_write,err_read,err_write;
    if(!CreatePipe(&out_read.value,&out_write.value,&security,0) ||
       !CreatePipe(&err_read.value,&err_write.value,&security,0) ||
       !SetHandleInformation(out_read.value,HANDLE_FLAG_INHERIT,0) ||
       !SetHandleInformation(err_read.value,HANDLE_FLAG_INHERIT,0))
        throw std::runtime_error("cannot create curve output pipes");
    Handle job;job.value=CreateJobObjectW(nullptr,nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!job.value || !SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))
        throw std::runtime_error("cannot create curve process job");
    STARTUPINFOW si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdOutput=out_write.value;si.hStdError=err_write.value;si.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    if(!CreateProcessW(exe.c_str(),cmd.data(),nullptr,nullptr,TRUE,CREATE_SUSPENDED|CREATE_NEW_PROCESS_GROUP,nullptr,nullptr,&si,&pi))
        throw std::runtime_error("cannot launch curve process: "+std::to_string(GetLastError()));
    Handle process,thread;process.value=pi.hProcess;thread.value=pi.hThread;
    if(!AssignProcessToJobObject(job.value,pi.hProcess)) {
        TerminateProcess(pi.hProcess,2);throw std::runtime_error("cannot bind curve process lifetime");
    }
    if(ResumeThread(pi.hThread)==static_cast<DWORD>(-1))throw std::runtime_error("cannot resume curve process");
    CloseHandle(out_write.value);out_write.value=INVALID_HANDLE_VALUE;
    CloseHandle(err_write.value);err_write.value=INVALID_HANDLE_VALUE;
    const auto begin=std::chrono::steady_clock::now();
    auto next=begin+std::chrono::seconds(30);
    std::string phase="Starting curve",stdout_pending,stderr_pending;
    bool heartbeat=false,stop_reported=false;
    stage2_console::Summary console_summary(batch,ini_settings.arena,ini_settings.owner_mb);
    auto console_line=[&](const std::string &text,bool error=false) {
        if(heartbeat){std::cout<<'\n';heartbeat=false;}
        (error?std::cerr:std::cout)<<text<<std::endl;
    };
    auto line=[&](const std::string &text,bool error) {
        if(output.is_open()){output<<text<<'\n';output.flush();if(!output)throw std::runtime_error("Stage2 log write failed");}
        const auto summary=error?stage2_console::Projection{}:console_summary.observe(text);
        if(stage2_log::enabled(stage2_log::phases))
            for(const auto &entry:summary.lines)console_line(entry);
        const bool milestone=text.compare(0,14,"stage2_phase: ")==0;
        const bool result=text.compare(0,15,"stage2_result: ")==0;
        if(milestone){phase=text.substr(14);const auto timing=phase.find(" previous=");if(timing!=phase.npos)phase.resize(timing);}
        if(!summary.replace && (error || (stage2_log::enabled(stage2_log::phases)&&milestone) ||
           (stage2_log::enabled(stage2_log::curve)&&result) ||
           stage2_log::enabled(stage2_log::batches) || text.find("WARNING")!=text.npos))console_line(text,error);
    };
    auto drain=[&](HANDLE pipe,std::string &pending,bool error) {
        DWORD available=0;
        while(PeekNamedPipe(pipe,nullptr,0,nullptr,&available,nullptr) && available) {
            char buffer[4096];DWORD got=0;
            if(!ReadFile(pipe,buffer,std::min<DWORD>(available,sizeof(buffer)),&got,nullptr) || !got)break;
            pending.append(buffer,got);
            size_t pos;
            while((pos=pending.find('\n'))!=std::string::npos) {
                std::string text=pending.substr(0,pos);pending.erase(0,pos+1);
                if(!text.empty()&&text.back()=='\r')text.pop_back();
                line(text,error);
            }
        }
    };
    DWORD wait=WAIT_TIMEOUT;
    for(;;) {
        if(stop_requests>=2) {
            TerminateJobObject(job.value,130);
            WaitForSingleObject(pi.hProcess,5000);
            throw std::runtime_error("immediate stop requested; incomplete curve will restart next time");
        }
        if(stop_requests && !stop_reported) {
            line("Stop requested: finishing current curve; press Ctrl+C again to terminate immediately.",true);
            stop_reported=true;
        }
        drain(out_read.value,stdout_pending,false);drain(err_read.value,stderr_pending,true);
        wait=WaitForSingleObject(pi.hProcess,50);
        if(wait==WAIT_OBJECT_0)break;
        if(wait!=WAIT_TIMEOUT)throw std::runtime_error("curve process wait failed");
        const auto now=std::chrono::steady_clock::now();
        if(now>=next && stage2_log::enabled(stage2_log::curve)) {
            std::cout<<'\r'<<"Curve "<<r.index<<": "<<phase<<" | elapsed="
                     <<static_cast<unsigned long long>(std::chrono::duration<double>(now-begin).count())<<" s     "<<std::flush;
            heartbeat=true;next=now+std::chrono::seconds(30);
        }
    }
    drain(out_read.value,stdout_pending,false);drain(err_read.value,stderr_pending,true);
    if(!stdout_pending.empty())line(stdout_pending,false);
    if(!stderr_pending.empty())line(stderr_pending,true);
    if(stage2_log::enabled(stage2_log::phases))
        for(const auto &entry:console_summary.finish().lines)console_line(entry);
    if(heartbeat)std::cout<<std::endl;
    DWORD code=1;
    if(!GetExitCodeProcess(pi.hProcess,&code))throw std::runtime_error("cannot read curve exit status");
    return code==0?0:1;
}

std::string select_auto(Options &o,const Record &r,bool apply=true) {
    namespace c=ecm_stage2::cost;
    if(o.carrier_exponent)throw std::runtime_error("Auto B2 needs a calibrated Mersenne-carrier profile; use explicit B2");
#if NTT_GL_ADD_SUB_MASK != 0
    throw std::runtime_error("Auto B2 has no calibrated cost profile for selected NTT add/sub arithmetic");
#endif
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
    require("NTT_GSCALE_DEVICE",0,0); // GPU Gamma correction requires new costs.
    require("NTT_SCALED_ROOT_DEVICE",0,0); // Resident root preparation needs new costs.
    require("NTT_SCALED_FRONTIER_DEVICE",0,0); // Resident descent layers require new costs.
    require("NTT_FOLD_OWNER_REUSE",0,0); // Owner aliases need a matching layout/cost scope.
    require("NTT_LADDER_CAP",8192,8192);
    require("CUDA_LAUNCH_BLOCKING",0,0);
    require("NTT_S4_SAMPLE",96,96);require("NTT_S4_CHECK_EVERY",8,8);
    require("NTT_ARENA_WORKSPACE_POOL",1,1);require("NTT_FUSE_COMPACT_SCRATCH",1,1);
    require("NTT_WORKSPACE_REUSE_BQ",0,0); // Two-buffer layouts require a matching memory/cost profile.
    require("NTT_PHASE_TRIM_RAW",0,0); // Phase reclamation also needs a matching cost scope.
    require("NTT_PHASE_TRIM_OUTPUT",0,0); // Output reclamation changes residency and timings.
    require("NTT_S4_WORKSPACE_BUDGET",0,0); // Chunk policy needs a matching cost scope.
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
    // Children ignore the first console interrupt; the parent owns stop policy.
    SetConsoleCtrlHandler(nullptr,TRUE);
    std::setvbuf(stdout,nullptr,_IONBF,0);std::setvbuf(stderr,nullptr,_IONBF,0);
    if(!o.debug_file.empty())stage2_log::open_debug(fs::path(o.debug_file));
    if(o.log_level<0)o.log_level=ecm_cuda_stage2_default_log_level();
    if(ecm_cuda_stage2_set_log_level(o.log_level))throw std::runtime_error("engine rejected log level (development requires debug)");
    stage2_log::level=o.log_level;
    stage2_log::phase("Read and validate Stage1 point");
    if(ecm_cuda_stage2_check_configuration())throw std::runtime_error("Stage2 engine configuration rejected");
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
    if(o.carrier_exponent) {
        ecm_stage2::ModulusContext modulus;std::string error;
        if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
    }
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
        if(stage2_log::enabled(stage2_log::curve))std::cout << "saved_X_factor: " << number(gcd.z, 10) << '\n';
    } else {
        if (!mpz_cmp(gcd.z, n.z)) throw std::runtime_error("saved X=0 gives no usable Stage1 point");
        configure_cuda_wait(o.device);
        code = ecm_cuda_stage2_run(r.n.c_str(), r.x.c_str(), r.sigma, r.b1, o.b2, o.d, o.device,
            [](const char *text, void *ctx) { *static_cast<std::string *>(ctx) = text; }, &result,o.carrier_exponent);
    }
    if (code || result.empty()) return code ? code : 1;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double factor_seconds = 0;
    if (o.factorize_hits) {
        stage2_log::phase("Optional factor decomposition");
        const auto begin = std::chrono::steady_clock::now();
        result += ',' + ecm_stage2::factor_details(result, n.z, fs::path(o.gp), o.factor_timeout,
                                                   fs::path(o.results).parent_path() / "factor_details");
        factor_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stage2_log::phase("Publish result");
    append(o.results, "{\"queue_receipt\":" + json_string(o.receipt) + ",\"status\":" + json_string(status) + ",\"save\":" + json_string(o.save) +
        ",\"record\":" + std::to_string(o.index) + ",\"record_hash\":" + json_string(std::to_string(o.hash)) +
        ",\"worker\":" + std::to_string(o.worker) + ",\"device\":" + std::to_string(o.device) +
        ",\"N_hex\":" + json_string(r.n) + ",\"sigma\":" + std::to_string(r.sigma) +
        ",\"carrier_exponent\":" + std::to_string(o.carrier_exponent) +
        ",\"B1\":" + std::to_string(r.b1) + ",\"B2\":" + std::to_string(o.b2) +
        ",\"requested_D\":" + std::to_string(requested_d) + ",\"seconds\":" + std::to_string(seconds) +
        ",\"factorization_seconds\":" + std::to_string(factor_seconds) +
        ",\"param\":0" +
        ",\"requested_factor_only\":" + std::string(o.factor_only ? "true" : "false") +
        ",\"auto_b2\":" + std::string(auto_json.empty() ? "false" : "true") +
        ",\"requested_B2\":" + std::to_string(requested_b2) + ",\"auto_planning_seconds\":" + std::to_string(auto_seconds) +
        (auto_json.empty() ? "" : ",\"auto_plan\":" + auto_json) +
        ",\"timestamp_ms\":" + std::to_string(timestamp) + "," + result + "}");
    std::cout << "stage2_result: record=" << o.index << " seconds=" << seconds
              << " factorization_seconds=" << factor_seconds << " " << result << std::endl;
    return 0;
}
void help() {
    std::cout << "ecm_cuda_stage2 - resume normalized Suyama PARAM=0 Stage1 text saves\n"
        "  ecm_cuda_stage2 --save FILE --b2 B2 [--curves N] [--skip-curves N]\n"
        "  ecm_cuda_stage2 [--ini ecm.ini] [--worktodo FILE] [--worker N] [--once]\n"
        "Options: --device N --d D --batch-mb MB --arena-mb MB --results FILE\n"
        "         --carrier-exponent p (experimental: arithmetic modulo 2^p-1, factors still target saved N; 0=off)\n"
        "         --log FILE --log-level quiet|curve|phases|batches|debug (0..4)\n"
        "         Production console default: phases; readable file: batches. --dry-run --help\n"
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
        "INI: stage2_worktodo, stage2_finished, stage2_progress_file, stage2_log_file;\n"
        "     shared tmp_dir/device/verbose; stage2_save_dir/device overrides; stage2_b2, stage2_d,\n"
        "     stage2_batch_mb, stage2_arena_mb, stage2_results_file, stage2_log_level; [Worker #N].\n"
        "     stage2_debug_log=true enables separate diagnostics (stage2_debug_log_file).\n"
        "D=0 chooses automatically; count/curves=0 means all remaining; shortages are clamped.\n"
        "Queue only: resumes completed curves. Ctrl+C finishes current curve; twice stops now.\n"
        "Missing implicit INI uses defaults. No Stage1 computation or checkpointing.\n";
}
int driver(Options o) {
    if (o.help) { help(); return 0; }
    if (o.child) return curve_worker(o);
    if(o.auto_b2&&o.b2)throw std::runtime_error("explicit --auto-b2 conflicts with nonzero --b2");
    if(o.cost_device_info) {
        if(!o.save.empty()||!o.worktodo.empty()||!o.tune.empty()||o.auto_b2||o.plan_only||o.dry||o.b2||o.carrier_exponent)
            throw std::runtime_error("--cost-device-info is independent of curve/queue options");
        EcmStage2DeviceInfo info;if(ecm_cuda_stage2_device_info(o.device<0?0:o.device,&info))throw std::runtime_error("cannot query cost-profile device");
        std::cout<<"{\"type\":\"cost_device\",\"uuid_hex\":\""<<info.uuid_hex<<"\",\"major\":"<<info.major<<",\"minor\":"<<info.minor
            <<",\"runtime\":"<<info.runtime<<",\"driver\":"<<info.driver<<",\"fixed_mode\":"<<info.fixed_mode<<",\"outer_unroll_u\":"<<info.outer_unroll_u
            <<",\"free_bytes\":"<<info.free_bytes<<",\"total_bytes\":"<<info.total_bytes<<"}"<<std::endl;return 0;
    }
    if (o.tune.empty() && o.tune_options) throw std::runtime_error("tune options require --tune ntt");
    if (!o.tune.empty() && (o.tune != "ntt" || !o.save.empty() || !o.worktodo.empty() ||
        o.b2 || o.has_d || o.selection || o.dry || o.plan_only || o.once || o.auto_b2 || !o.cost_profile.empty() || o.carrier_exponent))
        throw std::runtime_error("--tune ntt is independent of save/queue/curve planning options");
    if (o.plan_only && o.dry) throw std::runtime_error("choose --plan-only or --dry-run");
    const fs::path cwd = fs::current_path();
    const fs::path ini = o.ini.empty() ? executable().parent_path() / "ecm.ini" : absolute_from(cwd, o.ini);
    EcmQueueConfig cfg;
    if (!ecm_queue_config_load(ini.string(), o.worker, cfg) && !o.ini.empty())
        throw std::runtime_error("cannot read explicit ini: " + ini.string());
    Settings s = stage2_ini(ini, o.worker);
    if(o.log_level<0)o.log_level=s.log_level<0?(ecm_cuda_stage2_default_log_level()==stage2_log::debug ? stage2_log::debug : (cfg.verbose?stage2_log::phases:stage2_log::curve)):s.log_level;
    o.debug_log=o.debug_log||s.debug_log||o.log_level==stage2_log::debug;
    if(ecm_cuda_stage2_default_log_level()!=stage2_log::debug)o.log_level=std::min(o.log_level,static_cast<int>(stage2_log::batches));
    if(ecm_cuda_stage2_set_log_level(o.log_level))throw std::runtime_error("engine rejected log level (development requires debug)");
    stage2_log::level=o.log_level;
    if(ecm_cuda_stage2_check_configuration())throw std::runtime_error("Stage2 engine configuration rejected");
    if(s.factorize_hits)o.factorize_hits=true;
    if(s.factor_only)o.factor_only=true;
    o.auto_b2=o.auto_b2||s.auto_b2;
    if(!o.auto_min)o.auto_min=s.auto_min;if(!o.auto_max)o.auto_max=s.auto_max;
    if(!o.has_owner){if(s.has_owner)o.owner_mb=s.owner_mb;else if(const char *v=std::getenv("NTT_FOLD_DEVICE_MAX_MB"))o.owner_mb=ecm_stage2::cost::integer(v);}
    if(o.owner_mb>ecm_config::limits::stage2_stage2_fold_mb_maximum)throw std::runtime_error("invalid owner budget");
    if(!o.has_stage1_batch)o.stage1_batch=s.stage1_batch;
    if(!o.has_stage1_seconds)o.stage1_seconds=s.stage1_seconds;
    if(!o.has_ratio)o.ratio_adjust=s.ratio_adjust;
    if(!o.has_gp && !s.gp.empty())o.gp=s.gp;
    if(!o.has_factor_timeout)o.factor_timeout=static_cast<unsigned>(s.factor_timeout);
    const fs::path base = ini.parent_path();
    if(!o.cost_profile.empty())o.cost_profile=absolute_from(cwd,o.cost_profile).string();
    else if(!s.cost_profile.empty())o.cost_profile=absolute_from(base,s.cost_profile).string();
    const fs::path worktodo = o.worktodo.empty() ? absolute_from(base, s.worktodo) : absolute_from(cwd, o.worktodo);
    const fs::path finished = absolute_from(base, s.finished);
    const fs::path tmp = absolute_from(base, s.save_dir.empty()?cfg.tmp_dir:s.save_dir);
    const std::string suffix = o.worker == 1 ? "" : "_" + std::to_string(o.worker);
    const fs::path results = !o.results.empty() ? absolute_from(cwd, o.results) :
        absolute_from(base, s.results.empty() ? ecm_config::worker_file(ecm_config::defaults::stage2_stage2_results_file,o.worker) : s.results);
    fs::path log;
    if (!o.log.empty()) log = absolute_from(cwd, o.log);
    else if (!s.has_log) log = absolute_from(base, ecm_config::worker_file(ecm_config::defaults::stage2_stage2_log_file,o.worker));
    else if (!s.log.empty()) log = absolute_from(base, s.log);
    if(o.debug_log) {
        if(o.debug_file.empty())o.debug_file=absolute_from(base,s.debug_file.empty()?ecm_config::worker_file(ecm_config::defaults::stage2_stage2_debug_log_file,o.worker):s.debug_file).string();
        else o.debug_file=absolute_from(cwd,o.debug_file).string();
    }
    const fs::path progress=s.progress.empty()?fs::path(worktodo.string()+suffix+".progress"):absolute_from(base,s.progress);
    const int device = o.device < 0 ? (s.device<0?cfg.device:s.device) : o.device;
    const uint64_t d = o.has_d ? o.d : s.d;
    o.device=device;o.d=d;
    if (device < 0 || (d && (d < 6 || d % 2))) throw std::runtime_error("device must be >=0; D must be even and >=6");
    const uint64_t batch = o.has_batch ? o.batch : s.batch, arena = o.has_arena ? o.arena : s.arena;
    o.arena=arena;
    if (batch < ecm_config::limits::stage2_stage2_batch_mb_minimum || batch > ecm_config::limits::stage2_stage2_batch_mb_maximum || arena > ecm_config::limits::stage2_stage2_arena_mb_maximum) throw std::runtime_error("invalid Stage2 memory budget in MB");
    _putenv_s("NTT_FOLD_DEVICE_MAX_MB",std::to_string(o.owner_mb).c_str());
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
    auto same_path=[](const fs::path &a,const fs::path &b) {
        if(a.empty()||b.empty())return false;
        std::error_code error;
        if(fs::equivalent(a,b,error)&&!error)return true;
        return upper(a.lexically_normal().string())==upper(b.lexically_normal().string());
    };
    // A common INI may share device/save defaults, never writable queue state.
    const std::vector<fs::path> writable={worktodo,finished,results,log,progress,fs::path(o.debug_file)};
    for(size_t i=0;i<writable.size();++i) {
        if(same_path(writable[i],ini)||same_path(writable[i],executable()))throw std::runtime_error("output path conflicts with configuration or executable");
        for(size_t j=0;j<i;++j)if(same_path(writable[i],writable[j]))throw std::runtime_error("queue, finished, progress, result and log paths must be distinct");
    }
    if(queue && (same_path(worktodo,absolute_from(executable().parent_path(),cfg.worktodo)) ||
                  same_path(finished,absolute_from(executable().parent_path(),cfg.finished))))
        throw std::runtime_error("Stage2 queue/finished conflicts with Stage1 configuration; use distinct stage2_worktodo/stage2_finished");
    auto report=[&](const std::string &message,bool error=false) {
        (error?std::cerr:std::cout)<<message<<std::endl;
        if(!o.dry&&!o.plan_only&&!log.empty())append(log,message);
    };
    if(!SetConsoleCtrlHandler(console_control,TRUE))throw std::runtime_error("cannot install console stop handler");
    struct ControlGuard { ~ControlGuard(){SetConsoleCtrlHandler(console_control,FALSE);} } control_guard;
    uint64_t completed=0,failed_tasks=0;
    for (;;) {
        fatal_log.clear();
        if(stop_requests)break;
        fs::path save;
        std::string task_line,expected_n;
        std::vector<Record> plan;
        uint64_t skip=o.skip,count=o.curves,b2=o.b2?o.b2:s.b2;
        if(queue && !queue_first(worktodo,o.worker,task_line))break;
        try {
            if(queue) {
                QueueSelection fields;
                try { fields=queue_fields(task_line); }
                catch(const std::runtime_error &e){throw TaskInputError(e.what());}
                skip=fields.skip;count=fields.count;
                if(!o.b2&&fields.b2)b2=fields.b2;
                Big n;std::string error;
                if(!ecm_compute_stage2_n(fields.task,n.z,error))throw TaskInputError("worktodo N: "+error);
                expected_n=number(n.z);save=absolute_from(tmp,fields.task.save_name);
            } else save=absolute_from(cwd,o.save);
            for(const auto &path:writable)if(same_path(save,path))throw std::runtime_error("save path conflicts with writable queue/log/result state");
            if(!o.dry&&!o.plan_only)fatal_log=log;
            if(o.auto_b2&&!b2&&o.cost_profile.empty())throw std::runtime_error("Auto B2 requires --cost-profile FILE");
            if(o.carrier_exponent && o.auto_b2 && !b2)throw std::runtime_error("Mersenne carrier requires explicit B2 until cost profiles are calibrated");
            plan=records(save,skip,count);
            for(const auto &r:plan) {
                if((!(o.auto_b2&&!b2)&&b2<=r.b1)||b2>static_cast<uint64_t>(INT64_MAX)-8192)
                    throw TaskInputError("B2 must exceed every saved B1 and fit the engine's signed index range");
                if(!expected_n.empty()&&r.n!=expected_n)throw TaskInputError("worktodo N differs from save N");
                if(queue&&r.b1!=plan.front().b1)throw TaskInputError("queue saves must have the same B1");
                if(r.x=="0")throw TaskInputError("saved X=0 gives no usable Stage1 point");
                if(o.carrier_exponent) {
                    ecm_stage2::ModulusContext modulus;std::string error;
                    // An invalid command option must not mark a valid queue row as erroneous.
                    if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
                }
            }
        } catch(const TaskInputError &e) {
            if(!queue || o.dry || o.plan_only)throw;
            // Record the reason durably before commenting out the original row.
            report("ERROR: task input: "+std::string(e.what())+" | task="+task_line,true);
            append(finished,"# ERROR reason="+std::string(e.what())+"\n# "+task_line);
            if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::MarkError))
                throw std::runtime_error("cannot mark invalid task; queue changed or rewrite failed");
            ++failed_tasks;
            if(o.once)break;
            continue;
        }
        if(count && plan.size()<count)
            report("WARNING: requested="+std::to_string(count)+" available="+std::to_string(plan.size())+
                   " after skip="+std::to_string(skip)+"; running available curves only.",true);
        if(plan.empty())report("WARNING: no available curves after skip; task will finish with zero curves.",true);
        stage2_queue::State state;
        const bool persist=queue&&!o.dry&&!o.plan_only;
        if(persist) {
            std::ostringstream identity;
            identity << std::quoted(task_line) << ' ' << std::quoted(worktodo.string()) << ' '
                     << ecm_stage2::sha256_file(worktodo) << ' '
                     << std::quoted(save.string()) << ' ' << ecm_stage2::sha256_file(save) << ' '
                     << std::quoted(results.string()) << ' ' << skip << ' ' << count << ' ' << b2 << ' ' << d << ' '
                     << o.auto_b2 << ' ' << o.factor_only << ' ' << o.factorize_hits << ' '
                     << o.arena << ' ' << batch << ' ' << o.owner_mb << ' ' << std::setprecision(17)
                     << o.ratio_adjust << ' ' << o.stage1_seconds << ' ' << o.stage1_batch;
            if(o.carrier_exponent)identity << " carrier_exponent=" << o.carrier_exponent;
            if(o.auto_b2&&!b2)identity << ' ' << ecm_stage2::sha256_file(o.cost_profile) << ' ' << o.auto_min << ' ' << o.auto_max;
            const auto key=identity.str();
            if(state.load(progress)) {
                if(state.identity!=key) {
                    if(state.done!=state.total || !state.pending.empty())
                        throw std::runtime_error("unfinished queue task/save/settings changed; restore them or explicitly archive progress: "+progress.string());
                    state={};
                }
            }
            if(state.identity.empty()) {
                state.identity=key;state.total=plan.size();state.run=stage2_queue::token();state.save(progress);
            }
            if(state.total!=plan.size())throw std::runtime_error("progress curve count differs from validated task");
            if(!state.pending.empty() && stage2_queue::result_written(results,state.pending)) {
                ++state.done;state.pending.clear();state.save(progress);
                report("Recovered completed curve from durable result receipt.");
            }
        }
        if(stage2_log::enabled(stage2_log::curve)||o.dry) {
            std::ostringstream message;
            message << "stage2_plan: save=" << save.string() << " requested=" << (count?std::to_string(count):"all")
                    << " available=" << plan.size() << " completed=" << state.done << " remaining=" << plan.size()-state.done
                    << " B2=" << b2 << " D=" << d << " device=" << device << " worker=" << o.worker
                    << " log=" << log.string() << " results=" << results.string();
            report(message.str());
        }
        for(size_t i=static_cast<size_t>(state.done);i<plan.size();++i) {
            if(stop_requests)break;
            const auto &r=plan[i];
            if(stage2_log::enabled(stage2_log::curve)||o.dry)
                report("curve_start: "+std::to_string(i+1)+"/"+std::to_string(plan.size())+" record="+std::to_string(r.index)+
                       " sigma="+std::to_string(r.sigma)+" B1="+std::to_string(r.b1)+" checksum="+(r.checksum?"verified":"absent"));
            if(o.dry)continue;
            if(o.plan_only) {
                if(o.auto_b2&&!b2){Options local=o;local.b2=0;std::cout<<select_auto(local,r,false)<<std::endl;continue;}
                std::string result;
                const int code=ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,b2,d,device,
                    [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&result,o.carrier_exponent);
                if(code||result.empty())throw std::runtime_error("Stage2 planning failed");
                std::cout<<result<<std::endl;continue;
            }
            if(persist) {
                // Publish intent first, then launch. Only a complete result with
                // this random receipt can resolve an ambiguous interrupted exit.
                state.pending=stage2_queue::token();state.save(progress);o.receipt=state.pending;
            }
            const auto begin=std::chrono::steady_clock::now();
            if(child_run(o,save,r,b2,d,device,results,log,batch,s))
                throw std::runtime_error("curve failed; queue and completed progress retained; inspect "+log.string());
            if(persist) {
                if(!stage2_queue::result_written(results,state.pending))throw std::runtime_error("successful worker has no durable result receipt; queue retained");
                ++state.done;state.pending.clear();state.save(progress);
            }
            ++completed;
            if(stage2_log::enabled(stage2_log::curve)) {
                std::ostringstream message;
                message << "curve_done: " << i+1 << '/' << plan.size() << " record=" << r.index << " sigma=" << r.sigma
                        << " wall=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count() << " s";
                report(message.str());
            }
        }
        if(!queue||o.dry||o.plan_only)break;
        if(state.done<state.total)break;
        std::string current;
        if(!queue_first(worktodo,o.worker,current)||current!=task_line)
            throw std::runtime_error("task completed but worktodo changed; queue and progress retained");
        if(!stage2_queue::finished_written(finished,state.run,task_line))
            append(finished,"# stage2_task_id="+state.run+"\n"+task_line);
        if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::Remove))
            throw std::runtime_error("task completed but queue could not be advanced; progress retained");
        // A complete stale state is safe to replace if a crash happens here.
        if(!DeleteFileW(progress.c_str()) && GetLastError()!=ERROR_FILE_NOT_FOUND)
            throw std::runtime_error("task advanced but cannot remove completed progress file");
        report("task_done: requested="+(count?std::to_string(count):"all")+" available="+std::to_string(plan.size())+
               " completed="+std::to_string(state.done));
        if(o.once||stop_requests)break;
    }
    if(stage2_log::enabled(stage2_log::curve)||o.dry)
        report(std::string(o.plan_only?"plan_complete":o.dry?"dry_run_complete":stop_requests?"stage2_stopped":"stage2_complete")+
               ": curves_this_run="+std::to_string(completed)+" invalid_tasks="+std::to_string(failed_tasks));
    return failed_tasks?1:0;
}

} // namespace s2prod

int main(int argc, char **argv) {
    try { return s2prod::driver(s2prod::arguments(argc, argv)); }
    catch (const std::exception &e) {
        const std::string message=std::string("ecm_cuda_stage2: ")+e.what();
        std::cerr<<message<<std::endl;
        try {if(!s2prod::fatal_log.empty())s2prod::append(s2prod::fatal_log,message);}catch(...){}
        return 2;
    }
}
