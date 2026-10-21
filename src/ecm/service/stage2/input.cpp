// Complete Stage2 save/queue input validation and durable diagnostic append.
#include "input.h"
#include "../../application/configuration/ecm_expr.h"
#include "../../application/configuration/ecm_toml.h"
#include "../../core/stage2/ecm_stage2_modulus.h"
#include "../../storage/save/ecm_edwards_io.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
namespace s2prod {
std::string trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == s.npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string upper(std::string s) {
    for (char &c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}
std::string number(const mpz_t z, int base) {
    std::vector<char> buf(mpz_sizeinbase(z, base) + 3);
    mpz_get_str(buf.data(), base, z);
    return buf.data();
}
void set64(mpz_t z, uint64_t n) {
    mpz_set_ui(z, static_cast<unsigned long>(n >> 32));
    mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, static_cast<unsigned long>(n & 0xffffffffu));
}
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
    const bool edwards=upper(fields["METHOD"])=="ECM-EDWARDS-AM";
    if (!edwards && upper(fields["METHOD"]) != "ECM") throw std::runtime_error("save METHOD must be ECM or ECM-EDWARDS-AM");
    if(edwards && (fields.count("PARAM") || fields["VERSION"]!="1" || fields["SCALAR"]!="48LCM" ||
                   (fields["STATUS"]!="POINT" && fields["STATUS"]!="FACTOR") || !fields.count("CHECKSUM")))
        throw std::runtime_error("invalid Edwards save identity (VERSION=1, SCALAR=48LCM, STATUS and CHECKSUM required; PARAM forbidden)");
    if (fields.count("PARAM") && u64(fields["PARAM"], "PARAM") != 0)
        throw std::runtime_error("only Suyama PARAM=0 text saves are supported");
    std::string sigma = fields["SIGMA"];
    if (sigma.compare(0, 2, "0:") == 0) sigma.erase(0, 2);
    Record r;
    r.edwards=edwards;
    r.sigma = u64(sigma, "SIGMA");
    r.b1 = u64(fields["B1"], "B1");
    if (r.sigma < (edwards?1u:6u) || r.b1 < 2) throw std::runtime_error("invalid SIGMA or B1");
    Big n, x, z, chk, factor;
    std::string err;
    if (!ecm_parse_expression(fields["N"], n.z, &err)) throw std::runtime_error("N: " + err);
    if (mpz_cmp_ui(n.z, 3) <= 0 || !mpz_odd_p(n.z) || mpz_sizeinbase(n.z, 2) > ecm_stage2::max_input_bits)
        throw std::runtime_error("N must be odd, >3 and at most 16384 bits");
    if (mpz_set_str(x.z, fields["X"].c_str(), 0) || mpz_sgn(x.z) < 0 || mpz_cmp(x.z, n.z) >= 0)
        throw std::runtime_error("X must be an integer in [0,N), decimal or 0x hex");
    if (fields.count("Z") && (mpz_set_str(z.z, fields["Z"].c_str(), 0) || mpz_cmp_ui(z.z, 1)))
        throw std::runtime_error("only normalized affine saves (Z=1 or absent) are supported");
    if(edwards) {
        const bool hit=fields["STATUS"]=="FACTOR";
        if(u64(fields["CHECKSUM"],"CHECKSUM")!=ecm_edwards::save_checksum(number(n.z),number(x.z),r.sigma,r.b1,hit))
            throw std::runtime_error("Edwards save CHECKSUM mismatch");
        r.checksum=true;
        if(hit) {
            if(mpz_cmp_ui(x.z,1)<=0 || mpz_cmp(x.z,n.z)>=0 || !mpz_divisible_p(n.z,x.z))
                throw std::runtime_error("invalid Edwards saved factor");
        } else {
            Big d,px,py,a24;
            const auto setup=ecm_edwards::construct(d.z,px.z,py.z,a24.z,factor.z,r.sigma,n.z);
            if(setup==ecm_edwards::Setup::Factor)mpz_set(x.z,factor.z);
            else if(setup!=ecm_edwards::Setup::Ready)throw std::runtime_error("degenerate Edwards curve in save");
            else r.a24=number(a24.z);
        }
    } else if (fields.count("CHECKSUM")) {
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
}
