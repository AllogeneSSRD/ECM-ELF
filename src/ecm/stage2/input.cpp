// Complete Stage2 save/queue input validation and durable diagnostic append.
#include "input.h"
#include "io/expression.h"
#include "../../io/config_format.h"
#include "../../io/stage2_bounds.h"
#include "../../io/files.h"
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif
#include "modulus.h"
#include "ecm/io/save.h"
#include "ecm/curve/edwards.h"
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
Record parse_record(std::string line,unsigned input_max_bits,EdwardsPreparation preparation,
                    const std::string& projection_target) {
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
                   (fields["STATUS"]!="POINT" && fields["STATUS"]!="FACTOR" && fields["STATUS"]!="IDENTITY") ||
                   !fields.count("CHECKSUM")))
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
    if (mpz_cmp_ui(n.z, 3) <= 0 || !mpz_odd_p(n.z) ||
        (projection_target.empty() && input_max_bits && mpz_sizeinbase(n.z, 2) > input_max_bits))
        throw std::runtime_error(input_max_bits ?
            "N must be odd, >3 and at most "+std::to_string(input_max_bits)+" bits" : "N must be odd and >3");
    if (mpz_set_str(x.z, fields["X"].c_str(), 0) || mpz_sgn(x.z) < 0 || mpz_cmp(x.z, n.z) >= 0)
        throw std::runtime_error("X must be an integer in [0,N), decimal or 0x hex");
    if (fields.count("Z") && (mpz_set_str(z.z, fields["Z"].c_str(), 0) || mpz_cmp_ui(z.z, 1)))
        throw std::runtime_error("only normalized affine saves (Z=1 or absent) are supported");
    const auto saved_n = number(n.z), saved_x = number(x.z);
    if(edwards) {
        r.factor=fields["STATUS"]=="FACTOR";
        const bool identity=fields["STATUS"]=="IDENTITY";
        r.skip_stage2=r.factor || identity;
        r.hash = ecm_edwards::save_checksum(saved_n, saved_x, r.sigma, r.b1, r.factor, identity);
        if(u64(fields["CHECKSUM"],"CHECKSUM") != r.hash)
            throw std::runtime_error("Edwards save CHECKSUM mismatch");
        ecm_edwards::SavedCurve original{saved_n,saved_x,r.sigma,r.b1,0,r.factor,identity,{}};
        if (fields.count("QX") || fields.count("QZ") || fields.count("QCHECKSUM")) {
            original = ecm_edwards::parse_saved_curve(line);
            r.hash = fingerprint(std::to_string(r.hash) + '|' + fields.at("QCHECKSUM"));
        }
        r.checksum=true;
        if(r.factor) {
            if(mpz_cmp_ui(x.z,1)<=0 || mpz_cmp(x.z,n.z)>=0 || !mpz_divisible_p(n.z,x.z))
                throw std::runtime_error("invalid Edwards saved factor");
        } else if(identity && mpz_sgn(x.z)) throw std::runtime_error("invalid Edwards saved identity");
        if(!projection_target.empty()) {
            Big target;
            if(mpz_set_str(target.z,projection_target.c_str(),16) || mpz_cmp_ui(target.z,3)<=0 ||
               mpz_cmp(target.z,n.z)>=0 || !mpz_divisible_p(n.z,target.z) ||
               (input_max_bits && mpz_sizeinbase(target.z,2)>input_max_bits))
                throw std::runtime_error("Stage2 projection requires a supported proper divisor of the saved modulus");
            if(r.factor) {
                mpz_divexact(z.z,n.z,x.z);
                if(!mpz_divisible_p(z.z,target.z))
                    throw std::runtime_error("Stage2 cofactor does not divide the saved endpoint modulus");
            }
            r.projected=true;
            r.skip_stage2=!ecm_edwards::project_saved_curve(x.z,original,target.z);
            r.factor=false;
            mpz_swap(n.z,target.z);
        }
        if(!r.skip_stage2 && preparation==EdwardsPreparation::Construct) {
            Big d,px,py,a24;
            const auto setup=ecm_edwards::construct(d.z,px.z,py.z,a24.z,factor.z,r.sigma,n.z);
            if(setup==ecm_edwards::Setup::Factor){r.factor=true;mpz_set(x.z,factor.z);}
            else if(setup!=ecm_edwards::Setup::Ready)throw std::runtime_error("degenerate Edwards curve in save");
            else r.a24=number(a24.z);
        }
    } else if(!projection_target.empty()) {
        throw std::runtime_error("cofactor projection requires Edwards records");
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
    if (!edwards)
        r.hash = fingerprint(saved_n + '|' + saved_x + '|' + std::to_string(r.sigma) +
                             '|' + std::to_string(r.b1));
    r.n = r.projected ? projection_target : saved_n;
    r.x = r.projected || (r.factor && !r.skip_stage2) ? number(x.z) : saved_x;
    return r;
}
bool comment(const std::string &line) {
    const auto t = trim(line);
    return t.empty() || t[0] == '#' || t[0] == ';';
}
std::string saved_record_line(std::istream& input, uint64_t offset, uint64_t index, uint64_t key) {
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    std::string line;
    if (std::getline(input, line) && (offset == 0 || line.compare(0, 7, "METHOD=") == 0)) {
        try {
            if (parse_record(line, 0, EdwardsPreparation::ValidateOnly).hash == key)
                return line;
        } catch (const std::runtime_error&) {
            // An edit to metadata in an earlier record can shift this offset.
        }
    }
    input.clear();
    input.seekg(0);
    uint64_t position = 0;
    while (std::getline(input, line)) {
        if (comment(line) || ++position != index)
            continue;
        if (parse_record(line, 0, EdwardsPreparation::ValidateOnly).hash == key)
            return line;
        break;
    }
    throw TaskInputError("save record " + std::to_string(index) + " changed after planning");
}
std::vector<Record> records(const fs::path &path, uint64_t skip, uint64_t count, unsigned input_max_bits,
                            EdwardsPreparation preparation,const std::string& projection_target) {
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
            auto r = parse_record(line,input_max_bits,preparation,projection_target);
            r.offset = static_cast<uint64_t>(offset); r.index = index;
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
QueueSelection queue_fields(std::string line,unsigned input_max_bits) {
    QueueSelection q;
    std::string error;
    if (!ecm_parse_stage2_line(line, q.task, error))
        throw TaskInputError(error);
    if (input_max_bits && q.task.n > input_max_bits)
        throw TaskInputError("worktodo exponent exceeds supported input range");
    const auto bounds = ecm_stage2::parse_bounds(q.task.b2_text, "worktodo B2");
    q.b2 = bounds.upper;
    q.b2_min = bounds.lower;
    q.skip = q.task.skip_curves;
    q.count = q.task.curves_to_run;
    return q;
}
std::string queue_modulus(const EcmStage2Task &task,const Record *first_record) {
    if (task.modulus_from_save) {
        if (!first_record)
            throw TaskInputError("ECMSTAGE2N has no available save record after skip");
        return first_record->n;
    }
    Big expected;
    std::string error;
    if (!ecm_compute_stage2_n(task, expected.z, error))
        throw TaskInputError("worktodo N: " + error);
    return number(expected.z);
}
void append(const fs::path &path, const std::string &line) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string());
    bool separator=false;
    if(fs::exists(path) && fs::file_size(path)>0) {
        std::ifstream tail(path,std::ios::binary);tail.seekg(-1,std::ios::end);char last=0;
        if(!tail.get(last))throw std::runtime_error("cannot inspect append tail: "+path.string());
        separator=last!='\n';
    }
    const std::string data=(separator?"\n":"")+line+'\n';
#ifdef _WIN32
    HANDLE h=CreateFileW(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot append: "+path.string());
    LARGE_INTEGER end{};
    if(!SetFilePointerEx(h,end,nullptr,FILE_END)){CloseHandle(h);throw std::runtime_error("cannot seek append file");}
    DWORD written=0;
    const bool ok=data.size()<=MAXDWORD && WriteFile(h,data.data(),static_cast<DWORD>(data.size()),&written,nullptr) && written==data.size() && FlushFileBuffers(h);
    CloseHandle(h);
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC, 0666);
    if(fd < 0)throw std::runtime_error("cannot append: "+path.string());
    size_t offset=0;
    while(offset<data.size()) {
        const auto written=::write(fd,data.data()+offset,data.size()-offset);
        if(written<0 && errno==EINTR)continue;
        if(written<=0)break;
        offset+=static_cast<size_t>(written);
    }
    const bool flushed=offset==data.size() && fsync(fd)==0;
    const bool ok=::close(fd)==0 && flushed;
#endif
    if(!ok)throw std::runtime_error("cannot flush append: "+path.string());
}
}
