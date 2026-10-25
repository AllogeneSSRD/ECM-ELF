#include "ecm_save.h"

#include "../checkpoint/ecm_checkpoint.h"
#include "ecm_edwards_io.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr unsigned long CHKSUMMOD = 4294967291UL;

std::string mpz_to_dec_string(const mpz_t v) {
    char* s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
}

std::string build_who_field() {
    const char* uname = std::getenv("LOGNAME");
    if (!uname || !*uname) {
        uname = std::getenv("USERNAME");
    }
    std::string user = (uname && *uname) ? uname : "";

    std::string host;
#ifdef _WIN32
    char hbuf[MAX_COMPUTERNAME_LENGTH + 2] = {0};
    DWORD sz = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(hbuf, &sz) && sz > 0) {
        host.assign(hbuf, hbuf + sz);
    }
#else
    char hbuf[64] = {0};
    if (gethostname(hbuf, sizeof(hbuf) - 1) == 0) {
        host = hbuf;
    }
#endif

    if (user.empty() && host.empty()) {
        return "";
    }
    return user + "@" + host;
}

} // namespace

std::string opencl_ecm_resolve_data_path(const char* path) {
    return opencl_ecm_resolve_data_path_buf(path);
}

bool opencl_ecm_check_save_file_writable(const std::string& savefilename, bool saveappend) {
    if (savefilename.empty()) {
        return false;
    }
    if (!saveappend && access(savefilename.c_str(), 0) == 0) {
        ecm_ts_fprintf(stderr, "Save file %s already exists, will not overwrite\n",
                       savefilename.c_str());
        return false;
    }
    if (!opencl_ecm_ensure_parent_dir(savefilename.c_str())) {
        ecm_ts_fprintf(stderr, "Could not create parent directory for %s\n", savefilename.c_str());
        return false;
    }
    FILE* savefile = fopen(savefilename.c_str(), "a");
    if (savefile == nullptr) {
        ecm_ts_fprintf(stderr, "Could not open file %s for writing\n", savefilename.c_str());
        return false;
    }
    fclose(savefile);
    if (!saveappend) {
        struct stat st{};
        if (stat(savefilename.c_str(), &st) != 0 || st.st_size != 0) {
            ecm_ts_fprintf(stderr, "Save file %s initialization failed\n", savefilename.c_str());
            return false;
        }
        if (remove(savefilename.c_str()) != 0) {
            ecm_ts_fprintf(stderr, "Save file %s could not be cleaned up\n", savefilename.c_str());
            return false;
        }
    }
    return true;
}

std::string ecm_save_modulus_expression(const std::string& original_expr, const mpz_t N) {
    // X and CHECKSUM were computed modulo N. The published expression must
    // describe that same modulus even when this batch found a factor.
    return original_expr.empty() ? mpz_to_dec_string(N) : original_expr;
}

std::string ecm_encode_mont_save(mpz_srcptr n, double b1, int param,
                                 const std::vector<EcmSavePoint>& points,
                                 const std::string& expression) {
    std::ostringstream out;
    ecm_edwards::Integer sigma, checksum;
    const auto now = std::time(nullptr);
    char timebuf[128] = {0};
    if (const auto* local = std::localtime(&now))
        std::strftime(timebuf, sizeof(timebuf), "%a %b %d %H:%M:%S %Y", local);
    const auto who = build_who_field();
    for (const auto& point : points) {
        // unsigned long is only 32 bits on Windows; retain the full sigma.
        mpz_set_ui(sigma.v, static_cast<unsigned long>(point.sigma >> 32));
        mpz_mul_2exp(sigma.v, sigma.v, 32);
        mpz_add_ui(sigma.v, sigma.v, static_cast<unsigned long>(point.sigma & 0xffffffffull));
        mpz_set_d(checksum.v, b1);
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(sigma.v, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(n, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, mpz_fdiv_ui(point.value, CHKSUMMOD));
        mpz_mul_ui(checksum.v, checksum.v, static_cast<unsigned long>(param + 1) % CHKSUMMOD);
        out << "METHOD=ECM";
        if (param)
            out << "; PARAM=" << param;
        out << "; SIGMA=" << mpz_to_dec_string(sigma.v) << "; B1=" << std::llround(b1)
            << "; N=" << expression << "; X=0x" << ecm_edwards::hex(point.value)
            << "; CHECKSUM=" << mpz_fdiv_ui(checksum.v, CHKSUMMOD)
            << "; PROGRAM=" << ecm_program::name << "; X0=0x0; Y0=0x0;"
            << (who.empty() ? "" : " WHO=" + who + ";") << " TIME=" << timebuf << ";\n";
    }
    return out.str();
}

bool opencl_ecm_append_save_lines(const std::string& path, const mpz_t n, double b1, uint32_t first,
                                  uint32_t count, mpz_t* values, const std::string& expression,
                                  int param) {
    std::vector<EcmSavePoint> points;
    points.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        points.push_back({uint64_t(first) + i, values[i]});
    if (!ecm_save::publish(path, ecm_encode_mont_save(n, b1, param, points, expression), true))
        return false;
    ecm_ts_fprintf(stdout, "Saved %u curve line(s) to %s\n", count, path.c_str());
    return true;
}

bool ecm_append_save_lines_mont(const std::string& path, const mpz_t n, double b1,
                                const uint64_t* sigmas, uint32_t count, const mpz_t* values,
                                const int* hit, const std::string& expression) {
    std::vector<EcmSavePoint> points;
    points.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        points.push_back({sigmas[i], values[i]});
    (void)hit;
    if (!ecm_save::publish(path, ecm_encode_mont_save(n, b1, 0, points, expression), true))
        return false;
    ecm_ts_fprintf(stdout, "Saved %u Montgomery curve line(s) to %s\n", count, path.c_str());
    return true;
}
