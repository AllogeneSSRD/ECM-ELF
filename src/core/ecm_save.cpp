#include "ecm_save.h"

#include "ecm_checkpoint.h"
#include "opencl_ecm_log.h"

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
    char *s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
}

std::string build_who_field() {
    const char *uname = std::getenv("LOGNAME");
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

std::string opencl_ecm_resolve_data_path(const char *path) {
    return opencl_ecm_resolve_data_path_buf(path);
}


bool opencl_ecm_check_save_file_writable(const std::string &savefilename, bool saveappend) {
    if (savefilename.empty()) {
        return false;
    }
    if (!saveappend && access(savefilename.c_str(), 0) == 0) {
        ecm_ts_fprintf(stderr, "Save file %s already exists, will not overwrite\n",
                       savefilename.c_str());
        return false;
    }
    if (!opencl_ecm_ensure_parent_dir(savefilename.c_str())) {
        ecm_ts_fprintf(stderr, "Could not create parent directory for %s\n",
                       savefilename.c_str());
        return false;
    }
    FILE *savefile = fopen(savefilename.c_str(), "a");
    if (savefile == nullptr) {
        ecm_ts_fprintf(stderr, "Could not open file %s for writing\n", savefilename.c_str());
        return false;
    }
    fclose(savefile);
    if (!saveappend) {
        struct stat st {};
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

std::string opencl_ecm_build_saved_n_expr(const std::string &original_expr, const mpz_t N,
                                          uint32_t curves, mpz_t *factors, int *array_found) {
    std::string expr = original_expr.empty() ? mpz_to_dec_string(N) : original_expr;
    mpz_t remaining;
    mpz_init_set(remaining, N);

    std::vector<std::string> uniq_factors_dec;
    uniq_factors_dec.reserve(curves);
    for (uint32_t i = 0; i < curves; ++i) {
        if (array_found[i] == ECM_NO_FACTOR_FOUND) {
            continue;
        }
        if (mpz_cmp_ui(factors[i], 1) <= 0 || mpz_cmp(factors[i], N) >= 0) {
            continue;
        }
        std::string dec = mpz_to_dec_string(factors[i]);
        if (std::find(uniq_factors_dec.begin(), uniq_factors_dec.end(), dec) ==
            uniq_factors_dec.end()) {
            uniq_factors_dec.push_back(dec);
        }
    }

    std::sort(uniq_factors_dec.begin(), uniq_factors_dec.end(),
              [](const std::string &a, const std::string &b) {
                  if (a.size() != b.size()) {
                      return a.size() < b.size();
                  }
                  return a < b;
              });

    mpz_t f;
    mpz_init(f);
    for (const std::string &dec : uniq_factors_dec) {
        if (mpz_set_str(f, dec.c_str(), 10) != 0) {
            continue;
        }
        if (!mpz_divisible_p(remaining, f)) {
            continue;
        }
        expr = "(" + expr + ")/" + dec;
        mpz_divexact(remaining, remaining, f);
    }
    mpz_clear(f);
    mpz_clear(remaining);
    return expr;
}

bool opencl_ecm_append_save_lines(const std::string &savefilename, const mpz_t N, double B1,
                                  uint32_t firstsigma, uint32_t curves, mpz_t *factors,
                                  const std::string &n_expr_save, int param_id) {
    if (!opencl_ecm_ensure_parent_dir(savefilename.c_str())) {
        ecm_ts_fprintf(stderr, "Could not create parent directory for %s\n",
                       savefilename.c_str());
        return false;
    }
    std::ofstream out(savefilename, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        ecm_ts_fprintf(stderr, "Could not open file %s for appending\n", savefilename.c_str());
        return false;
    }

    mpz_t sigma_mpz, checksum;
    mpz_init(sigma_mpz);
    mpz_init(checksum);

    const time_t t = std::time(nullptr);
    char timebuf[128] = {0};
    const tm *lt = std::localtime(&t);
    if (lt) {
        std::strftime(timebuf, sizeof(timebuf), "%a %b %d %H:%M:%S %Y", lt);
    }
    const std::string who = build_who_field();

    for (uint32_t i = 0; i < curves; ++i) {
        mpz_set_ui(sigma_mpz, firstsigma + i);
        mpz_set_d(checksum, B1);
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(sigma_mpz, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(N, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(factors[i], CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, (unsigned long)(param_id + 1) % CHKSUMMOD);
        const unsigned long csum = mpz_fdiv_ui(checksum, CHKSUMMOD);

        char *sigma_dec = mpz_get_str(nullptr, 10, sigma_mpz);
        char *x_hex = mpz_get_str(nullptr, 16, factors[i]);

        out << "METHOD=ECM; PARAM=" << param_id
            << "; SIGMA=" << sigma_dec
            << "; B1=" << std::llround(B1)
            << "; N=" << n_expr_save
            << "; X=0x" << x_hex
            << "; CHECKSUM=" << csum
            << "; PROGRAM=GMP-ECM 7.0.6;"
            << " X0=0x0; Y0=0x0;"
            << (who.empty() ? "" : (" WHO=" + who + ";"))
            << " TIME=" << timebuf << ";"
            << "\n";

        free(sigma_dec);
        free(x_hex);
    }

    mpz_clear(sigma_mpz);
    mpz_clear(checksum);
    ecm_ts_fprintf(stdout, "Saved %u curve line(s) to %s\n", curves, savefilename.c_str());
    return true;
}

/* ---------------------------------------------------------------------------
 * Suyama-sigma Montgomery (gmp-ecm -param 0) save lines.
 *
 * Field family is the same one the param3 writer above uses; the three
 * differences are documented in ecm_save.h.  SIGMA is 64-bit, so it cannot go
 * through mpz_set_ui (unsigned long is 32-bit on Windows) -- it is assembled
 * from two halves instead.
 * ------------------------------------------------------------------------- */
bool ecm_append_save_lines_mont(const std::string &savefilename, const mpz_t N, double B1,
                                const uint64_t *sigmas, uint32_t curves, const mpz_t *xs,
                                const int *hit, const std::string &n_expr_save)
{
    std::ofstream out(savefilename, std::ios::app);
    if (!out.is_open()) {
        ecm_ts_fprintf(stderr, "Could not open file %s for appending\n", savefilename.c_str());
        return false;
    }

    mpz_t sigma_mpz, checksum;
    mpz_init(sigma_mpz);
    mpz_init(checksum);

    const time_t t = std::time(nullptr);
    char timebuf[128] = {0};
    const tm *lt = std::localtime(&t);
    if (lt) std::strftime(timebuf, sizeof(timebuf), "%a %b %d %H:%M:%S %Y", lt);
    const std::string who = build_who_field();

    for (uint32_t i = 0; i < curves; ++i) {
        /* the curve's own sigma, assembled for 64-bit safety (Windows'
           mpz_set_ui takes a 32-bit unsigned long) */
        mpz_set_ui(sigma_mpz, (unsigned long)(sigmas[i] >> 32));
        mpz_mul_2exp(sigma_mpz, sigma_mpz, 32);
        mpz_add_ui(sigma_mpz, sigma_mpz, (unsigned long)(sigmas[i] & 0xFFFFFFFFull));

        mpz_set_d(checksum, B1);
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(sigma_mpz, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(N, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(xs[i], CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, 1 % CHKSUMMOD);      /* param 0 -> (param+1) = 1 */
        const unsigned long csum = mpz_fdiv_ui(checksum, CHKSUMMOD);

        char *sigma_dec = mpz_get_str(nullptr, 10, sigma_mpz);
        char *x_hex = mpz_get_str(nullptr, 16, xs[i]);

        out << "METHOD=ECM"
            << "; SIGMA=" << sigma_dec
            << "; B1=" << std::llround(B1)
            << "; N=" << n_expr_save
            << "; X=0x" << x_hex
            << "; CHECKSUM=" << csum
            << "; PROGRAM=ECM-ELY;"
            << " X0=0x0; Y0=0x0;"
            << (who.empty() ? "" : (" WHO=" + who + ";"))
            << " TIME=" << timebuf << ";"
            << "\n";

        free(sigma_dec);
        free(x_hex);
        (void)hit;   /* callers already put the factor into xs[i] for hit curves */
    }

    mpz_clear(sigma_mpz);
    mpz_clear(checksum);
    ecm_ts_fprintf(stdout, "Saved %u Montgomery curve line(s) to %s\n", curves,
                   savefilename.c_str());
    return true;
}
