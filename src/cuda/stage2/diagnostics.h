#pragma once

#include "context.h"

struct PolyLayer;

struct FDump {
    bool ok = false;
    std::string err;
    std::string n_dec, n_hex, a24_hex, q_hex;
    long n_bits = 0;
    unsigned long long D = 0, B1 = 0, B2 = 0, sigma = 0, degree = 0;
    std::vector<unsigned long long> baby_j;      /* ascending, coprime to D, j <= D/2 */
    std::vector<std::string> baby_x;             /* the CPU's affine x of [j]Q, hex */
    std::vector<std::string> F;                  /* coefficients of F, ascending, hex */
};

bool read_hex_mpz(mpz_t out, const std::string &hex);

FDump read_f_dump(const char *path);

bool write_f_dump(const char *path, const mpz_t N, const mpz_t a24, const mpz_t Q,
                         unsigned long long D, unsigned long long B1, unsigned long long B2,
                         unsigned long long sigma, const std::vector<unsigned long long> &bj,
                         const std::vector<std::string> &bx,
                         const std::vector<std::string> &F, unsigned long long degree);

size_t words_for_bits(size_t bits);

void gfinv_batch_fixture(const mpz_t actual,size_t actualW);

void hex_of_words(std::string &out, const unsigned long long *w, size_t W, const mpz_t N);

void groot_device_fixture(PolyLayer &L);

void descent_slow(PolyLayer &L,
                         const std::vector<std::vector<unsigned long long>> &Ft,
                         const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
                         std::vector<std::vector<unsigned long long>> &values,
                         unsigned long long &divmods);

std::vector<unsigned long long> fold_gmp_reference(PolyLayer &L,
    const std::vector<unsigned long long> &G,const std::vector<unsigned long long> &H,
    const std::vector<unsigned long long> &F);

void fold_flat_fixture(PolyLayer &L);

void fold_device_fixture(PolyLayer &L);

std::vector<unsigned long long> scaled_state_gmp(
    const CPoly &H, const std::vector<unsigned long long> &F, size_t d, PolyLayer &L);

void scaled_descent_fixture(PolyLayer &L);

void s4_flat_input_check(PolyLayer &L, size_t slices = 3);

void s4_output_window_check(PolyLayer &L);

void groot_lifetime_check(PolyLayer &L);

void segment_product_fixture(const LadderCtx &C);

void block_product_fixture(const LadderCtx &C);

void device_baby_fixture(size_t w);

void s2g_install_crash_handler();
void s2g_install_terminate();
void s2g_state(const char *what);
void s2g_print_last_state(const char *why);
#if defined(_WIN32)
extern char g_last_state[512];
#endif

namespace stage2_giant_chunk { void fixture(); }
