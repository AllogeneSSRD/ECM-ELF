#pragma once

#include <gmp.h>

#include <cstdint>
#include <string>

/** Resolve relative paths against {@link opencl_ecm_set_work_dir}; absolute paths unchanged. */
std::string opencl_ecm_resolve_data_path(const char *path);

bool opencl_ecm_check_save_file_writable(const std::string &savefilename, bool saveappend);

std::string opencl_ecm_build_saved_n_expr(const std::string &original_expr, const mpz_t N,
                                          uint32_t curves, mpz_t *factors, int *array_found);

bool opencl_ecm_append_save_lines(const std::string &savefilename, const mpz_t N, double B1,
                                  uint32_t firstsigma, uint32_t curves, mpz_t *factors,
                                  const std::string &n_expr_save);

/**
 * Append one text save line per curve for the **Suyama-sigma Montgomery** path
 * (gmp-ecm `-param 0`), in the same field family as the param3 writer above.
 *
 * Differences that matter (see docs/ECM_Montgomery_STAGE1.md 9):
 *   * `SIGMA` is **64-bit** here (gmp-ecm generates sigmas beyond 32 bits; the
 *     param3 API above is uint32_t because param3's convention is 32-bit),
 *   * no `PARAM=` key is written (the reference omits it for param 0),
 *   * `X` is the normalised Montgomery x of [s]P for curves that did NOT hit, and
 *     the found factor for curves that DID hit (hit[i] != 0) -- a hit makes the
 *     normalised x meaningless (Z is not invertible).
 *
 * `sigmas` is the FULL per-curve array, not a base value: a run with random sigmas
 * has no sigmas[i] == sigmas[0] + i relationship, and writing a sigma that does not
 * belong to the stored X makes a stage-2 handoff rebuild a different curve (it then
 * searches a curve whose point it never had).
 */
bool ecm_append_save_lines_mont(const std::string &savefilename, const mpz_t N, double B1,
                                const uint64_t *sigmas, uint32_t curves, const mpz_t *xs,
                                const int *hit, const std::string &n_expr_save);
