#pragma once

#include <gmp.h>

#include <string>

// Parse decimal / hex / expression (e.g. "(2^421-1)") into mpz_t.
bool ecm_parse_expression(const std::string& text, mpz_t out, std::string* err_out);

// batch_s = prod_{p<=B1} p^{floor(log_p(B1))}
bool ecm_compute_batch_s(mpz_t s, double B1);
