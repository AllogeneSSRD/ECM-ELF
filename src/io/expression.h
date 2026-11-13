#pragma once
#include <gmp.h>
#include <string>

bool ecm_parse_expression(const std::string& text, mpz_t out, std::string* err_out);
