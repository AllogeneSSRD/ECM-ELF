#include "ecm_expr.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <vector>

namespace {

class ExprParser {
public:
    explicit ExprParser(const std::string& input) : text(input), pos(0), error(false) {}

    bool parse(mpz_t out) {
        skip_ws();
        parse_expr(out);
        skip_ws();
        if (!error && pos != text.size()) {
            set_error("unexpected trailing characters");
        }
        return !error;
    }

    const std::string& message() const { return message_text; }

private:
    const std::string& text;
    size_t pos;
    bool error;
    std::string message_text;

    void set_error(const std::string& msg) {
        if (!error) {
            error = true;
            message_text = msg;
        }
    }

    void skip_ws() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
    }

    bool match(char ch) {
        skip_ws();
        if (pos < text.size() && text[pos] == ch) {
            ++pos;
            return true;
        }
        return false;
    }

    void parse_expr(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_term(lhs);
        while (!error) {
            if (match('+')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_add(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('-')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_sub(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_term(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_power(lhs);
        while (!error) {
            if (match('*')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                mpz_mul(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('/')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                if (mpz_sgn(rhs) == 0) {
                    set_error("division by zero");
                    mpz_clear(rhs);
                    break;
                }
                if (!mpz_divisible_p(lhs, rhs)) {
                    set_error("division is not exact (result would not be an integer)");
                    mpz_clear(rhs);
                    break;
                }
                mpz_divexact(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_power(mpz_t out) {
        mpz_t base;
        mpz_init(base);
        parse_unary(base);
        if (error) {
            mpz_clear(base);
            return;
        }

        if (match('^')) {
            mpz_t exponent;
            mpz_init(exponent);
            parse_power(exponent);
            if (error) {
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            if (mpz_sgn(exponent) < 0 || !mpz_fits_ulong_p(exponent)) {
                set_error("exponent must be a non-negative integer that fits in unsigned long");
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            const unsigned long exp = mpz_get_ui(exponent);
            mpz_pow_ui(out, base, exp);
            mpz_clear(base);
            mpz_clear(exponent);
            return;
        }

        mpz_set(out, base);
        mpz_clear(base);
    }

    void parse_unary(mpz_t out) {
        skip_ws();
        if (match('+')) {
            parse_unary(out);
            return;
        }
        if (match('-')) {
            parse_unary(out);
            mpz_neg(out, out);
            return;
        }
        parse_primary(out);
    }

    void parse_primary(mpz_t out) {
        skip_ws();
        if (match('(')) {
            parse_expr(out);
            if (!match(')')) {
                set_error("missing closing parenthesis");
            }
            return;
        }

        const size_t start = pos;
        bool saw_digit = false;
        while (pos < text.size()) {
            const unsigned char ch = static_cast<unsigned char>(text[pos]);
            if (std::isalnum(ch) || ch == 'x' || ch == 'X') {
                saw_digit = true;
                ++pos;
            } else {
                break;
            }
        }
        if (!saw_digit) {
            set_error("expected number or parenthesized expression");
            return;
        }

        const std::string token = text.substr(start, pos - start);
        if (mpz_set_str(out, token.c_str(), 0) != 0) {
            set_error(std::string("invalid integer token: ") + token);
        }
    }
};

} // namespace

bool ecm_parse_expression(const std::string& text, mpz_t out, std::string* err_out) {
    ExprParser parser(text);
    if (!parser.parse(out)) {
        if (err_out != nullptr) {
            *err_out = parser.message();
        }
        return false;
    }
    return true;
}

bool ecm_compute_batch_s(mpz_t s, double B1) {
    static const unsigned MAX_HEIGHT = 32;

    if (B1 < 2.0) {
        mpz_set_ui(s, 1);
        return true;
    }

    const uint64_t limit64 = static_cast<uint64_t>(std::floor(B1 + 0.0001));
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }

    const uint32_t limit = static_cast<uint32_t>(limit64);

    std::vector<char> sieve(static_cast<size_t>(limit) + 1u, 1);
    sieve[0] = sieve[1] = 0;
    for (uint32_t p = 2; static_cast<uint64_t>(p) * static_cast<uint64_t>(p) <= limit; ++p) {
        if (!sieve[p]) {
            continue;
        }
        for (uint64_t q = static_cast<uint64_t>(p) * static_cast<uint64_t>(p); q <= limit; q += p) {
            sieve[static_cast<size_t>(q)] = 0;
        }
    }

    mpz_t acc[MAX_HEIGHT];
    mpz_t ppz;
    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_init(acc[j]);
    }
    mpz_init(ppz);

    unsigned i = 0;
    for (uint32_t pi = 2; pi <= limit; ++pi) {
        if (!sieve[pi]) {
            continue;
        }

        uint64_t pp = pi;
        const uint64_t maxpp = limit / pi;
        while (pp <= maxpp) {
            pp *= pi;
        }

        mpz_import(ppz, 1, -1, sizeof(pp), 0, 0, &pp);

        if ((i & 1u) == 0u) {
            mpz_set(acc[0], ppz);
        } else {
            mpz_mul(acc[0], acc[0], ppz);
        }

        unsigned j = 0;
        while ((i & (1u << j)) != 0u) {
            if (j + 1 >= MAX_HEIGHT - 1) {
                for (unsigned k = 0; k < MAX_HEIGHT; ++k) {
                    mpz_clear(acc[k]);
                }
                mpz_clear(ppz);
                return false;
            }

            if ((i & (1u << (j + 1))) == 0u) {
                mpz_swap(acc[j + 1], acc[j]);
            } else {
                mpz_mul(acc[j + 1], acc[j + 1], acc[j]);
            }
            mpz_set_ui(acc[j], 1);
            ++j;
        }

        ++i;
    }

    if (i == 0) {
        mpz_set_ui(s, 1);
    } else {
        mpz_set(s, acc[0]);
        for (unsigned j = 1; j < MAX_HEIGHT && mpz_cmp_ui(acc[j], 0) != 0; ++j) {
            mpz_mul(s, s, acc[j]);
        }
    }

    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_clear(acc[j]);
    }
    mpz_clear(ppz);
    return true;
}
