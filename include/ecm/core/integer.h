#pragma once
#include <gmp.h>
#include <string>
#include <vector>

// GMP values shared by host computation and persistent codecs.
namespace ecm_edwards {
struct Integer {
    mpz_t v;

    Integer() {
        mpz_init(v);
    }

    ~Integer() {
        mpz_clear(v);
    }

    Integer(const Integer&) = delete;
    Integer& operator=(const Integer&) = delete;
};

inline std::string hex(const mpz_t x) {
    std::vector<char> buf(mpz_sizeinbase(x, 16) + 3);
    mpz_get_str(buf.data(), 16, x);
    return buf.data();
}

class Array {
    mpz_t* p_;
    size_t size_;

public:
    explicit Array(size_t size) : p_(new mpz_t[size]), size_(size) {
        for (size_t i = 0; i < size_; ++i)
            mpz_init(p_[i]);
    }

    ~Array() {
        for (size_t i = 0; i < size_; ++i)
            mpz_clear(p_[i]);
        delete[] p_;
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    mpz_t* data() {
        return p_;
    }

    mpz_t& operator[](size_t i) {
        return p_[i];
    }
};
} // namespace ecm_edwards
