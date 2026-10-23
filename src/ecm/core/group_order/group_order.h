#pragma once
#include <gmp.h>
#include <cstdint>
#include <string>
#include <vector>

// Independent CPU/GMP arithmetic. No worker, GPU, PARI, or application dependency.
namespace ecm_group_order {
struct Integer {
    mpz_t v;
    Integer();
    explicit Integer(uint64_t value);
    explicit Integer(const std::string &decimal);
    Integer(const Integer &other);
    Integer(Integer &&other) noexcept;
    Integer &operator=(const Integer &other);
    Integer &operator=(Integer &&other) noexcept;
    ~Integer();
    std::string str() const;
};

enum class Status { Complete, Partial, ResourceLimit, InvalidInput };
enum class Primality { Composite, Probable, Proven };
const char *name(Status value);
const char *name(Primality value);

struct Limits {
    uint64_t milliseconds = 2000;
    uint64_t rho_iterations = 4000000;
    uint64_t baby_steps = 200000;
    uint32_t random_points = 24;
    uint32_t trial_limit = 1000;
    uint32_t enumeration_limit = 4096;
    uint32_t max_bits = 4096;
};
struct Statistics {
    double total_ms = 0, construction_ms = 0, factor_ms = 0, count_ms = 0, point_ms = 0;
    uint64_t rho_iterations = 0, baby_steps = 0, giant_steps = 0, point_additions = 0;
    uint32_t sampled_points = 0;
};
struct PrimePower {
    Integer prime;
    uint32_t exponent = 0;
    Primality primality = Primality::Probable;
};
struct Factorization {
    Status status = Status::InvalidInput;
    std::string reason;
    std::vector<PrimePower> factors;
    Integer cofactor{1};
    bool all_proven = false;
    Statistics statistics;
};

Primality classify_prime(const Integer &n);
Factorization factor_integer(const Integer &n, const Limits &limits = {});

struct Point {
    Integer x, y;
    bool infinity = true;
};
// Short Weierstrass y^2=x^3+a*x+b over a prime >3.
struct Curve {
    Integer prime, a, b;
    Point point;
};
enum class Family { Montgomery, Edwards };
struct CurveRequest {
    Integer prime;
    Family family = Family::Montgomery;
    int parameter = 0; // Ignored only for Edwards; pass -1 for that family.
    uint64_t sigma = 6;
};
struct Analysis {
    Status status = Status::InvalidInput;
    std::string reason;
    Primality modulus_primality = Primality::Composite;
    // Zero unless locally counted. An input multiple is never relabelled #E.
    bool group_order_computed = false;
    Integer group_order, multiple, point_order;
    Factorization multiple_factorization;
    std::vector<PrimePower> point_factorization;
    bool point_minimality_checked = false;
    bool all_proven = false;
    Statistics statistics;
};

// Throws std::invalid_argument for malformed or singular input.
Curve construct_curve(const CurveRequest &request);
bool point_on_curve(const Curve &curve, const Point &point);
// Curve and point must be valid over a prime field. Negative scalars are rejected.
Point multiply(const Curve &curve, const Point &point, const Integer &scalar);

// Known-multiple mode performs no group counting. Supplied factors are optional;
// their product and primality classifications are rechecked before using them.
Analysis analyze(const CurveRequest &request, const Limits &limits = {},
                 const Integer *known_multiple = nullptr,
                 const std::vector<PrimePower> *known_factors = nullptr);
// Also available for independent curves and noncyclic-group regression tests.
Analysis analyze_curve(const Curve &curve, const Limits &limits = {},
                       const Integer *known_multiple = nullptr,
                       const std::vector<PrimePower> *known_factors = nullptr);
} // namespace ecm_group_order
