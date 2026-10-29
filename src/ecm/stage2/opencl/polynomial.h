#pragma once
#include "ecm/opencl/ntt/runtime.h"

// Experimental exact modular polynomial product, independent of CUDA and GMP.
namespace ecm_opencl::stage2 {
using ntt::Word;
struct IntegerShape {
    unsigned coefficient_bits=0,digit_bits=0,log2_length=0;
    std::size_t capacity=0,slot_bits=0,slot_words=0,length=0;
};
// Same conservative term/slot contract as CUDA ntt_shape_query, computed exactly.
IntegerShape integer_shape(unsigned coefficient_bits,std::size_t capacity,unsigned forced_digit_bits=0);
// A request is synchronous at its verdict boundary, so different plans on the
// same host thread may safely share this workspace. It never owns results.
class PolynomialWorkspace {
    ntt::Buffer a_,b_,blocks_,flags_;
    friend class PolynomialPlan;
public:
    void release() { a_={};b_={};blocks_={};flags_={}; }
};
struct PolynomialOptions {
    std::vector<Word> modulus; // Minimal little-endian limbs; N >= 2, <= 16384 bits.
    std::size_t capacity=1;   // Maximum coefficient count of either input.
    bool mersenne=false;     // Requires modulus == 2^S-1, not merely a divisor.
    unsigned forced_digit_bits=0;
    ntt::PlanOptions transform; // log2_length is derived from integer_shape.
    std::shared_ptr<PolynomialWorkspace> workspace; // Null: per-request buffers.
};
struct PolynomialView {
    ntt::Buffer buffer;
    std::size_t offset=0,stride=0; // 64-bit limb indices, per polynomial.
};
struct ProductInfo {
    std::size_t coefficients=0,batch=0;
    Word verdict_words=0,carry_blocks=0;
};
struct PolynomialState;
class PolynomialPlan {
    std::shared_ptr<PolynomialState> state_;
public:
    PolynomialPlan(ntt::Runtime &,const PolynomialOptions &);
    const IntegerShape &shape()const;
    std::size_t coefficient_words()const;
    std::string build_log()const;
    // Ordinary residues, not Montgomery form. Inputs are unsigned S-bit integers;
    // they need not be < N. Exact integer slot coefficients are reduced mod N.
    // Read-only A/B may share an owner. Output must be distinct from both.
    // All batches use the same modulus.
    // Synchronous verdict drain: no result is valid if this method throws.
    // No per-pass host waits; one in-order queue for pack/NTT/carry/S4.
    ProductInfo multiply(PolynomialView a,std::size_t a_coefficients,
                         PolynomialView b,std::size_t b_coefficients,
                         PolynomialView out,std::size_t batch=1);
};
} // namespace ecm_opencl::stage2
