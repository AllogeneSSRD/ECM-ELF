#pragma once
#include "polynomial.h"
#include <limits>
#include <map>

namespace ecm_opencl::stage2 {
// A logical coefficient window. stride/offset count limbs, size counts coefficients.
// Results own their buffer through shared handles; windows never own raw cl_mem.
struct Polynomial {
    ntt::Buffer buffer;
    std::size_t size=0,batch=1,offset=0,stride=0;
    Polynomial element(std::size_t index)const;
};
struct ArithmeticOptions {
    PolynomialOptions product; // Engine derives capacity and owns the workspace.
    bool reuse_workspace=true;
    bool group_batches=true;
    bool borrow_contiguous=true; // Read-only regular slices need no gather copy.
};
struct ArithmeticStats {
    std::map<unsigned,Word> ntt_transforms; // Three transforms per convolution pair.
    Word multiply_calls=0,multiply_pairs=0,coefficient_launches=0,device_copies=0;
    Word upload_bytes=0,download_bytes=0,device_copy_bytes=0;
    Word borrowed_gathers=0,avoided_copy_bytes=0;
};
struct ArithmeticState;
class Arithmetic {
    std::unique_ptr<ArithmeticState> state_;
public:
    Arithmetic(ntt::Runtime &,const ArithmeticOptions &);
    ~Arithmetic();
    Arithmetic(const Arithmetic&)=delete;
    Arithmetic& operator=(const Arithmetic&)=delete;
    std::size_t words()const;
    bool group_batches()const;
    const ArithmeticStats &stats()const;
    Polynomial upload(const std::vector<Word> &,std::size_t count,std::size_t batch=1);
    std::vector<Word> download(const Polynomial &);
    Polynomial constant(Word value);
    Polynomial window(const Polynomial &,std::size_t first,std::size_t count)const;
    Polynomial reshape(const Polynomial &,std::size_t count,std::size_t first=0,bool reverse=false);
    Polynomial gather(const std::vector<Polynomial> &); // Equal-sized, batch=1 windows.
    Polynomial subtract(const Polynomial &,const Polynomial &);
    Polynomial linear_factors(const Polynomial &roots);
    Polynomial multiply(const Polynomial &,const Polynomial &,
        std::size_t first=0,std::size_t count=std::numeric_limits<std::size_t>::max());
    // a(0)=1. Used with the reverse of a monic polynomial over composite N.
    Polynomial inverse_series(const Polynomial &a,std::size_t count);
    // b monic, inverse = reverse(b)^-1 mod X^k, k >= a.size-b.degree.
    Polynomial remainder(const Polynomial &a,const Polynomial &b,const Polynomial &inverse);
    void finish(); // Drains async coefficient operations and releases their leases.
};
} // namespace ecm_opencl::stage2
