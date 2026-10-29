#pragma once
#include "arithmetic.h"

namespace ecm_opencl::stage2 {
struct ProductTree {
    std::size_t leaves=0,padded=0;
    std::vector<Polynomial> nodes; // Heap order; padded leaves are polynomial 1.
};
ProductTree product_tree(Arithmetic &,const Polynomial &roots,bool keep_children=true);
// Scaled remainder descent: identical root transform and sibling middle-product
// recurrence to the CUDA engine. No per-node polynomial inversions.
Polynomial evaluate_scaled(Arithmetic &,const ProductTree &,const Polynomial &h,const Polynomial &inverse);

// Retains F, inverse(reverse(F)), H and device tree coefficients. Point generation
// and factor extraction are separate operators. Must not outlive Arithmetic.
class Resultant {
    Arithmetic &arithmetic_;
    ProductTree tree_;
    Polynomial inverse_,fold_;
    std::size_t blocks_=0;
public:
    Resultant(Arithmetic &,const Polynomial &baby_roots);
    void append(const Polynomial &giant_roots); // 1..P roots, supports partial tail.
    Polynomial values();                       // H evaluated at each baby root.
    const ProductTree &tree()const{return tree_;}
    const Polynomial &inverse()const{return inverse_;}
    const Polynomial &fold()const{return fold_;}
    std::size_t blocks()const{return blocks_;}
};
} // namespace ecm_opencl::stage2
