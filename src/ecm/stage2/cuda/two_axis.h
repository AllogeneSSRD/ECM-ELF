#pragma once

#include <cstddef>
#include <algorithm>

namespace stage2_two_axis {
struct FoldWorkspace;

// For i in the window, i-N is negative and i+N is beyond the
// last linear coefficient. No cyclic alias can enter the window.
inline size_t window_span(size_t ma, size_t mb, size_t first, size_t count) {
    return std::max({ma, mb, first + count, ma + mb - 1 - first});
}

FoldWorkspace* create(unsigned p, unsigned max_coefficients, size_t limit, const char*& fallback);
FoldWorkspace* create_product(unsigned p, unsigned ma, unsigned mb, unsigned first, unsigned count,
                              bool host, size_t limit, const char*& fallback);
void release(FoldWorkspace*& workspace);
size_t bytes(const FoldWorkspace& workspace);
unsigned coefficient_length(const FoldWorkspace& workspace);
unsigned polynomial_length(const FoldWorkspace& workspace);
unsigned long long* host_staging(FoldWorkspace& workspace);
void multiply(FoldWorkspace& workspace, const unsigned long long* a, unsigned ma,
              const unsigned long long* b, unsigned mb, unsigned long long* out, unsigned count,
              unsigned first = 0, bool reverse_b = false);
}
