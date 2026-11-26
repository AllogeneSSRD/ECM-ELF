#pragma once

#include <cstddef>
#include <algorithm>

namespace stage2_two_axis {
struct FoldWorkspace;
enum class Operand { left, right };
struct Spectrum { const unsigned long long* data; };

// For i in the window, i-N is negative and i+N is beyond the
// last linear coefficient. No cyclic alias can enter the window.
inline size_t window_span(size_t ma, size_t mb, size_t first, size_t count) {
    return std::max({ma, mb, first + count, ma + mb - 1 - first});
}

FoldWorkspace* create(unsigned p, unsigned max_coefficients, size_t limit, const char*& fallback);
FoldWorkspace* create_product(unsigned p, unsigned ma, unsigned mb, unsigned first, unsigned count,
                              bool host, size_t limit, const char*& fallback, unsigned batch = 1);
void release(FoldWorkspace*& workspace);
size_t bytes(const FoldWorkspace& workspace);
unsigned coefficient_length(const FoldWorkspace& workspace);
unsigned polynomial_length(const FoldWorkspace& workspace);
unsigned batch_capacity(const FoldWorkspace& workspace);
unsigned long long* host_staging(FoldWorkspace& workspace);
// The plan bounds the input rows and the alias-free output window. All stages
// use the default stream. Spectra are valid only with the plan that built them.
// Prepare the left operand before the right when using workspace buffers.
// Caller-owned storage of spectrum_words() words survives inverse_window().
size_t spectrum_words(const FoldWorkspace& workspace);
Spectrum pretransform(FoldWorkspace& workspace, Operand operand, const unsigned long long* coefficients,
                      unsigned rows, unsigned reverse_rows = 0, unsigned long long* retained = nullptr);
// Two optional spectrum slots share the workspace's budget and lifetime.
Spectrum retain(FoldWorkspace& workspace, unsigned slot, const unsigned long long* coefficients,
                unsigned rows, size_t limit);
void multiply_retained(FoldWorkspace& workspace, const unsigned long long* coefficients, unsigned rows,
                       Spectrum fixed, unsigned long long* out, unsigned first, unsigned count,
                       unsigned reverse_rows = 0);
// After pretransform(left), multiply a sibling into the right matrix and use
// its compact staging area for the output rows. count must fit that area.
void multiply_preserving_left(FoldWorkspace& workspace, const unsigned long long* coefficients, unsigned rows,
                              unsigned long long* out, unsigned first, unsigned count, unsigned reverse_rows);
// Stores the normalized spectral product in the left workspace buffer.
void frequency_product(FoldWorkspace& workspace, Spectrum left, Spectrum right);
// Consumes the product and invalidates both workspace-backed spectra.
void inverse_window(FoldWorkspace& workspace, unsigned long long* out, unsigned first, unsigned count);
void multiply(FoldWorkspace& workspace, const unsigned long long* a, unsigned ma,
              const unsigned long long* b, unsigned mb, unsigned long long* out, unsigned count,
              unsigned first = 0, bool reverse_b = false);
// Full products of resident tree nodes. Offsets are device arrays of word
// offsets into src and dst, with one entry per product.
void multiply_tree_batch(FoldWorkspace& workspace, const unsigned long long* src,
                         const unsigned long long* left, const unsigned long long* right,
                         unsigned ma, unsigned mb, unsigned long long* dst,
                         const unsigned long long* output, unsigned batch);
}
