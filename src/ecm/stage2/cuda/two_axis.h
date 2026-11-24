#pragma once

#include <cstddef>

namespace stage2_two_axis {
struct FoldWorkspace;

FoldWorkspace* create(unsigned p, unsigned max_coefficients, size_t limit, const char*& fallback);
void release(FoldWorkspace*& workspace);
size_t bytes(const FoldWorkspace& workspace);
void multiply(FoldWorkspace& workspace, const unsigned long long* a, unsigned ma,
              const unsigned long long* b, unsigned mb, unsigned long long* out, unsigned count);
}
