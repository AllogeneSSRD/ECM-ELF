#include <cstdarg>
#include <cstdio>

// The host checkpoint codec does not need a CUDA factory just to report a save.
extern "C" void outputf(int, const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stdout, format, args);
    va_end(args);
}
