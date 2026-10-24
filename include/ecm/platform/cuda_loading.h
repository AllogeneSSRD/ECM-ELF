#pragma once
#include <cstdlib>
// Configure before the first CUDA context/API use. Explicit user diagnostics
// (including EAGER) remain visible and are checked by the qualification tools.
inline void ecm_cuda_default_module_loading() {
    if (std::getenv("CUDA_MODULE_LOADING")) return;
#if defined(_WIN32)
    _putenv_s("CUDA_MODULE_LOADING","LAZY");
#else
    setenv("CUDA_MODULE_LOADING","LAZY",0);
#endif
}
