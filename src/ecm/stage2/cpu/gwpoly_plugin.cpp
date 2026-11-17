#include "gwpoly.h"
#include "ecm/integer.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace {
template<class Function>
int guarded(char* error, uint32_t capacity, Function action) noexcept {
    try { return action(); }
    catch (const std::exception& e) { if (capacity && error) std::snprintf(error, capacity, "%s", e.what()); }
    catch (...) { if (capacity && error) std::snprintf(error, capacity, "GWNUM Stage2 failure"); }
    return -1;
}

int run(const ecm_gw_stage2_request* r, uint32_t* factor, ecm_gw_stage2_stats* stats,
        char* error, uint32_t capacity) noexcept {
    return guarded(error, capacity, [&] {
        if (!r || !stats || !factor || !r->words || !r->n || !r->x || !r->a24)
            throw std::invalid_argument("Invalid Stage2 buffers");
        ecm_edwards::Integer n, x, a24, f;
        auto import = [&](mpz_ptr z, const uint32_t* words) {
            mpz_import(z, r->words, -1, sizeof(uint32_t), 0, 0, words);
        };
        import(n.v, r->n);
        import(x.v, r->x);
        import(a24.v, r->a24);
        if (mpz_cmp_ui(n.v, 2) <= 0 || mpz_cmp(x.v, n.v) >= 0 || mpz_cmp(a24.v, n.v) >= 0)
            throw std::invalid_argument("Invalid Stage2 residues");
        ecm_gwpoly::Options options;
        options.d = r->d;
        options.threads = r->threads;
        options.memory_mib = r->memory_mib;
        options.exponent = r->exponent;
        options.check = r->check != 0;
        options.stop = r->stop;
        if (!ecm_gwpoly::stage2(f.v, n.v, x.v, a24.v, r->b1, r->b2, options, *stats)) return 0;
        std::fill_n(factor, r->words, 0u);
        mpz_export(factor, nullptr, -1, sizeof(uint32_t), 0, 0, f.v);
        return 1;
    });
}

int selftest(uint32_t threads, char* error, uint32_t capacity) noexcept {
    return guarded(error, capacity, [&] { ecm_gwpoly::selftest(threads); return 0; });
}
}

extern "C" __declspec(dllexport) const ecm_gw_stage2_api* ecm_gw_stage2_get_api(uint32_t abi) noexcept {
    if (abi != 1) return nullptr;
    static const ecm_gw_stage2_api api{1, sizeof(api), ecm_gwpoly::version(),
                                       ECM_GW_BUILD, ecm_gwpoly::cpu(), run, selftest};
    return &api;
}
