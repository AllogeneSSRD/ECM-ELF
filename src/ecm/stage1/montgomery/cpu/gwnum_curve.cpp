#include "gwnum_curve.h"

#include "ecm/integer.h"
#include "ecm/platform/logging.h"
#include "ecm/stage1/gwnum_abi.h"

#ifdef _WIN64
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Plugin {
    HMODULE module = nullptr;
    const ecm_gwnum_api* api = nullptr;
    std::string error;

    Plugin() {
        wchar_t executable[32768];
        const auto size = GetModuleFileNameW(nullptr, executable, static_cast<DWORD>(std::size(executable)));
        if (!size || size == std::size(executable)) {
            error = "GWNUM Stage1: cannot locate worker executable";
            return;
        }
        const auto path = std::filesystem::path(std::wstring_view(executable, size)).parent_path() /
                          ECM_GWNUM_PLUGIN_FILENAME;
        module = LoadLibraryExW(path.c_str(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) {
            const auto code = GetLastError();
            error = "GWNUM Stage1: cannot load gwnum.dll beside the worker (Windows error " +
                    std::to_string(code) + ")";
            return;
        }
        const auto get_api = reinterpret_cast<ecm_gwnum_get_api_fn>(
            GetProcAddress(module, ECM_GWNUM_GET_API_SYMBOL));
        const auto* table = get_api ? get_api(ECM_GWNUM_ABI_VERSION) : nullptr;
        if (!table || table->abi_version != ECM_GWNUM_ABI_VERSION ||
            table->struct_size < sizeof(ecm_gwnum_api) || !table->gwnum_version ||
            std::string_view(table->gwnum_version) != "31.6" || !table->create ||
            !table->advance || !table->read || !table->destroy) {
            FreeLibrary(module);
            module = nullptr;
            error = "GWNUM Stage1: gwnum.dll must provide ECM-ELF ABI 1 and GWNUM 31.6";
            return;
        }
        api = table;
    }

    ~Plugin() {
        if (module) FreeLibrary(module);
    }
};

const Plugin& gwnum_plugin() {
    static const Plugin plugin;
    return plugin;
}

using Words = std::vector<uint32_t>;

Words export_words(const mpz_t value, uint32_t count) {
    Words words(count);
    mpz_export(words.data(), nullptr, -1, sizeof(words[0]), 0, 0, value);
    return words;
}

void import_words(mpz_t value, const Words& words, const mpz_t modulus) {
    mpz_import(value, words.size(), -1, sizeof(words[0]), 0, 0, words.data());
    mpz_mod(value, value, modulus);
}

} // namespace
#endif

const char* mont_stage1_gwnum_load_error() {
#ifdef _WIN64
    const auto& plugin = gwnum_plugin();
    return plugin.api ? nullptr : plugin.error.c_str();
#else
    return "GWNUM Stage1 requires a Windows x64 worker and gwnum.dll";
#endif
}

int mont_stage1_curve_gwnum_bits_ex(mpz_t factor, mpz_t Qx, mpz_t Qz, const mpz_t N,
                                   uint64_t sigma, const uint8_t* bits, size_t nbits,
                                   size_t start_bit, mont_ladder_state_t* st,
                                   mont_progress_fn cb, void* cb_ctx, size_t chunk_bits) {
#ifdef _WIN64
    const auto* api = gwnum_plugin().api;
    if (!api || !bits || !nbits || start_bit > nbits || ((start_bit || (cb && chunk_bits)) && !st) ||
        mpz_cmp_ui(N, 3) < 0 || mpz_even_p(N)) {
        return MONT_LADDER_ERROR;
    }

    char error[256]{};
    const auto fail = [&](size_t bit) {
        ecm_ts_fprintf(stderr, "ERROR: GWNUM Stage1 at bit %zu: %s\n", bit,
                       *error ? error : "DLL operation failed");
        return MONT_LADDER_ERROR;
    };

    try {
        using ecm_edwards::Integer;
        Integer A, X0, Z0, a24, xdiff, inv, scratch;
        mont_suyama_curve(A.v, X0.v, Z0.v, sigma, N);
        mpz_add_ui(a24.v, A.v, 2);
        mpz_set_ui(inv.v, 4);
        mpz_invert(inv.v, inv.v, N);
        mpz_mul(a24.v, a24.v, inv.v);
        mpz_mod(a24.v, a24.v, N);
        // Match the scalar PARAM0 ladder when the starting Z is a non-unit.
        if (mpz_invert(scratch.v, Z0.v, N)) mpz_set(inv.v, scratch.v);
        mpz_mul(xdiff.v, X0.v, inv.v);
        mpz_mod(xdiff.v, xdiff.v, N);

        const auto modulus_bits = mpz_sizeinbase(N, 2);
        if (modulus_bits > UINT32_MAX) return MONT_LADDER_ERROR;
        const auto count = static_cast<uint32_t>((modulus_bits - 1) / 32 + 1);
        const auto p = mpz_scan0(N, 0) == modulus_bits ? static_cast<uint32_t>(modulus_bits) : 0u;
        const auto modulus = export_words(N, count);
        const auto a24_words = export_words(a24.v, count);
        const auto xdiff_words = export_words(xdiff.v, count);
        Words x1, z1;
        if (start_bit) {
            mpz_mod(X0.v, st->X0, N);
            mpz_mod(Z0.v, st->Z0, N);
            mpz_mod(scratch.v, st->X1, N);
            x1 = export_words(scratch.v, count);
            mpz_mod(scratch.v, st->Z1, N);
            z1 = export_words(scratch.v, count);
        } else if (cb && chunk_bits) {
            x1.resize(count);
            z1.resize(count);
        }
        auto x0 = export_words(X0.v, count);
        auto z0 = export_words(Z0.v, count);
        std::unique_ptr<void, decltype(api->destroy)> context(
            api->create(modulus.data(), count, p, a24_words.data(), xdiff_words.data(),
                        x0.data(), z0.data(), start_bit ? x1.data() : nullptr,
                        start_bit ? z1.data() : nullptr, error, sizeof(error)), api->destroy);
        if (!context) return fail(start_bit);

        const bool checkpoints = cb && chunk_bits;
        size_t bit = start_bit ? start_bit : 1;
        size_t end = checkpoints ? start_bit + std::min(chunk_bits, nbits - start_bit) : nbits;
        while (bit < nbits) {
            if (end > bit && api->advance(context.get(), bits + bit, end - bit, error, sizeof(error)))
                return fail(end);
            bit = end;
            if (bit == nbits) break;
            if (api->read(context.get(), x0.data(), z0.data(), x1.data(), z1.data(), error, sizeof(error)))
                return fail(bit);
            import_words(st->X0, x0, N);
            import_words(st->Z0, z0, N);
            import_words(st->X1, x1, N);
            import_words(st->Z1, z1, N);
            st->bitnum = bit;
            if (cb(cb_ctx, st)) return MONT_LADDER_PAUSED;
            end += std::min(chunk_bits, nbits - end);
        }

        if (api->read(context.get(), x0.data(), z0.data(), nullptr, nullptr, error, sizeof(error)))
            return fail(nbits);
        import_words(Qx, x0, N);
        import_words(Qz, z0, N);
        if (factor) {
            mpz_gcd(factor, Qz, N);
            if (mpz_cmp_ui(factor, 1) > 0 && mpz_cmp(factor, N) < 0) return MONT_LADDER_HIT;
        }
        return MONT_LADDER_MISS;
    } catch (const std::exception& exception) {
        ecm_ts_fprintf(stderr, "ERROR: GWNUM Stage1: %s\n", exception.what());
        return MONT_LADDER_ERROR;
    }
#else
    return MONT_LADDER_ERROR;
#endif
}
