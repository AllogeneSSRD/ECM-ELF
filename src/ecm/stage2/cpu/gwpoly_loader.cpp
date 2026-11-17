#include "gwpoly.h"
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecm_gwpoly {
namespace {
struct Plugin {
    HMODULE module{};
    const ecm_gw_stage2_api* api{};
    std::string error;
    Plugin() {
        wchar_t executable[32768];
        const auto size = GetModuleFileNameW(nullptr, executable, 32768);
        if (!size || size == 32768) { error = "Cannot locate CPU worker"; return; }
        auto path = std::filesystem::path(executable).parent_path() / L"gwnum.dll";
        module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) { error = "Cannot load adjacent gwnum.dll, Windows error " + std::to_string(GetLastError()); return; }
        auto get = reinterpret_cast<ecm_gw_stage2_get_api_fn>(GetProcAddress(module, "ecm_gw_stage2_get_api"));
        auto table = get ? get(1) : nullptr;
        if (!table || table->abi != 1 || table->size < sizeof(*table) || !table->version ||
            !table->build || !table->cpu || !table->run || !table->selftest) {
            error = "gwnum.dll lacks compatible Stage2 ABI 1"; return;
        }
        api = table;
    }
    ~Plugin() { if (module) FreeLibrary(module); }
};
const Plugin& plugin() { static const Plugin instance; return instance; }
const ecm_gw_stage2_api& api() {
    if (!plugin().api) throw std::runtime_error(plugin().error);
    return *plugin().api;
}
}
const char* load_error() { return plugin().api ? nullptr : plugin().error.c_str(); }
const char* version() { return api().version; }
const char* cpu() { return api().cpu; }
const char* identity() { return api().build; }

void selftest(unsigned threads) {
    char error[512]{};
    if (api().selftest(threads, error, sizeof(error)) < 0) throw std::runtime_error(error);
}

bool stage2(mpz_t factor, const mpz_t n, const mpz_t x, const mpz_t a24,
            uint64_t b1, uint64_t b2, const Options& o, Stats& stats) {
    const size_t words = (mpz_sizeinbase(n, 2) + 31) / 32;
    if (words > UINT32_MAX || mpz_sgn(x)<0 || mpz_cmp(x,n)>=0 || mpz_sgn(a24)<0 || mpz_cmp(a24,n)>=0)
        throw std::invalid_argument("Invalid Stage2 input");
    auto encode = [&](mpz_srcptr z) {
        std::vector<uint32_t> result(words);
        mpz_export(result.data(), nullptr, -1, sizeof(uint32_t), 0, 0, z);
        return result;
    };
    auto modulus = encode(n), point = encode(x), curve = encode(a24);
    std::vector<uint32_t> output(words);
    ecm_gw_stage2_request request{static_cast<uint32_t>(words), o.threads, o.check,
        b1,b2,o.d,o.exponent,o.memory_mib,modulus.data(),point.data(),curve.data(),o.stop};
    char error[512]{};
    const int rc = api().run(&request, output.data(), &stats, error, sizeof(error));
    if (rc < 0) throw std::runtime_error(error);
    if (rc == 1) mpz_import(factor, words, -1, sizeof(uint32_t), 0, 0, output.data());
    return rc == 1;
}
}
