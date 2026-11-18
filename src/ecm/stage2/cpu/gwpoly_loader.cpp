#include "gwpoly.h"
#include <windows.h>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecm_gwpoly {
namespace {
struct Plugin {
    HMODULE module{};
    const ecm_gw_stage2_api* api{};
    const ecm_gw_stage2_api_v2* api_v2{};
    const ecm_gw_stage2_api_v3* api_v3{};
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
        const auto header = get(2);
        if (header && header->abi == 2 && header->size >= sizeof(ecm_gw_stage2_api_v2)) {
            auto v2 = reinterpret_cast<const ecm_gw_stage2_api_v2*>(header);
            if (v2->run_with_phases && v2->tune_candidates) api_v2 = v2;
        }
        const auto streamed = get(3);
        if (api_v2 && streamed && streamed->abi == 3 && streamed->size >= sizeof(ecm_gw_stage2_api_v3)) {
            const auto v3 = reinterpret_cast<const ecm_gw_stage2_api_v3*>(streamed);
            if (v3->run_with_progress) api_v3 = v3;
        }
    }
    ~Plugin() { if (module) FreeLibrary(module); }
};
const Plugin& plugin() { static const Plugin instance; return instance; }
const ecm_gw_stage2_api& api() {
    if (!plugin().api) throw std::runtime_error(plugin().error);
    return *plugin().api;
}
uint32_t word_count(mpz_srcptr n) {
    const size_t words = (mpz_sizeinbase(n, 2) + 31) / 32;
    if (words > UINT32_MAX || mpz_cmp_ui(n,2)<=0) throw std::invalid_argument("Invalid Stage2 modulus");
    return static_cast<uint32_t>(words);
}
std::vector<uint32_t> encode(mpz_srcptr z, uint32_t words) {
    std::vector<uint32_t> result(words);
    mpz_export(result.data(), nullptr, -1, sizeof(uint32_t), 0, 0, z);
    return result;
}
}
const char* load_error() { return plugin().api ? nullptr : plugin().error.c_str(); }
unsigned stage2_abi() { return plugin().api_v3 ? 3 : plugin().api_v2 ? 2 : plugin().api ? 1 : 0; }
const char* version() { return api().version; }
const char* cpu() { return api().cpu; }
const char* identity() { return api().build; }

void selftest(unsigned threads) {
    char error[512]{};
    if (api().selftest(threads, error, sizeof(error)) < 0) throw std::runtime_error(error);
}

bool stage2(mpz_t factor, const mpz_t n, const mpz_t x, const mpz_t a24,
            uint64_t b1, uint64_t b2, const Options& o, Stats& stats) {
    const auto words = word_count(n);
    if (mpz_sgn(x)<0 || mpz_cmp(x,n)>=0 || mpz_sgn(a24)<0 || mpz_cmp(a24,n)>=0)
        throw std::invalid_argument("Invalid Stage2 input");
    auto modulus = encode(n,words), point = encode(x,words), curve = encode(a24,words);
    std::vector<uint32_t> output(words);
    ecm_gw_stage2_request request{words, o.threads, o.check,
        b1,b2,o.d,o.exponent,o.memory_mib,modulus.data(),point.data(),curve.data(),o.stop};
    stats = {};
    char error[512]{};
    auto v2 = plugin().api_v2;
    int rc;
    if (const auto v3 = plugin().api_v3)
        rc = v3->run_with_progress(&request,output.data(),&stats,&stats.phases,
                                   o.progress,o.progress_context,error,sizeof(error));
    else if (v2)
        rc = v2->run_with_phases(&request,output.data(),&stats,&stats.phases,error,sizeof(error));
    else
        rc = api().run(&request,output.data(),&stats,error,sizeof(error));
    if (rc < 0) throw std::runtime_error(error);
    if (rc == 1) mpz_import(factor, words, -1, sizeof(uint32_t), 0, 0, output.data());
    return rc == 1;
}

std::vector<TuneCandidate> tune_candidates(mpz_srcptr n, uint64_t b1, uint64_t b2, const Options& o) {
    auto v2 = plugin().api_v2;
    if (!v2) return {};
    const auto words = word_count(n);
    auto modulus = encode(n,words);
    ecm_gw_stage2_request request{words,o.threads,0,b1,b2,o.d,o.exponent,o.memory_mib,
                                  modulus.data(),nullptr,nullptr,o.stop};
    std::array<TuneCandidate,3> candidates{};
    char error[512]{};
    const int count = v2->tune_candidates(&request,candidates.data(),
                                          static_cast<uint32_t>(candidates.size()),error,sizeof(error));
    if (count < 0) throw std::runtime_error(error);
    if (static_cast<size_t>(count) > candidates.size())
        throw std::runtime_error("Invalid GWNUM Stage2 candidate count");
    return {candidates.begin(),candidates.begin()+count};
}
}
