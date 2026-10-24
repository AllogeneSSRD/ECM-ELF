#pragma once
#include "../../core/group_order/group_order.h"
#include <string>

namespace ecm_diagnostics {
// Auto: PARI DLL first, then external GP. GMP is explicit-only.
// Native is retained as the CLI spelling for the isolated PARI DLL engine.
enum class Backend { Auto, Gp, Native, Gmp };
Backend parse_backend(const std::string &text);
const char *name(Backend backend);
struct Settings {
    Backend backend = Backend::Auto;
    std::string gp = "gp";
    uint64_t gp_timeout_ms = 30000;
    uint64_t gp_stack_bytes = 128 * 1024 * 1024;
    // Empty resolves the host beside the calling executable, never the CWD/PATH.
    std::string pari_host;
    uint64_t pari_timeout_ms = 5000;
    uint64_t native_total_ms = 10000;
    ecm_group_order::Limits native;
};
struct GroupResult {
    ecm_group_order::Analysis value;
    std::string backend, gp_failure, gp_output, pari_failure;
};
struct FactorResult {
    ecm_group_order::Factorization value;
    std::string backend, gp_failure, gp_output, pari_failure;
};
// One session per task: memoizes unavailable GP and bounds cumulative native work.
// All diagnosis failures are data; callers publish primary factors independently.
class Session {
    Settings settings_;
    std::string unavailable_;
    double native_ms_ = 0;
public:
    explicit Session(Settings settings = {});
    GroupResult group(const ecm_group_order::CurveRequest &request);
    FactorResult factor(const ecm_group_order::Integer &integer);
};
std::string quote(const std::string &text);
} // namespace ecm_diagnostics
