#pragma once
#include "../../core/group_order/group_order.h"
#include <string>

namespace ecm_diagnostics {
enum class Backend { Auto, Gp, Native };
Backend parse_backend(const std::string &text);
const char *name(Backend backend);
struct Settings {
    Backend backend = Backend::Auto;
    std::string gp = "gp";
    uint64_t gp_timeout_ms = 30000;
    uint64_t native_total_ms = 10000;
    ecm_group_order::Limits native;
};
struct GroupResult {
    ecm_group_order::Analysis value;
    std::string backend, gp_failure, gp_output;
};
struct FactorResult {
    ecm_group_order::Factorization value;
    std::string backend, gp_failure, gp_output;
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
