#include "ecm/stage1/selection.h"
#include "../cpu/arithmetic/scalar/modular_mpn_limits.h"
#include <charconv>
#include <utility>

namespace ecm_stage1 {

std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char &ch : out) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return out;
}

bool parse_curve_model(std::string_view value, CurveModel &out) {
    const std::string v = lower_ascii(value);
    if (v == "mont") out = CurveModel::Montgomery;
    else if (v == "ed") out = CurveModel::Edwards;
    else return false;
    return true;
}

bool parse_execution_target(std::string_view value, ExecutionTarget &out) {
    const std::string v = lower_ascii(value);
    if (v == "cpu") out = ExecutionTarget::Cpu;
    else if (v == "gpu") out = ExecutionTarget::Gpu;
    else return false;
    return true;
}

bool parse_cpu_implementation(std::string_view value, CpuImplementation &out) {
    const std::string v = lower_ascii(value);
    if (v == "auto") out = CpuImplementation::Auto;
    else if (v == "simd") out = CpuImplementation::IfmaSimd;
    else if (v == "scalar") out = CpuImplementation::GmpScalar;
    else if (v == "gwnum") out = CpuImplementation::Gwnum;
    else return false;
    return true;
}

bool parse_cpu_implementation_policy(std::string_view value, Selection& selection) {
    std::string normalized;
    while (!value.empty()) {
        if (value.substr(0, 3) == "\xEF\xBC\x9A") { // Full-width colon.
            normalized += ':';
            value.remove_prefix(3);
            continue;
        }
        if (value.substr(0, 3) == "\xE3\x80\x80") { // Ideographic space.
            value.remove_prefix(3);
            continue;
        }
        if (std::string_view(" \t\r\n\f\v").find(value.front()) == std::string_view::npos)
            normalized += value.front();
        value.remove_prefix(1);
    }
    CpuImplementation single = CpuImplementation::Auto;
    if (normalized.empty() || parse_cpu_implementation(normalized, single)) {
        selection.cpu_implementation = single;
        selection.cpu_implementation_rules.clear();
        return true;
    }

    const auto parse_bit = [](std::string_view text, std::size_t& bits) {
        const auto end = text.data() + text.size();
        const auto parsed = std::from_chars(text.data(), end, bits);
        return parsed.ec == std::errc{} && parsed.ptr == end && bits > 0;
    };
    std::vector<CpuImplementationRule> rules;
    std::string_view rest = normalized;
    for (;;) {
        const auto comma = rest.find(',');
        const auto item = rest.substr(0, comma);
        const auto bracket = item.find('[');
        if (item.empty() || bracket == std::string_view::npos || item.back() != ']')
            return false;
        CpuImplementationRule rule{CpuImplementation::Auto, 1, std::nullopt};
        if (!parse_cpu_implementation(item.substr(0, bracket), rule.implementation))
            return false;
        const auto range = item.substr(bracket + 1, item.size() - bracket - 2);
        if (!range.empty()) {
            const auto colon = range.find(':');
            if (colon == std::string_view::npos ||
                (colon && !parse_bit(range.substr(0, colon), rule.first_bit)))
                return false;
            const auto end = range.substr(colon + 1);
            if (!end.empty()) {
                std::size_t bits = 0;
                if (!parse_bit(end, bits) || bits <= rule.first_bit) return false;
                rule.end_bit = bits;
            }
        }
        rules.push_back(rule);
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
    }
    selection.cpu_implementation = CpuImplementation::Auto;
    selection.cpu_implementation_rules = std::move(rules);
    return true;
}

namespace {

std::optional<CpuImplementation> resolve_implementation(
    CpuImplementation requested, CurveModel curve, std::size_t modulus_bits,
    std::uint32_t curves, bool has_ifma) {
    const bool wide = modulus_bits > ecm_cpu::gmp_fixed_width_bits;
    if (requested == CpuImplementation::Gwnum) {
        if (curve != CurveModel::Montgomery)
            return std::nullopt;
        return CpuImplementation::Gwnum;
    }
    if (requested == CpuImplementation::GmpScalar) {
        if (wide && curve == CurveModel::Montgomery)
            return std::nullopt;
        return CpuImplementation::GmpScalar;
    }
    if (wide || requested == CpuImplementation::IfmaSimd) {
        if (!has_ifma)
            return std::nullopt;
        return CpuImplementation::IfmaSimd;
    }
    const auto min_curves = curve == CurveModel::Edwards ? 8u : 2u;
    return has_ifma && curves >= min_curves ? CpuImplementation::IfmaSimd
                                          : CpuImplementation::GmpScalar;
}

const char* implementation_name(CpuImplementation implementation) {
    switch (implementation) {
    case CpuImplementation::Auto: return "auto";
    case CpuImplementation::IfmaSimd: return "simd";
    case CpuImplementation::GmpScalar: return "scalar";
    case CpuImplementation::Gwnum: return "gwnum";
    }
    return "unknown";
}

const char* unavailable_reason(const Selection& selection, CpuImplementation implementation,
                               std::size_t modulus_bits, const CpuImplementationCapabilities& capability) {
    if (selection.cpu_montgomery_chain == CpuMontgomeryChain::Prac &&
        (selection.curve != CurveModel::Montgomery || implementation != CpuImplementation::IfmaSimd ||
         !capability.prac_bound))
        return "PRAC requires Montgomery SIMD and B1 in 2..4294967295";
    if (selection.simd_reduction != SimdReduction::Auto && implementation != CpuImplementation::IfmaSimd)
        return "explicit reduction requires SIMD";
    if (implementation == CpuImplementation::GmpScalar && !capability.odd_modulus &&
        (selection.curve == CurveModel::Montgomery || modulus_bits <= ecm_cpu::gmp_fixed_width_bits))
        return "fixed-width scalar arithmetic requires an odd modulus greater than one";
    if (implementation == CpuImplementation::IfmaSimd) {
        if (!capability.odd_modulus) return "SIMD requires an odd modulus greater than one";
        if (selection.simd_reduction == SimdReduction::MersenneFold && !capability.mersenne_fold)
            return "mersenne-fold requires a full Mersenne modulus of at least 64 bits";
    }
    if (implementation == CpuImplementation::Gwnum) {
        if (!capability.odd_modulus) return "GWNUM requires an odd modulus greater than one";
        if (modulus_bits > UINT32_MAX) return "GWNUM exceeds its supported modulus width";
        return capability.gwnum ? capability.gwnum() : "GWNUM requires Montgomery PARAM0 and a usable DLL";
    }
    return nullptr;
}

} // namespace

std::optional<CpuImplementation> resolve_cpu_implementation(
    const Selection& selection, std::size_t modulus_bits, std::uint32_t curves, bool has_ifma) {
    return resolve_implementation(selection.cpu_implementation, selection.curve, modulus_bits, curves, has_ifma);
}

std::optional<CpuImplementation> resolve_cpu_implementation_policy(
    const Selection& selection, std::size_t modulus_bits, std::uint32_t curves,
    const CpuImplementationCapabilities& capability, std::string& decisions) {
    decisions.clear();
    for (const auto& rule : selection.cpu_implementation_rules) {
        if (!decisions.empty()) decisions += "; ";
        decisions += std::string(implementation_name(rule.implementation)) + '[' +
                     std::to_string(rule.first_bit) + ':';
        if (rule.end_bit) decisions += std::to_string(*rule.end_bit);
        decisions += "] ";
        if (modulus_bits < rule.first_bit || (rule.end_bit && modulus_bits >= *rule.end_bit)) {
            decisions += "outside range";
            continue;
        }
        const auto implementation = resolve_implementation(rule.implementation, selection.curve,
                                                           modulus_bits, curves, capability.has_ifma);
        if (!implementation) {
            if (rule.implementation == CpuImplementation::Gwnum)
                decisions += "requires Montgomery PARAM0";
            else if (rule.implementation == CpuImplementation::GmpScalar)
                decisions += "scalar Montgomery exceeds its supported modulus width";
            else
                decisions += "requires AVX512-F/DQ/IFMA";
            continue;
        }
        if (const char* error = unavailable_reason(selection, *implementation, modulus_bits, capability)) {
            decisions += error;
            continue;
        }
        decisions += std::string("selected ") + implementation_name(*implementation);
        return implementation;
    }
    return std::nullopt;
}

bool parse_simd_reduction(std::string_view value, SimdReduction &out) {
    const std::string v = lower_ascii(value);
    if (v == "auto") out = SimdReduction::Auto;
    else if (v == "mersenne-fold")
        out = SimdReduction::MersenneFold;
    else if (v == "montgomery")
        out = SimdReduction::Montgomery;
    else return false;
    return true;
}

bool parse_cpu_montgomery_chain(std::string_view value, CpuMontgomeryChain& out) {
    const auto mode = lower_ascii(value);
    if (mode == "ladder") out = CpuMontgomeryChain::Ladder;
    else if (mode == "prac") out = CpuMontgomeryChain::Prac;
    else return false;
    return true;
}

bool parse_gpu_algorithm(std::string_view value, GpuAlgorithm& out) {
    const auto v = lower_ascii(value);
    if (v == "1" || v == "+1" || v == "cgbn") out = GpuAlgorithm::Cgbn;
    else if (v == "2" || v == "+2" || v == "ntt") out = GpuAlgorithm::Ntt;
    else return false;
    return true;
}

bool parse_scalar_construction(std::string_view value, ScalarConstruction &out) {
    const std::string v = lower_ascii(value);
    if (v == "1") out = ScalarConstruction::PrimePowerLcm;
    else if (v == "12")
        out = ScalarConstruction::PrimePowerLcmTimes12;
    else return false;
    return true;
}

bool valid_route(const Selection &selection, bool gpu_edwards) {
    return selection.target != ExecutionTarget::Gpu ||
           selection.curve == CurveModel::Montgomery || gpu_edwards;
}

bool valid_gpu_param(int value) { return value == 0 || value == 2 || value == 3; }

// -sigma i:s selects i unless --gpu-param explicitly selects another value.
bool resolve_gpu_param(int configured, bool explicit_param, int sigma_claim,
                              int &resolved) {
    if (!valid_gpu_param(configured) ||
        (sigma_claim >= 0 && !valid_gpu_param(sigma_claim))) return false;
    if (sigma_claim >= 0 && explicit_param && sigma_claim != configured) return false;
    resolved = sigma_claim >= 0 ? sigma_claim : configured;
    return true;
}

const char *curve_model_name(CurveModel curve) {
    return curve == CurveModel::Edwards ? "ed" : "mont";
}

const char *execution_target_name(ExecutionTarget target) {
    return target == ExecutionTarget::Gpu ? "gpu" : "cpu";
}

} // namespace ecm_stage1
