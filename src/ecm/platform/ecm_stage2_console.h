#pragma once
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

// Console projection of existing engine output. The parent writes each original
// line to its log before calling this helper; derived lines never enter that log.
// No CUDA calls or estimates are used for the end-of-curve allocation summary.
namespace stage2_console {
struct Projection {
    bool replace = false;
    std::vector<std::string> lines;
};
inline bool prefix(const std::string &line, const char *value) {
    return line.compare(0, std::char_traits<char>::length(value), value) == 0;
}
inline bool integer_field(const std::string &line, const char *key, uint64_t &value) {
    const std::string token = std::string(" ") + key + '=';
    const auto pos = line.find(token);
    if (pos == line.npos) return false;
    size_t i = pos + token.size();
    if (i == line.size() || line[i] < '0' || line[i] > '9') return false;
    uint64_t parsed = 0;
    for (; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        const unsigned digit = static_cast<unsigned>(line[i] - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
        parsed = parsed * 10 + digit;
    }
    if (i < line.size() && line[i] != ' ' && line[i] != '\t') return false;
    value = parsed;
    return true;
}
inline std::string mib(uint64_t bytes) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3) << bytes / 1048576.0;
    return out.str();
}
class Summary {
    uint64_t batch_, ini_arena_, ini_fold_;
    uint64_t arena_peak_ = 0, fold_peak_ = 0;
    bool shape_seen_ = false, arena_seen_ = false, fold_seen_ = false, memory_printed_ = false;

    void append_memory(Projection &result) {
        if (memory_printed_ || !shape_seen_) return;
        result.lines.push_back("mem: batch=" + std::to_string(batch_) +
            " arena=" + (arena_seen_ ? mib(arena_peak_) : "n/a") + '/' + std::to_string(ini_arena_) +
            " fold=" + (fold_seen_ ? mib(fold_peak_) : "n/a") + '/' + std::to_string(ini_fold_));
        memory_printed_ = true;
    }
public:
    Summary(uint64_t batch, uint64_t ini_arena, uint64_t ini_fold)
        : batch_(batch), ini_arena_(ini_arena), ini_fold_(ini_fold) {}

    Projection observe(const std::string &line) {
        Projection result;
        if (prefix(line, "real_shape: ")) {
            uint64_t d=0, p=0, baby=0, giant=0, polynomials=0, loops=0, b1=0, b2=0, bits=0;
            // P is emitted as "P=phi(D)/2=<integer>" by the engine.
            if (integer_field(line,"D",d) && integer_field(line,"P=phi(D)/2",p) &&
                integer_field(line,"baby_points",baby) && integer_field(line,"giant_points",giant) &&
                integer_field(line,"num_poly_g",polynomials) && integer_field(line,"loops",loops) &&
                integer_field(line,"B1",b1) && integer_field(line,"B2",b2) && integer_field(line,"S_bits",bits)) {
                result.replace = shape_seen_ = true;
                result.lines.push_back("real_shape: B1=" + std::to_string(b1) + " B2=" + std::to_string(b2) +
                    " S_bits=" + std::to_string(bits));
                result.lines.push_back("real_shape: D=" + std::to_string(d) + " P=" + std::to_string(p) +
                    " baby=" + std::to_string(baby) + " giant=" + std::to_string(giant) +
                    " poly_g=" + std::to_string(polynomials) + " loops=" + std::to_string(loops));
            }
        } else if (prefix(line, "mem_budget: ")) {
            double free=0, total=0, reserve=0, cap=0, fold_mb=0, tree_mb=0;
            unsigned long long p=0, fold_coefficients=0, tree_coefficients=0;
            const int count = std::sscanf(line.c_str(),
                "mem_budget: free=%lf MB total=%lf MB reserve=%lf MB arena_cap=%lf MB ; "
                "P=%llu => largest transform (the fold, operand %llu coeffs) = %lf MB ; "
                "tree top (operand %llu coeffs) = %lf MB",
                &free,&total,&reserve,&cap,&p,&fold_coefficients,&fold_mb,&tree_coefficients,&tree_mb);
            if (count == 9) {
                result.replace = true;
                std::ostringstream budget, transforms;
                budget.imbue(std::locale::classic());transforms.imbue(std::locale::classic());
                budget << std::fixed << std::setprecision(0) << "mem: free=" << free << " MB total=" << total
                    << " MB reserve=" << reserve << " MB arena_cap=" << cap << " MB";
                transforms << std::fixed << std::setprecision(0) << "mem: P=" << p
                    << " -> largest transform (" << fold_coefficients << " coeffs) = " << fold_mb
                    << " MB, tree top (" << tree_coefficients << " coeffs) = " << tree_mb << " MB";
                result.lines.push_back(budget.str());result.lines.push_back(transforms.str());
            }
        } else if (prefix(line, "real_batched_folddevice: ")) {
            uint64_t enabled=0, bytes=0;
            if (integer_field(line,"enabled",enabled) &&
                (!enabled || integer_field(line,"peak_bytes",bytes))) {
                fold_seen_ = true;fold_peak_ = enabled ? bytes : 0;
            }
        } else if (prefix(line, "ntt_workspace_stats: ")) {
            uint64_t bytes=0;
            if (integer_field(line,"full_peak_bytes",bytes)) {
                arena_seen_ = true;arena_peak_ = bytes;
            }
        }
        if (prefix(line,"stage2_result: ")) append_memory(result);
        return result;
    }
    Projection finish() {
        Projection result;
        append_memory(result);
        return result;
    }
};
} // namespace stage2_console
