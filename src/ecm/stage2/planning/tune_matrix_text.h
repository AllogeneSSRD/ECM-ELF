#pragma once

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace ecm_stage2::tune {
// The first row contains column names. Every cell is already a TOML literal.
inline void write_aligned_matrix(std::ostream& out, const std::vector<std::vector<std::string>>& matrix) {
    std::vector<std::size_t> widths(matrix.front().size());
    for (const auto& row : matrix)
        for (std::size_t i = 0; i < row.size(); ++i)
            widths[i] = (std::max)(widths[i], row[i].size());

    out << "matrix = [\n";
    for (const auto& row : matrix) {
        out << "    [";
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i) out << ", ";
            out << std::setw(static_cast<int>(widths[i])) << row[i];
        }
        out << "],\n";
    }
    out << "]\n";
}
} // namespace ecm_stage2::tune
