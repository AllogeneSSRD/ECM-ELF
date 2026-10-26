#include "io/files.h"
#include <fstream>
#include <stdexcept>

// Compile the complete production save module and change only read scheduling.
namespace {
std::filesystem::path history_source;
std::string replacement;
bool swapped = false;
}

void test_history_swap(const std::filesystem::path& path, const std::string& bytes) {
    history_source = path;
    replacement = bytes;
    swapped = false;
}

bool test_history_swapped() { return swapped; }

namespace ecm_io {
std::string observed_history_read(const std::filesystem::path& path) {
    if (!swapped && path == history_source) {
        // The baseline keeps its first ifstream open, which can block rename on
        // Windows. A controlled shared write exposes the two-read window without
        // mistaking a blocked rename for a successful source replacement.
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << replacement;
        out.close();
        if (!out)
            throw std::runtime_error("history fixture could not replace source bytes");
        swapped = true;
    }
    return read_file(path);
}
}

#define read_file observed_history_read
#include "../../../src/io/save.cpp"
#undef read_file
