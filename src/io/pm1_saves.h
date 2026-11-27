#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ecm_pm1 {
struct OwlSave {
    uint64_t k = 0;   // Number of exponent bits consumed, including the leading one.
    uint32_t errors = 0;
    double seconds = 0;
    std::vector<uint32_t> residue;   // Exactly ceil(p/32) words, low word first.
    uint64_t bits = 0;              // Declared exponent length on import.
    uint64_t b2 = 0;
};

std::optional<OwlSave> load_owl(const std::filesystem::path& root, unsigned p,
                                uint64_t b1, uint64_t bits);
void write_owl(const std::filesystem::path& root, unsigned p, uint64_t b1,
               uint64_t b2, uint64_t bits, const OwlSave& save, unsigned backups = 3);
std::string prime95_name(unsigned p);
struct Prime95Save {
    uint64_t b1, fingerprint;
    std::vector<uint32_t> residue;
    bool done = false;
};
std::optional<Prime95Save> load_prime95(const std::filesystem::path& path, unsigned p);
struct ExtensionSave {
    uint64_t from = 0, bits = 0, k = 1;
    uint32_t errors = 0;
    std::vector<uint32_t> base, residue;
};
std::optional<ExtensionSave> load_extension(const std::filesystem::path& root, unsigned p, uint64_t b1);
void write_extension(const std::filesystem::path& root, unsigned p, uint64_t b1, const ExtensionSave& save);
struct ExtensionProgress {
    uint64_t from, old_iterations;
};
std::optional<ExtensionProgress> load_extension_progress(const std::filesystem::path& root, unsigned p, uint64_t b1);
void write_extension_progress(const std::filesystem::path& root, unsigned p, uint64_t b1,
                              const ExtensionProgress& progress);
std::filesystem::path write_prime95(const std::filesystem::path& dir, unsigned p,
                                    uint64_t b1, const std::vector<uint32_t>& residue,
                                    uint64_t* fingerprint = nullptr, const Prime95Save* previous = nullptr);
} // namespace ecm_pm1
