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
};

std::optional<OwlSave> load_owl(const std::filesystem::path& root, unsigned p,
                                uint64_t b1, uint64_t bits);
void write_owl(const std::filesystem::path& root, unsigned p, uint64_t b1,
               uint64_t b2, uint64_t bits, const OwlSave& save, unsigned backups = 3);
std::string prime95_name(unsigned p);
std::filesystem::path write_prime95(const std::filesystem::path& dir, unsigned p,
                                    uint64_t b1, const std::vector<uint32_t>& residue,
                                    uint64_t* fingerprint = nullptr);
} // namespace ecm_pm1
