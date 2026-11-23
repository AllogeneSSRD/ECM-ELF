#pragma once
#include "ecm/io/publication.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <gmp.h>

namespace ecm_results {
struct Submission {
    uint64_t exponent = 0, sigma = 0, b1 = 0, b2 = 0, curves = 1;
    uint64_t b2_min = 0;
    uint64_t d = 0, poly_size = 0,
             stage2_fft_length = 0; // Actual completed Stage2 geometry; zero means unavailable.
    int stage = 0;
    bool edwards = true;
    int param = 0;
    bool stage1_only = false, kbnc = false;
    std::string k = "1";
    uint32_t b = 2;
    int32_t c = -1;
    std::vector<std::string> factors, known;
    std::string user, computer, aid, when;
};

struct Stage1BatchRecord {
    Submission submission;
    std::string internal;
};

std::string quoted(const std::string&);
std::string decimal(mpz_srcptr);
std::string utc_timestamp();
uint64_t target_exponent(const Submission&, mpz_srcptr);
std::string submission_json(const Submission&, mpz_srcptr);
std::string internal_json(Submission, mpz_srcptr, uint64_t, const std::string&,
                          const std::string& extra_fields = {});
std::string aggregate_nf(const std::string&, uint64_t count);
std::string factor_identity(const std::string&);
std::filesystem::path stage1_publication_checkpoint(const std::filesystem::path&);
void discard_completed_stage1_publication(const std::filesystem::path&);
void publish_stage1_batch(const std::filesystem::path&, const std::filesystem::path&,
                          std::vector<Stage1BatchRecord>, mpz_srcptr, const std::string&,
                          const std::filesystem::path& checkpoint, bool early = false);
bool contains_stage1_curves(const std::filesystem::path&, mpz_srcptr, uint64_t, bool, unsigned,
                            const std::vector<uint64_t>&);
// Append one opaque CPU Stage2 result, retaining the previous text de-duplication contract.
void append_text_record(const std::filesystem::path&, const std::string&);
} // namespace ecm_results
