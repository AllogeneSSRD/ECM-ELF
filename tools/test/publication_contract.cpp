#include "ecm/io/results.h"
#include "ecm/integer.h"
#include "io/files.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace r = ecm_results;
void publication_observe(const fs::path&, int);
bool publication_unflushed();

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static std::vector<r::Stage1BatchRecord> batch(mpz_srcptr n, uint64_t first, unsigned count) {
    std::vector<r::Stage1BatchRecord> result;
    for (unsigned i = 0; i < count; ++i) {
        r::Submission s;
        s.exponent = 29;
        s.stage1_only = true;
        s.b1 = s.b2 = 20;
        s.sigma = first + i;
        s.when = "2026-10-26 00:00:00";
        result.push_back({s, r::internal_json(s, n, 0, "publication-contract",
                                            ",\"stage1-engine\":{}")});
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "MODE NEW_DIRECTORY required");
        const std::string mode = argv[1];
        const auto dir = fs::absolute(argv[2]);
        require(!fs::exists(dir), "fresh directory required");
        fs::create_directories(dir);
        const auto internal = dir / "internal.jsonl", submission = dir / "results.txt";
        ecm_edwards::Integer n;
        mpz_set_ui(n.v, 536870911);
        r::publish_stage1_batch(internal, submission, batch(n.v, 100, 1), n.v, "first");
        const auto before = ecm_io::read_file(internal);
        const auto before_public = ecm_io::read_file(submission);
        const int fault = mode == "flush-failure" ? 1 : mode == "rename-failure" ? 2
                                                     : mode == "uncertain" ? 3 : 0;
        publication_observe(internal, fault);
        bool failed = false, uncertain = false;
        try {
            r::publish_stage1_batch(internal, submission, batch(n.v, 200, 2), n.v, "next");
        } catch (const ecm_io::PublicationUncertain&) {
            failed = uncertain = true;
        } catch (const std::exception&) {
            failed = true;
        }
        if (!fault) {
            require(!failed && !publication_unflushed(), "final replacement preceded explicit file flush");
        } else if (fault < 3) {
            require(failed && !uncertain, "pre-publication failure was not reported");
            require(ecm_io::read_file(internal) == before &&
                        ecm_io::read_file(submission) == before_public,
                    "failed publication changed the final pair");
        } else {
            require(uncertain && ecm_io::read_file(internal) != before &&
                        ecm_io::read_file(submission) == before_public,
                    "visible replacement lost uncertain acknowledgement");
        }
        publication_observe(internal, 0);
        r::publish_stage1_batch(internal, submission, batch(n.v, 200, 2), n.v, "next");
        const auto final = ecm_io::read_file(internal);
        require(std::count(final.begin(), final.end(), '\n') == 3, "retry lost or duplicated curves");
        std::cout << "publication_contract: mode=" << mode << " passed=1\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
