#include "ecm/io/results.h"
#include "ecm/integer.h"
#include "io/files.h"
#include "io/json.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace r = ecm_results;

static void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static std::vector<r::json::Value> rows(const fs::path& path) {
    std::vector<r::json::Value> out;
    std::istringstream in(ecm_io::read_file(path));
    std::string line;
    while (std::getline(in, line))
        out.push_back(r::json::parse(line));
    return out;
}

static std::vector<r::Stage1BatchRecord> batch(mpz_srcptr n, const std::vector<uint64_t>& sigmas) {
    std::vector<r::Stage1BatchRecord> out;
    for (const auto sigma : sigmas) {
        r::Submission record;
        record.exponent = 29;
        record.stage1_only = true;
        record.b1 = record.b2 = 20;
        record.sigma = sigma;
        if (sigma == 3) {
            record.stage = 1;
            record.factors = {"233"};
        }
        out.push_back({record, r::internal_json(record, n, 0, "batch-contract",
                                                ",\"stage1-engine\":{\"coverage\":\"B1\"}")});
    }
    return out;
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "output directory required");
        const fs::path dir = argv[1];
        check(!fs::exists(dir), "fresh directory required");
        fs::create_directories(dir);
        ecm_edwards::Integer n;
        mpz_set_ui(n.v, 536870911);
        const auto internal = dir / "stage1.jsonl", submission = dir / "submission.txt";
        const auto checkpoint = dir / "stage1.publish.ckpt";
        const auto publish = [&](const std::vector<uint64_t>& sigmas, const char* identity, bool early) {
            r::publish_stage1_batch(internal, submission, batch(n.v, sigmas), n.v, identity,
                                    checkpoint, early);
        };
        publish({3}, "early-factor", true);
        publish({3}, "early-factor", true);
        check(rows(submission).size() == 1, "early factor was published twice");

        auto blocked = submission;
        blocked += ".publish.tmp";
        fs::create_directory(blocked);
        bool failed = false;
        try {
            publish({100, 3, 20}, "final-batch", false);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        check(failed && rows(submission).size() == 1, "failed replacement changed submission");
        const auto before_internal = ecm_io::read_file(internal);
        const auto internal_rows = rows(internal);
        check(internal_rows.size() == 3 &&
                  internal_rows[1].get("Edwards").get("sigma").text() == "20" &&
                  internal_rows[2].get("Edwards").get("sigma").text() == "100",
              "unsorted internal batch or duplicate early factor");
        check(!internal_rows[1].has("stage1-publication"), "publication state escaped checkpoint");
        fs::remove(blocked);
        publish({100, 3, 20}, "final-batch", false);
        const auto public_rows = rows(submission);
        check(public_rows.size() == 2 && public_rows[0].get("status").text() == "F" &&
                  public_rows[1].get("status").text() == "NF" &&
                  public_rows[1].get("curves").text() == "2",
              "failed retry changed early factor or combined NF");
        check(before_internal == ecm_io::read_file(internal), "retry duplicated internal results");

        fs::rename(submission, dir / "archived.txt");
        publish({100, 3, 20}, "final-batch", false);
        check(!fs::exists(submission), "acknowledged retry recreated archived results");
        const std::string prefix = "manual content without newline";
        ecm_io::write_atomic(submission, prefix);
        r::publish_stage1_batch(internal, submission, batch(n.v, {101}), n.v, "new-batch",
                                dir / "new.publish.ckpt");
        const auto appended = ecm_io::read_file(submission);
        check(appended.compare(0, prefix.size() + 1, prefix + '\n') == 0,
              "manual archive changed existing bytes");

        const auto shared = dir / "shared.txt";
        r::AppendCheckpoint pending;
        const auto noop = [] {};
        const auto write = [&](const std::string& line) {
            r::AppendCheckpoint fresh;
            r::append_checkpointed(shared, line, fresh, noop, noop);
        };
        // Another task may append after the first task saved its intent but
        // before it replaced the output. Both batches must survive.
        write("old\n");
        failed = false;
        try {
            r::append_checkpointed(shared, "first\n", pending,
                [] { throw std::runtime_error("simulated exit after persisting append intent"); },
                noop);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        check(failed && pending.pending, "append intent was not prepared");
        write("second\n");
        auto persisted = pending;
        failed = false;
        try {
            r::append_checkpointed(shared, "first\n", pending, noop,
                [] { throw std::runtime_error("simulated exit before checkpoint acknowledgement"); });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        check(failed && ecm_io::read_file(shared) == "old\nsecond\nfirst\n",
              "resuming append lost another task");
        write("third\n");
        r::append_checkpointed(shared, "first\n", persisted, noop, noop);
        check(ecm_io::read_file(shared) == "old\nsecond\nfirst\nthird\n",
              "retry after replacement duplicated the batch");
        check(!fs::exists(dir / "data"), "publication wrote a runtime data directory");
        std::cout << "result_batch_contract: passed=1\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
