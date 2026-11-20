#include "ecm/integer.h"
#include "ecm/io/results.h"
#include "ecm/io/stage2_checkpoint.h"
#include "io/files.h"
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace r = ecm_results;

static void append(const fs::path& path, const std::string& report) {
    r::AppendCheckpoint checkpoint;
    const auto noop = [] {};
    r::append_checkpointed(path, report, checkpoint, noop, noop);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            return 2;
        const fs::path out = argv[1];
        fs::create_directories(out);
        ecm_edwards::Integer prime, composite;
        mpz_ui_pow_ui(prime.v, 2, 127);
        mpz_sub_ui(prime.v, prime.v, 1);
        mpz_set_ui(composite.v, 536870911);
        r::Submission value;
        value.exponent = 127;
        value.b1 = value.b2 = 20;
        value.stage1_only = true;
        std::vector<r::Stage1BatchRecord> records;
        for (unsigned sigma : {900, 70, 1000}) {
            value.sigma = sigma;
            records.push_back({value, r::internal_json(value, prime.v, 0, "fixture",
                                                       ",\"stage1-engine\":{}")});
        }
        r::publish_stage1_batch(out / "internal.jsonl", out / "stage1.json.txt", records,
                                prime.v, "stage1-fixture", out / "stage1.publish.ckpt");
        value.stage1_only = false;
        value.b2 = 2000;
        append(out / "stage2.json.txt", r::aggregate_nf(r::submission_json(value, prime.v), 3) + '\n');

        stage2_queue::State state;
        for (unsigned i = 0; i < 3; ++i) {
            value.when = "2026-10-20 12:00:0" + std::to_string(i);
            value.d = 30030 + 2 * i;
            value.poly_size = 96 + 32 * i;
            state.record_outcome(i, r::submission_json(value, prime.v), std::string(64, char('a' + i)));
        }
        append(out / "geometry.json.txt", r::aggregate_nf(state.latest_nf, state.nf_keys.size()) + '\n');

        auto factor = value;
        factor.exponent = 29;
        factor.stage = 1;
        factor.d = factor.poly_size = 0;
        factor.factors = {"233"};
        const auto hit = r::submission_json(factor, composite.v);
        append(out / "factor.json.txt", hit);
        append(out / "mixed.json.txt", hit);
        auto nf = value;
        nf.exponent = 29;
        append(out / "mixed.json.txt", r::submission_json(nf, composite.v));

        for (int stage : {1, 2}) {
            const auto path = out / ("append" + std::to_string(stage) + ".json.txt");
            auto next = value;
            next.stage1_only = stage == 1;
            if (next.stage1_only) {
                next.b2 = next.b1;
                next.d = next.poly_size = 0;
            }
            for (unsigned i = 0; i < 3; ++i) {
                const auto prefix = fs::exists(path) ? ecm_io::read_file(path) : std::string{};
                next.when = "2026-10-21 12:00:0" + std::to_string(2 - i);
                const auto report = r::submission_json(next, prime.v);
                append(path, report);
                if (ecm_io::read_file(path) != prefix + report)
                    throw std::runtime_error("append changed existing result bytes");
            }
        }
        const auto alternating = out / "alternating.json.txt";
        append(alternating, r::submission_json(nf, composite.v));
        append(alternating, hit);
        nf.when = "2026-10-20 11:00:00";
        append(alternating, r::submission_json(nf, composite.v));

        const auto receipt = out / "receipt.jsonl";
        ecm_io::write_atomic(receipt, "{\"status\":\"NF\",\"queue_receipt\":\"ready\"}\n{\"queue_receipt\":\"torn\"}");
        if (!stage2_queue::result_written(receipt, "ready") ||
            stage2_queue::result_written(receipt, "torn"))
            throw std::runtime_error("invalid diagnostic receipt handling");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
