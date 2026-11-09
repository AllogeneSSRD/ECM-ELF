// Synthetic Stage1 ladder states let the production selector run at exact
// exponent boundaries where small factors can change the number of live curves.
#include "../../src/ecm/stage1/montgomery/cuda/ntt/ladder.h"
#include "../../src/ecm/stage1/montgomery/cuda/ntt/tune.h"
#include "../../src/io/files.h"
#include "ecm/platform/program_version.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::printf("ECM-ELF %s\n", ecm_program::version);
        return 0;
    }
    const bool experiment = argc >= 3 && std::string_view(argv[argc-2]) == "--dataflow";
    const int plan_argc = argc - (experiment ? 2 : 0);
    if (plan_argc != 6 && plan_argc != 13) {
        std::fprintf(stderr, "stage1_ntt_auto_probe device p batch steps tune.toml "
                             "[weighted length bpw tile outer window warp] [--dataflow 0..3]\n");
        return 2;
    }
    try {
        const int device = std::stoi(argv[1]);
        const unsigned p = std::stoul(argv[2]), batch = std::stoul(argv[3]), steps = std::stoul(argv[4]);
        if (p < 32 || p > 262144 || !batch || batch > ecm_stage1::stage1_ntt_max_curves ||
            !steps || steps > 1000000)
            throw std::invalid_argument("invalid benchmark width, batch or step count");
        ecm_stage1::Stage1NttOptions options;
        options.tune_file = argv[5];
        if (plan_argc == 13) {
            options.automatic = false;
            options.weighted = std::stoi(argv[6]) != 0;
            options.length = std::stoul(argv[7]);
            options.bpw = std::stoul(argv[8]);
            options.tile = std::stoul(argv[9]);
            options.outer_max = std::stoul(argv[10]);
            options.window = std::stoi(argv[11]);
            options.warp_tail = std::stoi(argv[12]);
        }
        const auto selected = ecm_mersenne_ntt::select_tuned_plan(device, p, batch, options);
        const unsigned words = (p + 31) / 32;
        std::mt19937 random(0x67ae128bu);
        std::vector<uint32_t> scalar((steps + 31) / 32), initial(size_t(6) * batch * words);
        for (auto& word : scalar) word = random();
        for (size_t i = 0; i < initial.size(); i += words) {
            for (unsigned j = 0; j < words; ++j) initial[i + j] = random();
            if (p % 32) initial[i + words - 1] &= (1u << (p % 32)) - 1;
            initial[i] &= 0xfffffffeu;
        }
        ecm_mersenne_ntt::Control control;
        if (experiment) {
            const std::string_view mode = argv[argc-1];
            if (mode.size()!=1 || mode[0]<'0' || mode[0]>'3')
                throw std::invalid_argument("invalid experimental dataflow");
            control.experimental_dataflow = unsigned(mode[0]-'0');
            control.warmup_iterations = std::min(100u, steps);
        }
        control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
            std::printf("stage1_ntt_plan: device=%d p=%u curves=%u length=%u bpw=%u "
                        "tile_layers=%u outer_max_layers=%u graph_bits=%u warp_tail=%u "
                        "forward_passes=%u forward_layers=%s inverse_layers=%s convolution=%s\n",
                        device, p, batch, plan.length, plan.bpw, plan.tile, plan.outer_max,
                        plan.window, unsigned(plan.warp_tail), plan.passes, plan.forward.c_str(),
                        plan.inverse.c_str(), plan.weighted ? "weighted" : "linear");
        };
        const auto run = ecm_mersenne_ntt::ladder(device, p, batch, initial, scalar, steps, selected, control);
        if (experiment) {
            const auto requested = control.experimental_dataflow;
            if (requested) {
                ecm_mersenne_ntt::Control baseline;
                const auto reference = ecm_mersenne_ntt::ladder(device, p, batch, initial, scalar, steps, selected, baseline);
                if (reference.points != run.points) throw std::runtime_error("experimental ladder endpoint differs");
            }
            std::printf("stage1_ntt_dataflow: requested=%u actual=%u verified_words=%zu warmup=%u\n",
                        requested,run.dataflow,requested ? run.points.size() : 0,control.warmup_iterations);
        }
        const std::string bytes(reinterpret_cast<const char*>(run.points.data()),
                                run.points.size() * sizeof(uint32_t));
        std::printf("stage1_ntt_timing: bits=%u device_ms=%.6f\nstage1_ntt_state: hash=%016llx\n",
                    run.completed_bits, run.device_ms.front(),
                    static_cast<unsigned long long>(ecm_io::fingerprint(bytes)));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "stage1_ntt_auto_probe: %s\n", e.what());
        return 1;
    }
}
