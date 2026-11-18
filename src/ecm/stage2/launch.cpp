#include "launch.h"
#include "queue_paths.h"
#include "../application/configuration/generated/ecm_config_generated.h"
#include "ecm/platform/program_version.h"
#include <iostream>

int ecm_stage2::run_combined_stage2(int argc, char** argv, GpuEntry gpu) {
    try {
        std::vector<char*> args{argv[0]};
        std::string config, target;
        int worker = 1;
        bool gpu_command = false, child = false, help = false, version = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--target" || option == "--config" || option == "--worker") {
                if (i + 1 == argc) throw std::runtime_error("missing value for " + option);
                if (option == "--target") {
                    target = ecm_config::lower(argv[++i]);
                    if (target != "cpu" && target != "gpu" && target != "auto")
                        throw std::runtime_error("--target must be auto, cpu or gpu");
                    continue;
                }
                args.push_back(argv[i++]);
                if (option == "--config") config = argv[i];
                else worker = launch_worker(argv[i]);
            } else if (option == "--gui-queue-protocol") {
                std::cout << "ECM_QUEUE_OWNER=1\n";
                return 0;
            } else if (option == "--curve-worker") child = true;
            else if (option == "--gpu-backend") gpu_command = true;
            else if (option == "--version") version = true;
            else if (option == "--help" || option == "-h") help = true;
            args.push_back(argv[i]);
        }
        const int count = static_cast<int>(args.size());
        args.push_back(nullptr);
        // GPU curve children carry a frozen queue route. They never reselect
        // the execution device after a configuration edit.
        if (child) return gpu(count, args.data(), nullptr);
        if (version && target.empty()) {
            std::cout << ecm_program::name << ' ' << ecm_program::version << '\n';
            return 0;
        }
        const auto launch = read_launch_config(config, worker, s2prod::executable());
        if (target.empty()) {
            if (gpu_command) target = "gpu";  // Also pins CUDA/OpenCL worker handoffs.
            else {
                const auto entries = ecm_config::cli_entries(launch.lines, worker);
                for (const auto& entry : entries) {
                    if (entry.first == "stage2_execution_device")
                        target = ecm_config::read_stage2({entry}).execution_device;
                }
            }
        }
        if (help)
            std::cout << "Stage2 target: --target auto|cpu|gpu (TOML: stage2.device; auto selects CPU)\n";
        return target == "gpu" ? gpu(count, args.data(), &launch)
                               : run_cpu_stage2(count, args.data(), &launch);
    } catch (const std::exception& error) {
        std::cerr << "ecm_stage2: " << error.what() << '\n';
        return 2;
    }
}
