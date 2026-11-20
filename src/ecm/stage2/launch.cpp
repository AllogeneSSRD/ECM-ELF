#include "launch.h"
#include "queue_paths.h"
#include "cpu/gwpoly.h"
#include "../application/configuration/generated/ecm_config_generated.h"
#include "ecm/platform/gpu_worker.h"
#include "ecm/platform/program_version.h"
#include <iostream>

namespace {
using ecm_stage2::Algorithm;
using namespace s2prod;

std::size_t modulus_bits(const Record& record) {
    Big n;
    mpz_set_str(n.z, record.n.c_str(), 16);
    return mpz_sizeinbase(n.z, 2);
}

struct Candidate {
    Algorithm algorithm;
    const char* cpu_backend = nullptr;
    const char* gpu_backend = nullptr;
};

Candidate choose_algorithm(const std::vector<ecm_stage2::AlgorithmRule>& rules,
                           const Record& record, uint64_t exponent, const std::string& gpu_mode,
                           int device, const fs::path& executable, std::size_t bits) {
    std::optional<bool> cuda, opencl;
    for (const auto& rule : rules) {
        if (!rule.matches(bits)) continue;
        switch (rule.algorithm) {
        case Algorithm::Gwnum:
            if (record.edwards && !ecm_gwpoly::load_error()) return {Algorithm::Gwnum, "gwpoly"};
            break;
        case Algorithm::CpuNtt:
            if (record.edwards) return {Algorithm::CpuNtt, "ntt"};
            break;
        case Algorithm::CpuPlain:
            if (record.edwards) return {Algorithm::CpuPlain, "plain"};
            break;
        case Algorithm::Cpu:
            if (record.edwards) return {Algorithm::Cpu};
            break;
        case Algorithm::Gpu:
        case Algorithm::Cuda:
        case Algorithm::OpenCl: {
            if (bits > ecm_stage2::max_input_bits || exponent > ecm_stage2::max_input_bits) break;
            const auto mode = rule.algorithm == Algorithm::Gpu ? gpu_mode :
                rule.algorithm == Algorithm::Cuda ? "cuda" : "opencl";
            if (mode != "opencl") {
                if (!cuda) cuda = ecm_gpu::cuda_visible(device);
                if (*cuda) return {Algorithm::Cuda, nullptr, "cuda"};
            }
            if (mode != "cuda") {
                if (!opencl)
                    opencl = fs::is_regular_file(executable.parent_path() / "ecm_opencl_stage2.exe");
                if (*opencl) return {Algorithm::OpenCl, nullptr, "opencl"};
            }
            break;
        }
        }
    }
    throw std::runtime_error("stage2_algorithm has no usable rule for " + std::to_string(bits) +
        "-bit " + (record.edwards ? "Edwards" : "Montgomery") +
        " input (GPU input and worktodo exponent limit: " + std::to_string(ecm_stage2::max_input_bits) + " bits)");
}
}

int ecm_stage2::run_combined_stage2(int argc, char** argv, GpuEntry gpu) {
    try {
        std::vector<char*> args{argv[0]};
        std::string config, target;
        int worker = 1;
        std::string save, worktodo;
        uint64_t skip = 0, stop_handle = 0;
        int device = -1;
        bool gpu_command = false, cpu_command = false, child = false, help = false, version = false, once = false;
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
            } else if (option == "--save" || option == "--worktodo" || option == "--skip" ||
                       option == "--skip-curves" || option == "--device" ||
                       option == "--stage2-stop-handle" || option == "--stop-handle") {
                if (i + 1 == argc) throw std::runtime_error("missing value for " + option);
                args.push_back(argv[i++]);
                if (option == "--save") save = argv[i];
                else if (option == "--worktodo") worktodo = argv[i];
                else if (option == "--device") device = std::stoi(argv[i]);
                else if (option == "--skip" || option == "--skip-curves") skip = u64(argv[i], option.c_str());
                else stop_handle = u64(argv[i], option.c_str());
            } else if (option == "--gui-queue-protocol") {
                std::cout << "ECM_QUEUE_OWNER=1\n";
                return 0;
            } else if (option == "--curve-worker") child = true;
            else if (option == "--gpu-backend" || option == "--tune" || option == "--plan-only" ||
                     option == "--check-stage1-profile" || option == "--cost-device-info" ||
                     option == "--list-devices")
                gpu_command = true;
            else if (option == "--backend" || option == "--threads" || option == "--memory-mib" ||
                     option == "--gw-check" || option == "--selftest-gwpoly")
                cpu_command = true;
            else if (option == "--once") once = true;
            else if (option == "--version") version = true;
            else if (option == "--help" || option == "-h") help = true;
            args.push_back(argv[i]);
        }
        const int count = static_cast<int>(args.size());
        // GPU curve children carry a frozen queue route. They never reselect
        // the execution device after a configuration edit.
        if (child) {
            args.push_back(nullptr);
            return gpu(count, args.data(), nullptr);
        }
        if (version && target.empty()) {
            std::cout << ecm_program::name << ' ' << ecm_program::version << '\n';
            return 0;
        }
        const auto image = executable();
        const auto launch = read_launch_config(config, worker, image);
        const auto settings = ecm_config::read_stage2(ecm_config::cli_entries(launch.lines, worker, true));
        if (target.empty()) target = gpu_command ? "gpu" : settings.execution_device;
        if (target == "auto" && (gpu_command || cpu_command))
            target = gpu_command ? "gpu" : "cpu";
        if (help)
            std::cout << "Stage2 target: --target auto|cpu|gpu (TOML: [stage2] device; auto uses stage2_algorithm)\n";
        if (target != "auto" || help || version) {
            args.push_back(nullptr);
            return target == "gpu" ? gpu(count, args.data(), &launch)
                                   : run_cpu_stage2(count, args.data(), &launch);
        }

        const auto rules = parse_algorithm_policy(settings.algorithm);
        EcmQueueConfig queue_config;
        std::vector<const char*> inherited;
        if (save.empty()) inherited.push_back("finished");
        if (settings.save_dir.empty()) inherited.push_back("tmp_dir");
        if (device < 0 && settings.device < 0) inherited.push_back("device");
        ecm_queue_config_project(launch.lines, launch.path.u8string(), worker, queue_config, inherited);
        if (device < 0) device = settings.device < 0 ? queue_config.device : settings.device;
        Options options;
        options.worker = worker;
        options.worktodo = worktodo;
        const auto cwd = fs::current_path();
        const auto paths = task_paths(options, cwd, launch.path, settings, queue_config,
                                      ecm_config::toml_document_version(launch.lines) >= 3);
        auto invoke = [&](const Candidate& route, const std::string& task) {
            auto arguments = args;
            std::string row;
            if (!task.empty()) {
                row = std::to_string(fingerprint(task));
                arguments.push_back(const_cast<char*>("--stage2-dispatch-row"));
                arguments.push_back(row.data());
            }
            if (route.gpu_backend) {
                arguments.push_back(const_cast<char*>("--gpu-backend"));
                arguments.push_back(const_cast<char*>(route.gpu_backend));
            } else if (route.cpu_backend) {
                arguments.push_back(const_cast<char*>("--backend"));
                arguments.push_back(const_cast<char*>(route.cpu_backend));
            }
            const auto size = static_cast<int>(arguments.size());
            arguments.push_back(nullptr);
            return route.gpu_backend ? gpu(size, arguments.data(), &launch)
                                     : run_cpu_stage2(size, arguments.data(), &launch);
        };
        auto select = [&](const Record& record, uint64_t exponent) {
            const auto bits = modulus_bits(record);
            const auto route = choose_algorithm(rules, record, exponent, settings.gpu_backend, device, image, bits);
            std::cout << "stage2_algorithm: bits=" << bits << " selected=" << algorithm_name(route.algorithm)
                      << " device=" << (route.gpu_backend ? device : -1) << std::endl;
            return route;
        };
        if (!save.empty()) {
            const auto plan = records(absolute_from(cwd, save), skip, 1, 0, EdwardsPreparation::ValidateOnly);
            if (plan.empty()) throw std::runtime_error("skip exceeds saved curve count");
            const auto route = select(plan.front(), 0);
            return invoke(route, {});
        }

        if (!fs::exists(paths.worktodo))
            throw std::runtime_error("worktodo not found: " + paths.worktodo.string());
        if (stop_handle && !set_inherited_stop(stop_handle))
            throw std::runtime_error("invalid inherited Stage2 stop event");
        std::cout << ecm_program::name << ' ' << ecm_program::version << " (Stage2)" << std::endl;
        uint64_t finished_tasks = 0, invalid_tasks = 0;
        auto finish = [&](bool stopped) {
            std::cout << (stopped ? "stage2_stopped:" : "stage2_complete:");
            if (!finished_tasks) std::cout << " curves_this_run=0";
            else std::cout << " tasks_this_run=" << finished_tasks;
            std::cout << " invalid_tasks=" << invalid_tasks << std::endl;
            return invalid_tasks ? 1 : 0;
        };
        for (;;) {
            poll_inherited_stop();
            if (stop_requests) return finish(true);
            std::string task, error;
            const auto read = ecm_worktodo_read_next_line(paths.worktodo.string(), worker,
                {queue_config.worktodo_priority, queue_config.worktodo_skip}, WorktodoStage::Stage2, task, error);
            if (read == WorktodoReadResult::Error) throw std::runtime_error(error);
            if (read == WorktodoReadResult::Empty) return finish(false);
            Candidate route{Algorithm::Cpu};
            try {
                const auto fields = queue_fields(task, 0);
                const auto plan = records(absolute_from(paths.tmp, fields.task.save_name), fields.skip, 1, 0,
                                          EdwardsPreparation::ValidateOnly);
                if (!plan.empty()) route = select(plan.front(), fields.task.n);
            } catch (const TaskInputError&) {
                // Use the shared queue's durable error handling for malformed tasks.
            }
            const auto status = invoke(route, task);
            if (status == DispatchChanged) continue;
            if (status == DispatchInvalid) ++invalid_tasks;
            else if (status == DispatchAdvanced) ++finished_tasks;
            else return status ? status : (invalid_tasks ? 1 : 0);
            if (once) return finish(false);
        }
    } catch (const std::exception& error) {
        std::cerr << "ecm_stage2: " << error.what() << '\n';
        return 2;
    }
}
