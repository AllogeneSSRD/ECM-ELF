#pragma once
#include "ecm/service/stage1/task.h"
#include <filesystem>
#include <string>
#include <vector>
struct EcmQueueConfig;
struct ParsedConfig;
struct Ecm2Task;

namespace ecm_stage1_app {
struct CliInput;

// Frozen, owned computation input. Queue routing and I/O handles stay in main.
struct NormalizedStage1Task {
    mpz_t modulus;
    double b1 = 0, b2 = 0;
    uint32_t curves = 0;
    bool append_save = false;
    std::string save_file, expression;
    ecm_stage1::Stage1Options options;

    ecm_stage1::Stage1Request request() const {
        return {modulus, b1, b2, curves, save_file, append_save, expression, options};
    }

    NormalizedStage1Task() {
        mpz_init(modulus);
    }

    ~NormalizedStage1Task() {
        mpz_clear(modulus);
    }

    NormalizedStage1Task(const NormalizedStage1Task&) = delete;
    NormalizedStage1Task& operator=(const NormalizedStage1Task&) = delete;
};

// Queue editing metadata never enters the computation request.
struct WorktodoStage1Task {
    NormalizedStage1Task input;
    bool save_assignment = false;
    std::string ticket, published_line, stage2_line, origin;
};

bool normalize_direct(const CliInput&, const EcmQueueConfig&, const std::string& expression,
                      const std::string& exe, const std::string& config_directory,
                      NormalizedStage1Task&, std::string& error);
bool normalize_worktodo(const std::string& line, const ecm_stage1::Stage1Options& defaults,
                        const std::string& exe, WorktodoStage1Task&, std::string& error);
// Parsing retains explicit presence. Defaults and precedence are applied here.
void normalize_cli(CliInput&);
void normalize_cli(CliInput&, const EcmQueueConfig&);
bool validate_cli_mode(const CliInput&, std::string& error);
bool normalize_config(EcmQueueConfig&, const CliInput&, std::string& error);
bool normalize_config(const ParsedConfig&, const CliInput&, EcmQueueConfig&, std::string& error);
void normalize_assignment(Ecm2Task&);
ecm_stage1::Stage1Options normalize_options(const EcmQueueConfig&, const std::string& exe,
                                            const std::string& config_directory, int worker);
std::string normalize_path_local(std::filesystem::path);
std::string resolve_rel_local(const std::string&, const std::string&);
std::string get_exe_dir_local();
bool stage1_save_construction(const std::string&, ecm_stage1::Stage1Options&, std::string&);
bool stage1_selection_error(const ecm_stage1::Selection&, std::string&);
bool stage1_task_error(const ecm_stage1::Stage1Options&, uint32_t, std::string&);
void stage1_result_settings(ecm_stage1::Stage1Options&, const EcmQueueConfig&, const std::string&,
                            int);
} // namespace ecm_stage1_app
