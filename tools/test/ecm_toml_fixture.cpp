#include "../../src/ecm/application/gui/toml_file.h"
#include "../../src/io/config.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using ecm_config::TomlReadMode;

static void expect(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
static void write(const fs::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << body;
    expect(bool(out), "cannot write fixture");
}
static void rejected(const fs::path& path, const std::string& body,
                     TomlReadMode mode = TomlReadMode::Strict) {
    write(path, body);
    std::vector<ecm_config::TomlLine> lines;
    try {
        if (!ecm_config::read_toml(path.string(), lines, mode, nullptr, nullptr, 2)) return;
        const auto entries = ecm_config::cli_entries(lines, 2);
        if (mode != TomlReadMode::Stage2) (void)ecm_config::read_stage1(entries);
        if (mode != TomlReadMode::Stage1) (void)ecm_config::read_stage2(entries);
    } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid TOML was accepted: " + body);
}

static void versions_and_addresses(const fs::path& path) {
    for (const auto* invalid : {
            "", "[shared]\nworktodo_file='queue.txt'\n",
            "config_version=1\n", "config_version=2\n", "config_version=3\n",
            "config_version=5\n", "config_version='4'\n", "config_version=4.0\n",
            "config_version=4\nconfig_version=4\n", "[stage1]\nconfig_version=4\n",
            "config_version=4\n[stage1]\nhandoff.mode='off'\n",
            "config_version=4\n[stage1.handoff]\nmode='off'\n",
            "config_version=4\n[worker.2]\nstage2.device='cpu'\n",
            "config_version=4\n[worker.2]\n'stage2.device'='cpu'\n",
            "config_version=4\n[worker.2.stage2]\ndevice='cpu'\n",
            "config_version=4\n[workers.2]\nautostart=true\n",
            "config_version=4\n[worker.0]\nautostart=true\n",
            "config_version=4\n[worker.1000001]\nautostart=true\n",
            "config_version=4\n[worker.2]\nworker_count=2\n",
            "config_version=4\n[worker.2]\nlog_file='override.log'\n",
            "config_version=4\n[stage1]\ntune_file='wrong.toml'\n",
            "config_version=4\n[stage1]\nstage2_gpu_device=1\n",
            "config_version=4\n[worker.2]\ndevice='cpu'\ndevice='gpu'\n"})
        rejected(path, invalid);
}

static void schema_addresses() {
    std::set<std::string> keys;
    for (const auto& option : ecm_config::toml_options) {
        const std::string key = option.toml_key;
        const std::string owner = option.toml_table;
        expect(key.find('.') == key.npos && keys.insert(key).second,
               "public TOML key is dotted or repeated");
        if (owner != "worker.1")
            expect(ecm_config::toml_address(owner, key).option == &option,
                   "option missing from global owner");
        if (owner != "gui" && !option.global_only) {
            const auto address = ecm_config::toml_address("worker.2", key);
            expect(address.option == &option && address.worker_scope == 2 && address.key == option.key,
                   "same key does not resolve under a worker");
        }
    }
}

static void projection(const fs::path& path) {
    write(path,
        "config_version=4\n[shared]\nworker_count=3\nworktodo_file='queue#1.txt' # preserved\n"
        "log_file='common.log'\nfinished_file='done.txt'\nlog_file_max_kb=1000\ntune_file='shared.toml'\n"
        "[stage1]\nparam='m0'\nstage1_gpu_device=0\nhandoff_mode='off'\n"
        "cpu_implementation='simd[:6144], scalar[:4096], gwnum[4096:], auto[]'\n"
        "cuda_slice_depth=2\ncuda_low_cpu=1\nedwards_doubling=0\ncpu_prac_search=50\n"
        "[stage2]\ndevice='gpu'\nstage2_gpu_device=1\ncurve_concurrency='auto'\nb2_ratio=20\n"
        "tune_stage1_seconds_per_curve=48.78\n"
        "[gui]\nstage2_exe='ecm_stage2.exe'\n"
        "[worker.2]\nautostart=true\nparam='e'\nstage1_gpu_device=2\nntt_length=1024\n"
        "cpu_implementation=''\ncuda_slice_depth=8\ncuda_low_cpu=0\nedwards_doubling=1\ncpu_prac_search=1\n"
        "cuda_prac_search=12\ntune_file='worker.toml'\ndevice='cPu'\nstage2_gpu_device=3\nb2_ratio=30\n"
        "tune_stage1_seconds_per_curve=0.0\n"
        "[worker.3]\ntune_file=''\n");
    std::vector<ecm_config::TomlLine> lines;
    int version = 0;
    expect(ecm_config::read_toml(path.string(), lines, TomlReadMode::Strict, nullptr, &version) &&
           version == 4, "wrong TOML version");
    const auto stage2_base = ecm_config::read_stage2(ecm_config::cli_entries(lines, 1));
    const auto stage2_worker = ecm_config::read_stage2(ecm_config::cli_entries(lines, 2));
    expect(stage2_base.device == 1 && stage2_base.execution_device == "gpu" &&
           stage2_worker.device == 3 && stage2_worker.execution_device == "cpu",
           "Stage2 switch/index or worker override lost");
    expect(stage2_base.stage1_seconds == 48.78 && stage2_worker.stage1_seconds == 0.0,
           "fractional Stage1 cost or zero fallback lost");
    expect(stage2_base.parallel == 0 && stage2_worker.b2_ratio == 30 &&
           stage2_worker.worktodo == "queue#1.txt", "Stage2 values lost");
    expect(ecm_config::read_worker(ecm_config::section_entries(lines, "workers.2")).autostart == 1,
           "worker launch flag lost");
    expect(ecm_config::configured_workflow_worker_count(lines, path.string()) == 3,
           "shared worker count lost");

    ParsedConfig parsed;
    expect(read_config(path.string(), parsed), "Stage1 did not read v4 config");
    EcmQueueConfig stage1;
    project_stage1_config(parsed, 1, stage1);
    expect(stage1.device == 0 && stage1.param == "m0" && stage1.handoff_mode == "off" &&
           stage1.shared_log_file == "common.log" && stage1.shared_finished_file == "done.txt" &&
           stage1.log_file_max_kb == 1000 && stage1.ntt_tune_file == "shared.toml",
           "Stage1 global values lost");
    expect(stage1.selection.cpu_implementation_rules.size() == 4 &&
           stage1.selection.cpu_implementation_rules.front().end_bit == 6144 &&
           stage1.cuda_slice_depth == 2 && stage1.cuda_low_cpu &&
           stage1.edwards_doubling == 0 && stage1.cpu_prac_search == 50,
           "Stage1 host/GPU fields lost");
    project_stage1_config(parsed, 2, stage1);
    expect(stage1.device == 2 && stage1.param == "e" && stage1.ntt_length == 1024 &&
           stage1.cuda_slice_depth == 8 && !stage1.cuda_low_cpu && stage1.edwards_doubling == 1 &&
           stage1.cpu_prac_search == 1 && stage1.cuda_prac_search == 12 &&
           stage1.selection.cpu_implementation_rules.empty(),
           "Stage1 worker override lost");
    expect(stage1.ntt_tune_file == stage2_worker.shared_tune_file &&
           stage1.ntt_tune_file == "worker.toml", "Stage1 and Stage2 resolved different tune files");
    project_stage1_config(parsed, 3, stage1);
    expect(stage1.ntt_tune_file.empty() &&
           ecm_config::read_stage2(ecm_config::cli_entries(lines, 3)).shared_tune_file.empty(),
           "empty worker tune path lost");
}

static void consumer_isolation(const fs::path& path) {
    write(path, "config_version=4\n[shared]\nworktodo_file='queue.txt'\n"
                "[stage1]\nstage1_gpu_device=0\n"
                "[stage2]\ngiant_backend=42\n"
                "[gui]\ngpu_clock_ranges=''\n"
                "[worker.2]\nstage1_gpu_device=1\ngiant_backend='coop16'\n"
                "[worker.3]\nstage2_gpu_device='invalid'\n");
    std::vector<ecm_config::TomlLine> lines;
    expect(ecm_config::read_toml(path.string(), lines, TomlReadMode::Gui),
           "GUI eagerly validated computation");
    expect(ecm_config::read_toml(path.string(), lines, TomlReadMode::Stage1, nullptr, nullptr, 2),
           "active Stage1 override did not isolate foreign/overridden values");
    const auto inherited = ecm_config::selected_entries(lines, 2, {"device"}, path.string());
    expect(ecm_config::read_stage1(inherited).device == 1,
           "selected Stage1 inheritance ignored worker override");
    write(path, "config_version=4\n[stage1]\nstage1_gpu_device='invalid'\n"
                "[stage2]\ngiant_backend='auto'\n"
                "[worker.2]\ngiant_backend='coop16'\n"
                "[worker.3]\nstage2_gpu_device='invalid'\n");
    expect(ecm_config::read_toml(path.string(), lines, TomlReadMode::Stage2, nullptr, nullptr, 2),
           "active Stage2 override did not isolate foreign/inactive values");
    expect(ecm_config::read_stage2(ecm_config::cli_entries(lines, 2)).giant_backend == "coop16",
           "selected worker backend lost");
}

static void domains_and_editor(const fs::path& path) {
    for (const auto& [key, values] : std::vector<std::pair<std::string, std::vector<const char*>>>{
            {"cuda_slice_depth", {"0", "4", "9", "'3'", "3.0"}},
            {"cuda_low_cpu", {"2", "'1'"}},
            {"edwards_doubling", {"-1", "2", "'1'", "1.0", "true"}},
            {"cpu_prac_search", {"0", "51", "7.0", "'7'", "true"}},
            {"progress_average_seconds", {"-0.01", "60.01", "nan", "inf", "'5'"}}}) {
        for (const auto* value : values)
            rejected(path, "config_version=4\n[stage1]\n" + key + "=" + value + "\n");
    }
    write(path, "config_version=4\n[stage1]\ncpu_implementation='simd[3:2]'\n");
    ParsedConfig parsed;
    expect(!read_config(path.string(), parsed), "Stage1 accepted invalid CPU slice");
    for (const auto& [key, values] : std::vector<std::pair<std::string, std::vector<const char*>>>{
            {"groot_balanced", {"2", "1.0", "'invalid'"}},
            {"pack_zero", {"2", "'1'"}},
            {"baby_backend", {"'coop16'", "'wrong'", "32"}},
            {"giant_backend", {"'coop8'", "'wrong'", "32"}},
            {"tune_stage1_seconds_per_curve", {"-0.01", "nan", "inf", "1e309", "'48.78'"}}}) {
        for (const auto* value : values)
            rejected(path, "config_version=4\n[stage2]\n" + key + "=" + value + "\n");
    }

    write(path, "# retain this comment\nconfig_version=4\n[stage2]\n"
                "groot_balanced='auto'\ngiant_tail_balance = 1 # enabled\n"
                "[worker.2]\ngroot_balanced=0\n");
    ecmgui::TomlFile editor;
    std::string error;
    expect(editor.load(path.string(), error), "GUI editor rejected v4");
    expect(editor.get("stage2", "stage2_giant_tail_balance") == "true",
           "GUI loaded 1 as false");
    for (const auto* value : {"true", "false", "1", "0", "auto"}) {
        editor.set("stage2", "stage2_groot_balanced", value);
        editor.set("workers.2.stage2", "stage2_device", "3");
        const auto canonical = ecm_config::automatic_boolean_text(value, "groot fixture");
        expect(editor.save(error), "GUI editor failed to save v4");
        std::vector<ecm_config::TomlLine> lines;
        expect(ecm_config::read_toml(path.string(), lines), "GUI wrote invalid v4");
        expect(lines.front().raw == "# retain this comment", "GUI changed an unrelated source line");
        const auto stage2_global = ecm_config::read_stage2(ecm_config::cli_entries(lines, 1));
        const auto stage2_worker = ecm_config::read_stage2(ecm_config::cli_entries(lines, 2));
        expect(stage2_global.groot_balanced == canonical && stage2_worker.groot_balanced == "false" &&
               stage2_worker.device == 3, "GUI roundtrip changed worker or global values");
    }
}

int main(int argc, char** argv) {
    try {
        expect(argc == 3, "usage: fixture DEFAULT_TOML TEMP_TOML");
        std::vector<ecm_config::TomlLine> lines;
        expect(ecm_config::read_toml(argv[1], lines), "default TOML unavailable");
        const auto stage1 = ecm_config::read_stage1(ecm_config::cli_entries(lines, 1));
        const auto stage2 = ecm_config::read_stage2(ecm_config::cli_entries(lines, 1));
        expect(stage1.progress_average_seconds == 0 && stage1.edwards_doubling == 1 &&
               stage1.cuda_slice_depth == 3 && !stage1.cuda_low_cpu,
               "unexpected default Stage1 tuning policy");
        expect(stage2.execution_device == "auto" && stage2.device == -1 &&
               stage2.groot_balanced == "auto", "unexpected default Stage2 selection");
        const fs::path temp = argv[2];
        versions_and_addresses(temp);
        schema_addresses();
        projection(temp);
        consumer_isolation(temp);
        domains_and_editor(temp);
        fs::remove(temp);
        fs::remove(temp.string() + ".bak");
        std::cout << "TOML v4 parser, worker projections, GUI editor and generated defaults: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
