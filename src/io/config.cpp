#include "io/config.h"
#include "ecm/application/configuration/generated/ecm_toml_template.h"
#include "ecm/core/stage1/selection.h"
#include <fstream>

bool read_config(const std::string& path, ParsedConfig& parsed) {
    try {
        parsed = ParsedConfig{};
        parsed.path = path;
        std::vector<std::string> ignored_gui;
        if (!ecm_config::read_toml(path, parsed.lines, ecm_config::TomlReadMode::Computation,
                                   &ignored_gui))
            return false;
        for (const auto& key : ignored_gui)
            std::fprintf(stderr, "[ecm] WARNING: ignoring unknown GUI-only TOML key: %s\n",
                         key.c_str());
        for (const auto& line : parsed.lines)
            if (line.kind == ecm_config::TomlLine::Kind::KeyValue) {
                const auto* meta = ecm_config::toml_option(line.section, line.key);
                try {
                    ecm_config::Stage1Values scratch;
                    ecm_config::apply(scratch, {{line.key, line.value}},
                                      ecm_config::stage1_bindings);
                    ecm_config::Stage2Values stage2_scratch;
                    ecm_config::apply(stage2_scratch, {{line.key, line.value}},
                                      ecm_config::stage2_bindings);
                    ecm_stage1::Selection s;
                    std::vector<unsigned> affinity;
                    bool valid = true;
                    const auto &key = line.key, &v = line.value;
                    if (key == "curve")
                        valid = ecm_stage1::parse_curve_model(v, s.curve);
                    else if (key == "target")
                        valid = ecm_stage1::parse_execution_target(v, s.target);
                    else if (key == "impl")
                        valid = ecm_stage1::parse_cpu_implementation(v, s.cpu_implementation);
                    else if (key == "reduction")
                        valid = ecm_stage1::parse_simd_reduction(v, s.simd_reduction);
                    else if (key == "lcm")
                        valid = ecm_stage1::parse_scalar_construction(v, s.scalar_construction);
                    else if (key == "gpu_param")
                        valid = v == "0" || v == "2" || v == "3";
                    else if (key == "device")
                        valid = scratch.device >= 0;
                    else if (key == "naf_w")
                        valid = scratch.naf_w == 0 || (scratch.naf_w >= 2 && scratch.naf_w <= 12);
                    else if (key == "tpi")
                        valid = scratch.tpi >= 1 && scratch.tpi <= 32 &&
                                !(scratch.tpi & (scratch.tpi - 1));
                    else if (key == "wg_size")
                        valid = scratch.wg_size >= 0;
                    else if (key == "sync_mode")
                        valid = v == "incremental" || v == "full";
                    else if (key == "affinity")
                        affinity = ecm_config::affinity_indices(v);
                    if (!valid)
                        throw std::runtime_error("value is outside the supported domain");
                    const bool global = line.section == "shared" || line.section == "stage1" ||
                                        line.section == "stage2";
                    const bool worker =
                        line.worker_scope > 0 &&
                        line.section != "workers." + std::to_string(line.worker_scope);
                    if (global || worker)
                        for (const auto& binding : ecm_config::stage1_bindings)
                            if (line.key == binding.key) {
                                auto& scope = parsed.stage1[global ? 0 : line.worker_scope];
                                binding.copy(scope.values, scratch);
                                if (key == "curve")
                                    scope.selection.curve = s.curve;
                                else if (key == "target")
                                    scope.selection.target = s.target;
                                else if (key == "impl")
                                    scope.selection.cpu_implementation = s.cpu_implementation;
                                else if (key == "reduction")
                                    scope.selection.simd_reduction = s.simd_reduction;
                                else if (key == "lcm")
                                    scope.selection.scalar_construction = s.scalar_construction;
                                else if (key == "affinity")
                                    scope.affinity_cpus = std::move(affinity);
                                if (std::find(scope.bindings.begin(), scope.bindings.end(),
                                              &binding) == scope.bindings.end())
                                    scope.bindings.push_back(&binding);
                                break;
                            }
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        "INPUT[config] " + path + ":" + std::to_string(line.line_number) +
                        " worker=" + std::to_string(line.worker_scope) + " key=" + line.key +
                        " value='" + line.value + "'; expected " +
                        (meta ? meta->domain : "documented scalar") + "; " + e.what() +
                        ". Correct this value and retry.");
                }
            }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return false;
    }
}

void project_stage1_config(const ParsedConfig& parsed, int worker, EcmQueueConfig& cfg) {
    cfg = EcmQueueConfig{};
    // Defaults are supplied at projection time, never inferred from absent keys.
    ecm_stage1::parse_curve_model(cfg.curve_model, cfg.selection.curve);
    ecm_stage1::parse_execution_target(cfg.execution_target, cfg.selection.target);
    ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation, cfg.selection.cpu_implementation);
    ecm_stage1::parse_simd_reduction(cfg.simd_reduction, cfg.selection.simd_reduction);
    ecm_stage1::parse_scalar_construction(cfg.scalar_construction,
                                          cfg.selection.scalar_construction);
    cfg.affinity_cpus = ecm_config::affinity_indices(cfg.affinity);
    for (const int scope : {0, worker}) {
        const auto found = parsed.stage1.find(scope);
        if (found == parsed.stage1.end())
            continue;
        for (const auto* binding : found->second.bindings) {
            binding->copy(cfg, found->second.values);
            cfg.config_keys.insert(binding->key);
            const std::string key = binding->key;
            const auto& selected = found->second;
            if (key == "curve")
                cfg.selection.curve = selected.selection.curve;
            else if (key == "target")
                cfg.selection.target = selected.selection.target;
            else if (key == "impl")
                cfg.selection.cpu_implementation = selected.selection.cpu_implementation;
            else if (key == "reduction")
                cfg.selection.simd_reduction = selected.selection.simd_reduction;
            else if (key == "lcm")
                cfg.selection.scalar_construction = selected.selection.scalar_construction;
            else if (key == "affinity")
                cfg.affinity_cpus = selected.affinity_cpus;
        }
        if (worker <= 0)
            break;
    }
    cfg.selection.gpu_param = cfg.gpu_param;
}

// Stage2 compatibility projection; Stage1 uses parsed sources + io/normalize.
bool ecm_queue_config_load(const std::string& path, EcmQueueConfig& cfg) {
    return ecm_queue_config_load(path, 1, cfg);
}

bool ecm_queue_config_load(const std::string& path, int worker, EcmQueueConfig& cfg) {
    ParsedConfig parsed;
    if (!read_config(path, parsed))
        return false;
    try {
        project_stage1_config(parsed, worker, cfg);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "INPUT[config] %s worker=%d: %s\n", path.c_str(), worker, e.what());
        return false;
    }
}

bool ecm_queue_config_write_default(const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out)
        return false;
    out << ecm_config::default_toml;
    out.close();
    return !out.fail();
}
