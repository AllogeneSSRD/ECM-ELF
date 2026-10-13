#include "ecm_queue_config.h"
#include "ecm_stage1_selection.h"
#include "generated/ecm_ini_template.h"
#include <fstream>

static bool removed_stage1_key(const std::string &key) {
    static constexpr const char *removed[] = {
        "method", "backend", "field", "exponent", "cpu_affinity",
        "curve_model", "execution_target", "cpu_implementation",
        "simd_reduction", "scalar_construction",
        "gpuckpt_seconds", "edwards_backend", "mont_backend", "edbackend",
        "edwards_threads", "mont_threads", "edthreads", "edwards_naf_w",
        "ednafw", "mont_save_pattern", "edwards", "mont",
        "edwards_mersenne", "emersenne", "mont_torsion"
    };
    const std::string normalized = ecm_config::lower(key);
    return std::any_of(std::begin(removed), std::end(removed),
                       [&](const char *candidate) { return normalized == candidate; });
}

bool ecm_queue_config_load(const std::string &path,EcmQueueConfig &cfg) {
    return ecm_queue_config_load(path,1,cfg);
}
bool ecm_queue_config_load(const std::string &path,int worker,EcmQueueConfig &cfg) {
    std::vector<ecm_config::IniLine> lines;
    if(!ecm_config::read_ini(path,lines))return false;
    if(worker>1 && std::none_of(lines.begin(),lines.end(),[&](const auto &l){
        return l.kind==ecm_config::IniLine::Kind::Section && ecm_config::worker_header(ecm_config::trim(l.raw))==worker;
    }))std::fprintf(stderr,"[ecm] NOTE: %s has no [Worker #%d] section; worker uses global values.\n",path.c_str(),worker);
    const auto entries = ecm_config::cli_entries(lines,worker);
    for (const auto &entry : entries) {
        if (removed_stage1_key(entry.first)) {
            std::fprintf(stderr, "[ecm] ERROR: removed Stage1 INI key '%s'; see ECM_INI_REFERENCE.md\n",
                         entry.first.c_str());
            return false;
        }
        bool valid = true;
        ecm_stage1::Selection selection;
        if (entry.first == "curve") valid = ecm_stage1::parse_curve_model(entry.second, selection.curve);
        else if (entry.first == "target") valid = ecm_stage1::parse_execution_target(entry.second, selection.target);
        else if (entry.first == "impl") valid = ecm_stage1::parse_cpu_implementation(entry.second, selection.cpu_implementation);
        else if (entry.first == "reduction") valid = ecm_stage1::parse_simd_reduction(entry.second, selection.simd_reduction);
        else if (entry.first == "lcm") valid = ecm_stage1::parse_scalar_construction(entry.second, selection.scalar_construction);
        else if (entry.first == "gpu_param") valid = entry.second == "0" || entry.second == "2" || entry.second == "3";
        if (!valid) {
            std::fprintf(stderr, "[ecm] ERROR: invalid Stage1 INI value '%s=%s'; see ECM_INI_REFERENCE.md\n",
                         entry.first.c_str(), entry.second.c_str());
            return false;
        }
    }
    ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),
                      entries,ecm_config::stage1_bindings);
    return true;
}
bool ecm_queue_config_write_default(const std::string &path) {
    std::ofstream out(path,std::ios::trunc);
    if(!out)return false;
    out<<ecm_config::default_ini;out.close();return !out.fail();
}
