#include "ecm_queue_config.h"
#include "ecm/core/stage1/selection.h"
#include "generated/ecm_toml_template.h"
#include <fstream>

bool ecm_queue_config_load(const std::string &path,EcmQueueConfig &cfg) {
    return ecm_queue_config_load(path,1,cfg);
}
bool ecm_queue_config_load(const std::string &path,int worker,EcmQueueConfig &cfg) {
  try {
    std::vector<ecm_config::TomlLine> lines;
    if(!ecm_config::read_toml(path,lines))return false;
    if(worker>1 && std::none_of(lines.begin(),lines.end(),[&](const auto &l){
        return l.kind==ecm_config::TomlLine::Kind::Section && l.worker_scope==worker;
    }))std::fprintf(stderr,"[ecm] NOTE: %s has no [workers.%d] table; worker uses global values.\n",path.c_str(),worker);
    const auto entries = ecm_config::cli_entries(lines,worker);
    for (const auto &entry : entries) {
        bool valid = true;
        ecm_stage1::Selection selection;
        if (entry.first == "curve") valid = ecm_stage1::parse_curve_model(entry.second, selection.curve);
        else if (entry.first == "target") valid = ecm_stage1::parse_execution_target(entry.second, selection.target);
        else if (entry.first == "impl") valid = ecm_stage1::parse_cpu_implementation(entry.second, selection.cpu_implementation);
        else if (entry.first == "reduction") valid = ecm_stage1::parse_simd_reduction(entry.second, selection.simd_reduction);
        else if (entry.first == "lcm") valid = ecm_stage1::parse_scalar_construction(entry.second, selection.scalar_construction);
        else if (entry.first == "gpu_param") valid = entry.second == "0" || entry.second == "2" || entry.second == "3";
        if (!valid) {
            std::fprintf(stderr, "[ecm] ERROR: invalid Stage1 TOML value '%s=%s'; see ECM_TOML_REFERENCE.md\n",
                         entry.first.c_str(), entry.second.c_str());
            return false;
        }
    }
    ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),
                      entries,ecm_config::stage1_bindings);
    return true;
  } catch(const std::exception &e) {
    std::fprintf(stderr,"[ecm] ERROR: invalid configuration: %s\n",e.what());
    return false;
  }
}
bool ecm_queue_config_write_default(const std::string &path) {
    std::ofstream out(path,std::ios::trunc);
    if(!out)return false;
    out<<ecm_config::default_toml;out.close();return !out.fail();
}
