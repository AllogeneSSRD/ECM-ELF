#include "ecm_queue_config.h"
#include "generated/ecm_ini_template.h"
#include <fstream>

bool ecm_queue_config_load(const std::string &path,EcmQueueConfig &cfg) {
    return ecm_queue_config_load(path,1,cfg);
}
bool ecm_queue_config_load(const std::string &path,int worker,EcmQueueConfig &cfg) {
    std::vector<ecm_config::IniLine> lines;
    if(!ecm_config::read_ini(path,lines))return false;
    if(worker>1 && std::none_of(lines.begin(),lines.end(),[&](const auto &l){
        return l.kind==ecm_config::IniLine::Kind::Section && ecm_config::worker_header(ecm_config::trim(l.raw))==worker;
    }))std::fprintf(stderr,"[ecm] NOTE: %s has no [Worker #%d] section; worker uses global values.\n",path.c_str(),worker);
    ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),
        ecm_config::cli_entries(lines,worker),ecm_config::stage1_bindings);
    return true;
}
bool ecm_queue_config_write_default(const std::string &path) {
    std::ofstream out(path,std::ios::trunc);
    if(!out)return false;
    out<<ecm_config::default_ini;out.close();return !out.fail();
}
