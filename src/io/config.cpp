#include "io/config.h"
#include "ecm/core/stage1/selection.h"
#include "ecm/application/configuration/generated/ecm_toml_template.h"
#include <fstream>


bool read_config(const std::string &path,ParsedConfig &parsed) {
    try {
        parsed.path=path;std::vector<std::string> ignored_gui;
        if(!ecm_config::read_toml(path,parsed.lines,ecm_config::TomlReadMode::Computation,&ignored_gui))return false;
        for(const auto &key:ignored_gui)std::fprintf(stderr,"[ecm] WARNING: ignoring unknown GUI-only TOML key: %s\n",key.c_str());
        for(const auto &line:parsed.lines)if(line.kind==ecm_config::TomlLine::Kind::KeyValue) {
            const auto *meta=ecm_config::toml_option(line.section,line.key);
            try {
                ecm_config::Stage1Values scratch;
                ecm_config::apply(scratch,{{line.key,line.value}},ecm_config::stage1_bindings);
                ecm_config::Stage2Values stage2_scratch;
                ecm_config::apply(stage2_scratch,{{line.key,line.value}},ecm_config::stage2_bindings);
                ecm_stage1::Selection s;bool valid=true;
                const auto &key=line.key,&v=line.value;
                if(key=="curve")valid=ecm_stage1::parse_curve_model(v,s.curve);
                else if(key=="target")valid=ecm_stage1::parse_execution_target(v,s.target);
                else if(key=="impl")valid=ecm_stage1::parse_cpu_implementation(v,s.cpu_implementation);
                else if(key=="reduction")valid=ecm_stage1::parse_simd_reduction(v,s.simd_reduction);
                else if(key=="lcm")valid=ecm_stage1::parse_scalar_construction(v,s.scalar_construction);
                else if(key=="gpu_param")valid=v=="0"||v=="2"||v=="3";
                else if(key=="device")valid=scratch.device>=0;
                else if(key=="naf_w")valid=scratch.naf_w==0 || (scratch.naf_w>=2 && scratch.naf_w<=12);
                else if(key=="tpi")valid=scratch.tpi>=1 && scratch.tpi<=32 && !(scratch.tpi&(scratch.tpi-1));
                else if(key=="wg_size")valid=scratch.wg_size>=0;
                else if(key=="sync_mode")valid=v=="incremental"||v=="full";
                else if(key=="affinity")(void)ecm_config::affinity_indices(v);
                if(!valid)throw std::runtime_error("value is outside the supported domain");
            }catch(const std::exception &e){
                throw std::runtime_error("INPUT[config] "+path+":"+std::to_string(line.line_number)+" worker="+std::to_string(line.worker_scope)+
                    " key="+line.key+" value='"+line.value+"'; expected "+(meta?meta->domain:"documented scalar")+"; "+e.what()+". Correct this value and retry.");
            }
        }
        return true;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return false;}
}
// Stage2 compatibility projection; Stage1 uses parsed sources + io/normalize.
bool ecm_queue_config_load(const std::string &path,EcmQueueConfig &cfg){return ecm_queue_config_load(path,1,cfg);}
bool ecm_queue_config_load(const std::string &path,int worker,EcmQueueConfig &cfg) {
    ParsedConfig parsed;if(!read_config(path,parsed))return false;
    try {
        cfg=EcmQueueConfig{};
        const auto entries=ecm_config::cli_entries(parsed.lines,worker);
        ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),entries,ecm_config::stage1_bindings);
        for(const auto &entry:entries)cfg.config_keys.insert(entry.first);
        return true;
    }catch(const std::exception &e){std::fprintf(stderr,"INPUT[config] %s worker=%d: %s\n",path.c_str(),worker,e.what());return false;}
}
bool ecm_queue_config_write_default(const std::string &path) {
    std::ofstream out(path,std::ios::trunc);
    if(!out)return false;
    out<<ecm_config::default_toml;out.close();return !out.fail();
}
