#include "io/normalize.h"
#include "io/config.h"
#include "io/cli.h"
#include "io/worktodo.h"
#include <cmath>
#include <limits>
#include "ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "ecm/service/stage1/preparation.h"
#include <sstream>
#include "ecm/service/execution/legacy_gpu.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
namespace ecm_stage1_app {
void normalize_assignment(Ecm2Task &task) {
    if(!task.has_curves)task.curves_to_run=100; // Prime95 ECM/ECM2 grammar default.
    if(!task.has_b2){task.B2=0;task.b2_text="0";}
}
template<class Task> static void target_identity(const Task &task,NormalizedStage1Task &out) {
    auto &id=out.options.identity;id.target_kbnc=true;id.handoff_k_str=task.k;
    id.handoff_k=std::strtod(task.k.c_str(),nullptr);id.handoff_b=static_cast<uint32_t>(ecm_config::bounded_integer(task.b,"b",2,UINT32_MAX));
    id.handoff_n=static_cast<uint32_t>(task.n);id.handoff_c=ecm_config::signed_integer(task.c,"c",INT32_MIN,INT32_MAX);
    id.handoff_factors.clear();for(const auto &factor:task.factors){if(!id.handoff_factors.empty())id.handoff_factors+=',';id.handoff_factors+=factor;}
    out.expression="("+task.k+"*"+task.b+"^"+std::to_string(task.n)+(task.c.empty() || task.c[0]!='-'?"+":"")+task.c+")";
    if(!task.factors.empty()){out.expression+="/(";for(size_t i=0;i<task.factors.size();++i){if(i)out.expression+='*';out.expression+=task.factors[i];}out.expression+=')';}
    out.aid=task.aid;id.result_aid=task.aid;out.curves=task.curves_to_run;
}
template<class Task> static std::string saved_stage2_line(const Task &task,const std::string &save) {
    std::string quoted="\"";for(char c:save){quoted+=c;if(c=='"')quoted+='"';}quoted+='"';
    std::ostringstream out;out<<"ECMSTAGE2=";
    if(!task.aid.empty())out<<task.aid<<',';
    out<<task.k<<','<<task.b<<','<<task.n<<','<<task.c<<','<<quoted<<','<<task.b2_text<<",0,"<<task.curves_to_run;
    if(!task.factors.empty()){out<<",\"";for(size_t i=0;i<task.factors.size();++i){if(i)out<<',';out<<task.factors[i];}out<<'"';}
    return out.str();
}
bool normalize_worktodo(const std::string &line,const ecm_stage1::Stage1Options &defaults,const std::string &exe,NormalizedStage1Task &out,std::string &error) {
    try {
        out.options=defaults;out.ticket=line;
        if(line.compare(0,4,"ECM=")==0 || line.compare(0,5,"ECM2=")==0) {
            Ecm2Task task;if(!ecm_parse_ecm2_line(line,task,error))return false;
            normalize_assignment(task);
            if(!ecm_compute_ecm2_n(task,out.modulus,error))return false;
            target_identity(task,out);out.b1=task.B1;out.b2=task.B2;
            if(task.has_sigma){out.options.identity.sigma_fixed=true;out.options.identity.fixed_sigma64=task.sigma;}
            (void)ecm_config::unsigned_integer(task.b2_text,"B2 (integer >=0 for saved Stage2 handoff)");
            out.save_file=resolve_rel_local(out.options.persistence.tmp_dir,ecm_stage1::mont_format_save_name(out.options.persistence.save_name_pattern,out.modulus,out.b1,&out.options));
            out.stage2_line=saved_stage2_line(task,out.save_file);
            out.published_line=out.stage2_line;out.published_line.replace(0,10,"ECMSTAGE1=");
        }else {
            EcmStage2Task task;if(!ecm_parse_stage1_line(line,task,error))return false;
            if(!ecm_extract_b1_from_save_name(task.save_name,&out.b1,error) || !ecm_compute_stage2_n(task,out.modulus,error))return false;
            target_identity(task,out);out.save_assignment=true;
            if(!stage1_save_construction(task.save_name,out.options,error))return false;
            out.save_file=resolve_rel_local(exe,ecm_stage1::stage1_prefixed_save(task.save_name,ecm_stage1::stage1_save_prefix(out.options)));
            const auto original=resolve_rel_local(exe,task.save_name);
            if(out.save_file!=original)out.options.persistence.legacy_save_source=original;
            if(!ecm_rewrite_stage2_save(line,out.save_file,out.published_line,error))return false;
            out.stage2_line=out.published_line;out.stage2_line.replace(0,10,"ECMSTAGE2=");
        }
        if(stage1_selection_error(out.options.execution.selection,error) || stage1_task_error(out.options,out.curves,error))return false;
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
void normalize_cli(CliInput &input) {
    const EcmQueueConfig defaults;
    if(!input.device_set)input.gpu_device_index=defaults.device;
    if(!input.gpu_param_set)input.gpu_param_cli=defaults.gpu_param;
    if(!input.explicit_options.count("--stage1-threads"))input.stage1_threads=defaults.stage1_threads;
    if(!input.explicit_options.count("--naf-w"))input.naf_w=defaults.naf_w;
    if(input.pos.size()>2)throw std::runtime_error("INPUT[cli]: expected B1 [B2], received extra positional arguments");
    for(size_t i=0;i<input.pos.size();++i) {
        const auto &raw=input.pos[i];size_t used=0;double n=0;
        try{n=std::stod(raw,&used);}catch(...){}
        if(used!=raw.size() || !std::isfinite(n) || (i==0?n<=0:n<0))
            throw std::runtime_error("INPUT[cli] key="+std::string(i==0?"B1":"B2")+" value='"+raw+"'; expected finite "+(i==0?"positive":"nonnegative")+" bound");
    }
}
void normalize_cli(CliInput &input,const EcmQueueConfig &cfg) {
    const auto absent=[&](const char *key){return !input.explicit_options.count(key);};
    if(absent("-v"))input.verbose=(input.pos.empty() || cfg.config_keys.count("verbose"))?cfg.verbose:false;
    if(absent("--curve"))ecm_stage1::parse_curve_model(cfg.curve_model,input.selection.curve);
    if(absent("--target"))ecm_stage1::parse_execution_target(cfg.execution_target,input.selection.target);
    if(absent("--impl"))ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation,input.selection.cpu_implementation);
    if(absent("--reduction"))ecm_stage1::parse_simd_reduction(cfg.simd_reduction,input.selection.simd_reduction);
    if(absent("--lcm"))ecm_stage1::parse_scalar_construction(cfg.scalar_construction,input.selection.scalar_construction);
    if(!input.device_set)input.gpu_device_index=cfg.device;
    if(!input.gpu_param_set)input.gpu_param_cli=cfg.gpu_param;
    if(absent("--stage1-threads"))input.stage1_threads=cfg.stage1_threads;
    if(absent("--naf-w"))input.naf_w=cfg.naf_w;
    if(absent("--affinity"))input.affinity_spec=cfg.affinity;
    if(!input.ckpt_set){input.ckpt_seconds=cfg.ckpt_seconds;input.ckpt_set=true;}
    if(!input.sigma_fixed && cfg.sigma){input.fixed_sigma64=cfg.sigma;input.sigma_fixed=true;}
    if(absent("--tmp-dir"))input.tmp_dir=cfg.config_keys.count("tmp_dir")?resolve_rel_local(get_exe_dir_local(),cfg.tmp_dir):".";
    if(absent("--exp-cache"))input.exp_cache=cfg.exp_cache;
    if(absent("--mul"))input.gpu_mul_path=cfg.kernel_mul;
    if(absent("--sqr"))input.gpu_sqr_path=cfg.kernel_sqr;
    if(absent("--add"))input.gpu_add_path=cfg.kernel_add;
    if(absent("--sub"))input.gpu_sub_path=cfg.kernel_sub;
    if(absent("--special-mult"))input.gpu_special_mult_path=cfg.kernel_special_mult;
    if(absent("--tpi"))input.runtime.tpi=cfg.tpi;
    if(absent("--wg"))input.runtime.wg_size=cfg.wg_size;
}
bool normalize_config(const ParsedConfig &parsed,const CliInput &input,EcmQueueConfig &cfg,std::string &error) {
    try {
        cfg=EcmQueueConfig{};
        const auto entries=ecm_config::cli_entries(parsed.lines,input.worker_index);
        ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),entries,ecm_config::stage1_bindings);
        for(const auto &entry:entries)cfg.config_keys.insert(entry.first);
    }catch(const std::exception &e){error="INPUT[config] "+parsed.path+" worker="+std::to_string(input.worker_index)+" "+e.what();return false;}
    if(!normalize_config(cfg,input,error)){error=parsed.path+": "+error;return false;}return true;
}
ecm_stage1::Stage1Options normalize_options(const EcmQueueConfig &cfg,const std::string &exe,const std::string &base,int worker) {
    ecm_stage1::Stage1Options opt;
    auto &s=opt.execution.selection;
    if(!ecm_stage1::parse_curve_model(cfg.curve_model,s.curve) ||
       !ecm_stage1::parse_execution_target(cfg.execution_target,s.target) ||
       !ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation,s.cpu_implementation) ||
       !ecm_stage1::parse_simd_reduction(cfg.simd_reduction,s.simd_reduction) ||
       !ecm_stage1::parse_scalar_construction(cfg.scalar_construction,s.scalar_construction))
        throw std::runtime_error("INPUT[normalize]: invalid execution selection");
    s.gpu_param=cfg.gpu_param;
    opt.execution.affinity_cpus=ecm_config::affinity_indices(cfg.affinity);
    opt.execution.stage1_threads=cfg.stage1_threads;opt.execution.naf_w=cfg.naf_w;
    opt.execution.verbose=cfg.verbose;opt.execution.device_index=cfg.device;
    opt.execution.gpu_mul_path=cfg.kernel_mul;opt.execution.gpu_sqr_path=cfg.kernel_sqr;
    opt.execution.gpu_add_path=cfg.kernel_add;opt.execution.gpu_sub_path=cfg.kernel_sub;
    opt.execution.gpu_special_mult_path=cfg.kernel_special_mult;
    opt.persistence.save_name_pattern=cfg.save_name_pattern;
    const auto ms=cfg.ckpt_seconds*1000.0;
    opt.persistence.ckpt_ms=ms<=0?0:ms>=double(ULONG_MAX)?ULONG_MAX:static_cast<unsigned long>(std::llround(ms));
    opt.persistence.ckpt_backups=static_cast<unsigned>(std::clamp(cfg.ckpt_backups,0,int(ECM_STAGE1_MAX_BACKUPS)));
    opt.persistence.tmp_dir=resolve_rel_local(exe,cfg.tmp_dir);
    opt.identity.sigma_fixed=cfg.sigma!=0;opt.identity.fixed_sigma64=cfg.sigma;
    stage1_result_settings(opt,cfg,base,worker);return opt;
}
bool validate_cli_mode(const CliInput &input,std::string &error) {
    const bool queue=input.pos.empty() && !input.show_gpu_info && !input.show_kernels;
    const std::vector<std::string> forbidden={"--curve","--gpu-param","--lcm","-gpucurves","-save","-savea","--go","--go-backend","--go-gp-timeout","--bits"};
    for(const auto &key:forbidden)if(queue && input.explicit_options.count(key)) {
        const auto &at=input.explicit_options.at(key);
        error="INPUT[conflict] argv["+std::to_string(at.first)+"] key="+key+" value='"+at.second+"'; task fields belong to worktodo in queue mode. Remove this CLI option and edit the task/config instead.";
        return false;
    }
    if(queue && input.sigma_param_set){error="INPUT[conflict] key=sigma; queue mode accepts an unprefixed sigma only; curve/param belong to the task";return false;}
    return true;
}
bool normalize_config(EcmQueueConfig &cfg,const CliInput &input,std::string &error) {
    if(!validate_cli_mode(input,error))return false;
    static const std::pair<const char*,const char*> keys[]={
        {"-v","verbose"},{"--target","target"},{"--impl","impl"},{"--reduction","reduction"},
        {"--stage1-threads","stage1_threads"},{"--naf-w","naf_w"},{"--affinity","affinity"},
        {"-d","device"},{"--ckpt","ckpt_seconds"},{"-gpuckpt","ckpt_seconds"},
        {"--tmp-dir","tmp_dir"},{"--exp-cache","exp_cache"},{"--mul","kernel_mul"},{"--sqr","kernel_sqr"},
        {"--add","kernel_add"},{"--sub","kernel_sub"},{"--special-mult","kernel_special_mult"},
        {"--tpi","tpi"},{"--wg","wg_size"},{"-sigma","sigma"},{"--sigma","sigma"}};
    try {
        ecm_config::Entries entries;
        // CLI order matters for aliases; the last explicit spelling wins.
        std::vector<std::pair<int,std::pair<std::string,std::string>>> ordered;
        for(const auto &pair:keys)if(input.explicit_options.count(pair.first)) {
            const auto &at=input.explicit_options.at(pair.first);
            auto value=std::string(pair.first)=="-v"?"true":at.second;
            if(std::string(pair.second)=="sigma")value=std::to_string(input.fixed_sigma64);
            ordered.push_back({at.first,{pair.second,value}});
        }
        std::sort(ordered.begin(),ordered.end(),[](const auto &a,const auto &b){return a.first<b.first;});
        for(const auto &entry:ordered)ecm_config::replace_entry(entries,entry.second.first,entry.second.second);
        ecm_config::apply(static_cast<ecm_config::Stage1Values&>(cfg),entries,ecm_config::stage1_bindings);
        if(cfg.device<0)throw std::runtime_error("key=device value='"+std::to_string(cfg.device)+"'; expected integer >=0");
        if(cfg.naf_w!=0 && (cfg.naf_w<2 || cfg.naf_w>12))throw std::runtime_error("key=naf_w; expected 0 (auto) or [2,12], CPU requires [3,12]");
        if(cfg.tpi<1 || cfg.tpi>32 || (cfg.tpi&(cfg.tpi-1)))throw std::runtime_error("key=tpi; expected {1,2,4,8,16,32}");
        if(cfg.wg_size<0)throw std::runtime_error("key=wg_size; expected integer >=0");
        if(cfg.sync_mode!="incremental" && cfg.sync_mode!="full")throw std::runtime_error("key=sync_mode value='"+cfg.sync_mode+"'; expected incremental|full");
        ecm_stage1::ExecutionTarget target;ecm_stage1::CpuImplementation implementation;ecm_stage1::SimdReduction reduction;
        if(!ecm_stage1::parse_execution_target(cfg.execution_target,target) ||
           !ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation,implementation) ||
           !ecm_stage1::parse_simd_reduction(cfg.simd_reduction,reduction))
            throw std::runtime_error("invalid target/impl/reduction; expected cpu|gpu, auto|simd|scalar, auto|montgomery|mersenne-fold");
        if(target==ecm_stage1::ExecutionTarget::Gpu && (implementation!=ecm_stage1::CpuImplementation::Auto || reduction!=ecm_stage1::SimdReduction::Auto))
            throw std::runtime_error("keys=target,impl,reduction values='"+cfg.execution_target+","+cfg.cpu_implementation+","+cfg.simd_reduction+"'; GPU requires impl=auto and reduction=auto; choose target=cpu for CPU preferences");
        if(target==ecm_stage1::ExecutionTarget::Cpu && reduction!=ecm_stage1::SimdReduction::Auto && implementation!=ecm_stage1::CpuImplementation::IfmaSimd)
            throw std::runtime_error("keys=impl,reduction values='"+cfg.cpu_implementation+","+cfg.simd_reduction+"'; explicit reduction requires impl=simd");
        if(target==ecm_stage1::ExecutionTarget::Cpu && cfg.naf_w==2)
            throw std::runtime_error("key=naf_w value='2'; CPU requires 0 (auto) or [3,12]");
        (void)ecm_config::affinity_indices(cfg.affinity);
        return true;
    }catch(const std::exception &e){error="INPUT[normalize] worker="+std::to_string(input.worker_index)+" "+e.what();return false;}
}
using Stage1Options=ecm_stage1::Stage1Options;
static bool is_absolute_path_local(const std::string &p) {
#ifdef _WIN32
    if (p.size() >= 2 && std::isalpha((unsigned char)p[0]) && p[1] == ':') return true;
    if (p.size() >= 1 && (p[0] == '\\' || p[0] == '/')) return true;
    return false;
#else
    return !p.empty() && p[0] == '/';
#endif
}

std::string normalize_path_local(std::filesystem::path path) {
    if(path.empty())return {};
    path=path.lexically_normal();
    // A directory's trailing separator is unnecessary, except at a root.
    if(path.filename().empty() && path!=path.root_path())path=path.parent_path();
    return path.make_preferred().string();
}

std::string resolve_rel_local(const std::string &base, const std::string &p) {
    if(p.empty())return {};
    return normalize_path_local(is_absolute_path_local(p)?std::filesystem::path(p):
                                std::filesystem::path(base)/p);
}

std::string get_exe_dir_local() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::string p(buf, n);
        const std::size_t slash = p.find_last_of("\\/");
        if (slash != std::string::npos) return p.substr(0, slash);
    }
    return "";
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p(buf);
        const std::size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) return p.substr(0, slash);
    }
    return "";
#endif
}

bool stage1_save_construction(const std::string &name,Stage1Options &opt,std::string &error) {
    const auto leaf=ecm_stage1::lower_ascii(std::filesystem::path(name).filename().string());
    const auto prefix=leaf.substr(0,leaf.find('_'));
    if(prefix=="e" || prefix=="ed") {
        opt.execution.selection.curve=ecm_stage1::CurveModel::Edwards;
        opt.execution.selection.scalar_construction=ecm_stage1::ScalarConstruction::PrimePowerLcm;
    } else if(prefix=="m0" || prefix=="m2" || prefix=="m3") {
        opt.execution.selection.curve=ecm_stage1::CurveModel::Montgomery;
        opt.execution.selection.gpu_param=static_cast<unsigned>(prefix[1]-'0');
        if(opt.execution.selection.target==ecm_stage1::ExecutionTarget::Cpu && opt.execution.selection.gpu_param!=0) {
            error="save prefix selects Montgomery PARAM"+prefix.substr(1)+", which requires target=gpu";
            return false;
        }
    }
    return true;
}

bool stage1_selection_error(const ecm_stage1::Selection &selection,
                                   std::string &error) {
    if (!ecm_stage1::valid_route(selection, ECM_HAS_CUDA_STAGE1 != 0)) {
        error = "GPU Edwards requires the CUDA backend";
        return true;
    }
    if (selection.curve == ecm_stage1::CurveModel::Edwards &&
        selection.scalar_construction != ecm_stage1::ScalarConstruction::PrimePowerLcm) {
        error = "Edwards uses its fixed 48*lcm(1..B1) scalar; --lcm 12 is unsupported";
        return true;
    }
    if (selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        (selection.cpu_implementation != ecm_stage1::CpuImplementation::Auto ||
         selection.simd_reduction != ecm_stage1::SimdReduction::Auto)) {
        error = "--impl and --reduction require --target cpu";
        return true;
    }
    if (selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        selection.simd_reduction != ecm_stage1::SimdReduction::Auto &&
        selection.cpu_implementation != ecm_stage1::CpuImplementation::IfmaSimd) {
        error = "explicit --reduction requires --impl simd";
        return true;
    }
    if (selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        selection.curve == ecm_stage1::CurveModel::Edwards) return false;
    if (!ecm_stage1::valid_gpu_param(selection.gpu_param)) {
        error = "invalid gpu_param (expected 0, 2 or 3)";
        return true;
    }
    if (const char *reason = ecm_backend_stage1_param_error(selection.gpu_param)) {
        error = std::string(ecm_backend_name()) + ": " + reason;
        return true;
    }
    return false;
}

bool stage1_task_error(const Stage1Options &opt, uint32_t curves,
                              std::string &error) {
    if(opt.execution.selection.target==ecm_stage1::ExecutionTarget::Cpu &&
       opt.execution.naf_w!=0 && (opt.execution.naf_w<3 || opt.execution.naf_w>12)) {
        error="key=naf_w value='"+std::to_string(opt.execution.naf_w)+"'; CPU requires 0 (auto) or [3,12]";return true;
    }
    if (opt.identity.sigma_fixed && curves != 0 &&
        static_cast<uint64_t>(curves - 1) > UINT64_MAX - opt.identity.fixed_sigma64) {
        error = "sigma + curve index exceeds 2^64-1";
        return true;
    }
    if (opt.execution.selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards ||
        opt.execution.selection.gpu_param == 0 || !opt.identity.sigma_fixed) return false;
    if (opt.identity.fixed_sigma64 > 0xFFFFFFFFull) {
        error = "GPU batch sigma exceeds 2^32-1";
        return true;
    }
    if (curves != 0 && curves > 0x100000000ull - opt.identity.fixed_sigma64) {
        error = "GPU batch sigma + curves must be <= 2^32";
        return true;
    }
    return false;
}


void stage1_result_settings(Stage1Options &opt,const EcmQueueConfig &cfg,
                                   const std::string &base,int worker) {
    opt.identity.result_user=cfg.result_user;opt.identity.result_computer=cfg.result_computer;
    opt.persistence.results_file=(std::filesystem::path(base)/(cfg.has_results?cfg.results:ecm_config::worker_file(cfg.results,worker))).string();
    opt.persistence.submission_file=(std::filesystem::path(base)/(cfg.has_submission?cfg.submission:ecm_config::worker_file(cfg.submission,worker))).string();
}
} // namespace ecm_stage1_app
