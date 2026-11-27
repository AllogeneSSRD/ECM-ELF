#include "i18n.h"
#pragma once
#include "../../gui/worker_controller.h"
#include "../../gui/task_display.h"
#include "stage2_progress.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iterator>
namespace ecmgui_v2::ui {
struct WorkerProgress {float fraction=0;std::string label="--",speed="--",iterations="--",eta="--",eta_detail="--";};
struct Stage2Batch {bool visible=false;float fraction=0;std::string label,remaining;};
inline Stage2Batch stage2_batch(const ecmgui::Stage2Curve& s,bool zh) {
    Stage2Batch out;if(!s.valid||!s.batches)return out;out.visible=true;
    out.fraction=static_cast<float>(std::clamp(double(s.batch)/double(s.batches),0.,1.));
    const bool giant=s.phase.find("Giant")!=s.phase.npos||s.phase.find("giant")!=s.phase.npos;
    out.label=std::string(giant?(ecmgui_v2::ui::text("giant_batch",zh)):(ecmgui_v2::ui::text("batch",zh)))+" "+std::to_string(s.batch)+" / "+std::to_string(s.batches);
    char text[96];if(std::isfinite(s.batch_left)&&s.batch_left>=0)std::snprintf(text,sizeof(text),ecmgui_v2::ui::text("worker.phase_remaining_seconds",zh),s.batch_left);
    else std::snprintf(text,sizeof(text),"%s",ecmgui_v2::ui::text("worker.phase_remaining_unknown",zh));
    out.remaining=text;return out;
}
inline std::string eta_text(double seconds,bool detail=false) {
    if(!std::isfinite(seconds)||seconds<0)return "--";
    char text[160];const double rounded=std::round(seconds);
    std::snprintf(text,sizeof(text),"%.0f s",rounded);std::string result=text;
    if(detail&&rounded>=60){
        if(rounded<3600)std::snprintf(text,sizeof(text)," (%.0f m %.0f s)",std::floor(rounded/60),std::fmod(rounded,60));
        else if(rounded<86400)std::snprintf(text,sizeof(text)," (%.0f h %.0f m %.0f s)",std::floor(rounded/3600),std::floor(std::fmod(rounded,3600)/60),std::fmod(rounded,60));
        else std::snprintf(text,sizeof(text)," (%.0f d %.0f h %.0f m %.0f s)",std::floor(rounded/86400),std::floor(std::fmod(rounded,86400)/3600),std::floor(std::fmod(rounded,3600)/60),std::fmod(rounded,60));
        result+=text;
    }
    return result;
}
inline Stage2Batch stage1_ntt_batch(const ecmgui::Stage1NttStatus& s) {
    Stage2Batch out;
    if(!s.valid||!s.curves)return out;
    out.visible=true;
    out.fraction=static_cast<float>(s.batch_fraction());
    char label[128];
    std::snprintf(label,sizeof(label),"curve %llu–%llu  %.1f%%",s.first+1,s.first+s.curves,100.*out.fraction);
    out.label=label;
    if(std::isfinite(s.batch_eta_s)&&s.batch_eta_s>=0)
        out.label+="  ETA "+eta_text(s.batch_eta_s);
    return out;
}
inline Stage2Batch pm1_batch(const ecmgui::WorkerView& worker) {
    Stage2Batch out;
    const auto& progress=worker.progress;
    if (!progress.pm1) return out;
    const auto count=worker.stage1_execution.details.find("batch");
    out.visible=true;
    out.fraction=static_cast<float>(std::clamp(progress.pct/100.,0.,1.));
    out.label="Pminus1 1";
    if (count!=worker.stage1_execution.details.end() && count->second!="1")
        out.label+="–"+count->second;
    out.label+="  "+std::to_string(progress.pm1_done)+" / "+std::to_string(progress.pm1_total)+" iter";
    return out;
}
inline Stage2Batch worker_stage1_batch(const ecmgui::WorkerView& worker) {
    return worker.progress.pm1?pm1_batch(worker):stage1_ntt_batch(worker.stage1_ntt);
}
inline WorkerProgress worker_progress(const ecmgui::WorkerView& worker,bool zh) {
    WorkerProgress out;const auto p=worker.stage2?worker.stage2_status.progress(false):worker.stage1_ntt.valid?worker.stage1_ntt.progress():worker.progress;
    char text[128];
    if(!worker.stage2&&std::isfinite(p.iter_per_s)&&p.iter_per_s>0){std::snprintf(text,sizeof(text),"%.0f",p.iter_per_s);out.iterations=text;}
    if(p.valid&&std::isfinite(p.pct)) {
        const double pct=std::clamp(p.pct,0.,100.);out.fraction=static_cast<float>(pct/100.);
        if(p.curves_total&&(worker.stage2||!worker.stage1_ntt.valid))std::snprintf(text,sizeof(text),"%llu / %llu  (%.1f%%)",p.curves_done,p.curves_total,pct);
        else std::snprintf(text,sizeof(text),"%.1f%%",pct);
        out.label=text;
        if(std::isfinite(p.s_per_curve)&&p.s_per_curve>0){
            std::snprintf(text,sizeof(text),"%.2f",p.s_per_curve);out.speed=text;
        }
        if((worker.stage2&&worker.stage2_status.queue_events?!worker.stop_requested:(p.pm1 || p.s_per_curve>0))&&
            std::isfinite(p.eta_s)&&p.eta_s>=0){
            out.eta=eta_text(p.eta_s);out.eta_detail=eta_text(p.eta_s,true);
        }
    } else if(std::isfinite(worker.resume_pct)&&worker.resume_pct>=0) {
        const double pct=std::clamp(worker.resume_pct,0.,100.);out.fraction=static_cast<float>(pct/100.);
        std::snprintf(text,sizeof(text),ecmgui_v2::ui::text("worker.resumed_percent",zh),pct);out.label=text;
    } else if(worker.running()||worker.state()==ecmgui::WorkerRunState::Restarting)out.label=ecmgui_v2::ui::text("waiting_for_progress",zh);
    return out;
}

inline std::string worker_device_name(const ecmgui::WorkerView& worker) {
    const auto device=worker.execution_target=="cpu"?std::string("CPU"):
        worker.execution_target=="gpu"?"GPU "+std::to_string(worker.device):std::string("Auto");
    return device+": "+worker.name;
}

inline std::string worker_task(const ecmgui::WorkerView& worker,bool zh) {
    const auto task=ecmgui::task_display(worker.task);
    if(!task.valid)return text("worker.no_task",zh);
    std::string line=task.pm1?"P-1 "+task.number:task.number;
    if(task.from_save)line+=" (Stage1 save)";
    if(task.has_factors)line+=" / "+std::string(text("worker.known_factors",zh));
    if(task.pm1) {
        const auto b2=worker.stage1_execution.details.find("B2");
        if (const auto& progress=worker.progress; progress.pm1_old_b1 && progress.pm1_target_b1) {
            const char* phase=progress.pm1_phase=="finish_old"?"pm1.finish_old":"pm1.extend";
            return line+"  B1="+std::to_string(progress.pm1_old_b1)+" → "+
                std::to_string(progress.pm1_target_b1)+"  B2="+
                (b2==worker.stage1_execution.details.end()?task.b2:b2->second)+"  |  "+text(phase,zh);
        }
        return line+"  B1="+task.b1+"  B2="+(b2==worker.stage1_execution.details.end()?task.b2:b2->second);
    }
    const auto curves=worker.stage2&&worker.stage2_status.valid?worker.stage2_status.total:
        worker.stage1_ntt.valid?worker.stage1_ntt.total:task.curves;
    line+="  curves = "+(curves?std::to_string(curves):std::string("--"));
    return line;
}

inline std::string stage1_execution(const ecmgui::Stage1Execution& execution, bool zh, bool details = false,
                                    const ecmgui::ProgressInfo* progress = nullptr) {
    if (!execution.valid())
        return {};
    if (execution.details.count("method") && execution.details.at("method")=="pm1") {
        std::string line="Stage1 CUDA/NTT";
        if (progress && progress->pm1_first_known)
            line+=" | iter="+std::to_string(progress->pm1_first_total);
        line+=" | length="+execution.details.at("length");
        for (const auto* key:{"bpw","shape","graph"}) {
            const auto field=execution.details.find(key);
            if (field!=execution.details.end()) line+=" | "+field->first+"="+field->second;
        }
        return line;
    }
    std::string line = std::string(!details && execution.backend=="ntt" ? "Stage1" : text("worker.stage1_execution", zh)) + " ";
    line += text(execution.curve == "ed" ? "worker.curve_edwards" : "worker.curve_montgomery", zh);
    if (execution.curve == "mont")
        line += " PARAM" + execution.param.substr(1);
    const auto& backend = execution.backend;
    line += " | ";
    line += backend == "cgbn" ? "CUDA/CGBN" : backend == "ntt" ? "CUDA/NTT" :
        backend == "cuda-thread" ? "CUDA/thread" :
        backend == "cpu" ? "CPU" : backend == "opencl" ? "OpenCL" : backend;
    const auto field = [&](const char* key) -> std::string {
        const auto found = execution.details.find(key);
        return found == execution.details.end() ? std::string{} : found->second;
    };
    const auto append = [&](const std::string& value) { line += " | " + value; };
    const auto append_field = [&](const char* key, const char* label) {
        if (const auto value = field(key); !value.empty())
            append(std::string(label) + "=" + value);
    };
    if(!details && backend=="ntt"){
        if(const auto bits=field("ntt_bits");!bits.empty())line+=" "+bits+" bits";
        append_field("B1","B1");append_field("iter_total","iter");
        if(const auto length=field("ntt_length");!length.empty())append("NTT: "+length);
        if(const auto shape=field("ntt_shape");!shape.empty())line+=" "+shape;
        if(const auto naf=field("naf");!naf.empty())line+=" NAF="+naf;
        append_field("ntt_graph_bits", "Graph");
        return line;
    }
    if (backend == "cgbn" || backend == "opencl" || backend == "cuda-thread") {
        const auto weight = field("kernel_weight"), tpi = field("tpi");
        if (!weight.empty() && !tpi.empty())
            line += "＜" + weight + ", " + tpi + "＞";
        else {
            append_field("kernel_weight", text("worker.stage1_kernel_bits", zh));
            append_field("tpi", "TPI");
        }
        if (backend == "cuda-thread")
            append_field("tpb", "TPB");
    } else if (backend == "ntt") {
        if (const auto bits = field("ntt_bits"); !bits.empty())
            line += " " + bits + " bits";
        append_field("ntt_length", "L");
        append_field("ntt_bpw", "bpw");
        if (const auto shape = field("ntt_shape"); !shape.empty())
            append(std::string(text("worker.stage1_ntt_shape", zh)) + "=" + shape);
        else
            append_field("ntt_type", text("worker.stage1_ntt_type", zh));
        append_field("ntt_graph_bits", "Graph");
    } else if (backend == "cpu") {
        if (const auto impl = field("cpu_impl"); !impl.empty())
            append(impl == "ifma" ? "IFMA SIMD" : impl == "scalar" ? "GMP scalar" :
                   impl == "gwnum" ? "GWNUM" : impl);
    }
    const auto naf = field("naf"), algorithm = field("algorithm"), chain = field("chain");
    const auto prac = field("prac");
    if (!naf.empty() && naf != "0")
        append(std::string(text("worker.stage1_naf_window", zh)) + "=" + naf);
    else if (algorithm == "binary")
        append(text("worker.stage1_binary", zh));
    else if (chain == "x-only" || (backend == "ntt" && execution.curve == "ed" && naf == "0"))
        append(text("worker.stage1_x_only", zh));
    if (!prac.empty())
        append(std::string(text("worker.stage1_prac_level", zh)) + "=" + prac);
    if (!chain.empty() && chain != "x-only" && chain != "naf" && (chain != "prac" || prac.empty()))
        append("chain=" + chain);

    static constexpr std::string_view shown[] = {
        "cpu_impl", "kernel_weight", "tpi", "ntt_type", "ntt_bits", "ntt_length", "ntt_bpw", "ntt_shape",
        "ntt_graph_bits", "ntt_max_radix_bits",
        "naf", "algorithm", "chain", "prac", "ntt_memory_per_curve_bytes", "ntt_memory_config_mib",
        "ntt_memory_device_bytes", "ntt_naf_memory_mb", "max_curves_in_memory"
    };
    if (details)
        for (const auto& [key, value] : execution.details)
            if (std::find(std::begin(shown), std::end(shown), key) == std::end(shown))
                append(key + "=" + value);
    if (details && backend == "ntt") {
        const auto number = [&](const char* key, unsigned long long& value) {
            const auto s = field(key);
            const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
            return parsed.ec == std::errc{} && parsed.ptr == s.data() + s.size();
        };
        unsigned long long per_curve = 0, configured = 0, device = 0, capacity = 0, naf_mb = 0;
        if (number("ntt_memory_per_curve_bytes", per_curve) && number("ntt_memory_config_mib", configured) &&
            number("ntt_memory_device_bytes", device) && number("max_curves_in_memory", capacity)) {
            constexpr double mib = 1024. * 1024.;
            const auto config = configured ? std::to_string(configured) + " MiB" : text("worker.stage1_ntt_memory_auto", zh);
            char summary[256];
            std::snprintf(summary, sizeof(summary), text("worker.stage1_ntt_memory", zh),
                          per_curve / mib, config.c_str(), device / mib, capacity);
            line += "\n";
            line += summary;
            if (number("ntt_naf_memory_mb", naf_mb) && naf_mb)
                line += " | NAF " + std::to_string(naf_mb) + " MB";
        }
    }
    return line;
}

inline std::string stage1_handoff(const ecmgui::Stage1HandoffPlan& plan,
                                  const ecmgui::Stage1Execution& execution, bool zh, bool details = false) {
    std::string line = "Stage2 → ";
    if (!plan.prime95 && !plan.local)
        return line + text("worker.stage1_handoff_off", zh);
    bool local = plan.local, prime95 = plan.prime95, local_fallback = plan.local_fallback;
    bool fallback_selected = false;
    if (!plan.legacy && execution.valid()) {
        const bool prime95_compatible = execution.param == "m0" || execution.param == "m3";
        fallback_selected = prime95 && !prime95_compatible && local_fallback;
        local = local || fallback_selected;
        prime95 = prime95 && prime95_compatible;
        local_fallback = prime95 && local_fallback;
        if (!local && !prime95)
            return line + text("worker.stage1_handoff_incompatible", zh);
    }
    const auto append_workers = [&](const std::string& workers) {
        if (!workers.empty())
            line += " (" + std::string(text("worker.stage1_handoff_workers", zh)) +
                (workers == "auto" ? " " : "") + workers + ")";
    };
    if (prime95) {
        line += "Prime95";
        append_workers(plan.prime95_workers);
        if (!local && !local_fallback)
            line += " (" + std::string(text("worker.stage1_handoff_keep_unsupported", zh)) + ")";
    }
    if (local || local_fallback) {
        if (prime95)
            line += local_fallback ? " → " : " / ";
        line += "ECM-ELF";
        if (local_fallback || fallback_selected)
            line += " (" + std::string(text("worker.stage1_handoff_fallback", zh)) + ")";
        if (plan.edwards_only)
            line += " (" + std::string(text("worker.stage1_handoff_edwards_only", zh)) + ")";
        if (!stage2_worker_slices(plan.local_workers)) {
            append_workers(plan.local_workers);
        } else if (!plan.local_candidates.empty()) {
            std::string workers;
            for (std::size_t i = 0; i < (std::min)(std::size_t{8}, plan.local_candidates.size()); ++i) {
                if (i) workers += ", ";
                workers += std::to_string(plan.local_candidates[i]);
            }
            if (plan.local_candidates.size() > 8) workers += ", …";
            append_workers(workers);
        } else {
            const char* key = plan.local_worker_rules.empty() ? "worker.stage1_handoff_invalid_workers" :
                plan.local_status.bits ? "worker.stage1_handoff_no_worker" : "worker.stage1_handoff_waiting_width";
            line += " (" + std::string(text(key, zh)) + ")";
        }
    }
    if (details) {
        if (prime95) {
            line += "\n" + std::string(text("worker.stage1_handoff_prime95_queue", zh)) + "  " +
                (plan.prime95_queue.empty() ? text("worker.stage1_handoff_missing", zh) : plan.prime95_queue);
            line += "\n" + std::string(text("worker.stage1_handoff_prime95_save", zh)) + "  " +
                (plan.prime95_save_dir.empty() ? text("worker.stage1_handoff_missing", zh) : plan.prime95_save_dir);
        }
        if ((local || local_fallback) && !plan.local_queue.empty())
            line += "\n" + std::string(text("worker.stage1_handoff_local_queue", zh)) + "  " + plan.local_queue;
        if ((local || local_fallback) && stage2_worker_slices(plan.local_workers))
            line += "\nhandoff_workers = " + plan.local_workers;
    }
    return line;
}

inline std::string worker_handoff(const ecmgui::WorkerView& worker,bool zh,bool details=false) {
    if(ecmgui::task_display(worker.task).pm1)
        return std::string("Stage2 → ")+(worker.stage1_handoff.pm1_prime95?"Prime95":text("worker.stage1_handoff_off",zh));
    return stage1_handoff(worker.stage1_handoff,worker.stage1_execution,zh,details);
}
inline std::string stage2_shape(const ecmgui::Stage2Curve& s) {
    if(!s.valid)return {};
    const auto value=[](unsigned long long n){return n?std::to_string(n):std::string("--");};
    return "B1="+value(s.b1)+"  B2="+value(s.b2)+"  S="+value(s.bits)+" bits  D="+value(s.d)+"  P="+value(s.p);
}
inline std::string worker_shape(const ecmgui::WorkerView& worker) {
    return worker.stage2?stage2_shape(worker.stage2_status):std::string{};
}
inline std::string stage2_phase_name(const ecmgui::Stage2Curve& s,bool zh){
    const int phase=stage2_phase_index(s.phase);
    if(phase>=0){const auto key=std::string("stage2.phase.")+stage2_phase_keys[phase];return text(key.c_str(),zh);}
    if(s.phase=="curve_done")return text("stage2.phase.curve_done",zh);
    if(s.phase=="task_done")return text("stage2.phase.task_done",zh);
    if(s.phase=="complete")return text("stage2.phase.ended",zh);
    return s.phase.empty()?std::string(text("waiting_for_progress",zh)):s.phase;
}
inline std::string stage2_construction(const ecmgui::Stage2Curve& s,unsigned long long ratio,bool zh){
    const bool mont=s.curve_model=="mont"||s.curve_model=="montgomery";
    const bool ed=s.curve_model=="ed"||s.curve_model=="edwards";
    std::string line=!s.valid?text("worker.curve_unknown",zh):mont?text("worker.curve_montgomery",zh):ed?text("worker.curve_edwards",zh):s.curve_model;
    if(s.valid&&mont&&s.param>=0)line+=" param="+std::to_string(s.param);
    const auto value=[](unsigned long long n){return n?std::to_string(n):std::string("--");};
    line+="  B1="+value(s.b1)+"  B2=";
    if(ratio&&s.b1&&s.b2&&s.b1<=std::numeric_limits<unsigned long long>::max()/ratio&&s.b1*ratio==s.b2)line+=std::to_string(ratio)+"*B1=";
    line+=value(s.b2)+"  sigma="+value(s.sigma)+"  D="+value(s.d)+"  P="+value(s.p);
    return line;
}
inline std::string worker_construction(const ecmgui::WorkerView& worker,unsigned long long ratio,bool zh){
    return stage2_construction(worker.stage2_status,ratio,zh);
}
inline std::string stage2_curve_label(const ecmgui::Stage2Curve& s,bool zh) {
    char label[100];std::snprintf(label,sizeof(label),text("stage2.curve_identity",zh),s.current,s.total);
    return label;
}
inline std::string worker_phase(const ecmgui::WorkerView& worker,bool zh) {
    if(!worker.stage2||!worker.stage2_status.valid)return {};
    const auto& s=worker.stage2_status;char text[256];
    std::snprintf(text,sizeof(text),ecmgui_v2::ui::text("worker.stage2_phase_details",zh),s.phase.c_str(),s.current,s.total,s.batch,s.batches,s.b1,s.b2);
    std::string result=text;
    if(s.sigma){result+=" | "+s.curve_model;if(s.curve_model!="ed"&&s.curve_model!="edwards")result+=" param="+std::to_string(s.param);result+=" sigma="+std::to_string(s.sigma);}
    return result;
}
}
