#include "i18n.h"
#pragma once
#include "../../gui/worker_controller.h"
#include "stage2_progress.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
namespace ecmgui_v2::ui {
struct WorkerProgress {float fraction=0;std::string label="--",speed="--",eta="--",eta_detail="--";};
struct Stage2Batch {bool visible=false;float fraction=0;std::string label,remaining;};
inline Stage2Batch stage2_batch(const ecmgui::Stage2Status& s,bool zh) {
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
inline WorkerProgress worker_progress(const ecmgui::WorkerView& worker,bool zh) {
    WorkerProgress out;const auto p=worker.stage2?worker.stage2_status.progress():worker.progress;
    char text[128];
    if(p.valid&&std::isfinite(p.pct)) {
        const double pct=std::clamp(p.pct,0.,100.);out.fraction=static_cast<float>(pct/100.);
        if(p.curves_total)std::snprintf(text,sizeof(text),"%llu / %llu  (%.1f%%)",p.curves_done,p.curves_total,pct);
        else std::snprintf(text,sizeof(text),"%.1f%%",pct);
        out.label=text;
        if(std::isfinite(p.s_per_curve)&&p.s_per_curve>0){
            std::snprintf(text,sizeof(text),"%.2f",p.s_per_curve);out.speed=text;
            if(std::isfinite(p.eta_s)&&p.eta_s>=0){
                out.eta=eta_text(p.eta_s);out.eta_detail=eta_text(p.eta_s,true);
            }
        }
    } else if(std::isfinite(worker.resume_pct)&&worker.resume_pct>=0) {
        const double pct=std::clamp(worker.resume_pct,0.,100.);out.fraction=static_cast<float>(pct/100.);
        std::snprintf(text,sizeof(text),ecmgui_v2::ui::text("worker.resumed_percent",zh),pct);out.label=text;
    } else if(worker.running()||worker.state()==ecmgui::WorkerRunState::Restarting)out.label=ecmgui_v2::ui::text("waiting_for_progress",zh);
    return out;
}
inline std::string worker_shape(const ecmgui::WorkerView& worker) {
    if(!worker.stage2||!worker.stage2_status.valid)return {};
    const auto& s=worker.stage2_status;
    const auto value=[](unsigned long long n){return n?std::to_string(n):std::string("--");};
    return "B1="+value(s.b1)+"  B2="+value(s.b2)+"  S="+value(s.bits)+" bits  D="+value(s.d)+"  P="+value(s.p);
}
inline std::string stage2_phase_name(const ecmgui::Stage2Status& s,bool zh){
    const int phase=stage2_phase_index(s.phase);
    if(phase>=0){const auto key=std::string("stage2.phase.")+stage2_phase_keys[phase];return text(key.c_str(),zh);}
    if(s.phase=="curve_done")return text("stage2.phase.curve_done",zh);
    if(s.phase=="task_done")return text("stage2.phase.task_done",zh);
    if(s.phase=="complete")return text("stage2.phase.ended",zh);
    return s.phase.empty()?std::string(text("waiting_for_progress",zh)):s.phase;
}
inline std::string worker_construction(const ecmgui::WorkerView& worker,unsigned long long ratio,bool zh){
    const auto& s=worker.stage2_status;
    const bool mont=s.curve_model=="mont"||s.curve_model=="montgomery";
    const bool ed=s.curve_model=="ed"||s.curve_model=="edwards";
    std::string line=!s.valid?text("worker.curve_unknown",zh):mont?text("worker.curve_montgomery",zh):ed?text("worker.curve_edwards",zh):s.curve_model;
    if(s.valid&&mont&&s.param>=0)line+=" param="+std::to_string(s.param);
    const auto value=[](unsigned long long n){return n?std::to_string(n):std::string("--");};
    line+="  B1="+value(s.b1)+"  B2=";
    if(ratio&&s.b1&&s.b2&&s.b1<=std::numeric_limits<unsigned long long>::max()/ratio&&s.b1*ratio==s.b2)line+=std::to_string(ratio)+"*B1=";
    line+=value(s.b2)+"  sigma="+value(s.sigma);return line;
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
