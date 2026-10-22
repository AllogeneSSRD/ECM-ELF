#pragma once
#include "../../gui/worker_controller.h"
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
    out.label=std::string(giant?(zh?"Giant 批次":"Giant batch"):(zh?"批次":"Batch"))+" "+std::to_string(s.batch)+" / "+std::to_string(s.batches);
    char text[96];if(std::isfinite(s.batch_left)&&s.batch_left>=0)std::snprintf(text,sizeof(text),zh?"阶段剩余 %.1f s":"Phase remaining %.1f s",s.batch_left);
    else std::snprintf(text,sizeof(text),"%s",zh?"阶段剩余 --":"Phase remaining --");
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
        std::snprintf(text,sizeof(text),zh?"%.1f%% (已恢复)":"%.1f%% (resumed)",pct);out.label=text;
    } else if(worker.running()||worker.state()==ecmgui::WorkerRunState::Restarting)out.label=zh?"等待首次进度":"Waiting for progress";
    return out;
}
inline std::string worker_shape(const ecmgui::WorkerView& worker) {
    if(!worker.stage2||!worker.stage2_status.valid)return {};
    const auto& s=worker.stage2_status;
    const auto value=[](unsigned long long n){return n?std::to_string(n):std::string("--");};
    return "B1="+value(s.b1)+"  B2="+value(s.b2)+"  S="+value(s.bits)+" bits  D="+value(s.d)+"  P="+value(s.p);
}
inline std::string worker_phase(const ecmgui::WorkerView& worker,bool zh) {
    if(!worker.stage2||!worker.stage2_status.valid)return {};
    const auto& s=worker.stage2_status;char text[256];
    std::snprintf(text,sizeof(text),zh?"阶段 %s | 曲线 %llu/%llu | 批次 %llu/%llu | B1=%llu B2=%llu":"Phase %s | curve %llu/%llu | batch %llu/%llu | B1=%llu B2=%llu",s.phase.c_str(),s.current,s.total,s.batch,s.batches,s.b1,s.b2);
    std::string result=text;
    if(s.sigma){result+=" | "+s.curve_model;if(s.curve_model!="ed"&&s.curve_model!="edwards")result+=" param="+std::to_string(s.param);result+=" sigma="+std::to_string(s.sigma);}
    return result;
}
}
