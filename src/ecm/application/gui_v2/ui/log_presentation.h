#pragma once
#include "theme.h"
#include "worker_presentation.h"
namespace ecmgui_v2::ui {
enum class LogTone {Text,Info,Stage1,Stage2,Success,Warning,Error};
inline LogTone log_tone(const ecmgui::ParsedLine& line) {
    const auto& text=line.plain;
    if(line.kind==ecmgui::LogKind::Error)return text.find("WARN")!=text.npos||text.find("warn")!=text.npos?LogTone::Warning:LogTone::Error;
    if(line.is_hit||line.queue_done)return LogTone::Success;
    if(line.stage2.kind!=ecmgui::Stage2Event::Kind::None)return LogTone::Stage2;
    if(line.kind==ecmgui::LogKind::Progress)return LogTone::Stage1;
    if(text.find("START:")==0||text.find("Resuming")==0)return text.find("ECMSTAGE2")!=text.npos?LogTone::Stage2:LogTone::Stage1;
    if(text.find("Checkpoint")==0||text.find("queue_owner:")==0||text.find("p95_add:")==0)return LogTone::Info;
    return LogTone::Text;
}
inline ImVec4 log_color(LogTone tone,const Palette& p) {
    switch(tone){case LogTone::Info:return p.info;case LogTone::Stage1:return p.stage1;case LogTone::Stage2:return p.stage2;
    case LogTone::Success:return p.success;case LogTone::Warning:return p.warning;case LogTone::Error:return p.danger;default:return p.fg;}
}
// One fixed-height row: timestamp stays muted, event tag and body use semantic colors.
inline void log_line(const std::string& source,const Palette& p) {
    const auto line=ecmgui::parse_line(source);auto text=line.plain;
    if(line.progress.valid&&line.progress.s_per_curve>0&&std::isfinite(line.progress.eta_s)&&line.progress.eta_s>=60){
        const auto eta=eta_text(line.progress.eta_s,true);const auto reduced=eta.find('(');
        if(reduced!=eta.npos)text+="  "+eta.substr(reduced);
    }
    const auto origin=ImGui::GetCursorScreenPos();auto* draw=ImGui::GetWindowDrawList();float x=origin.x;
    if(!line.timestamp.empty()){
        const auto stamp="["+line.timestamp+"] ";draw->AddText({x,origin.y},ImGui::GetColorU32(p.muted),stamp.c_str());x+=ImGui::CalcTextSize(stamp.c_str()).x;
    }
    const auto tone=log_tone(line);const auto color=log_color(tone,p);
    const auto colon=text.find(':');
    if(colon!=text.npos&&colon<72){
        const auto tag=text.substr(0,colon+1);draw->AddText({x,origin.y},ImGui::GetColorU32(color),tag.c_str());x+=ImGui::CalcTextSize(tag.c_str()).x;
        const auto body=text.substr(colon+1);draw->AddText({x,origin.y},ImGui::GetColorU32(tone==LogTone::Info?p.fg:color),body.c_str());x+=ImGui::CalcTextSize(body.c_str()).x;
    }else{draw->AddText({x,origin.y},ImGui::GetColorU32(color),text.c_str());x+=ImGui::CalcTextSize(text.c_str()).x;}
    ImGui::Dummy({x-origin.x,ImGui::GetTextLineHeight()});
}
}
