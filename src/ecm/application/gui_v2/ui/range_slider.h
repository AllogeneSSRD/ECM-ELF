#pragma once
#include "widgets.h"
#include "../gpu_clocks.h"
#include <algorithm>
#include <cmath>

namespace ecmgui_v2::ui {
struct RangeSliderAudit {ImVec2 min{},max{},low{},high{};};
// Two independently focusable handles share a single MHz axis. Equal endpoints
// are allowed; at overlap, click its left/right half to choose the lower/upper.
inline bool clock_range_slider(const char* id,ClockRange& value,ClockRange bounds,const Palette& p,RangeSliderAudit* audit=nullptr){
    ImGui::PushID(id);
    const auto origin=ImGui::GetCursorScreenPos();const float scale=ImGui::GetFontSize()/15.f;
    const float width=std::max(48.f*scale,ImGui::GetContentRegionAvail().x),height=32*scale,radius=6*scale;
    const float left=origin.x+radius+2*scale,right=origin.x+width-radius-2*scale,mid_y=origin.y+height*.5f;
    const auto x=[&](int n){return bounds.high>bounds.low?left+(right-left)*float(n-bounds.low)/float(bounds.high-bounds.low):(left+right)*.5f;};
    const float split=(x(value.low)+x(value.high))*.5f;
    bool disabled=false;
    bool changed=false;int focused=-1;
    for(int i=0;i<2;++i){
        const float a=i?split:origin.x,b=i?origin.x+width:split;
        ImGui::SetCursorScreenPos({a,origin.y});ImGui::InvisibleButton(i?"max":"min",{std::max(1.f,b-a),height},ImGuiButtonFlags_EnableNav);
        disabled=(ImGui::GetItemFlags()&ImGuiItemFlags_Disabled)!=0;
        int n=i?value.high:value.low;
        if(ImGui::IsItemActive()&&ImGui::IsMouseDown(ImGuiMouseButton_Left)&&!disabled){
            const float fraction=std::clamp((ImGui::GetIO().MousePos.x-left)/(right-left),0.f,1.f);
            n=bounds.low+int(std::lround(fraction*(bounds.high-bounds.low)));
        }
        if(ImGui::IsItemFocused()&&!disabled){
            focused=i;const int step=ImGui::GetIO().KeyCtrl?10:1;
            if(ImGui::IsKeyPressed(ImGuiKey_LeftArrow))n-=step;
            if(ImGui::IsKeyPressed(ImGuiKey_RightArrow))n+=step;
            if(ImGui::IsKeyPressed(ImGuiKey_Home))n=bounds.low;
            if(ImGui::IsKeyPressed(ImGuiKey_End))n=bounds.high;
        }
        n=std::clamp(n,i?value.low:bounds.low,i?bounds.high:value.high);
        auto& current=i?value.high:value.low;if(n!=current){current=n;changed=true;}
    }
    auto* draw=ImGui::GetWindowDrawList();
    draw->AddLine({left,mid_y},{right,mid_y},ImGui::GetColorU32(p.border),3*scale);
    draw->AddLine({x(value.low),mid_y},{x(value.high),mid_y},ImGui::GetColorU32(p.focus),4*scale);
    for(int i=0;i<2;++i){const ImVec2 center{x(i?value.high:value.low),mid_y};
        draw->AddCircleFilled(center,radius+2*scale,ImGui::GetColorU32(p.surface));
        draw->AddCircleFilled(center,radius,ImGui::GetColorU32(disabled?p.muted:p.focus));
        draw->AddCircleFilled(center,2.3f*scale,ImGui::GetColorU32(p.bg));
        if(focused==i)draw->AddCircle(center,radius+3*scale,ImGui::GetColorU32(p.focus),0,scale);
    }
    if(audit)*audit={origin,{origin.x+width,origin.y+height},{x(value.low),mid_y},{x(value.high),mid_y}};
    ImGui::SetCursorScreenPos({origin.x,origin.y+height+ImGui::GetStyle().ItemSpacing.y});
    ImGui::PopID();return changed;
}
}
