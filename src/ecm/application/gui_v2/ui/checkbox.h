#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include <algorithm>
#include <string>

namespace ecmgui_v2::ui {
struct CheckboxMetrics {ImVec2 min{},max{};float box=0,stroke=0,native_frame=0;bool mixed=false;};
inline float checkbox_size(){return ImGui::GetFontSize()+2*std::max(1.f,ImGui::GetFontSize()/15.f);}
inline float checkbox_width(const char* label){
    const float text=ImGui::CalcTextSize(label,nullptr,true).x;
    return checkbox_size()+(text>0?ImGui::GetStyle().ItemInnerSpacing.x+text:0);
}
inline bool checkbox(const char* label,bool* value,CheckboxMetrics* metrics=nullptr){
    if(ImGui::GetCurrentWindow()->SkipItems)return false;
    const auto& style=ImGui::GetStyle();const float scale=ImGui::GetFontSize()/15.f;
    const float padding=std::max(1.f,scale),box=checkbox_size(),stroke=1.1f*scale;
    const auto pos=ImGui::GetCursorScreenPos();
    const float offset=std::max(0.f,ImGui::GetCurrentWindow()->DC.CurrLineTextBaseOffset-padding);
    ImGui::SetCursorScreenPos({pos.x,pos.y+offset});
    const auto normal=ImGui::GetStyleColorVec4(ImGuiCol_FrameBg),hover=ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered);
    const auto active=ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive),selected=ImGui::GetStyleColorVec4(ImGuiCol_CheckboxSelectedBg);
    const auto border=ImGui::GetColorU32(ImGuiCol_Border),ink=ImGui::GetColorU32(ImGuiCol_CheckMark);
    const float native_frame=ImGui::GetFrameHeight();
    const char* end=ImGui::FindRenderedTextEnd(label);std::string shown(label,end);
    const float label_width=std::max(0.f,ImGui::GetContentRegionAvail().x-box-style.ItemInnerSpacing.x);
    bool clipped=ImGui::CalcTextSize(shown.c_str()).x>label_width;
    if(clipped){
        const float ellipsis=ImGui::CalcTextSize("...").x;
        while(!shown.empty()&&ImGui::CalcTextSize(shown.c_str()).x+ellipsis>label_width){
            size_t last=shown.size()-1;while(last&&(static_cast<unsigned char>(shown[last])&0xc0)==0x80)--last;shown.resize(last);
        }
        if(ellipsis<=label_width)shown+="...";else shown.clear();
    }
    // Keep native interaction, keyboard navigation, disabled and mixed-value behavior.
    // Only the native checkbox drawing is transparent; its label remains untouched.
    for(auto color:{ImGuiCol_FrameBg,ImGuiCol_FrameBgHovered,ImGuiCol_FrameBgActive,ImGuiCol_CheckboxSelectedBg,ImGuiCol_CheckMark})ImGui::PushStyleColor(color,ImVec4{0,0,0,0});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{style.FramePadding.x,padding});ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,0);
    ImGui::PushID(label);const auto native_label=shown+"###check";
    const bool pressed=ImGui::Checkbox(native_label.c_str(),value);ImGui::PopID();
    ImGui::PopStyleVar(2);ImGui::PopStyleColor(5);
    const auto lo=ImGui::GetItemRectMin();const ImVec2 hi{lo.x+box,lo.y+box};
    const bool mixed=(GImGui->LastItemData.ItemFlags&ImGuiItemFlags_MixedValue)!=0;
    if(metrics)*metrics={lo,hi,box,stroke,native_frame,mixed};
    if(ImGui::IsItemVisible()){
        auto* draw=ImGui::GetWindowDrawList();
        const auto bg=ImGui::GetColorU32(ImGui::IsItemActive()?active:ImGui::IsItemHovered()?hover:(*value||mixed)?selected:normal);
        const float rounding=3.5f*scale;draw->AddRectFilled(lo,hi,bg,rounding);draw->AddRect(lo,hi,border,rounding,0,scale);
        if(mixed)draw->AddLine({lo.x+box*.25f,lo.y+box*.5f},{lo.x+box*.75f,lo.y+box*.5f},ink,stroke);
        else if(*value){const ImVec2 points[]{{lo.x+box*.23f,lo.y+box*.51f},{lo.x+box*.43f,lo.y+box*.71f},{lo.x+box*.78f,lo.y+box*.29f}};draw->AddPolyline(points,3,ink,0,stroke);}
    }
    if(clipped&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))ImGui::SetTooltip("%.*s",static_cast<int>(end-label),label);
    return pressed;
}
}
