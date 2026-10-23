#pragma once
#include "probability_presets.h"
#include "i18n.h"
#include <map>
namespace ecmgui_v2::ui {
inline bool curve_buttons(const char* id,int& value,bool generator,bool zh,std::map<int,std::pair<ImVec2,ImVec2>>& controls,bool order=false){
    bool changed=false;controls.clear();ImGui::PushID(id);
    if(ImGui::BeginTable("families",2,ImGuiTableFlags_SizingStretchProp)){
        const auto ed=ui::text("curve.edwards_title",zh),mont=ui::text("curve.montgomery_title",zh);
        ImGui::TableSetupColumn("family",ImGuiTableColumnFlags_WidthFixed,std::max(ImGui::CalcTextSize(ed).x,ImGui::CalcTextSize(mont).x));
        ImGui::TableSetupColumn("profiles",ImGuiTableColumnFlags_WidthStretch);
        for(bool edwards:{true,false}){
            ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(edwards?ed:mont);ImGui::TableNextColumn();
            const auto ids=edwards?ui::edwards_profiles():std::array<int,4>{0,3,5,4};
            const float left=ImGui::GetCursorScreenPos().x,right=left+ImGui::GetContentRegionAvail().x;
            bool first=true;
            for(int index:ids){const auto& preset=ui::curve_presets[index];if(generator&&!preset.generator)continue;
                const float width=ImGui::CalcTextSize(preset.label).x+ImGui::GetStyle().FramePadding.x*2;
                if(!first&&ImGui::GetItemRectMax().x+ImGui::GetStyle().ItemSpacing.x+width<=right)ImGui::SameLine();
                const bool selected=value==index;if(selected)ImGui::PushStyleColor(ImGuiCol_Button,ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
                const bool unsupported=order&&index!=0&&index!=1&&index!=3&&index!=4&&index!=5;ImGui::BeginDisabled(unsupported);
                if(ImGui::Button(preset.label)){value=index;changed=true;}
                ImGui::EndDisabled();
                controls[index]={ImGui::GetItemRectMin(),ImGui::GetItemRectMax()};
                if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){if(unsupported)ImGui::SetTooltip("%s",ui::text("go.unsupported",zh));else ImGui::SetTooltip(ui::text("probability.profile_values",zh),preset.source,ui::curve_divisor(index,true),ui::curve_divisor(index,false));}
                if(selected)ImGui::PopStyleColor();first=false;
            }
        }ImGui::EndTable();
    }ImGui::PopID();return changed;
}
}
