#pragma once
#include "widgets.h"
#include <functional>
#include <map>

namespace ecmgui_v2::ui {
struct FormField {ImVec2 label_min{},label_max{},input_min{},input_max{},action_min{},action_max{};float right=0;};
struct CompactFormAudit {int columns=0;std::map<std::string,FormField> fields;};
// Pair short fields across the available width; narrow panels use one pair per row.
// Labels stay on the left, and optional field actions stay on the right.
class CompactForm {
public:
    CompactForm(const char* id,CompactFormAudit& audit,bool paired=true):audit_(audit){
        audit.fields.clear();scale_=ImGui::GetFontSize()/15.f;
        audit.columns=paired&&ImGui::GetContentRegionAvail().x>=780*scale_?2:1;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{10*scale_,ImGui::GetStyle().CellPadding.y});
        open_=ImGui::BeginTable(id,audit.columns,ImGuiTableFlags_SizingStretchSame);
    }
    ~CompactForm(){if(open_)ImGui::EndTable();ImGui::PopStyleVar();}
    void field(const char* id,const char* label,const std::function<void()>& input,
               const char* action_label=nullptr,const std::function<void()>& action={}){
        if(!open_)return;
        ImGui::TableNextColumn();ImGui::PushID(id);ImGui::BeginGroup();
        auto& rect=audit_.fields[id];
        const float available=ImGui::GetContentRegionAvail().x,gap=ImGui::GetStyle().ItemInnerSpacing.x;
        const float action_width=action_label?ImGui::CalcTextSize(action_label).x+ImGui::GetFrameHeight()+gap:0;
        const float label_width=std::max(16.f,std::min(126*scale_,available*.28f));
        const float x=ImGui::GetCursorScreenPos().x;rect.right=x+available;
        ImGui::AlignTextToFramePadding();const auto shown=fitted_label(label,label_width);
        ImGui::TextUnformatted(shown.c_str());rect.label_min=ImGui::GetItemRectMin();rect.label_max=ImGui::GetItemRectMax();
        if(shown!=label&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",label);
        ImGui::SameLine(0,gap);ImGui::SetCursorScreenPos({x+label_width+gap,ImGui::GetCursorScreenPos().y});
        ImGui::SetNextItemWidth(std::max(24.f,available-label_width-gap-action_width-(action_label?gap:0)));
        input();rect.input_min=ImGui::GetItemRectMin();rect.input_max=ImGui::GetItemRectMax();
        if(action_label){ImGui::SameLine(0,gap);action();rect.action_min=ImGui::GetItemRectMin();rect.action_max=ImGui::GetItemRectMax();}
        ImGui::EndGroup();ImGui::PopID();
    }
private:
    CompactFormAudit& audit_;bool open_=false;float scale_=1;
};
}
