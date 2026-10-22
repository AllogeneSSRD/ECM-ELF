#pragma once
#include "widgets.h"
#include <functional>
#include <map>

namespace ecmgui_v2::ui {
struct FormField {ImVec2 label_min{},label_max{},input_min{},input_max{},action_min{},action_max{};float right=0;};
struct CompactFormAudit {int columns=0;std::map<std::string,FormField> fields;};
// Measure the full form before submission so every row reserves the same action width.
class CompactForm {
public:
    CompactForm(const char* id,CompactFormAudit& audit,bool paired=true):id_(id),audit_(audit),paired_(paired){}
    ~CompactForm(){draw();}
    void field(const char* id,const char* label,const std::function<void()>& input,
               const char* action_label=nullptr,const std::function<void()>& action={}){
        fields_.push_back({id,label,action_label?action_label:"",input,action});
    }
private:
    struct Field {std::string id,label,action_label;std::function<void()> input,action;};
    std::string id_;CompactFormAudit& audit_;bool paired_;std::vector<Field> fields_;
    void draw(){
        audit_.fields.clear();if(fields_.empty())return;
        const float scale=ImGui::GetFontSize()/15.f,gap=ImGui::GetStyle().ItemInnerSpacing.x;
        float action_width=0;
        for(const auto& field:fields_)if(!field.action_label.empty())action_width=std::max(action_width,checkbox_width(field.action_label.c_str()));
        const float action_reserve=action_width>0?action_width+gap:0;
        const float minimum_cell=126*scale+96*scale+gap+action_reserve+20*scale;
        audit_.columns=paired_&&ImGui::GetContentRegionAvail().x>=std::max(780*scale,minimum_cell*2)?2:1;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{10*scale,ImGui::GetStyle().CellPadding.y});
        if(ImGui::BeginTable(id_.c_str(),audit_.columns,ImGuiTableFlags_SizingStretchSame)){
            for(const auto& field:fields_){
                ImGui::TableNextColumn();ImGui::PushID(field.id.c_str());ImGui::BeginGroup();
                auto& rect=audit_.fields[field.id];
                const float available=ImGui::GetContentRegionAvail().x;
                const float label_width=std::max(16.f,std::min(126*scale,available*.28f));
                const float reserve=std::min(action_reserve,std::max(0.f,available-label_width-gap-std::min(96*scale,available*.35f)));
                const float input_width=std::max(24.f,available-label_width-gap-reserve);
                const float x=ImGui::GetCursorScreenPos().x;rect.right=x+available;
                ImGui::AlignTextToFramePadding();const auto shown=fitted_label(field.label.c_str(),label_width);
                ImGui::TextUnformatted(shown.c_str());rect.label_min=ImGui::GetItemRectMin();rect.label_max=ImGui::GetItemRectMax();
                if(shown!=field.label&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",field.label.c_str());
                ImGui::SameLine(0,gap);ImGui::SetCursorScreenPos({x+label_width+gap,ImGui::GetCursorScreenPos().y});ImGui::SetNextItemWidth(input_width);
                field.input();rect.input_min=ImGui::GetItemRectMin();rect.input_max=ImGui::GetItemRectMax();
                if(!field.action_label.empty()){ImGui::SameLine(0,gap);field.action();rect.action_min=ImGui::GetItemRectMin();rect.action_max=ImGui::GetItemRectMax();}
                ImGui::EndGroup();ImGui::PopID();
            }ImGui::EndTable();
        }ImGui::PopStyleVar();
    }
};
}
