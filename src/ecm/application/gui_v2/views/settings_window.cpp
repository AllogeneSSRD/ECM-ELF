#include "../app.h"
#include "../ui/i18n.h"
#include <algorithm>
namespace ecmgui_v2 {
void App::open_settings(int category){
    if(category>=0&&category<6)settings_category=category;
    settings_open=true;focus_settings_=true;
}
void App::draw_settings_window(const ui::Settings& active,const ui::Fonts& fonts){
    if(!settings_open)return;
    const bool zh=fonts.chinese;const auto p=ui::palette(active.light);
    const auto* viewport=ImGui::GetMainViewport();const auto origin=viewport->WorkPos,area=viewport->WorkSize;
    fonts.push(ui::FontRole::Body);
    // Live font preview must not move a slider horizontally while it is held.
    // Reflow the category shell on release; font rendering still updates live.
    if(settings_geometry_scale_<=0||!ImGui::IsMouseDown(0))settings_geometry_scale_=ImGui::GetFontSize()/15.f;
    const float scale=settings_geometry_scale_;
    const ImVec2 limit{std::max(1.f,area.x-24),std::max(1.f,area.y-24)};
    ImGui::SetNextWindowViewport(viewport->ID);
    if(place_settings_){
        ImGui::SetNextWindowPos({origin.x+area.x*.5f,origin.y+area.y*.5f},ImGuiCond_Always,{.5f,.5f});
        ImGui::SetNextWindowSize({std::min(1100*scale,limit.x),std::min(760*scale,limit.y)});place_settings_=false;
    }else if(settings_maximized){ImGui::SetNextWindowPos({origin.x+12,origin.y+12});ImGui::SetNextWindowSize(limit);}
    ImGui::SetNextWindowSizeConstraints({std::min(480*scale,limit.x),std::min(420*scale,limit.y)},limit);
    if(focus_settings_){ImGui::SetNextWindowFocus();focus_settings_=false;}
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{16*scale,12*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize,scale);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{10*scale,5*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{8*scale,8*scale});
    ImGui::PushStyleColor(ImGuiCol_WindowBg,p.bg);
    const auto title=std::string(ui::text("settings",zh))+"###v2.settings";
    ImGui::Begin(title.c_str(),nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
    const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    const float header=32*scale;auto position=ImGui::GetWindowPos(),size=ImGui::GetWindowSize();
    if(size.x>limit.x||size.y>limit.y)ImGui::SetWindowSize({std::min(size.x,limit.x),std::min(size.y,limit.y)});
    position.x=std::clamp(position.x,origin.x+12,origin.x+12+std::max(0.f,limit.x-size.x));
    position.y=std::clamp(position.y,origin.y+12,origin.y+12+std::max(0.f,limit.y-size.y));ImGui::SetWindowPos(position);
    settings_min=position;settings_max={position.x+size.x,position.y+size.y};
    const auto header_pos=ImGui::GetCursorScreenPos();
    const float drag_width=std::max(1.f,ImGui::GetContentRegionAvail().x-2*header-16*scale);
    ImGui::InvisibleButton("settings-drag",{drag_width,header});
    if(ImGui::IsItemActive()&&ImGui::IsMouseDragging(0)&&!settings_maximized){
        const auto delta=ImGui::GetIO().MouseDelta;ImGui::SetWindowPos({position.x+delta.x,position.y+delta.y});
    }
    ui::line_icon(ImGui::GetWindowDrawList(),ui::Icon::Settings,{header_pos.x,header_pos.y+7*scale},18*scale,ImGui::GetColorU32(p.muted));
    ImGui::GetWindowDrawList()->AddText({header_pos.x+28*scale,header_pos.y+(header-ImGui::GetFontSize())*.5f},ImGui::GetColorU32(p.fg),ui::text("settings",zh));
    ImGui::SameLine();if(ImGui::Button("##settings-expand",{header,header})){
        if(settings_maximized){settings_maximized=false;ImGui::SetWindowPos(settings_restore_pos_);ImGui::SetWindowSize(settings_restore_size_);}
        else {settings_restore_pos_=position;settings_restore_size_=size;settings_maximized=true;}
    }
    settings_expand=center();const float radius=4*scale;auto* draw=ImGui::GetWindowDrawList();
    draw->AddRect({settings_expand.x-radius,settings_expand.y-radius},{settings_expand.x+radius,settings_expand.y+radius},ImGui::GetColorU32(p.muted),0,0,scale);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("window.maximize_restore",zh));
    ImGui::SameLine();if(ImGui::Button("##settings-close",{header,header}))settings_open=false;
    settings_close=center();draw->AddLine({settings_close.x-radius,settings_close.y-radius},{settings_close.x+radius,settings_close.y+radius},ImGui::GetColorU32(p.muted),scale);
    draw->AddLine({settings_close.x-radius,settings_close.y+radius},{settings_close.x+radius,settings_close.y-radius},ImGui::GetColorU32(p.muted),scale);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("window.close",zh));
    if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)&&!ImGui::GetIO().WantTextInput&&ImGui::IsKeyPressed(ImGuiKey_Escape))settings_open=false;
    ImGui::Separator();
    const char* categories[]{"settings_appearance","settings_layout","settings_monitor","settings_pet","settings_workflow","settings_compute"};
    const ui::Icon icons[]{ui::Icon::Palette,ui::Icon::Layout,ui::Icon::Chart,ui::Icon::Pet,ui::Icon::Folder,ui::Icon::Cpu};
    const float available=ImGui::GetContentRegionAvail().x;
    const float footer=ImGui::GetTextLineHeight()*2+56*scale+(config.dirty()?ImGui::GetTextLineHeight()+8*scale:0);
    settings_sidebar_stacked=available<850*scale;
    for(auto& c:settings_categories)c={};for(auto& c:theme_controls)c={};
    font_size_control=primary_offset_control=secondary_offset_control=settings_search_control={};
    for(auto& c:gpu_range_controls)c={};for(auto& c:pet_skin_controls)c={};
    const float row_height=std::max(36*scale,ImGui::GetTextLineHeight()+12*scale);
    if(settings_sidebar_stacked){
        ImGui::SetNextItemWidth(-1);
        if(ImGui::BeginCombo("##settings-category",ui::text(categories[settings_category],zh))){
            for(int i=0;i<6;++i){if(ImGui::Selectable(ui::text(categories[i],zh),settings_category==i))settings_category=i;settings_categories[i]=center();}
            ImGui::EndCombo();
        }
    }else{
        ImGui::BeginChild("settings-categories",{210*scale,std::max(1.f,ImGui::GetContentRegionAvail().y-footer)},ImGuiChildFlags_None);
        for(int i=0;i<6;++i){ImGui::PushID(i);
            if(ui::navigation_button("##section",ui::text(categories[i],zh),icons[i],settings_category==i,p,scale,row_height))settings_category=i;
            settings_categories[i]=center();ImGui::PopID();
        }
        ImGui::EndChild();ImGui::SameLine();
    }
    const auto content_id="settings-content-"+std::to_string(settings_category);
    ImGui::BeginChild(content_id.c_str(),{0,std::max(1.f,ImGui::GetContentRegionAvail().y-footer)},ImGuiChildFlags_Borders);
    settings_content_min=ImGui::GetCursorScreenPos();
    const auto content_pos=ImGui::GetWindowPos(),content_size=ImGui::GetWindowSize(),padding=ImGui::GetStyle().WindowPadding;
    settings_content_max={content_pos.x+content_size.x-padding.x,content_pos.y+content_size.y-padding.y};
    fonts.push(ui::FontRole::Primary);ImGui::TextUnformatted(ui::text(categories[settings_category],zh));ImGui::PopFont();ImGui::Spacing();
    ImGui::PushTextWrapPos(0);
    if(settings_category==5&&!ready)ui::secondary_text(ui::text("settings_no_workflow",zh));
    draw_settings(zh);ImGui::PopTextWrapPos();ImGui::EndChild();
    ImGui::Separator();
    if(ui::button("save-ui",ui::text("save_ui_preferences",zh),ui::ButtonKind::Primary,p,scale))save_preferences();
    save_ui_control=center();ImGui::SameLine();
    if(ui::button("settings-done",ui::text("settings_done",zh),ui::ButtonKind::Secondary,p,scale))settings_open=false;
    if(config.dirty())ui::secondary_text(ui::text("unsaved_workflow_changes",zh));
    ui::secondary_text(state.writable?ui::text("settings_ui_saved_hint",zh):state.notice.c_str());
    ImGui::End();ImGui::PopStyleColor();ImGui::PopStyleVar(4);ImGui::PopFont();
}
}
