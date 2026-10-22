#include "i18n.h"
#include "widgets.h"
#include "../version.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace ecmgui_v2::ui {
bool navigation_button(const char* id,const char* label,Icon shape,bool selected,const Palette& p,float scale,float height){
    ImGui::PushStyleColor(ImGuiCol_Button,selected?p.active:ImVec4{0,0,0,0});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,p.hover);ImGui::PushStyleColor(ImGuiCol_ButtonActive,p.active);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,0);ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,8*scale);
    const float width=std::max(1.f,ImGui::GetContentRegionAvail().x);
    const bool clicked=ImGui::Button(id,{width,height});const auto pos=ImGui::GetItemRectMin();
    const auto color=ImGui::GetColorU32(selected?p.fg:p.muted);auto* draw=ImGui::GetWindowDrawList();
    const auto shown=fitted_label(label,width-48*scale);
    draw->PushClipRect(pos,{pos.x+width,pos.y+height},true);
    line_icon(draw,shape,{pos.x+12*scale,pos.y+(height-18*scale)*.5f},18*scale,color);
    draw->AddText({pos.x+40*scale,pos.y+(height-ImGui::GetFontSize())*.5f},color,shown.c_str());draw->PopClipRect();
    if(shown!=label&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",label);
    ImGui::PopStyleVar(2);ImGui::PopStyleColor(3);return clicked;
}
bool icon_button(const char* id,Icon shape,const Palette& p,float scale){
    const bool clicked=ImGui::Button(id,{30*scale,30*scale});const auto pos=ImGui::GetItemRectMin();
    line_icon(ImGui::GetWindowDrawList(),shape,{pos.x+6*scale,pos.y+6*scale},18*scale,ImGui::GetColorU32(p.muted));return clicked;
}
bool theme_card(const char* id,const char* label,bool light,bool selected,const Palette& p,float scale,float width){
    const float height=150*scale;ImGui::PushStyleColor(ImGuiCol_Button,p.surface);
    ImGui::PushStyleColor(ImGuiCol_Border,selected?p.focus:p.border);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,8*scale);ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,scale);
    const bool clicked=ImGui::Button(id,{width,height});const auto pos=ImGui::GetItemRectMin();auto* draw=ImGui::GetWindowDrawList();
    const auto dot=ImVec2{pos.x+20*scale,pos.y+24*scale};draw->AddCircle(dot,6*scale,ImGui::GetColorU32(selected?p.focus:p.border),16,scale);
    if(selected)draw->AddCircleFilled(dot,3*scale,ImGui::GetColorU32(p.focus));
    draw->AddText({pos.x+34*scale,pos.y+24*scale-ImGui::GetFontSize()*.5f},ImGui::GetColorU32(p.fg),label);
    const auto preview=palette(light);const auto lo=ImVec2{pos.x+18*scale,pos.y+50*scale},hi=ImVec2{pos.x+width-18*scale,pos.y+height-16*scale};
    draw->AddRectFilled(lo,hi,ImGui::GetColorU32(preview.bg),4*scale);draw->AddRect(lo,hi,ImGui::GetColorU32(p.border),4*scale);
    const float sidebar=(hi.x-lo.x)*.22f;draw->AddRectFilled(lo,{lo.x+sidebar,hi.y},ImGui::GetColorU32(preview.sunken),4*scale);
    for(int i=0;i<3;++i){const float y=lo.y+(14+12*i)*scale;draw->AddLine({lo.x+8*scale,y},{lo.x+sidebar-8*scale,y},ImGui::GetColorU32(preview.muted),3*scale);}
    const float start=lo.x+sidebar+12*scale,space=hi.x-start-12*scale;
    const ImVec4 colors[]{preview.fg,preview.info,preview.stage2,preview.success};
    for(int i=0;i<4;++i){const float y=lo.y+(17+i*14)*scale;draw->AddLine({start,y},{start+space*(i==1?.9f:i==2?.75f:.55f),y},ImGui::GetColorU32(colors[i]),3*scale);}
    ImGui::PopStyleVar(2);ImGui::PopStyleColor(2);return clicked;
}
namespace {
void flow_next(float width) {
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    if (ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + width <= right) ImGui::SameLine();
}
}
std::string fitted_label(const char* text,float width){
    if(width<=0)return {};
    if(ImGui::CalcTextSize(text).x<=width)return text;
    constexpr const char* ellipsis="\xe2\x80\xa6";
    const float tail=ImGui::CalcTextSize(ellipsis).x;if(width<tail)return {};
    const char* remaining=text;
    ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(),width-tail,0,text,nullptr,&remaining);
    return std::string(text,remaining)+ellipsis;
}
bool button(const char* id, const char* label, ButtonKind kind, const Palette& p, float scale, bool disabled,float* rendered_size,bool compact,float width,float height,Icon shape) {
    const bool primary = kind == ButtonKind::Primary || kind == ButtonKind::Selected;
    if(kind==ButtonKind::Primary)ImGui::PushFont(ImGui::GetFont(),primary_font_size(ImGui::GetStyle().FontSizeBase));
    if(rendered_size)*rendered_size=ImGui::GetFontSize();
    const auto normal = primary ? p.accent : kind == ButtonKind::Ghost ? ImVec4(0, 0, 0, 0) : p.surface;
    const auto text = primary ? p.on_accent : kind == ButtonKind::Danger ? p.danger : p.fg;
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    ImGui::PushStyleColor(ImGuiCol_Button, normal);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, primary ? p.accent_hover : p.hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, primary ? p.muted : p.active);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, kind == ButtonKind::Ghost ? 0.f : scale);
    if(compact)ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{(width>0?4.f:8.f)*scale,4*scale});
    ImGui::BeginDisabled(disabled);
    const bool with_icon=shape!=Icon::None;
    const float icon_size=ImGui::GetFontSize()*14/15,icon_gap=ImGui::GetFontSize()/3;
    if(with_icon&&width<=0)width=ImGui::CalcTextSize(label).x+icon_size+icon_gap+ImGui::GetStyle().FramePadding.x*2;
    const std::string stable_label = std::string(with_icon?"":label) + "###" + id;
    const auto clicked = ImGui::Button(stable_label.c_str(), {width, height>0?height:std::max((compact?24:32)*scale,ImGui::GetFontSize()+(compact?8:12)*scale)});
    if(with_icon){
        const auto lo=ImGui::GetItemRectMin(),hi=ImGui::GetItemRectMax();
        const float inner=std::max(0.f,hi.x-lo.x-8*scale);const bool show_icon=inner>=icon_size;
        const float span=show_icon?icon_size+icon_gap:0;
        const auto shown=fitted_label(label,inner-span);const float left=lo.x+(hi.x-lo.x-span-ImGui::CalcTextSize(shown.c_str()).x)*.5f;
        auto* draw=ImGui::GetWindowDrawList();draw->PushClipRect(lo,hi,true);
        if(show_icon)line_icon(draw,shape,{left,lo.y+(hi.y-lo.y-icon_size)*.5f},icon_size,ImGui::GetColorU32(ImGuiCol_Text));
        draw->AddText({left+span,lo.y+(hi.y-lo.y-ImGui::GetFontSize())*.5f},ImGui::GetColorU32(ImGuiCol_Text),shown.c_str());draw->PopClipRect();
        if(shown!=label&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))ImGui::SetTooltip("%s",label);
    }
    ImGui::EndDisabled(); ImGui::PopStyleVar(compact?2:1); ImGui::PopStyleColor(4);
    if(kind==ButtonKind::Primary)ImGui::PopFont();
    return clicked;
}
void secondary_text(const char* text,float* rendered_size){
    ImGui::PushFont(ImGui::GetFont(),secondary_font_size(ImGui::GetStyle().FontSizeBase));
    if(rendered_size)*rendered_size=ImGui::GetFontSize();
    ImGui::PushTextWrapPos(0);ImGui::TextDisabled("%s",text);ImGui::PopTextWrapPos();ImGui::PopFont();
}
void status_badge(const char* label, Status status, const Palette& p, float scale,float max_width) {
    const std::array<ImVec4, 4> foreground{p.success, p.warning, p.danger, p.info};
    const std::array<ImVec4, 4> background{p.success_bg, p.warning_bg, p.danger_bg, p.info_bg};
    const auto index = static_cast<size_t>(status);
    const auto pos = ImGui::GetCursorScreenPos();
    const auto text = ImGui::CalcTextSize(label);
    const ImVec2 size{max_width>0?std::min(text.x+26*scale,max_width):text.x+26*scale,text.y+8*scale};
    auto draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, ImGui::ColorConvertFloat4ToU32(background[index]), 4 * scale);
    draw->PushClipRect(pos,{pos.x+size.x,pos.y+size.y},true);
    draw->AddCircleFilled({pos.x + 9 * scale, pos.y + size.y / 2}, 2.5f * scale, ImGui::ColorConvertFloat4ToU32(foreground[index]));
    const auto shown=max_width>0?fitted_label(label,size.x-21*scale):std::string(label);
    draw->AddText({pos.x + 17 * scale, pos.y + 4 * scale}, ImGui::ColorConvertFloat4ToU32(foreground[index]), shown.c_str());
    draw->PopClipRect();
    ImGui::Dummy(size);
}
void section_header(const char* title, const char* detail, const Fonts& fonts, const Palette& p) {
    fonts.push(FontRole::Heading); ImGui::TextUnformatted(title); ImGui::PopFont();
    fonts.push(FontRole::Caption); ImGui::PushStyleColor(ImGuiCol_Text, p.muted);
    ImGui::TextWrapped("%s", detail); ImGui::PopStyleColor(); ImGui::PopFont();
    ImGui::Spacing();
}
void form_row(const char* label, const char* id, char* value, size_t capacity, float scale) {
    // Stack the label on narrow surfaces, preserving an accessible full-width field.
    const bool wide = ImGui::GetContentRegionAvail().x > 560 * scale;
    ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted(label);
    if (wide) ImGui::SameLine(180 * scale);
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText(id, value, capacity);
}
void icon(Icon shape, const Palette& p, float scale) {
    const auto pos = ImGui::GetCursorScreenPos();
    const float x = pos.x, y = pos.y + 2 * scale, s = 14 * scale;
    auto draw = ImGui::GetWindowDrawList(); const auto c = ImGui::ColorConvertFloat4ToU32(p.muted);
    if (shape == Icon::Grid) {
        for (int i = 0; i < 4; ++i) {
            const ImVec2 a{x + (i % 2) * s * .57f, y + (i / 2) * s * .57f};
            draw->AddRect(a, {a.x + s * .35f, a.y + s * .35f}, c, 1 * scale, 0, scale);
        }
    } else if (shape == Icon::Check) {
        draw->AddLine({x, y + s / 2}, {x + s * .35f, y + s}, c, 2 * scale);
        draw->AddLine({x + s * .35f, y + s}, {x + s, y}, c, 2 * scale);
    } else if (shape == Icon::Folder) {
        draw->AddRect({x, y + s * .25f}, {x + s, y + s}, c, 2 * scale, 0, scale);
        draw->AddLine({x, y}, {x + s * .5f, y}, c, scale);
    } else {
        for (int i = 0; i < 3; ++i) {
            const float row = y + s * i / 2;
            draw->AddLine({x, row}, {x + s, row}, c, scale);
            draw->AddCircleFilled({x + s * (i == 1 ? .7f : .3f), row}, 2 * scale, c);
        }
    }
    ImGui::Dummy({s + 3 * scale, s + 4 * scale});
}
void state_strip(const Palette& p, float scale, bool chinese) {
    const char* keys[]{"normal","hover","pressed","disabled","focus","selected"};
    const ImVec4 fills[] = {p.surface, p.hover, p.active, p.surface, p.surface, p.selection};
    const float width = 88 * scale, height = 32 * scale;
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (int i = 0; i < 6; ++i) {
        if (i && ImGui::GetItemRectMax().x + 8 * scale + width < right) ImGui::SameLine();
        const auto pos = ImGui::GetCursorScreenPos(); auto draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(pos, {pos.x + width, pos.y + height}, ImGui::ColorConvertFloat4ToU32(fills[i]), 6 * scale);
        draw->AddRect(pos, {pos.x + width, pos.y + height}, ImGui::ColorConvertFloat4ToU32(i == 4 ? p.focus : p.border), 6 * scale, 0, (i == 4 ? 2 : 1) * scale);
        const auto label = ecmgui_v2::ui::text(keys[i],chinese); const auto text = ImGui::CalcTextSize(label);
        draw->AddText({pos.x + (width - text.x) / 2, pos.y + (height - text.y) / 2}, ImGui::ColorConvertFloat4ToU32(i == 3 ? p.disabled : p.fg), label);
        ImGui::Dummy({width, height});
    }
}
void Showcase::draw(Settings& pending, const Settings& active, const Fonts& fonts, const char* gpu) {
    const auto p = palette(active.light); const float s = active.scale; const bool zh = fonts.chinese;

    const auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos); ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("ECM-ELF##v2_showcase", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    fonts.push(FontRole::Body); body_pixels = ImGui::GetFontSize();
    const bool wide = ImGui::GetContentRegionAvail().x > 760 * s;
    ImGui::BeginChild("navigation", {wide ? 174 * s : 40 * s, 0}, ImGuiChildFlags_None);
    fonts.push(FontRole::Heading); ImGui::TextUnformatted(wide ? "ECM-ELF" : "E"); ImGui::PopFont();
    if (wide) { fonts.push(FontRole::Caption); ImGui::TextDisabled("GUI v2 / %s", version); ImGui::PopFont(); }
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    icon(Icon::Grid, p, s); if (wide) { ImGui::SameLine(); ImGui::TextUnformatted(ecmgui_v2::ui::text("design_system",zh)); }
    if (ImGui::Selectable(wide ? ecmgui_v2::ui::text("components.components_tab",zh) : "01###components", page == 0, 0, {0, 32 * s})) page = 0;
    if (ImGui::Selectable(wide ? ecmgui_v2::ui::text("components.typography_tab",zh) : "02###typography", page == 1, 0, {0, 32 * s})) page = 1;
    ImGui::EndChild(); ImGui::SameLine();
    ImGui::BeginChild("content", {0, 0}, ImGuiChildFlags_None);
    ImGui::PushTextWrapPos(0);
    section_header(ecmgui_v2::ui::text("components.heading",zh),
        ecmgui_v2::ui::text("components.preview_note",zh), fonts, p);
    if (!fonts.notice.empty()) ImGui::TextWrapped("%s", fonts.notice.c_str());
    // Preferences are applied at the next frame boundary, before any font is used.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * s, 4 * s});
    if (ImGui::BeginTable("preferences", wide ? 3 : 1, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("appearance",zh));
        int theme = pending.light ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##theme", &theme, "Dark\0Light\0")) pending.light = theme != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("menu.language",zh));
        const char* language_names[]{ui::text("language_english",zh),ui::text("language_chinese",zh)}; int language = pending.chinese ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##language", &language, language_names, 2)) pending.chinese = language != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("interface_scale",zh));
        const float scales[]{1, 1.25f, 1.5f, 2}; int index = 0;
        for (int i = 0; i < 4; ++i) if (std::abs(pending.scale - scales[i]) < .01f) index = i;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##scale", &index, "100%\0 125%\0 150%\0 200%\0")) pending.scale = scales[index];
        ImGui::EndTable();
    }
    ImGui::PopStyleVar(); ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (page == 0) {
        ImGui::TextUnformatted(ecmgui_v2::ui::text("actions",zh));
        if (button("primary", ecmgui_v2::ui::text("primary_action",zh), ButtonKind::Primary, p, s)) ++clicks;
        primary_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("secondary",zh)).x + 20 * s);
        button("secondary", ecmgui_v2::ui::text("secondary",zh), ButtonKind::Secondary, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("disabled",zh)).x + 20 * s);
        button("disabled", ecmgui_v2::ui::text("disabled",zh), ButtonKind::Secondary, p, s, true);
        button("ghost", ecmgui_v2::ui::text("ghost_action",zh), ButtonKind::Ghost, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("danger_action",zh)).x + 20 * s);
        button("danger", ecmgui_v2::ui::text("danger_action",zh), ButtonKind::Danger, p, s);
        ImGui::Spacing(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("interaction_states_specimen",zh));
        state_strip(p, s, zh);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(ecmgui_v2::ui::text("form_selection",zh));
        icon(Icon::Folder, p, s); ImGui::SameLine(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("components.path_preview_hint",zh));
        form_row(ecmgui_v2::ui::text("workflow_path",zh), "##path", path, sizeof(path), s);
        field_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("components.enable_option",zh), &selected);
        ImGui::TextDisabled(ecmgui_v2::ui::text("components.primary_click_count",zh), clicks);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(ecmgui_v2::ui::text("status_vocabulary",zh));
        status_badge(ecmgui_v2::ui::text("status.running",zh), Status::Success, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("status.waiting",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("status.waiting",zh), Status::Warning, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("status.error",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("status.error",zh), Status::Danger, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("status.stopped",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("status.stopped",zh), Status::Info, p, s);
        fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("sample_states_no_active_workers",zh)); ImGui::PopFont();
    }
    ImGui::Spacing();
    fonts.push(FontRole::Heading); heading_pixels = ImGui::GetFontSize(); ImGui::TextUnformatted(ecmgui_v2::ui::text("type_precision",zh)); ImGui::PopFont();
    ImGui::TextUnformatted(ecmgui_v2::ui::text("typography.body_example",zh));
    fonts.push(FontRole::Caption); caption_pixels = ImGui::GetFontSize();
    ImGui::TextDisabled("%s", ecmgui_v2::ui::text("typography.caption_focus_hint",zh)); ImGui::PopFont();
    fonts.push(FontRole::Mono); mono_pixels = ImGui::GetFontSize();
    mono_i = ImGui::CalcTextSize("iiiiiiii").x; mono_w = ImGui::CalcTextSize("WWWWWWWW").x;
    ImGui::TextUnformatted("B1  1000000     B2  100000000\nM127  /  sigma=123456789");
    if (fonts.cjk) ImGui::TextUnformatted(ecmgui_v2::ui::text("typography.cjk_example",zh));
    ImGui::PopFont();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage1); ImGui::TextUnformatted("Stage 1"); ImGui::PopStyleColor(); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage2); ImGui::TextUnformatted("Stage 2"); ImGui::PopStyleColor();
    fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", gpu); ImGui::PopFont();
    ImGui::PopTextWrapPos(); ImGui::EndChild(); ImGui::PopFont(); ImGui::End();
}
}
