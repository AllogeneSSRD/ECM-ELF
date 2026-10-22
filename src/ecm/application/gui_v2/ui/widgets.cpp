#include "i18n.h"
#include "widgets.h"
#include "../version.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace ecmgui_v2::ui {
namespace {
void flow_next(float width) {
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    if (ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + width <= right) ImGui::SameLine();
}
}
bool button(const char* id, const char* label, ButtonKind kind, const Palette& p, float scale, bool disabled,float* rendered_size,bool compact) {
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
    if(compact)ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*scale,4*scale});
    ImGui::BeginDisabled(disabled);
    const std::string stable_label = std::string(label) + "###" + id;
    const auto clicked = ImGui::Button(stable_label.c_str(), {0, std::max((compact?24:32)*scale,ImGui::GetFontSize()+(compact?8:12)*scale)});
    ImGui::EndDisabled(); ImGui::PopStyleVar(compact?2:1); ImGui::PopStyleColor(4);
    if(kind==ButtonKind::Primary)ImGui::PopFont();
    return clicked;
}
void secondary_text(const char* text,float* rendered_size){
    ImGui::PushFont(ImGui::GetFont(),secondary_font_size(ImGui::GetStyle().FontSizeBase));
    if(rendered_size)*rendered_size=ImGui::GetFontSize();
    ImGui::PushTextWrapPos(0);ImGui::TextDisabled("%s",text);ImGui::PopTextWrapPos();ImGui::PopFont();
}
void status_badge(const char* label, Status status, const Palette& p, float scale) {
    const std::array<ImVec4, 4> foreground{p.success, p.warning, p.danger, p.info};
    const std::array<ImVec4, 4> background{p.success_bg, p.warning_bg, p.danger_bg, p.info_bg};
    const auto index = static_cast<size_t>(status);
    const auto pos = ImGui::GetCursorScreenPos();
    const auto text = ImGui::CalcTextSize(label);
    const ImVec2 size{text.x + 26 * scale, text.y + 8 * scale};
    auto draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, ImGui::ColorConvertFloat4ToU32(background[index]), 4 * scale);
    draw->AddCircleFilled({pos.x + 9 * scale, pos.y + size.y / 2}, 2.5f * scale, ImGui::ColorConvertFloat4ToU32(foreground[index]));
    draw->AddText({pos.x + 17 * scale, pos.y + 4 * scale}, ImGui::ColorConvertFloat4ToU32(foreground[index]), label);
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
    const char* keys[]{"normal_a7248eeb","hover_c76ae86a","pressed_57997acc","disabled_75081b59","focus_9d3cab2b","selected_57fd7a0c"};
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
    icon(Icon::Grid, p, s); if (wide) { ImGui::SameLine(); ImGui::TextUnformatted(ecmgui_v2::ui::text("design_system_4a866105",zh)); }
    if (ImGui::Selectable(wide ? ecmgui_v2::ui::text("components_components_4cf95529",zh) : "01###components", page == 0, 0, {0, 32 * s})) page = 0;
    if (ImGui::Selectable(wide ? ecmgui_v2::ui::text("typography_typography_f1f32d35",zh) : "02###typography", page == 1, 0, {0, 32 * s})) page = 1;
    ImGui::EndChild(); ImGui::SameLine();
    ImGui::BeginChild("content", {0, 0}, ImGuiChildFlags_None);
    ImGui::PushTextWrapPos(0);
    section_header(ecmgui_v2::ui::text("a_quieter_workspace_4771eb10",zh),
        ecmgui_v2::ui::text("p2_component_preview_worker_controls_and_b_c9ecbaff",zh), fonts, p);
    if (!fonts.notice.empty()) ImGui::TextWrapped("%s", fonts.notice.c_str());
    // Preferences are applied at the next frame boundary, before any font is used.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * s, 4 * s});
    if (ImGui::BeginTable("preferences", wide ? 3 : 1, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("appearance_3907fa7f",zh));
        int theme = pending.light ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##theme", &theme, "Dark\0Light\0")) pending.light = theme != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("language_a4fe6526",zh));
        const char* language_names[]{ui::text("language_english",zh),ui::text("language_chinese",zh)}; int language = pending.chinese ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##language", &language, language_names, 2)) pending.chinese = language != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("interface_scale_6ba5218d",zh));
        const float scales[]{1, 1.25f, 1.5f, 2}; int index = 0;
        for (int i = 0; i < 4; ++i) if (std::abs(pending.scale - scales[i]) < .01f) index = i;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##scale", &index, "100%\0 125%\0 150%\0 200%\0")) pending.scale = scales[index];
        ImGui::EndTable();
    }
    ImGui::PopStyleVar(); ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (page == 0) {
        ImGui::TextUnformatted(ecmgui_v2::ui::text("actions_ff8059dc",zh));
        if (button("primary", ecmgui_v2::ui::text("primary_action_da2eb576",zh), ButtonKind::Primary, p, s)) ++clicks;
        primary_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("secondary_62f2ccff",zh)).x + 20 * s);
        button("secondary", ecmgui_v2::ui::text("secondary_62f2ccff",zh), ButtonKind::Secondary, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("disabled_75081b59",zh)).x + 20 * s);
        button("disabled", ecmgui_v2::ui::text("disabled_75081b59",zh), ButtonKind::Secondary, p, s, true);
        button("ghost", ecmgui_v2::ui::text("ghost_action_e10eaa58",zh), ButtonKind::Ghost, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("danger_action_7c4d0637",zh)).x + 20 * s);
        button("danger", ecmgui_v2::ui::text("danger_action_7c4d0637",zh), ButtonKind::Danger, p, s);
        ImGui::Spacing(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("interaction_states_specimen_dea5e57a",zh));
        state_strip(p, s, zh);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(ecmgui_v2::ui::text("form_selection_7eb99b4c",zh));
        icon(Icon::Folder, p, s); ImGui::SameLine(); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("preview_path_no_file_will_be_opened_or_sav_84370819",zh));
        form_row(ecmgui_v2::ui::text("workflow_path_d11ce1d6",zh), "##path", path, sizeof(path), s);
        field_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        ImGui::Checkbox(ecmgui_v2::ui::text("enable_sample_option_option_0a4f69ac",zh), &selected);
        ImGui::TextDisabled(ecmgui_v2::ui::text("primary_action_clicks_u_11904dd4",zh), clicks);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(ecmgui_v2::ui::text("status_vocabulary_ff0020b0",zh));
        status_badge(ecmgui_v2::ui::text("running_f4ccae29",zh), Status::Success, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("waiting_6e293a8c",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("waiting_6e293a8c",zh), Status::Warning, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("error_54a0e8c1",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("error_54a0e8c1",zh), Status::Danger, p, s);
        flow_next(ImGui::CalcTextSize(ecmgui_v2::ui::text("stopped_1a4f630a",zh)).x + 26 * s);
        status_badge(ecmgui_v2::ui::text("stopped_1a4f630a",zh), Status::Info, p, s);
        fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", ecmgui_v2::ui::text("sample_states_no_active_workers_27fa1752",zh)); ImGui::PopFont();
    }
    ImGui::Spacing();
    fonts.push(FontRole::Heading); heading_pixels = ImGui::GetFontSize(); ImGui::TextUnformatted(ecmgui_v2::ui::text("type_precision_8f686fc3",zh)); ImGui::PopFont();
    ImGui::TextUnformatted(ecmgui_v2::ui::text("body_segoe_ui_with_cjk_rescue_54f51b52",zh));
    fonts.push(FontRole::Caption); caption_pixels = ImGui::GetFontSize();
    ImGui::TextDisabled("%s", ecmgui_v2::ui::text("caption_13_logical_px_use_tab_to_inspect_k_05a5bddf",zh)); ImGui::PopFont();
    fonts.push(FontRole::Mono); mono_pixels = ImGui::GetFontSize();
    mono_i = ImGui::CalcTextSize("iiiiiiii").x; mono_w = ImGui::CalcTextSize("WWWWWWWW").x;
    ImGui::TextUnformatted("B1  1000000     B2  100000000\nM127  /  sigma=123456789");
    if (fonts.cjk) ImGui::TextUnformatted(ecmgui_v2::ui::text("cjk_fallback_task_log_d147f3c4",zh));
    ImGui::PopFont();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage1); ImGui::TextUnformatted("Stage 1"); ImGui::PopStyleColor(); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage2); ImGui::TextUnformatted("Stage 2"); ImGui::PopStyleColor();
    fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", gpu); ImGui::PopFont();
    ImGui::PopTextWrapPos(); ImGui::EndChild(); ImGui::PopFont(); ImGui::End();
}
}
