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
bool button(const char* id, const char* label, ButtonKind kind, const Palette& p, float scale, bool disabled) {
    const bool primary = kind == ButtonKind::Primary;
    const auto normal = primary ? p.accent : kind == ButtonKind::Ghost ? ImVec4(0, 0, 0, 0) : p.surface;
    const auto text = primary ? p.on_accent : kind == ButtonKind::Danger ? p.danger : p.fg;
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    ImGui::PushStyleColor(ImGuiCol_Button, normal);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, primary ? p.accent_hover : p.hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, primary ? p.muted : p.active);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, kind == ButtonKind::Ghost ? 0.f : scale);
    ImGui::BeginDisabled(disabled);
    const std::string stable_label = std::string(label) + "###" + id;
    const auto clicked = ImGui::Button(stable_label.c_str(), {0, std::max(32 * scale, ImGui::GetFontSize() + 12 * scale)});
    ImGui::EndDisabled(); ImGui::PopStyleVar(); ImGui::PopStyleColor(4);
    return clicked;
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
    const char* en[] = {"Normal", "Hover", "Pressed", "Disabled", "Focus", "Selected"};
    const char* zh[] = {"常态", "悬停", "按下", "禁用", "焦点", "选中"};
    const ImVec4 fills[] = {p.surface, p.hover, p.active, p.surface, p.surface, p.selection};
    const float width = 88 * scale, height = 32 * scale;
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (int i = 0; i < 6; ++i) {
        if (i && ImGui::GetItemRectMax().x + 8 * scale + width < right) ImGui::SameLine();
        const auto pos = ImGui::GetCursorScreenPos(); auto draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(pos, {pos.x + width, pos.y + height}, ImGui::ColorConvertFloat4ToU32(fills[i]), 6 * scale);
        draw->AddRect(pos, {pos.x + width, pos.y + height}, ImGui::ColorConvertFloat4ToU32(i == 4 ? p.focus : p.border), 6 * scale, 0, (i == 4 ? 2 : 1) * scale);
        const auto label = chinese ? zh[i] : en[i]; const auto text = ImGui::CalcTextSize(label);
        draw->AddText({pos.x + (width - text.x) / 2, pos.y + (height - text.y) / 2}, ImGui::ColorConvertFloat4ToU32(i == 3 ? p.disabled : p.fg), label);
        ImGui::Dummy({width, height});
    }
}
void Showcase::draw(Settings& pending, const Settings& active, const Fonts& fonts, const char* gpu) {
    const auto p = palette(active.light); const float s = active.scale; const bool zh = fonts.chinese;
    const auto t = [zh](const char* en, const char* cn) { return zh ? cn : en; };
    const auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos); ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("ECM-ELF##v2_showcase", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    fonts.push(FontRole::Body); body_pixels = ImGui::GetFontSize();
    const bool wide = ImGui::GetContentRegionAvail().x > 760 * s;
    ImGui::BeginChild("navigation", {wide ? 174 * s : 40 * s, 0}, ImGuiChildFlags_None);
    fonts.push(FontRole::Heading); ImGui::TextUnformatted(wide ? "ECM-ELF" : "E"); ImGui::PopFont();
    if (wide) { fonts.push(FontRole::Caption); ImGui::TextDisabled("GUI v2 / %s", version); ImGui::PopFont(); }
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    icon(Icon::Grid, p, s); if (wide) { ImGui::SameLine(); ImGui::TextUnformatted(t("Design system", "设计系统")); }
    if (ImGui::Selectable(wide ? t("Components###components", "组件样板###components") : "01###components", page == 0, 0, {0, 32 * s})) page = 0;
    if (ImGui::Selectable(wide ? t("Typography###typography", "文字与数据###typography") : "02###typography", page == 1, 0, {0, 32 * s})) page = 1;
    ImGui::EndChild(); ImGui::SameLine();
    ImGui::BeginChild("content", {0, 0}, ImGuiChildFlags_None);
    ImGui::PushTextWrapPos(0);
    section_header(t("A quieter workspace", "清晰、专注的工作空间"),
        t("P2 component preview. Worker controls and business pages arrive in later phases.", "P2 组件预览。任务控制与业务页面将在后续阶段接入。"), fonts, p);
    if (!fonts.notice.empty()) ImGui::TextWrapped("%s", fonts.notice.c_str());
    // Preferences are applied at the next frame boundary, before any font is used.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * s, 4 * s});
    if (ImGui::BeginTable("preferences", wide ? 3 : 1, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", t("Appearance", "外观"));
        int theme = pending.light ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##theme", &theme, "Dark\0Light\0")) pending.light = theme != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", t("Language", "语言"));
        int language = pending.chinese ? 1 : 0; ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##language", &language, "English\0中文\0")) pending.chinese = language != 0;
        ImGui::TableNextColumn(); ImGui::TextDisabled("%s", t("Interface scale", "界面缩放"));
        const float scales[]{1, 1.25f, 1.5f, 2}; int index = 0;
        for (int i = 0; i < 4; ++i) if (std::abs(pending.scale - scales[i]) < .01f) index = i;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##scale", &index, "100%\0 125%\0 150%\0 200%\0")) pending.scale = scales[index];
        ImGui::EndTable();
    }
    ImGui::PopStyleVar(); ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (page == 0) {
        ImGui::TextUnformatted(t("Actions", "操作"));
        if (button("primary", t("Primary action", "主要操作"), ButtonKind::Primary, p, s)) ++clicks;
        primary_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        flow_next(ImGui::CalcTextSize(t("Secondary", "次要操作")).x + 20 * s);
        button("secondary", t("Secondary", "次要操作"), ButtonKind::Secondary, p, s);
        flow_next(ImGui::CalcTextSize(t("Disabled", "不可用")).x + 20 * s);
        button("disabled", t("Disabled", "不可用"), ButtonKind::Secondary, p, s, true);
        button("ghost", t("Ghost action", "轻量操作"), ButtonKind::Ghost, p, s);
        flow_next(ImGui::CalcTextSize(t("Danger action", "危险操作")).x + 20 * s);
        button("danger", t("Danger action", "危险操作"), ButtonKind::Danger, p, s);
        ImGui::Spacing(); ImGui::TextDisabled("%s", t("Interaction states (specimen)", "交互状态（样板）"));
        state_strip(p, s, zh);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(t("Form & selection", "表单与选择"));
        icon(Icon::Folder, p, s); ImGui::SameLine(); ImGui::TextDisabled("%s", t("Preview path; no file will be opened or saved", "预览路径；不会打开或保存文件"));
        form_row(t("Workflow path", "工作流路径"), "##path", path, sizeof(path), s);
        field_center = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) / 2};
        ImGui::Checkbox(t("Enable sample option###option", "启用样板选项###option"), &selected);
        ImGui::TextDisabled(t("Primary action clicks: %u", "主要操作点击次数：%u"), clicks);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(t("Status vocabulary", "状态语义"));
        status_badge(t("Running", "运行"), Status::Success, p, s);
        flow_next(ImGui::CalcTextSize(t("Waiting", "等待")).x + 26 * s);
        status_badge(t("Waiting", "等待"), Status::Warning, p, s);
        flow_next(ImGui::CalcTextSize(t("Error", "错误")).x + 26 * s);
        status_badge(t("Error", "错误"), Status::Danger, p, s);
        flow_next(ImGui::CalcTextSize(t("Stopped", "停止")).x + 26 * s);
        status_badge(t("Stopped", "停止"), Status::Info, p, s);
        fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", t("Sample states; no active workers.", "以上为状态样板，当前未连接 worker。")); ImGui::PopFont();
    }
    ImGui::Spacing();
    fonts.push(FontRole::Heading); heading_pixels = ImGui::GetFontSize(); ImGui::TextUnformatted(t("Type & precision", "文字与精确数据")); ImGui::PopFont();
    ImGui::TextUnformatted(t("Body / Segoe UI with CJK rescue", "正文 / 微软雅黑与中文救援"));
    fonts.push(FontRole::Caption); caption_pixels = ImGui::GetFontSize();
    ImGui::TextDisabled("%s", t("Caption / 13 logical px. Use Tab to inspect keyboard focus.", "辅助文字 / 13 逻辑像素。使用 Tab 查看键盘焦点。")); ImGui::PopFont();
    fonts.push(FontRole::Mono); mono_pixels = ImGui::GetFontSize();
    mono_i = ImGui::CalcTextSize("iiiiiiii").x; mono_w = ImGui::CalcTextSize("WWWWWWWW").x;
    ImGui::TextUnformatted("B1  1000000     B2  100000000\nM127  /  sigma=123456789");
    if (fonts.cjk) ImGui::TextUnformatted("中文救援 / 任务日志");
    ImGui::PopFont();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage1); ImGui::TextUnformatted("Stage 1"); ImGui::PopStyleColor(); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, p.stage2); ImGui::TextUnformatted("Stage 2"); ImGui::PopStyleColor();
    fonts.push(FontRole::Caption); ImGui::TextDisabled("%s", gpu); ImGui::PopFont();
    ImGui::PopTextWrapPos(); ImGui::EndChild(); ImGui::PopFont(); ImGui::End();
}
}
