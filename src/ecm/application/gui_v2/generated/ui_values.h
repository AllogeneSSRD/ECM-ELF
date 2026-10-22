// Generated; edit config/ecm_options.json, not this file.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
namespace ecmgui_v2 {
struct UiValues {
    int state_schema = 1;
    std::string ui_theme = "dark";
    std::string ui_language = "en";
    float ui_scale = 0.0f;
    int ui_page = 0;
    int ui_worker = 1;
    bool ui_inspector = true;
    bool ui_logs = true;
    bool ui_follow_log = true;
    bool ui_raw_log = false;
    std::string ui_dock = "";
    int ui_right_pages = 0;
    int ui_right_page = 4;
    bool ui_right_visible = true;
    bool ui_smooth_scroll = true;
    bool ui_toasts = true;
    int window_x = (-2147483647 - 1);
    int window_y = (-2147483647 - 1);
    int window_width = 1280;
    int window_height = 850;
    bool window_maximized = false;
    std::string worker_stage1_exe = "";
    std::string worker_stage2_exe = "";
    int worker_count = 0;
    bool pet_enabled = false;
    bool pet_animate = true;
    float pet_left = 72.0f;
};
template<class V,class F> void visit_ui_fields(V& value,F&& field) {
    field("state","schema",value.state_schema);
    field("ui","theme",value.ui_theme);
    field("ui","language",value.ui_language);
    field("ui","scale",value.ui_scale);
    field("ui","page",value.ui_page);
    field("ui","worker",value.ui_worker);
    field("ui","inspector",value.ui_inspector);
    field("ui","logs",value.ui_logs);
    field("ui","follow_log",value.ui_follow_log);
    field("ui","raw_log",value.ui_raw_log);
    field("ui","dock",value.ui_dock);
    field("ui","right_pages",value.ui_right_pages);
    field("ui","right_page",value.ui_right_page);
    field("ui","right_visible",value.ui_right_visible);
    field("ui","smooth_scroll",value.ui_smooth_scroll);
    field("ui","toasts",value.ui_toasts);
    field("window","x",value.window_x);
    field("window","y",value.window_y);
    field("window","width",value.window_width);
    field("window","height",value.window_height);
    field("window","maximized",value.window_maximized);
    field("worker","stage1_exe",value.worker_stage1_exe);
    field("worker","stage2_exe",value.worker_stage2_exe);
    field("worker","count",value.worker_count);
    field("pet","enabled",value.pet_enabled);
    field("pet","animate",value.pet_animate);
    field("pet","left",value.pet_left);
}
inline void validate_ui(const UiValues& v) {
    if (v.state_schema < 1 || v.state_schema > 1) throw std::runtime_error("Invalid UI state: state.schema");
    if ((v.ui_theme != "dark" && v.ui_theme != "light") || v.ui_theme.size() > 4096) throw std::runtime_error("Invalid UI state: ui.theme");
    if ((v.ui_language != "en" && v.ui_language != "zh") || v.ui_language.size() > 4096) throw std::runtime_error("Invalid UI state: ui.language");
    if (v.ui_scale < 0 || v.ui_scale > 4 || !std::isfinite(v.ui_scale)) throw std::runtime_error("Invalid UI state: ui.scale");
    if (v.ui_page < 0 || v.ui_page > 6) throw std::runtime_error("Invalid UI state: ui.page");
    if (v.ui_worker < 1 || v.ui_worker > 1024) throw std::runtime_error("Invalid UI state: ui.worker");
    if (v.ui_dock.size() > 524288) throw std::runtime_error("Invalid UI state: ui.dock");
    if (v.ui_right_pages < 0 || v.ui_right_pages > 126) throw std::runtime_error("Invalid UI state: ui.right_pages");
    if (v.ui_right_page < 0 || v.ui_right_page > 6) throw std::runtime_error("Invalid UI state: ui.right_page");
    if (v.window_x < (-2147483647 - 1) || v.window_x > 2147483647) throw std::runtime_error("Invalid UI state: window.x");
    if (v.window_y < (-2147483647 - 1) || v.window_y > 2147483647) throw std::runtime_error("Invalid UI state: window.y");
    if (v.window_width < 480 || v.window_width > 16384) throw std::runtime_error("Invalid UI state: window.width");
    if (v.window_height < 360 || v.window_height > 16384) throw std::runtime_error("Invalid UI state: window.height");
    if (v.worker_stage1_exe.size() > 4096) throw std::runtime_error("Invalid UI state: worker.stage1_exe");
    if (v.worker_stage2_exe.size() > 4096) throw std::runtime_error("Invalid UI state: worker.stage2_exe");
    if (v.worker_count < 0 || v.worker_count > 64) throw std::runtime_error("Invalid UI state: worker.count");
    if (v.pet_left < 0 || v.pet_left > 16384 || !std::isfinite(v.pet_left)) throw std::runtime_error("Invalid UI state: pet.left");
}
}
