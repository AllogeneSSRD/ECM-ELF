// Generated; edit config/ecm_options.json, not this file.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace ecmgui_v2 {
struct UiValues {
    std::string ui_theme = "dark";
    std::string ui_language = "en";
    std::string ui_language_file = "";
    float ui_scale = 0.0f;
    int ui_font_size = 15;
    int ui_primary_font_offset = 1;
    int ui_secondary_font_offset = -2;
    int ui_worker_chart_columns = 1;
    int ui_gpu_history_seconds = 60;
    std::string ui_stage2_timing_source = "auto";
    std::string ui_stage2_tune_file = "";
    std::string ui_nav_order = "01237456";
    int ui_page = 0;
    int ui_worker = 1;
    bool ui_inspector = true;
    bool ui_logs = true;
    bool ui_follow_log = true;
    bool ui_raw_log = false;
    std::string ui_dock = "";
    int window_x = (-2147483647 - 1);
    int window_y = (-2147483647 - 1);
    int window_width = 1280;
    int window_height = 850;
    bool window_title_bar = false;
    bool window_maximized = false;
    bool pet_enabled = false;
    bool pet_animate = true;
    float pet_left = 72.0f;
    std::string pet_skin = "classic";
    int ui_right_pages = 0;
    int ui_right_page = 4;
    bool ui_right_visible = true;
    bool ui_smooth_scroll = true;
    bool ui_toasts = true;
    bool ui_external_windows = false;
    bool ui_worker_chart_auto_y = false;
    bool ui_logs_keep_visible = false;
    bool ui_nav_collapsed = false;
    int ui_nav_expanded_width = 0;
    bool ui_nav_hide_tab_bar = true;
    bool ui_workspace_hide_tab_bar = true;
    bool ui_right_hide_tab_bar = false;
};
namespace ui_limits {
inline constexpr float ui_scale_min = 0.0f;
inline constexpr float ui_scale_max = 4.0f;
inline constexpr int ui_font_size_min = 10;
inline constexpr int ui_font_size_max = 30;
inline constexpr int ui_primary_font_offset_min = 0;
inline constexpr int ui_primary_font_offset_max = 6;
inline constexpr int ui_secondary_font_offset_min = -6;
inline constexpr int ui_secondary_font_offset_max = 0;
inline constexpr int ui_page_min = 0;
inline constexpr int ui_page_max = 7;
inline constexpr int ui_worker_min = 1;
inline constexpr int ui_worker_max = 1024;
inline constexpr int window_x_min = (-2147483647 - 1);
inline constexpr int window_x_max = 2147483647;
inline constexpr int window_y_min = (-2147483647 - 1);
inline constexpr int window_y_max = 2147483647;
inline constexpr int window_width_min = 480;
inline constexpr int window_width_max = 16384;
inline constexpr int window_height_min = 360;
inline constexpr int window_height_max = 16384;
inline constexpr float pet_left_min = 0.0f;
inline constexpr float pet_left_max = 16384.0f;
inline constexpr int ui_right_pages_min = 0;
inline constexpr int ui_right_pages_max = 254;
inline constexpr int ui_right_page_min = 0;
inline constexpr int ui_right_page_max = 7;
inline constexpr int ui_nav_expanded_width_min = 0;
inline constexpr int ui_nav_expanded_width_max = 16384;
}
template<class F> void visit_ui_fields(F&& field) {
    field("theme",&UiValues::ui_theme);
    field("display_language",&UiValues::ui_language);
    field("language_file",&UiValues::ui_language_file);
    field("scale",&UiValues::ui_scale);
    field("body_font_size",&UiValues::ui_font_size);
    field("primary_font_offset",&UiValues::ui_primary_font_offset);
    field("secondary_font_offset",&UiValues::ui_secondary_font_offset);
    field("worker_chart_columns",&UiValues::ui_worker_chart_columns);
    field("gpu_history_seconds",&UiValues::ui_gpu_history_seconds);
    field("stage2_timing_source",&UiValues::ui_stage2_timing_source);
    field("stage2_timing_profile",&UiValues::ui_stage2_tune_file);
    field("nav_order",&UiValues::ui_nav_order);
    field("page",&UiValues::ui_page);
    field("selected_worker",&UiValues::ui_worker);
    field("inspector",&UiValues::ui_inspector);
    field("logs",&UiValues::ui_logs);
    field("follow_log",&UiValues::ui_follow_log);
    field("raw_log",&UiValues::ui_raw_log);
    field("workspace_layout",&UiValues::ui_dock);
    field("window_x",&UiValues::window_x);
    field("window_y",&UiValues::window_y);
    field("window_width",&UiValues::window_width);
    field("window_height",&UiValues::window_height);
    field("window_title_bar",&UiValues::window_title_bar);
    field("window_maximized",&UiValues::window_maximized);
    field("pet_enabled",&UiValues::pet_enabled);
    field("pet_animate",&UiValues::pet_animate);
    field("pet_left",&UiValues::pet_left);
    field("pet_skin",&UiValues::pet_skin);
    field("right_pages",&UiValues::ui_right_pages);
    field("right_page",&UiValues::ui_right_page);
    field("right_visible",&UiValues::ui_right_visible);
    field("smooth_scroll",&UiValues::ui_smooth_scroll);
    field("toasts",&UiValues::ui_toasts);
    field("external_windows",&UiValues::ui_external_windows);
    field("worker_chart_auto_y",&UiValues::ui_worker_chart_auto_y);
    field("logs_keep_visible",&UiValues::ui_logs_keep_visible);
    field("nav_collapsed",&UiValues::ui_nav_collapsed);
    field("nav_expanded_width",&UiValues::ui_nav_expanded_width);
    field("nav_hide_tab_bar",&UiValues::ui_nav_hide_tab_bar);
    field("workspace_hide_tab_bar",&UiValues::ui_workspace_hide_tab_bar);
    field("right_hide_tab_bar",&UiValues::ui_right_hide_tab_bar);
}
inline void validate_ui(const UiValues& v) {
    if ((v.ui_theme != "dark" && v.ui_theme != "light") || v.ui_theme.size() > 4096) throw std::runtime_error("Invalid GUI preference: theme");
    if ((v.ui_language != "en" && v.ui_language != "zh") || v.ui_language.size() > 4096) throw std::runtime_error("Invalid GUI preference: display_language");
    if (v.ui_language_file.size() > 255) throw std::runtime_error("Invalid GUI preference: language_file");
    if (v.ui_scale < 0 || v.ui_scale > 4 || !std::isfinite(v.ui_scale)) throw std::runtime_error("Invalid GUI preference: scale");
    if (v.ui_font_size < 10 || v.ui_font_size > 30) throw std::runtime_error("Invalid GUI preference: body_font_size");
    if (v.ui_primary_font_offset < 0 || v.ui_primary_font_offset > 6) throw std::runtime_error("Invalid GUI preference: primary_font_offset");
    if (v.ui_secondary_font_offset < -6 || v.ui_secondary_font_offset > 0) throw std::runtime_error("Invalid GUI preference: secondary_font_offset");
    if ((v.ui_worker_chart_columns != 1 && v.ui_worker_chart_columns != 2)) throw std::runtime_error("Invalid GUI preference: worker_chart_columns");
    if ((v.ui_gpu_history_seconds != 15 && v.ui_gpu_history_seconds != 30 && v.ui_gpu_history_seconds != 60 && v.ui_gpu_history_seconds != 120)) throw std::runtime_error("Invalid GUI preference: gpu_history_seconds");
    if ((v.ui_stage2_timing_source != "auto" && v.ui_stage2_timing_source != "previous" && v.ui_stage2_timing_source != "tune") || v.ui_stage2_timing_source.size() > 8) throw std::runtime_error("Invalid GUI preference: stage2_timing_source");
    if (v.ui_stage2_tune_file.size() > 4096) throw std::runtime_error("Invalid GUI preference: stage2_timing_profile");
    auto sorted_ui_nav_order = v.ui_nav_order; std::sort(sorted_ui_nav_order.begin(),sorted_ui_nav_order.end());
    if (sorted_ui_nav_order != "01234567" || v.ui_nav_order.size() > 8) throw std::runtime_error("Invalid GUI preference: nav_order");
    if (v.ui_page < 0 || v.ui_page > 7) throw std::runtime_error("Invalid GUI preference: page");
    if (v.ui_worker < 1 || v.ui_worker > 1024) throw std::runtime_error("Invalid GUI preference: selected_worker");
    if (v.ui_dock.size() > 524288) throw std::runtime_error("Invalid GUI preference: workspace_layout");
    if (v.window_x < (-2147483647 - 1) || v.window_x > 2147483647) throw std::runtime_error("Invalid GUI preference: window_x");
    if (v.window_y < (-2147483647 - 1) || v.window_y > 2147483647) throw std::runtime_error("Invalid GUI preference: window_y");
    if (v.window_width < 480 || v.window_width > 16384) throw std::runtime_error("Invalid GUI preference: window_width");
    if (v.window_height < 360 || v.window_height > 16384) throw std::runtime_error("Invalid GUI preference: window_height");
    if (v.pet_left < 0 || v.pet_left > 16384 || !std::isfinite(v.pet_left)) throw std::runtime_error("Invalid GUI preference: pet_left");
    if ((v.pet_skin != "classic" && v.pet_skin != "miss_pink_elf" && v.pet_skin != "herrscher_of_human_ego" && v.pet_skin != "hi_love_elf") || v.pet_skin.size() > 4096) throw std::runtime_error("Invalid GUI preference: pet_skin");
    if (v.ui_right_pages < 0 || v.ui_right_pages > 254) throw std::runtime_error("Invalid GUI preference: right_pages");
    if (v.ui_right_page < 0 || v.ui_right_page > 7) throw std::runtime_error("Invalid GUI preference: right_page");
    if (v.ui_nav_expanded_width < 0 || v.ui_nav_expanded_width > 16384) throw std::runtime_error("Invalid GUI preference: nav_expanded_width");
}
}
