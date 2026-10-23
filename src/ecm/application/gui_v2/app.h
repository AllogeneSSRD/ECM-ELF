#pragma once
#include "ui_state.h"
#include "group_order_job.h"
#include "gpu_details.h"
#include "gpu_clocks.h"
#include "gpu_clock_automation.h"
#include "ui/widgets.h"
#include "ui/chart.h"
#include "ui/compact_form.h"
#include "ui/stage2_progress.h"
#include "ui/range_slider.h"
#include "../gui/worker_controller.h"
#include "../gui/toml_file.h"
#include "../gui/results.h"
#include "../gui/worktodo_gen.h"
#include "../gui/probability_model.h"
#include "../gui/gpu_monitor.h"
#include "../../storage/queue/owner_lease.h"
#include <map>
#include <memory>
#include <deque>
#include <set>

namespace ecmgui_v2 {
class App {
public:
    explicit App(UiState& state, FILE* log);
    ~App();
    void initialize(const std::string& config, bool no_nvml);
    bool load_config(const std::string& config);
    void configure_workers();
    bool start(int worker);
    void tick();
    bool request_close();
    void begin_exit();
    bool done() const { return controller.quit_requested(); }
    void draw(ui::Settings& pending, const ui::Settings& active, const ui::Fonts& fonts);
    void draw_page(const ui::Settings& active, const ui::Fonts& fonts, int page=-1);
    bool move_page(int page,bool right);
    void open_settings(int category=-1);
    bool save_preferences(const char* success_key="message.ui_saved");
    bool save_layout_config();
    void set_logs_visible(bool visible);
    const ecmgui::WorkerView* selected() const;
    void trace(const std::string& text);
    void toast(const std::string& text,bool error=false,bool repeat=false);
    struct Toast {std::string text; unsigned long long until; bool error;};
    std::deque<Toast> toasts;
    std::string last_status;
    int running_count() const;
    UiState& state;
    ecmgui::TomlFile config;
    ecmgui::WorkerController controller;
    ecmgui::ResultStore results;
    ecmgui::GpuMonitor gpu;
    GpuDetailsCache gpu_details;
    GpuClocks gpu_clocks;
    struct ClockDraft {ClockRange range;bool initialized=false,automatic=false;std::string message_key;};
    ClockProfiles saved_clock_profiles;
    std::set<std::string> saved_clock_automatic;
    std::map<std::string,ClockAutoPolicy> clock_auto_policies;
    struct ClockPending {ClockRange range;bool automatic=false;ClockAutoReason reason=ClockAutoReason::None;};
    std::map<int,ClockPending> clock_pending;
    std::map<std::string,ClockDraft> clock_drafts;
    std::map<int,unsigned> clock_completions;
    bool clock_profiles_valid=true,clock_policies_seeded=false;
    bool save_gpu_clock_range(const std::string& uuid,ClockRange range,bool automatic=false);
    void tick_gpu_clocks();
    void draw_gpu_clock_control(int index,const std::string& uuid,bool zh);
    ImVec2 gpu_range_controls[4]{};
    struct GpuPresentation {
        ImVec2 header{};
        size_t charts=0;
        bool hovered=false;
        std::string tooltip,basic,usage,power,clocks;
        ui::ChartMetrics chart_metrics[3];
        float height=0;
        ImVec2 gear{},popover_min{},popover_max{},apply{},reset{},save{},refresh{},input_low{},input_high{},automatic{};
        ui::RangeSliderAudit range_slider;
        bool popover=false;
        ClockRange supported,draft;
    };
    std::map<int,GpuPresentation> gpu_presentations;
    std::string status, config_path;
    std::deque<std::string> messages;
    std::vector<ecmgui::WorkerNotice> notices;
    bool ready = false, no_nvml = false, reset_layout = false;
    unsigned long long last_results = 0;
    ecmgui::ResultFilter result_filter;
    std::set<std::string> selected_results;
    std::vector<size_t> result_indices;
    std::set<std::string> visible_result_ids;
    std::string result_view_signature;
    bool result_view_dirty=true;
    ImVec2 result_chips[3]{},result_wheel{},start_control{},stop_control{};
    ImVec2 nav_items[8]{},right_items[8]{},nav_drop{},nav_return{},language_items[2]{},save_ui_control{},calculate_control{};
    ImVec2 language_control{};
    ImVec2 view_control{},view_logs_control{},view_save_control{};
    bool view_menu_open=false,view_save_separator=false;
    std::vector<std::string> view_item_keys;
    unsigned layout_build_count=0,view_save_count=0;
    ui::CompactFormAudit generator_form,probability_form;
    std::map<int,std::pair<ImVec2,ImVec2>> generator_curve_controls,probability_curve_controls;
    std::map<std::string,ImVec2> language_file_controls;
    int nav_drag_page=-1;
    float nav_rounding=0,nav_item_height=0,nav_item_gap=0;
    float nav_padding=0;
    ImVec2 nav_scroll{},nav_scrollbars{},nav_size{};
    std::map<int,ImVec2> right_close_controls;
    std::map<int,unsigned> right_docks;
    std::map<int,unsigned> log_docks;
    bool log_nested_tabs=false;
    float log_stage2_fraction=0,log_stage2_width=0;
    int log_stage2_phase=-1;
    std::string log_stage2_sources;
    bool right_container_present=false;
    ImVec2 settings_search_control{};
    ImVec2 font_size_control{},exit_controls[3]{},worker_speed_min{},worker_speed_max{};
    ImVec2 primary_offset_control{},secondary_offset_control{};
    float primary_font_height=0,secondary_button_font_height=0,secondary_text_font_height=0;
    size_t worker_speed_samples=0,workspace_chart_count=0;
    ImVec2 start_button_size{},stop_button_size{},worker_chart_toggle{};
    int worker_chart_columns=1;
    std::vector<std::pair<ImVec2,ImVec2>> worker_chart_rects;
    std::vector<ui::ChartMetrics> worker_chart_metrics;
    std::map<int,ImVec2> worker_number_controls;
    ImVec2 pet_skin_controls[4]{};
    std::map<int,ImVec2> worker_select_controls,worker_action_controls,log_tab_controls;
    ImVec2 log_raw_control{},log_follow_control{},title_drag_min{},title_drag_max{},title_controls[3]{};
    int log_active_worker=-1;
    bool log_current_raw=false,log_current_follow=true;
    float log_scroll=0;
    std::vector<std::pair<ImVec2,ImVec2>> worker_summary_rects;
    std::map<int,std::string> worker_summary_texts;
    float log_line_pitch=0,log_font_height=0,log_view_height=0;
    float worker_table_scroll_max_y=0,worker_table_height=0,worker_table_content_height=0;
    float worker_table_required_height=0,worker_table_data_height=0;
    float worker_toolbar_height=0,worker_row_height=0,worker_button_height=0;
    float worker_table_scroll_max_x=0,worker_field_overflow=0;
    unsigned worker_table_columns=0;
    size_t worker_table_rows=0;
    float body_font_height=0,body_glyph_height=0,log_glyph_height=0;
    bool log_uses_body=false;
    size_t log_speed_samples=0;
    int settings_matches=0;
    bool settings_open=false,settings_maximized=false;
    int settings_category=0;
    ImVec2 settings_entry{},settings_close{},settings_expand{},settings_categories[6]{},theme_controls[2]{};
    ImVec2 settings_min{},settings_max{};
    ImVec2 settings_content_min{},settings_content_max{};
    bool settings_sidebar_stacked=false;
    int navigation_icon_count=0;
    float result_scroll=0;
    char result_search[256]{}, gen_input[65536]{}, config_input[4096]{}, stage1_input[4096]{}, stage2_input[4096]{};
    ecmgui::GenOptions gen_options;
    ecmgui::GpuProfile gen_gpu;
    ecmgui::GenResult gen_preview;
    ecmgui::FileStamp gen_stamp;
    std::string gen_path, gen_signature;
    bool gen_valid = false;
    int gen_worker = 1;
    char gen_b1[64]{}, gen_b2[64]{}, gen_pattern[512] = "{curve}_{target}_{b1}.save";
    char edit_table[128] = "stage1", edit_key[128]{}, edit_value[4096]{};
    char settings_search[256]{};
    int settings_group=0;
    bool settings_bool=false;
    ecmgui::FileStamp config_stamp;
    bool autostart_pending = false;
    bool preview_generator();
    bool apply_generator();
    bool edit_business(const std::string& table, const std::string& key, const std::string& value);
    bool save_business();
    std::string generator_signature() const;
    ecmgui::probability_ui::Request probability_request;
    ecmgui::probability_ui::Result probability_result;
    bool probability_ready = false;
    int probability_profile=0;
    int probability_tab=0;
    bool probability_empirical=false;
    OrderInput group_order_input;
    GroupOrderJob group_order_job;
    std::vector<std::string> group_order_files;
    std::vector<OrderRecord> group_order_records;
    int group_order_file=-1;
    std::string group_order_file_error,group_order_base;
    bool group_order_loaded=false;
    ui::CompactFormAudit group_order_form;
    std::map<int,std::pair<ImVec2,ImVec2>> group_order_curve_controls;
    std::map<std::string,ImVec2> group_order_controls;
    ImVec2 group_order_calculate{},group_order_save{},group_order_cancel{};
    unsigned group_order_completions=0;
    bool group_order_success=false;
    std::string group_order_copy;
    void notify_group_order();
    void draw_group_order(bool zh);
    bool save_group_order_gp();
    ui::Showcase showcase;
    bool test_float = false;
    bool workspace_docked=false;
    int pet_test_mood = -1;
    float probe_dpi = 0, probe_font = 0, probe_padding = 0;
private:
    FILE* log_;
    std::map<int,std::unique_ptr<ecm_queue::OwnerLease>> leases_;
    std::map<std::string,std::filesystem::file_time_type> verified_;
    bool layout_ready_ = false, layout_wide_ = false;
    bool focus_right_ = true;
    bool focus_settings_=false,place_settings_=true;
    ImVec2 settings_restore_pos_{},settings_restore_size_{};
    float settings_geometry_scale_=0;
    struct LogView {bool raw=false,follow=true,jump=true;};
    std::map<int,LogView> log_views_;
    std::map<int,unsigned long long> worker_b2_ratios_;
    ui::Stage2TimingHistory stage2_history_;
    std::map<int,ui::AsyncStage2Tune> stage2_tunes_;
    std::map<int,std::string> stage2_engine_ids_;
    unsigned long long stage2_tune_poll_=0;
    struct PhaseView {std::string identity,phase,clock_curve;unsigned long long phase_at=0,batch_at=0,batch=0;ui::Stage2Timing timing;float high_water=0;};
    std::map<int,PhaseView> stage2_views_;
    unsigned log_dock_=0;
    std::map<std::string,unsigned> hidden_log_docks_;
    float log_height_ratio_=.28f;
    int log_last_worker_=-1;
    int drawn_center_page_=-1;
    std::map<std::string,float> window_dpi_;
    bool begin_window(const char* name, bool* open, const ui::Settings& active,ImGuiWindowFlags flags=0);
    void end_window();
    void shutdown_workers() noexcept;
    bool supports_owner(const std::string& exe);
    void draw_workers(const ui::Settings& active, const ui::Fonts& fonts);
    void draw_worker_speed(const ui::Fonts& fonts);
    std::string worker_summary(const ecmgui::WorkerView& worker,bool zh)const;
    bool select_language_file(const std::string& file);
    void reload_language_file();
    void draw_exit(const ui::Settings& active,const ui::Fonts& fonts);
    void draw_results(bool zh);
    void draw_generator(bool zh);
    void draw_probability(bool zh);
    void draw_gpu(bool zh);
    void draw_settings(bool zh);
    void draw_settings_window(const ui::Settings& active,const ui::Fonts& fonts);
    void draw_log_windows(const ui::Settings& active,const ui::Fonts& fonts);
    void draw_logs(const ui::Fonts& fonts,const ecmgui::WorkerView* worker);
    void draw_stage2_progress(const ecmgui::WorkerView& worker,const ui::Palette& palette,bool zh);
    void draw_inspector(const ui::Fonts& fonts);
    void build_layout(unsigned dock, float scale);
};
}
