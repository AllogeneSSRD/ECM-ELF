#pragma once
#include "ui_state.h"
#include "gpu_details.h"
#include "ui/widgets.h"
#include "ui/chart.h"
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
    bool save_preferences();
    const ecmgui::WorkerView* selected() const;
    void trace(const std::string& text);
    void toast(const std::string& text,bool error=false);
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
    ImVec2 gpu_range_controls[4]{};
    struct GpuPresentation {
        ImVec2 header{};
        size_t charts=0;
        bool hovered=false;
        std::string tooltip,basic,usage,power,clocks;
        ui::ChartMetrics chart_metrics[3];
        float height=0;
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
    ImVec2 nav_items[7]{},right_items[7]{},nav_drop{},nav_return{},language_items[2]{},save_ui_control{},calculate_control{};
    int nav_drag_page=-1;
    float nav_rounding=0,nav_item_height=0,nav_item_gap=0;
    std::map<int,ImVec2> right_close_controls;
    std::map<int,unsigned> right_docks;
    bool right_container_present=false;
    ImVec2 settings_search_control{};
    ImVec2 font_size_control{},exit_controls[3]{},worker_speed_min{},worker_speed_max{};
    ImVec2 primary_offset_control{},secondary_offset_control{};
    float primary_font_height=0,secondary_button_font_height=0,secondary_text_font_height=0;
    size_t worker_speed_samples=0,workspace_chart_count=0;
    ImVec2 pet_skin_controls[4]{};
    std::map<int,ImVec2> worker_select_controls,worker_action_controls,log_tab_controls;
    ImVec2 log_raw_control{},log_follow_control{},title_drag_min{},title_drag_max{},title_controls[3]{};
    int log_active_worker=-1;
    bool log_current_raw=false,log_current_follow=true;
    float log_scroll=0;
    std::vector<std::pair<ImVec2,ImVec2>> worker_summary_rects;
    float log_line_pitch=0,log_font_height=0,log_view_height=0;
    float worker_table_scroll_max_y=0,worker_table_height=0,worker_table_content_height=0;
    float worker_table_required_height=0,worker_table_data_height=0;
    float worker_toolbar_height=0,worker_row_height=0,worker_button_height=0;
    float worker_table_scroll_max_x=0,worker_field_overflow=0;
    bool worker_flexible=false;
    size_t worker_table_rows=0;
    float body_font_height=0,body_glyph_height=0,log_glyph_height=0;
    bool log_uses_body=false;
    size_t log_speed_samples=0;
    int settings_matches=0;
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
    struct LogView {bool raw=false,follow=true,jump=true;};
    std::map<int,LogView> log_views_;
    int log_last_worker_=-1;
    int drawn_center_page_=-1;
    std::map<std::string,float> window_dpi_;
    bool begin_window(const char* name, bool* open, const ui::Settings& active);
    void end_window();
    void shutdown_workers() noexcept;
    bool supports_owner(const std::string& exe);
    void draw_workers(const ui::Settings& active, const ui::Fonts& fonts);
    void draw_worker_speed(const ui::Fonts& fonts);
    void draw_exit(const ui::Settings& active,const ui::Fonts& fonts);
    void draw_results(bool zh);
    void draw_generator(bool zh);
    void draw_probability(bool zh);
    void draw_gpu(bool zh);
    void draw_settings(bool zh);
    void draw_logs(const ui::Fonts& fonts);
    void draw_inspector(const ui::Fonts& fonts);
    void build_layout(unsigned dock, float scale);
};
}
