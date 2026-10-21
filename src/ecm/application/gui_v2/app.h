#pragma once
#include "ui_state.h"
#include "ui/widgets.h"
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
    void draw_page(const ui::Settings& active, const ui::Fonts& fonts);
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
    std::string status, config_path;
    std::deque<std::string> messages;
    std::vector<ecmgui::WorkerNotice> notices;
    bool ready = false, no_nvml = false, reset_layout = false;
    unsigned long long last_results = 0;
    ecmgui::ResultFilter result_filter;
    std::set<std::string> selected_results;
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
    bool probability_empirical=false;
    ui::Showcase showcase;
    bool test_float = false;
    int pet_test_mood = -1;
    float probe_dpi = 0, probe_font = 0, probe_padding = 0;
private:
    FILE* log_;
    std::map<int,std::unique_ptr<ecm_queue::OwnerLease>> leases_;
    std::map<std::string,std::filesystem::file_time_type> verified_;
    bool layout_ready_ = false, layout_wide_ = false;
    std::map<std::string,float> window_dpi_;
    void begin_window(const char* name, bool* open, const ui::Settings& active);
    void end_window();
    void shutdown_workers() noexcept;
    bool supports_owner(const std::string& exe);
    void draw_workers(const ui::Settings& active, const ui::Fonts& fonts);
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
