#include "app.h"
#include "version.h"
#include "imgui_internal.h" // DockBuilder is kept in the v2 layout module.
#include <algorithm>
#include <cstdio>
#include <thread>

namespace ecmgui_v2 {
namespace {
std::string executable_path(const std::string& input, const std::string& base) {
    if (input.empty()) return {};
    auto path = std::filesystem::u8path(input);
    if (!path.is_absolute()) {
        auto candidate = std::filesystem::u8path(base) / path;
        path = std::filesystem::exists(candidate) ? candidate : std::filesystem::u8path(ecmgui::exe_dir()) / path;
    }
    return std::filesystem::weakly_canonical(path).u8string();
}
}
App::App(UiState& value, FILE* log) : state(value), controller(config), log_(log) {
    controller.set_sinks([this](const std::string& text) { trace(text); }, [this](const ecmgui::WorkerNotice& notice) {
        if (notices.size() == 32) notices.erase(notices.begin()); notices.push_back(notice);
        if (!notice.text.empty()) status = notice.text;
    });
}
App::~App() { shutdown_workers(); }
void App::trace(const std::string& text) {
    if (messages.size() == 4000) messages.pop_front(); messages.push_back(text);
    std::fprintf(log_, "app: %s\n", text.c_str()); std::fflush(log_);
}
void App::initialize(const std::string& path, bool disable_nvml) {
    no_nvml = disable_nvml;
    if (!path.empty()) load_config(path); else status = "Open a workflow TOML to manage workers.";
    if (!no_nvml && !gpu.start(500)) trace("GPU monitor unavailable: " + gpu.reason());
}
bool App::load_config(const std::string& input) {
    if (controller.busy()) { status = "Stop all workers before changing workflow."; return false; }
    try {
        const auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(std::filesystem::u8path(input))).u8string();
        std::error_code same_error;
        if (!state.path.empty() && (std::filesystem::equivalent(std::filesystem::u8path(path), state.path, same_error) ||
            _wcsicmp(std::filesystem::weakly_canonical(std::filesystem::u8path(path)).c_str(), std::filesystem::weakly_canonical(state.path).c_str()) == 0))
            throw std::runtime_error("Business TOML and UI sidecar must be distinct");
        std::string error; ecmgui::TomlFile candidate;
        if (!candidate.load(path, error)) throw std::runtime_error(error);
        config = std::move(candidate); config_path = path; leases_.clear(); verified_.clear();
        std::snprintf(config_input, sizeof(config_input), "%s", path.c_str());
        configure_workers(); ready = true; gen_valid = false; selected_results.clear();
        results.set_paths(ecmgui::result_paths(config.lines(), config_path, static_cast<int>(controller.workers().size())));
        status = "Workflow opened; worker control requires queue-owner protocol 1."; trace("config=" + path);
        return true;
    } catch (const std::exception& error) { status = error.what(); trace(status); return false; }
}
void App::configure_workers() {
    auto settings = ecm_config::read_gui(ecm_config::section_entries(config.lines(), "gui"));
    ecmgui::WorkerSettings workers;
    workers.ini_path = config_path; workers.num_workers = std::clamp(settings.NumWorkers, 1, 1024);
    const auto base = ecmgui::path_dir(config_path);
    auto first = state.worker_stage1_exe.empty() ? settings.exe : state.worker_stage1_exe;
    auto second = state.worker_stage2_exe.empty() ? settings.stage2_exe : state.worker_stage2_exe;
    if (first.empty()) first = "ecm_cuda.exe";
    if (second.empty()) second = "ecm_cuda_stage2.exe";
    workers.worker_exe = executable_path(first, base); workers.stage2_exe = executable_path(second, base);
    workers.priority = settings.priority; workers.exit_confirm = settings.exit_confirm;
    workers.graceful_stop_ms = static_cast<unsigned long long>(settings.graceful_stop_ms);
    controller.configure(workers); controller.suppress_autostart();
    std::snprintf(stage1_input, sizeof(stage1_input), "%s", workers.worker_exe.c_str());
    std::snprintf(stage2_input, sizeof(stage2_input), "%s", workers.stage2_exe.c_str());
}
bool App::supports_owner(const std::string& path) {
    if (!std::filesystem::is_regular_file(std::filesystem::u8path(path))) { status = "Worker executable missing: " + path; return false; }
    const auto stamp = std::filesystem::last_write_time(std::filesystem::u8path(path));
    const auto found = verified_.find(path); if (found != verified_.end() && found->second == stamp) return true;
    std::string output; int code = -1;
    if (!ecmgui::run_capture('"' + path + "\" --gui-queue-protocol", output, code, 3000) || code != 0 || ecm_config::trim(output) != ecm_config::trim(ecm_queue::owner_protocol)) {
        status = "Worker lacks queue ownership protocol 1; install the updated worker: " + path; trace(status); return false;
    }
    verified_[path] = stamp; return true;
}
bool App::start(int index) {
    if (!ready || controller.exit_phase() != ecmgui::WorkerController::ExitPhase::Idle) return false;
    try {
        for (const auto& worker : controller.workers()) if (worker.index == index) {
            if (worker.running()) return true;
            if (!supports_owner(controller.resolve_worker_exe(index))) return false;
            // Stage transitions run inside the unchanged Controller; validate both present engines.
            for (const auto& exe : {controller.resolve_stage1_exe(), controller.resolve_stage2_exe()})
                if (std::filesystem::exists(std::filesystem::u8path(exe)) && !supports_owner(exe)) return false;
            auto lease = std::make_unique<ecm_queue::OwnerLease>(worker.worktodo, index, ".gui-control.lock");
            { ecm_queue::OwnerLease probe(worker.worktodo, index); }
            if (!controller.start(index, true)) return false;
            leases_[index] = std::move(lease); return true;
        }
    } catch (const std::exception& error) { status = error.what(); trace(status); }
    return false;
}
void App::tick() {
    controller.tick();
    for (const auto& worker : controller.workers())
        if (!worker.running() && worker.state() != ecmgui::WorkerRunState::Restarting && worker.state() != ecmgui::WorkerRunState::Starting) leases_.erase(worker.index);
    const auto now = GetTickCount64();
    if (state.ui_page == 1 && now - last_results >= 500) {
        last_results = now; if (results.refresh()) {
            std::set<std::string> present; for (const auto& row : results.rows()) present.insert(row.id);
            for (auto it = selected_results.begin(); it != selected_results.end();) if (!present.count(*it)) it = selected_results.erase(it); else ++it;
        }
    }
}
int App::running_count() const { int count = 0; for (const auto& worker : controller.workers()) count += worker.running(); return count; }
const ecmgui::WorkerView* App::selected() const {
    for (const auto& worker : controller.workers()) if (worker.index == state.ui_worker) return &worker;
    return controller.workers().empty() ? nullptr : &controller.workers().front();
}
bool App::request_close() { return controller.request_close(); }
void App::begin_exit() { controller.begin_graceful_exit(); }
void App::shutdown_workers() noexcept {
    try {
        if (controller.busy()) {
            controller.begin_graceful_exit(); const auto end = controller.exit_deadline() + 1000;
            while (controller.busy() && GetTickCount64() < end) { controller.tick(); Sleep(20); }
        }
        controller.force_close(); leases_.clear();
    } catch (...) { controller.force_close(); }
}
void App::build_layout(unsigned dock, float scale) {
    const auto viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(dock); ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock, viewport->WorkSize);
    ImGuiID center = dock, nav = 0, inspector = 0, logs = 0;
    const bool wide = viewport->WorkSize.x >= 1100 * scale;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, wide ? .16f : .11f, &nav, &center);
    if (wide && state.ui_inspector) ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, .29f, &inspector, &center);
    if (state.ui_logs) ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, .28f, &logs, &center);
    ImGui::DockBuilderDockWindow("Navigation###v2.navigation", nav);
    ImGui::DockBuilderDockWindow("Workspace###v2.workspace", center);
    if (inspector) ImGui::DockBuilderDockWindow("Inspector###v2.inspector", inspector);
    if (logs) ImGui::DockBuilderDockWindow("Logs###v2.logs", logs);
    ImGui::DockBuilderFinish(dock);
}
void App::begin_window(const char* name, bool* open, const ui::Settings& active) {
    auto found = window_dpi_.find(name);
    const float dpi = state.ui_scale > 0 || found == window_dpi_.end() ? active.scale : found->second;
    const float ratio = dpi / active.scale;
    const auto& style = ImGui::GetStyle();
    const auto scaled = [ratio](ImVec2 v) { return ImVec2(v.x*ratio,v.y*ratio); };
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,scaled(style.WindowPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,scaled(style.FramePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,scaled(style.ItemSpacing));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,scaled(style.CellPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,style.FrameRounding*ratio);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize,style.ScrollbarSize*ratio);
    ImGui::Begin(name,open); window_dpi_[name] = ImGui::GetWindowViewport()->DpiScale;
}
void App::end_window() { ImGui::End(); ImGui::PopStyleVar(6); }
void App::draw(ui::Settings& pending, const ui::Settings& active, const ui::Fonts& fonts) {
    const bool zh = fonts.chinese; const auto t = [zh](const char* en, const char* cn) { return zh ? cn : en; };
    if (ImGui::BeginMainMenuBar()) {
        ImGui::Text("ECM-ELF %s", version); ImGui::Separator();
        if (ImGui::MenuItem(t("Reset layout", "重置布局"))) reset_layout = true;
        ImGui::Checkbox(t("Inspector", "检查区"), &state.ui_inspector); ImGui::Checkbox(t("Logs", "日志"), &state.ui_logs);
        if (ImGui::MenuItem(t("Notices", "通知"))) ImGui::OpenPopup("Notices###v2.notices");
        if (ImGui::BeginPopup("Notices###v2.notices")) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize()*34);
            const auto begin = messages.size() > 20 ? messages.size()-20 : 0;
            for (size_t i=begin;i<messages.size();++i) ImGui::TextUnformatted(messages[i].c_str());
            ImGui::PopTextWrapPos(); ImGui::EndPopup();
        }
        ImGui::TextDisabled("%s", status.c_str()); ImGui::EndMainMenuBar();
    }
    const auto dock = ImGui::DockSpaceOverViewport(ImGui::GetID("v2.dock"), ImGui::GetMainViewport());
    const bool wide = ImGui::GetMainViewport()->WorkSize.x >= 1100 * active.scale;
    const auto node = ImGui::DockBuilderGetNode(dock);
    if (reset_layout || (!layout_ready_ && (!node || !node->ChildNodes[0])) || (layout_ready_ && layout_wide_ != wide)) {
        build_layout(dock, active.scale); reset_layout = false;
    }
    layout_ready_ = true; layout_wide_ = wide;
    begin_window("Navigation###v2.navigation",nullptr,active); fonts.push(ui::FontRole::Body);
    const char* en[]{"Workers", "Results", "Generator", "Probability", "GPU", "Settings", "Components"};
    const char* cn[]{"任务", "结果", "生成器", "概率", "GPU", "设置", "组件"};
    const bool compact = ImGui::GetContentRegionAvail().x < 110 * active.scale;
    for (int i = 0; i < 7; ++i) {
        ImGui::PushID(i); const auto label = (compact ? std::to_string(i+1) : std::string(zh ? cn[i] : en[i])) + "###page";
        if (ImGui::Selectable(label.c_str(), state.ui_page == i)) state.ui_page = i;
        if (compact && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", zh ? cn[i] : en[i]); ImGui::PopID();
    }
    ImGui::PopFont(); end_window();
    begin_window("Workspace###v2.workspace",nullptr,active); fonts.push(ui::FontRole::Body);
    ImGui::PushTextWrapPos(0); draw_page(active, fonts); ImGui::PopTextWrapPos(); ImGui::PopFont(); end_window();
    if (state.ui_inspector && wide) { begin_window("Inspector###v2.inspector", &state.ui_inspector,active); draw_inspector(fonts); end_window(); }
    if (state.ui_logs) { begin_window("Logs###v2.logs", &state.ui_logs,active); draw_logs(fonts); end_window(); }
    if (test_float) {
        ImGui::SetNextWindowPos({ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x + 30, ImGui::GetMainViewport()->Pos.y + 30}, ImGuiCond_FirstUseEver);
        begin_window("Detached lifecycle probe###v2.float", &test_float,active);
        probe_dpi = ImGui::GetWindowViewport()->DpiScale; probe_font = ImGui::GetFontSize(); probe_padding = ImGui::GetStyle().FramePadding.y;
        ImGui::TextUnformatted("Vulkan platform viewport"); end_window();
    }
    if (controller.exit_phase() == ecmgui::WorkerController::ExitPhase::Confirm) ImGui::OpenPopup("Exit###v2.exit");
    if (ImGui::BeginPopupModal("Exit###v2.exit", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(t("Stop workers safely before closing?", "安全停止 worker 后退出？"));
        if (ImGui::Button(t("Stop and exit", "停止并退出"))) { begin_exit(); ImGui::CloseCurrentPopup(); }
        ImGui::SameLine(); if (ImGui::Button(t("Cancel", "取消"))) { controller.cancel_close(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    if (state.ui_page == 5) {
        // Preferences are applied by the host at the next frame boundary.
        pending.light = state.ui_theme == "light"; pending.chinese = state.ui_language == "zh";
        if (state.ui_scale > 0) pending.scale = state.ui_scale;
    }
}
void App::draw_workers(const ui::Settings& active, const ui::Fonts& fonts) {
    const auto p = ui::palette(active.light);
    ui::section_header(fonts.chinese ? "计算任务" : "Workers", config_path.empty() ? "Open a workflow in Settings." : config_path.c_str(), fonts, p);
    if (!ready) return;
    if (ui::button("start-all", fonts.chinese ? "全部启动" : "Start all", ui::ButtonKind::Primary, p, active.scale))
        for (const auto& worker : controller.workers()) start(worker.index);
    ImGui::SameLine(); if (ui::button("stop-all", fonts.chinese ? "安全停止" : "Stop safely", ui::ButtonKind::Secondary, p, active.scale)) controller.stop_all();
    if (ImGui::BeginTable("workers", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollX)) {
        for (const char* name : {"Worker", "State", "Stage", "Progress", "Action"}) ImGui::TableSetupColumn(name);
        ImGui::TableHeadersRow();
        for (const auto& worker : controller.workers()) {
            ImGui::PushID(worker.index); ImGui::TableNextRow(); ImGui::TableNextColumn();
            if (ImGui::Selectable(worker.name.c_str(), state.ui_worker == worker.index)) state.ui_worker = worker.index;
            ImGui::TableNextColumn(); ImGui::TextUnformatted(ecmgui::state_name(worker.state()));
            ImGui::TableNextColumn(); ImGui::TextUnformatted(worker.stage2 ? "Stage 2" : "Stage 1");
            ImGui::TableNextColumn(); const auto progress = worker.stage2 ? worker.stage2_status.progress() : worker.progress;
            ImGui::ProgressBar(static_cast<float>(progress.valid ? progress.pct / 100 : 0), {-1, 0});
            ImGui::TableNextColumn();
            if (worker.running()) { if (ImGui::SmallButton("Stop")) controller.request_stop(worker.index); }
            else if (ImGui::SmallButton("Start")) start(worker.index);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}
void App::draw_inspector(const ui::Fonts& fonts) {
    fonts.push(ui::FontRole::Body);
    if (const auto worker = selected()) {
        ImGui::Text("%s / %s", worker->name.c_str(), ecmgui::state_name(worker->state()));
        ImGui::TextWrapped("%s", worker->task.c_str()); ImGui::Separator();
        ImGui::TextWrapped("%s", worker->effective_config.c_str());
        ImGui::TextWrapped("Worker version: %s", worker->application_version.empty() ? "unknown" : worker->application_version.c_str());
        if (ImGui::Button("Copy command")) ImGui::SetClipboardText(worker->command_line().c_str());
        ImGui::TextWrapped("%s", worker->command_line().c_str());
    } else ImGui::TextUnformatted("No worker selected");
    ImGui::PopFont();
}
void App::draw_logs(const ui::Fonts& fonts) {
    ImGui::Checkbox("Raw##log", &state.ui_raw_log); ImGui::SameLine(); ImGui::Checkbox("Follow##log", &state.ui_follow_log);
    ImGui::BeginChild("output", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar); fonts.push(ui::FontRole::Mono);
    if (const auto worker = selected()) {
        const auto& lines = state.ui_raw_log ? worker->raw : worker->events;
        ImGuiListClipper clip; clip.Begin(static_cast<int>(lines.size()));
        while (clip.Step()) for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) ImGui::TextUnformatted(lines[static_cast<size_t>(i)].c_str());
    } else for (const auto& line : messages) ImGui::TextUnformatted(line.c_str());
    if (state.ui_follow_log) ImGui::SetScrollHereY(1);
    ImGui::PopFont(); ImGui::EndChild();
}
void App::draw_page(const ui::Settings& active, const ui::Fonts& fonts) {
    draw_workers(active, fonts);
}
}
