#include "app.h"
#include "version.h"
#include "ui/navigation.h"
#include "ui/i18n.h"
#include "ui/chart.h"
#include "imgui_internal.h" // DockBuilder is kept in the v2 layout module.
#include <algorithm>
#include <cstdio>
#include <thread>
#include <cstring>

namespace ecmgui_v2 {
namespace {
// The existing Controller announces a stage change before spawning its child.
// A typed refusal stops that dispatch without changing the legacy Controller.
struct DispatchBlocked : std::runtime_error { using std::runtime_error::runtime_error; };
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
    if(ui::pinned(state.ui_right_pages,state.ui_page))for(int i=0;i<7;++i)if(!ui::pinned(state.ui_right_pages,i)){state.ui_page=i;break;}
    controller.set_sinks([this](const std::string& text) { trace(text); }, [this](const ecmgui::WorkerNotice& notice) {
        if(notice.kind==ecmgui::WorkerNotice::Kind::StageChanged)
            for(const auto& worker:controller.workers())if(worker.index==notice.worker&&!supports_owner(worker.exe))
                throw DispatchBlocked(status);
        if (notices.size() == 32) notices.erase(notices.begin()); notices.push_back(notice);
        if (!notice.text.empty()) {status = notice.text;toast(notice.text,notice.kind==ecmgui::WorkerNotice::Kind::Error);}
    });
}
App::~App() { shutdown_workers(); }
void App::toast(const std::string& text,bool error) {
    if(text.empty())return;
    if(!toasts.empty()&&toasts.back().text==text){toasts.back().until=GetTickCount64()+4000;toasts.back().error|=error;return;}
    if(toasts.size()==4)toasts.pop_front();toasts.push_back({text,GetTickCount64()+4000,error});
}
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
        configure_workers(); ready = true; gen_valid = false; selected_results.clear(); result_filter=ecmgui::ResultFilter{};result_view_dirty=true;
        ecmgui::stamp_file(config_path, config_stamp); autostart_pending = true;
        results.set_paths(ecmgui::result_paths(config.lines(), config_path, static_cast<int>(controller.workers().size())));
        status = "Workflow opened; worker control requires queue-owner protocol 1."; trace("config=" + path);
        return true;
    } catch (const std::exception& error) { status = error.what(); trace(status); return false; }
}
void App::configure_workers() {
    auto settings = ecm_config::read_gui(ecm_config::section_entries(config.lines(), "gui"));
    ecmgui::WorkerSettings workers;
    workers.ini_path = config_path; workers.num_workers = std::clamp(settings.NumWorkers, 1, 1024);
    if(state.worker_count>0)workers.num_workers=state.worker_count;
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
    if (config.dirty()) { status="Save or reload workflow changes before starting workers."; return false; }
    try {
        for (const auto& worker : controller.workers()) if (worker.index == index) {
            if (worker.running()) return true;
            if (!supports_owner(controller.resolve_worker_exe(index))) return false;
            // Only the selected engine is needed now. StageChanged validates any
            // later dispatch before Controller can start the replacement child.
            auto lease = std::make_unique<ecm_queue::OwnerLease>(worker.worktodo, index, ".gui-control.lock");
            { ecm_queue::OwnerLease probe(worker.worktodo, index); }
            if (!controller.start(index, true)) return false;
            leases_[index] = std::move(lease); toast((state.ui_language=="zh"?"已启动 worker ":"Worker started: ")+std::to_string(index)); return true;
        }
    } catch (const std::exception& error) { status = error.what(); trace(status); }
    return false;
}
void App::tick() {
    if (autostart_pending) {
        autostart_pending = false;
        for (const auto& worker : controller.workers()) if (worker.autostart) start(worker.index);
    }
    for(const auto& worker:controller.workers())
        if(worker.state()==ecmgui::WorkerRunState::Restarting&&!supports_owner(worker.exe)){
            controller.request_stop(worker.index);toast(status,true);
        }
    try { controller.tick(); }
    catch(const DispatchBlocked& error) { status=error.what();toast(status,true); }
    for (const auto& worker : controller.workers())
        if (!worker.running() && worker.state() != ecmgui::WorkerRunState::Restarting && worker.state() != ecmgui::WorkerRunState::Starting) leases_.erase(worker.index);
    const auto now = GetTickCount64();
    if ((state.ui_page == 1 || (state.ui_right_visible && ui::pinned(state.ui_right_pages,1))) && now - last_results >= 500) {
        last_results = now; if (results.refresh()) {
            result_view_dirty=true;
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
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, std::clamp(170*scale/viewport->WorkSize.x,.16f,.35f), &nav, &center);
    if ((wide && state.ui_inspector)||(state.ui_right_pages && state.ui_right_visible)) ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, wide?.34f:.43f, &inspector, &center);
    if (state.ui_logs) ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, .28f, &logs, &center);
    ImGui::DockBuilderDockWindow("Navigation###v2.navigation", nav);
    ImGui::DockBuilderDockWindow("Workspace###v2.workspace", center);
    if (inspector) ImGui::DockBuilderDockWindow("Inspector###v2.inspector", inspector);
    if (inspector && state.ui_right_pages && state.ui_right_visible) ImGui::DockBuilderDockWindow("Tools###v2.tools", inspector);
    if (logs) ImGui::DockBuilderDockWindow("Logs###v2.logs", logs);
    ImGui::DockBuilderFinish(dock);
    if(state.ui_right_pages&&state.ui_right_visible)focus_right_=true;
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
    ImGui::Begin(name,open,std::strstr(name,"###v2.navigation")?ImGuiWindowFlags_HorizontalScrollbar:0); window_dpi_[name] = ImGui::GetWindowViewport()->DpiScale;
}
void App::end_window() { ImGui::End(); ImGui::PopStyleVar(6); }
bool App::save_preferences() {
    try {
        size_t length=0;const auto* layout=ImGui::SaveIniSettingsToMemory(&length);state.layout(std::string(layout,length));
        const bool saved=state.save();status=saved?"UI preferences saved":state.notice;toast(status,!saved);return saved;
    }catch(const std::exception& error){status=error.what();toast(status,true);return false;}
}
bool App::move_page(int page,bool right) {
    if(!ui::move_page(state.ui_right_pages,state.ui_page,state.ui_right_page,page,right)){toast(state.ui_language=="zh"?"至少保留一个中央页面":"Keep at least one central page");return false;}
    if(right){state.ui_right_visible=true;focus_right_=true;}
    reset_layout=true;
    toast(state.ui_language=="zh"?(right?"已移到右侧工具栏":"已移回左侧导航"):(right?"Moved to right tools":"Moved to left navigation"));return true;
}
void App::draw(ui::Settings& pending, const ui::Settings& active, const ui::Fonts& fonts) {
    if(status!=last_status){last_status=status;toast(status);}
    while(!toasts.empty()&&toasts.front().until<GetTickCount64())toasts.pop_front();
    const bool zh = fonts.chinese; const auto t = [zh](const char* en, const char* cn) { return zh ? cn : en; };
    if (ImGui::BeginMainMenuBar()) {
        ImGui::Text("ECM-ELF %s", version); ImGui::Separator();
        if(ImGui::BeginMenu(t("View","视图"))){
            if (ImGui::MenuItem(t("Reset layout", "重置布局"))) reset_layout = true;
            if(ImGui::MenuItem(t("Inspector", "检查区"),nullptr,&state.ui_inspector))reset_layout=true;
            if(ImGui::MenuItem(t("Logs", "日志"),nullptr,&state.ui_logs))reset_layout=true;
            if(state.ui_right_pages && ImGui::MenuItem(t("Tools", "工具栏"),nullptr,&state.ui_right_visible))reset_layout=true;
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu(t("Language","语言"))){
            if(ImGui::MenuItem("English",nullptr,state.ui_language=="en"))state.ui_language="en";
            if(ImGui::MenuItem("中文",nullptr,state.ui_language=="zh"))state.ui_language="zh";
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem(t("Notices", "通知"))) ImGui::OpenPopup("Notices###v2.notices");
        if (ImGui::BeginPopup("Notices###v2.notices")) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize()*34);
            const auto begin = messages.size() > 20 ? messages.size()-20 : 0;
            for (size_t i=begin;i<messages.size();++i) ImGui::TextUnformatted(messages[i].c_str());
            ImGui::PopTextWrapPos(); ImGui::EndPopup();
        }
        if(ImGui::GetMainViewport()->WorkSize.x>=1000*active.scale)ImGui::TextDisabled("%s", ui::message(status,zh).c_str()); ImGui::EndMainMenuBar();
    }
    const auto dock = ImGui::DockSpaceOverViewport(ImGui::GetID("v2.dock"), ImGui::GetMainViewport());
    const bool wide = ImGui::GetMainViewport()->WorkSize.x >= 1100 * active.scale;
    const auto node = ImGui::DockBuilderGetNode(dock);
    if (reset_layout || (!layout_ready_ && (!node || !node->ChildNodes[0])) || (layout_ready_ && layout_wide_ != wide)) {
        build_layout(dock, active.scale); reset_layout = false;
    }
    layout_ready_ = true; layout_wide_ = wide;
    begin_window(t("Navigation###v2.navigation","导航###v2.navigation"),nullptr,active); fonts.push(ui::FontRole::Body);
    const bool compact = ImGui::GetContentRegionAvail().x < 110 * active.scale;
    const auto item_center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    for(auto& pos:nav_items)pos={};for(auto& pos:right_items)pos={};
    for (int i = 0; i < 7; ++i) {
        if(ui::pinned(state.ui_right_pages,i))continue;
        ImGui::PushID(i); const auto label = std::string(ui::page_name(i,zh)) + "###page";
        if (ImGui::Selectable(label.c_str(), state.ui_page == i,0,{std::max(ImGui::GetContentRegionAvail().x,ImGui::CalcTextSize(ui::page_name(i,zh)).x+12*active.scale),0})) state.ui_page = i;
        nav_items[i]=item_center();
        if (compact && ImGui::IsItemHovered()) ImGui::SetTooltip("%s",ui::page_name(i,zh));
        if(ImGui::BeginDragDropSource()){ImGui::SetDragDropPayload("V2_PAGE",&i,sizeof(i));ImGui::TextUnformatted(ui::page_name(i,zh));ImGui::EndDragDropSource();}
        if(ImGui::BeginPopupContextItem("location")){if(ImGui::MenuItem(t("Move to right tools","移到右侧工具栏")))move_page(i,true);ImGui::EndPopup();}
        ImGui::PopID();
    }
    ImGui::Separator();ImGui::Button(compact?">###tools-drop":t("Tools >###tools-drop","工具 >###tools-drop"),{-1,0});
    nav_drop=item_center();
    if(compact&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",t("Drop a page here to move it to right tools","将页面拖到此处，移入右侧工具栏"));
    if(ImGui::IsItemClicked()&&state.ui_right_pages){state.ui_right_visible=true;reset_layout=true;}
    if(ImGui::BeginDragDropTarget()){if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE"))if(payload->DataSize==sizeof(int))move_page(*static_cast<const int*>(payload->Data),true);ImGui::EndDragDropTarget();}
    ImGui::TextWrapped("%s",compact?"":t("Right-click or drag a page to Tools.","右键页面或拖到工具按钮。"));
    ImGui::Button(compact?"<###nav-drop":t("< Navigation###nav-drop","< 导航###nav-drop"),{-1,0});
    nav_return=item_center();
    if(compact&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",t("Drop a tool here to return it to navigation","将工具拖到此处，移回导航"));
    if(ImGui::BeginDragDropTarget()){if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE"))if(payload->DataSize==sizeof(int))move_page(*static_cast<const int*>(payload->Data),false);ImGui::EndDragDropTarget();}
    ImGui::PopFont(); end_window();
    begin_window(t("Workspace###v2.workspace","工作区###v2.workspace"),nullptr,active);
    workspace_docked=ImGui::IsWindowDocked();
    if(drawn_center_page_!=state.ui_page){ImGui::SetScrollY(0);drawn_center_page_=state.ui_page;}
    fonts.push(ui::FontRole::Body);
    ImGui::PushTextWrapPos(0); draw_page(active, fonts); ImGui::PopTextWrapPos(); ImGui::PopFont(); end_window();
    if (state.ui_inspector && wide) { begin_window(t("Inspector###v2.inspector","检查区###v2.inspector"), &state.ui_inspector,active); draw_inspector(fonts); end_window(); }
    if (state.ui_logs) { begin_window(t("Logs###v2.logs","日志###v2.logs"), &state.ui_logs,active); draw_logs(fonts); end_window(); }
    if(state.ui_right_pages && state.ui_right_visible){
        if(focus_right_){ImGui::SetNextWindowFocus();focus_right_=false;}
        begin_window(t("Tools###v2.tools","工具###v2.tools"),&state.ui_right_visible,active);fonts.push(ui::FontRole::Body);
        if(!ui::pinned(state.ui_right_pages,state.ui_right_page))for(int i=0;i<7;++i)if(ui::pinned(state.ui_right_pages,i)){state.ui_right_page=i;break;}
        {
            bool first=true;
            for(int i=0;i<7;++i)if(ui::pinned(state.ui_right_pages,i)){
                const auto name=ui::page_name(i,zh);const auto width=ImGui::CalcTextSize(name).x+ImGui::GetStyle().FramePadding.x*2;
                if(!first&&ImGui::GetItemRectMax().x+width+ImGui::GetStyle().ItemSpacing.x<ImGui::GetWindowPos().x+ImGui::GetWindowWidth()-ImGui::GetStyle().WindowPadding.x)ImGui::SameLine();first=false;
                if(state.ui_right_page==i)ImGui::PushStyleColor(ImGuiCol_Button,ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
                const bool selected=state.ui_right_page==i;
                if(ImGui::Button(name))state.ui_right_page=i;
                right_items[i]=item_center();
                if(selected)ImGui::PopStyleColor();
                if(ImGui::BeginDragDropSource()){ImGui::SetDragDropPayload("V2_PAGE",&i,sizeof(i));ImGui::TextUnformatted(ui::page_name(i,zh));ImGui::EndDragDropSource();}
                if(ImGui::BeginPopupContextItem()){if(ImGui::MenuItem(t("Move to navigation","移回导航")))move_page(i,false);ImGui::EndPopup();}
            }
        }
        const int right=state.ui_right_page;
        if(drawn_right_page_!=right){ImGui::SetScrollY(0);drawn_right_page_=right;}
        if(ImGui::SmallButton(t("< Move to navigation","< 移回导航")))move_page(right,false);
        if(ui::pinned(state.ui_right_pages,right)){ImGui::Separator();ImGui::PushID("right-content");ImGui::PushTextWrapPos(0);draw_page(active,fonts,right);ImGui::PopTextWrapPos();ImGui::PopID();}
        ImGui::PopFont();end_window();
    }
    if (test_float) {
        ImGui::SetNextWindowPos({ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x + 30, ImGui::GetMainViewport()->Pos.y + 30}, ImGuiCond_FirstUseEver);
        begin_window("Detached lifecycle probe###v2.float", &test_float,active);
        probe_dpi = ImGui::GetWindowViewport()->DpiScale; probe_font = ImGui::GetFontSize(); probe_padding = ImGui::GetStyle().FramePadding.y;
        ImGui::TextUnformatted("Platform viewport probe"); end_window();
    }
    if (controller.exit_phase() == ecmgui::WorkerController::ExitPhase::Confirm) ImGui::OpenPopup("Exit###v2.exit");
    if (ImGui::BeginPopupModal("Exit###v2.exit", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(t("Stop workers safely before closing?", "安全停止 worker 后退出？"));
        if (ImGui::Button(t("Stop and exit", "停止并退出"))) { begin_exit(); ImGui::CloseCurrentPopup(); }
        ImGui::SameLine(); if (ImGui::Button(t("Cancel", "取消"))) { controller.cancel_close(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    const auto viewport=ImGui::GetMainViewport();float toast_bottom=viewport->WorkPos.y+viewport->WorkSize.y-14*active.scale;
    int toast_index=0;
    for(auto it=toasts.rbegin();it!=toasts.rend();++it){
        if(!state.ui_toasts&&!it->error)continue;
        ImGui::SetNextWindowViewport(viewport->ID);ImGui::SetNextWindowPos({viewport->WorkPos.x+viewport->WorkSize.x-14*active.scale,toast_bottom},ImGuiCond_Always,{1,1});
        ImGui::SetNextWindowSize({std::min(400*active.scale,viewport->WorkSize.x-28*active.scale),0});
        ImGui::SetNextWindowBgAlpha(.96f);const auto name="Notice###v2.toast"+std::to_string(toast_index++);
        ImGui::Begin(name.c_str(),nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoFocusOnAppearing|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoInputs|ImGuiWindowFlags_AlwaysAutoResize);
        if(it->error)ImGui::PushStyleColor(ImGuiCol_Text,ui::palette(active.light).danger);
        ImGui::TextWrapped("%s",ui::message(it->text,zh).c_str());if(it->error)ImGui::PopStyleColor();toast_bottom-=ImGui::GetWindowSize().y+8*active.scale;ImGui::End();
    }
    {
        // Preferences are applied by the host at the next frame boundary.
        pending.light = state.ui_theme == "light"; pending.chinese = state.ui_language == "zh";
        if (state.ui_scale > 0) pending.scale = state.ui_scale;
    }
}
void App::draw_workers(const ui::Settings& active, const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;const auto t=[zh](const char* en,const char* cn){return zh?cn:en;};
    const auto p = ui::palette(active.light);
    ui::section_header(fonts.chinese ? "计算任务" : "Workers", config_path.empty() ? t("Open a workflow in Settings.","请在设置中打开工作流。") : config_path.c_str(), fonts, p);
    if (!ready) return;
    if (ui::button("start-all", fonts.chinese ? "全部启动" : "Start all", ui::ButtonKind::Primary, p, active.scale))
        for (const auto& worker : controller.workers()) if(!start(worker.index))toast(status,true);
    start_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    ImGui::SameLine(); if (ui::button("stop-all", fonts.chinese ? "安全停止" : "Stop safely", ui::ButtonKind::Secondary, p, active.scale)) {controller.stop_all();toast(fonts.chinese?"已请求安全停止，等待检查点边界":"Safe stop requested; waiting for checkpoint boundary");}
    stop_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    for (const auto& worker : controller.workers()) if (worker.p95.valid) {
        const bool pending = worker.p95.level == ecmgui::P95Notice::Level::Pending;
        ImGui::TextColored(pending ? ImVec4(1,.35f,.4f,1) : ImVec4(.8f,.65f,.35f,1), "Prime95 %s: %s %s", worker.name.c_str(),worker.p95.error.c_str(),worker.p95.note.c_str());
    }
    if (ImGui::BeginTable("workers", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable)) {
        for (const char* name : {t("Worker","Worker"),t("State","状态"),t("Stage","阶段"),t("Task progress","任务进度"),t("s/curve","秒/曲线"),t("Done / hits","完成/命中"),t("Action","操作")}) ImGui::TableSetupColumn(name,ImGuiTableColumnFlags_WidthFixed,100*active.scale);
        ImGui::TableHeadersRow();
        for (const auto& worker : controller.workers()) {
            ImGui::PushID(worker.index); ImGui::TableNextRow(); ImGui::TableNextColumn();
            if (ImGui::Selectable(worker.name.c_str(), state.ui_worker == worker.index)) state.ui_worker = worker.index;
            ImGui::TableNextColumn(); ImGui::TextUnformatted(ui::message(ecmgui::state_name(worker.state()),zh).c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(worker.stage2 ? "Stage 2" : "Stage 1");
            ImGui::TableNextColumn(); const auto progress = worker.stage2 ? worker.stage2_status.progress() : worker.progress;
            ImGui::ProgressBar(static_cast<float>(progress.valid ? progress.pct / 100 : 0), {-1, 0});
            ImGui::TableNextColumn(); if (progress.valid) ImGui::Text("%.3f",progress.s_per_curve); else ImGui::TextUnformatted("--");
            ImGui::TableNextColumn(); ImGui::Text("%lld / %lld",worker.tasks_done,worker.hits);
            ImGui::TableNextColumn();
            if (worker.running()) { if (ImGui::SmallButton(t("Stop","停止"))) {controller.request_stop(worker.index);toast("Safe stop requested: "+worker.name);} }
            else if (ImGui::SmallButton(t("Start","启动"))&&!start(worker.index))toast(status,true);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}
void App::draw_inspector(const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;const auto t=[zh](const char* en,const char* cn){return zh?cn:en;};
    fonts.push(ui::FontRole::Body);
    if (const auto worker = selected()) {
        ImGui::Text("%s / %s", worker->name.c_str(), ui::message(ecmgui::state_name(worker->state()),zh).c_str());
        ImGui::TextWrapped("%s", worker->task.c_str()); ImGui::Separator();
        if (worker->stage2 && worker->stage2_status.valid) {
            const auto& s=worker->stage2_status;
            ImGui::Text(t("Task curves: %llu / %llu","任务曲线数：%llu / %llu"),s.done,s.total);
            ImGui::Text(t("Phase: %s | batch %llu / %llu","阶段：%s | 批次 %llu / %llu"),s.phase.c_str(),s.batch,s.batches);
            if(s.curve_model=="ed")ImGui::Text("Edwards sigma=%llu",s.sigma);
            else ImGui::Text("%s param=%d sigma=%llu",s.curve_model.c_str(),s.param,s.sigma);
            ImGui::Text("B1=%llu B2=%llu",s.b1,s.b2);
        }
        if (!worker->hist_s_per_curve.empty()) {
            std::vector<ui::ChartSeries> series{{t("Seconds per curve","每曲线耗时"),ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f).stage1,{}}};
            for(size_t i=0;i<worker->hist_s_per_curve.size();++i)series[0].points.push_back({static_cast<double>(i+1),worker->hist_s_per_curve[i]});
            ui::chart("worker-speed",series,t("Sample","采样序号"),t("s/curve","秒/曲线"),zh,0,160);
        }
        if (ImGui::Button(t("Open worker folder","打开 worker 目录"))) ecmgui::open_in_explorer(worker->working_dir);
        ImGui::TextWrapped("%s", worker->effective_config.c_str());
        ImGui::TextWrapped(t("Worker version: %s","计算程序版本：%s"), worker->application_version.empty() ? t("unknown","未知") : worker->application_version.c_str());
        if (ImGui::Button(t("Copy command","复制命令"))) ImGui::SetClipboardText(worker->command_line().c_str());
        ImGui::TextWrapped("%s", worker->command_line().c_str());
    } else ImGui::TextUnformatted(t("No worker selected","未选中 worker"));
    ImGui::PopFont();
}
void App::draw_logs(const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;const auto t=[zh](const char* en,const char* cn){return zh?cn:en;};
    ImGui::Checkbox(t("Raw##log","原始##log"), &state.ui_raw_log); ImGui::SameLine(); ImGui::Checkbox(t("Follow##log","跟随##log"), &state.ui_follow_log);
    ImGui::BeginChild("output", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar); fonts.push(ui::FontRole::Mono);
    if (const auto worker = selected()) {
        const auto& lines = state.ui_raw_log ? worker->raw : worker->events;
        ImGuiListClipper clip; clip.Begin(static_cast<int>(lines.size()));
        while (clip.Step()) for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) ImGui::TextUnformatted(lines[static_cast<size_t>(i)].c_str());
    } else for (const auto& line : messages) ImGui::TextUnformatted(line.c_str());
    if (state.ui_follow_log) ImGui::SetScrollHereY(1);
    ImGui::PopFont(); ImGui::EndChild();
}
void App::draw_page(const ui::Settings& active, const ui::Fonts& fonts,int page) {
    switch (page<0?state.ui_page:page) {
    case 0: draw_workers(active, fonts); break;
    case 1: draw_results(fonts.chinese); break;
    case 2: draw_generator(fonts.chinese); break;
    case 3: draw_probability(fonts.chinese); break;
    case 4: draw_gpu(fonts.chinese); break;
    case 5: draw_settings(fonts.chinese); break;
    case 6: ui::state_strip(ui::palette(active.light),active.scale,fonts.chinese); break;
    }
}
}
