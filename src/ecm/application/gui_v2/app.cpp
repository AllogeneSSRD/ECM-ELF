#include "app.h"
#include "version.h"
#include "ui/navigation.h"
#include "ui/i18n.h"
#include "ui/chart.h"
#include "ui/worker_presentation.h"
#include "ui/log_presentation.h"
#include "ui/group_order_presentation.h"
#include "../gui/task_display.h"
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
    if(state.ui_language_file.empty())state.ui_language_file=state.ui_language=="zh"?"chineseSimplified.xml":"english.xml";
    if(ui::language_directory().empty()){std::string error;ui::load_languages(ecmgui::exe_dir()+"/localization/v2",error,state.ui_language_file);if(!error.empty())trace("localization: "+error);}
    // Read old page/sidecar IDs without keeping Settings in the dock workspace.
    if(state.ui_page==5){state.ui_page=0;open_settings();}
    if(state.ui_right_pages&(1<<5)){state.ui_right_pages&=~(1<<5);reset_layout=true;}
    if((state.ui_right_pages&223)==223){state.ui_right_pages&=~1;reset_layout=true;}
    if(state.ui_right_page==5)for(int i:{4,3,2,1,0,6,7})if(ui::pinned(state.ui_right_pages,i)){state.ui_right_page=i;break;}
    if(state.ui_right_pages){
        if(!state.ui_right_visible||state.layout().find("v2.tools")!=std::string::npos)reset_layout=true;
        state.ui_right_visible=true;
    }
    if(ui::pinned(state.ui_right_pages,state.ui_page))for(int i=0;i<8;++i)if(i!=5&&!ui::pinned(state.ui_right_pages,i)){state.ui_page=i;break;}
    controller.set_sinks([this](const std::string& text) { trace(text); }, [this](const ecmgui::WorkerNotice& notice) {
        if(notice.kind==ecmgui::WorkerNotice::Kind::StageChanged)
            for(const auto& worker:controller.workers())if(worker.index==notice.worker&&!supports_owner(worker.exe))
                throw DispatchBlocked(status);
        if (notices.size() == 32) notices.erase(notices.begin()); notices.push_back(notice);
        if (!notice.text.empty()) {status = notice.text;toast(notice.text,notice.kind==ecmgui::WorkerNotice::Kind::Error);}
    });
}
App::~App() { shutdown_workers(); }
void App::toast(const std::string& text,bool error,bool repeat) {
    if(text.empty())return;
    if(!repeat&&!toasts.empty()&&toasts.back().text==text){toasts.back().until=GetTickCount64()+4000;toasts.back().error|=error;return;}
    if(toasts.size()==4)toasts.pop_front();toasts.push_back({text,GetTickCount64()+4000,error});
}
void App::notify_group_order(){
    const bool zh=state.ui_language=="zh";
    group_order_success=ui::order_succeeded(group_order_job.result,group_order_job.error);
    ++group_order_completions;
    std::string message=ui::text(group_order_success?"go.finished":"go.failed",zh);
    if(!group_order_success){
        const auto status=group_order_job.result.get("status").text();
        if(!status.empty())message+=" · "+std::string(ui::text(("go.status."+status).c_str(),zh));
        else if(!group_order_job.error.empty())message+=" · "+group_order_job.error;
    }
    toast(message,!group_order_success,true);
}
void App::trace(const std::string& text) {
    if (messages.size() == 4000) messages.pop_front(); messages.push_back(text);
    std::fprintf(log_, "app: %s\n", text.c_str()); std::fflush(log_);
}
void App::initialize(const std::string& path, bool disable_nvml) {
    no_nvml = disable_nvml;
    if (!path.empty()) load_config(path); else status = "Open a workflow TOML to manage workers.";
    if (!no_nvml) {
        if(!gpu.start(500))trace("GPU monitor unavailable: " + gpu.reason());
        else {gpu_details.start(gpu.nvml_source(),gpu.devices());gpu_clocks.start(gpu.nvml_source(),system_clock_query_tool());}
    }
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
        group_order_job.cancel();
        group_order_loaded=false;
        config = std::move(candidate); config_path = path; leases_.clear(); verified_.clear();log_views_.clear();log_last_worker_=-1;
        saved_clock_profiles.clear();saved_clock_automatic.clear();clock_drafts.clear();
        clock_profiles_valid=parse_clock_profiles(config.get("gui","gpu_clock_ranges"),saved_clock_profiles,&saved_clock_automatic);
        // Loading a different workflow does not replay startup commands. The
        // first valid workflow seeds per-GPU policies once for this process.
        if(!clock_policies_seeded){
            clock_policies_seeded=true;
            for(const auto& entry:saved_clock_profiles)if(!clock_auto_policies.count(entry.first)){
                auto& policy=clock_auto_policies[entry.first];policy.range=entry.second;policy.enabled=saved_clock_automatic.count(entry.first)!=0;
            }
        }
        if(!clock_profiles_valid)trace("Invalid gui.gpu_clock_ranges; clock profile saving disabled until corrected and reloaded.");
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
    worker_b2_ratios_.clear();
    for(const auto& worker:controller.workers())for(const auto& entry:ecm_config::cli_entries(config.lines(),worker.index))if(entry.first=="b2_ratio")
        worker_b2_ratios_[worker.index]=ecm_config::read_stage2(ecm_config::Entries{entry}).b2_ratio;
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
            leases_[index] = std::move(lease); toast((ecmgui_v2::ui::text("worker_started",state.ui_language=="zh"))+std::to_string(index)); return true;
        }
    } catch (const std::exception& error) { status = error.what(); trace(status); }
    return false;
}
void App::tick() {
    tick_gpu_clocks();
    if(group_order_job.poll())notify_group_order();
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
    auto timing_file=state.path;timing_file+=".stage2-timing.json";stage2_history_.load(timing_file);
    const bool poll_tune=now>=stage2_tune_poll_;if(poll_tune)stage2_tune_poll_=now+1000;
    for(const auto& worker:controller.workers())if(worker.stage2){
        const auto device=gpu_details.get(worker.device).uuid;
        stage2_history_.observe(worker,device);
        const auto& stage=worker.stage2_status;auto& view=stage2_views_[worker.index];
        const auto curve=std::to_string(stage.current)+"|"+std::to_string(stage.record)+"|"+std::to_string(stage.sigma);
        if(view.clock_curve!=curve||view.phase!=stage.phase){view.clock_curve=curve;view.phase=stage.phase;view.phase_at=view.batch_at=now;view.batch=stage.batch;}
        if(view.batch!=stage.batch){view.batch=stage.batch;view.batch_at=now;}
        if(poll_tune){
            const auto base=std::filesystem::u8path(config_path).parent_path();
            auto file=state.ui_stage2_tune_file;
            if(file.empty())file=config.get("workers."+std::to_string(worker.index)+".stage2","stage2_tune_profile",config.get("stage2","stage2_tune_profile"));
            std::filesystem::path path;
            if(!file.empty()){path=std::filesystem::u8path(file);if(path.is_relative())path=base/path;}
            else for(const auto& dir:{base,std::filesystem::u8path(worker.exe).parent_path()})for(const char* name:{"tune.toml","stage2_ecm_tune.toml"}){
                std::error_code error;const auto candidate=dir/name;if(path.empty()&&std::filesystem::is_regular_file(candidate,error))path=candidate;
            }
            std::error_code error;const auto stamp=std::filesystem::last_write_time(std::filesystem::u8path(worker.exe),error);
            stage2_engine_ids_[worker.index]=error?"unknown":std::to_string(stamp.time_since_epoch().count());
            stage2_tunes_[worker.index].refresh(path);
        }
    }
    if ((state.ui_page == 1 || ui::pinned(state.ui_right_pages,1)) && now - last_results >= 500) {
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
bool App::request_close() {
    if(controller.exit_phase()!=ecmgui::WorkerController::ExitPhase::Idle)return false;
    return controller.request_close();
}
void App::begin_exit() {
    if(controller.exit_phase()!=ecmgui::WorkerController::ExitPhase::Stopping)controller.begin_graceful_exit();
}
void App::shutdown_workers() noexcept {
    try {
        if (controller.busy()) {
            begin_exit(); const auto end = controller.exit_deadline() + 1000;
            while (controller.busy() && GetTickCount64() < end) { controller.tick(); Sleep(20); }
        }
        controller.force_close(); leases_.clear();
    } catch (...) { controller.force_close(); }
}
void App::build_layout(unsigned dock, float scale) {
    ++layout_build_count;
    const auto viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(dock); ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock, viewport->WorkSize);
    ImGuiID center = dock, nav = 0, inspector = 0, logs = 0;
    const bool wide = viewport->WorkSize.x >= 1100 * scale;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, std::clamp(170*scale/viewport->WorkSize.x,.16f,.35f), &nav, &center);
    if ((wide && state.ui_inspector)||state.ui_right_pages) ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, wide?.34f:.43f, &inspector, &center);
    if (state.ui_logs) ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, .28f, &logs, &center);
    ImGui::DockBuilderDockWindow("Navigation###v2.navigation", nav);
    ImGui::DockBuilderDockWindow("Workspace###v2.workspace", center);
    if (inspector) ImGui::DockBuilderDockWindow("Inspector###v2.inspector", inspector);
    if(inspector)for(char id:state.ui_nav_order){const int page=id-'0';if(ui::pinned(state.ui_right_pages,page))ImGui::DockBuilderDockWindow(ui::right_window(page,false).c_str(),inspector);}
    log_dock_=logs;
    if (logs){
        ImGui::DockBuilderDockWindow("Manager###v2.logs", logs);
        for(const auto& w:controller.workers())ImGui::DockBuilderDockWindow(("Worker###v2.log.worker."+std::to_string(w.index)).c_str(),logs);
    }
    ImGui::DockBuilderFinish(dock);
    if(state.ui_right_pages)focus_right_=true;
}
bool App::begin_window(const char* name, bool* open, const ui::Settings& active,ImGuiWindowFlags flags) {
    auto found = window_dpi_.find(name);
    const float dpi = state.ui_scale > 0 || found == window_dpi_.end() ? active.scale : found->second;
    const float ratio = dpi / active.scale;
    const auto& style = ImGui::GetStyle();
    const auto scaled = [ratio](ImVec2 v) { return ImVec2(v.x*ratio,v.y*ratio); };
    const bool navigation=std::strstr(name,"###v2.navigation")!=nullptr;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,navigation?ImVec2{8*dpi,style.WindowPadding.y*ratio}:scaled(style.WindowPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,scaled(style.FramePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,scaled(style.ItemSpacing));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,scaled(style.CellPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,style.FrameRounding*ratio);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize,style.ScrollbarSize*ratio);
    if(navigation)ImGui::SetNextWindowScroll({0,0});
    const bool visible=ImGui::Begin(name,open,flags|(navigation?ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse:0)); window_dpi_[name] = ImGui::GetWindowViewport()->DpiScale;return visible;
}
void App::end_window() { ImGui::End(); ImGui::PopStyleVar(6); }
bool App::save_preferences(const char* success_key) {
    try {
        size_t length=0;const auto* layout=ImGui::SaveIniSettingsToMemory(&length);state.layout(std::string(layout,length));
        const auto* viewport=ImGui::GetMainViewport();
        const auto hwnd=static_cast<HWND>(viewport->PlatformHandleRaw?viewport->PlatformHandleRaw:viewport->PlatformHandle);
        if(hwnd&&!IsIconic(hwnd)){
            state.window_maximized=IsZoomed(hwnd)!=FALSE;
            if(!state.window_maximized){RECT rect{};if(GetWindowRect(hwnd,&rect)){state.window_x=rect.left;state.window_y=rect.top;state.window_width=rect.right-rect.left;state.window_height=rect.bottom-rect.top;}}
        }
        const bool saved=state.save();status=saved?ui::text(success_key,state.ui_language!="en"):state.notice;toast(status,!saved);return saved;
    }catch(const std::exception& error){status=error.what();toast(status,true);return false;}
}
bool App::save_layout_config(){
    if(!state.writable){status=state.notice;toast(status,true);return false;}
    if(config.dirty()&&!save_business()){toast(status,true);return false;}
    const bool saved=save_preferences("message.layout_config_saved");if(saved)++view_save_count;return saved;
}
void App::set_logs_visible(bool visible){
    if(visible==state.ui_logs)return;
    std::vector<std::string> names{"Manager###v2.logs"};
    for(const auto& worker:controller.workers())names.push_back("Worker###v2.log.worker."+std::to_string(worker.index));
    const auto dock_id=[](const std::string& name,unsigned& id){
        if(const auto* window=ImGui::FindWindowByName(name.c_str())){id=window->DockId;return true;}
        if(const auto* settings=ImGui::FindWindowSettingsByID(ImHashStr(name.c_str()))){id=settings->DockId;return true;}return false;
    };
    if(!visible){
        hidden_log_docks_.clear();
        for(const auto& name:names){unsigned id=0;if(dock_id(name,id))hidden_log_docks_[name]=id;}
        if(const auto* manager=ImGui::FindWindowByName(names.front().c_str()))if(const auto* node=manager->DockNode)
            if(node->ParentNode&&node->ParentNode->SplitAxis==ImGuiAxis_Y&&node->ParentNode->Size.y>0)
                log_height_ratio_=std::clamp(node->Size.y/node->ParentNode->Size.y,.1f,.75f);
    }else{
        ImGuiID fallback=0,root=0;
        for(const auto& name:names){
            unsigned id=0;const auto found=hidden_log_docks_.find(name);
            const bool known=found!=hidden_log_docks_.end()?(id=found->second,true):dock_id(name,id);
            if(known&&!id)continue; // A floating log window keeps its existing geometry.
            if(!id||!ImGui::DockBuilderGetNode(id)){
                if(!fallback)if(const auto* workspace=ImGui::FindWindowByName("Workspace###v2.workspace"))if(workspace->DockId){
                    ImGuiID center=workspace->DockId;ImGui::DockBuilderSplitNode(center,ImGuiDir_Down,log_height_ratio_,&fallback,&center);
                }
                id=fallback;
            }
            if(id){ImGui::DockBuilderDockWindow(name.c_str(),id);root=ImGui::DockNodeGetRootNode(ImGui::DockBuilderGetNode(id))->ID;if(name==names.front())log_dock_=id;}
        }
        if(root)ImGui::DockBuilderFinish(root);hidden_log_docks_.clear();
    }
    state.ui_logs=visible;
}
bool App::select_language_file(const std::string& file){
    std::string error;const auto directory=ui::language_directory();
    if(!ui::load_languages(directory,error,file)){toast(error,true);trace(error);return false;}
    state.ui_language_file=file;state.ui_language=file=="english.xml"?"en":"zh";
    if(!error.empty()){toast(error,true);trace(error);}return true;
}
void App::reload_language_file(){
    const bool ok=select_language_file(state.ui_language_file);
    if(ok)toast(ui::text("localization_reloaded",state.ui_language=="zh"));
}
bool App::move_page(int page,bool right) {
    if(!ui::move_page(state.ui_right_pages,state.ui_page,state.ui_right_page,page,right)){toast(ecmgui_v2::ui::text("keep_at_least_one_central_page",state.ui_language=="zh"));return false;}
    if(right){
        state.ui_right_visible=true;focus_right_=true;
        ImGuiID target=0;
        for(const auto& tab:right_docks)if(tab.second&&ImGui::DockBuilderGetNode(tab.second)){target=tab.second;break;}
        if(!target&&state.ui_inspector)if(const auto* inspector=ImGui::FindWindowByName("Inspector###v2.inspector"))target=inspector->DockId;
        if(!target)if(const auto* workspace=ImGui::FindWindowByName("Workspace###v2.workspace"))if(workspace->DockId){
            ImGuiID center=workspace->DockId;ImGui::DockBuilderSplitNode(center,ImGuiDir_Right,.34f,&target,&center);
        }
        if(target){ImGui::DockBuilderDockWindow(ui::right_window(page,false).c_str(),target);if(auto* node=ImGui::DockBuilderGetNode(target))ImGui::DockBuilderFinish(ImGui::DockNodeGetRootNode(node)->ID);}
    }
    toast(ui::text(right?"moved_right":"moved_left",state.ui_language=="zh"));return true;
}
void App::draw(ui::Settings& pending, const ui::Settings& active, const ui::Fonts& fonts) {
    if(state.ui_page==5){state.ui_page=drawn_center_page_>=0&&drawn_center_page_!=5?drawn_center_page_:0;open_settings();}
    if(ImGui::GetIO().KeyCtrl&&ImGui::IsKeyPressed(ImGuiKey_Comma,false))open_settings();
    navigation_icon_count=0;
    view_control=view_logs_control=view_save_control={};view_menu_open=view_save_separator=false;
    view_item_keys.clear();
    language_control={};language_items[0]=language_items[1]={};language_file_controls.clear();
    worker_speed_samples=workspace_chart_count=0;worker_speed_min=worker_speed_max={};
    worker_chart_rects.clear();worker_chart_metrics.clear();worker_chart_toggle={};
    for(auto& control:exit_controls)control={};
    if(status!=last_status){last_status=status;toast(status);}
    while(!toasts.empty()&&toasts.front().until<GetTickCount64())toasts.pop_front();
    const bool zh = fonts.chinese;
    if (ImGui::BeginMainMenuBar()) {
        ImGui::Text("ECM-ELF %s", version); ImGui::Separator();
        view_menu_open=ImGui::BeginMenu(ecmgui_v2::ui::text("menu.view",zh));
        view_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
        if(view_menu_open){
            const auto menu_item=[&](const char* key,bool selected=false){view_item_keys.push_back(key);return ImGui::MenuItem(ui::text(key,zh),nullptr,selected);};
            if(menu_item("menu.inspector",state.ui_inspector)){state.ui_inspector=!state.ui_inspector;reset_layout=true;}
            if(menu_item("menu.logs",state.ui_logs))set_logs_visible(!state.ui_logs);
            view_logs_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
            if(menu_item("menu.title_bar",state.window_title_bar))state.window_title_bar=!state.window_title_bar;
            ImGui::Separator();view_save_separator=true;
            if(menu_item("menu.save_layout_config"))save_layout_config();
            view_save_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu(ecmgui_v2::ui::text("menu.language",zh))){
            if(ImGui::IsWindowAppearing())ui::refresh_language_files();
            const auto files=ui::language_files();
            for(const auto& file:files){const auto label=ui::language_label(file)+"###language."+file;if(ImGui::MenuItem(label.c_str(),nullptr,state.ui_language_file==file))select_language_file(file);}
            if(ImGui::MenuItem(ui::text("reload_localization",zh)))reload_language_file();
            ImGui::EndMenu();
        }
        if(ImGui::MenuItem(ui::text("settings",zh)))open_settings();
        if (ImGui::MenuItem(ecmgui_v2::ui::text("menu.notices",zh))) ImGui::OpenPopup("Notices###v2.notices");
        if (ImGui::BeginPopup("Notices###v2.notices")) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize()*34);
            const auto begin = messages.size() > 20 ? messages.size()-20 : 0;
            for (size_t i=begin;i<messages.size();++i) ImGui::TextUnformatted(messages[i].c_str());
            ImGui::PopTextWrapPos(); ImGui::EndPopup();
        }
        title_drag_min=title_drag_max={};
        if(state.window_title_bar){
            if(ImGui::GetMainViewport()->WorkSize.x>=1000*active.scale)ImGui::TextDisabled("%s",ui::message(status,zh).c_str());
        }else{
            const float h=ImGui::GetFrameHeight(),spacing=ImGui::GetStyle().ItemSpacing.x;
            const float right=ImGui::GetWindowWidth()-ImGui::GetStyle().WindowPadding.x-3*h-3*spacing;
            ImGui::InvisibleButton("window-drag",{std::max(4.f,right-ImGui::GetCursorPosX()),h});title_drag_min=ImGui::GetItemRectMin();title_drag_max=ImGui::GetItemRectMax();
            const auto viewport=ImGui::GetMainViewport();const auto hwnd=static_cast<HWND>(viewport->PlatformHandleRaw?viewport->PlatformHandleRaw:viewport->PlatformHandle);
            const char* labels[]{"##window-min","##window-max","##window-close"};
            const char* tips[]{ecmgui_v2::ui::text("window.minimize",zh),ecmgui_v2::ui::text("window.maximize_restore",zh),ecmgui_v2::ui::text("window.close",zh)};
            for(int i=0;i<3;++i){ImGui::SameLine();if(i==2)ImGui::PushStyleColor(ImGuiCol_ButtonHovered,ui::palette(active.light).danger);
                if(ImGui::Button(labels[i],{h,h}))PostMessageW(hwnd,i==2?WM_CLOSE:WM_SYSCOMMAND,i==0?SC_MINIMIZE:i==1?(IsZoomed(hwnd)?SC_RESTORE:SC_MAXIMIZE):0,0);
                title_controls[i]={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
                const auto center=title_controls[i];const float r=4*active.scale;auto draw=ImGui::GetWindowDrawList();const auto ink=ImGui::GetColorU32(ImGuiCol_Text);
                if(i==0)draw->AddLine({center.x-r,center.y},{center.x+r,center.y},ink,active.scale);
                else if(i==1)draw->AddRect({center.x-r,center.y-r},{center.x+r,center.y+r},ink,0,0,active.scale);
                else{draw->AddLine({center.x-r,center.y-r},{center.x+r,center.y+r},ink,active.scale);draw->AddLine({center.x-r,center.y+r},{center.x+r,center.y-r},ink,active.scale);}
                if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",tips[i]);if(i==2)ImGui::PopStyleColor();
            }
        }
        ImGui::EndMainMenuBar();
    }
    const auto dock = ImGui::DockSpaceOverViewport(ImGui::GetID("v2.dock"), ImGui::GetMainViewport());
    const bool wide = ImGui::GetMainViewport()->WorkSize.x >= 1100 * active.scale;
    const auto node = ImGui::DockBuilderGetNode(dock);
    if (reset_layout || (!layout_ready_ && (!node || !node->ChildNodes[0])) || (layout_ready_ && layout_wide_ != wide)) {
        build_layout(dock, active.scale); reset_layout = false;
    }
    layout_ready_ = true; layout_wide_ = wide;
    begin_window(ecmgui_v2::ui::text("navigation.title",zh),nullptr,active); fonts.push(ui::FontRole::Body);
    const bool compact = ImGui::GetContentRegionAvail().x < 110 * active.scale;
    const auto item_center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    for(auto& pos:nav_items)pos={};for(auto& pos:right_items)pos={};
    nav_drag_page=-1;nav_return={};
    const auto nav_palette=ui::palette(active.light);const float nav_scale=ImGui::GetFontSize()/15.f;
    nav_rounding=8*nav_scale;nav_item_height=std::max(40*nav_scale,ImGui::GetTextLineHeight()+16*nav_scale);nav_item_gap=8*nav_scale;
    size_t nav_count=1;for(int i=0;i<8;++i)if(i!=5&&!ui::pinned(state.ui_right_pages,i))++nav_count;
    const float nav_top=ImGui::GetCursorPosY(),nav_bottom=ImGui::GetWindowSize().y-ImGui::GetStyle().WindowPadding.y;
    const float available_height=nav_bottom-nav_top,drop_height=ImGui::GetFrameHeight()+8*nav_scale,row_count=static_cast<float>(nav_count);
    const bool short_nav=available_height<row_count*(nav_item_height+nav_item_gap)+drop_height+60*nav_scale;
    if(short_nav){nav_item_gap=4*nav_scale;nav_item_height=std::max(ImGui::GetTextLineHeight()+4*nav_scale,std::min(nav_item_height,(available_height-drop_height)/row_count-nav_item_gap));}
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,nav_rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,0);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,{0,.5f});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{12*nav_scale,8*nav_scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,nav_item_gap});
    const auto order=state.ui_nav_order;
    for(char id:order){const int i=id-'0';
        if(i==5||ui::pinned(state.ui_right_pages,i))continue;
        ImGui::PushID(i);
        if(ui::navigation_button("##page",ui::page_name(i,zh),ui::page_icon(i),state.ui_page==i,nav_palette,nav_scale,nav_item_height))state.ui_page=i;
        ++navigation_icon_count;nav_items[i]=item_center();
        const auto lo=ImGui::GetItemRectMin(),hi=ImGui::GetItemRectMax();
        if(ImGui::IsItemActive()&&ImGui::GetIO().MouseDownDuration[0]>=.32f&&ImGui::BeginDragDropSource()){
            nav_drag_page=i;ImGui::SetDragDropPayload("V2_PAGE",&i,sizeof(i));ImGui::TextUnformatted(ui::page_name(i,zh));ImGui::EndDragDropSource();}
        if(ImGui::BeginDragDropTarget()){
            if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE",ImGuiDragDropFlags_AcceptBeforeDelivery))if(payload->DataSize==sizeof(int)){
                const bool after=ImGui::GetIO().MousePos.y>nav_items[i].y;const float y=after?hi.y:lo.y;
                ImGui::GetWindowDrawList()->AddLine({lo.x,y},{hi.x,y},ImGui::GetColorU32(nav_palette.focus),2*nav_scale);
                if(payload->IsDelivery())ui::reorder_page(state.ui_nav_order,*static_cast<const int*>(payload->Data),i,after);
            }ImGui::EndDragDropTarget();}
        if(ImGui::BeginPopupContextItem("location")){if(ImGui::MenuItem(ecmgui_v2::ui::text("move_to_right_tabs",zh)))move_page(i,true);ImGui::EndPopup();}
        ImGui::PopID();
    }
    ImGui::PopStyleVar(5);
    ImGui::Separator();ImGui::Button(compact?">###right-drop":ecmgui_v2::ui::text("navigation.right_tabs",zh),{-1,0});
    nav_drop=item_center();
    if(ImGui::BeginDragDropTarget()){if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE"))if(payload->DataSize==sizeof(int))move_page(*static_cast<const int*>(payload->Data),true);ImGui::EndDragDropTarget();}
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("navigation.move_right_hint",zh));
    if(!compact&&!short_nav)ui::secondary_text(ecmgui_v2::ui::text("navigation.reorder_hint",zh));
    const float settings_y=ImGui::GetWindowSize().y-ImGui::GetStyle().WindowPadding.y-nav_item_height;
    ImGui::SetCursorPosY(std::max(nav_top,settings_y));
    if(ui::navigation_button("##open-settings",ui::page_name(5,zh),ui::Icon::Settings,settings_open,nav_palette,nav_scale,nav_item_height))open_settings();
    settings_entry=nav_items[5]=item_center();++navigation_icon_count;
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("settings_shortcut",zh));
    nav_padding=ImGui::GetStyle().WindowPadding.x;nav_scroll={ImGui::GetScrollX(),ImGui::GetScrollY()};nav_size=ImGui::GetWindowSize();
    nav_scrollbars={ImGui::GetCurrentWindow()->ScrollbarX?1.f:0.f,ImGui::GetCurrentWindow()->ScrollbarY?1.f:0.f};
    ImGui::PopFont(); end_window();
    begin_window(ecmgui_v2::ui::text("workspace.title",zh),nullptr,active);
    workspace_docked=ImGui::IsWindowDocked();
    if(drawn_center_page_!=state.ui_page){ImGui::SetScrollY(0);drawn_center_page_=state.ui_page;}
    fonts.push(ui::FontRole::Body);
    ImGui::PushTextWrapPos(0); draw_page(active, fonts); ImGui::PopTextWrapPos(); ImGui::PopFont(); end_window();
    if (state.ui_inspector && wide) { begin_window(ecmgui_v2::ui::text("inspector.title",zh), &state.ui_inspector,active); draw_inspector(fonts); end_window(); }
    draw_log_windows(active,fonts);
    right_close_controls.clear();right_docks.clear();right_container_present=false;
    const auto right_order=state.ui_nav_order;std::vector<int> closed_pages;
    for(char id:right_order){const int page=id-'0';if(!ui::pinned(state.ui_right_pages,page))continue;
        if(focus_right_&&state.ui_right_page==page)ImGui::SetNextWindowFocus();
        bool open=true;const auto name=ui::right_window(page,zh);
        const bool visible=begin_window(name.c_str(),&open,active);
        if(visible&&open){
            if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))state.ui_right_page=page;
            fonts.push(ui::FontRole::Body);ImGui::PushTextWrapPos(0);draw_page(active,fonts,page);ImGui::PopTextWrapPos();ImGui::PopFont();
        }
        // Audit actual native dock tab positions; there is no nested Tools shell.
        const auto window=ImGui::GetCurrentWindow();
        right_docks[page]=window->DockId;
        if(window->DockNode&&window->DockNode->TabBar){
            const auto bar=window->DockNode->TabBar;
            if(const auto tab=ImGui::TabBarFindTabByID(bar,window->TabId)){
                const float x=bar->BarRect.Min.x+tab->Offset-bar->ScrollingAnim;
                right_items[page]={x+tab->Width*.4f,(bar->BarRect.Min.y+bar->BarRect.Max.y)*.5f};
                right_close_controls[page]={x+tab->Width-bar->FramePadding.x-ImGui::GetFontSize()*.5f,bar->BarRect.Min.y+bar->FramePadding.y+ImGui::GetFontSize()*.5f};
            }
        }
        end_window();if(!open)closed_pages.push_back(page);
    }
    focus_right_=false;
    for(int page:closed_pages)move_page(page,false);
    if(const auto old=ImGui::FindWindowByName("Tools###v2.tools"))right_container_present=old->Active;
    if (test_float) {
        ImGui::SetNextWindowPos({ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x + 30, ImGui::GetMainViewport()->Pos.y + 30}, ImGuiCond_FirstUseEver);
        begin_window("Detached lifecycle probe###v2.float", &test_float,active);
        probe_dpi = ImGui::GetWindowViewport()->DpiScale; probe_font = ImGui::GetFontSize(); probe_padding = ImGui::GetStyle().FramePadding.y;
        ImGui::TextUnformatted("Platform viewport probe"); end_window();
    }
    draw_settings_window(active,fonts);
    draw_exit(active,fonts);
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
        pending.font_size=static_cast<float>(state.ui_font_size);
        pending.primary_offset=state.ui_primary_font_offset;pending.secondary_offset=state.ui_secondary_font_offset;
        if (state.ui_scale > 0) pending.scale = state.ui_scale;
    }
}
std::string App::worker_summary(const ecmgui::WorkerView& worker,bool zh)const {
    const auto task=ecmgui::task_caption(worker.task,ui::text("worker.known_factors",zh),"assignment ID",ui::text("worker.no_task",zh));
    std::string summary=std::string(ui::text("worker.task_label",zh))+task;
    if(worker.stage2){
        const auto ratio=worker_b2_ratios_.find(worker.index);
        summary+="\n"+ui::worker_construction(worker,ratio==worker_b2_ratios_.end()?0:ratio->second,zh);
        char phase[512];std::snprintf(phase,sizeof(phase),ui::text("logs.phase_only",zh),ui::stage2_phase_name(worker.stage2_status,zh).c_str());
        summary+="\n"+std::string(phase);
    }
    return summary;
}
void App::draw_workers(const ui::Settings& active, const ui::Fonts& fonts) {
    body_font_height=ImGui::GetFontSize();const auto* body_glyph=ImGui::GetFontBaked()->FindGlyph('M');body_glyph_height=body_glyph->Y1-body_glyph->Y0;
    const bool zh=fonts.chinese;
    const auto p = ui::palette(active.light);
    const float toolbar_top=ImGui::GetCursorScreenPos().y;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*active.scale,4*active.scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6*active.scale,6*active.scale});
    fonts.push(ui::FontRole::Primary);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ecmgui_v2::ui::text("worker.title",zh));ImGui::PopFont();
    if(ImGui::IsItemHovered()&&!config_path.empty())ImGui::SetTooltip("%s",config_path.c_str());
    if(!ready){ui::secondary_text(ecmgui_v2::ui::text("worker.open_workflow_hint",zh));ImGui::PopStyleVar(2);return;}
    const auto title_end=ImGui::GetItemRectMax();
    const char* start_label=ui::text("action.start_all",zh);const char* stop_label=ui::text("action.stop_safely",zh);
    const float stop_width=ImGui::CalcTextSize(stop_label).x+ImGui::GetFontSize()*19/15,stop_height=ImGui::GetFontSize();
    fonts.push(ui::FontRole::Primary);
    const float button_width=std::max(stop_width,ImGui::CalcTextSize(start_label).x+ImGui::GetFontSize()*19/15)+16*active.scale;
    const float button_height=std::max(24*active.scale,std::max(stop_height,ImGui::GetFontSize())+8*active.scale);ImGui::PopFont();
    const float toolbar_right=ImGui::GetCursorScreenPos().x+ImGui::GetContentRegionAvail().x,gap=ImGui::GetStyle().ItemSpacing.x;
    if(title_end.x+2*button_width+2*gap<=toolbar_right)ImGui::SameLine();
    if (ui::button("start-all", start_label, ui::ButtonKind::Primary, p, active.scale,false,&primary_font_height,true,button_width,button_height,ui::Icon::Play))
        for (const auto& worker : controller.workers()) if(!start(worker.index))toast(status,true);
    start_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    start_button_size=ImGui::GetItemRectSize();
    if(ImGui::GetItemRectMax().x+gap+button_width<=toolbar_right)ImGui::SameLine();
    if (ui::button("stop-all", stop_label, ui::ButtonKind::Secondary, p, active.scale,false,&secondary_button_font_height,true,button_width,button_height,ui::Icon::Stop)) {controller.stop_all();toast(ecmgui_v2::ui::text("worker.safe_stop_requested",fonts.chinese));}
    stop_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    stop_button_size=ImGui::GetItemRectSize();
    worker_toolbar_height=ImGui::GetCursorScreenPos().y-toolbar_top;ImGui::PopStyleVar(2);
    for (const auto& worker : controller.workers()) if (worker.p95.valid) {
        const bool pending = worker.p95.level == ecmgui::P95Notice::Level::Pending;
        ImGui::TextColored(pending ? ImVec4(1,.35f,.4f,1) : ImVec4(.8f,.65f,.35f,1), "Prime95 %s: %s %s", worker.name.c_str(),worker.p95.error.c_str(),worker.p95.note.c_str());
    }
    worker_select_controls.clear();worker_number_controls.clear();worker_action_controls.clear();worker_table_rows=0;
    const float scale=ImGui::GetFontSize()/15.f;
    const auto left=ImGui::GetCursorScreenPos();const float visible_width=ImGui::GetContentRegionAvail().x;
    const float cell_pad=(visible_width>=800*scale?8.f:3.f)*scale;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{cell_pad,4*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*scale,4*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6*scale,2*scale});
    const float control_height=std::max(24*scale,ImGui::GetTextLineHeight()+8*scale);
    // Proportional columns retain one information row at every width.
    // Text elides within its cell; the full value remains available on hover.
    float widths[]{28,72,112,80,84,150,64,76,76};
    for(const auto& worker:controller.workers()){
        const auto progress=ui::worker_progress(worker,zh);
        const auto label=ui::message(worker.stop_requested?"Stopping":ecmgui::state_name(worker.state()),zh);
        widths[0]=std::max(widths[0],ImGui::CalcTextSize(std::to_string(worker.index).c_str()).x/scale);
        widths[3]=std::max(widths[3],ImGui::CalcTextSize(label.c_str()).x/scale+26);
        widths[6]=std::max(widths[6],ImGui::CalcTextSize(progress.speed.c_str()).x/scale);
        widths[7]=std::max(widths[7],ImGui::CalcTextSize(progress.eta.c_str()).x/scale);
    }
    float preferred_width=18*cell_pad;for(float width:widths)preferred_width+=width*scale;
    const bool compress_columns=visible_width<preferred_width;
    worker_table_scroll_max_x=0;worker_field_overflow=0;worker_table_columns=0;
    worker_summary_rects.clear();worker_summary_texts.clear();
    const float padding=ImGui::GetStyle().CellPadding.y*2;float content_height=ImGui::GetTextLineHeight()+padding+4*scale;
    for(const auto& worker:controller.workers()){
        const auto summary=worker_summary(worker,zh);
        // Selectable's baseline follows the button's FramePadding.y in each row.
        content_height+=control_height+padding+ImGui::GetStyle().FramePadding.y;
        content_height+=ImGui::CalcTextSize(summary.c_str(),nullptr,false,std::max(1.f,visible_width-24*scale)).y+padding;
    }
    // Reserve child viewport padding so every row remains visible.
    content_height+=ImGui::GetStyle().ScrollbarSize;
    // Keep every worker row in the table. Only the surrounding workspace scrolls
    // vertically; the same elastic columns are used on narrow surfaces.
    const float table_height=content_height;
    // New column order gets a fresh table identity instead of reusing old widths.
    if (ImGui::BeginTable("workers-elastic-v4",9,ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerH|ImGuiTableFlags_BordersOuterH|ImGuiTableFlags_ScrollY|ImGuiTableFlags_PadOuterX|ImGuiTableFlags_SizingStretchProp|ImGuiTableFlags_NoSavedSettings,{0,table_height})) {
        worker_table_scroll_max_x=ImGui::GetScrollMaxX();
        worker_table_scroll_max_y=ImGui::GetScrollMaxY();worker_table_height=ImGui::GetWindowSize().y;worker_table_content_height=ImGui::GetCurrentWindow()->ContentSize.y;
        const auto table_origin=ImGui::GetWindowPos();worker_table_data_height=ImGui::GetCurrentWindow()->InnerRect.GetHeight();
        struct SummaryRow {ImVec2 min,max,pos;float wrap;std::string text;};
        std::vector<SummaryRow> summaries;auto* table_draw=ImGui::GetWindowDrawList();
        const auto table_clip_min=ImGui::GetWindowDrawList()->GetClipRectMin(),table_clip_max=ImGui::GetWindowDrawList()->GetClipRectMax();
        float body_top=table_clip_min.y;
        const char* labels[]{"#",ecmgui_v2::ui::text("worker.action",zh),ecmgui_v2::ui::text("worker_name",zh),ecmgui_v2::ui::text("worker.state",zh),ecmgui_v2::ui::text("worker.device_engine",zh),ecmgui_v2::ui::text("worker.progress",zh),ecmgui_v2::ui::text("metrics.seconds_per_curve",zh),"ETA",ecmgui_v2::ui::text("worker.completed_hits",zh)};
        for(int i=0;i<9;++i){const bool stretch=i!=0&&(compress_columns||i==5);
            ImGui::TableSetupColumn(labels[i],stretch?ImGuiTableColumnFlags_WidthStretch:ImGuiTableColumnFlags_WidthFixed,stretch?widths[i]:widths[i]*scale);}
        ImGui::TableHeadersRow();
        for(int i=0;i<9;++i)if(ImGui::TableGetColumnFlags(i)&ImGuiTableColumnFlags_IsVisible)++worker_table_columns;
        const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
        for (const auto& worker : controller.workers()) {
            ++worker_table_rows;
            ImGui::PushID(worker.index);ImGui::TableNextRow(0,control_height+padding);
            const auto st=worker.state();const auto progress=ui::worker_progress(worker,zh);
            const auto value_cell=[&](const std::string& value,const ImVec4& color,bool selectable=false){
                const auto pos=ImGui::GetCursorScreenPos();const float width=std::max(1.f,ImGui::GetContentRegionAvail().x);
                const auto shown=ui::fitted_label(value.c_str(),width);
                if(selectable){
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{0,ImGui::GetStyle().ItemSpacing.y});
                    if(ImGui::Selectable("##worker",state.ui_worker==worker.index,0,{width,control_height})){state.ui_worker=worker.index;log_active_worker=worker.index;log_last_worker_=-1;}
                    ImGui::PopStyleVar();
                }
                else ImGui::Dummy({width,control_height});
                if(ImGui::IsItemHovered()&&(selectable||shown!=value))ImGui::SetTooltip("%s",value.c_str());
                auto* draw=ImGui::GetWindowDrawList();draw->PushClipRect(pos,{pos.x+width,pos.y+control_height},true);
                draw->AddText({pos.x,pos.y+(control_height-ImGui::GetFontSize())*.5f},ImGui::GetColorU32(color),shown.c_str());draw->PopClipRect();
                worker_field_overflow=std::max(worker_field_overflow,ImGui::GetItemRectMax().x-pos.x-width);
            };
            ImGui::TableSetColumnIndex(0);
            if(summaries.empty())body_top=std::max(body_top,ImGui::GetCursorScreenPos().y-ImGui::GetStyle().CellPadding.y);
            const float main_row_top=ImGui::GetCursorScreenPos().y;
            value_cell(std::to_string(worker.index),p.muted);worker_number_controls[worker.index]=center();
            ImGui::TableSetColumnIndex(1);
            const bool busy=worker.running()||st==ecmgui::WorkerRunState::Restarting;
            const auto action_label=worker.stop_requested?ui::text("status.stopping",zh):busy?ui::text("action.stop",zh):ui::text("action.start",zh);
            const float action_width=std::max(1.f,std::min(ImGui::GetContentRegionAvail().x,ImGui::CalcTextSize(action_label).x+35*scale));
            const auto action_shown=ui::fitted_label(action_label,action_width-8*scale);
            if(ui::button("worker-action",action_shown.c_str(),busy?ui::ButtonKind::Danger:ui::ButtonKind::Secondary,p,scale,worker.stop_requested,nullptr,true,action_width,0,busy||worker.stop_requested?ui::Icon::Stop:ui::Icon::Play)){
                if(busy){controller.request_stop(worker.index);toast("Safe stop requested: "+worker.name);}else if(!start(worker.index))toast(status,true);
            }
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)&&action_shown!=action_label)ImGui::SetTooltip("%s",action_label);
            worker_action_controls[worker.index]=center();worker_button_height=ImGui::GetItemRectSize().y;
            ImGui::TableSetColumnIndex(2);
            value_cell(worker.name,p.fg,true);worker_select_controls[worker.index]=center();
            ImGui::TableSetColumnIndex(3);
            const auto state_kind=st==ecmgui::WorkerRunState::Error?ui::Status::Danger:worker.stop_requested||st==ecmgui::WorkerRunState::Restarting?ui::Status::Warning:worker.running()?ui::Status::Success:ui::Status::Info;
            const auto state_label=worker.stop_requested?std::string(ui::text("status.stopping",zh)):ui::message(ecmgui::state_name(st),zh);
            ui::status_badge(state_label.c_str(),state_kind,p,scale,std::max(1.f,ImGui::GetContentRegionAvail().x));
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",state_label.c_str(),worker.state_text().c_str());
            ImGui::TableSetColumnIndex(4);ImGui::AlignTextToFramePadding();
            const auto device=worker.execution_target=="gpu"?"GPU"+std::to_string(worker.device):std::string("CPU");
            value_cell(device+" / S"+std::to_string(worker.stage2?2:1)+(worker.duplicate_device?"*":""),worker.stage2?p.stage2:p.stage1);
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s | Stage %d | %s%s",device.c_str(),worker.stage2?2:1,worker.curve_model.c_str(),worker.duplicate_device?ecmgui_v2::ui::text("worker.shared_gpu_hint",zh):"");
            ImGui::TableSetColumnIndex(5);ImGui::PushStyleColor(ImGuiCol_PlotHistogram,worker.stage2?p.stage2:p.stage1);
            const auto progress_shown=ui::fitted_label(progress.label.c_str(),ImGui::GetContentRegionAvail().x-8*scale);
            ImGui::ProgressBar(progress.fraction,{-1,control_height},progress_shown.c_str());ImGui::PopStyleColor();
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",progress.label.c_str());
            ImGui::TableSetColumnIndex(6);value_cell(progress.speed,p.fg);
            ImGui::TableSetColumnIndex(7);value_cell(progress.eta,p.fg);
            ImGui::TableSetColumnIndex(8);value_cell(std::to_string(worker.tasks_done)+" / "+std::to_string(worker.hits),p.fg);
            const auto summary=worker_summary(worker,zh);worker_summary_texts[worker.index]=summary;
            ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);
            const auto row=ImGui::GetCursorScreenPos();worker_row_height=row.y-main_row_top;
            const ImVec2 pos{std::max(left.x+12*scale,row.x),row.y};const float right=left.x+visible_width-12*scale;
            const float wrap=std::max(1.f,right-pos.x),height=ImGui::CalcTextSize(summary.c_str(),nullptr,false,wrap).y;
            ImGui::Dummy({1,height});
            summaries.push_back({{table_clip_min.x,row.y-ImGui::GetStyle().CellPadding.y},{table_clip_max.x,row.y+height+ImGui::GetStyle().CellPadding.y},pos,wrap,summary});
            if(ImGui::IsWindowHovered()&&ImGui::IsMouseHoveringRect(pos,{right,pos.y+height})&&!worker.task.empty())ImGui::SetTooltip("%s",worker.task.c_str());
            ImGui::PopID();
        }
        worker_table_required_height=summaries.empty()?0:summaries.back().max.y-table_origin.y;
        ImGui::EndTable();
        // The table adds resize borders at EndTable. Paint spanning rows only after
        // its channels merge, in the same child draw list and body clip rectangle.
        table_draw->PushClipRect({table_clip_min.x,body_top},table_clip_max,true);
        worker_summary_rects.clear();
        if(!summaries.empty())summaries.back().max.y=std::max(summaries.back().max.y,table_clip_max.y);
        for(const auto& row:summaries){
            table_draw->AddRectFilled(row.min,row.max,ImGui::GetColorU32(p.surface));
            table_draw->AddText(ImGui::GetFont(),ImGui::GetFontSize(),row.pos,ImGui::GetColorU32(p.muted),row.text.c_str(),nullptr,row.wrap);
            worker_summary_rects.push_back({row.min,row.max});
        }
        table_draw->PopClipRect();
    }
    ImGui::PopStyleVar(3);
    draw_worker_speed(fonts);
}

void App::draw_worker_speed(const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;
    const auto p=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
    const float scale=ImGui::GetFontSize()/15.f;
    worker_speed_min=ImGui::GetCursorScreenPos();
    const float right=worker_speed_min.x+ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ui::text("worker.curve_time",zh));
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(),right-ImGui::GetWindowPos().x-30*scale));
    if(ui::icon_button("##worker-chart-layout",state.ui_worker_chart_columns==1?ui::Icon::Grid:ui::Icon::List,p,scale))
        state.ui_worker_chart_columns=state.ui_worker_chart_columns==1?2:1;
    worker_chart_toggle={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text(state.ui_worker_chart_columns==1?"worker_charts_two_columns":"worker_charts_list",zh));
    // Keep each chart above the shared chart widget's minimum width. Restore
    // the saved two-column choice automatically when the workspace grows.
    worker_chart_columns=state.ui_worker_chart_columns==2&&ImGui::GetContentRegionAvail().x>=360*scale?2:1;
    const auto draw_chart=[&](const auto& worker){
        ImGui::PushID(worker.index);++workspace_chart_count;
        ImGui::BeginGroup();const auto top=ImGui::GetCursorScreenPos();
        const auto title=worker.name+" / "+(worker.stage2?"Stage2":"Stage1");
        const auto shown=ui::fitted_label(title.c_str(),ImGui::GetContentRegionAvail().x);
        ImGui::TextUnformatted(shown.c_str());if(ImGui::IsItemHovered())ImGui::SetTooltip("#%d %s",worker.index,title.c_str());
        worker_speed_samples+=worker.hist_s_per_curve.size();ui::ChartMetrics metrics;
        if(!worker.hist_s_per_curve.empty()){
            std::vector<ui::ChartSeries> series{{ecmgui_v2::ui::text("metrics.seconds_per_curve",zh),worker.stage2?p.stage2:p.stage1,{}}};
            for(size_t i=0;i<worker.hist_s_per_curve.size();++i)series[0].points.push_back({static_cast<double>(i+1),worker.hist_s_per_curve[i]});
            const auto value=ui::worker_progress(worker,zh).speed+" s/curve";
            ui::ChartFormat format;format.legend_controls=false;format.show_y_ticks=false;format.show_axis_titles=false;
            format.integer_x=true;format.corner_text=value.c_str();format.y_format="%.2f s/curve";format.x_format="%.0f";format.metrics=&metrics;
            ui::chart("workspace-speed",series,ecmgui_v2::ui::text("chart.sample",zh),"s/curve",zh,0,88,format);
        }else {ImGui::PushTextWrapPos(0);ImGui::TextDisabled("%s",ecmgui_v2::ui::text("worker.waiting_curve_time",zh));ImGui::PopTextWrapPos();}
        ImGui::EndGroup();worker_chart_rects.push_back({top,ImGui::GetItemRectMax()});worker_chart_metrics.push_back(metrics);ImGui::PopID();
    };
    if(worker_chart_columns==2){
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{6*scale,4*scale});
        if(ImGui::BeginTable("worker-speed-grid",2,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_NoSavedSettings)){
            for(const auto& worker:controller.workers()){ImGui::TableNextColumn();draw_chart(worker);}
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }else{
        for(const auto& worker:controller.workers())draw_chart(worker);
    }
    worker_speed_max={right,ImGui::GetCursorScreenPos().y};
}

void App::draw_exit(const ui::Settings& active,const ui::Fonts& fonts) {
    using Phase=ecmgui::WorkerController::ExitPhase;
    const auto phase=controller.exit_phase();if(phase==Phase::Idle)return;
    const bool zh=fonts.chinese;
    const auto p=ui::palette(active.light);const auto* vp=ImGui::GetMainViewport();
    const float width=std::max(240.f,std::min(600*active.scale,vp->WorkSize.x-24*active.scale));
    bool stage1=false,stage2=false;
    for(const auto& w:controller.workers())if(w.running()){stage1|=!w.stage2;stage2|=w.stage2;}
    const auto explain=stage1&&stage2?ecmgui_v2::ui::text("exit.mixed_stage_stop_hint",zh):
        stage2?ecmgui_v2::ui::text("exit.stage2_stop_hint",zh):
        ecmgui_v2::ui::text("exit.stage1_stop_hint",zh);
    const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    fonts.push(ui::FontRole::Body);
    const char* popup=ecmgui_v2::ui::text("exit.confirm_title",zh);
    if(phase==Phase::Confirm){
        if(!ImGui::IsPopupOpen(popup))ImGui::OpenPopup(popup);
        ImGui::SetNextWindowViewport(vp->ID);ImGui::SetNextWindowPos(vp->GetCenter(),ImGuiCond_Always,{.5f,.5f});
        ImGui::SetNextWindowSize({width,0});
        if(ImGui::BeginPopupModal(popup,nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove)){
            ImGui::PushTextWrapPos(0);
            ImGui::Text(ecmgui_v2::ui::text("exit.running_workers_prompt",zh),running_count());
            ImGui::TextWrapped("%s",explain);ImGui::Separator();
            const bool keyboard=!ImGui::IsWindowAppearing()&&ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            const bool accept=ui::button("exit-stop",ecmgui_v2::ui::text("exit.stop_and_exit",zh),ui::ButtonKind::Primary,p,active.scale);
            exit_controls[0]=center();
            const bool cancel=ui::button("exit-cancel",ecmgui_v2::ui::text("exit.cancel",zh),ui::ButtonKind::Secondary,p,active.scale);
            exit_controls[1]=center();
            if(cancel||(keyboard&&ImGui::IsKeyPressed(ImGuiKey_Escape))){controller.cancel_close();ImGui::CloseCurrentPopup();}
            else if(accept||(keyboard&&(ImGui::IsKeyPressed(ImGuiKey_Enter)||ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)))){begin_exit();ImGui::CloseCurrentPopup();}
            ImGui::PopTextWrapPos();ImGui::EndPopup();
        }
    }
    if(controller.exit_phase()==Phase::Stopping){
        const auto now=GetTickCount64(),deadline=controller.exit_deadline();const double left=deadline>now?(deadline-now)/1000.:0;
        ImGui::SetNextWindowViewport(vp->ID);ImGui::SetNextWindowPos(vp->GetCenter(),ImGuiCond_Always,{.5f,.5f});ImGui::SetNextWindowSize({width,0});
        ImGui::Begin(ecmgui_v2::ui::text("exit.stopping_title",zh),nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushTextWrapPos(0);
        ImGui::TextWrapped("%s",explain);ImGui::Text(ecmgui_v2::ui::text("exit.remaining_seconds",zh),left);
        ImGui::TextColored(p.warning,"%s",ecmgui_v2::ui::text("exit.deadline_hint",zh));
        ImGui::Separator();
        for(const auto& w:controller.workers()){
            ImGui::TextColored(w.running()?p.warning:p.success,"#%d %s | %s | %s",w.index,w.name.c_str(),w.stage2?"Stage2":"Stage1",w.running()?(w.stage2?ecmgui_v2::ui::text("exit.finishing_curve",zh):ecmgui_v2::ui::text("exit.waiting_checkpoint",zh)):ecmgui_v2::ui::text("status.stopped",zh));
        }
        ImGui::Separator();
        const bool force=ui::button("exit-force",ecmgui_v2::ui::text("exit.force_exit",zh),ui::ButtonKind::Danger,p,active.scale);exit_controls[2]=center();
        if(force||(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)&&ImGui::IsKeyPressed(ImGuiKey_Escape))){trace("exit: forced by user without waiting for a fresh checkpoint");controller.force_close();}
        ImGui::PopTextWrapPos();ImGui::End();
    }
    ImGui::PopFont();
}

void App::draw_inspector(const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;
    fonts.push(ui::FontRole::Body);
    if (const auto worker = selected()) {
        ImGui::Text("%s / %s", worker->name.c_str(), ui::message(ecmgui::state_name(worker->state()),zh).c_str());
        ImGui::TextWrapped("%s", worker->task.c_str()); ImGui::Separator();
        if (worker->stage2 && worker->stage2_status.valid) {
            const auto& s=worker->stage2_status;
            ImGui::Text(ecmgui_v2::ui::text("worker.task_curves",zh),s.done,s.total);
            ImGui::Text(ecmgui_v2::ui::text("worker.phase_batch",zh),s.phase.c_str(),s.batch,s.batches);
            if(s.curve_model=="ed")ImGui::Text("Edwards sigma=%llu",s.sigma);
            else ImGui::Text("%s param=%d sigma=%llu",s.curve_model.c_str(),s.param,s.sigma);
            ImGui::Text("B1=%llu B2=%llu",s.b1,s.b2);
        }
        if (ImGui::Button(ecmgui_v2::ui::text("open_worker_folder",zh))) ecmgui::open_in_explorer(worker->working_dir);
        ImGui::TextWrapped("%s", worker->effective_config.c_str());
        ImGui::TextWrapped(ecmgui_v2::ui::text("worker.version",zh), worker->application_version.empty() ? ecmgui_v2::ui::text("unknown",zh) : worker->application_version.c_str());
        if (ImGui::Button(ecmgui_v2::ui::text("copy_command",zh))) ImGui::SetClipboardText(worker->command_line().c_str());
        ImGui::TextWrapped("%s", worker->command_line().c_str());
    } else ImGui::TextUnformatted(ecmgui_v2::ui::text("no_worker_selected",zh));
    ImGui::PopFont();
}
void App::draw_log_windows(const ui::Settings& active,const ui::Fonts& fonts){
    log_tab_controls.clear();log_docks.clear();log_raw_control=log_follow_control={};
    log_stage2_fraction=log_stage2_width=0;log_stage2_phase=-1;log_stage2_sources.clear();
    if(!state.ui_logs)return;
    const bool request=log_last_worker_!=state.ui_worker||log_active_worker<0;
    if(request){log_active_worker=selected()?selected()->index:0;log_last_worker_=state.ui_worker;}
    const int requested=log_active_worker;
    for(size_t i=0;i<=controller.workers().size();++i){
        const auto* worker=i?&controller.workers()[i-1]:nullptr;const int id=worker?worker->index:0;
        const auto label=worker?"#"+std::to_string(id)+" "+worker->name+"###v2.log.worker."+std::to_string(id):std::string(ui::text("logs.title",fonts.chinese));
        if(worker&&log_dock_)ImGui::SetNextWindowDockID(log_dock_,ImGuiCond_FirstUseEver);
        if(request&&id==requested)ImGui::SetNextWindowFocus();
        const bool visible=begin_window(label.c_str(),nullptr,active,ImGuiWindowFlags_NoFocusOnAppearing);auto* window=ImGui::GetCurrentWindow();
        if(!worker&&window->DockId)log_dock_=window->DockId;
        log_docks[id]=window->DockId;
        if(window->DockNode&&window->DockNode->TabBar){
            auto* bar=window->DockNode->TabBar;
            if(const auto* tab=ImGui::TabBarFindTabByID(bar,window->TabId)){
                const float x=bar->BarRect.Min.x+tab->Offset-bar->ScrollingAnim;
                log_tab_controls[id]={x+tab->Width*.5f,(bar->BarRect.Min.y+bar->BarRect.Max.y)*.5f};
            }
        }
        if(visible){
            if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)){
                log_active_worker=id;if(worker)state.ui_worker=id;log_last_worker_=state.ui_worker;
            }
            draw_logs(fonts,worker);
        }
        end_window();
    }
}
void App::draw_stage2_progress(const ecmgui::WorkerView& worker,const ui::Palette& p,bool zh){
    const auto& s=worker.stage2_status;auto& view=stage2_views_[worker.index];const auto now=GetTickCount64();
    const auto engine=stage2_engine_ids_.find(worker.index);
    const auto scope=ui::stage2_scope(worker,gpu_details.get(worker.device).uuid,engine==stage2_engine_ids_.end()?"unknown":engine->second);
    const auto tune=stage2_tunes_.find(worker.index);
    const auto identity=scope+"|"+std::to_string(s.current)+"|"+std::to_string(s.record)+"|"+std::to_string(s.sigma)+"|"+state.ui_stage2_timing_source+"|"+state.ui_stage2_tune_file+"|"+std::to_string(tune==stage2_tunes_.end()?0:tune->second.revision);
    if(view.identity!=identity){
        const auto phase_at=view.phase_at,batch_at=view.batch_at,batch=view.batch;const auto clock_curve=view.clock_curve;
        view=PhaseView{};view.identity=identity;view.phase=s.phase;view.clock_curve=clock_curve;
        view.phase_at=phase_at?phase_at:now;view.batch_at=batch_at?batch_at:now;view.batch=batch;
        const int mode=state.ui_stage2_timing_source=="previous"?1:state.ui_stage2_timing_source=="tune"?2:0;
        view.timing=ui::stage2_timing(stage2_history_.lookup(scope),tune==stage2_tunes_.end()?ui::no_phase_seconds():tune->second.value.lookup(s,gpu_details.get(worker.device).uuid),mode);
    }
    if(view.phase!=s.phase){view.phase=s.phase;view.phase_at=view.batch_at=now;view.batch=0;}
    if(view.batch!=s.batch){view.batch=s.batch;view.batch_at=now;}
    const int phase=ui::stage2_phase_index(s.phase);const auto weights=view.timing.fractions();double fraction=0;
    if(phase>=0){
        for(int i=0;i<phase;++i)fraction+=weights[i];double local=0;
        if(s.batches){
            local=std::clamp(double(s.batch)/double(s.batches),0.,1.);
            if(worker.running()&&s.batch_left>0&&std::isfinite(s.batch_left))local=std::min(.99,local+(1-local)*double(now-view.batch_at)/1000/s.batch_left);
        }else if(worker.running()&&view.timing.seconds[phase]>0)local=std::min(.95,double(now-view.phase_at)/1000/view.timing.seconds[phase]);
        fraction+=weights[phase]*local;
    }
    view.high_water=std::max(view.high_water,ui::stage2_bounded_fraction(s,fraction));
    char overlay[128];std::snprintf(overlay,sizeof(overlay),ui::text("stage2.curve_progress",zh),100.*view.high_water);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,p.stage2);ImGui::ProgressBar(view.high_water,{-1,ImGui::GetFrameHeight()},overlay);ImGui::PopStyleColor();
    const auto lo=ImGui::GetItemRectMin(),hi=ImGui::GetItemRectMax();auto* draw=ImGui::GetWindowDrawList();double boundary=0;
    for(size_t i=0;i<weights.size()-1;++i){boundary+=weights[i];if(boundary>0&&boundary<1){const float x=lo.x+float(boundary)*(hi.x-lo.x);draw->AddLine({x,lo.y+2},{x,hi.y-2},ImGui::GetColorU32(p.border));}}
    if(ImGui::IsItemHovered()){
        ImGui::BeginTooltip();
        if(!view.timing.any())ImGui::TextUnformatted(ui::text("stage2.source.placeholder",zh));
        else for(size_t i=0;i<weights.size();++i){const auto key=std::string("stage2.phase.")+ui::stage2_phase_keys[i];
            const auto source=ui::text(view.timing.sources[i]==1?"stage2.source.previous":view.timing.sources[i]==2?"stage2.source.tune":"stage2.source.unknown",zh);
            if(view.timing.seconds[i]>=0)ImGui::Text("%s | %.3f s | %.1f%% | %s",ui::text(key.c_str(),zh),view.timing.seconds[i],weights[i]*100,source);
            else ImGui::Text("%s | -- | %s",ui::text(key.c_str(),zh),source);
        }
        if(tune!=stage2_tunes_.end()&&!tune->second.value.notice.empty())ImGui::TextUnformatted(tune->second.value.notice.c_str());
        if(!stage2_history_.notice.empty())ImGui::TextUnformatted(stage2_history_.notice.c_str());ImGui::EndTooltip();
    }
    log_stage2_fraction=view.high_water;log_stage2_width=hi.x-lo.x;log_stage2_phase=phase;
    for(int source:view.timing.sources)log_stage2_sources+=char('0'+source);
}
void App::draw_logs(const ui::Fonts& fonts,const ecmgui::WorkerView* active_worker) {
    const bool zh=fonts.chinese;
    const auto p=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);const float scale=ImGui::GetFontSize()/15.f;
    fonts.push(ui::FontRole::Body);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,4*scale});
    const int log_id=active_worker?active_worker->index:0;
    if(active_worker){
        const auto progress=ui::worker_progress(*active_worker,zh);
        const auto task=ecmgui::task_caption(active_worker->task,ecmgui_v2::ui::text("worker.known_factors",zh),"assignment ID",ecmgui_v2::ui::text("worker.no_task",zh));
        ImGui::PushTextWrapPos(0);ImGui::TextColored(active_worker->stage2?p.stage2:p.stage1,ecmgui_v2::ui::text("logs.worker_summary",zh),active_worker->stage2?"Stage 2":"Stage 1",task.c_str(),progress.label.c_str(),progress.speed.c_str(),progress.eta_detail.c_str());ImGui::PopTextWrapPos();
        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",active_worker->task.c_str(),ui::worker_phase(*active_worker,zh).c_str());
        const auto shape=ui::worker_shape(*active_worker);
        if(!shape.empty()){
            // Keep the shape on one line; hover to read it in a narrow pane.
            ImGui::TextColored(p.stage2,"%s",shape.c_str());
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",shape.c_str());
        }else if(active_worker->stage2)ImGui::Dummy({0,ImGui::GetTextLineHeight()});
        if(active_worker->stage2){
            const auto& s=active_worker->stage2_status;const auto name=ui::stage2_phase_name(s,zh);
            char phase_line[512];std::snprintf(phase_line,sizeof(phase_line),ui::text("logs.phase_only",zh),name.c_str());
            std::string label=phase_line;const auto batch=ui::stage2_batch(s,zh);
            if(batch.visible)label+=" | "+batch.label+" | "+batch.remaining;
            ImGui::TextColored(p.stage2,"%s",label.c_str());if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",label.c_str(),s.phase.c_str());
            draw_stage2_progress(*active_worker,p,zh);
            if(active_worker->stop_requested)ImGui::TextColored(p.warning,"%s",ui::text("stop_after_the_current_stage2_curve",zh));
        }
    }
    auto found=log_views_.find(log_id);if(found==log_views_.end())found=log_views_.emplace(log_id,LogView{state.ui_raw_log,state.ui_follow_log,true}).first;
    auto& view=found->second;ImGui::PushID(log_id);
    if(ui::button("raw",ecmgui_v2::ui::text("raw_output",zh),view.raw?ui::ButtonKind::Selected:ui::ButtonKind::Secondary,p,scale)){view.raw=!view.raw;state.ui_raw_log=view.raw;view.jump=true;}
    log_raw_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SameLine();if(ui::button("follow",ecmgui_v2::ui::text("follow",zh),view.follow?ui::ButtonKind::Selected:ui::ButtonKind::Secondary,p,scale)){view.follow=!view.follow;state.ui_follow_log=view.follow;view.jump=view.follow;}
    log_follow_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(active_worker){
        ImGui::SameLine();if(ui::button("log-folder",ecmgui_v2::ui::text("open_log_folder",zh),ui::ButtonKind::Secondary,p,scale))ecmgui::open_in_explorer(ecmgui::path_dir(active_worker->log_file));
        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",active_worker->log_file.c_str());
        if(ImGui::GetContentRegionAvail().x>ImGui::CalcTextSize(ecmgui::file_name(active_worker->log_file).c_str()).x+16*scale){ImGui::SameLine();ImGui::TextDisabled("%s",ecmgui::file_name(active_worker->log_file).c_str());if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",active_worker->log_file.c_str());}
    }
    log_current_raw=view.raw;log_current_follow=view.follow;
    log_speed_samples=0;
    ImGui::BeginChild(view.raw?"raw-output":"event-output",{0,0},false,ImGuiWindowFlags_HorizontalScrollbar);
    log_view_height=ImGui::GetWindowSize().y;
    const bool follow=view.follow&&(view.jump||ImGui::GetScrollY()>=ImGui::GetScrollMaxY()-1.f);
    fonts.push(ui::FontRole::Body);
    log_uses_body=ImGui::GetFont()==fonts.body;const auto* log_glyph=ImGui::GetFontBaked()->FindGlyph('M');log_glyph_height=log_glyph->Y1-log_glyph->Y0;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,2*scale});
    log_font_height=ImGui::GetTextLineHeight();log_line_pitch=ImGui::GetTextLineHeightWithSpacing();
    const auto* lines=active_worker?(view.raw?&active_worker->raw:&active_worker->events):nullptr;
    const size_t count=lines?lines->size():messages.size();
    if(!count)ImGui::TextDisabled("%s",ecmgui_v2::ui::text("no_output_yet",zh));
    ImGuiListClipper clip;clip.Begin(static_cast<int>(count),log_line_pitch);
    while(clip.Step())for(int i=clip.DisplayStart;i<clip.DisplayEnd;++i){const auto& line=lines?(*lines)[static_cast<size_t>(i)]:messages[static_cast<size_t>(i)];
        ui::log_line(line,p);}
    if(follow)ImGui::SetScrollHereY(1);view.jump=false;log_scroll=ImGui::GetScrollY();
    ImGui::PopStyleVar();ImGui::PopFont();ImGui::EndChild();
    ImGui::PopID();ImGui::PopStyleVar();ImGui::PopFont();
}

void App::draw_page(const ui::Settings& active, const ui::Fonts& fonts,int page) {
    switch (page<0?state.ui_page:page) {
    case 0: draw_workers(active, fonts); break;
    case 1: draw_results(fonts.chinese); break;
    case 2: draw_generator(fonts.chinese); break;
    case 3: draw_probability(fonts.chinese); break;
    case 4: draw_gpu(fonts.chinese); break;
    case 5: open_settings(); break;
    case 7: draw_group_order(fonts.chinese); break;
    case 6: ui::state_strip(ui::palette(active.light),active.scale,fonts.chinese); break;
    }
}
}
