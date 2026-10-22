#include "app.h"
#include "version.h"
#include "ui/navigation.h"
#include "ui/i18n.h"
#include "ui/chart.h"
#include "ui/worker_presentation.h"
#include "ui/log_presentation.h"
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
    if(ui::language_directory().empty()){std::string error;ui::load_languages(ecmgui::exe_dir()+"/localization/v2",error);if(!error.empty())trace("localization: "+error);}
    if(state.ui_right_pages){
        if(!state.ui_right_visible||state.layout().find("v2.tools")!=std::string::npos)reset_layout=true;
        state.ui_right_visible=true;
    }
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
    if (!no_nvml) {
        if(!gpu.start(500))trace("GPU monitor unavailable: " + gpu.reason());
        else gpu_details.start(gpu.nvml_source(),gpu.devices());
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
        config = std::move(candidate); config_path = path; leases_.clear(); verified_.clear();log_views_.clear();log_last_worker_=-1;
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
            leases_[index] = std::move(lease); toast((ecmgui_v2::ui::text("worker_started_009c206a",state.ui_language=="zh"))+std::to_string(index)); return true;
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
    if (logs) ImGui::DockBuilderDockWindow("Logs###v2.logs", logs);
    ImGui::DockBuilderFinish(dock);
    if(state.ui_right_pages)focus_right_=true;
}
bool App::begin_window(const char* name, bool* open, const ui::Settings& active) {
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
    const bool visible=ImGui::Begin(name,open,std::strstr(name,"###v2.navigation")?ImGuiWindowFlags_HorizontalScrollbar:0); window_dpi_[name] = ImGui::GetWindowViewport()->DpiScale;return visible;
}
void App::end_window() { ImGui::End(); ImGui::PopStyleVar(6); }
bool App::save_preferences() {
    try {
        size_t length=0;const auto* layout=ImGui::SaveIniSettingsToMemory(&length);state.layout(std::string(layout,length));
        const bool saved=state.save();status=saved?"UI preferences saved":state.notice;toast(status,!saved);return saved;
    }catch(const std::exception& error){status=error.what();toast(status,true);return false;}
}
bool App::move_page(int page,bool right) {
    if(!ui::move_page(state.ui_right_pages,state.ui_page,state.ui_right_page,page,right)){toast(ecmgui_v2::ui::text("keep_at_least_one_central_page_d623b219",state.ui_language=="zh"));return false;}
    if(right){state.ui_right_visible=true;focus_right_=true;}
    reset_layout=true;
    toast(ui::text(right?"moved_right":"moved_left",state.ui_language=="zh"));return true;
}
void App::draw(ui::Settings& pending, const ui::Settings& active, const ui::Fonts& fonts) {
    worker_speed_samples=workspace_chart_count=0;worker_speed_min=worker_speed_max={};
    for(auto& control:exit_controls)control={};
    if(status!=last_status){last_status=status;toast(status);}
    while(!toasts.empty()&&toasts.front().until<GetTickCount64())toasts.pop_front();
    const bool zh = fonts.chinese;
    if (ImGui::BeginMainMenuBar()) {
        ImGui::Text("ECM-ELF %s", version); ImGui::Separator();
        if(ImGui::BeginMenu(ecmgui_v2::ui::text("view_dcc839a4",zh))){
            if (ImGui::MenuItem(ecmgui_v2::ui::text("reset_layout_2e2a1784",zh))) reset_layout = true;
            if(ImGui::MenuItem(ecmgui_v2::ui::text("inspector_da188e3b",zh),nullptr,&state.ui_inspector))reset_layout=true;
            if(ImGui::MenuItem(ecmgui_v2::ui::text("logs_ea2100dc",zh),nullptr,&state.ui_logs))reset_layout=true;
            ImGui::MenuItem(ecmgui_v2::ui::text("system_title_bar_285c57a3",zh),nullptr,&state.window_title_bar);
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu(ecmgui_v2::ui::text("language_a4fe6526",zh))){
            if(ImGui::MenuItem(ui::text("language_english",zh),nullptr,state.ui_language=="en"))state.ui_language="en";
            if(ImGui::MenuItem(ui::text("language_chinese",zh),nullptr,state.ui_language=="zh"))state.ui_language="zh";
            if(ImGui::MenuItem(ui::text("reload_localization",zh))){std::string error;const auto directory=ui::language_directory();const bool ok=ui::load_languages(directory,error);toast(ok?ui::text("localization_reloaded",zh):error,!ok);if(!error.empty())trace(error);}
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem(ecmgui_v2::ui::text("notices_00a05fce",zh))) ImGui::OpenPopup("Notices###v2.notices");
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
            const char* tips[]{ecmgui_v2::ui::text("minimize_d72e311c",zh),ecmgui_v2::ui::text("maximize_restore_ab9616c1",zh),ecmgui_v2::ui::text("close_7d9eb7ac",zh)};
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
    begin_window(ecmgui_v2::ui::text("navigation_v2_navigation_fa8b2bee",zh),nullptr,active); fonts.push(ui::FontRole::Body);
    const bool compact = ImGui::GetContentRegionAvail().x < 110 * active.scale;
    const auto item_center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    for(auto& pos:nav_items)pos={};for(auto& pos:right_items)pos={};
    nav_drag_page=-1;nav_return={};
    const auto nav_palette=ui::palette(active.light);const float nav_scale=ImGui::GetFontSize()/15.f;
    nav_rounding=8*nav_scale;nav_item_height=std::max(40*nav_scale,ImGui::GetTextLineHeight()+16*nav_scale);nav_item_gap=8*nav_scale;
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,nav_rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,0);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,{0,.5f});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{12*nav_scale,8*nav_scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,nav_item_gap});
    const auto order=state.ui_nav_order;
    for(char id:order){const int i=id-'0';
        if(ui::pinned(state.ui_right_pages,i))continue;
        ImGui::PushID(i);const auto label=std::string(ui::page_name(i,zh))+"###page";
        ImGui::PushStyleColor(ImGuiCol_Button,state.ui_page==i?nav_palette.active:ImVec4{0,0,0,0});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,nav_palette.hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,nav_palette.active);
        if(ImGui::Button(label.c_str(),{std::max(ImGui::GetContentRegionAvail().x,ImGui::CalcTextSize(ui::page_name(i,zh)).x+24*nav_scale),nav_item_height}))state.ui_page=i;
        ImGui::PopStyleColor(3);nav_items[i]=item_center();
        const auto lo=ImGui::GetItemRectMin(),hi=ImGui::GetItemRectMax();
        if(compact&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::page_name(i,zh));
        if(ImGui::IsItemActive()&&ImGui::GetIO().MouseDownDuration[0]>=.32f&&ImGui::BeginDragDropSource()){
            nav_drag_page=i;ImGui::SetDragDropPayload("V2_PAGE",&i,sizeof(i));ImGui::TextUnformatted(ui::page_name(i,zh));ImGui::EndDragDropSource();}
        if(ImGui::BeginDragDropTarget()){
            if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE",ImGuiDragDropFlags_AcceptBeforeDelivery))if(payload->DataSize==sizeof(int)){
                const bool after=ImGui::GetIO().MousePos.y>nav_items[i].y;const float y=after?hi.y:lo.y;
                ImGui::GetWindowDrawList()->AddLine({lo.x,y},{hi.x,y},ImGui::GetColorU32(nav_palette.focus),2*nav_scale);
                if(payload->IsDelivery())ui::reorder_page(state.ui_nav_order,*static_cast<const int*>(payload->Data),i,after);
            }ImGui::EndDragDropTarget();}
        if(ImGui::BeginPopupContextItem("location")){if(ImGui::MenuItem(ecmgui_v2::ui::text("move_to_right_tabs_80d3c7aa",zh)))move_page(i,true);ImGui::EndPopup();}
        ImGui::PopID();
    }
    ImGui::PopStyleVar(5);
    ImGui::Separator();ImGui::Button(compact?">###right-drop":ecmgui_v2::ui::text("right_tabs_right_drop_3dd2712f",zh),{-1,0});
    nav_drop=item_center();
    if(ImGui::BeginDragDropTarget()){if(const auto* payload=ImGui::AcceptDragDropPayload("V2_PAGE"))if(payload->DataSize==sizeof(int))move_page(*static_cast<const int*>(payload->Data),true);ImGui::EndDragDropTarget();}
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("long_press_a_page_then_drag_here_to_open_i_bded2000",zh));
    if(!compact)ui::secondary_text(ecmgui_v2::ui::text("hold_and_drag_to_reorder_close_a_right_tab_ba726d0c",zh));
    ImGui::PopFont(); end_window();
    begin_window(ecmgui_v2::ui::text("workspace_v2_workspace_ebc1df62",zh),nullptr,active);
    workspace_docked=ImGui::IsWindowDocked();
    if(drawn_center_page_!=state.ui_page){ImGui::SetScrollY(0);drawn_center_page_=state.ui_page;}
    fonts.push(ui::FontRole::Body);
    ImGui::PushTextWrapPos(0); draw_page(active, fonts); ImGui::PopTextWrapPos(); ImGui::PopFont(); end_window();
    if (state.ui_inspector && wide) { begin_window(ecmgui_v2::ui::text("inspector_v2_inspector_dc966628",zh), &state.ui_inspector,active); draw_inspector(fonts); end_window(); }
    if (state.ui_logs) { begin_window(ecmgui_v2::ui::text("logs_v2_logs_a97cbe80",zh), &state.ui_logs,active); draw_logs(fonts); end_window(); }
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
void App::draw_workers(const ui::Settings& active, const ui::Fonts& fonts) {
    body_font_height=ImGui::GetFontSize();const auto* body_glyph=ImGui::GetFontBaked()->FindGlyph('M');body_glyph_height=body_glyph->Y1-body_glyph->Y0;
    const bool zh=fonts.chinese;
    const auto p = ui::palette(active.light);
    const float toolbar_top=ImGui::GetCursorScreenPos().y;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*active.scale,4*active.scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6*active.scale,6*active.scale});
    fonts.push(ui::FontRole::Primary);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ecmgui_v2::ui::text("workers_7a1ec9fe",zh));ImGui::PopFont();
    if(ImGui::IsItemHovered()&&!config_path.empty())ImGui::SetTooltip("%s",config_path.c_str());
    if(!ready){ui::secondary_text(ecmgui_v2::ui::text("open_a_workflow_in_settings_55a5439d",zh));ImGui::PopStyleVar(2);return;}
    const float actions_width=ImGui::CalcTextSize(ecmgui_v2::ui::text("start_allstop_safely_13031d69",zh)).x+60*active.scale;
    if(ImGui::GetContentRegionAvail().x>ImGui::GetItemRectSize().x+actions_width)ImGui::SameLine();
    if (ui::button("start-all", ecmgui_v2::ui::text("start_all_1a7e6299",fonts.chinese), ui::ButtonKind::Primary, p, active.scale,false,&primary_font_height,true))
        for (const auto& worker : controller.workers()) if(!start(worker.index))toast(status,true);
    start_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    if(ImGui::GetContentRegionAvail().x>ImGui::GetItemRectSize().x+ImGui::CalcTextSize(ecmgui_v2::ui::text("stop_safely_a2e61809",zh)).x+24*active.scale)ImGui::SameLine();
    if (ui::button("stop-all", ecmgui_v2::ui::text("stop_safely_a2e61809",fonts.chinese), ui::ButtonKind::Secondary, p, active.scale,false,&secondary_button_font_height,true)) {controller.stop_all();toast(ecmgui_v2::ui::text("safe_stop_requested_waiting_for_checkpoint_c4c6c1c2",fonts.chinese));}
    stop_control=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
    worker_toolbar_height=ImGui::GetCursorScreenPos().y-toolbar_top;ImGui::PopStyleVar(2);
    for (const auto& worker : controller.workers()) if (worker.p95.valid) {
        const bool pending = worker.p95.level == ecmgui::P95Notice::Level::Pending;
        ImGui::TextColored(pending ? ImVec4(1,.35f,.4f,1) : ImVec4(.8f,.65f,.35f,1), "Prime95 %s: %s %s", worker.name.c_str(),worker.p95.error.c_str(),worker.p95.note.c_str());
    }
    worker_select_controls.clear();worker_action_controls.clear();worker_table_rows=0;
    const float scale=ImGui::GetFontSize()/15.f;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{8*scale,4*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*scale,4*scale});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6*scale,2*scale});
    const float control_height=std::max(24*scale,ImGui::GetTextLineHeight()+8*scale);
    float widths[]{72,112,80,84,150,64,76,76};
    // Size non-flex columns from the actual localized values before deciding
    // whether all eight columns fit. A long ETA/status must not be clipped.
    for(const auto& worker:controller.workers()){
        const auto progress=ui::worker_progress(worker,zh);
        const auto label=ui::message(worker.stop_requested?"Stopping":ecmgui::state_name(worker.state()),zh);
        widths[0]=std::max(widths[0],ImGui::CalcTextSize(ui::message("Stopping",zh).c_str()).x/scale+16);
        widths[2]=std::max(widths[2],ImGui::CalcTextSize(label.c_str()).x/scale+26);
        widths[5]=std::max(widths[5],ImGui::CalcTextSize(progress.speed.c_str()).x/scale);
        widths[6]=std::max(widths[6],ImGui::CalcTextSize(progress.eta.c_str()).x/scale);
        const auto done=std::to_string(worker.tasks_done)+" / "+std::to_string(worker.hits);
        widths[7]=std::max(widths[7],ImGui::CalcTextSize(done.c_str()).x/scale);
    }
    float minimum=0;for(float w:widths)minimum+=(w+16)*scale;
    const auto left=ImGui::GetCursorScreenPos();const float visible_width=ImGui::GetContentRegionAvail().x;
    worker_flexible=visible_width<minimum;worker_table_scroll_max_x=0;worker_field_overflow=0;
    worker_summary_rects.clear();
    if(worker_flexible){
        // Flow whole labelled fields; the enclosing workspace owns vertical scroll.
        // Never hide columns or shrink the font to fit a narrow dock / large font.
        const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
        const float right=left.x+visible_width;
        for(const auto& worker:controller.workers()){
            ++worker_table_rows;ImGui::PushID(worker.index);ImGui::BeginGroup();
            const auto st=worker.state();const bool busy=worker.running()||st==ecmgui::WorkerRunState::Restarting;
            const auto progress=ui::worker_progress(worker,zh);
            const auto name="#"+std::to_string(worker.index)+" "+worker.name;
            const float name_height=std::max(control_height,ImGui::CalcTextSize(name.c_str(),nullptr,false,std::max(1.f,visible_width-16*scale)).y+8*scale);
            const auto name_pos=ImGui::GetCursorScreenPos();
            if(ImGui::Selectable("##worker-name",state.ui_worker==worker.index,0,{visible_width,name_height})){state.ui_worker=worker.index;log_active_worker=worker.index;log_last_worker_=-1;}
            worker_select_controls[worker.index]=center();
            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),ImGui::GetFontSize(),{name_pos.x+8*scale,name_pos.y+4*scale},ImGui::GetColorU32(p.fg),name.c_str(),nullptr,std::max(1.f,visible_width-16*scale));
            const auto row_top=ImGui::GetCursorScreenPos().y;
            if(ui::button("worker-action",worker.stop_requested?ui::text("message_23",zh):busy?ui::text("stop_cae7d57b",zh):ui::text("start_e4bb9f1e",zh),busy?ui::ButtonKind::Danger:ui::ButtonKind::Secondary,p,scale,worker.stop_requested,nullptr,true)){
                if(busy){controller.request_stop(worker.index);toast("Safe stop requested: "+worker.name);}else if(!start(worker.index))toast(status,true);
            }
            worker_action_controls[worker.index]=center();worker_button_height=ImGui::GetItemRectSize().y;
            const auto state_label=ui::message(worker.stop_requested?"Stopping":ecmgui::state_name(st),zh);
            const float badge_width=ImGui::CalcTextSize(state_label.c_str()).x+26*scale;
            if(ImGui::GetItemRectMax().x+ImGui::GetStyle().ItemSpacing.x+badge_width<=right)ImGui::SameLine();
            ui::status_badge(state_label.c_str(),st==ecmgui::WorkerRunState::Error?ui::Status::Danger:worker.stop_requested?ui::Status::Warning:busy?ui::Status::Success:ui::Status::Info,p,scale);
            const auto device=(worker.execution_target=="gpu"?"GPU"+std::to_string(worker.device):std::string("CPU"))+" / S"+std::to_string(worker.stage2?2:1)+(worker.duplicate_device?"*":"");
            if(ImGui::GetItemRectMax().x+ImGui::GetStyle().ItemSpacing.x+ImGui::CalcTextSize(device.c_str()).x<=right)ImGui::SameLine();
            ImGui::AlignTextToFramePadding();ImGui::TextColored(worker.stage2?p.stage2:p.stage1,"%s",device.c_str());
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s | %s",device.c_str(),worker.curve_model.c_str());
            worker_row_height=ImGui::GetCursorScreenPos().y-row_top;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,worker.stage2?p.stage2:p.stage1);
            const bool separate=ImGui::CalcTextSize(progress.label.c_str()).x+16*scale>visible_width;
            ImGui::ProgressBar(progress.fraction,{visible_width,control_height},separate?"":progress.label.c_str());ImGui::PopStyleColor();
            if(separate){ImGui::PushTextWrapPos(0);ImGui::TextUnformatted(progress.label.c_str());ImGui::PopTextWrapPos();}
            const std::string fields[]{std::string(ui::text("s_curve_1715c3bd",zh))+": "+progress.speed,"ETA: "+progress.eta,std::string(ui::text("done_hits_8c10e20d",zh))+": "+std::to_string(worker.tasks_done)+" / "+std::to_string(worker.hits)};
            for(int i=0;i<3;++i){
                const float width=std::min(visible_width,ImGui::CalcTextSize(fields[i].c_str()).x);
                if(i&&ImGui::GetItemRectMax().x+ImGui::GetStyle().ItemSpacing.x+width<=right)ImGui::SameLine();
                ImGui::PushTextWrapPos(ImGui::GetCursorPosX()+std::max(1.f,width));ImGui::TextUnformatted(fields[i].c_str());ImGui::PopTextWrapPos();
                worker_field_overflow=std::max(worker_field_overflow,ImGui::GetItemRectMax().x-right);
            }
            const auto task=ecmgui::task_caption(worker.task,ui::text("known_factors_f88cad5f",zh),"assignment ID",ui::text("no_queued_task_b954d32f",zh));
            const auto phase=ui::worker_phase(worker,zh);const auto summary=std::string(ui::text("task_953516d5",zh))+task+(phase.empty()?"":"\n"+phase);
            const auto summary_pos=ImGui::GetCursorScreenPos();
            ImGui::PushTextWrapPos(0);ImGui::TextColored(p.muted,"%s",summary.c_str());ImGui::PopTextWrapPos();
            worker_summary_rects.push_back({summary_pos,ImGui::GetItemRectMax()});
            ImGui::EndGroup();ImGui::Separator();ImGui::PopID();
        }
        worker_table_height=worker_table_content_height=worker_table_required_height=worker_table_data_height=ImGui::GetCursorScreenPos().y-left.y;
        worker_table_scroll_max_y=0;ImGui::PopStyleVar(3);draw_worker_speed(fonts);return;
    }
    const float padding=ImGui::GetStyle().CellPadding.y*2;float content_height=ImGui::GetTextLineHeight()+padding+4*scale;
    for(const auto& worker:controller.workers()){
        const auto task=ecmgui::task_caption(worker.task,ecmgui_v2::ui::text("known_factors_f88cad5f",zh),"assignment ID",ecmgui_v2::ui::text("no_queued_task_b954d32f",zh));
        const auto phase=ui::worker_phase(worker,zh);const auto summary=std::string(ecmgui_v2::ui::text("task_953516d5",zh))+task+(phase.empty()?"":"\n"+phase);
        // Selectable's baseline follows the button's FramePadding.y in each row.
        content_height+=control_height+padding+ImGui::GetStyle().FramePadding.y;
        content_height+=ImGui::CalcTextSize(summary.c_str(),nullptr,false,std::max(1.f,visible_width-24*scale)).y+padding;
    }
    // Reserve child viewport padding so every row remains visible.
    content_height+=ImGui::GetStyle().ScrollbarSize;
    // Keep every worker row in the table. Only the surrounding workspace scrolls
    // vertically; narrow surfaces use the flow layout above.
    const float table_height=content_height;
    // New column order gets a fresh table identity instead of reusing old widths.
    if (ImGui::BeginTable("workers-v2",8,ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerH|ImGuiTableFlags_BordersOuterH|ImGuiTableFlags_ScrollY|ImGuiTableFlags_PadOuterX|ImGuiTableFlags_SizingFixedFit,{0,table_height})) {
        worker_table_scroll_max_x=ImGui::GetScrollMaxX();
        worker_table_scroll_max_y=ImGui::GetScrollMaxY();worker_table_height=ImGui::GetWindowSize().y;worker_table_content_height=ImGui::GetCurrentWindow()->ContentSize.y;
        const auto table_origin=ImGui::GetWindowPos();worker_table_data_height=ImGui::GetCurrentWindow()->InnerRect.GetHeight();
        struct SummaryRow {ImVec2 min,max,pos;float wrap;std::string text;};
        std::vector<SummaryRow> summaries;auto* table_draw=ImGui::GetWindowDrawList();
        const auto table_clip_min=ImGui::GetWindowDrawList()->GetClipRectMin(),table_clip_max=ImGui::GetWindowDrawList()->GetClipRectMax();
        float body_top=table_clip_min.y;
        const char* labels[]{ecmgui_v2::ui::text("action_64cff131",zh),"Worker",ecmgui_v2::ui::text("state_a3b50c47",zh),ecmgui_v2::ui::text("device_engine_1041b2a2",zh),ecmgui_v2::ui::text("task_progress_e38b9d46",zh),ecmgui_v2::ui::text("s_curve_1715c3bd",zh),"ETA",ecmgui_v2::ui::text("done_hits_8c10e20d",zh)};
        for(int i=0;i<8;++i)ImGui::TableSetupColumn(labels[i],i==4?ImGuiTableColumnFlags_WidthStretch:ImGuiTableColumnFlags_WidthFixed,widths[i]*scale);
        ImGui::TableHeadersRow();
        const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
        for (const auto& worker : controller.workers()) {
            ++worker_table_rows;
            ImGui::PushID(worker.index);ImGui::TableNextRow(0,control_height+padding);
            const auto st=worker.state();const auto progress=ui::worker_progress(worker,zh);
            const auto name="#"+std::to_string(worker.index)+" "+worker.name;
            ImGui::TableSetColumnIndex(0);
            if(summaries.empty())body_top=std::max(body_top,ImGui::GetCursorScreenPos().y-ImGui::GetStyle().CellPadding.y);
            const float main_row_top=ImGui::GetCursorScreenPos().y;
            const bool busy=worker.running()||st==ecmgui::WorkerRunState::Restarting;
            if(ui::button("worker-action",worker.stop_requested?ecmgui_v2::ui::text("stopping_a71ee1d4",zh):busy?ecmgui_v2::ui::text("stop_cae7d57b",zh):ecmgui_v2::ui::text("start_e4bb9f1e",zh),busy?ui::ButtonKind::Danger:ui::ButtonKind::Secondary,p,scale,worker.stop_requested,nullptr,true)){
                if(busy){controller.request_stop(worker.index);toast("Safe stop requested: "+worker.name);}else if(!start(worker.index))toast(status,true);
            }
            worker_action_controls[worker.index]=center();worker_button_height=ImGui::GetItemRectSize().y;
            ImGui::TableSetColumnIndex(1);
            if(ImGui::Selectable(name.c_str(),state.ui_worker==worker.index,0,{0,control_height})){state.ui_worker=worker.index;log_active_worker=worker.index;log_last_worker_=-1;}
            worker_select_controls[worker.index]=center();if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",worker.name.c_str());
            ImGui::TableSetColumnIndex(2);
            const auto state_kind=st==ecmgui::WorkerRunState::Error?ui::Status::Danger:worker.stop_requested||st==ecmgui::WorkerRunState::Restarting?ui::Status::Warning:worker.running()?ui::Status::Success:ui::Status::Info;
            ui::status_badge(worker.stop_requested?ecmgui_v2::ui::text("stopping_a71ee1d4",zh):ui::message(ecmgui::state_name(st),zh).c_str(),state_kind,p,scale);
            if(ImGui::IsItemHovered()&&!worker.state_text().empty())ImGui::SetTooltip("%s",worker.state_text().c_str());
            ImGui::TableSetColumnIndex(3);ImGui::AlignTextToFramePadding();
            const auto device=worker.execution_target=="gpu"?"GPU"+std::to_string(worker.device):std::string("CPU");
            ImGui::TextColored(worker.stage2?p.stage2:p.stage1,"%s / S%d%s",device.c_str(),worker.stage2?2:1,worker.duplicate_device?"*":"");
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s | Stage %d | %s%s",device.c_str(),worker.stage2?2:1,worker.curve_model.c_str(),worker.duplicate_device?ecmgui_v2::ui::text("multiple_workers_use_this_gpu_ad48019a",zh):"");
            ImGui::TableSetColumnIndex(4);ImGui::PushStyleColor(ImGuiCol_PlotHistogram,worker.stage2?p.stage2:p.stage1);
            ImGui::ProgressBar(progress.fraction,{-1,control_height},progress.label.c_str());ImGui::PopStyleColor();
            ImGui::TableSetColumnIndex(5);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(progress.speed.c_str());
            ImGui::TableSetColumnIndex(6);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(progress.eta.c_str());
            ImGui::TableSetColumnIndex(7);ImGui::AlignTextToFramePadding();ImGui::Text("%lld / %lld",worker.tasks_done,worker.hits);
            const auto task=ecmgui::task_caption(worker.task,ecmgui_v2::ui::text("known_factors_f88cad5f",zh),"assignment ID",ecmgui_v2::ui::text("no_queued_task_b954d32f",zh));
            const auto phase=ui::worker_phase(worker,zh);const auto summary=std::string(ecmgui_v2::ui::text("task_953516d5",zh))+task+(phase.empty()?"":"\n"+phase);
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
    worker_speed_min=ImGui::GetCursorScreenPos();
    for(const auto& worker:controller.workers()){
        ImGui::PushID(worker.index);++workspace_chart_count;
        const auto title=std::string(ecmgui_v2::ui::text("curve_time_369c1d1c",zh))+"  #"+std::to_string(worker.index)+" "+worker.name+" / "+(worker.stage2?"Stage2":"Stage1");
        ImGui::SeparatorText(title.c_str());worker_speed_samples+=worker.hist_s_per_curve.size();
        if(!worker.hist_s_per_curve.empty()){
            std::vector<ui::ChartSeries> series{{ecmgui_v2::ui::text("s_curve_1715c3bd",zh),worker.stage2?p.stage2:p.stage1,{}}};
            for(size_t i=0;i<worker.hist_s_per_curve.size();++i)series[0].points.push_back({static_cast<double>(i+1),worker.hist_s_per_curve[i]});
            const auto value=ui::worker_progress(worker,zh).speed+" s/curve";
            ui::ChartFormat format;format.legend_controls=false;format.show_y_ticks=false;format.show_axis_titles=false;
            format.integer_x=true;format.corner_text=value.c_str();format.y_format="%.2f s/curve";format.x_format="%.0f";
            ui::chart("workspace-speed",series,ecmgui_v2::ui::text("sample_95a2f1ed",zh),"s/curve",zh,0,120,format);
        }else ImGui::TextDisabled("%s",ecmgui_v2::ui::text("waiting_for_a_measured_curve_time_c1472a27",zh));
        worker_speed_max=ImGui::GetItemRectMax();ImGui::PopID();
    }
}

void App::draw_exit(const ui::Settings& active,const ui::Fonts& fonts) {
    using Phase=ecmgui::WorkerController::ExitPhase;
    const auto phase=controller.exit_phase();if(phase==Phase::Idle)return;
    const bool zh=fonts.chinese;
    const auto p=ui::palette(active.light);const auto* vp=ImGui::GetMainViewport();
    const float width=std::max(240.f,std::min(600*active.scale,vp->WorkSize.x-24*active.scale));
    bool stage1=false,stage2=false;
    for(const auto& w:controller.workers())if(w.running()){stage1|=!w.stage2;stage2|=w.stage2;}
    const auto explain=stage1&&stage2?ecmgui_v2::ui::text("stage1_saves_a_checkpoint_stage2_finishes__f16b2dc8",zh):
        stage2?ecmgui_v2::ui::text("stage2_finishes_its_current_curve_before_s_23f91e06",zh):
        ecmgui_v2::ui::text("stage1_saves_a_checkpoint_before_stopping_fd6d9575",zh);
    const auto center=[](){return ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};};
    fonts.push(ui::FontRole::Body);
    const char* popup=ecmgui_v2::ui::text("confirm_exit_v2_exit_bd0b37b2",zh);
    if(phase==Phase::Confirm){
        if(!ImGui::IsPopupOpen(popup))ImGui::OpenPopup(popup);
        ImGui::SetNextWindowViewport(vp->ID);ImGui::SetNextWindowPos(vp->GetCenter(),ImGuiCond_Always,{.5f,.5f});
        ImGui::SetNextWindowSize({width,0});
        if(ImGui::BeginPopupModal(popup,nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove)){
            ImGui::PushTextWrapPos(0);
            ImGui::Text(ecmgui_v2::ui::text("d_worker_s_are_running_stop_safely_and_exi_24856f27",zh),running_count());
            ImGui::TextWrapped("%s",explain);ImGui::Separator();
            const bool keyboard=!ImGui::IsWindowAppearing()&&ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            const bool accept=ui::button("exit-stop",ecmgui_v2::ui::text("stop_and_exit_enter_61229fa0",zh),ui::ButtonKind::Primary,p,active.scale);
            exit_controls[0]=center();
            const bool cancel=ui::button("exit-cancel",ecmgui_v2::ui::text("cancel_esc_862c2324",zh),ui::ButtonKind::Secondary,p,active.scale);
            exit_controls[1]=center();
            if(cancel||(keyboard&&ImGui::IsKeyPressed(ImGuiKey_Escape))){controller.cancel_close();ImGui::CloseCurrentPopup();}
            else if(accept||(keyboard&&(ImGui::IsKeyPressed(ImGuiKey_Enter)||ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)))){begin_exit();ImGui::CloseCurrentPopup();}
            ImGui::PopTextWrapPos();ImGui::EndPopup();
        }
    }
    if(controller.exit_phase()==Phase::Stopping){
        const auto now=GetTickCount64(),deadline=controller.exit_deadline();const double left=deadline>now?(deadline-now)/1000.:0;
        ImGui::SetNextWindowViewport(vp->ID);ImGui::SetNextWindowPos(vp->GetCenter(),ImGuiCond_Always,{.5f,.5f});ImGui::SetNextWindowSize({width,0});
        ImGui::Begin(ecmgui_v2::ui::text("stopping_workers_v2_stopping_53c8f7ae",zh),nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushTextWrapPos(0);
        ImGui::TextWrapped("%s",explain);ImGui::Text(ecmgui_v2::ui::text("up_to_1f_s_remaining_2d6201db",zh),left);
        ImGui::TextColored(p.warning,"%s",ecmgui_v2::ui::text("at_the_deadline_remaining_processes_are_te_0b34a996",zh));
        ImGui::Separator();
        for(const auto& w:controller.workers()){
            ImGui::TextColored(w.running()?p.warning:p.success,"#%d %s | %s | %s",w.index,w.name.c_str(),w.stage2?"Stage2":"Stage1",w.running()?(w.stage2?ecmgui_v2::ui::text("finishing_current_curve_1fef6708",zh):ecmgui_v2::ui::text("waiting_for_checkpoint_b160a8dc",zh)):ecmgui_v2::ui::text("stopped_1a4f630a",zh));
        }
        ImGui::Separator();
        const bool force=ui::button("exit-force",ecmgui_v2::ui::text("force_exit_esc_8774db86",zh),ui::ButtonKind::Danger,p,active.scale);exit_controls[2]=center();
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
            ImGui::Text(ecmgui_v2::ui::text("task_curves_llu_llu_58f847e9",zh),s.done,s.total);
            ImGui::Text(ecmgui_v2::ui::text("phase_s_batch_llu_llu_f8541ba8",zh),s.phase.c_str(),s.batch,s.batches);
            if(s.curve_model=="ed")ImGui::Text("Edwards sigma=%llu",s.sigma);
            else ImGui::Text("%s param=%d sigma=%llu",s.curve_model.c_str(),s.param,s.sigma);
            ImGui::Text("B1=%llu B2=%llu",s.b1,s.b2);
        }
        if (ImGui::Button(ecmgui_v2::ui::text("open_worker_folder_5650fc18",zh))) ecmgui::open_in_explorer(worker->working_dir);
        ImGui::TextWrapped("%s", worker->effective_config.c_str());
        ImGui::TextWrapped(ecmgui_v2::ui::text("worker_version_s_8c3af2a1",zh), worker->application_version.empty() ? ecmgui_v2::ui::text("unknown_b23a6a84",zh) : worker->application_version.c_str());
        if (ImGui::Button(ecmgui_v2::ui::text("copy_command_9a01feec",zh))) ImGui::SetClipboardText(worker->command_line().c_str());
        ImGui::TextWrapped("%s", worker->command_line().c_str());
    } else ImGui::TextUnformatted(ecmgui_v2::ui::text("no_worker_selected_947b0173",zh));
    ImGui::PopFont();
}
void App::draw_logs(const ui::Fonts& fonts) {
    const bool zh=fonts.chinese;
    const auto p=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);const float scale=ImGui::GetFontSize()/15.f;
    fonts.push(ui::FontRole::Body);log_tab_controls.clear();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,4*scale});
    const bool request=log_last_worker_!=state.ui_worker;
    if(request){log_active_worker=selected()?selected()->index:0;log_last_worker_=state.ui_worker;}
    const int requested_worker=log_active_worker;
    const ecmgui::WorkerView* active_worker=nullptr;
    if(ImGui::BeginTabBar("worker-logs",ImGuiTabBarFlags_FittingPolicyScroll)){
        for(int item=0;item<=static_cast<int>(controller.workers().size());++item){
            const auto* worker=item?&controller.workers()[static_cast<size_t>(item-1)]:nullptr;const int id=worker?worker->index:0;
            const auto label=(worker?"#"+std::to_string(id)+" "+worker->name:std::string(ecmgui_v2::ui::text("application_e7ad522e",zh)))+"###log-"+std::to_string(id);
            const bool open=ImGui::BeginTabItem(label.c_str(),nullptr,request&&id==requested_worker?ImGuiTabItemFlags_SetSelected:0);
            log_tab_controls[id]={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
            if(open){log_active_worker=id;active_worker=worker;if(worker){state.ui_worker=id;log_last_worker_=id;}ImGui::EndTabItem();}
        }
        ImGui::EndTabBar();
    }
    // TabBar may defer a newly selected tab by a frame; always use its stable id.
    for(const auto& w:controller.workers())if(w.index==log_active_worker)active_worker=&w;
    if(active_worker){
        const auto progress=ui::worker_progress(*active_worker,zh);
        const auto task=ecmgui::task_caption(active_worker->task,ecmgui_v2::ui::text("known_factors_f88cad5f",zh),"assignment ID",ecmgui_v2::ui::text("no_queued_task_b954d32f",zh));
        ImGui::PushTextWrapPos(0);ImGui::TextColored(active_worker->stage2?p.stage2:p.stage1,ecmgui_v2::ui::text("s_s_s_s_s_curve_eta_s_6c18c89b",zh),active_worker->stage2?"Stage 2":"Stage 1",task.c_str(),progress.label.c_str(),progress.speed.c_str(),progress.eta_detail.c_str());ImGui::PopTextWrapPos();
        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",active_worker->task.c_str(),ui::worker_phase(*active_worker,zh).c_str());
        const auto shape=ui::worker_shape(*active_worker);
        if(!shape.empty()){
            // Keep the shape on one line; hover to read it in a narrow pane.
            ImGui::TextColored(p.stage2,"%s",shape.c_str());
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",shape.c_str());
        }
        if(active_worker->stage2&&active_worker->stage2_status.valid){
            const auto& s=active_worker->stage2_status;
            ImGui::TextColored(p.stage2,ecmgui_v2::ui::text("phase_s_curve_llu_llu_221f1861",zh),s.phase.c_str(),s.current,s.total);
            const auto batch=ui::stage2_batch(s,zh);
            if(batch.visible){
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,p.stage2);
                ImGui::ProgressBar(batch.fraction,{std::min(420*scale,ImGui::GetContentRegionAvail().x),ImGui::GetFrameHeight()},batch.label.c_str());ImGui::PopStyleColor();
                ImGui::TextDisabled("%s",batch.remaining.c_str());
            }
            if(active_worker->stop_requested)ImGui::TextColored(p.warning,"%s",ecmgui_v2::ui::text("stop_after_the_current_stage2_curve_d315e7ec",zh));
        }
    }
    auto found=log_views_.find(log_active_worker);if(found==log_views_.end())found=log_views_.emplace(log_active_worker,LogView{state.ui_raw_log,state.ui_follow_log,true}).first;
    auto& view=found->second;ImGui::PushID(log_active_worker);
    if(ui::button("raw",ecmgui_v2::ui::text("raw_output_f3dbf20a",zh),view.raw?ui::ButtonKind::Selected:ui::ButtonKind::Secondary,p,scale)){view.raw=!view.raw;state.ui_raw_log=view.raw;view.jump=true;}
    log_raw_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SameLine();if(ui::button("follow",ecmgui_v2::ui::text("follow_641d1ef6",zh),view.follow?ui::ButtonKind::Selected:ui::ButtonKind::Secondary,p,scale)){view.follow=!view.follow;state.ui_follow_log=view.follow;view.jump=view.follow;}
    log_follow_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(active_worker){
        ImGui::SameLine();if(ui::button("log-folder",ecmgui_v2::ui::text("open_log_folder_4862da00",zh),ui::ButtonKind::Secondary,p,scale))ecmgui::open_in_explorer(ecmgui::path_dir(active_worker->log_file));
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
    if(!count)ImGui::TextDisabled("%s",ecmgui_v2::ui::text("no_output_yet_7b7658d7",zh));
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
    case 5: draw_settings(fonts.chinese); break;
    case 6: ui::state_strip(ui::palette(active.light),active.scale,fonts.chinese); break;
    }
}
}
