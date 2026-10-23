#include "../app.h"
#include "../ui/i18n.h"
#include <algorithm>

namespace ecmgui_v2 {
bool App::save_gpu_clock_range(const std::string& uuid,ClockRange range,bool automatic){
    const bool zh=state.ui_language=="zh";
    const auto fail=[&](const char* key){status=ui::text(key,zh);return false;};
    if(!ready)return fail("gpu.clock_open_config");
    if(!clock_uuid(uuid)||!range.valid())return fail("gpu.clock_invalid");
    if(!clock_profiles_valid)return fail("gpu.clock_config_invalid");
    ecmgui::FileStamp current;
    if(!ecmgui::stamp_file(config_path,current)||current.exists!=config_stamp.exists||current.size!=config_stamp.size||current.mtime!=config_stamp.mtime||current.hash!=config_stamp.hash)
        return fail("gpu.clock_config_changed");
    std::string error;ecmgui::TomlFile disk;
    if(!disk.load(config_path,error)){status=error;return false;}
    ClockProfiles profiles;std::set<std::string> autos;
    if(!parse_clock_profiles(disk.get("gui","gpu_clock_ranges"),profiles,&autos))return fail("gpu.clock_config_invalid");
    profiles[uuid]=range;if(automatic)autos.insert(uuid);else autos.erase(uuid);
    const auto encoded=encode_clock_profiles(profiles,autos);
    if(encoded.empty()||encoded.size()>16384)return fail("gpu.clock_invalid");
    // Save only this UI-owned field, including while workers run. Unrelated
    // unsaved computation edits stay in memory; no worker is reconfigured.
    auto draft_config=config;draft_config.set("gui","gpu_clock_ranges",encoded);
    disk.set("gui","gpu_clock_ranges",encoded);
    if(!disk.save(error)){status=error;return false;}
    const bool dirty=config.dirty();config=dirty?std::move(draft_config):std::move(disk);
    saved_clock_profiles=std::move(profiles);saved_clock_automatic=std::move(autos);ecmgui::stamp_file(config_path,config_stamp);
    status=ui::text("gpu.clock_saved",zh);return true;
}
void App::draw_gpu_clock_control(int index,const std::string& expected_uuid,bool zh){
    auto& display=gpu_presentations[index];
    const auto p=ui::palette(state.ui_theme=="light");const float scale=ImGui::GetFontSize()/15.f;
    const auto center=[](){const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();return ImVec2{(a.x+b.x)*.5f,(a.y+b.y)*.5f};};
    ImGui::SameLine(0,6*scale);
    ImGui::PushStyleColor(ImGuiCol_Button,{0,0,0,0});
    const bool open=ImGui::Button("##clock-settings",{20*scale,20*scale});
    display.gear=center();const auto anchor=ImGui::GetItemRectMax();
    ui::line_icon(ImGui::GetWindowDrawList(),ui::Icon::Settings,{display.gear.x-8*scale,display.gear.y-8*scale},16*scale,ImGui::GetColorU32(p.muted));
    ImGui::PopStyleColor();
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("gpu.clock_settings",zh));
    if(open){ImGui::OpenPopup("clock-range");gpu_clocks.query(index,expected_uuid);}
    if(!ImGui::IsPopupOpen("clock-range"))return;
    const auto* viewport=ImGui::GetMainViewport();
    const float width=std::min(460*scale,viewport->WorkSize.x-24*scale);
    const float pos_x=std::clamp(anchor.x-width,viewport->WorkPos.x+8*scale,viewport->WorkPos.x+viewport->WorkSize.x-width-8*scale);
    ImGui::SetNextWindowPos({pos_x,anchor.y+4*scale},ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({width,0},ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints({width,0},{width,std::max(80*scale,viewport->WorkSize.y-16*scale)});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,10*scale);
    const bool visible=ImGui::BeginPopup("clock-range",ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove);
    if(visible){
        const auto popup_pos=ImGui::GetWindowPos(),popup_size=ImGui::GetWindowSize();
        const float top=viewport->WorkPos.y+8*scale,bottom=viewport->WorkPos.y+viewport->WorkSize.y-8*scale;
        ImGui::SetWindowPos({pos_x,std::clamp(popup_pos.y,top,std::max(top,bottom-popup_size.y))});
        display.popover=true;display.popover_min=ImGui::GetWindowPos();const auto size=ImGui::GetWindowSize();
        display.popover_max={display.popover_min.x+size.x,display.popover_min.y+size.y};
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(ui::text("gpu.clock_settings",zh));
        auto snapshot=gpu_clocks.get(index);display.supported=snapshot.supported;
        const auto& uuid=snapshot.uuid;
        if(!uuid.empty())ui::secondary_text(uuid.c_str());
        ImGui::Separator();
        if(!snapshot.queried||snapshot.busy)ImGui::TextUnformatted(ui::text("gpu.clock_busy",zh));
        if(snapshot.error){
            ImGui::TextColored(p.warning,"%s",ui::text(clock_error_key(snapshot.error),zh));
            if(!snapshot.detail.empty())ui::secondary_text(snapshot.detail.c_str());
        }
        if(snapshot.supported.valid()&&clock_uuid(uuid)){
            auto& draft=clock_drafts[uuid];
            if(!draft.initialized){
                const auto saved=saved_clock_profiles.find(uuid);
                draft.range=saved==saved_clock_profiles.end()?snapshot.supported:saved->second;
                draft.automatic=saved_clock_automatic.count(uuid)!=0;draft.initialized=true;
            }
            ImGui::Text(ui::text("gpu.clock_limits",zh),snapshot.supported.low,snapshot.supported.high);
            ui::secondary_text(ui::text(snapshot.smi_range?"gpu.clock_smi_source":snapshot.pstate_range?"gpu.clock_pstate_source":"gpu.clock_supported_source",zh));
            ImGui::BeginDisabled(snapshot.busy);
            ClockRange shown{std::clamp(draft.range.low,snapshot.supported.low,snapshot.supported.high),std::clamp(draft.range.high,snapshot.supported.low,snapshot.supported.high)};
            shown.high=std::max(shown.high,shown.low);
            if(ui::clock_range_slider("range",shown,snapshot.supported,p,&display.range_slider)){draft.range=shown;draft.message_key.clear();}
            const float gap=ImGui::GetStyle().ItemSpacing.x,field=(ImGui::GetContentRegionAvail().x-gap)*.5f;
            const float label=std::max(ImGui::CalcTextSize(ui::text("gpu.clock_min",zh)).x,ImGui::CalcTextSize(ui::text("gpu.clock_max",zh)).x);
            const float start=ImGui::GetCursorPosX();
            ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ui::text("gpu.clock_min",zh));ImGui::SameLine(start+label+gap);
            ImGui::SetNextItemWidth(std::max(35*scale,field-label-gap));
            if(ImGui::InputInt("##clock-min",&draft.range.low,0,0))draft.message_key.clear();display.input_low=center();
            ImGui::SameLine(start+field+gap);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ui::text("gpu.clock_max",zh));ImGui::SameLine(start+field+gap+label+gap);
            ImGui::SetNextItemWidth(std::max(35*scale,field-label-gap));
            if(ImGui::InputInt("##clock-max",&draft.range.high,0,0))draft.message_key.clear();display.input_high=center();
            const bool valid=draft.range.valid()&&draft.range.low>=snapshot.supported.low&&draft.range.high<=snapshot.supported.high;
            if(!valid)ImGui::TextColored(p.warning,"%s",ui::text("gpu.clock_invalid",zh));
            ui::secondary_text(ui::text("gpu.clock_draft_hint",zh));
            if(ui::checkbox(ui::text("gpu.clock_automatic",zh),&draft.automatic))draft.message_key.clear();display.automatic=center();
            ui::secondary_text(ui::text("gpu.clock_auto_hint",zh));
            if(snapshot.applied_known){
                if(snapshot.reset_accepted)ImGui::TextUnformatted(ui::text("gpu.clock_reset_done",zh));
                else ImGui::Text(ui::text("gpu.clock_applied_range",zh),snapshot.applied.low,snapshot.applied.high);
            }else ui::secondary_text(ui::text("gpu.clock_current_unknown",zh));
            if(ui::button("clock-apply",ui::text("gpu.clock_apply",zh),ui::ButtonKind::Primary,p,scale,!valid||!snapshot.can_apply,nullptr,true)){
                if(gpu_clocks.apply(index,uuid,draft.range)){clock_pending[index]={draft.range,draft.automatic,ClockAutoReason::None};draft.message_key.clear();}
            }display.apply=center();
            ImGui::SameLine();
            if(ui::button("clock-save",ui::text("gpu.clock_save",zh),ui::ButtonKind::Secondary,p,scale,!valid||!ready||!clock_profiles_valid,nullptr,true)){
                const bool success=save_gpu_clock_range(uuid,draft.range,draft.automatic);toast(status,!success);draft.message_key=success?"gpu.clock_saved":"";
            }display.save=center();
            if(!draft.message_key.empty())ui::secondary_text(ui::text(draft.message_key.c_str(),zh));
            ImGui::EndDisabled();display.draft=draft.range;
        }
        ImGui::BeginDisabled(snapshot.busy);
        if(ui::button("clock-reset",ui::text("gpu.clock_reset",zh),ui::ButtonKind::Secondary,p,scale,!snapshot.can_reset,nullptr,true))gpu_clocks.reset(index,uuid);
        display.reset=center();ImGui::SameLine();
        if(ui::button("clock-refresh",ui::text("gpu.clock_refresh",zh),ui::ButtonKind::Ghost,p,scale,false,nullptr,true))gpu_clocks.query(index,expected_uuid);
        display.refresh=center();ImGui::EndDisabled();
        if(!ready)ui::secondary_text(ui::text("gpu.clock_open_config",zh));
        if(!clock_profiles_valid)ui::secondary_text(ui::text("gpu.clock_config_invalid",zh));
        ui::secondary_text(ui::text("gpu.clock_actual_hint",zh));
        ImGui::PopTextWrapPos();ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
}
void App::tick_gpu_clocks(){
    const bool zh=state.ui_language=="zh";const auto now=GetTickCount64();const auto hour=clock_local_hour();
    for(const auto& device:gpu.devices()){
        auto snapshot=gpu_clocks.get(device.index);auto& seen=clock_completions[device.index];
        if(snapshot.completed!=seen){
            seen=snapshot.completed;
            if(snapshot.action!=ClockAction::Query){
                const auto pending=clock_pending.find(device.index);
                if(!snapshot.error){
                    auto& policy=clock_auto_policies[snapshot.uuid];
                    if(snapshot.action==ClockAction::Reset){policy.enabled=false;policy.reset_violation();clock_drafts[snapshot.uuid].automatic=false;}
                    else if(pending!=clock_pending.end()&&pending->second.reason==ClockAutoReason::None){
                        policy.range=pending->second.range;policy.enabled=pending->second.automatic;policy.startup=false;policy.checked_hour=hour;policy.reset_violation();
                    }
                }
                const auto* key=snapshot.error?clock_error_key(snapshot.error):snapshot.action==ClockAction::Reset?"gpu.clock_reset_done":
                    pending!=clock_pending.end()&&pending->second.reason!=ClockAutoReason::None?clock_auto_key(pending->second.reason):"gpu.clock_applied";
                toast("#"+std::to_string(device.index)+" "+ui::text(key,zh),snapshot.error!=0);
                trace("GPU clocks: "+snapshot.uuid+" action="+std::to_string(int(snapshot.action))+" reason="+std::to_string(pending==clock_pending.end()?0:int(pending->second.reason))+" NVML="+std::to_string(snapshot.error));
                clock_pending.erase(device.index);
            }
        }
        const auto uuid=gpu_details.get(device.index).uuid;if(uuid.empty())continue;
        const auto found=clock_auto_policies.find(uuid);if(found==clock_auto_policies.end()||!found->second.enabled)continue;
        auto& policy=found->second;
        if(!snapshot.queried){if(!snapshot.busy)gpu_clocks.query(device.index,uuid);continue;}
        if(snapshot.busy)continue;
        if(snapshot.uuid!=uuid||!snapshot.can_apply||!snapshot.supported.valid()){
            if(policy.startup||policy.checked_hour!=hour){policy.startup=false;policy.checked_hour=hour;
                toast(ui::text(snapshot.error?clock_error_key(snapshot.error):"gpu.clock_unsupported",zh),true);trace("GPU clock auto apply unavailable: "+uuid);}
            continue;
        }
        ecmgui::GpuSample sample;gpu.latest(device.index,sample); // Same 500 ms snapshot used by the charts; no driver read here.
        const auto reason=policy.evaluate(now,hour,sample,false);if(reason==ClockAutoReason::None)continue;
        if(reason==ClockAutoReason::Deviation){
            char text[256];std::snprintf(text,sizeof(text),ui::text("gpu.clock_unexpected",zh),device.index,sample.clock_sm_mhz,policy.range.low,policy.range.high);
            trace(text);
        }
        if(gpu_clocks.apply(device.index,uuid,policy.range))clock_pending[device.index]={policy.range,true,reason};
        else{toast(ui::text("gpu.clock_invalid",zh),true);trace("GPU clock auto apply rejected by current range: "+uuid);}
    }
}
}
