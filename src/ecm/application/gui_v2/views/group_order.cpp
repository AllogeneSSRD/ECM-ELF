#include "../app.h"
#include "../ui/i18n.h"
#include "../ui/curve_selector.h"
#include "../ui/presentation.h"
#include "../ui/group_order_presentation.h"
#include <commdlg.h>
#include <filesystem>
#include <algorithm>
namespace ecmgui_v2 {
namespace {
bool edit(const char*id,std::string&value,size_t limit=4097,const char*hint=nullptr){std::vector<char> data(std::max(limit,value.size()+1));std::copy(value.begin(),value.end(),data.begin());if(!ImGui::InputTextWithHint(id,hint?hint:"",data.data(),data.size()))return false;value=data.data();return true;}
std::string browse(bool executable){
    wchar_t name[32768]{};OPENFILENAMEW dialog{sizeof(dialog)};dialog.hwndOwner=GetActiveWindow();dialog.lpstrFile=name;dialog.nMaxFile=32768;
    dialog.lpstrFilter=executable?L"PARI/GP\0*.exe\0All files\0*.*\0":L"ECM results\0*.txt;*.json;*.jsonl\0All files\0*.*\0";
    dialog.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|OFN_NOCHANGEDIR|OFN_EXPLORER;
    return GetOpenFileNameW(&dialog)?std::filesystem::path(name).u8string():std::string();
}
bool same(const ecmgui::FileStamp&a,const ecmgui::FileStamp&b){return a.exists==b.exists&&a.size==b.size&&a.mtime==b.mtime&&a.hash==b.hash;}
}
bool App::save_group_order_gp(){
    const bool zh=state.ui_language=="zh";if(!ready){toast(ui::text("go.open_config",zh),true);return false;}
    ecmgui::FileStamp current;std::string error;ecmgui::TomlFile disk;
    if(!ecmgui::stamp_file(config_path,current)||!same(current,config_stamp)){toast(ui::text("gpu.clock_config_changed",zh),true);return false;}
    if(!disk.load(config_path,error)){toast(error,true);return false;}
    const bool dirty=config.dirty();auto draft=config;draft.set("gui","group_order_gp",group_order_input.gp);disk.set("gui","group_order_gp",group_order_input.gp);
    if(!disk.save(error)){toast(error,true);return false;}config=dirty?std::move(draft):std::move(disk);ecmgui::stamp_file(config_path,config_stamp);toast(ui::text("go.saved",zh));return true;
}
void App::draw_group_order(bool zh){
    namespace fs=std::filesystem;const auto p=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);const float scale=ImGui::GetFontSize()/15.f;
    const auto center=[](){auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();return ImVec2{(a.x+b.x)*.5f,(a.y+b.y)*.5f};};
    const std::string base=ready?fs::u8path(config_path).parent_path().u8string():ecmgui::exe_dir();
    const auto refresh=[&](){
        const std::string selected=group_order_file>=0&&group_order_file<int(group_order_files.size())?group_order_files[group_order_file]:"";
        std::vector<std::string> configured;for(const auto&file:results.files())configured.push_back(file.path);if(!selected.empty())configured.push_back(selected);
        group_order_files=discover_order_files(base,configured);const auto it=std::find(group_order_files.begin(),group_order_files.end(),selected);
        group_order_file=it==group_order_files.end()?-1:int(it-group_order_files.begin());group_order_records.clear();group_order_file_error.clear();
        if(group_order_file>=0)group_order_records=read_order_records(group_order_files[group_order_file],group_order_file_error);
    };
    if(!group_order_loaded){group_order_input.gp=ready?config.get("gui","group_order_gp"):"";if(group_order_input.gp.empty())group_order_input.gp=find_order_gp();group_order_loaded=true;group_order_base=base;group_order_records.clear();group_order_file=-1;refresh();}
    group_order_controls.clear();
    ImGui::TextUnformatted(ui::text("go.title",zh));ui::secondary_text(ui::text("go.description",zh));
    ui::curve_buttons("go-curves",group_order_input.profile,false,zh,group_order_curve_controls,true);
    {ui::CompactForm form("go-fields",group_order_form);
        form.field("factor",ui::text("go.factor",zh),[&]{edit("##factor",group_order_input.factor,1235);});
        form.field("sigma","sigma",[&]{edit("##sigma",group_order_input.sigma,21,ui::text(group_order_input.profile==0?"go.sigma.suyama_range":"go.sigma.range",zh));if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("go.sigma.hint",zh));});}
    const char*backends[]{ui::text("go.auto",zh),"PARI/GP",ui::text("go.native",zh)};
    const float available=ImGui::GetContentRegionAvail().x,row_height=24*scale;
    float combo_width=0;for(auto name:backends)combo_width=std::max(combo_width,ImGui::CalcTextSize(name).x);
    combo_width=std::min(combo_width+ImGui::CalcTextSize(ui::text("go.algorithm",zh)).x+50*scale,available*.4f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*scale,4.5f*scale});
    if(ImGui::BeginTable("go-gp-row",5,ImGuiTableFlags_SizingStretchProp|ImGuiTableFlags_NoPadOuterX)){
        ImGui::TableSetupColumn("label",ImGuiTableColumnFlags_WidthFixed,ImGui::CalcTextSize("GP").x);
        ImGui::TableSetupColumn("path",ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("browse",ImGuiTableColumnFlags_WidthFixed,ImGui::CalcTextSize(ui::text("go.browse",zh)).x+16*scale);
        ImGui::TableSetupColumn("save",ImGuiTableColumnFlags_WidthFixed,ImGui::CalcTextSize(ui::text("go.save_path",zh)).x+16*scale);
        ImGui::TableSetupColumn("algorithm",ImGuiTableColumnFlags_WidthFixed,combo_width);
        ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::AlignTextToFramePadding();ui::secondary_text("GP");ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);std::vector<char> path(std::max(size_t(4097),group_order_input.gp.size()+1));std::copy(group_order_input.gp.begin(),group_order_input.gp.end(),path.begin());
        if(ImGui::InputTextWithHint("##go-gp",ui::text("go.select_gp",zh),path.data(),path.size()))group_order_input.gp=path.data();group_order_controls["gp"]=center();
        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",group_order_input.gp.c_str(),ui::text("go.gp_hint",zh));ImGui::TableNextColumn();
        if(ui::button("go-browse-gp",ui::text("go.browse",zh),ui::ButtonKind::Secondary,p,scale,false,nullptr,true,-1,row_height)){auto selected=browse(true);if(!selected.empty())group_order_input.gp=selected;}group_order_controls["browse"]=center();ImGui::TableNextColumn();
        if(ui::button("go-save",ui::text("go.save_path",zh),ui::ButtonKind::Secondary,p,scale,!ready,nullptr,true,-1,row_height))save_group_order_gp();group_order_save=center();ImGui::TableNextColumn();
        const std::string preview=std::string(ui::text("go.algorithm",zh))+": "+backends[group_order_input.backend];ImGui::SetNextItemWidth(-1);
        const bool open=ImGui::BeginCombo("##go-algorithm",preview.c_str());group_order_controls["algorithm"]=center();
        if(open){for(int i=0;i<3;++i){if(ImGui::Selectable(backends[i],group_order_input.backend==i))group_order_input.backend=i;group_order_controls["backend"+std::to_string(i)]=center();}ImGui::EndCombo();}
        ImGui::EndTable();
    }ImGui::PopStyleVar();
    if(ui::button("go-calculate",ui::text("go.calculate",zh),ui::ButtonKind::Primary,p,scale,group_order_job.busy(),nullptr,true,0,0,ui::Icon::Calculate)){
        auto request=group_order_input;
        if(!request.gp.empty()&&fs::u8path(request.gp).is_relative()){
            const auto path=fs::u8path(base)/fs::u8path(request.gp);if(request.gp.find_first_of("/\\")!=std::string::npos||fs::exists(path))request.gp=path.lexically_normal().u8string();}
        const bool started=group_order_job.start(request,ecmgui::exe_dir()+"/ecm_group_order_ui.exe");
        if(started&&request.gp==group_order_input.gp&&!group_order_job.snapshot.gp.empty())group_order_input.gp=group_order_job.snapshot.gp;
        // GP-only refusal can finish synchronously; poll() has no process to observe.
        if(!group_order_job.busy())notify_group_order();
    }group_order_calculate=center();ImGui::SameLine();
    if(ui::button("go-cancel",ui::text("go.cancel",zh),ui::ButtonKind::Secondary,p,scale,!group_order_job.busy(),nullptr,true)){group_order_job.cancel();toast(ui::text("go.cancelled",zh),false,true);}group_order_cancel=center();
    if(group_order_job.busy())ui::secondary_text(ui::text("go.running",zh));
    if(!group_order_job.error.empty())ImGui::TextWrapped("%s",group_order_job.error.c_str());
    group_order_copy.clear();group_order_selection=0;group_order_output_min=group_order_output_max={};
    const auto&r=group_order_job.result;
    if(r.type=='o'&&r.has("status")){
        ImGui::Separator();const auto&snapshot=group_order_job.snapshot;
        ImGui::TextWrapped("p = %s | sigma = %s | %s",snapshot.factor.c_str(),snapshot.sigma.c_str(),ui::curve_presets[snapshot.profile].label);
        // Path resolution belongs to the snapshot; compare the editable mathematical identity.
        if(snapshot.factor!=group_order_input.factor||snapshot.sigma!=group_order_input.sigma||snapshot.profile!=group_order_input.profile)ui::secondary_text(ui::text("go.stale",zh));
        const std::string status=ui::text(("go.status."+r.get("status").text()).c_str(),zh);
        const std::string summary=status+" | "+r.get("backend").text()+" | "+ui::text(r.get("all_proven").scalar=="true"?"go.proven":"go.conditional",zh);
        ImGui::TextWrapped("%s",summary.c_str());
        if(ImGui::IsItemHovered()){
            ImGui::BeginTooltip();ImGui::Text("%s: %s",ui::text("go.primality",zh),ui::text(("go.primality."+r.get("modulus_primality").text()).c_str(),zh));
            if(!r.get("fallback").text().empty())ImGui::TextWrapped("%s: %s",ui::text("go.fallback",zh),r.get("fallback").text().c_str());ImGui::EndTooltip();
        }
        const bool group=r.get("group_computed").scalar=="true",point=r.get("status").text()=="complete"&&r.get("point_minimality").scalar=="true";
        group_order_copy="p = "+snapshot.factor+" | sigma = "+snapshot.sigma+" | "+ui::curve_presets[snapshot.profile].label+"\n"+status+" | "+r.get("backend").text()+"\n";
        std::string output;
        const auto show_order=[&](bool is_point,bool computed){
            const std::string first=std::string(ui::text(is_point?"go.point_order":"go.group_order",zh))+": "+(computed?r.get(is_point?"point_order":"group_order").text():ui::text("go.incomplete",zh));
            if(!output.empty())output+="\n";output+=first;group_order_copy+=first+"\n";
            if(computed){
                auto factors=ui::order_factorization(r,is_point);
                const std::string second="= "+(factors.empty()?std::string(ui::text("go.incomplete",zh)):factors);
                output+="\n    "+second;
                group_order_copy+="    "+second+"\n";
            }
        };
        show_order(false,group);show_order(true,point);
        // Use native text editing interaction in read-only mode, with the ordinary
        // body font and transparent frame. Keep one selection across both orders.
        std::vector<char> selectable(output.begin(),output.end());selectable.push_back(0);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,0);ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{0,2*scale});
        ImGui::PushStyleColor(ImGuiCol_FrameBg,{0,0,0,0});
        const float wrap=std::max(1.f,ImGui::GetContentRegionAvail().x-ImGui::GetStyle().ScrollbarSize);
        const float height=ImGui::CalcTextSize(output.c_str(),nullptr,false,wrap).y+ImGui::GetTextLineHeight()+4*scale;
        ImGui::PushID(int(group_order_completions));
        ImGui::InputTextMultiline("##go-orders",selectable.data(),selectable.size(),{-1,height},ImGuiInputTextFlags_ReadOnly|ImGuiInputTextFlags_WordWrap|ImGuiInputTextFlags_NoUndoRedo);
        group_order_output_min=ImGui::GetItemRectMin();group_order_output_max=ImGui::GetItemRectMax();group_order_controls["output"]=center();
        if(auto* input=ImGui::GetInputTextState(ImGui::GetItemID()))group_order_selection=std::abs(input->GetSelectionEnd()-input->GetSelectionStart());
        if(r.get("cofactor").text()!="1"&&!r.get("cofactor").text().empty()&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text("go.remaining_hint",zh));
        ImGui::PopID();ImGui::PopStyleColor();ImGui::PopStyleVar(2);
        if(!r.get("reason").text().empty())ImGui::TextWrapped("%s",r.get("reason").text().c_str());
        if(ui::button("go-copy",ui::text("go.copy",zh),ui::ButtonKind::Secondary,p,scale,false,nullptr,true))ImGui::SetClipboardText(group_order_copy.c_str());group_order_controls["copy"]=center();
    }
    ImGui::Separator();ImGui::TextUnformatted(ui::text("go.result_files",zh));
    if(ImGui::Button(ui::text("go.refresh",zh)))refresh();group_order_controls["refresh"]=center();ImGui::SameLine();
    if(ImGui::Button(ui::text("go.choose_file",zh))){auto path=browse(false);if(!path.empty()){auto it=std::find(group_order_files.begin(),group_order_files.end(),path);if(it==group_order_files.end()){group_order_files.push_back(path);group_order_file=int(group_order_files.size())-1;}else group_order_file=int(it-group_order_files.begin());group_order_records=read_order_records(path,group_order_file_error);}}
    if(group_order_file>=int(group_order_files.size()))group_order_file=-1;
    const std::string preview=group_order_file<0?ui::text("go.select_file",zh):fs::u8path(group_order_files[group_order_file]).filename().u8string();
    const bool open=ImGui::BeginCombo(ui::text("source",zh),preview.c_str());group_order_controls["files"]=center();
    if(open){
        for(size_t i=0;i<group_order_files.size();++i){if(ImGui::Selectable(group_order_files[i].c_str(),group_order_file==int(i))){group_order_file=int(i);group_order_records=read_order_records(group_order_files[i],group_order_file_error);}group_order_controls["file"+std::to_string(i)]=center();}ImGui::EndCombo();}
    if(!group_order_file_error.empty())ImGui::TextWrapped("%s",group_order_file_error.c_str());
    ui::secondary_text(ui::text("go.records_hint",zh));
    if(ImGui::BeginChild("go-records",{0,170*scale},ImGuiChildFlags_Borders)){
        ImGuiListClipper clip;clip.Begin(int(group_order_records.size()));while(clip.Step())for(int i=clip.DisplayStart;i<clip.DisplayEnd;++i){const auto&record=group_order_records[i];ImGui::PushID(i);
            const auto label=ui::fitted_label(record.label.c_str(),ImGui::GetContentRegionAvail().x);
            if(ImGui::Selectable(label.c_str())){group_order_input.factor=record.input.factor;group_order_input.sigma=record.input.sigma;group_order_input.profile=record.input.profile;}
            group_order_controls["record"+std::to_string(i)]=center();
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",record.label.c_str());ImGui::PopID();}
    }ImGui::EndChild();
}
}
