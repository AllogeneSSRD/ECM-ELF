#include "../app.h"
#include "../ui/i18n.h"
#include "../ui/curve_selector.h"
#include "../ui/presentation.h"
#include <commdlg.h>
#include <filesystem>
#include <algorithm>
namespace ecmgui_v2 {
namespace {
bool edit(const char*id,std::string&value,size_t limit=4097){std::vector<char> data(std::max(limit,value.size()+1));std::copy(value.begin(),value.end(),data.begin());if(!ImGui::InputText(id,data.data(),data.size()))return false;value=data.data();return true;}
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
    if(!group_order_loaded){group_order_input.gp=ready?config.get("gui","group_order_gp"):"";group_order_loaded=true;group_order_base=base;group_order_records.clear();group_order_file=-1;refresh();}
    group_order_controls.clear();
    ImGui::TextUnformatted(ui::text("go.title",zh));ui::secondary_text(ui::text("go.description",zh));
    ui::curve_buttons("go-curves",group_order_input.profile,false,zh,group_order_curve_controls,true);
    {ui::CompactForm form("go-fields",group_order_form);
        form.field("factor",ui::text("go.factor",zh),[&]{edit("##factor",group_order_input.factor,1235);});
        form.field("sigma","sigma",[&]{edit("##sigma",group_order_input.sigma,21);});}
    ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(ui::text("go.gp_path",zh));ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(80*scale,ImGui::GetContentRegionAvail().x-140*scale));edit("##go-gp",group_order_input.gp);group_order_controls["gp"]=center();
    ImGui::SameLine();if(ui::button("go-browse-gp",ui::text("go.browse",zh),ui::ButtonKind::Secondary,p,scale,false,nullptr,true)){auto path=browse(true);if(!path.empty())group_order_input.gp=path;}
    if(ui::button("go-save",ui::text("go.save_path",zh),ui::ButtonKind::Secondary,p,scale,!ready,nullptr,true))save_group_order_gp();group_order_save=center();
    ImGui::TextWrapped("%s",ui::text("go.gp_hint",zh));
    const char*backends[]{ui::text("go.auto",zh),"PARI/GP",ui::text("go.native",zh)};
    for(int i=0;i<3;++i){if(i)ImGui::SameLine();const bool selected=group_order_input.backend==i;if(selected)ImGui::PushStyleColor(ImGuiCol_Button,ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));ImGui::PushID(i);if(ImGui::Button(backends[i]))group_order_input.backend=i;group_order_controls["backend"+std::to_string(i)]=center();ImGui::PopID();if(selected)ImGui::PopStyleColor();}
    if(ui::button("go-calculate",ui::text("go.calculate",zh),ui::ButtonKind::Primary,p,scale,group_order_job.busy(),nullptr,true)){
        auto request=group_order_input;
        if(!request.gp.empty()&&fs::u8path(request.gp).is_relative()){
            const auto path=fs::u8path(base)/fs::u8path(request.gp);if(request.gp.find_first_of("/\\")!=std::string::npos||fs::exists(path))request.gp=path.lexically_normal().u8string();}
        if(!group_order_job.start(request,ecmgui::exe_dir()+"/ecm_group_order_ui.exe"))toast(group_order_job.error,true);
    }group_order_calculate=center();ImGui::SameLine();
    if(ui::button("go-cancel",ui::text("go.cancel",zh),ui::ButtonKind::Secondary,p,scale,!group_order_job.busy(),nullptr,true))group_order_job.cancel();group_order_cancel=center();
    if(group_order_job.busy())ui::secondary_text(ui::text("go.running",zh));
    if(!group_order_job.error.empty())ImGui::TextWrapped("%s",group_order_job.error.c_str());
    const auto&r=group_order_job.result;
    if(r.type=='o'){
        ImGui::Separator();const auto&snapshot=group_order_job.snapshot;
        ImGui::TextWrapped("p = %s | sigma = %s | %s",snapshot.factor.c_str(),snapshot.sigma.c_str(),ui::curve_presets[snapshot.profile].label);
        // Path resolution belongs to the snapshot; compare the editable mathematical identity.
        if(snapshot.factor!=group_order_input.factor||snapshot.sigma!=group_order_input.sigma||snapshot.profile!=group_order_input.profile)ui::secondary_text(ui::text("go.stale",zh));
        ImGui::Text("%s: %s | %s: %s",ui::text("go.status",zh),ui::text(("go.status."+r.get("status").text()).c_str(),zh),ui::text("go.backend",zh),r.get("backend").text().c_str());
        ImGui::Text("%s: %s | %s: %s",ui::text("go.primality",zh),ui::text(("go.primality."+r.get("modulus_primality").text()).c_str(),zh),ui::text("go.proof",zh),ui::text(r.get("all_proven").scalar=="true"?"go.proven":"go.conditional",zh));
        const bool group=r.get("group_computed").scalar=="true",point=r.get("status").text()=="complete"&&r.get("point_minimality").scalar=="true";
        ImGui::TextWrapped("%s: %s",ui::text("go.group_order",zh),group?r.get("group_order").text().c_str():ui::text("go.incomplete",zh));
        ImGui::TextWrapped("%s: %s",ui::text("go.point_order",zh),point?r.get("point_order").text().c_str():ui::text("go.incomplete",zh));
        const auto show_factors=[&](const char*key,const char*title){ImGui::TextUnformatted(ui::text(title,zh));std::string value;
            for(const auto&f:r.get(key).items){if(!value.empty())value+=" * ";value+=f.get("prime").text()+"^"+f.get("exponent").text()+" ["+ui::text(("go.primality."+f.get("primality").text()).c_str(),zh)+"]";}if(value.empty()&&point&&r.get("point_order").text()=="1"&&std::string(key)=="point_factors")value="1";ImGui::TextWrapped("%s",value.empty()?ui::text("go.incomplete",zh):value.c_str());};
        if(group)show_factors("group_factors","go.group_factorization");if(point)show_factors("point_factors","go.point_factorization");
        if(r.get("cofactor").text()!="1"&&r.get("cofactor").text()!="")ImGui::TextWrapped("%s: %s",ui::text("go.remaining",zh),r.get("cofactor").text().c_str());
        if(!r.get("reason").text().empty())ImGui::TextWrapped("%s",r.get("reason").text().c_str());
        if(!r.get("fallback").text().empty())ImGui::TextWrapped("%s: %s",ui::text("go.fallback",zh),r.get("fallback").text().c_str());
        if(ImGui::Button(ui::text("go.copy",zh)))ImGui::SetClipboardText(r.dump().c_str());
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
