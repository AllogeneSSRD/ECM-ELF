#include "../app.h"
#include "../generated/business_options.h"
#include "../ui/probability_presets.h"
#include "../ui/curve_selector.h"
#include "../ui/chart.h"
#include "../ui/navigation.h"
#include "../ui/i18n.h"
#include "../ui/presentation.h"
#include "../generated/pet_skins.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <tuple>

namespace ecmgui_v2 {
namespace {
bool text(const char* label,std::string& value,size_t limit=1024) {
    std::vector<char> buffer(std::max(limit,value.size()+1)); std::copy(value.begin(),value.end(),buffer.begin());
    if (!ImGui::InputText(label,buffer.data(),buffer.size())) return false;
    value=buffer.data(); return true;
}
bool same_stamp(const ecmgui::FileStamp& a,const ecmgui::FileStamp& b) {
    return a.exists==b.exists && a.size==b.size && a.mtime==b.mtime && a.hash==b.hash;
}
void number(const char* label,long long& value) { ImGui::InputScalar(label,ImGuiDataType_S64,&value); }
std::string option_table(std::string table) {
    if(ecm_config::toml_worker(table)>0){const auto dot=table.find('.',8);return dot==std::string::npos?"workers.1":table.substr(dot+1);}return table;
}
const BusinessOption* selected_option(const char* table,const char* key) {
    const auto base=option_table(table);for(const auto& option:business_options)if(base==option.table&&std::string(key)==option.key)return &option;return nullptr;
}
bool matches(const std::string& value,const std::string& query) {
    auto lower=[](std::string s){for(auto& c:s)if(static_cast<unsigned char>(c)<128)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;};
    return query.empty()||lower(value).find(lower(query))!=std::string::npos;
}
bool chips(const char* id,int& value,const std::vector<const char*>& labels,ImVec2* centers=nullptr){
    bool changed=false;ImGui::PushID(id);int index=0;
    for(const auto label:labels){
        const auto width=ImGui::CalcTextSize(label).x+ImGui::GetStyle().FramePadding.x*2;
        if(index&&ImGui::GetItemRectMax().x+width+ImGui::GetStyle().ItemSpacing.x<ImGui::GetWindowPos().x+ImGui::GetWindowWidth()-ImGui::GetStyle().WindowPadding.x)ImGui::SameLine();
        const bool selected=value==index;
        if(selected)ImGui::PushStyleColor(ImGuiCol_Button,ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
        if(ImGui::Button(label)){value=index;changed=true;}
        if(centers)centers[index]=ImVec2((ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f);
        if(selected)ImGui::PopStyleColor();
        ++index;
    }ImGui::PopID();return changed;
}

}
void App::draw_results(bool zh) {
    if(result_filter.source>=static_cast<int>(results.files().size()))result_filter.source=-1;
    ImGui::TextUnformatted(ecmgui_v2::ui::text("published_results",zh));
    const char* statuses[]{"All","F","NF"}; int status_index=result_filter.status=="F"?1:result_filter.status=="NF"?2:0;
    if (chips("result-status",status_index,{ecmgui_v2::ui::text("all_statuses",zh),"F","NF"},result_chips)) result_filter.status=status_index?statuses[status_index]:"";
    chips("result-stage",result_filter.stage,{ecmgui_v2::ui::text("all_stages",zh),"Stage 1","Stage 2"});
    ImGui::InputText(ecmgui_v2::ui::text("search",zh),result_search,sizeof(result_search)); result_filter.search=result_search;
    if (ImGui::BeginCombo(ecmgui_v2::ui::text("source",zh),result_filter.source<0?ecmgui_v2::ui::text("all",zh):results.files()[static_cast<size_t>(result_filter.source)].path.c_str())) {
        if (ImGui::Selectable(ecmgui_v2::ui::text("all",zh),result_filter.source<0)) result_filter.source=-1;
        for(size_t i=0;i<results.files().size();++i) if(ImGui::Selectable(results.files()[i].path.c_str(),result_filter.source==static_cast<int>(i))) result_filter.source=static_cast<int>(i);
        ImGui::EndCombo();
    }
    const auto signature=result_filter.status+'\n'+std::to_string(result_filter.stage)+'\n'+std::to_string(result_filter.source)+'\n'+result_filter.search;
    if(result_view_dirty||signature!=result_view_signature){
        result_indices=results.filtered(result_filter);visible_result_ids.clear();
        for(auto i:result_indices)visible_result_ids.insert(results.rows()[i].id);
        for(auto it=selected_results.begin();it!=selected_results.end();)if(!visible_result_ids.count(*it))it=selected_results.erase(it);else ++it;
        result_view_dirty=false;result_view_signature=signature;
    }
    const auto& indices=result_indices;const auto& visible=visible_result_ids;
    if(ImGui::Button(ecmgui_v2::ui::text("select_visible",zh))) selected_results=visible;
    ImGui::SameLine(); if(ImGui::Button(ecmgui_v2::ui::text("clear_selection",zh)))selected_results.clear();
    ImGui::SameLine(); if(ImGui::Button(ecmgui_v2::ui::text("copy_original_json_checksum",zh))) {
        std::vector<size_t> copy; for(auto i:indices)if(selected_results.count(results.rows()[i].id))copy.push_back(i);
        ImGui::SetClipboardText(results.copy_text(copy).c_str());
    }
    ImGui::Text(ecmgui_v2::ui::text("results.selection_summary",zh),indices.size(),results.rows().size(),selected_results.size());
    for(const auto& f:results.files()) if(!f.error.empty()||f.invalid_lines||f.pending_lines)
        ImGui::TextWrapped(ecmgui_v2::ui::text("results.source_summary",zh),f.path.c_str(),f.error.c_str(),f.invalid_lines,f.pending_lines);
    if(ImGui::BeginTable("results",11,ImGuiTableFlags_ScrollX|ImGuiTableFlags_ScrollY|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable,ImVec2(0,ImGui::GetContentRegionAvail().y))) {
        result_scroll=ImGui::GetScrollY();result_wheel=ImVec2(ImGui::GetWindowPos().x+80,ImGui::GetWindowPos().y+ImGui::GetWindowSize().y*.6f);
        for(const char* title:{ecmgui_v2::ui::text("select",zh),ecmgui_v2::ui::text("status",zh),ecmgui_v2::ui::text("target",zh),"B1","B2",ecmgui_v2::ui::text("curves",zh),ecmgui_v2::ui::text("model",zh),ecmgui_v2::ui::text("time",zh),ecmgui_v2::ui::text("factors",zh),ecmgui_v2::ui::text("stage",zh),ecmgui_v2::ui::text("source",zh)}) ImGui::TableSetupColumn(title,ImGuiTableColumnFlags_WidthFixed,110);
        ImGui::TableSetupScrollFreeze(2,1);ImGui::TableHeadersRow(); ImGuiListClipper clip;clip.Begin(static_cast<int>(indices.size()));
        while(clip.Step())for(int j=clip.DisplayStart;j<clip.DisplayEnd;++j) {
            const auto& row=results.rows()[indices[static_cast<size_t>(j)]]; ImGui::PushID(row.id.c_str());ImGui::TableNextRow();ImGui::TableNextColumn();
            bool chosen=selected_results.count(row.id)!=0;if(ecmgui_v2::ui::checkbox("##select",&chosen)){if(chosen)selected_results.insert(row.id);else selected_results.erase(row.id);}
            for(const auto* value:{&row.status,&row.target,&row.b1,&row.b2,&row.curves,&row.model,&row.timestamp,&row.factors}){ImGui::TableNextColumn();ImGui::TextUnformatted(value->c_str());}
            ImGui::TableNextColumn();ImGui::Text("%d",row.stage);ImGui::TableNextColumn();ImGui::Text("%zu:%zu",row.source+1,row.line);ImGui::PopID();
        } ImGui::EndTable();
    }
}
std::string App::generator_signature() const {
    std::ostringstream s;s<<gen_input<<'\n'<<gen_worker<<'\n'<<gen_b1<<'\n'<<gen_b2<<'\n'<<gen_pattern<<'\n'
        <<gen_options.curve_type<<' '<<gen_options.use_recommended<<' '<<gen_options.curves_fixed<<' '<<gen_options.blocks_per_sm
        <<' '<<gen_options.dedup<<' '<<gen_options.sort_by<<' '<<gen_options.sort_desc<<' '<<gen_options.sort_factors
        <<' '<<gen_options.skip_curves<<' '<<gen_options.min_n<<' '<<gen_options.max_n<<' '<<gen_options.min_curves<<' '<<gen_options.max_curves<<' '<<gen_options.set_has_na;
    return s.str();
}
bool App::preview_generator() {
    gen_valid=false; if(!ready || controller.busy() || config.dirty()){status="Stop workers and save workflow before generating.";return false;}
    const ecmgui::WorkerView* target=nullptr;for(const auto& w:controller.workers())if(w.index==gen_worker)target=&w;
    if(!target){status="Select a valid target worker.";return false;}
    if(target->stage1_target=="cpu" && (gen_options.use_recommended||gen_options.curve_type=="m2"||gen_options.curve_type=="m3")) {
        status="CPU supports e/m0 with explicit curve count; GPU recommendation and m2/m3 are unavailable.";return false;
    }
    gen_options.valid=true;gen_options.set_b1=gen_b1;gen_options.set_b2=gen_b2;gen_options.save_pattern=gen_pattern;
    gen_options.target_device=target->stage1_device;gen_options.worker_devices={{gen_worker,target->stage1_device}};
    gen_path=target->worktodo;
    if(gen_options.use_recommended) {
        std::string cmd='"'+controller.resolve_stage1_exe()+"\" --config \""+config_path+"\" --worker "+std::to_string(gen_worker)+" --gpu-info -d "+std::to_string(target->stage1_device)+" --curve "+(gen_options.curve_type=="e"?"ed":"mont");
        if(gen_options.curve_type!="e")cmd+=" --gpu-param "+gen_options.curve_type.substr(1);
        std::string output,error;int code=-1;
        if(!ecmgui::run_capture(cmd,output,code,20000)||code||!ecmgui::parse_gpu_info(output,gen_gpu,error)) {status="GPU profile unavailable: "+error;return false;}
    }
    gen_preview=ecmgui::generate(gen_input,gen_gpu,gen_options);
    if(!gen_preview.ok){status=gen_preview.error;return false;}
    if(!ecmgui::stamp_file(gen_path,gen_stamp)&&gen_stamp.exists){status="Cannot read target queue";return false;}
    gen_signature=generator_signature();gen_valid=true;status="Preview ready: "+gen_path;trace("generator: preview ready");return true;
}
bool App::apply_generator() {
    if(!gen_valid || generator_signature()!=gen_signature || controller.busy() || config.dirty()) {gen_valid=false;status="Preview expired; generate again.";return false;}
    try {
        // Block a consumer during the short append, including an unchanged old GUI's updated worker.
        ecm_queue::OwnerLease lease(gen_path,gen_worker);
        std::string error;size_t bytes=0;
        if(!ecmgui::apply_append(gen_path,gen_preview.text,gen_stamp,error,&bytes)){status=error;gen_valid=false;return false;}
        gen_valid=false;status="Queue appended: "+std::to_string(bytes)+" bytes";trace("generator: "+status);return true;
    }catch(const std::exception& e){status=e.what();return false;}
}
void App::draw_generator(bool zh) {
    ImGui::TextUnformatted(ecmgui_v2::ui::text("task_generator",zh));
    ImGui::TextUnformatted("ECM / ECM2");ImGui::InputTextMultiline("##assignments",gen_input,sizeof(gen_input),ImVec2(-1,140));
    int curve=0;for(const auto& preset:ui::curve_presets)if(preset.generator&&gen_options.curve_type==preset.generator)curve=preset.id;
    if(ui::curve_buttons("generator-curve",curve,true,zh,generator_curve_controls))gen_options.curve_type=ui::curve_presets[curve].generator;
    {
        ui::CompactForm form("generator-fields",generator_form);
        form.field("worker",ui::text("target_worker",zh),[&]{ImGui::InputInt("##value",&gen_worker,0,0);});
        form.field("curves",ui::text(gen_options.use_recommended?"blocks_per_sm":"fixed_curves",zh),[&]{
            if(gen_options.use_recommended)ImGui::InputInt("##value",&gen_options.blocks_per_sm,0,0);
            else number("##value",gen_options.curves_fixed);
        },ui::text("gpu_recommended_curves",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("gpu_recommended_curves",zh),&gen_options.use_recommended);});
        form.field("b1","B1",[&]{ImGui::InputTextWithHint("##value",ui::text("generator.b1_hint",zh),gen_b1,sizeof(gen_b1));});
        form.field("b2","B2",[&]{ImGui::InputTextWithHint("##value",ui::text("generator.b2_hint",zh),gen_b2,sizeof(gen_b2));});
    }
    {ui::CompactFormAudit path_audit;ui::CompactForm form("generator-path",path_audit,false);
        form.field("pattern",ui::text("save_pattern",zh),[&]{ImGui::InputText("##value",gen_pattern,sizeof(gen_pattern));});}
    if(ImGui::CollapsingHeader(ui::text("filters_and_ordering",zh))) {
        ecmgui_v2::ui::checkbox(ui::text("deduplicate",zh),&gen_options.dedup);ImGui::SameLine();ecmgui_v2::ui::checkbox(ui::text("descending",zh),&gen_options.sort_desc);
        ecmgui_v2::ui::checkbox(ui::text("sort_known_factors",zh),&gen_options.sort_factors);
        ui::CompactFormAudit filter_audit;ui::CompactForm form("generator-filters",filter_audit);
        form.field("sort",ui::text("sort_keys",zh),[&]{text("##value",gen_options.sort_by);});
        form.field("skip",ui::text("skip_curves",zh),[&]{number("##value",gen_options.skip_curves);});
        form.field("n-min",ui::text("generator.min_exponent",zh),[&]{number("##value",gen_options.min_n);});
        form.field("n-max",ui::text("max_exponent",zh),[&]{number("##value",gen_options.max_n);});
        form.field("curves-min",ui::text("min_input_curves",zh),[&]{number("##value",gen_options.min_curves);});
        form.field("curves-max",ui::text("max_input_curves",zh),[&]{number("##value",gen_options.max_curves);});
        form.field("assignment","AID",[]{ImGui::TextUnformatted("N/A");},ui::text("set_assignment_n_a",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("set_assignment_n_a",zh),&gen_options.set_has_na);});
    }
    if(gen_valid && generator_signature()!=gen_signature)gen_valid=false;
    if(ui::button("generate-preview",ecmgui_v2::ui::text("generate_preview",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!preview_generator())toast(status,true);ImGui::SameLine();
    ImGui::BeginDisabled(!gen_valid||controller.busy());if(ui::button("apply-preview",ecmgui_v2::ui::text("append_to_queue",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!apply_generator())toast(status,true);ImGui::EndDisabled();
    if(!gen_preview.text.empty()) {
        ImGui::Text(ecmgui_v2::ui::text("generator.preview_summary",zh),gen_valid?ecmgui_v2::ui::text("current_preview",zh):ecmgui_v2::ui::text("expired_preview",zh),gen_preview.read,gen_preview.duplicates);
        if(ImGui::Button(ecmgui_v2::ui::text("copy_preview",zh)))ImGui::SetClipboardText(gen_preview.text.c_str());
        for(const auto& e:gen_preview.parse_errors)ImGui::TextWrapped("%s",e.c_str());for(const auto& e:gen_preview.warnings)ImGui::TextWrapped("%s",e.c_str());
        ImGui::BeginChild("preview",ImVec2(0,180),true,ImGuiWindowFlags_HorizontalScrollbar);ImGui::TextUnformatted(gen_preview.text.c_str());ImGui::EndChild();
    }
}
void App::draw_probability(bool zh) {
    using namespace ecmgui::probability_ui; auto& request=probability_request;auto& p=request.parameters;
    ImGui::TextUnformatted(ecmgui_v2::ui::text("ecm_probability",zh));
    chips("probability-view",probability_tab,{ecmgui_v2::ui::text("parameters",zh),ecmgui_v2::ui::text("charts_and_snapshot",zh)});
    if(ui::button("calculate-snapshot",ecmgui_v2::ui::text("calculate_snapshot",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)){probability_result=calculate(request);probability_ready=true;probability_tab=1;toast(probability_result.error.empty()?ecmgui_v2::ui::text("probability_snapshot_ready",zh):probability_result.error,!probability_result.error.empty());}
    calculate_control=ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(probability_ready&&probability_result.error.empty()){ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("copy_snapshot_csv",zh))){ImGui::SetClipboardText(curve_csv(probability_result).c_str());toast(ecmgui_v2::ui::text("snapshot_csv_copied",zh));}}
    if(probability_tab==0){
    const bool profile_changed=ui::curve_buttons("curve-profile",probability_profile,false,zh,probability_curve_controls);
    double size=size_from_bits(p.factor_bits,request.decimal_digits);
    {
        ui::CompactForm form("probability-fields",probability_form);
        form.field("size",ui::text("factor_size",zh),[&]{if(ImGui::InputDouble("##value",&size,0,0,"%.0f"))p.factor_bits=bits_from_size(size,request.decimal_digits);},
            ui::text("decimal_digits",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("decimal_digits",zh),&request.decimal_digits);});
        form.field("curves",ui::text("curves",zh),[&]{ImGui::InputDouble("##value",&p.curves,0,0,"%.0f");});
        form.field("b1","B1",[&]{ImGui::InputDouble("##value",&p.b1,0,0,"%.0f");},ui::text("probability.stage2_enabled",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("probability.stage2_enabled",zh),&p.stage2);});
        form.field("b2",p.ratio_mode?"B2/B1":"B2",[&]{if(p.ratio_mode)ImGui::InputDouble("##value",&p.b2_ratio,0,0,"%.6g");else ImGui::InputDouble("##value",&p.b2,0,0,"%.0f");},
            ui::text("use_b2_b1_ratio",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("use_b2_b1_ratio",zh),&p.ratio_mode);});
        form.field("divisor","D_eff",[&]{
            if(probability_profile<ui::custom_curve_profile&&profile_changed)p.effective_divisor=ui::curve_divisor(probability_profile,probability_empirical);
            if(ImGui::InputDouble("##value",&p.effective_divisor,0,0,"%.6g"))probability_profile=ui::custom_curve_profile;
        },ui::text("use_stage1_empirical_calibration",zh),[&]{
            if(ecmgui_v2::ui::checkbox(ui::text("use_stage1_empirical_calibration",zh),&probability_empirical)&&probability_profile<ui::custom_curve_profile)p.effective_divisor=ui::curve_divisor(probability_profile,probability_empirical);
        });
        form.field("target",ui::text("probability.target_percent",zh),[&]{double target=ui::percent(request.target);if(ImGui::InputDouble("##value",&target,0,0,"%.6g%%"))request.target=ui::probability(target);},
            ui::text("cumulative_target",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("cumulative_target",zh),&request.cumulative_target);});
    }
    ImGui::TextWrapped("%s",ui::text("probability.edwards_divisor_note",zh));
    int scope=static_cast<int>(p.scope);if(chips("probability-scope",scope,{"Stage 1",ui::text("combined",zh),ui::text("probability.stage2_conditional",zh)}))p.scope=static_cast<Scope>(scope);
    int solve=static_cast<int>(request.solve);if(chips("solve",solve,{ui::text("forward",zh),ui::text("curves",zh),ui::text("factor_size",zh),"B1","B2","D_eff"}))request.solve=static_cast<Solve>(solve);
    if(ImGui::CollapsingHeader(ui::text("plot_range",zh))){
        ui::CompactFormAudit range_audit;ui::CompactForm form("probability-range",range_audit);
        form.field("min",ui::text("size_min",zh),[&]{ImGui::InputDouble("##value",&request.size_min,0,0,"%.0f");});
        form.field("max",ui::text("size_max",zh),[&]{ImGui::InputDouble("##value",&request.size_max,0,0,"%.0f");});
        form.field("curves-max",ui::text("curves_max",zh),[&]{ImGui::InputDouble("##value",&request.curves_max,0,0,"%.0f");},
            ui::text("auto_curves_range",zh),[&]{ecmgui_v2::ui::checkbox(ui::text("auto_curves_range",zh),&request.auto_curves_range);});
    }
        return;
    }
    if(!probability_ready){ImGui::TextWrapped("%s",ecmgui_v2::ui::text("probability.calculate_hint",zh));return;}
    const auto& r=probability_result;if(!r.error.empty()){ImGui::TextWrapped("%s",r.error.c_str());return;}
    const auto values=[](const Request& q){const auto& v=q.parameters;return std::make_tuple(v.factor_bits,v.b1,v.b2,v.b2_ratio,v.curves,v.effective_divisor,v.stage2,v.ratio_mode,v.scope,q.solve,q.target,q.cumulative_target,q.decimal_digits,q.auto_curves_range,q.size_min,q.size_max,q.curves_max);};
    if(values(request)!=values(r.request))ImGui::TextWrapped("%s",ecmgui_v2::ui::text("probability.stale_snapshot_hint",zh));
    ImGui::Text(ecmgui_v2::ui::text("probability.snapshot_parameters",zh),r.parameters.b1,effective_b2(r.parameters),r.parameters.curves,r.parameters.factor_bits);
    ImGui::Text(ecmgui_v2::ui::text("probability.stage_summary",zh),ui::percent(r.probability.stage1),ui::percent(r.probability.combined),ui::percent(r.probability.conditional));
    ImGui::Text(ecmgui_v2::ui::text("probability.selected_summary",zh),ui::percent(r.probability.selected),ui::percent(cumulative(r.probability.selected,r.parameters.curves)));
    if(r.request.solve!=Solve::Forward){const bool integral=r.request.solve==Solve::Curves||r.request.solve==Solve::B1||r.request.solve==Solve::B2;
        char value[64];std::snprintf(value,sizeof(value),integral?"%.0f":"%.12g",r.solved_value);
        ImGui::Text(ecmgui_v2::ui::text("probability.solution_summary",zh),value,ui::percent(r.residual),r.lower_bound_sufficient?ecmgui_v2::ui::text("lower_bound_sufficient",zh):"");}
    const std::vector<Point>* series[]{&r.sizes,&r.size_totals,&r.curve_counts};
    const char* labels[]{ecmgui_v2::ui::text("per_curve_probability_size",zh),ecmgui_v2::ui::text("cumulative_probability_size",zh),ecmgui_v2::ui::text("cumulative_probability_curves",zh)};
    const auto colors=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
    for(int i=0;i<3;++i){ImGui::PushID(i);ImGui::SeparatorText(labels[i]);
        std::vector<ui::ChartSeries> lines{{"S1",colors.stage1,{}},{"S1+S2",colors.stage2,{}},{ecmgui_v2::ui::text("probability.stage2_given_stage1_missed",zh),colors.warning,{}}};
        for(const auto& v:*series[i]){lines[0].points.push_back({v.x,ui::percent(v.stage1)});lines[1].points.push_back({v.x,ui::percent(v.combined)});lines[2].points.push_back({v.x,ui::percent(v.conditional)});}
        ui::chart("probability",lines,i==2?ecmgui_v2::ui::text("curves",zh):r.request.decimal_digits?ecmgui_v2::ui::text("factor_digits",zh):ecmgui_v2::ui::text("factor_bits",zh),ecmgui_v2::ui::text("probability.axis_percent",zh),zh,100,180,{true,true,"%.0f","%.6g%%"});
        ImGui::PopID();}
}
void App::draw_gpu(bool zh) {
    gpu_presentations.clear();
    if(!gpu.available()){ImGui::TextWrapped("%s",no_nvml?ecmgui_v2::ui::text("monitoring_disabled",zh):gpu.reason().c_str());return;}
    for(const auto& device:gpu.devices()) {
        ImGui::PushID(device.index);
        auto& display=gpu_presentations[device.index];
        const auto title="#"+std::to_string(device.index)+"  "+device.name;
        const bool expanded=ImGui::CollapsingHeader(title.c_str(),ImGuiTreeNodeFlags_DefaultOpen);
        const float block_top=ImGui::GetItemRectMin().y;
        display.header={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
        const auto details=gpu_details.get(device.index);
        const auto known=[](int n){return n>0?std::to_string(n):std::string("--");};
        const auto capability=details.major>=0?std::to_string(details.major)+"."+std::to_string(details.minor):std::string("--");
        const auto cuda=details.cuda_version>0?std::to_string(details.cuda_version/1000)+"."+std::to_string((details.cuda_version%1000)/10):std::string("--");
        display.tooltip=device.name+"\n"+ecmgui_v2::ui::text("driver",zh)+(details.driver.empty()?"--":details.driver)+" | "+ecmgui_v2::ui::text("cuda_driver",zh)+cuda+
            "\nSM "+capability+" | "+ecmgui_v2::ui::text("sm_count",zh)+known(details.sm_count)+" | "+ecmgui_v2::ui::text("cuda_cores",zh)+known(device.cores)+
            "\n"+ecmgui_v2::ui::text("vram",zh)+known(static_cast<int>(device.mem_total_mb))+" MiB\nNVML / 500 ms";
        display.hovered=ImGui::IsItemHovered();
        if(display.hovered)ImGui::SetTooltip("%s",display.tooltip.c_str());
        if(!expanded){ImGui::PopID();continue;}
        const auto history=gpu.history(device.index);
        if(history.empty()||!history.back().valid){ui::secondary_text(ecmgui_v2::ui::text("unknown_sample",zh));draw_gpu_clock_control(device.index,details.uuid,zh);ImGui::PopID();continue;}
        const auto& sample=history.back();const double occupied=ui::memory_occupancy(sample.mem_used_mb,device.mem_total_mb);
        char basic[192];std::snprintf(basic,sizeof(basic),ecmgui_v2::ui::text("gpu.memory_temperature",zh),sample.mem_used_mb,device.mem_total_mb,sample.temp_c);
        display.basic=basic;ui::secondary_text(basic);
        draw_gpu_clock_control(device.index,details.uuid,zh);
        const auto throttle=ecmgui::GpuMonitor::throttle_text(sample.throttle_reasons);if(!throttle.empty())ui::secondary_text(throttle.c_str());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,4*ImGui::GetFontSize()/15.f});
        const int seconds=state.ui_gpu_history_seconds;const auto color=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
        std::vector<ui::ChartSeries> usage{{ecmgui_v2::ui::text("gpu_load",zh),color.stage1,{}},{ecmgui_v2::ui::text("memory_occupied",zh),color.stage2,{}}};
        std::vector<ui::ChartSeries> power{{ecmgui_v2::ui::text("power",zh),color.warning,{}}};
        std::vector<ui::ChartSeries> clocks{{ecmgui_v2::ui::text("core_frequency",zh),color.stage1,{}},{ecmgui_v2::ui::text("memory_frequency",zh),color.stage2,{}}};
        for(const auto& v:history){const double x=(static_cast<double>(v.ts_ms)-static_cast<double>(sample.ts_ms))/1000.;if(x < -seconds)continue;
            const double missing=std::numeric_limits<double>::quiet_NaN();usage[0].points.push_back({x,v.valid?double(v.util_gpu):missing});usage[1].points.push_back({x,v.valid?ui::memory_occupancy(v.mem_used_mb,device.mem_total_mb):missing});power[0].points.push_back({x,ui::power_value(v.power_w,v.power_limit_w,v.valid)});
            clocks[0].points.push_back({x,v.valid?double(v.clock_sm_mhz):missing});clocks[1].points.push_back({x,v.valid?double(v.clock_mem_mhz):missing});}
        ui::ChartFormat format{true,true,"%.1f s","%.0f%%",-double(seconds),0};
        format.show_y_ticks=false;format.show_axis_titles=false;
        format.legend_controls=false;format.compact=true;format.show_x_ticks=false;format.show_x_grid=false;
        char readout[96];std::snprintf(readout,sizeof(readout),"%d / 100%%",sample.util_gpu);usage[0].readout=readout;
        if(std::isfinite(occupied))std::snprintf(readout,sizeof(readout),"%.1f / 100%%",occupied);else std::snprintf(readout,sizeof(readout),"-- / 100%%");usage[1].readout=readout;
        display.usage=std::string(usage[0].label)+" "+usage[0].readout+" | "+usage[1].label+" "+usage[1].readout;
        format.metrics=&display.chart_metrics[0];
        ui::chart("usage",usage,ecmgui_v2::ui::text("seconds_from_latest_sample",zh),"%",zh,100,64,format);++display.charts;
        format.percent_y=false;format.y_format="%.1f W";
        power[0].readout=ui::power_readout(sample.power_w,sample.power_limit_w,sample.valid);display.power=power[0].readout;format.metrics=&display.chart_metrics[1];
        if(std::isfinite(sample.power_limit_w)&&sample.power_limit_w>0){std::snprintf(readout,sizeof(readout),"%.1f / %.0f W",3*sample.power_limit_w,sample.power_limit_w);power[0].readout_capacity=readout;}
        else power[0].readout_capacity="99999.9 / 99999 W";
        ui::chart("power",power,ecmgui_v2::ui::text("seconds_from_latest_sample",zh),"W",zh,0,64,format);++display.charts;
        format.y_format="%.0f MHz";
        std::snprintf(readout,sizeof(readout),"%u MHz",sample.clock_sm_mhz);clocks[0].readout=readout;
        std::snprintf(readout,sizeof(readout),"%u MHz",sample.clock_mem_mhz);clocks[1].readout=readout;
        clocks[0].readout_capacity=clocks[1].readout_capacity="99999 MHz";
        display.clocks=std::string(clocks[0].label)+" "+clocks[0].readout+" | "+clocks[1].label+" "+clocks[1].readout;
        format.metrics=&display.chart_metrics[2];
        ui::chart("clocks",clocks,ecmgui_v2::ui::text("seconds_from_latest_sample",zh),"MHz",zh,0,64,format);++display.charts;
        ImGui::PopStyleVar();display.height=ImGui::GetItemRectMax().y-block_top;
        ImGui::PopID();
    }
}
bool App::edit_business(const std::string& table,const std::string& key,const std::string& value) {
    if(!ready||controller.busy()){status="Stop workers before editing workflow.";return false;}
    if(table=="gui"||table.rfind("gui.",0)==0){status="Legacy GUI preferences are preserved; use v2 appearance controls.";return false;}
    try {
        auto candidate=config;candidate.set(table,key,value);
        const auto* option=ecm_config::toml_option(table,key);const std::string type=option->type;
        auto base=table;if(ecm_config::toml_worker(table)>0){const auto dot=table.find('.',8);base=dot==std::string::npos?"workers.1":table.substr(dot+1);}
        for(const auto& field:business_options)if(base==field.table&&key==field.key&&(field.has_min||field.has_max)){
            size_t used=0;const double n=std::stod(value,&used);
            if(used!=value.size()||!std::isfinite(n)||(field.has_min&&n<field.minimum)||(field.has_max&&n>field.maximum))throw std::runtime_error("Value outside schema range: "+std::string(field.domain));
        }
        if(type=="bool")ecm_config::boolean(value,key.c_str());
        if(type=="int"){int n=0;ecm_config::assign_value(n,value);}
        if(type=="u32"){uint32_t n=0;ecm_config::assign_value(n,value);}
        if(type=="u64")ecm_config::unsigned_integer(value,key.c_str());
        if(type=="float"||type=="double"){double n=0;ecm_config::assign_value(n,value);}
        for(const auto& line:candidate.lines())if(line.kind==ecm_config::TomlLine::Kind::KeyValue&&line.section==table&&line.key==key){char kind=0;ecm_config::toml_value(line.raw.substr(line.raw.find('=')+1),kind);}
        // Use the same generated assignment contracts as the unchanged GUI/worker.
        ecm_config::read_stage1(ecm_config::cli_entries(candidate.lines(),ecm_config::toml_worker(table)));
        ecm_config::read_stage2(ecm_config::cli_entries(candidate.lines(),ecm_config::toml_worker(table)));
        config=std::move(candidate);gen_valid=false;status="Unsaved workflow edit";return true;
    }catch(const std::exception& error){status=error.what();return false;}
}
bool App::save_business() {
    if(!ready||controller.busy()){status="Stop workers before saving workflow.";return false;}
    ecmgui::FileStamp current;if(!ecmgui::stamp_file(config_path,current)||!same_stamp(current,config_stamp)){status="Workflow changed externally; reload before saving.";return false;}
    std::string error;if(!config.save(error)){status=error;return false;}
    const auto path=config_path;return load_config(path);
}
void App::draw_settings(bool zh) {
    if(settings_category==0){
    ImGui::SeparatorText(ui::text("settings_theme",zh));
    const auto p=ui::palette(state.ui_theme=="light");const float scale=ImGui::GetFontSize()/15.f;
    const float available=ImGui::GetContentRegionAvail().x,gap=ImGui::GetStyle().ItemSpacing.x;
    const bool two=available>=440*scale;const float width=two?(available-gap)*.5f:available;
    if(ui::theme_card("##theme-dark",ui::text("settings_theme_dark",zh),false,state.ui_theme=="dark",p,scale,width))state.ui_theme="dark";
    theme_controls[0]={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(two)ImGui::SameLine();
    if(ui::theme_card("##theme-light",ui::text("settings_theme_light",zh),true,state.ui_theme=="light",p,scale,width))state.ui_theme="light";
    theme_controls[1]={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SeparatorText(ui::text("menu.language",zh));
    const auto language_pos=ImGui::GetCursorScreenPos();ImGui::SetNextItemWidth(-1);
    language_control={language_pos.x+ImGui::GetContentRegionAvail().x*.5f,language_pos.y+ImGui::GetFrameHeight()*.5f};
    const auto selected_file=state.ui_language_file;
    const auto selected_label=ui::language_label(selected_file);
    if(ImGui::BeginCombo("##language-file",selected_label.c_str())){
        if(ImGui::IsWindowAppearing())ui::refresh_language_files();
        const auto files=ui::language_files();
        for(const auto& file:files){
            const auto label=ui::language_label(file)+"###language."+file;
            if(ImGui::Selectable(label.c_str(),file==selected_file))select_language_file(file);
            const ImVec2 center{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};language_file_controls[file]=center;
            if(file=="english.xml")language_items[0]=center;else if(file=="chineseSimplified.xml")language_items[1]=center;
        }
        ImGui::EndCombo();
    }
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",(ui::language_directory()+"/"+selected_file).c_str());
    if(ImGui::Button(ui::text("reload_localization",zh)))reload_language_file();
    ImGui::SeparatorText(ui::text("settings_typography",zh));
    const float scales[]{0,1,1.25f,1.5f,2};int scale_index=0;for(int i=1;i<5;++i)if(std::abs(state.ui_scale-scales[i])<.01f)scale_index=i;
    if(chips("scale",scale_index,{ecmgui_v2::ui::text("auto_dpi",zh),"100%","125%","150%","200%"}))state.ui_scale=scales[scale_index];
    ImGui::TextWrapped("%s",ui::text("font_size",zh));
    ImGui::SetNextItemWidth(std::min(180*ImGui::GetStyle().FontScaleDpi,ImGui::GetContentRegionAvail().x));
    ImGui::SliderInt("##font-size",&state.ui_font_size,ui_limits::ui_font_size_min,ui_limits::ui_font_size_max,"%d px",ImGuiSliderFlags_AlwaysClamp);
    font_size_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    const auto default_size=UiValues{}.ui_font_size;const auto default_label=std::string(ecmgui_v2::ui::text("default",zh))+std::to_string(default_size);
    ImGui::SameLine();if(ImGui::SmallButton(default_label.c_str()))state.ui_font_size=default_size;
    ui::secondary_text(ecmgui_v2::ui::text("settings.font_size_hint",zh),&secondary_text_font_height);
    ImGui::TextWrapped("%s",ui::text("primary_action_font_offset",zh));
    ImGui::SetNextItemWidth(std::min(180*ImGui::GetStyle().FontScaleDpi,ImGui::GetContentRegionAvail().x));
    ImGui::SliderInt("##primary-offset",&state.ui_primary_font_offset,ui_limits::ui_primary_font_offset_min,ui_limits::ui_primary_font_offset_max,"%+d px",ImGuiSliderFlags_AlwaysClamp);
    primary_offset_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::TextWrapped("%s",ui::text("secondary_text_font_offset",zh));
    ImGui::SetNextItemWidth(std::min(180*ImGui::GetStyle().FontScaleDpi,ImGui::GetContentRegionAvail().x));
    ImGui::SliderInt("##secondary-offset",&state.ui_secondary_font_offset,ui_limits::ui_secondary_font_offset_min,ui_limits::ui_secondary_font_offset_max,"%+d px",ImGuiSliderFlags_AlwaysClamp);
    secondary_offset_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SameLine();if(ImGui::SmallButton(ecmgui_v2::ui::text("reset_offsets",zh))){const UiValues defaults;state.ui_primary_font_offset=defaults.ui_primary_font_offset;state.ui_secondary_font_offset=defaults.ui_secondary_font_offset;}

    }
    if(settings_category==1){
    ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("smooth_scrolling",zh),&state.ui_smooth_scroll);
    ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("settings.external_windows",zh),&state.ui_external_windows);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("settings.external_windows_hint",zh));
    ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("show_system_title_bar",zh),&state.window_title_bar);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("settings.title_bar_hint",zh));
    ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("action_toasts_errors_always_shown",zh),&state.ui_toasts);
    {
        ImGui::SeparatorText(ecmgui_v2::ui::text("navigation_and_right_tabs",zh));
        ImGui::TextWrapped("%s",ecmgui_v2::ui::text("settings.navigation_hint",zh));
        for(int i=0;i<8;++i){if(i==5)continue;ImGui::PushID(i);bool right=ui::pinned(state.ui_right_pages,i);
            if(ecmgui_v2::ui::checkbox(ui::page_name(i,zh),&right))move_page(i,right);ImGui::PopID();}
        if(ImGui::Button(ecmgui_v2::ui::text("restore_default_layout",zh))){state.ui_right_pages=0;state.ui_right_visible=true;state.ui_inspector=true;state.ui_logs=true;reset_layout=true;}
        if(ImGui::Button(ecmgui_v2::ui::text("save_layout",zh)))save_preferences();
    }

    }
    if(settings_category==2){
    ImGui::TextUnformatted(ecmgui_v2::ui::text("gpu_chart_time_range",zh));
    const int gpu_seconds[]{15,30,60,120};int gpu_range=0;
    for(int i=0;i<4;++i)if(state.ui_gpu_history_seconds==gpu_seconds[i])gpu_range=i;
    if(chips("gpu-history",gpu_range,{"15 s","30 s","60 s","120 s"},gpu_range_controls))state.ui_gpu_history_seconds=gpu_seconds[gpu_range];
    ImGui::Separator();ImGui::TextUnformatted(ui::text("stage2.timing_source",zh));
    int timing_source=state.ui_stage2_timing_source=="previous"?1:state.ui_stage2_timing_source=="tune"?2:0;
    if(chips("stage2-timing",timing_source,{ui::text("stage2.source.auto",zh),ui::text("stage2.source.previous",zh),ui::text("stage2.source.tune",zh)})){
        const char* modes[]{"auto","previous","tune"};state.ui_stage2_timing_source=modes[timing_source];
    }
    ImGui::TextUnformatted(ui::text("stage2.tune_file",zh));char tune_path[4096];std::snprintf(tune_path,sizeof(tune_path),"%s",state.ui_stage2_tune_file.c_str());
    ImGui::SetNextItemWidth(-1);if(ImGui::InputText("##stage2-tune-path",tune_path,sizeof(tune_path)))state.ui_stage2_tune_file=tune_path;
    ImGui::TextWrapped("%s",ui::text("stage2.timing_hint",zh));

    }
    if(settings_category==3){
    ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("ely_pet",zh),&state.pet_enabled);ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("animate_pet",zh),&state.pet_animate);
    ImGui::TextUnformatted(ecmgui_v2::ui::text("pet.skin_label",zh));
    int pet_skin=0;std::vector<const char*> skin_labels;
    for(size_t i=0;i<pet::skins.size();++i){skin_labels.push_back(pet::skins[i].english);if(state.pet_skin==pet::skins[i].id)pet_skin=static_cast<int>(i);}
    if(chips("pet-skin",pet_skin,skin_labels,pet_skin_controls)){
        state.pet_skin=pet::skins[pet_skin].id;
        toast(std::string(ecmgui_v2::ui::text("pet.skin_selected",zh))+pet::skins[pet_skin].english);
    }
    if(zh)ImGui::TextDisabled("%s",pet::skins[pet_skin].chinese);

    }
    if(settings_category==4){
    ImGui::SeparatorText(ecmgui_v2::ui::text("workflow",zh));ImGui::TextUnformatted("TOML");ImGui::SetNextItemWidth(-1);ImGui::InputText("##workflow-path",config_input,sizeof(config_input));
    if(ImGui::Button(ecmgui_v2::ui::text("browse_workflow",zh))){std::string path=config_input;const char filter[]="TOML\0*.toml\0All\0*.*\0\0";if(ecmgui::browse_for_file(path,ecmgui_v2::ui::text("open_workflow",zh),std::string(filter,sizeof(filter))))std::snprintf(config_input,sizeof(config_input),"%s",path.c_str());}
    ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("open_reload",zh))&&!load_config(config_input))toast(status,true);
    if(ImGui::CollapsingHeader(ecmgui_v2::ui::text("worker_settings",zh),ImGuiTreeNodeFlags_DefaultOpen)){
    ImGui::BeginDisabled(controller.busy()||config.dirty());
    ImGui::TextWrapped("%s",ui::text("stage1_executable",zh));ImGui::SetNextItemWidth(-1);ImGui::InputText("##stage1-path",stage1_input,sizeof(stage1_input));
    ImGui::TextWrapped("%s",ui::text("stage2_executable",zh));ImGui::SetNextItemWidth(-1);ImGui::InputText("##stage2-path",stage2_input,sizeof(stage2_input));
    auto browse_exe=[&](const char* id,char* value,size_t capacity){if(ImGui::Button(id)){std::string path=value;const char filter[]="Executable\0*.exe\0All\0*.*\0\0";if(ecmgui::browse_for_file(path,ecmgui_v2::ui::text("choose_worker_executable",zh),std::string(filter,sizeof(filter))))std::snprintf(value,capacity,"%s",path.c_str());}};
    browse_exe(ecmgui_v2::ui::text("browse_stage1",zh),stage1_input,sizeof(stage1_input));ImGui::SameLine();browse_exe(ecmgui_v2::ui::text("browse_stage2",zh),stage2_input,sizeof(stage2_input));
    ImGui::EndDisabled();
    ImGui::BeginDisabled(controller.busy());ImGui::TextWrapped("%s",ui::text("settings.worker_count",zh));ImGui::SetNextItemWidth(std::min(240*ImGui::GetStyle().FontScaleDpi,ImGui::GetContentRegionAvail().x));ImGui::InputInt("##worker-count",&state.worker_count);state.worker_count=std::clamp(state.worker_count,0,64);
    ImGui::BeginDisabled(config.dirty());
    if(ImGui::Button(ecmgui_v2::ui::text("apply_worker_settings",zh))){state.worker_stage1_exe=stage1_input;state.worker_stage2_exe=stage2_input;if(ready)configure_workers();toast(ecmgui_v2::ui::text("worker_settings_applied",zh));}
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    }

    }
    if(settings_category==5){
    ImGui::BeginDisabled(controller.busy());
    if(ready && ImGui::CollapsingHeader(ecmgui_v2::ui::text("edit_computation_settings",zh),ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("%s",ecmgui_v2::ui::text("settings.computation_hint",zh));
        ImGui::InputText(ecmgui_v2::ui::text("search_settings",zh),settings_search,sizeof(settings_search));
        settings_search_control=ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
        chips("settings-group",settings_group,{ecmgui_v2::ui::text("all",zh),ecmgui_v2::ui::text("shared",zh),"Stage1","Stage2",ecmgui_v2::ui::text("worker",zh)});
        const char* groups[]{"","shared","stage1","stage2","workers.1"};
        const auto visible=[&](const char* table,const char* key,const BusinessOption* option){const auto base=option_table(table);return (!settings_group||base==groups[settings_group])&&matches(std::string(table)+"."+key+" "+(option?(ui::text(option->description_key,zh)):""),settings_search);};
        auto select_field=[&](const char* table,const char* key){std::snprintf(edit_table,sizeof(edit_table),"%s",table);std::snprintf(edit_key,sizeof(edit_key),"%s",key);std::snprintf(edit_value,sizeof(edit_value),"%s",config.get(table,key).c_str());};
        int shown=0;
        if(ImGui::BeginChild("fields",ImVec2(0,180),true))for(const auto& line:config.lines())if(line.kind==ecm_config::TomlLine::Kind::KeyValue&&line.section!="gui"){
            const auto* option=selected_option(line.section.c_str(),line.key.c_str());if(!visible(line.section.c_str(),line.key.c_str(),option))continue;++shown;
            const auto name=line.section+"."+line.key+" = "+line.value;if(ImGui::Selectable(name.c_str(),line.section==edit_table&&line.key==edit_key))select_field(line.section.c_str(),line.key.c_str());
            if(option&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text(option->description_key,zh));}
        settings_matches=shown;if(!shown)ImGui::TextUnformatted(ecmgui_v2::ui::text("no_matching_existing_settings",zh));
        ImGui::EndChild();ImGui::InputText(ecmgui_v2::ui::text("section",zh),edit_table,sizeof(edit_table));ImGui::InputText(ecmgui_v2::ui::text("key",zh),edit_key,sizeof(edit_key));
        if(ImGui::BeginCombo(ecmgui_v2::ui::text("available_settings",zh),edit_key)){
            for(const auto& option:business_options){if(!visible(option.table,option.key,&option))continue;const auto name=std::string(option.table)+"."+option.key;if(ImGui::Selectable(name.c_str()))select_field(option.table,option.key);if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text(option.description_key,zh));}ImGui::EndCombo();
        }
        const auto* option=selected_option(edit_table,edit_key);
        if(option){ImGui::TextWrapped("%s\n%s: %s | %s",ui::text(option->description_key,zh),ecmgui_v2::ui::text("type",zh),option->type,option->domain);}
        const bool toggle=option&&(std::string(option->type)=="bool"||std::string(option->domain)=="[0|1]");
        if(toggle){settings_bool=std::string(edit_value)=="true"||std::string(edit_value)=="1";if(ecmgui_v2::ui::checkbox(ecmgui_v2::ui::text("enabled",zh),&settings_bool))std::snprintf(edit_value,sizeof(edit_value),"%s",std::string(option->type)=="bool"?(settings_bool?"true":"false"):(settings_bool?"1":"0"));}
        else ImGui::InputText(ecmgui_v2::ui::text("value",zh),edit_value,sizeof(edit_value));
        if(ImGui::Button(ecmgui_v2::ui::text("stage_edit",zh))&&!edit_business(edit_table,edit_key,edit_value))toast(status,true);ImGui::SameLine();if(ui::button("save-workflow",ecmgui_v2::ui::text("save_workflow",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!save_business())toast(status,true);
        ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("reload_discard_edits",zh)))load_config(config_path);
        if(config.dirty())ImGui::TextUnformatted(ecmgui_v2::ui::text("unsaved_workflow_changes",zh));
    } ImGui::EndDisabled();
    }
}
}
