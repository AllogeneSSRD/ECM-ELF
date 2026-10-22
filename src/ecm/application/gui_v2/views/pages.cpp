#include "../app.h"
#include "../generated/business_options.h"
#include "../ui/probability_presets.h"
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
    ImGui::TextUnformatted(ecmgui_v2::ui::text("published_results_8d015a9f",zh));
    const char* statuses[]{"All","F","NF"}; int status_index=result_filter.status=="F"?1:result_filter.status=="NF"?2:0;
    if (chips("result-status",status_index,{ecmgui_v2::ui::text("all_statuses_8ee57323",zh),"F","NF"},result_chips)) result_filter.status=status_index?statuses[status_index]:"";
    chips("result-stage",result_filter.stage,{ecmgui_v2::ui::text("all_stages_e7974b8d",zh),"Stage 1","Stage 2"});
    ImGui::InputText(ecmgui_v2::ui::text("search_49c266ba",zh),result_search,sizeof(result_search)); result_filter.search=result_search;
    if (ImGui::BeginCombo(ecmgui_v2::ui::text("source_0e570ca6",zh),result_filter.source<0?ecmgui_v2::ui::text("all_a52ace42",zh):results.files()[static_cast<size_t>(result_filter.source)].path.c_str())) {
        if (ImGui::Selectable(ecmgui_v2::ui::text("all_a52ace42",zh),result_filter.source<0)) result_filter.source=-1;
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
    if(ImGui::Button(ecmgui_v2::ui::text("select_visible_70f58736",zh))) selected_results=visible;
    ImGui::SameLine(); if(ImGui::Button(ecmgui_v2::ui::text("clear_selection_cea4d2e0",zh)))selected_results.clear();
    ImGui::SameLine(); if(ImGui::Button(ecmgui_v2::ui::text("copy_original_json_checksum_55c6ba36",zh))) {
        std::vector<size_t> copy; for(auto i:indices)if(selected_results.count(results.rows()[i].id))copy.push_back(i);
        ImGui::SetClipboardText(results.copy_text(copy).c_str());
    }
    ImGui::Text(ecmgui_v2::ui::text("zu_zu_selected_zu_7c9cde22",zh),indices.size(),results.rows().size(),selected_results.size());
    for(const auto& f:results.files()) if(!f.error.empty()||f.invalid_lines||f.pending_lines)
        ImGui::TextWrapped(ecmgui_v2::ui::text("s_s_invalid_zu_pending_zu_0f1d8942",zh),f.path.c_str(),f.error.c_str(),f.invalid_lines,f.pending_lines);
    if(ImGui::BeginTable("results",11,ImGuiTableFlags_ScrollX|ImGuiTableFlags_ScrollY|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable,ImVec2(0,ImGui::GetContentRegionAvail().y))) {
        result_scroll=ImGui::GetScrollY();result_wheel=ImVec2(ImGui::GetWindowPos().x+80,ImGui::GetWindowPos().y+ImGui::GetWindowSize().y*.6f);
        for(const char* title:{ecmgui_v2::ui::text("select_2a78025d",zh),ecmgui_v2::ui::text("status_920e413c",zh),ecmgui_v2::ui::text("target_978354db",zh),"B1","B2",ecmgui_v2::ui::text("curves_4c2c115c",zh),ecmgui_v2::ui::text("model_5e2c614c",zh),ecmgui_v2::ui::text("time_33b93476",zh),ecmgui_v2::ui::text("factors_d7cf50c1",zh),ecmgui_v2::ui::text("stage_de838855",zh),ecmgui_v2::ui::text("source_0e570ca6",zh)}) ImGui::TableSetupColumn(title,ImGuiTableColumnFlags_WidthFixed,110);
        ImGui::TableSetupScrollFreeze(2,1);ImGui::TableHeadersRow(); ImGuiListClipper clip;clip.Begin(static_cast<int>(indices.size()));
        while(clip.Step())for(int j=clip.DisplayStart;j<clip.DisplayEnd;++j) {
            const auto& row=results.rows()[indices[static_cast<size_t>(j)]]; ImGui::PushID(row.id.c_str());ImGui::TableNextRow();ImGui::TableNextColumn();
            bool chosen=selected_results.count(row.id)!=0;if(ImGui::Checkbox("##select",&chosen)){if(chosen)selected_results.insert(row.id);else selected_results.erase(row.id);}
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
    ImGui::TextUnformatted(ecmgui_v2::ui::text("task_generator_7faad54f",zh));
    ImGui::InputTextMultiline("ECM / ECM2",gen_input,sizeof(gen_input),ImVec2(-1,140));
    ImGui::InputInt(ecmgui_v2::ui::text("target_worker_b23c5ad4",zh),&gen_worker);
    int curve=gen_options.curve_type=="e"?0:gen_options.curve_type=="m0"?1:gen_options.curve_type=="m2"?2:3;
    if(chips("generator-curve",curve,{"Edwards (e)","Montgomery PARAM0","Montgomery PARAM2","Montgomery PARAM3"}))gen_options.curve_type=std::vector<std::string>{"e","m0","m2","m3"}[curve];
    ImGui::Checkbox(ecmgui_v2::ui::text("gpu_recommended_curves_9c4a8ce8",zh),&gen_options.use_recommended);
    if(gen_options.use_recommended)ImGui::InputInt(ecmgui_v2::ui::text("blocks_per_sm_1e2788f4",zh),&gen_options.blocks_per_sm);
    else number(ecmgui_v2::ui::text("fixed_curves_6258ac95",zh),gen_options.curves_fixed);
    ImGui::InputText(ecmgui_v2::ui::text("b1_override_empty_keeps_input_0765337e",zh),gen_b1,sizeof(gen_b1));
    ImGui::InputText(ecmgui_v2::ui::text("b2_override_0_disables_d7b40a2b",zh),gen_b2,sizeof(gen_b2));
    ImGui::InputText(ecmgui_v2::ui::text("save_pattern_dcf76fd4",zh),gen_pattern,sizeof(gen_pattern));
    if(ImGui::CollapsingHeader(ecmgui_v2::ui::text("filters_and_ordering_9cfc39f8",zh))) {
        ImGui::Checkbox(ecmgui_v2::ui::text("deduplicate_f0dd6a72",zh),&gen_options.dedup);ImGui::Checkbox(ecmgui_v2::ui::text("descending_79479a6c",zh),&gen_options.sort_desc);
        ImGui::Checkbox(ecmgui_v2::ui::text("sort_known_factors_3b6981b8",zh),&gen_options.sort_factors);text(ecmgui_v2::ui::text("sort_keys_4c73617e",zh),gen_options.sort_by);
        number(ecmgui_v2::ui::text("skip_curves_8d9ed55f",zh),gen_options.skip_curves);number(ecmgui_v2::ui::text("min_exponent_1_any_8a0bccc6",zh),gen_options.min_n);number(ecmgui_v2::ui::text("max_exponent_2c38acf2",zh),gen_options.max_n);
        number(ecmgui_v2::ui::text("min_input_curves_e78af800",zh),gen_options.min_curves);number(ecmgui_v2::ui::text("max_input_curves_1901b341",zh),gen_options.max_curves);ImGui::Checkbox(ecmgui_v2::ui::text("set_assignment_n_a_5ff15e72",zh),&gen_options.set_has_na);
    }
    if(gen_valid && generator_signature()!=gen_signature)gen_valid=false;
    if(ui::button("generate-preview",ecmgui_v2::ui::text("generate_preview_f82333b8",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!preview_generator())toast(status,true);ImGui::SameLine();
    ImGui::BeginDisabled(!gen_valid||controller.busy());if(ui::button("apply-preview",ecmgui_v2::ui::text("append_to_queue_170d8d59",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!apply_generator())toast(status,true);ImGui::EndDisabled();
    if(!gen_preview.text.empty()) {
        ImGui::Text(ecmgui_v2::ui::text("s_read_d_duplicates_d_ae76c89e",zh),gen_valid?ecmgui_v2::ui::text("current_preview_40f8da45",zh):ecmgui_v2::ui::text("expired_preview_37dfc620",zh),gen_preview.read,gen_preview.duplicates);
        if(ImGui::Button(ecmgui_v2::ui::text("copy_preview_311cbd9b",zh)))ImGui::SetClipboardText(gen_preview.text.c_str());
        for(const auto& e:gen_preview.parse_errors)ImGui::TextWrapped("%s",e.c_str());for(const auto& e:gen_preview.warnings)ImGui::TextWrapped("%s",e.c_str());
        ImGui::BeginChild("preview",ImVec2(0,180),true,ImGuiWindowFlags_HorizontalScrollbar);ImGui::TextUnformatted(gen_preview.text.c_str());ImGui::EndChild();
    }
}
void App::draw_probability(bool zh) {
    using namespace ecmgui::probability_ui; auto& request=probability_request;auto& p=request.parameters;
    ImGui::TextUnformatted(ecmgui_v2::ui::text("ecm_probability_43508814",zh));
    chips("probability-view",probability_tab,{ecmgui_v2::ui::text("parameters_e68b36b1",zh),ecmgui_v2::ui::text("charts_and_snapshot_94e2f72c",zh)});
    if(ui::button("calculate-snapshot",ecmgui_v2::ui::text("calculate_snapshot_a22049db",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)){probability_result=calculate(request);probability_ready=true;probability_tab=1;toast(probability_result.error.empty()?ecmgui_v2::ui::text("probability_snapshot_ready_64bbce9f",zh):probability_result.error,!probability_result.error.empty());}
    calculate_control=ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    if(probability_ready&&probability_result.error.empty()){ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("copy_snapshot_csv_bea99b05",zh))){ImGui::SetClipboardText(curve_csv(probability_result).c_str());toast(ecmgui_v2::ui::text("snapshot_csv_copied_a15d488a",zh));}}
    if(probability_tab==0){
    double size=size_from_bits(p.factor_bits,request.decimal_digits);
    if(ImGui::InputDouble(ecmgui_v2::ui::text("factor_size_f27e87f9",zh),&size,0,0,"%.0f"))p.factor_bits=bits_from_size(size,request.decimal_digits);
    ImGui::Checkbox(ecmgui_v2::ui::text("decimal_digits_b45d60b0",zh),&request.decimal_digits);
    ImGui::InputDouble("B1",&p.b1,0,0,"%.0f");ImGui::Checkbox(ecmgui_v2::ui::text("stage_2_enabled_8976ac2a",zh),&p.stage2);
    ImGui::Checkbox(ecmgui_v2::ui::text("use_b2_b1_ratio_5f0b38ea",zh),&p.ratio_mode);if(p.ratio_mode)ImGui::InputDouble("B2/B1",&p.b2_ratio,0,0,"%.6g");else ImGui::InputDouble("B2",&p.b2,0,0,"%.0f");
    ImGui::InputDouble(ecmgui_v2::ui::text("curves_4c2c115c",zh),&p.curves,0,0,"%.0f");
    const bool profile_changed=chips("curve-profile",probability_profile,{"Suyama / PARAM2","Edwards Z2xZ8","Edwards Z12","PARAM1","PARAM3",ecmgui_v2::ui::text("custom_494ca78f",zh)});
    const bool basis_changed=ImGui::Checkbox(ecmgui_v2::ui::text("use_stage1_empirical_calibration_b45c1f82",zh),&probability_empirical);
    if(probability_profile<5&&(profile_changed||basis_changed))p.effective_divisor=ui::curve_divisor(probability_profile,probability_empirical);
    if(ImGui::InputDouble("D_eff",&p.effective_divisor,0,0,"%.6g"))probability_profile=5;
    ImGui::TextWrapped("%s",ecmgui_v2::ui::text("torsion_order_is_not_d_eff_edwards_z2xz8_m_917a40eb",zh));
    int scope=static_cast<int>(p.scope);if(chips("probability-scope",scope,{"Stage 1",ecmgui_v2::ui::text("combined_8036ed37",zh),ecmgui_v2::ui::text("stage_2_conditional_18266507",zh)}))p.scope=static_cast<Scope>(scope);
    int solve=static_cast<int>(request.solve);if(chips("solve",solve,{ecmgui_v2::ui::text("forward_f1c65e14",zh),ecmgui_v2::ui::text("curves_4c2c115c",zh),ecmgui_v2::ui::text("factor_size_f27e87f9",zh),"B1","B2","D_eff"}))request.solve=static_cast<Solve>(solve);
    double target=ui::percent(request.target);if(ImGui::InputDouble(ecmgui_v2::ui::text("target_probability_ff15b758",zh),&target,0,0,"%.6g%%"))request.target=ui::probability(target);
    ImGui::Checkbox(ecmgui_v2::ui::text("cumulative_target_b5b1c689",zh),&request.cumulative_target);
    if(ImGui::CollapsingHeader(ecmgui_v2::ui::text("plot_range_6e66c843",zh))){ImGui::InputDouble(ecmgui_v2::ui::text("size_min_0426839f",zh),&request.size_min,0,0,"%.0f");ImGui::InputDouble(ecmgui_v2::ui::text("size_max_a05f6c92",zh),&request.size_max,0,0,"%.0f");ImGui::Checkbox(ecmgui_v2::ui::text("auto_curves_range_f85bb5f0",zh),&request.auto_curves_range);ImGui::InputDouble(ecmgui_v2::ui::text("curves_max_eea3d4c0",zh),&request.curves_max,0,0,"%.0f");}
        return;
    }
    if(!probability_ready){ImGui::TextWrapped("%s",ecmgui_v2::ui::text("set_parameters_then_calculate_a_snapshot_t_6f098e9f",zh));return;}
    const auto& r=probability_result;if(!r.error.empty()){ImGui::TextWrapped("%s",r.error.c_str());return;}
    const auto values=[](const Request& q){const auto& v=q.parameters;return std::make_tuple(v.factor_bits,v.b1,v.b2,v.b2_ratio,v.curves,v.effective_divisor,v.stage2,v.ratio_mode,v.scope,q.solve,q.target,q.cumulative_target,q.decimal_digits,q.auto_curves_range,q.size_min,q.size_max,q.curves_max);};
    if(values(request)!=values(r.request))ImGui::TextWrapped("%s",ecmgui_v2::ui::text("parameters_changed_the_charts_and_csv_stil_d32359e5",zh));
    ImGui::Text(ecmgui_v2::ui::text("snapshot_b1_0f_b2_0f_curves_0f_bits_6g_b2ea4d63",zh),r.parameters.b1,effective_b2(r.parameters),r.parameters.curves,r.parameters.factor_bits);
    ImGui::Text(ecmgui_v2::ui::text("s1_6g_s1_s2_6g_s2_conditional_6g_88ed90d8",zh),ui::percent(r.probability.stage1),ui::percent(r.probability.combined),ui::percent(r.probability.conditional));
    ImGui::Text(ecmgui_v2::ui::text("selected_per_curve_6g_cumulative_6g_8caa0715",zh),ui::percent(r.probability.selected),ui::percent(cumulative(r.probability.selected,r.parameters.curves)));
    if(r.request.solve!=Solve::Forward){const bool integral=r.request.solve==Solve::Curves||r.request.solve==Solve::B1||r.request.solve==Solve::B2;
        char value[64];std::snprintf(value,sizeof(value),integral?"%.0f":"%.12g",r.solved_value);
        ImGui::Text(ecmgui_v2::ui::text("solution_s_residual_6g_pp_s_80b3192b",zh),value,ui::percent(r.residual),r.lower_bound_sufficient?ecmgui_v2::ui::text("lower_bound_sufficient_cf517509",zh):"");}
    const std::vector<Point>* series[]{&r.sizes,&r.size_totals,&r.curve_counts};
    const char* labels[]{ecmgui_v2::ui::text("per_curve_probability_size_71a03e06",zh),ecmgui_v2::ui::text("cumulative_probability_size_0bcf67ec",zh),ecmgui_v2::ui::text("cumulative_probability_curves_3769b2e1",zh)};
    const auto colors=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
    for(int i=0;i<3;++i){ImGui::PushID(i);ImGui::SeparatorText(labels[i]);
        std::vector<ui::ChartSeries> lines{{"S1",colors.stage1,{}},{"S1+S2",colors.stage2,{}},{ecmgui_v2::ui::text("s2_s1_missed_7420a12d",zh),colors.warning,{}}};
        for(const auto& v:*series[i]){lines[0].points.push_back({v.x,ui::percent(v.stage1)});lines[1].points.push_back({v.x,ui::percent(v.combined)});lines[2].points.push_back({v.x,ui::percent(v.conditional)});}
        ui::chart("probability",lines,i==2?ecmgui_v2::ui::text("curves_4c2c115c",zh):r.request.decimal_digits?ecmgui_v2::ui::text("factor_digits_496da08e",zh):ecmgui_v2::ui::text("factor_bits_15c9189c",zh),ecmgui_v2::ui::text("probability_de7e6ede",zh),zh,100,180,{true,true,"%.0f","%.6g%%"});
        ImGui::PopID();}
}
void App::draw_gpu(bool zh) {
    gpu_presentations.clear();
    if(!gpu.available()){ImGui::TextWrapped("%s",no_nvml?ecmgui_v2::ui::text("monitoring_disabled_9432cd8b",zh):gpu.reason().c_str());return;}
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
        display.tooltip=device.name+"\n"+ecmgui_v2::ui::text("driver_0c1299f4",zh)+(details.driver.empty()?"--":details.driver)+" | "+ecmgui_v2::ui::text("cuda_driver_b9ddcb1e",zh)+cuda+
            "\nSM "+capability+" | "+ecmgui_v2::ui::text("sm_count_d07f07a8",zh)+known(details.sm_count)+" | "+ecmgui_v2::ui::text("cuda_cores_bc6ad6f3",zh)+known(device.cores)+
            "\n"+ecmgui_v2::ui::text("vram_d04c2bb5",zh)+known(static_cast<int>(device.mem_total_mb))+" MiB\nNVML / 500 ms";
        display.hovered=ImGui::IsItemHovered();
        if(display.hovered)ImGui::SetTooltip("%s",display.tooltip.c_str());
        if(!expanded){ImGui::PopID();continue;}
        const auto history=gpu.history(device.index);
        if(history.empty()||!history.back().valid){ImGui::TextUnformatted(ecmgui_v2::ui::text("unknown_sample_e6665d60",zh));ImGui::PopID();continue;}
        const auto& sample=history.back();const double occupied=ui::memory_occupancy(sample.mem_used_mb,device.mem_total_mb);
        char basic[192];std::snprintf(basic,sizeof(basic),ecmgui_v2::ui::text("vram_u_u_mib_d_c_d3c46e35",zh),sample.mem_used_mb,device.mem_total_mb,sample.temp_c);
        display.basic=basic;ui::secondary_text(basic);
        const auto throttle=ecmgui::GpuMonitor::throttle_text(sample.throttle_reasons);if(!throttle.empty())ui::secondary_text(throttle.c_str());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{ImGui::GetStyle().ItemSpacing.x,4*ImGui::GetFontSize()/15.f});
        const int seconds=state.ui_gpu_history_seconds;const auto color=ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
        std::vector<ui::ChartSeries> usage{{ecmgui_v2::ui::text("gpu_load_30c049fb",zh),color.stage1,{}},{ecmgui_v2::ui::text("memory_occupied_03c116ce",zh),color.stage2,{}}};
        std::vector<ui::ChartSeries> power{{ecmgui_v2::ui::text("power_848e9656",zh),color.warning,{}}};
        std::vector<ui::ChartSeries> clocks{{ecmgui_v2::ui::text("core_frequency_63975ac8",zh),color.stage1,{}},{ecmgui_v2::ui::text("memory_frequency_b5886939",zh),color.stage2,{}}};
        for(const auto& v:history){const double x=(static_cast<double>(v.ts_ms)-static_cast<double>(sample.ts_ms))/1000.;if(x < -seconds)continue;
            const double missing=std::numeric_limits<double>::quiet_NaN();usage[0].points.push_back({x,v.valid?double(v.util_gpu):missing});usage[1].points.push_back({x,v.valid?ui::memory_occupancy(v.mem_used_mb,device.mem_total_mb):missing});power[0].points.push_back({x,v.valid?v.power_w:missing});
            clocks[0].points.push_back({x,v.valid?double(v.clock_sm_mhz):missing});clocks[1].points.push_back({x,v.valid?double(v.clock_mem_mhz):missing});}
        ui::ChartFormat format{true,true,"%.1f s","%.0f%%",-double(seconds),0};
        format.show_y_ticks=false;format.show_axis_titles=false;
        format.legend_controls=false;format.compact=true;format.show_x_ticks=false;format.show_x_grid=false;
        char readout[96];std::snprintf(readout,sizeof(readout),"%d / 100%%",sample.util_gpu);usage[0].readout=readout;
        if(std::isfinite(occupied))std::snprintf(readout,sizeof(readout),"%.1f / 100%%",occupied);else std::snprintf(readout,sizeof(readout),"-- / 100%%");usage[1].readout=readout;
        display.usage=std::string(usage[0].label)+" "+usage[0].readout+" | "+usage[1].label+" "+usage[1].readout;
        format.metrics=&display.chart_metrics[0];
        ui::chart("usage",usage,ecmgui_v2::ui::text("seconds_from_latest_sample_7e115cd0",zh),"%",zh,100,64,format);++display.charts;
        format.percent_y=false;format.y_format="%.1f W";
        if(sample.power_limit_w>0)std::snprintf(readout,sizeof(readout),"%.1f / %.0f W",sample.power_w,sample.power_limit_w);else std::snprintf(readout,sizeof(readout),"%.1f / -- W",sample.power_w);
        power[0].readout=readout;display.power=readout;format.metrics=&display.chart_metrics[1];
        ui::chart("power",power,ecmgui_v2::ui::text("seconds_from_latest_sample_7e115cd0",zh),"W",zh,0,64,format);++display.charts;
        format.y_format="%.0f MHz";
        std::snprintf(readout,sizeof(readout),"%u MHz",sample.clock_sm_mhz);clocks[0].readout=readout;
        std::snprintf(readout,sizeof(readout),"%u MHz",sample.clock_mem_mhz);clocks[1].readout=readout;
        display.clocks=std::string(clocks[0].label)+" "+clocks[0].readout+" | "+clocks[1].label+" "+clocks[1].readout;
        format.metrics=&display.chart_metrics[2];
        ui::chart("clocks",clocks,ecmgui_v2::ui::text("seconds_from_latest_sample_7e115cd0",zh),"MHz",zh,0,64,format);++display.charts;
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
    ImGui::SeparatorText(ecmgui_v2::ui::text("appearance_and_interaction_16ace6fc",zh));
    int theme=state.ui_theme=="light";if(chips("theme",theme,{ecmgui_v2::ui::text("dark_60acc53f",zh),ecmgui_v2::ui::text("light_dbcd5e7b",zh)}))state.ui_theme=theme?"light":"dark";
    int language=state.ui_language=="zh";if(chips("language",language,{ui::text("language_english",zh),ui::text("language_chinese",zh)},language_items))state.ui_language=language?"zh":"en";
    const float scales[]{0,1,1.25f,1.5f,2};int scale=0;for(int i=1;i<5;++i)if(std::abs(state.ui_scale-scales[i])<.01f)scale=i;
    if(chips("scale",scale,{ecmgui_v2::ui::text("auto_dpi_38f46a94",zh),"100%","125%","150%","200%"}))state.ui_scale=scales[scale];
    ImGui::SetNextItemWidth(180*ImGui::GetStyle().FontScaleDpi);
    ImGui::SliderInt(ecmgui_v2::ui::text("font_size_6d784d95",zh),&state.ui_font_size,ui_limits::ui_font_size_min,ui_limits::ui_font_size_max,"%d px",ImGuiSliderFlags_AlwaysClamp);
    font_size_control={ImGui::GetItemRectMin().x+90*ImGui::GetStyle().FontScaleDpi,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    const auto default_size=UiValues{}.ui_font_size;const auto default_label=std::string(ecmgui_v2::ui::text("default_f12bc6ea",zh))+std::to_string(default_size);
    ImGui::SameLine();if(ImGui::SmallButton(default_label.c_str()))state.ui_font_size=default_size;
    ui::secondary_text(ecmgui_v2::ui::text("font_size_uses_logical_pixels_monitor_dpi__fad8e1e4",zh),&secondary_text_font_height);
    ImGui::SetNextItemWidth(180*ImGui::GetStyle().FontScaleDpi);
    ImGui::SliderInt(ecmgui_v2::ui::text("primary_action_font_offset_034fcf3c",zh),&state.ui_primary_font_offset,ui_limits::ui_primary_font_offset_min,ui_limits::ui_primary_font_offset_max,"%+d px",ImGuiSliderFlags_AlwaysClamp);
    primary_offset_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SetNextItemWidth(180*ImGui::GetStyle().FontScaleDpi);
    ImGui::SliderInt(ecmgui_v2::ui::text("secondary_text_font_offset_78d4f12a",zh),&state.ui_secondary_font_offset,ui_limits::ui_secondary_font_offset_min,ui_limits::ui_secondary_font_offset_max,"%+d px",ImGuiSliderFlags_AlwaysClamp);
    secondary_offset_control={(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ImGui::SameLine();if(ImGui::SmallButton(ecmgui_v2::ui::text("reset_offsets_1e6247e3",zh))){const UiValues defaults;state.ui_primary_font_offset=defaults.ui_primary_font_offset;state.ui_secondary_font_offset=defaults.ui_secondary_font_offset;}
    ImGui::TextUnformatted(ecmgui_v2::ui::text("gpu_chart_time_range_19468e7b",zh));
    const int gpu_seconds[]{15,30,60,120};int gpu_range=0;
    for(int i=0;i<4;++i)if(state.ui_gpu_history_seconds==gpu_seconds[i])gpu_range=i;
    if(chips("gpu-history",gpu_range,{"15 s","30 s","60 s","120 s"},gpu_range_controls))state.ui_gpu_history_seconds=gpu_seconds[gpu_range];
    ImGui::Checkbox(ecmgui_v2::ui::text("smooth_scrolling_f62e8490",zh),&state.ui_smooth_scroll);
    ImGui::Checkbox(ecmgui_v2::ui::text("allow_panels_outside_the_main_window_resta_972e3b76",zh),&state.ui_external_windows);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("by_default_panels_dock_and_float_inside_th_469a5486",zh));
    ImGui::Checkbox(ecmgui_v2::ui::text("show_system_title_bar_6e15fd01",zh),&state.window_title_bar);
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ecmgui_v2::ui::text("when_hidden_drag_the_empty_top_bar_window__83f3723f",zh));
    ImGui::Checkbox(ecmgui_v2::ui::text("action_toasts_errors_always_shown_2ef6996e",zh),&state.ui_toasts);
    ImGui::Checkbox(ecmgui_v2::ui::text("ely_pet_1100b952",zh),&state.pet_enabled);ImGui::Checkbox(ecmgui_v2::ui::text("animate_pet_d53550fa",zh),&state.pet_animate);
    ImGui::TextUnformatted(ecmgui_v2::ui::text("ely_skin_67bfe14c",zh));
    int pet_skin=0;std::vector<const char*> skin_labels;
    for(size_t i=0;i<pet::skins.size();++i){skin_labels.push_back(pet::skins[i].english);if(state.pet_skin==pet::skins[i].id)pet_skin=static_cast<int>(i);}
    if(chips("pet-skin",pet_skin,skin_labels,pet_skin_controls)){
        state.pet_skin=pet::skins[pet_skin].id;
        toast(std::string(ecmgui_v2::ui::text("ely_skin_cf778253",zh))+pet::skins[pet_skin].english);
    }
    if(zh)ImGui::TextDisabled("%s",pet::skins[pet_skin].chinese);
    if(ui::button("save-ui",ecmgui_v2::ui::text("save_ui_preferences_b7e00f2f",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi))save_preferences();
    save_ui_control=ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
    ui::secondary_text(ecmgui_v2::ui::text("appearance_and_layout_are_also_saved_on_a__9b4c60a6",zh));
    if(!state.writable)ImGui::TextWrapped("%s",state.notice.c_str());
    ui::secondary_text(state.path.u8string().c_str());
    {
        ImGui::SeparatorText(ecmgui_v2::ui::text("navigation_and_right_tabs_c90aebf6",zh));
        ImGui::TextWrapped("%s",ecmgui_v2::ui::text("hold_a_navigation_page_and_drag_to_reorder_244eb05e",zh));
        for(int i=0;i<7;++i){ImGui::PushID(i);bool right=ui::pinned(state.ui_right_pages,i);
            if(ImGui::Checkbox(ui::page_name(i,zh),&right))move_page(i,right);ImGui::PopID();}
        if(ImGui::Button(ecmgui_v2::ui::text("restore_default_layout_c8df250c",zh))){state.ui_right_pages=0;state.ui_right_visible=true;state.ui_inspector=true;state.ui_logs=true;reset_layout=true;}
        if(ImGui::Button(ecmgui_v2::ui::text("save_layout_26ac39b1",zh)))save_preferences();
    }
    ImGui::SeparatorText(ecmgui_v2::ui::text("workflow_2e2d5c56",zh));ImGui::InputText("TOML",config_input,sizeof(config_input));
    if(ImGui::Button(ecmgui_v2::ui::text("browse_workflow_f6379582",zh))){std::string path=config_input;const char filter[]="TOML\0*.toml\0All\0*.*\0\0";if(ecmgui::browse_for_file(path,ecmgui_v2::ui::text("open_workflow_8d5f0289",zh),std::string(filter,sizeof(filter))))std::snprintf(config_input,sizeof(config_input),"%s",path.c_str());}
    ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("open_reload_002e2127",zh))&&!load_config(config_input))toast(status,true);
    if(ImGui::CollapsingHeader(ecmgui_v2::ui::text("worker_settings_bd2e3420",zh))){
    ImGui::BeginDisabled(controller.busy()||config.dirty());
    ImGui::InputText(ecmgui_v2::ui::text("stage1_executable_50902b05",zh),stage1_input,sizeof(stage1_input));
    ImGui::InputText(ecmgui_v2::ui::text("stage2_executable_3b6f5722",zh),stage2_input,sizeof(stage2_input));
    auto browse_exe=[&](const char* id,char* value,size_t capacity){if(ImGui::Button(id)){std::string path=value;const char filter[]="Executable\0*.exe\0All\0*.*\0\0";if(ecmgui::browse_for_file(path,ecmgui_v2::ui::text("choose_worker_executable_8157cf69",zh),std::string(filter,sizeof(filter))))std::snprintf(value,capacity,"%s",path.c_str());}};
    browse_exe(ecmgui_v2::ui::text("browse_stage1_f27f693d",zh),stage1_input,sizeof(stage1_input));ImGui::SameLine();browse_exe(ecmgui_v2::ui::text("browse_stage2_b26e15b0",zh),stage2_input,sizeof(stage2_input));
    ImGui::EndDisabled();
    ImGui::BeginDisabled(controller.busy());ImGui::InputInt(ecmgui_v2::ui::text("worker_count_0_follows_workflow_291604da",zh),&state.worker_count);state.worker_count=std::clamp(state.worker_count,0,64);
    ImGui::BeginDisabled(config.dirty());
    if(ImGui::Button(ecmgui_v2::ui::text("apply_worker_settings_5e29a826",zh))){state.worker_stage1_exe=stage1_input;state.worker_stage2_exe=stage2_input;if(ready)configure_workers();toast(ecmgui_v2::ui::text("worker_settings_applied_849be3a6",zh));}
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    }
    ImGui::BeginDisabled(controller.busy());
    if(ready && ImGui::CollapsingHeader(ecmgui_v2::ui::text("edit_computation_settings_58236f98",zh),ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("%s",ecmgui_v2::ui::text("search_descriptions_or_keys_choose_a_field_92349d21",zh));
        ImGui::InputText(ecmgui_v2::ui::text("search_settings_255b076d",zh),settings_search,sizeof(settings_search));
        settings_search_control=ImVec2{(ImGui::GetItemRectMin().x+ImGui::GetItemRectMax().x)*.5f,(ImGui::GetItemRectMin().y+ImGui::GetItemRectMax().y)*.5f};
        chips("settings-group",settings_group,{ecmgui_v2::ui::text("all_a52ace42",zh),ecmgui_v2::ui::text("shared_e3c4b39d",zh),"Stage1","Stage2",ecmgui_v2::ui::text("worker_a67b04cd",zh)});
        const char* groups[]{"","shared","stage1","stage2","workers.1"};
        const auto visible=[&](const char* table,const char* key,const BusinessOption* option){const auto base=option_table(table);return (!settings_group||base==groups[settings_group])&&matches(std::string(table)+"."+key+" "+(option?(ui::text(option->description_key,zh)):""),settings_search);};
        auto select_field=[&](const char* table,const char* key){std::snprintf(edit_table,sizeof(edit_table),"%s",table);std::snprintf(edit_key,sizeof(edit_key),"%s",key);std::snprintf(edit_value,sizeof(edit_value),"%s",config.get(table,key).c_str());};
        int shown=0;
        if(ImGui::BeginChild("fields",ImVec2(0,180),true))for(const auto& line:config.lines())if(line.kind==ecm_config::TomlLine::Kind::KeyValue&&line.section!="gui"){
            const auto* option=selected_option(line.section.c_str(),line.key.c_str());if(!visible(line.section.c_str(),line.key.c_str(),option))continue;++shown;
            const auto name=line.section+"."+line.key+" = "+line.value;if(ImGui::Selectable(name.c_str(),line.section==edit_table&&line.key==edit_key))select_field(line.section.c_str(),line.key.c_str());
            if(option&&ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text(option->description_key,zh));}
        settings_matches=shown;if(!shown)ImGui::TextUnformatted(ecmgui_v2::ui::text("no_matching_existing_settings_5461a9f5",zh));
        ImGui::EndChild();ImGui::InputText(ecmgui_v2::ui::text("section_dfca5da5",zh),edit_table,sizeof(edit_table));ImGui::InputText(ecmgui_v2::ui::text("key_99a52df3",zh),edit_key,sizeof(edit_key));
        if(ImGui::BeginCombo(ecmgui_v2::ui::text("available_settings_018544ed",zh),edit_key)){
            for(const auto& option:business_options){if(!visible(option.table,option.key,&option))continue;const auto name=std::string(option.table)+"."+option.key;if(ImGui::Selectable(name.c_str()))select_field(option.table,option.key);if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",ui::text(option.description_key,zh));}ImGui::EndCombo();
        }
        const auto* option=selected_option(edit_table,edit_key);
        if(option){ImGui::TextWrapped("%s\n%s: %s | %s",ui::text(option->description_key,zh),ecmgui_v2::ui::text("type_baaddf70",zh),option->type,option->domain);}
        const bool toggle=option&&(std::string(option->type)=="bool"||std::string(option->domain)=="[0|1]");
        if(toggle){settings_bool=std::string(edit_value)=="true"||std::string(edit_value)=="1";if(ImGui::Checkbox(ecmgui_v2::ui::text("enabled_92c1cdfd",zh),&settings_bool))std::snprintf(edit_value,sizeof(edit_value),"%s",std::string(option->type)=="bool"?(settings_bool?"true":"false"):(settings_bool?"1":"0"));}
        else ImGui::InputText(ecmgui_v2::ui::text("value_8e37953d",zh),edit_value,sizeof(edit_value));
        if(ImGui::Button(ecmgui_v2::ui::text("stage_edit_12101dc2",zh))&&!edit_business(edit_table,edit_key,edit_value))toast(status,true);ImGui::SameLine();if(ui::button("save-workflow",ecmgui_v2::ui::text("save_workflow_e77921c8",zh),ui::ButtonKind::Primary,ui::palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f),ImGui::GetStyle().FontScaleDpi)&&!save_business())toast(status,true);
        ImGui::SameLine();if(ImGui::Button(ecmgui_v2::ui::text("reload_discard_edits_bd211ada",zh)))load_config(config_path);
        if(config.dirty())ImGui::TextUnformatted(ecmgui_v2::ui::text("unsaved_workflow_changes_01a66bda",zh));
    } ImGui::EndDisabled();
}
}
