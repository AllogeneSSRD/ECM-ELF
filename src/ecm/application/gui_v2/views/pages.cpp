#include "../app.h"
#include "../generated/business_options.h"
#include "../ui/probability_presets.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

namespace ecmgui_v2 {
namespace {
const char* tr(bool zh,const char* en,const char* cn) { return zh?cn:en; }
bool text(const char* label,std::string& value,size_t limit=1024) {
    std::vector<char> buffer(std::max(limit,value.size()+1)); std::copy(value.begin(),value.end(),buffer.begin());
    if (!ImGui::InputText(label,buffer.data(),buffer.size())) return false;
    value=buffer.data(); return true;
}
bool same_stamp(const ecmgui::FileStamp& a,const ecmgui::FileStamp& b) {
    return a.exists==b.exists && a.size==b.size && a.mtime==b.mtime && a.hash==b.hash;
}
void number(const char* label,long long& value) { ImGui::InputScalar(label,ImGuiDataType_S64,&value); }
bool chips(const char* id,int& value,const std::vector<const char*>& labels){
    bool changed=false;ImGui::PushID(id);int index=0;
    for(const auto label:labels){
        const auto width=ImGui::CalcTextSize(label).x+ImGui::GetStyle().FramePadding.x*2;
        if(index&&ImGui::GetItemRectMax().x+width+ImGui::GetStyle().ItemSpacing.x<ImGui::GetWindowPos().x+ImGui::GetWindowWidth()-ImGui::GetStyle().WindowPadding.x)ImGui::SameLine();
        const bool selected=value==index;
        if(selected)ImGui::PushStyleColor(ImGuiCol_Button,ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
        if(ImGui::Button(label)){value=index;changed=true;}
        if(selected)ImGui::PopStyleColor();
        ++index;
    }ImGui::PopID();return changed;
}
}
void App::draw_results(bool zh) {
    if(result_filter.source>=static_cast<int>(results.files().size()))result_filter.source=-1;
    ImGui::TextUnformatted(tr(zh,"Published results","已发布结果"));
    const char* statuses[]{"All","F","NF"}; int status_index=result_filter.status=="F"?1:result_filter.status=="NF"?2:0;
    if (chips("result-status",status_index,{tr(zh,"All statuses","全部状态"),"F","NF"})) result_filter.status=status_index?statuses[status_index]:"";
    chips("result-stage",result_filter.stage,{tr(zh,"All stages","全部阶段"),"Stage 1","Stage 2"});
    ImGui::InputText(tr(zh,"Search","搜索"),result_search,sizeof(result_search)); result_filter.search=result_search;
    if (ImGui::BeginCombo(tr(zh,"Source","来源"),result_filter.source<0?"All":results.files()[static_cast<size_t>(result_filter.source)].path.c_str())) {
        if (ImGui::Selectable("All",result_filter.source<0)) result_filter.source=-1;
        for(size_t i=0;i<results.files().size();++i) if(ImGui::Selectable(results.files()[i].path.c_str(),result_filter.source==static_cast<int>(i))) result_filter.source=static_cast<int>(i);
        ImGui::EndCombo();
    }
    const auto indices=results.filtered(result_filter);
    std::set<std::string> visible; for(auto i:indices)visible.insert(results.rows()[i].id);
    for(auto it=selected_results.begin();it!=selected_results.end();) if(!visible.count(*it))it=selected_results.erase(it);else ++it;
    if(ImGui::Button(tr(zh,"Select visible","选择可见行"))) selected_results=visible;
    ImGui::SameLine(); if(ImGui::Button(tr(zh,"Clear selection","清除选择")))selected_results.clear();
    ImGui::SameLine(); if(ImGui::Button(tr(zh,"Copy original JSON / checksum","复制原始 JSON / checksum"))) {
        std::vector<size_t> copy; for(auto i:indices)if(selected_results.count(results.rows()[i].id))copy.push_back(i);
        ImGui::SetClipboardText(results.copy_text(copy).c_str());
    }
    ImGui::Text("%zu / %zu | selected %zu",indices.size(),results.rows().size(),selected_results.size());
    for(const auto& f:results.files()) if(!f.error.empty()||f.invalid_lines||f.pending_lines)
        ImGui::TextWrapped("%s: %s | invalid %zu | pending %zu",f.path.c_str(),f.error.c_str(),f.invalid_lines,f.pending_lines);
    if(ImGui::BeginTable("results",11,ImGuiTableFlags_ScrollX|ImGuiTableFlags_ScrollY|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable,ImVec2(0,ImGui::GetContentRegionAvail().y))) {
        for(const char* title:{"Select","Status","Target","B1","B2","Curves","Model","Time","Factors","Stage","Source"}) ImGui::TableSetupColumn(title,ImGuiTableColumnFlags_WidthFixed,110);
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
    ImGui::TextUnformatted(tr(zh,"Task generator","任务生成器"));
    ImGui::InputTextMultiline("ECM / ECM2",gen_input,sizeof(gen_input),ImVec2(-1,140));
    ImGui::InputInt(tr(zh,"Target worker","目标 worker"),&gen_worker);
    int curve=gen_options.curve_type=="e"?0:gen_options.curve_type=="m0"?1:gen_options.curve_type=="m2"?2:3;
    if(ImGui::Combo(tr(zh,"Curve / parametrization","曲线 / 参数化"),&curve,"Edwards (e)\0Montgomery PARAM0\0Montgomery PARAM2\0Montgomery PARAM3\0"))gen_options.curve_type=std::vector<std::string>{"e","m0","m2","m3"}[curve];
    ImGui::Checkbox(tr(zh,"GPU recommended curves","GPU 推荐曲线数"),&gen_options.use_recommended);
    if(gen_options.use_recommended)ImGui::InputInt(tr(zh,"Blocks per SM","每 SM 块数"),&gen_options.blocks_per_sm);
    else number(tr(zh,"Fixed curves","固定曲线数"),gen_options.curves_fixed);
    ImGui::InputText(tr(zh,"B1 override (empty keeps input)","B1 覆盖（留空保留输入）"),gen_b1,sizeof(gen_b1));
    ImGui::InputText(tr(zh,"B2 override (0 disables)","B2 覆盖（0 关闭）"),gen_b2,sizeof(gen_b2));
    ImGui::InputText(tr(zh,"Save pattern","保存文件名"),gen_pattern,sizeof(gen_pattern));
    if(ImGui::CollapsingHeader(tr(zh,"Filters and ordering","筛选与排序"))) {
        ImGui::Checkbox(tr(zh,"Deduplicate","去重"),&gen_options.dedup);ImGui::Checkbox(tr(zh,"Descending","降序"),&gen_options.sort_desc);
        ImGui::Checkbox(tr(zh,"Sort known factors","排序已知因子"),&gen_options.sort_factors);text("Sort keys",gen_options.sort_by);
        number("Skip curves",gen_options.skip_curves);number("Min exponent (-1: any)",gen_options.min_n);number("Max exponent",gen_options.max_n);
        number("Min input curves",gen_options.min_curves);number("Max input curves",gen_options.max_curves);ImGui::Checkbox("Set assignment N/A",&gen_options.set_has_na);
    }
    if(gen_valid && generator_signature()!=gen_signature)gen_valid=false;
    if(ImGui::Button(tr(zh,"Generate preview","生成预览")))preview_generator();ImGui::SameLine();
    ImGui::BeginDisabled(!gen_valid||controller.busy());if(ImGui::Button(tr(zh,"Append to queue","追加到队列")))apply_generator();ImGui::EndDisabled();
    if(!gen_preview.text.empty()) {
        ImGui::Text("%s | read %d | duplicates %d",gen_valid?"Current preview":"Expired preview",gen_preview.read,gen_preview.duplicates);
        if(ImGui::Button("Copy preview"))ImGui::SetClipboardText(gen_preview.text.c_str());
        for(const auto& e:gen_preview.parse_errors)ImGui::TextWrapped("%s",e.c_str());for(const auto& e:gen_preview.warnings)ImGui::TextWrapped("%s",e.c_str());
        ImGui::BeginChild("preview",ImVec2(0,180),true,ImGuiWindowFlags_HorizontalScrollbar);ImGui::TextUnformatted(gen_preview.text.c_str());ImGui::EndChild();
    }
}
void App::draw_probability(bool zh) {
    using namespace ecmgui::probability_ui; auto& request=probability_request;auto& p=request.parameters;
    ImGui::TextUnformatted(tr(zh,"ECM probability","ECM 概率"));
    double size=size_from_bits(p.factor_bits,request.decimal_digits);
    if(ImGui::InputDouble(tr(zh,"Factor size","因子大小"),&size))p.factor_bits=bits_from_size(size,request.decimal_digits);
    ImGui::Checkbox(tr(zh,"Decimal digits","十进制位数"),&request.decimal_digits);
    ImGui::InputDouble("B1",&p.b1);ImGui::Checkbox(tr(zh,"Stage 2 enabled","启用 Stage2"),&p.stage2);
    ImGui::Checkbox(tr(zh,"Use B2/B1 ratio","使用 B2/B1 比率"),&p.ratio_mode);if(p.ratio_mode)ImGui::InputDouble("B2/B1",&p.b2_ratio);else ImGui::InputDouble("B2",&p.b2);
    ImGui::InputDouble(tr(zh,"Curves","曲线数"),&p.curves);
    const bool profile_changed=chips("curve-profile",probability_profile,{"Suyama / PARAM2","Edwards Z2xZ8","Edwards Z12","PARAM1","PARAM3",tr(zh,"Custom","自定义")});
    const bool basis_changed=ImGui::Checkbox(tr(zh,"Use stage1 empirical calibration","使用 Stage1 经验标定口径"),&probability_empirical);
    if(probability_profile<5&&(profile_changed||basis_changed))p.effective_divisor=ui::curve_divisor(probability_profile,probability_empirical);
    if(ImGui::InputDouble("D_eff",&p.effective_divisor))probability_profile=5;
    ImGui::TextWrapped("%s",tr(zh,"Torsion order is not D_eff. Edwards Z2xZ8: measured stage1 D=26.4081; relative Suyama rebase ~29.0 is a combined-model heuristic, not a direct Stage2 calibration.","挠群阶不等于 D_eff。Edwards Z2xZ8 的 Stage1 经验 D=26.4081；相对 Suyama 归一约 29.0，是联合模型启发值，未经独立 Stage2 标定。"));
    int scope=static_cast<int>(p.scope);if(chips("probability-scope",scope,{"Stage 1",tr(zh,"Combined","联合"),tr(zh,"Stage 2 conditional","Stage2 条件概率")}))p.scope=static_cast<Scope>(scope);
    int solve=static_cast<int>(request.solve);if(ImGui::Combo(tr(zh,"Solve","求解"),&solve,"Forward\0Curves\0Factor size\0B1\0B2\0Effective divisor\0"))request.solve=static_cast<Solve>(solve);
    ImGui::InputDouble(tr(zh,"Target probability","目标概率"),&request.target);ImGui::Checkbox(tr(zh,"Cumulative target","累计目标"),&request.cumulative_target);
    if(ImGui::CollapsingHeader(tr(zh,"Plot range","绘图范围"))){ImGui::InputDouble("Size min",&request.size_min);ImGui::InputDouble("Size max",&request.size_max);ImGui::Checkbox("Auto curves range",&request.auto_curves_range);ImGui::InputDouble("Curves max",&request.curves_max);}
    if(ImGui::Button(tr(zh,"Calculate snapshot","计算快照"))){probability_result=calculate(request);probability_ready=true;}
    if(!probability_ready)return;
    const auto& r=probability_result;if(!r.error.empty()){ImGui::TextWrapped("%s",r.error.c_str());return;}
    ImGui::Text("Snapshot B1=%.9g B2=%.9g curves=%.9g bits=%.6g",r.parameters.b1,effective_b2(r.parameters),r.parameters.curves,r.parameters.factor_bits);
    ImGui::Text("S1 %.8g | S1+S2 %.8g | S2 conditional %.8g",r.probability.stage1,r.probability.combined,r.probability.conditional);
    ImGui::Text("Selected per curve %.8g | cumulative %.8g",r.probability.selected,cumulative(r.probability.selected,r.parameters.curves));
    if(r.request.solve!=Solve::Forward)ImGui::Text("Solution %.12g | residual %.6g%s",r.solved_value,r.residual,r.lower_bound_sufficient?" | lower bound sufficient":"");
    if(ImGui::Button(tr(zh,"Copy snapshot CSV","复制快照 CSV")))ImGui::SetClipboardText(curve_csv(r).c_str());
    const std::vector<Point>* series[]{&r.sizes,&r.size_totals,&r.curve_counts};const char* labels[]{"Per-curve probability / size","Cumulative probability / size","Cumulative probability / curves"};
    for(int i=0;i<3;++i){std::vector<float> s1,s2,conditional;for(const auto& v:*series[i]){s1.push_back(static_cast<float>(v.stage1));s2.push_back(static_cast<float>(v.combined));conditional.push_back(static_cast<float>(v.conditional));}
        ImGui::TextUnformatted(labels[i]);if(!s1.empty()){ImGui::PushID(i);ImGui::PlotLines("S1",s1.data(),static_cast<int>(s1.size()),0,nullptr,0,1,ImVec2(0,65));ImGui::PlotLines("S1+S2",s2.data(),static_cast<int>(s2.size()),0,nullptr,0,1,ImVec2(0,65));ImGui::PlotLines("S2 | S1 missed",conditional.data(),static_cast<int>(conditional.size()),0,nullptr,0,1,ImVec2(0,65));ImGui::PopID();}}
}
void App::draw_gpu(bool zh) {
    ImGui::TextUnformatted(tr(zh,"GPU monitoring (500 ms)","GPU 监控（500 ms）"));
    if(!gpu.available()){ImGui::TextWrapped("%s",no_nvml?tr(zh,"Monitoring disabled","监控已禁用"):gpu.reason().c_str());return;}
    for(const auto& device:gpu.devices()) {
        ImGui::PushID(device.index);ImGui::SeparatorText(device.name.c_str());ecmgui::GpuSample sample;
        if(!gpu.latest(device.index,sample)||!sample.valid){ImGui::TextUnformatted(tr(zh,"Unknown sample","采样未知"));ImGui::PopID();continue;}
        ImGui::Text("GPU %d%% | memory %d%% | %u / %u MB",sample.util_gpu,sample.util_mem,sample.mem_used_mb,device.mem_total_mb);
        ImGui::Text("%u MHz | %u MHz memory | %d C | %.2f / %.2f W",sample.clock_sm_mhz,sample.clock_mem_mhz,sample.temp_c,sample.power_w,sample.power_limit_w);
        ImGui::TextWrapped("%s",ecmgui::GpuMonitor::throttle_text(sample.throttle_reasons).c_str());
        const auto history=gpu.history(device.index);std::vector<float> utilization,power;for(const auto& v:history)if(v.valid){utilization.push_back(static_cast<float>(v.util_gpu));power.push_back(static_cast<float>(v.power_w));}
        if(!utilization.empty()){ImGui::PlotLines("GPU %",utilization.data(),static_cast<int>(utilization.size()),0,nullptr,0,100,ImVec2(0,80));ImGui::PlotLines("Power W",power.data(),static_cast<int>(power.size()),0,nullptr,0,FLT_MAX,ImVec2(0,80));}ImGui::PopID();
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
    ImGui::TextUnformatted(tr(zh,"Appearance and companion","外观与宠物"));
    int theme=state.ui_theme=="light";if(ImGui::Combo(tr(zh,"Theme","主题"),&theme,"Dark\0Light\0"))state.ui_theme=theme?"light":"dark";
    int language=state.ui_language=="zh";if(ImGui::Combo(tr(zh,"Language","语言"),&language,"English\0中文\0"))state.ui_language=language?"zh":"en";
    const float scales[]{0,1,1.25f,1.5f,2};int scale=0;for(int i=1;i<5;++i)if(std::abs(state.ui_scale-scales[i])<.01f)scale=i;
    if(ImGui::Combo(tr(zh,"Scale","缩放"),&scale,"Auto (monitor DPI)\0 100%\0 125%\0 150%\0 200%\0"))state.ui_scale=scales[scale];
    ImGui::Checkbox(tr(zh,"Ely pet","Ely 宠物"),&state.pet_enabled);ImGui::Checkbox(tr(zh,"Animate pet","宠物动画"),&state.pet_animate);
    if(ImGui::Button(tr(zh,"Save UI preferences","保存界面偏好")))status=state.save()?"UI preferences saved":state.notice;
    ImGui::TextWrapped("%s",state.path.u8string().c_str());
    ImGui::SeparatorText(tr(zh,"Workflow","工作流"));ImGui::InputText("TOML",config_input,sizeof(config_input));
    if(ImGui::Button(tr(zh,"Browse workflow","选择工作流"))){std::string path=config_input;const char filter[]="TOML\0*.toml\0All\0*.*\0\0";if(ecmgui::browse_for_file(path,"Open workflow",std::string(filter,sizeof(filter))))std::snprintf(config_input,sizeof(config_input),"%s",path.c_str());}
    ImGui::SameLine();if(ImGui::Button(tr(zh,"Open / reload","打开 / 重载")))load_config(config_input);
    ImGui::InputText("Stage1 executable",stage1_input,sizeof(stage1_input));ImGui::InputText("Stage2 executable",stage2_input,sizeof(stage2_input));
    ImGui::BeginDisabled(controller.busy());ImGui::InputInt(tr(zh,"Worker count (0 follows workflow)","worker 数量（0 跟随工作流）"),&state.worker_count);state.worker_count=std::clamp(state.worker_count,0,64);
    if(ImGui::Button(tr(zh,"Apply worker settings","应用 worker 设置"))){state.worker_stage1_exe=stage1_input;state.worker_stage2_exe=stage2_input;if(ready)configure_workers();}
    if(ready && ImGui::CollapsingHeader(tr(zh,"Edit computation settings","编辑计算设置"))) {
        ImGui::TextWrapped("%s",tr(zh,"Choose an existing field or enter a known schema key. Edits require an explicit save.","选择现有字段或填写已知配置键；编辑后需显式保存。"));
        if(ImGui::BeginChild("fields",ImVec2(0,180),true))for(const auto& line:config.lines())if(line.kind==ecm_config::TomlLine::Kind::KeyValue&&line.section!="gui"){
            const auto name=line.section+"."+line.key+" = "+line.value;if(ImGui::Selectable(name.c_str())){std::snprintf(edit_table,sizeof(edit_table),"%s",line.section.c_str());std::snprintf(edit_key,sizeof(edit_key),"%s",line.key.c_str());std::snprintf(edit_value,sizeof(edit_value),"%s",line.value.c_str());}}
        ImGui::EndChild();ImGui::InputText("Table",edit_table,sizeof(edit_table));ImGui::InputText("Key",edit_key,sizeof(edit_key));ImGui::InputText("Value",edit_value,sizeof(edit_value));
        if(ImGui::BeginCombo(tr(zh,"Available settings","可用设置"),edit_key)){
            for(const auto& option:business_options){const auto name=std::string(option.table)+"."+option.key;if(ImGui::Selectable(name.c_str())){std::snprintf(edit_table,sizeof(edit_table),"%s",option.table);std::snprintf(edit_key,sizeof(edit_key),"%s",option.key);std::snprintf(edit_value,sizeof(edit_value),"%s",config.get(option.table,option.key).c_str());}if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",zh?option.chinese:option.english);}ImGui::EndCombo();
        }
        for(const auto& option:business_options)if(std::string(edit_key)==option.key){ImGui::TextWrapped("%s\n%s",zh?option.chinese:option.english,option.domain);break;}
        if(ImGui::Button(tr(zh,"Stage edit","暂存编辑")))edit_business(edit_table,edit_key,edit_value);ImGui::SameLine();if(ImGui::Button(tr(zh,"Save workflow","保存工作流")))save_business();
        if(config.dirty())ImGui::TextUnformatted(tr(zh,"Unsaved workflow changes","工作流有未保存修改"));
    } ImGui::EndDisabled();
}
}
