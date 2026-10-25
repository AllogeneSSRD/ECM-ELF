#include "../../../src/ecm/application/gui_v2/app.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>
#include "../../../src/ecm/application/gui_v2/ui/scroll.h"
#include "../../../src/ecm/application/gui_v2/ui/probability_presets.h"
#include "../../../src/ecm/application/gui_v2/ui/navigation.h"
#include "../../../src/ecm/application/gui_v2/ui/chart.h"
#include "../../../src/ecm/application/gui_v2/ui/i18n.h"
#include "../../../src/ecm/application/gui_v2/ui/presentation.h"
#include "../../../src/ecm/application/gui_v2/ui/worker_presentation.h"
#include "../../../src/ecm/application/gui_v2/ui/log_presentation.h"
namespace fs=std::filesystem;
static void write(const fs::path& p,const std::string& text){std::ofstream f(p,std::ios::binary);f<<text;assert(f.good());}
static std::string read(const fs::path& p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
// Run with a locking worker whose protocol-query API is deliberately disabled.
static void no_protocol_probe(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"no-protocol";fs::create_directories(dir);
    const auto probe=dir/"unexpected-probe.txt",queue=dir/"worktodo.txt",cfg=dir/"ecm.toml";
    assert(_putenv_s("ECM_TEST_PROTOCOL_PROBE",probe.u8string().c_str())==0);
    const auto setup=[&](const std::string& args,bool autostart=false,const fs::path& exe=fs::path{}){
        write(queue,"[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n");
        write(cfg,"[shared]\nworktodo=\"worktodo.txt\"\n[stage1]\ntarget=\"cpu\"\n[gui]\nNumWorkers=1\nexe=\""+(exe.empty()?fixture:exe).generic_u8string()+"\"\nstage2_exe=\""+fixture.generic_u8string()+"\"\n[workers.1]\nautostart="+(autostart?"1":"0")+"\nextra_args='"+args+"'\n");
    };
    const auto wait=[&](ecmgui_v2::App& app,const auto& done){
        const auto deadline=GetTickCount64()+15000;
        while(!done()&&GetTickCount64()<deadline){app.tick();Sleep(10);}assert(done());
    };
    const auto acquired=[](const ecmgui_v2::App& app){
        for(const auto& line:app.controller.workers()[0].raw)if(line.find("queue_owner: acquired")!=line.npos)return true;
        return false;
    };
    setup("--scenario v2-checkpoint-stop");
    {ecmgui_v2::UiState state;state.path=dir/"manual.ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
     // A real worker owner blocks dispatch even without the query handshake.
     {ecm_queue::OwnerLease owner(queue.u8string(),1);assert(!app.start(1));}
     {ecm_queue::OwnerLease gui(queue.u8string(),1,".gui-control.lock");assert(!app.start(1));}
     assert(app.start(1));wait(app,[&]{return acquired(app);});
     ecmgui_v2::UiState second_state;second_state.path=dir/"second.ui.toml";ecmgui_v2::App second(second_state,stdout);second.initialize(cfg.u8string(),true);assert(!second.start(1));
     app.controller.request_stop(1);wait(app,[&]{return !app.controller.busy();});}
    setup("--scenario v2-checkpoint-stop",true);
    {ecmgui_v2::UiState state;state.path=dir/"auto.ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);app.tick();wait(app,[&]{return acquired(app);});
     app.controller.request_stop(1);wait(app,[&]{return !app.controller.busy();});}
    setup("--scenario ok --lines 0");
    {ecmgui_v2::UiState state;state.path=dir/"switch.ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(!app.controller.workers()[0].stage2);
     write(queue,"[Worker #1]\nECMSTAGE2=1,2,137,-1,\"ed_m137.save\",0,0,10\n");
     assert(app.start(1)&&app.controller.workers()[0].stage2);wait(app,[&]{return acquired(app)&&!app.controller.busy();});}
    setup("--scenario crash-once --lines 0 --marker \""+(dir/"crashed-once").generic_u8string()+"\"");
    {ecmgui_v2::UiState state;state.path=dir/"restart.ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.start(1));
     wait(app,[&]{return app.controller.workers()[0].restarts()==1&&!app.controller.busy();});}
    for(const auto& exe:{dir/"missing.exe",dir/"invalid.exe"}){
        if(exe.filename()=="invalid.exe")write(exe,"not an executable");setup("",false,exe);
        ecmgui_v2::UiState state;state.path=dir/"error.ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
        assert(!app.start(1)&&app.status.find("CreateProcessW failed")!=app.status.npos);
    }
    assert(!fs::exists(probe));assert(_putenv_s("ECM_TEST_PROTOCOL_PROBE","")==0);
    std::cout<<"{\"no_protocol_probe\":true,\"manual_autostart_restart_stage_switch\":true,\"queue_locks_preserved\":true,\"spawn_errors_preserved\":true}\n";
}
int main(int argc,char** argv){
    std::string localization_error;assert(ecmgui_v2::ui::load_languages(ecmgui::exe_dir()+"/localization/v2",localization_error));
    {ecmgui::WorkerView worker;worker.resume_pct=23.8;
     auto display=ecmgui_v2::ui::worker_progress(worker,false);assert(display.label=="23.8% (resumed)"&&display.eta=="--"&&display.fraction>.237f&&display.fraction<.239f);
     worker.progress.valid=true;worker.progress.pct=50;worker.progress.curves_done=5;worker.progress.curves_total=10;worker.progress.s_per_curve=2;worker.progress.eta_s=65;
     display=ecmgui_v2::ui::worker_progress(worker,false);assert(display.label=="5 / 10  (50.0%)"&&display.speed=="2.00"&&display.eta=="65 s"&&display.eta_detail=="65 s (1 m 5 s)");
     using ecmgui_v2::ui::eta_text;
     assert(eta_text(3599.8,true)=="3600 s (1 h 0 m 0 s)");
     assert(eta_text(90061,true)=="90061 s (1 d 1 h 1 m 1 s)");
     assert(eta_text(-1)=="--"&&eta_text(std::numeric_limits<double>::infinity())=="--");
     worker.stage2=true;worker.stage2_status.valid=true;worker.stage2_status.total=10;worker.stage2_status.done=2;
     worker.stage2_status.current=3;worker.stage2_status.sigma=17;worker.stage2_status.curve_model="ed";
     display=ecmgui_v2::ui::worker_progress(worker,false);assert(display.label=="2 / 10  (20.0%)"&&display.speed=="--"&&display.eta=="--");
     const auto phase=ecmgui_v2::ui::worker_phase(worker,false);assert(phase.find("sigma=17")!=std::string::npos&&phase.find("param=")==std::string::npos);
     assert(ecmgui_v2::ui::worker_shape(worker)=="B1=--  B2=--  S=-- bits  D=--  P=--");
     worker.stage2_status.b1=260000000;worker.stage2_status.b2=1300000000000ULL;worker.stage2_status.bits=4283;worker.stage2_status.d=1381380;worker.stage2_status.p=126720;
     assert(ecmgui_v2::ui::worker_shape(worker)=="B1=260000000  B2=1300000000000  S=4283 bits  D=1381380  P=126720");
     const auto direct=ecmgui_v2::ui::worker_construction(worker,0,false);
     assert(direct.find("Edwards")!=std::string::npos&&direct.find("param=")==std::string::npos&&direct.find("B2=1300000000000")!=std::string::npos&&direct.find("sigma=17")!=std::string::npos);
     assert(ecmgui_v2::ui::worker_construction(worker,5000,false).find("B2=5000*B1=1300000000000")!=std::string::npos);
     assert(ecmgui_v2::ui::worker_construction(worker,4999,false).find("*B1=")==std::string::npos);
     assert(ecmgui_v2::ui::worker_construction(worker,std::numeric_limits<unsigned long long>::max(),false).find("*B1=")==std::string::npos);
     worker.stage2_status.curve_model="mont";worker.stage2_status.param=3;
     assert(ecmgui_v2::ui::worker_construction(worker,5000,false).find("Montgomery param=3")!=std::string::npos);
     worker.stage2_status.curve_model="ed";worker.stage2_status.param=-1;
     auto batch=ecmgui_v2::ui::stage2_batch(worker.stage2_status,false);assert(!batch.visible);
     worker.stage2_status.phase="Giant points";worker.stage2_status.batch=2;worker.stage2_status.batches=8;worker.stage2_status.batch_left=12;
     batch=ecmgui_v2::ui::stage2_batch(worker.stage2_status,false);assert(batch.visible&&batch.fraction==.25f&&batch.label=="Giant batch 2 / 8"&&batch.remaining=="Phase remaining 12.0 s");
     worker.stage2_status.batch=9;worker.stage2_status.batch_left=-1;
     batch=ecmgui_v2::ui::stage2_batch(worker.stage2_status,true);assert(batch.fraction==1&&batch.remaining=="阶段剩余 --");
     using ecmgui_v2::ui::log_tone;using ecmgui_v2::ui::LogTone;
     assert(log_tone(ecmgui::parse_line("[2026-10-22 12:00:00] Checkpoint saved: test"))==LogTone::Info);
     assert(log_tone(ecmgui::parse_line("stage2_phase: Giant points elapsed=4"))==LogTone::Stage2);
     assert(log_tone(ecmgui::parse_line("FATAL: test"))==LogTone::Error);
     assert(log_tone(ecmgui::parse_line("WARNING: test"))==LogTone::Warning);
     assert(log_tone(ecmgui::parse_line("factor[0]=123"))==LogTone::Success);}
    int mask=0,center=0,right=4;
    for(int i=0;i<5;++i)assert(ecmgui_v2::ui::move_page(mask,center,right,i,true));
    assert(mask==31&&center==6&&ecmgui_v2::ui::move_page(mask,center,right,6,true));
    assert(center==7&&!ecmgui_v2::ui::move_page(mask,center,right,7,true));
    assert(!ecmgui_v2::ui::move_page(mask,center,right,5,true));
    assert(ecmgui_v2::ui::move_page(mask,center,right,3,false)&&center==7&&!ecmgui_v2::ui::pinned(mask,3));
    assert(!ecmgui_v2::ui::move_page(mask,center,right,-1,true));
    std::string order="0123456";
    assert(ecmgui_v2::ui::reorder_page(order,4,1,false)&&order=="0412356");
    assert(ecmgui_v2::ui::reorder_page(order,0,6,true)&&order=="4123560");
    assert(!ecmgui_v2::ui::reorder_page(order,3,3,false)&&!ecmgui_v2::ui::reorder_page(order,8,2,false));
    using namespace ecmgui_v2::ui;
    {LogVisibility logs;logs.change_view(0);assert(logs.visible(true,false,true)&&!logs.visible(true,false,false));
     logs.change_view(1);logs.manual=true;assert(logs.visible(false,false,false));logs.change_view(2);assert(!logs.visible(true,false,false));
     assert(logs.visible(true,true,false)&&!logs.visible(false,true,false));
     std::vector<ChartSeries> samples{{"speed",{},{{0,100},{1,1.75},{2,2.25},{3,std::numeric_limits<double>::quiet_NaN()}}}};
     const auto fit=chart_bounds(samples,0,true,1,3);assert(fit.ymin>0&&fit.ymin<1.75&&fit.ymax>2.25&&fit.ymax<3);
     assert(chart_bounds(samples).ymin==0&&chart_bounds(samples).ymax>=100);
     const auto none=chart_bounds(samples,0,true,4,5);assert(none.ymin==0&&none.ymax==1);
     samples[0].points={{0,.001},{1,.001}};const auto flat=chart_bounds(samples,0,true);assert(flat.ymin<.001&&flat.ymax>.001&&flat.ymin>=0);
     samples[0].visible=false;const auto hidden=chart_bounds(samples,0,true);assert(hidden.ymin==0&&hidden.ymax==1);}
    assert(language_label("english.xml")=="english"&&language_label("custom.name.XML")=="custom.name");
    assert(language_label("自定义语言.xml")=="自定义语言");
    assert(memory_occupancy(4096,8192)==50);assert(std::isnan(memory_occupancy(1,0)));
    assert(power_value(9.4,55)==9.4&&power_value(165,55)==165);
    assert(power_value(0,55)==0&&power_readout(0,55)=="0.0 / 55 W");
    assert(std::isnan(power_value(590,55))&&std::isnan(power_value(165.000001,55)));
    assert(power_readout(590,55)=="-- / 55 W"&&power_readout(165,55)=="165.0 / 55 W");
    assert(power_value(590,0)==590&&power_readout(590,0)=="590.0 / -- W");
    assert(power_readout(590,std::numeric_limits<double>::quiet_NaN())=="590.0 / -- W");
    assert(power_value(590,285)==590); // Per-device/sample limit, not a global ceiling.
    for(double bad:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})assert(std::isnan(power_value(bad,55)));
    assert(std::isnan(power_value(9,55,false))&&power_readout(9,55,false)=="-- / 55 W");
    std::vector<ChartSeries> power_charts{{"power",{},{{-1,power_value(9.2,55)},{-.5,power_value(590,55)},{0,power_value(9.4,55)}}}};
    assert(power_charts[0].points.size()==3&&std::isnan(power_charts[0].points[1].y));
    assert(chart_bounds(power_charts).ymax<11&&chart_nearest(power_charts[0].points,-.5)!=1);
    power_charts[0].points={{0,power_value(590,55)}};assert(chart_nearest(power_charts[0].points,0)==-1&&chart_bounds(power_charts).ymax==1);
    assert(percent(.95)==95&&probability(95)==.95);assert(integer_step(29.9,3)==10&&integer_step(15,3)==5);
    std::vector<ChartSeries> charts{{"measured",{},{{-60,20},{-30,std::numeric_limits<double>::quiet_NaN()},{0,80}}}};
    const auto bounds=chart_bounds(charts);assert(bounds.xmin==-60&&bounds.xmax==0&&bounds.ymax>=80);
    assert(chart_nearest(charts[0].points,-29)==2);assert(chart_bounds(charts,100).ymax==100);
    charts[0].visible=false;const auto empty=chart_bounds(charts);assert(empty.xmax>empty.xmin&&empty.ymax>empty.ymin);
    assert(message("Preview ready: C:/test.toml",true)=="预览已就绪：C:/test.toml");assert(message("raw worker payload",true)=="raw worker payload");
    const auto edwards=ecmgui_v2::ui::curve_divisor(1);assert(edwards>28&&edwards<31);
    assert(ecmgui_v2::ui::curve_divisor(1,true)==26.408122047632308);
    assert(edwards_profiles()==(std::array<int,4>{1,2,6,7}));
    assert(curve_divisor(5,true)==20.15667357813588&&curve_divisor(5)==curve_divisor(0));
    assert(curve_divisor(6,true)==14.16200621274284&&curve_divisor(7,true)==8.510017237617587);
    for(const auto& preset:curve_presets){assert(curve_divisor(preset.id)>0);if(preset.generator)assert(std::string(preset.generator)=="e"||std::string(preset.generator)=="m0"||std::string(preset.generator)=="m2"||std::string(preset.generator)=="m3");}
    ecmgui_v2::ui::SmoothScroll scroll;scroll.add(0,-3);double total=0,x=0,y=0;int steps=0;
    while(scroll.active()){scroll.step(.016,x,y);assert(y<=0);total+=y;assert(++steps<100);}
    assert(std::abs(total+3)<1e-9&&steps>1);
    {auto* context=ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={600,400};io.DeltaTime=1.f/60;
     io.ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;io.Fonts->AddFontDefault();unsigned char* pixels;int width,height;io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
     for(float size:{15.f,24.f}){
         ImGui::GetStyle().FontSizeBase=size;ImGui::GetStyle().FramePadding={10,6};bool checked=false;CheckboxMetrics metrics;
         const auto frame=[&](bool disabled=false,bool mixed=false,bool focus=false){
             ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({500,300});ImGui::Begin("checkbox-contract",nullptr,ImGuiWindowFlags_NoSavedSettings);
             if(focus)ImGui::SetKeyboardFocusHere();ImGui::BeginDisabled(disabled);ImGui::PushItemFlag(ImGuiItemFlags_MixedValue,mixed);
             const bool changed=checkbox("Option",&checked,&metrics);ImGui::PopItemFlag();ImGui::EndDisabled();ImGui::End();ImGui::Render();return changed;
         };
         frame();assert(metrics.box<metrics.native_frame*.8f&&metrics.stroke<metrics.box*.1f);
         io.AddMousePosEvent(metrics.min.x+metrics.box*.5f,metrics.min.y+metrics.box*.5f);frame();
         io.AddMouseButtonEvent(0,true);frame();io.AddMouseButtonEvent(0,false);assert(frame()&&checked);
         io.AddMousePosEvent(metrics.max.x+15,metrics.min.y+metrics.box*.5f);frame();
         io.AddMouseButtonEvent(0,true);frame();io.AddMouseButtonEvent(0,false);assert(frame()&&!checked);
         io.AddMouseButtonEvent(0,true);frame(true);io.AddMouseButtonEvent(0,false);assert(!frame(true)&&!checked);
         io.AddMousePosEvent(-100,-100);frame(false,false,true);frame();
         io.AddKeyEvent(ImGuiKey_Tab,true);frame();io.AddKeyEvent(ImGuiKey_Tab,false);frame();
         io.AddKeyEvent(ImGuiKey_Space,true);const bool key_down=frame();io.AddKeyEvent(ImGuiKey_Space,false);const bool key_up=frame();assert((key_down||key_up)&&checked);
         frame(false,true);assert(metrics.mixed);
     }ImGui::DestroyContext(context);}
    assert(argc==2||argc==3);const auto root=fs::absolute(argv[1]);assert(!fs::exists(root));fs::create_directories(root);
    {using namespace ecmgui_v2::ui;
     auto previous=no_phase_seconds(),tuned=no_phase_seconds();previous[5]=20;tuned[5]=80;tuned[2]=10;
     auto timing=stage2_timing(previous,tuned,0);assert(timing.seconds[5]==20&&timing.sources[5]==1&&timing.seconds[2]==10&&timing.sources[2]==2);
     assert(std::abs(timing.fractions()[5]-2./3)<1e-6);
     assert(stage2_timing(previous,tuned,1).seconds[2]==-1&&stage2_timing(previous,tuned,2).seconds[5]==80);
     tuned[2]=std::numeric_limits<double>::infinity();assert(stage2_timing(no_phase_seconds(),tuned,2).seconds[2]==-1);
     assert(stage2_phase_index("Giant points, G trees and fold")==5&&stage2_phase_index("unrecognized")==-1);
     ecmgui::Stage2Status status;status.bits=4283;status.d=1381380;status.b2=1300000000000ULL;
     status.phase="Publish result";assert(stage2_bounded_fraction(status,1)==.99f);
     status.phase="complete";status.current=2;status.done=1;assert(stage2_bounded_fraction(status,1)==.99f);
     status.phase="curve_done";assert(stage2_bounded_fraction(status,0)==1);status.phase.clear();
     const auto tune=root/"tune.toml";
     write(tune,"[profile]\nformat=8\nunit=\"full_stage2\"\n[condition.0]\nuuid_hex=\"aabb\"\n[width.4283.table.0]\ncondition=0\narithmetic_bits=4283\nphase_accounting=\"exclusive_engine_v1\"\nphase_baby_seconds=10\nmatrix=[\n[\"d\",\"b2\",\"phase_giant_loop_seconds\",\"checked\"],\n[1381380,1300000000000,80,6591],\n]\n");
     Stage2Tune table;table.load(tune);auto row=table.lookup(status,"GPU-aabb");assert(table.notice.empty()&&row[2]==10&&row[5]==80);
     assert(table.lookup(status,"GPU-ccdd")[5]==-1);status.d=210;assert(table.lookup(status,"GPU-aabb")[5]==-1);status.d=1381380;
     const auto bad=root/"bad-tune.toml";write(bad,"[profile]\nformat=8\nunit=\"full_stage2\"\n[width.4283.table.0]\nmatrix=[\n[\"d\",\"d\"],\n");table.load(bad);assert(!table.notice.empty()&&table.lookup(status,"GPU-aabb")[5]==-1);
     ecmgui::WorkerView w;w.exe="fixture.exe";w.device=0;
     w.raw={"stage2_plan: save=test.save requested=2 available=2 completed=0 B2=1300000000000 D=1381380",
       "curve_start: 1/2 record=1 sigma=17 B1=20 checksum=verified curve_model=ed",
       "real_shape: B1=20 B2=1300000000000 S_bits=4283 D=1381380 P=126720",
       "stage2_phase: Generate and normalize baby points previous=0 elapsed=0",
       "stage2_phase: Giant points previous=10 elapsed=10",
       "stage2_phase: Normalize fold and descend F tree previous=20 elapsed=30",
       "curve_done: 1/2 record=1 seconds=35"};
     Stage2TimingHistory history;const auto cache=root/"stage2-times.json";history.load(cache);history.observe(w,"GPU-aabb");
     for(const auto& line:w.raw){const auto parsed=ecmgui::parse_line(line);if(parsed.stage2.kind!=ecmgui::Stage2Event::Kind::None)w.stage2_status.apply(parsed.stage2);}
     const auto scope=stage2_scope(w,"GPU-aabb");assert(history.lookup(scope)[2]==10&&history.lookup(scope)[5]==20);
     Stage2TimingHistory restored;restored.load(cache);assert(restored.notice.empty()&&restored.lookup(scope)[5]==20);
     history.observe(w,"GPU-aabb");assert(history.lookup(scope)[5]==20); // No double-count on repeated view/tick.
     const auto corrupt=root/"corrupt-times.json";write(corrupt,"{broken");Stage2TimingHistory retained;retained.load(corrupt);retained.observe(w,"GPU-aabb");assert(read(corrupt)=="{broken"&&!retained.notice.empty());
    }
    {
        // Exercise external edits, Unicode paths, fallback and transactional reload.
        const auto loc=root/fs::u8path("语言资源");fs::create_directories(loc);
        const auto original=fs::u8path(language_directory());
        for(const auto& item:fs::directory_iterator(original))if(item.path().extension()==".xml")fs::copy_file(item.path(),loc/item.path().filename());
        const auto english=read(loc/"english.xml"),chinese=read(loc/"chineseSimplified.xml");
        auto changed=chinese;size_t at=0;bool found=false;
        while((at=changed.find("name=\"已停止\"",at))!=changed.npos){found=true;changed.replace(at,std::strlen("name=\"已停止\""),"name=\"外部编辑测试\"");at+=std::strlen("name=\"外部编辑测试\"");}
        assert(found);write(loc/"chineseSimplified.xml",changed);
        std::string error;assert(load_languages(loc.u8string(),error));assert(message("Stopped",true)=="外部编辑测试");
        // A missing translated entry falls back to the external English baseline.
        auto fallback=chinese;const auto begin=fallback.find("<Item id=\"status.stopped\"");assert(begin!=fallback.npos);fallback.erase(begin,fallback.find("/>",begin)+2-begin);write(loc/"chineseSimplified.xml",fallback);
        assert(load_languages(loc.u8string(),error));assert(std::string(text("status.stopped",true))=="Stopped");
        auto invalid=chinese;const auto format=invalid.find("%.1f s");assert(format!=invalid.npos);invalid.replace(format,4,"%s");write(loc/"chineseSimplified.xml",invalid);
        assert(!load_languages(loc.u8string(),error)&&error.find("format mismatch")!=error.npos);
        assert(std::string(text("status.stopped",true))=="Stopped");
        write(loc/"chineseSimplified.xml",chinese);write(loc/"english.xml","<EcmGui><Native-Langue><Panel id=\"ui\"><Item id=\"bad\" name=\"broken\"/>");
        assert(!load_languages(loc.u8string(),error));
        write(loc/"english.xml",english);assert(load_languages(loc.u8string(),error));
        auto duplicate=chinese;const auto close=duplicate.find("</Panel>");assert(close!=duplicate.npos);duplicate.insert(close,"<Item id=\"status.stopped\" name=\"Duplicate\"/>");write(loc/"chineseSimplified.xml",duplicate);
        assert(!load_languages(loc.u8string(),error)&&error.find("Duplicate")!=error.npos);
        assert(message("Stopped",true)=="已停止");fs::remove(loc/"chineseSimplified.xml");
        assert(load_languages(loc.u8string(),error)&&!error.empty());assert(message("Stopped",true)=="Stopped");
        write(loc/"chineseSimplified.xml",chinese);assert(load_languages(loc.u8string(),error));
        assert(message("Stopped",true)=="已停止");
        write(loc/fs::u8path("自定义语言.xml"),changed);assert(load_languages(loc.u8string(),error,"自定义语言.xml"));
        assert(selected_language_file()=="自定义语言.xml"&&message("Stopped",true)=="外部编辑测试");
        const auto files=language_files();assert(std::find(files.begin(),files.end(),"自定义语言.xml")!=files.end());
        assert(std::none_of(files.begin(),files.end(),[](const auto& name){return name.rfind("options.",0)==0;}));
        assert(!load_languages(loc.u8string(),error,"../chineseSimplified.xml")&&selected_language_file()=="自定义语言.xml");
        assert(!load_languages(loc.u8string(),error,"options.english.xml")&&selected_language_file()=="自定义语言.xml");
        write(loc/"bad-custom.xml",invalid);assert(!load_languages(loc.u8string(),error,"bad-custom.xml")&&selected_language_file()=="自定义语言.xml");
        const std::string legacy="<EcmGui><Native-Langue><Panel id=\"ui\"><Item id=\"start_e4bb9f1e\" name=\"旧版开始\"/><Item id=\"message_24\" name=\"旧版停止\"/></Panel></Native-Langue></EcmGui>";
        write(loc/"legacy.xml",legacy);assert(load_languages(loc.u8string(),error,"legacy.xml"));
        assert(std::string(text("action.start",true))=="旧版开始"&&std::string(text("start_e4bb9f1e",true))=="旧版开始"&&message("Stopped",true)=="旧版停止");
        auto mixed=legacy;const auto panel_end=mixed.find("</Panel>");mixed.insert(panel_end,"<Item id=\"action.start\" name=\"Duplicate\"/>");write(loc/"mixed.xml",mixed);
        assert(!load_languages(loc.u8string(),error,"mixed.xml")&&error.find("Duplicate")!=error.npos&&selected_language_file()=="legacy.xml");
        write(loc/"bad-legacy.xml","<EcmGui><Native-Langue><Panel id=\"ui\"><Item id=\"worker_version_s_8c3af2a1\" name=\"%d\"/></Panel></Native-Langue></EcmGui>");
        assert(!load_languages(loc.u8string(),error,"bad-legacy.xml")&&error.find("format mismatch")!=error.npos&&selected_language_file()=="legacy.xml");
        assert(load_languages(original.u8string(),error));
    }
    const auto cfg=root/"ecm.toml",queue=root/"worktodo.txt";
    write(cfg,"# preserve comment\n[shared]\nworktodo=\"worktodo.txt\"\n[stage1]\ntarget=\"cpu\"\n[gui]\nNumWorkers=2\nfont_size=\"auto\"\n[workers.1]\nautostart=0\n[workers.2]\nautostart=0\n");
    write(queue,"# queue preserved\n[Worker #1]\n[Worker #2]\n");
    ecmgui_v2::UiState state;state.path=root/"ui.toml";ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.ready);
    const auto original=read(cfg);
    std::strcpy(app.gen_input,"ECM2=1,2,127,-1,100,1,0\n");app.gen_options.use_recommended=false;app.gen_options.curves_fixed=4;app.gen_worker=2;
    assert(app.preview_generator());assert(app.gen_preview.segments.size()==1&&app.gen_preview.segments.front().worker==2);
    std::strcpy(app.gen_b1,"200");assert(!app.apply_generator());assert(read(queue)=="# queue preserved\n[Worker #1]\n[Worker #2]\n");
    assert(app.preview_generator());write(queue,read(queue)+"# external edit\n");assert(!app.apply_generator());
    assert(app.preview_generator());assert(app.apply_generator());assert(!app.apply_generator());
    const auto appended=read(queue);assert(appended.find("ECMSTAGE1=")>appended.find("[Worker #2]"));
    app.gen_options.curve_type="m3";assert(!app.preview_generator());app.gen_options.curve_type="m0";
    assert(!app.edit_business("gui","font_size","20"));assert(!app.edit_business("stage1","not_known","4"));
    assert(app.edit_business("shared","b2_ratio","20"));assert(!app.start(1));assert(app.save_business());
    assert(read(cfg).find("# preserve comment")!=std::string::npos&&read(cfg).find("font_size=\"auto\"")!=std::string::npos);
    assert(app.edit_business("shared","b2_ratio","30"));write(cfg,read(cfg)+"# changed externally\n");assert(!app.save_business());assert(app.load_config(cfg.u8string()));
    using namespace ecmgui::probability_ui;Request request;const auto forward=calculate(request);assert(forward.error.empty());
    auto inverse=request;inverse.solve=Solve::Curves;inverse.target=.75;const auto solved=calculate(inverse);assert(solved.error.empty());assert(cumulative(solved.probability.selected,solved.parameters.curves)>=.75);
    const auto csv=curve_csv(forward);request.parameters.b1=10;assert(curve_csv(forward)==csv);
    {auto* context=ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;
     assert(app.edit_business("shared","b2_ratio","40")&&app.save_layout_config());
     assert(!app.config.dirty()&&read(cfg).find("b2_ratio = 40")!=std::string::npos&&fs::exists(state.path));
     const auto saved_state=read(state.path);const auto saved_config=read(cfg);
     assert(app.save_layout_config()&&read(cfg)==saved_config); // Clean workflow is not rewritten.
     assert(app.edit_business("shared","b2_ratio","50"));write(cfg,saved_config+"# external save conflict\n");
     assert(!app.save_layout_config()&&read(state.path)==saved_state&&app.config.dirty());
     assert(app.load_config(cfg.u8string()));state.writable=false;state.notice="fixture read-only sidecar";
     assert(app.edit_business("shared","b2_ratio","60")&&!app.save_layout_config()&&read(cfg)==saved_config+"# external save conflict\n");
     state.writable=true;assert(app.load_config(cfg.u8string()));ImGui::DestroyContext(context);}
    assert(original.find("font_size=\"auto\"")!=std::string::npos);
    {const auto path=root/"gp-config.toml";write(path,"# preserve GP comment\n[shared]\nb2_ratio=10\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n");
     ecmgui_v2::UiState preferences;ecmgui_v2::App gp_app(preferences,stdout);assert(gp_app.load_config(path.u8string()));
     assert(gp_app.edit_business("shared","b2_ratio","25"));gp_app.group_order_input.gp="C:/GP Folder/gp.exe";assert(gp_app.save_group_order_gp());
     assert(gp_app.config.dirty()&&gp_app.config.get("shared","b2_ratio")=="25");
     const auto saved=read(path);assert(saved.find("b2_ratio=10")!=saved.npos&&saved.find("# preserve GP comment")!=saved.npos&&saved.find("C:/GP Folder/gp.exe")!=saved.npos);
     write(path,saved+"# external edit\n");gp_app.group_order_input.gp="other.exe";assert(!gp_app.save_group_order_gp()&&read(path)==saved+"# external edit\n");}
    {ecmgui_v2::UiState preferences;ecmgui_v2::App notify(preferences,stdout);
     auto& job=notify.group_order_job;job.result.type='o';job.result.set("status",ecmgui_v2::go_json::Value::string("complete"));
     notify.notify_group_order();notify.notify_group_order();
     assert(notify.group_order_completions==2&&notify.toasts.size()==2&&notify.group_order_success&&!notify.toasts.back().error);
     for(const char* status:{"partial","resource_limit","invalid_input"}){
         job.result.set("status",ecmgui_v2::go_json::Value::string(status));notify.notify_group_order();assert(!notify.group_order_success&&notify.toasts.back().error);}
     ecmgui_v2::OrderInput input{"1009","26",(root/"missing-gp.exe").u8string(),0,1};
     assert(job.start(input,"missing-helper.exe")&&!job.busy());notify.notify_group_order();assert(!notify.group_order_success&&notify.toasts.back().error);
     input.sigma="invalid";assert(!job.start(input,"missing-helper.exe"));notify.notify_group_order();assert(!notify.group_order_success&&notify.toasts.back().error);
     assert(notify.group_order_completions==7);}
    {ecmgui_v2::UiState preferences;ecmgui_v2::App text_app(preferences,stdout);
     text_app.group_order_loaded=true;text_app.group_order_input={"1009","26","",0,1};text_app.group_order_job.snapshot=text_app.group_order_input;
     text_app.group_order_job.result=ecmgui_v2::go_json::parse(R"({"status":"complete","backend":"gp","all_proven":true,"modulus_primality":"proven","group_computed":true,"point_minimality":true,"group_order":"996","point_order":"249","cofactor":"1","group_factors":[{"prime":"2","exponent":"2"},{"prime":"3","exponent":"1"},{"prime":"83","exponent":"1"}],"point_factors":[{"prime":"3","exponent":"1"},{"prime":"83","exponent":"1"}]})");
     auto* context=ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={1800,1400};io.DeltaTime=1.f/60;
     io.Fonts->AddFontDefault();unsigned char* pixels;int width,height;io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
     static std::string copied;auto& platform=ImGui::GetPlatformIO();
     platform.Platform_SetClipboardTextFn=[](ImGuiContext*,const char* value){copied=value;};
     platform.Platform_GetClipboardTextFn=[](ImGuiContext*){return copied.c_str();};
     const auto frame=[&]{ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({1500,1200});ImGui::Begin("order-text-contract",nullptr,ImGuiWindowFlags_NoSavedSettings);text_app.draw_group_order(false);ImGui::End();ImGui::Render();};
     frame();frame();const auto lo=text_app.group_order_output_min,hi=text_app.group_order_output_max;
     io.AddMousePosEvent(lo.x+1,lo.y+ImGui::GetFontSize()*.5f+2);frame();io.AddMouseButtonEvent(0,true);frame();
     io.AddMousePosEvent(hi.x-4,hi.y-3);frame();io.AddMouseButtonEvent(0,false);frame();assert(text_app.group_order_selection>0);
     const auto copy=[&]{copied.clear();io.AddKeyEvent(ImGuiMod_Ctrl,true);io.AddKeyEvent(ImGuiKey_C,true);frame();io.AddKeyEvent(ImGuiKey_C,false);io.AddKeyEvent(ImGuiMod_Ctrl,false);frame();};
     copy();assert(copied=="Group order #E(F_p): 996\n    = 2^2 × 3 × 83\nPoint order ord(P): 249\n    = 3 × 83");
     const auto original=copied;io.AddInputCharactersUTF8("overwrite");frame();copy();assert(copied==original);
     ImGui::DestroyContext(context);}
    if(argc==3)no_protocol_probe(root,fs::absolute(argv[2]));
    std::cout<<"{\"passed\":true,\"generator_target\":2,\"stale_preview_refused\":true,\"mtime_refused\":true,\"cpu_m3_refused\":true,\"legacy_ui_preserved\":true,\"external_config_refused\":true,\"probability_snapshot\":true,\"order_mouse_selection_clipboard_readonly\":true}\n";
}
