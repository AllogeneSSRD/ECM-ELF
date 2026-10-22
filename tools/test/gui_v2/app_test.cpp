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
int main(int argc,char** argv){
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
    for(int i=0;i<6;++i)assert(ecmgui_v2::ui::move_page(mask,center,right,i,true));
    assert(mask==63&&center==6&&!ecmgui_v2::ui::move_page(mask,center,right,6,true));
    assert(ecmgui_v2::ui::move_page(mask,center,right,3,false)&&center==3&&!ecmgui_v2::ui::pinned(mask,3));
    assert(!ecmgui_v2::ui::move_page(mask,center,right,-1,true));
    using namespace ecmgui_v2::ui;
    assert(memory_occupancy(4096,8192)==50);assert(std::isnan(memory_occupancy(1,0)));
    assert(percent(.95)==95&&probability(95)==.95);assert(integer_step(29.9,3)==10&&integer_step(15,3)==5);
    std::vector<ChartSeries> charts{{"measured",{},{{-60,20},{-30,std::numeric_limits<double>::quiet_NaN()},{0,80}}}};
    const auto bounds=chart_bounds(charts);assert(bounds.xmin==-60&&bounds.xmax==0&&bounds.ymax>=80);
    assert(chart_nearest(charts[0].points,-29)==2);assert(chart_bounds(charts,100).ymax==100);
    charts[0].visible=false;const auto empty=chart_bounds(charts);assert(empty.xmax>empty.xmin&&empty.ymax>empty.ymin);
    assert(message("Preview ready: C:/test.toml",true)=="预览已就绪：C:/test.toml");assert(message("raw worker payload",true)=="raw worker payload");
    const auto edwards=ecmgui_v2::ui::curve_divisor(1);assert(edwards>28&&edwards<31);
    assert(ecmgui_v2::ui::curve_divisor(1,true)==26.408122047632308);
    ecmgui_v2::ui::SmoothScroll scroll;scroll.add(0,-3);double total=0,x=0,y=0;int steps=0;
    while(scroll.active()){scroll.step(.016,x,y);assert(y<=0);total+=y;assert(++steps<100);}
    assert(std::abs(total+3)<1e-9&&steps>1);
    assert(argc==2);const auto root=fs::absolute(argv[1]);assert(!fs::exists(root));fs::create_directories(root);
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
    assert(original.find("font_size=\"auto\"")!=std::string::npos);
    std::cout<<"{\"passed\":true,\"generator_target\":2,\"stale_preview_refused\":true,\"mtime_refused\":true,\"cpu_m3_refused\":true,\"legacy_ui_preserved\":true,\"external_config_refused\":true,\"probability_snapshot\":true}\n";
}
