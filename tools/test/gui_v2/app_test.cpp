#include "../../../src/ecm/application/gui_v2/app.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>
#include "../../../src/ecm/application/gui_v2/ui/scroll.h"
#include "../../../src/ecm/application/gui_v2/ui/probability_presets.h"
namespace fs=std::filesystem;
static void write(const fs::path& p,const std::string& text){std::ofstream f(p,std::ios::binary);f<<text;assert(f.good());}
static std::string read(const fs::path& p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv){
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
