#include "../../../src/ecm/application/gui_v2/app.h"
#include "../../../src/ecm/application/gui_v2/ui/i18n.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <chrono>
#include <functional>
#include <cstring>
namespace fs=std::filesystem;
using namespace ecmgui_v2;
static const std::string a="GPU-11111111-1111-1111-1111-111111111111",b="GPU-22222222-2222-2222-2222-222222222222";
static void until(const std::function<bool()>& done){const auto end=GetTickCount64()+8000;while(!done()){assert(GetTickCount64()<end);Sleep(5);}}
static ClockSnapshot wait(GpuClocks& clocks,int index){until([&]{const auto s=clocks.get(index);return s.queried&&!s.busy;});return clocks.get(index);}
static void write(const fs::path& path,const std::string& text){std::ofstream f(path,std::ios::binary);f<<text;assert(f.good());}
static std::string read(const fs::path& path){std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
static void automation_contract(){
    ClockAutoPolicy policy;policy.range={2600,2600};policy.enabled=true;
    ecmgui::GpuSample sample;sample.valid=true;sample.clock_sm_mhz=2600;
    const auto at=[&](unsigned long long ms,long long hour,unsigned mhz,bool busy=false){sample.ts_ms=ms;sample.clock_sm_mhz=mhz;return policy.evaluate(ms,hour,sample,busy);};
    assert(at(10000,1,2600,true)==ClockAutoReason::None&&policy.startup);
    assert(at(10000,1,2600)==ClockAutoReason::Startup&&!policy.startup);
    assert(at(10500,1,2600)==ClockAutoReason::None);
    assert(at(11000,2,2600)==ClockAutoReason::Hourly); // Every hour, even in range.
    assert(at(11500,2,2600)==ClockAutoReason::None);
    for(unsigned long long ms=12000;ms<17000;ms+=500)assert(at(ms,2,2700)==ClockAutoReason::None);
    assert(at(16999,2,2700)==ClockAutoReason::None);
    assert(at(17000,2,2700)==ClockAutoReason::Deviation);
    for(unsigned long long ms=17500;ms<23000;ms+=500)assert(at(ms,2,2700)==ClockAutoReason::None); // One write per episode.
    assert(at(23000,2,2600)==ClockAutoReason::None);
    for(unsigned long long ms=23500;ms<28500;ms+=500)assert(at(ms,2,2500)==ClockAutoReason::None);
    assert(at(28500,2,2500)==ClockAutoReason::Deviation); // Lower bound is checked too.
    sample.valid=false;assert(at(29000,2,2700)==ClockAutoReason::None);sample.valid=true;
    assert(at(29500,2,2700)==ClockAutoReason::None);sample.ts_ms=29500;
    assert(policy.evaluate(32000,2,sample,false)==ClockAutoReason::None&&!policy.outside); // Stale samples break continuity.
    assert(at(32500,2,2700)==ClockAutoReason::None);
    assert(at(35000,2,2700)==ClockAutoReason::None); // Missing sample gap starts a new interval.
    assert(at(35500,3,2600)==ClockAutoReason::Hourly);
    policy.enabled=false;assert(at(36000,4,3000)==ClockAutoReason::None);
}
int main(int argc,char** argv){
    if(argc==4&&std::strcmp(argv[1],"--read-only-probe")==0){
        ecmgui::GpuMonitor monitor;GpuDetailsCache details;GpuClocks clocks;
        const std::string mode=argv[2];assert(monitor.start(500));
        if(mode=="metadata")details.start(monitor.nvml_source(),monitor.devices());
        clocks.start(argv[3]);
        if(mode=="metadata")until([&]{return details.ready();});
        for(int repeat=0;repeat<3;++repeat)for(const auto& device:monitor.devices()){
            const auto uuid=mode=="metadata"?details.get(device.index).uuid:std::string{};
            assert(clocks.query(device.index,uuid));const auto s=wait(clocks,device.index);
            std::cout<<"index="<<device.index<<" repeat="<<repeat<<" error="<<s.error<<" range="<<s.supported.low<<","<<s.supported.high<<" detail="<<s.detail<<"\n";
        }return 0;
    }
    automation_contract();
    assert(argc==4);const auto out=fs::absolute(argv[3]);assert(!fs::exists(out));fs::create_directories(out);
    const auto module=LoadLibraryW(fs::absolute(argv[1]).c_str());assert(module);
    const auto configure=reinterpret_cast<void(*)(int,int,int,int,int)>(GetProcAddress(module,"fakeConfigure"));
    const auto count=reinterpret_cast<int(*)(int)>(GetProcAddress(module,"fakeRead"));assert(configure&&count);
    ClockProfiles profiles;assert(parse_clock_profiles("",profiles)&&profiles.empty());
    assert(parse_clock_profiles(a+":2600-2600;"+b+":210-3000",profiles)&&profiles.size()==2);
    assert(encode_clock_profiles(profiles)==a+":2600-2600;"+b+":210-3000");
    std::set<std::string> autos;assert(parse_clock_profiles(a+":2600-2600:auto;"+b+":210-3000",profiles,&autos)&&autos==std::set<std::string>{a});
    assert(encode_clock_profiles(profiles,autos)==a+":2600-2600:auto;"+b+":210-3000");
    assert(!parse_clock_profiles(a+":2600-2600:false",profiles,&autos));
    for(const auto& text:{a+":0-2600",a+":2700-2600",a+":210-100001",a+":210-3000;",a+":210-3000;"+a+":210-3000",a+":+210-3000",a+":210-3000junk",std::string("GPU-bad:1-2")}){
        const auto before=profiles;assert(!parse_clock_profiles(text,profiles)&&profiles==before);
    }
    {
        GpuClocks clocks;clocks.start(fs::absolute(argv[1]).u8string());assert(clocks.query(0,a));auto s=wait(clocks,0);
        assert(s.uuid==a&&s.supported==ClockRange({210,3105})&&s.can_apply&&s.can_reset&&!s.applied_known&&count(0)==0);
        assert(!clocks.apply(0,b,{2600,2600})&&!clocks.apply(0,a,{200,2600})&&!clocks.apply(0,a,{2700,2600}));
        configure(0,0,1,0,0); // Index 0 now refers to the other GPU; writes must still resolve UUID a.
        assert(clocks.apply(0,a,{2600,2600}));s=wait(clocks,0);
        assert(!s.error&&s.applied==ClockRange({2600,2600})&&count(0)==1&&count(2)==0&&count(3)==2600&&count(4)==2600);
        configure(4,0,0,0,0);assert(clocks.apply(0,a,{2500,2700}));s=wait(clocks,0);assert(s.error==4&&s.applied==ClockRange({2600,2600}));
        configure(0,0,0,0,0);assert(clocks.reset(0,a));s=wait(clocks,0);assert(!s.error&&s.reset_accepted&&count(1)==1);
        configure(0,4,0,0,0);const int writes=count(0);assert(clocks.apply(0,a,{2600,2600}));s=wait(clocks,0);assert(s.error==2&&count(0)==writes);
        configure(0,0,0,1,0);assert(clocks.query(0,a));s=wait(clocks,0);assert(s.error==2&&!clocks.apply(0,a,{2600,2600}));
        configure(0,2,0,0,0);assert(clocks.query(0,a));s=wait(clocks,0);assert(!s.error&&s.pstate_range&&s.supported==ClockRange({210,3000}));
        configure(0,1,0,0,0);assert(clocks.query(0,a));s=wait(clocks,0);assert(s.error==3&&!s.can_apply&&s.can_reset);
        assert(clocks.reset(0,a));assert(!wait(clocks,0).error); // Reset does not need a supported-clock list.
        configure(0,3,0,0,0);assert(clocks.query(0,a));s=wait(clocks,0);assert(s.error&&!s.supported.valid());
        configure(0,5,0,0,0);assert(clocks.query(0,a));s=wait(clocks,0);assert(s.error==999&&!s.supported.valid());
        assert(clocks.query(0,a));s=wait(clocks,0);assert(!s.error&&s.supported.valid()); // Explicit refresh after a transient driver failure.
        configure(0,0,0,0,100);const auto before=GetTickCount64();assert(clocks.query(0,a));assert(GetTickCount64()-before<80&&clocks.get(0).busy);
        assert(!clocks.query(0,a)&&!clocks.apply(0,a,{2600,2600}));wait(clocks,0);
    }
    configure(0,0,0,0,0);
    {GpuClocks clocks;clocks.start(fs::absolute(argv[1]).u8string(),ecmgui::exe_dir()+"/ecm_gui_v2_fake_clock_query.exe");
     configure(0,1,0,0,0);assert(clocks.query(0,a));auto s=wait(clocks,0);assert(!s.error&&s.smi_range&&s.supported==ClockRange({210,3105}));
     const int writes=count(0);assert(clocks.apply(0,a,{2600,2600}));s=wait(clocks,0);assert(!s.error&&count(0)==writes+1&&s.applied==ClockRange({2600,2600}));
     assert(clocks.query(1,b));s=wait(clocks,1);assert(s.error&&!s.supported.valid()); // N/A cannot become guessed limits.
     configure(0,0,0,0,0);}
    {GpuClocks clocks;clocks.start(fs::absolute(argv[2]).u8string());assert(clocks.query(0,a));const auto s=wait(clocks,0);assert(s.supported.valid()&&!s.can_apply&&!s.can_reset);}
    std::string error;assert(ui::load_languages(ecmgui::exe_dir()+"/localization/v2",error));
    FILE* log=nullptr;_wfopen_s(&log,(out/"app.log").c_str(),L"wb");assert(log);
    const int writes=count(0),resets=count(1);
    {
        UiState state;state.ui_language="en";App app(state,log);
        const auto toml=out/"ecm.toml";const std::string original="# retained comment\n[shared]\nworktodo = \"queue.txt\"\nb2_ratio = 1000\n[stage1]\ntarget = \"cpu\"\n[gui]\nNumWorkers = 1\n";
        write(toml,original);assert(app.load_config(toml.u8string()));app.autostart_pending=false;
        assert(app.save_gpu_clock_range(a,{2600,2600}));assert(!app.config.dirty());
        assert(read(toml).find("# retained comment")!=std::string::npos&&read(toml).find(a+":2600-2600")!=std::string::npos);
        assert(app.edit_business("shared","b2_ratio","2000"));assert(app.save_gpu_clock_range(b,{210,3000}));
        assert(app.config.dirty()&&app.config.get("shared","b2_ratio")=="2000");
        ecmgui::TomlFile disk;assert(disk.load(toml.u8string(),error));assert(disk.get("shared","b2_ratio")=="1000");
        assert(parse_clock_profiles(disk.get("gui","gpu_clock_ranges"),profiles)&&profiles.size()==2);
        assert(app.load_config(toml.u8string()));app.autostart_pending=false;assert(app.saved_clock_profiles.size()==2);
        auto changed=read(toml)+"# external edit\n";write(toml,changed);
        assert(!app.save_gpu_clock_range(a,{2500,2700})&&read(toml)==changed);
        assert(app.load_config(toml.u8string()));app.autostart_pending=false;
        app.gpu_clocks.start(fs::absolute(argv[1]).u8string());
        auto* context=ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;
        io.DisplaySize={1100,850};io.DeltaTime=1.f/60;io.Fonts->AddFontDefault();unsigned char* pixels=nullptr;int w=0,h=0;io.Fonts->GetTexDataAsRGBA32(&pixels,&w,&h);
        const auto frame=[&](){ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({1050,820});ImGui::Begin("GPU test",nullptr,ImGuiWindowFlags_NoSavedSettings);
            app.gpu_presentations.clear();ImGui::TextUnformatted("VRAM 123 / 16000 MiB | 45 C");app.draw_gpu_clock_control(0,a,false);ImGui::End();ImGui::Render();};
        const auto move=[&](ImVec2 p){io.AddMousePosEvent(p.x,p.y);frame();};
        const auto click=[&](ImVec2 p){move(p);io.AddMouseButtonEvent(0,true);frame();io.AddMouseButtonEvent(0,false);frame();};
        frame();frame();click(app.gpu_presentations[0].gear);wait(app.gpu_clocks,0);frame();frame();
        assert(app.gpu_presentations[0].popover&&app.clock_drafts[a].range==ClockRange({2600,2600}));
        const auto slider=app.gpu_presentations[0].range_slider;
        move({slider.low.x-3,slider.low.y});io.AddMouseButtonEvent(0,true);frame();
        move({slider.low.x-60,slider.low.y});io.AddMouseButtonEvent(0,false);frame();
        const auto edited=app.clock_drafts[a].range;assert(edited.low<2600&&edited.high==2600&&count(0)==writes&&count(1)==resets);
        click({950,750});frame();assert(!app.gpu_presentations[0].popover&&app.clock_drafts[a].range==edited);
        click(app.gpu_presentations[0].gear);wait(app.gpu_clocks,0);frame();frame();assert(app.gpu_presentations[0].popover&&app.clock_drafts[a].range==edited);
        click(app.gpu_presentations[0].automatic);assert(app.clock_drafts[a].automatic);
        click(app.gpu_presentations[0].save);assert(count(0)==writes&&count(1)==resets);
        assert(app.saved_clock_profiles[a]==edited&&app.saved_clock_automatic.count(a)&&read(toml).find(":auto")!=std::string::npos);
        click(app.gpu_presentations[0].apply);wait(app.gpu_clocks,0);frame();assert(count(0)==writes+1&&count(2)==0);
        click(app.gpu_presentations[0].reset);wait(app.gpu_clocks,0);frame();assert(count(1)==resets+1&&app.clock_drafts[a].range==edited);
        assert(app.load_config(toml.u8string()));app.autostart_pending=false;assert(app.saved_clock_profiles[a]==edited&&count(0)==writes+1);
        assert(app.clock_auto_policies.empty()); // Save + reload in the same session must not replay startup.
        {UiState fresh_state;fresh_state.ui_language="en";App fresh(fresh_state,log);
         assert(fresh.load_config(toml.u8string()));assert(fresh.clock_auto_policies[a].enabled&&fresh.clock_auto_policies[a].startup);
         fresh.clock_auto_policies[a].startup=false;assert(fresh.load_config(toml.u8string())&&!fresh.clock_auto_policies[a].startup);}
        ImGui::DestroyContext(context);
    }
    fclose(log);FreeLibrary(module);
    std::cout<<"GPU clocks: UUID isolation, async queries, bounds, unsupported APIs, permissions, explicit writes, draft popover, TOML persistence, startup/hourly and continuous 5-second detection passed\n";
}
