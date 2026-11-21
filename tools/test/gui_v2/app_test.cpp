#include "../../../src/ecm/application/gui_v2/app.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>
#include "imgui_internal.h"
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
static void generator_while_running(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"generator-running";fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml",queue=dir/"worktodo.txt";
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\nworker_count=2\n"
              "[stage1]\ntarget=\"cpu\"\n[gui]\nstage1_exe=\""+fixture.generic_u8string()+"\"\n"
              "[worker.1]\nautostart=false\nextra_args='--scenario v2-checkpoint-stop'\n[worker.2]\nautostart=false\n");
    write(queue,"[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n[Worker #2]\n");
    ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
    assert(app.start(1));
    const auto wait=[&](const auto& done){const auto deadline=GetTickCount64()+15000;
        while(!done()&&GetTickCount64()<deadline){app.tick();Sleep(10);}assert(done());};
    wait([&]{for(const auto& line:app.controller.workers()[0].raw)
        if(line.find("queue_owner: acquired")!=line.npos)return true;return false;});
    std::strcpy(app.gen_input,"ECM2=1,2,137,-1,100,1,0\n");
    app.gen_options.use_recommended=false;app.gen_options.curves_fixed=4;app.gen_worker=2;
    assert(app.preview_generator());assert(app.apply_generator());
    assert(app.controller.busy()&&read(queue).find("ECMSTAGE1=",read(queue).find("[Worker #2]"))!=std::string::npos);
    // A running target can be previewed; its ownership lock still protects writes.
    app.gen_worker=1;assert(app.preview_generator());const auto before=read(queue);
    assert(!app.apply_generator());assert(app.status.find("already owned")!=app.status.npos&&read(queue)==before);
    app.controller.request_stop(1);wait([&]{return !app.controller.busy();});
    assert(app.apply_generator());
    std::cout<<"{\"generator_while_running\":true,\"target_queue_lock_preserved\":true}\n";
}
// Run with a locking worker whose protocol-query API is deliberately disabled.
static void no_protocol_probe(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"no-protocol";fs::create_directories(dir);
    const auto probe=dir/"unexpected-probe.txt",queue=dir/"worktodo.txt",cfg=dir/"ecm.toml";
    assert(_putenv_s("ECM_TEST_PROTOCOL_PROBE",probe.u8string().c_str())==0);
    const auto setup=[&](const std::string& args,bool autostart=false,const fs::path& exe=fs::path{}){
        write(queue,"[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n");
        write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\n[stage1]\ntarget=\"cpu\"\n[gui]\nstage1_exe=\""+(exe.empty()?fixture:exe).generic_u8string()+"\"\nstage2_exe=\""+fixture.generic_u8string()+"\"\n[worker.1]\nautostart="+(autostart?"true":"false")+"\nextra_args='"+args+"'\n");
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
    {ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
     // A real worker owner blocks dispatch even without the query handshake.
     {ecm_queue::OwnerLease owner(queue.u8string(),1);assert(!app.start(1));}
     {ecm_queue::OwnerLease gui(queue.u8string(),1,".gui-control.lock");assert(!app.start(1));}
     assert(app.start(1));wait(app,[&]{return acquired(app);});
     ecmgui_v2::UiState second_state;ecmgui_v2::App second(second_state,stdout);second.initialize(cfg.u8string(),true);assert(!second.start(1));
     app.controller.request_stop(1);wait(app,[&]{return !app.controller.busy();});}
    setup("--scenario v2-checkpoint-stop",true);
    {ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);app.tick();wait(app,[&]{return acquired(app);});
     app.controller.request_stop(1);wait(app,[&]{return !app.controller.busy();});}
    setup("--scenario ok --lines 0");
    {ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(!app.controller.workers()[0].stage2);
     write(queue,"[Worker #1]\nECMSTAGE2=1,2,137,-1,\"ed_m137.save\",0,0,10\n");
     assert(app.start(1)&&app.controller.workers()[0].stage2);wait(app,[&]{return acquired(app)&&!app.controller.busy();});}
    setup("--scenario crash-once --lines 0 --marker \""+(dir/"crashed-once").generic_u8string()+"\"");
    {ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.start(1));
     wait(app,[&]{return app.controller.workers()[0].restarts()==1&&!app.controller.busy();});}
    for(const auto& exe:{dir/"missing.exe",dir/"invalid.exe"}){
        if(exe.filename()=="invalid.exe")write(exe,"not an executable");setup("",false,exe);
        ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
        assert(!app.start(1)&&app.status.find("CreateProcessW failed")!=app.status.npos);
    }
    assert(!fs::exists(probe));assert(_putenv_s("ECM_TEST_PROTOCOL_PROBE","")==0);
    std::cout<<"{\"no_protocol_probe\":true,\"manual_autostart_restart_stage_switch\":true,\"queue_locks_preserved\":true,\"spawn_errors_preserved\":true}\n";
}
static void task_watch_dispatch(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"task-watch";
    fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml",queue=dir/"worktodo.txt";
    const std::string header="[Worker #1]\n";
    const std::string task="ECMSTAGE1=1,2,127,-1,100,1\n";
    write(queue,header);
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\n[stage1]\ntarget=\"cpu\"\n[gui]\nstage1_exe=\""+
        fixture.generic_u8string()+"\"\n[worker.1]\nautostart=false\nextra_args='--scenario v2-checkpoint-stop'\n");
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);
    app.initialize(cfg.u8string(),true);
    const auto worker=[&]() -> const ecmgui::WorkerView& {return app.controller.workers().front();};
    const auto running=[&]{return worker().running();};
    const auto stopped=[&]{return !app.controller.busy()&&!worker().stop_requested;};
    const auto wait=[&](const auto& done){
        const auto deadline=GetTickCount64()+10000;
        while(!done()&&GetTickCount64()<deadline){app.tick();Sleep(10);}
        assert(done());
    };
    const auto owner_acquisitions=[&]{
        size_t count=0;
        for(const auto& line:worker().raw)
            if(line.find("queue_owner: acquired")!=line.npos)++count;
        return count;
    };
    const auto base=GetTickCount64();
    constexpr unsigned long long interval=5*60*1000;
    assert(!app.task_watch_enabled());
    app.set_task_watch(true);
    app.poll_task_watch(base);
    assert(!running());

    // A changed queue triggers immediately, but its owner can block dispatch.
    {
        ecm_queue::OwnerLease owner(queue.u8string(),1);
        write(queue,header+task);
        app.poll_task_watch(base+1000);
        assert(!running()&&app.status.find("queue")!=app.status.npos);
    }
    app.poll_task_watch(base+2000);
    assert(!running());
    write(queue,header+task+"\n");
    app.poll_task_watch(base+3000);
    assert(running());
    wait([&]{return owner_acquisitions()==1;});

    app.stop_worker(1);
    wait(stopped);
    app.poll_task_watch(base+interval);
    assert(app.task_watch_enabled()&&!running());
    assert(app.start(1));
    wait([&]{return owner_acquisitions()==2;});
    app.stop_all();
    wait(stopped);
    app.poll_task_watch(base+2*interval);
    assert(!app.task_watch_enabled()&&!running());

    // Without a second edit, the five-minute scan retries a previously busy queue.
    app.set_task_watch(true);
    app.status.clear();
    {
        ecm_queue::OwnerLease owner(queue.u8string(),1);
        app.poll_task_watch(base+3*interval);
        assert(!running()&&app.status.find("queue")!=app.status.npos);
    }
    app.poll_task_watch(base+4*interval);
    assert(running());
    wait([&]{return owner_acquisitions()==3;});
    app.stop_all();
    wait(stopped);
    std::cout<<"{\"task_watch\":true,\"queue_change\":true,\"periodic_scan\":true,\"queue_owner_respected\":true,\"safe_stop_respected\":true}\n";
}
static void pause_schedule_rules(){
    ecmgui_v2::PauseSchedule schedule;
    ecmgui_v2::PauseScheduleError error;
    assert(schedule.parse({"[1,3-4], 1-5, 08:30-12:00|13:00-17:30", "2, Mon, 08:30-09:30", "#5, 7, 22:00-02:00"},error));
    std::vector<ecmgui_v2::PauseWorker> workers{{1,true},{2,true},{3,true},{4,true},{5,false},{6,true}};
    assert(schedule.paused_workers(0,509,workers).empty());
    assert((schedule.paused_workers(0,510,workers)==std::set<int>{1,2,3,4,6}));
    workers[1].active=false;
    workers[4].active=true;
    assert((schedule.paused_workers(0,511,workers)==std::set<int>{1,2,3,4,6}));
    assert((schedule.paused_workers(0,570,workers)==std::set<int>{1,3,4}));
    assert(schedule.paused_workers(0,720,workers).empty());
    assert((schedule.paused_workers(0,780,workers)==std::set<int>{1,3,4}));
    assert((schedule.paused_workers(6,22*60,workers)==std::set<int>{5}));
    assert((schedule.paused_workers(0,119,workers)==std::set<int>{5}));
    assert(schedule.paused_workers(0,120,workers).empty());
    assert(schedule.parse({",, 03:00-04:00"},error));
    assert(schedule.paused_workers(6,180,workers).size()==workers.size());
    assert(schedule.parse({"*,Tue|4,*"},error));
    assert(schedule.paused_workers(3,0,workers).size()==workers.size());
    assert(!schedule.parse({"#2","[],*,*"},error)&&error.item==2&&error.group==ecmgui_v2::PauseScheduleError::Worker);
    assert(!schedule.parse({"1,8,*"},error)&&error.group==ecmgui_v2::PauseScheduleError::Weekday);
    assert(!schedule.parse({"1,1,08:00-08:00"},error)&&error.group==ecmgui_v2::PauseScheduleError::Time);
    assert(schedule.parse({"foo bar, [1, 3-4], 1, 08 : 30 - 09:00"},error));
    assert(!schedule.needs_applications(0,509)&&schedule.needs_applications(0,510));
    assert(schedule.paused_workers(0,510,workers,{LR"(C:\other.exe)"}).empty());
    for (const std::wstring& path : {LR"(c:\foobar.exe)", LR"(C:\FOOBAR\name.exe)", LR"(C:\myFoo Barprog.exe)"})
        assert((schedule.paused_workers(0,510,workers,{path})==std::set<int>{1,3,4}));
    assert(schedule.paused_workers(0,540,workers,{LR"(C:\FOOBAR\name.exe)"}).empty());
    assert(schedule.paused_workers(0,510,workers).empty());
    for (const std::string row : {"*,1,*,*", "*, 1, *, *   "}) {
        assert(schedule.parse({row},error)&&!schedule.needs_applications(0,510));
        assert((schedule.paused_workers(0,510,workers)==std::set<int>{6}));
    }
}
static void scheduled_worker_dispatch(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"pause-schedule";
    fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml",queue=dir/"worktodo.txt";
    write(queue,"[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n[Worker #2]\nECMSTAGE1=1,2,137,-1,100,1\n");
    const auto args="\nautostart=false\nextra_args='--scenario v2-checkpoint-stop'\n";
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\nworker_count=2\n[stage1]\ntarget=\"cpu\"\n[gui]\nstage1_exe=\""+
        fixture.generic_u8string()+"\"\npause_schedule = [\n  '#1, 1, 08:30-09:00', # preserve rule note\n] # preserve schedule note\n[worker.1]"+args+"[worker.2]"+args);

    // Computation readers accept the GUI array and ignore its scheduling semantics.
    for(const auto mode:{ecm_config::TomlReadMode::Computation,ecm_config::TomlReadMode::Stage1,ecm_config::TomlReadMode::Stage2}){
        std::vector<ecm_config::TomlLine> lines;
        assert(ecm_config::read_toml(cfg.u8string(),lines,mode));
    }
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);
    app.initialize(cfg.u8string(),true);
    assert(app.ready);
    SYSTEMTIME time{};time.wDayOfWeek=1;time.wHour=8;time.wMinute=29;
    const auto worker=[&](int i) -> const ecmgui::WorkerView& {return app.controller.workers().at(i-1);};
    const auto wait=[&](const auto& done){
        const auto deadline=GetTickCount64()+10000;
        while(!done()&&GetTickCount64()<deadline){app.tick(time);Sleep(10);}
        assert(done());
    };
    const auto acquired=[&](int i,int count){
        for(const auto& line:worker(i).raw)if(line.find("queue_owner: acquired")!=line.npos)--count;
        return count==0;
    };
    app.tick(time);
    assert(app.start(1));
    wait([&]{return acquired(1,1);});
    time.wMinute=30;
    app.set_task_watch(true);
    app.tick(time);
    wait([&]{return !worker(1).running()&&!worker(1).stop_requested&&acquired(2,1);});
    assert(!app.start(1)&&worker(2).running()&&app.task_watch_enabled());
    app.stop_worker(2);
    wait([&]{return !worker(2).running()&&!worker(2).stop_requested;});
    time.wHour=9;time.wMinute=0;
    app.tick(time);
    wait([&]{return acquired(1,2);});
    assert(!worker(2).running());
    const auto before=read(cfg);
    assert(!app.save_pause_schedule({"#2,8,*"})&&read(cfg)==before);
    assert(app.save_pause_schedule({"#1, Mon, 09:00-09:30"}));
    const auto saved=read(cfg);
    assert(saved.find("# preserve rule note")!=saved.npos&&saved.find("# preserve schedule note")!=saved.npos);
    app.tick(time);
    wait([&]{return !worker(1).running()&&!worker(1).stop_requested;});
    app.stop_all();
    time.wMinute=30;
    app.tick(time);
    assert(!worker(1).running()&&!worker(2).running()&&!app.task_watch_enabled());
    assert(app.load_config(cfg.u8string()));
    assert((ecm_config::toml_string_array(app.config.get("gui","pause_schedule"))==std::vector<std::string>{"#1, Mon, 09:00-09:30"}));

    // Only the directory name matches. Keep the clock in the same minute for both polls.
    const std::string rule="ecm pause foobar, #1, *, *";
    assert(app.save_pause_schedule({rule}));
    app.tick(time);
    assert(app.start(1));
    wait([&]{return acquired(1,1);});
    const auto program=dir/"ECM Pause FooBar"/"name.exe";
    fs::create_directories(program.parent_path());
    fs::copy_file(fixture,program);
    auto command=L"\""+program.wstring()+L"\" --scenario hang --lines 0";
    STARTUPINFOW startup{};startup.cb=sizeof(startup);
    PROCESS_INFORMATION process{};
    assert(CreateProcessW(program.c_str(),command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,
                          nullptr,nullptr,&startup,&process));
    CloseHandle(process.hThread);
    wait([&]{return !worker(1).running()&&!worker(1).stop_requested;});
    assert(!app.start(1)&&!app.task_watch_enabled());
    assert(TerminateProcess(process.hProcess,0));
    assert(WaitForSingleObject(process.hProcess,5000)==WAIT_OBJECT_0);
    CloseHandle(process.hProcess);
    wait([&]{return acquired(1,2);});
    app.stop_all();
    wait([&]{return !app.controller.busy()&&!worker(1).stop_requested;});
    std::cout<<"{\"pause_schedule\":true,\"task_watch_obeys_pause\":true,\"only_scheduled_worker_resumed\":true,\"toml_round_trip\":true,\"application_path_pause_and_resume_same_minute\":true}\n";
}
static void worker_breaker_recovery(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"breaker-recovery";fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml",queue=dir/"worktodo.txt",fixed=dir/"fault-fixed";
    const std::string queued="[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n";
    write(queue,queued);
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\n[gui]\nstage1_exe=\""+fixture.generic_u8string()+
        "\"\n[worker.1]\nautostart=true\nextra_args='--scenario v2-recover-after-breaker --marker \""+
        fixed.generic_u8string()+"\"'\n");
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);
    const auto worker=[&]() -> const ecmgui::WorkerView& {return app.controller.workers().at(0);};
    const auto wait=[&](ecmgui::WorkerRunState expected){
        const auto deadline=GetTickCount64()+20000;
        do {app.tick();if(worker().state()==expected)return;Sleep(10);}while(GetTickCount64()<deadline);
        assert(false);
    };
    wait(ecmgui::WorkerRunState::Error);
    assert(worker().breaker_tripped()&&worker().restarts()==2&&!app.controller.busy());
    write(fixed,"ready");
    app.set_task_watch(true);
    app.poll_task_watch(GetTickCount64());
    assert(worker().breaker_tripped()&&!app.controller.busy());
    app.set_task_watch(false);
    assert(!app.start(1,ecmgui::WorkerStart::Automatic));
    {ecm_queue::OwnerLease owner(queue.u8string(),1);assert(!app.start(1));}
    {ecm_queue::OwnerLease control(queue.u8string(),1,".gui-control.lock");app.start_all();}
    assert(worker().breaker_tripped()&&worker().restarts()==2&&!app.controller.busy());
    assert(app.start(1));
    wait(ecmgui::WorkerRunState::QueueEmpty);
    assert(!worker().breaker_tripped()&&worker().restarts()==0);

    fs::remove(fixed);
    app.start_all();
    wait(ecmgui::WorkerRunState::Error);
    assert(worker().breaker_tripped()&&worker().restarts()==2);
    write(fixed,"ready");
    app.start_all();
    wait(ecmgui::WorkerRunState::QueueEmpty);
    assert(!worker().breaker_tripped()&&worker().restarts()==0&&read(queue)==queued);
    std::cout<<"{\"worker_breaker_manual_recovery\":true,\"start_all_recovery\":true,\"retry_budget_reset\":true,\"queue_locks_preserved\":true}\n";
}
static void stage2_raw_times(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"stage2-raw";fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml";
    write(dir/"worktodo.txt","[Worker #1]\nECMSTAGE2=1,2,137,-1,\"fixture.save\",0,0,4\n");
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\n[gui]\nstage2_exe=\""+fixture.generic_u8string()+"\"\n[worker.1]\nautostart=false\nextra_args=\"--scenario v2-stage2-raw-times\"\n");
    const auto drain=[](auto& controller,const auto& tick){
        const auto deadline=GetTickCount64()+5000;
        while(controller.busy()&&GetTickCount64()<deadline){tick();Sleep(10);}assert(!controller.busy());
    };
    {ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.start(1));
     drain(app.controller,[&]{app.tick();});const auto& worker=app.controller.workers()[0];
     assert(worker.hist_s_per_curve==(std::vector<float>{10,30,5}));
     assert(worker.progress.s_per_curve==5&&worker.progress.eta_s==5);
     const auto shown=ecmgui_v2::ui::worker_progress(worker,false);assert(shown.speed=="5.00"&&shown.eta=="5 s");}
    // The original GUI retains its existing presentation policy.
    {ecmgui::TomlFile config;std::string error;assert(config.load(cfg.u8string(),error));ecmgui::WorkerController legacy(config);
     ecmgui::WorkerSettings settings;settings.ini_path=cfg.u8string();settings.stage2_exe=fixture.u8string();settings.num_workers=1;
     legacy.configure(settings);legacy.suppress_autostart();assert(legacy.start(1));drain(legacy,[&]{legacy.tick();});
     assert(legacy.workers()[0].hist_s_per_curve==(std::vector<float>{10,20,15}));}
    std::cout<<"{\"stage2_raw_history\":true,\"raw_display_and_eta\":true,\"legacy_average_preserved\":true}\n";
}
static void stage2_paired_views(const fs::path& root,const fs::path& fixture){
    const auto dir=root/"stage2-paired";fs::create_directories(dir);
    const auto cfg=dir/"ecm.toml";
    write(dir/"worktodo.txt","[Worker #1]\nECMSTAGE2=1,2,137,-1,\"fixture.save\",0,0,5\n");
    write(cfg,"config_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\n[gui]\nstage2_exe=\""+fixture.generic_u8string()+
        "\"\n[worker.1]\nautostart=false\nextra_args=\"--scenario v2-stage2-paired\"\n");
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.start(1));
    const auto worker=[&]() -> const ecmgui::WorkerView& {return app.controller.workers()[0];};
    const auto wait=[&](const auto& predicate){
        const auto deadline=GetTickCount64()+5000;
        do{app.tick();if(predicate())return;Sleep(10);}while(GetTickCount64()<deadline);
        assert(false);
    };
    wait([&]{return worker().stage2_status.curves.size()==3;});
    const auto& status=worker().stage2_status;
    const auto shown=status.visible_curves();
    assert(shown.size()==2&&shown[0]->current==2&&shown[1]->current==3);
    assert(status.done==1&&shown[0]->batch==2&&shown[0]->bits==318&&shown[1]->batch==0);
    assert(shown[0]->phase=="Giant points"&&shown[1]->phase=="Generate and normalize baby points");
    assert(ecmgui_v2::ui::worker_progress(worker(),false).eta=="--");
    auto* context=ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={1500,850};io.DeltaTime=1.f/60;
    ecmgui_v2::ui::Settings style;ecmgui_v2::ui::Fonts fonts;
    auto* font=io.Fonts->AddFontDefault();fonts.body=fonts.mono=font;io.Fonts->Build();
    for(int i=0;i<2;++i){
        ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({1500,850});
        ImGui::Begin("paired-workspace");app.draw_page(style,fonts,0);ImGui::End();ImGui::Render();
    }
    const auto& summary=app.worker_summary_texts.at(1);
    assert(summary.find("Curve #")==summary.npos);
    assert(summary.find("Edwards  B1=20  B2=1000  sigma=17  D=210  P=24")!=summary.npos);
    assert(summary.find("Edwards  B1=20  B2=1100  sigma=18  D=330  P=40")!=summary.npos);
    assert(summary.find("sigma=19")==summary.npos);
    assert(ecmgui_v2::ui::stage2_curve_label(*shown[0],false)=="Curve #2 / 5");
    assert(summary.find("Phase:")==summary.npos&&std::count(summary.begin(),summary.end(),'\n')==2);
    ImGui::DestroyContext(context);
    wait([&]{return status.done==4&&status.workers==1;});
    assert(status.visible_curves().size()==1&&status.visible_curves().front()->current==5&&status.progress().eta_s<0);
    wait([&]{return !app.controller.busy();});
    assert(worker().hist_s_per_curve==(std::vector<float>{8,12,5,3}));
    assert(status.done==5&&status.progress().eta_s==0&&status.curves.empty());
    std::cout<<"{\"stage2_parallel_views\":true,\"pipeline_hidden\":true,\"serial_fallback\":true,\"all_curve_samples\":true}\n";
}
static void stage1_ntt_progress(const fs::path& root,const fs::path& fixture) {
    const auto dir=root/"stage1-ntt";fs::create_directories(dir);
    write(dir/"worktodo.txt","[Worker #1]\nECM2=1,2,15919,-1,44000000,4400000000,1000,\"1782929\"\n");
    const auto cfg=dir/"ecm.toml";
    write(cfg,"config_version=4\n[shared]\nworktodo_file='worktodo.txt'\n[gui]\nstage1_exe='"+fixture.generic_u8string()+
        "'\n[worker.1]\nautostart=false\nextra_args='--scenario v2-ntt-progress'\n"
        "handoff_workers='2[:15910], 3[15910:]'\n");
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.start(1));
    const auto deadline=GetTickCount64()+5000;
    while(!app.controller.workers()[0].stage1_handoff.local_status.worker&&GetTickCount64()<deadline){app.tick();Sleep(10);}
    const auto& worker=app.controller.workers()[0];
    assert(worker.stage1_handoff.local_status.bits==15890);
    assert(ecmgui_v2::ui::stage1_handoff(worker.stage1_handoff,worker.stage1_execution,false)==
           "Stage2 → ECM-ELF (Worker2)");
    const auto shown=ecmgui_v2::ui::worker_progress(worker,false);
    assert(shown.label=="14.4%"&&shown.speed=="1.00"&&shown.eta=="856 s");
    const auto batch=ecmgui_v2::ui::stage1_ntt_batch(worker.stage1_ntt);
    assert(batch.visible&&batch.fraction==.5f&&batch.label=="curve 129–160  50.0%");
    app.stop_worker(1);
    const auto stop_deadline=GetTickCount64()+5000;
    while(app.controller.busy()&&GetTickCount64()<stop_deadline){app.tick();Sleep(10);}
    assert(!app.controller.busy());
    std::cout<<"{\"stage1_ntt_total\":true,\"resumed_batch\":true,\"speed_eta\":true}\n";
}
static void stage1_details(const fs::path& root) {
    {
        const auto dir=root/"worker-device-labels";fs::create_directories(dir);
        const auto cfg=dir/"ecm.toml";
        write(dir/"worktodo.txt","[Worker #1]\nECMSTAGE2=1,2,127,-1,\"e_m127_100.save\",0,0,1\n"
            "[Worker #2]\nECMSTAGE2=1,2,127,-1,\"e_m127_100.save\",0,0,1\n"
            "[Worker #3]\nECMSTAGE2=1,2,127,-1,\"e_m127_100.save\",0,0,1\n"
            "[Worker #4]\nECMSTAGE1=1,2,127,-1,100,1\n");
        const std::string content="config_version=4\n[shared]\nworker_count=4\nworktodo_file=\"worktodo.txt\"\n"
            "[stage1]\ntarget=\"cpu\"\nstage1_gpu_device=7\n[stage2]\ndevice=\"cpu\"\nstage2_gpu_device=2\n"
            "[worker.1]\nname=\"Alpha\"\nautostart=false\n[worker.2]\nname=\"Beta\"\ndevice=\"gpu\"\nautostart=false\n"
            "[worker.3]\nname=\"Gamma\"\ndevice=\"cpu\"\nautostart=false\n[worker.4]\nname=\"Delta\"\nautostart=false\n";
        write(cfg,content);ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);
        assert(app.load_config(cfg.u8string()));
        const auto label=[&](size_t i){return ecmgui_v2::ui::worker_device_name(app.controller.workers()[i]);};
        assert(label(0)=="CPU: Alpha"&&label(1)=="GPU 2: Beta"&&label(2)=="CPU: Gamma"&&label(3)=="CPU: Delta");
        auto changed=content;const auto at=changed.find("[stage2]\ndevice=\"cpu\"");
        changed.replace(at,std::strlen("[stage2]\ndevice=\"cpu\""),"[stage2]\ndevice=\"gpu\"");write(cfg,changed);
        assert(app.load_config(cfg.u8string()));
        assert(label(0)=="GPU 2: Alpha"&&label(2)=="CPU: Gamma"&&label(3)=="CPU: Delta");
    }
    {
        std::string error;
        assert(ecmgui_v2::ui::load_languages(ecmgui::exe_dir()+"/localization/v2",error,"chineseSimplified.xml"));
        ecmgui::WorkerView worker;worker.task="ECM2=1,2,15919,-1,44000000,4400000000,1000,\"1782929\"";
        assert(ecmgui_v2::ui::worker_task(worker,true)=="M15919 / 已知因子  curves = 1000");
        worker.task="ECMSTAGE1=1,2,5351,-1,\"e_m5351_260e6.save\",0,0,300";
        assert(ecmgui_v2::ui::worker_task(worker,false)=="M5351  curves = 300");
        worker.task="ECMSTAGE2=1,2,5351,-1,\"e_m5351_260e6.save\",0,0,300";
        assert(!ecmgui::task_display(worker.task).has_factors);
        worker.task="ECMSTAGE2N=N/A,\"ed.save\",0,0,300,\"1782929\"";
        assert(ecmgui_v2::ui::worker_task(worker,false)=="N (Stage1 save) / Known factors  curves = 300");
        worker.task="ECM=1,2,5351,-1,44000000";
        assert(ecmgui::task_display(worker.task).curves==100);
        worker.task="ECM=1,2,5351,-1,44000000,\"7\"";
        assert(ecmgui::task_display(worker.task).has_factors&&ecmgui::task_display(worker.task).curves==100);
        worker.task="ECM=1,2,5351,-1,44000000,0,\"7\"";
        assert(ecmgui::task_display(worker.task).curves==100);
        worker.task="ECMSTAGE2N=\"ed.save\",\"7\"";
        assert(ecmgui::task_display(worker.task).has_factors&&ecmgui::task_display(worker.task).curves==0);
    }
    const auto execution=ecmgui::parse_line("Stage1 execution: curve=ed param=e backend=ntt ntt_type=weighted "
        "B1=1024 iter_total=1500 ntt_bits=127 ntt_length=32 ntt_bpw=24 naf=0 ntt_shape=1:3:4:5:1:2,3:3,2 "
        "ntt_forward_pass_bits=2+3").stage1;
    const auto compact=ecmgui_v2::ui::stage1_execution(execution,false);
    {ecmgui::WorkerView worker;
     worker.progress=ecmgui::parse_line("GPU: [bar] 50.0%  500, +100 bits (~1.20 s/curve) elapsed 1s remaining 1s iter_per_s=1000").progress;
     assert(ecmgui_v2::ui::worker_progress(worker,false).iterations=="1000");
     worker.stage2=true;assert(ecmgui_v2::ui::worker_progress(worker,false).iterations=="--");}
    assert(compact.find("CUDA/NTT 127 bits | B1=1024 | iter=1500 | NTT: 32 1:3:4:5:1:2,3:3,2 NAF=0")!=compact.npos &&
           compact.find("bpw=")==compact.npos && compact.find('\n')==compact.npos &&
           compact.find("ntt_forward_pass_bits")==compact.npos);
    assert(ecmgui_v2::ui::stage1_execution(execution,false,true).find("ntt_forward_pass_bits=2+3")!=compact.npos);
    const auto memory = ecmgui::parse_line("Stage1 execution: curve=ed param=e backend=ntt ntt_type=weighted "
        "ntt_length=1024 ntt_memory_per_curve_bytes=41867520 ntt_memory_config_mib=6144 "
        "ntt_memory_device_bytes=8388608000 ntt_naf_memory_mb=4096 max_curves_in_memory=98").stage1;
    const auto estimate = ecmgui_v2::ui::stage1_execution(memory,false,true);
    assert(estimate.find("39.93 MiB/curve | Limit (config / device) 6144 MiB / 8000 MiB | Max 98 curves | NAF 4096 MB") != estimate.npos);
    assert(estimate.find("ntt_memory_per_curve_bytes=") == estimate.npos);
    const auto zh_estimate = ecmgui_v2::ui::stage1_execution(memory,true);
    assert(zh_estimate.find("MiB")==zh_estimate.npos && zh_estimate.find("最多")==zh_estimate.npos);
    const auto binary=ecmgui::parse_line("Stage1 execution: curve=ed param=e backend=cgbn "
        "kernel_weight=1024 tpi=8 naf=0 algorithm=binary").stage1;
    const auto cgbn=ecmgui_v2::ui::stage1_execution(binary,false);
    assert(cgbn.find("CGBN＜1024, 8＞ | Binary")!=cgbn.npos && cgbn.find("x-only")==cgbn.npos);
    const auto naf=ecmgui::parse_line("Stage1 execution: curve=ed param=e backend=cgbn "
        "kernel_weight=2048 tpi=16 naf=5 algorithm=naf").stage1;
    assert(ecmgui_v2::ui::stage1_execution(naf,false).find("CGBN＜2048, 16＞ | NAF window=5")!=std::string::npos);
    const auto prac=ecmgui::parse_line("Stage1 execution: curve=mont param=m0 backend=cgbn "
        "kernel_weight=1024 tpi=8 chain=prac prac=3").stage1;
    assert(ecmgui_v2::ui::stage1_execution(prac,false).find("CGBN＜1024, 8＞ | PRAC level=3")!=std::string::npos);
    const auto mont=ecmgui::parse_line("Stage1 execution: curve=mont param=m0 backend=ntt naf=0 chain=ladder").stage1;
    const auto ladder=ecmgui_v2::ui::stage1_execution(mont,false,true);
    assert(ladder.find("chain=ladder")!=ladder.npos && ladder.find("x-only")==ladder.npos &&
           ladder.find("NAF=")==ladder.npos);
    const auto path=root/"stage1-details.toml";
    write(path,"config_version=4\n[shared]\nworker_count=2\n[stage1]\nhandoff_mode='off'\n"
        "[worker.2]\nhandoff_mode='prime95'\nhandoff_unsupported='ecm-elf'\n"
        "handoff_workers='auto'\nhandoff_worktodo_file='local-queue.txt'\n"
        "prime95_worktodo_file='prime95.txt'\n");
    ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);
    assert(app.load_config(path.u8string()));
    const auto& workers=app.controller.workers();
    assert(ecmgui_v2::ui::stage1_handoff(workers.at(0).stage1_handoff,{},false).find("Off")!=std::string::npos);
    const auto& plan=workers.at(1).stage1_handoff;
    assert(plan.prime95 && plan.local_fallback && !plan.local && plan.local_workers=="auto");
    const auto details=ecmgui_v2::ui::stage1_handoff(plan,{},false,true);
    assert(details.find("compatible saved points")==details.npos &&
           details.find("local-queue.txt")!=details.npos && details.find("prime95.txt")!=details.npos);
    ecmgui::Stage1HandoffPlan direct;
    direct.local=true;
    direct.local_workers="2";
    assert(ecmgui_v2::ui::stage1_handoff(direct,prac,false)=="Stage2 → ECM-ELF (Worker2)");
    const auto sliced_config=root/"stage1-sliced.toml";
    write(root/"sliced.txt","[Worker #1]\nECM2=1,2,4095,-1,1000,2000,1\n"
         "[Worker #2]\nECM2=1,2,4096,-1,1000,2000,1\n"
         "[Worker #3]\nECM2=1,2,4096,-1,1000,2000,1,\"3\"\n"
         "[Worker #4]\nECM2=1,2,4096,-1,1000,2000,1\n");
    write(sliced_config,"config_version=4\n[shared]\nworker_count=5\nworktodo_file='sliced.txt'\n"
        "[stage1]\nhandoff_workers='2[:4096], 3[4096:]'\n"
        "[worker.4]\nhandoff_workers='2[:4096]'\n[worker.5]\nhandoff_workers='2[], 3[:]'\n");
    ecmgui_v2::UiState sliced_state;
    ecmgui_v2::App sliced_app(sliced_state,stdout);
    assert(sliced_app.load_config(sliced_config.u8string()));
    const auto target=[&](std::size_t i) {
        return ecmgui_v2::ui::stage1_handoff(sliced_app.controller.workers().at(i).stage1_handoff,{},false);
    };
    assert(target(0)=="Stage2 → ECM-ELF (Worker2)");
    assert(target(1)=="Stage2 → ECM-ELF (Worker3)");
    assert(target(2).find("Waiting for cofactor width")!=std::string::npos);
    assert(target(3).find("No worker matches")!=std::string::npos);
    assert(target(4)=="Stage2 → ECM-ELF (Worker2, 3)");
    const auto fallback=ecmgui_v2::ui::stage1_handoff(plan,execution,false,true);
    assert(fallback.find("local-queue.txt")!=fallback.npos && fallback.find("prime95.txt")==fallback.npos);
    const auto param3=ecmgui::parse_line("Stage1 execution: curve=mont param=m3 backend=cgbn").stage1;
    const auto prime95=ecmgui_v2::ui::stage1_handoff(plan,param3,false,true);
    assert(prime95.find("prime95.txt")!=prime95.npos && prime95.find("local-queue.txt")!=prime95.npos);
    const auto param2=ecmgui::parse_line("Stage1 execution: curve=mont param=m2 backend=cgbn").stage1;
    assert(ecmgui_v2::ui::stage1_handoff(direct,param2,false)=="Stage2 → ECM-ELF (Worker2)");
    const auto fallback_for_param2=ecmgui_v2::ui::stage1_handoff(plan,param2,false,true);
    assert(fallback_for_param2.find("local-queue.txt")!=fallback_for_param2.npos &&
           fallback_for_param2.find("prime95.txt")==fallback_for_param2.npos);
    auto keep_plan=plan;
    keep_plan.local_fallback=false;
    const auto keep=ecmgui_v2::ui::stage1_handoff(keep_plan,param2,false,true);
    assert(keep.find("incompatible")!=keep.npos && keep.find("prime95.txt")==keep.npos &&
           keep.find("local-queue.txt")==keep.npos);
}
static void workflow_errors(const fs::path& root){
    const auto path=root/"workflow-error.toml";
    write(path,"config_version=4\n[shared]\nworker_count=\"invalid\"\n");
    ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(path.u8string(),true);
    assert(!app.ready&&app.workflow_load_error.find("worker_count")!=std::string::npos);
    assert(app.workflow_load_error.find(":3")!=std::string::npos&&std::string(app.config_input)==path.u8string());
    const auto error=app.workflow_load_error;app.status="unrelated UI notification";
    assert(app.workflow_load_error==error); // The Workers view keeps the diagnostic.
    write(path,"config_version=4\n[shared]\nworker_count=2\n[worker.1]\nautostart=false\ntune_file=\"gpu0.toml\"\n"
        "[worker.2]\nautostart=false\ntune_file=\"gpu1.toml\"\n");
    assert(app.load_config(path.u8string())&&app.ready&&app.controller.workers().size()==2&&app.workflow_load_error.empty());
    write(path,read(path)+"[stage1]\nfuture_stage1=true\ncurve=\"future\"\n[stage2]\n"
        "giant_backend=\"future\"\nfuture_stage2=[\"future\"]\n");
    assert(app.load_config(path.u8string())&&app.ready&&app.controller.workers().size()==2);
    assert(app.edit_business("stage2","b2_ratio","40")&&app.save_business());
    assert(read(path).find("giant_backend=\"future\"")!=std::string::npos);
    write(path,"config_version=4\n[shared]\nworker_count=\"invalid\"\n");
    assert(!app.load_config(path.u8string())&&app.ready&&app.controller.workers().size()==2&&!app.workflow_load_error.empty());
    write(path,"config_version=4\n[shared]\nworker_count=65\n");
    assert(!app.load_config(path.u8string())&&app.controller.workers().size()==2&&app.workflow_worker_count==2);
    write(path,"config_version=4\n[shared]\nworker_count=3\n");
    assert(app.load_config(path.u8string())&&app.controller.workers().size()==3&&app.workflow_worker_count==3);
    write(path,"config_version=3\n[shared]\nworker_count=4\n");
    assert(!app.load_config(path.u8string())&&app.controller.workers().size()==3);
    write(path,"config_version=4\n");
    assert(app.load_config(path.u8string())&&app.controller.workers().size()==1);
}
static void result_table_contract(const fs::path& root){
    const auto row=[](int exponent,int b1,int day,bool factored=false){
        return std::string("{\"status\":\"")+(factored?"F":"NF")+"\",\"exponent\":"+std::to_string(exponent)+
            ",\"worktype\":\"ECM\",\"b1\":"+std::to_string(b1)+",\"b2\":"+std::to_string(b1)+
            ",\"curves\":1,\"timestamp\":\"2026-11-0"+std::to_string(day)+" 00:00:00\","
            "\"program\":{\"name\":\"ECM-ELF\",\"version\":\"fixture\"},\"checksum\":{\"version\":1,\"checksum\":\"1234ABCD\"}"+
            (factored?",\"stage\":1,\"factors\":[\"233\"]":"")+"}\n";
    };
    const auto a=root/"sort-first.txt",b=root/"sort-second.txt";
    const auto first=row(100,100,2)+row(10,10,3,true)+row(2,2,1);
    const auto second=row(20,20,4);
    write(a,first);write(b,second);
    ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);
    const auto workflow=root/"result-workflow.toml";
    write(workflow,"config_version = 4\n[shared]\nworktodo_file = \"worktodo.txt\"\nworker_count = 1\n");
    assert(app.load_config(workflow.u8string()));
    app.results.set_paths({a.u8string(),b.u8string()});assert(app.results.refresh());
    auto* context=ImGui::CreateContext();auto& io=ImGui::GetIO();
    io.IniFilename=nullptr;io.DisplaySize={1600,900};io.DeltaTime=1.f/60;
    io.Fonts->AddFontDefault();unsigned char* pixels;int width,height;io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
    static std::string clipboard;
    static bool allow_clipboard=false;
    auto& platform=ImGui::GetPlatformIO();
    platform.Platform_SetClipboardTextFn=[](ImGuiContext*,const char* value){if(allow_clipboard)clipboard=value;};
    platform.Platform_GetClipboardTextFn=[](ImGuiContext*){return clipboard.c_str();};
    ecmgui_v2::ui::Fonts fonts;ecmgui_v2::ui::Settings style;
    ImGuiID table_id=0;
    const auto draw_frame=[&](ecmgui_v2::App& view){
        ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({1500,800});
        ImGui::Begin("result-sort-contract",nullptr,ImGuiWindowFlags_NoSavedSettings);
        table_id=ImGui::GetID("results");view.draw_page(style,fonts,1);
        ImGui::End();ImGui::Render();
    };
    const auto frame=[&]{draw_frame(app);};
    const auto click_at=[&](ImVec2 pos){
        io.AddMousePosEvent(pos.x,pos.y);frame();
        io.AddMouseButtonEvent(0,true);frame();io.AddMouseButtonEvent(0,false);frame();frame();
    };
    const auto click=[&](int column){
        const auto* table=ImGui::TableFindByID(table_id);assert(table);
        click_at({(table->Columns[column].MinX+table->Columns[column].MaxX)*.5f,table->OuterRect.Min.y+9});
    };
    const auto has_color=[](ImU32 color){
        const auto* draw=ImGui::GetDrawData();
        for(int i=0;i<draw->CmdListsCount;++i)
            for(const auto& vertex:draw->CmdLists[i]->VtxBuffer)if(vertex.col==color)return true;
        return false;
    };
    ecmgui_v2::ui::apply_theme(style);frame();frame();
    assert((app.result_indices==std::vector<size_t>{3,1,0,2}));
    const auto selected_id=app.results.rows()[1].id;app.selected_results.insert(selected_id);
    click(0);assert(app.result_sort_column==-1);
    click(3);assert((app.result_indices==std::vector<size_t>{2,1,3,0}));
    click(3);assert((app.result_indices==std::vector<size_t>{0,3,1,2}));
    write(b,second+row(50,50,5));assert(app.results.refresh());app.result_view_dirty=true;frame();
    assert((app.result_indices==std::vector<size_t>{0,4,3,1,2}));
    click(3);assert(app.result_sort_column==-1);
    assert((app.result_indices==std::vector<size_t>{4,3,1,0,2}));
    assert(app.selected_results.size()==1&&app.selected_results.count(selected_id));
    for(bool light:{false,true}){
        state.ui_theme=light?"light":"dark";style.light=light;ecmgui_v2::ui::apply_theme(style);frame();
        const auto tint=ImGui::GetColorU32(ecmgui_v2::ui::palette(light).success_bg);
        const auto p=ecmgui_v2::ui::palette(light);
        const auto unread=ImGui::GetColorU32(ImVec4{(p.bg.x+p.info_bg.x)*.5f,(p.bg.y+p.info_bg.y)*.5f,(p.bg.z+p.info_bg.z)*.5f,1});
        assert(has_color(tint)&&has_color(unread)&&tint!=unread);
    }
    assert(app.result_uncopied_count==5);
    click_at(app.result_copy_controls[0]);assert(app.result_uncopied_count==5&&app.result_copies.last_copy().empty());
    assert(read(workflow).find("results_last_copy_utc")==std::string::npos);
    app.config.set("shared","workflow_worker_count","2");
    allow_clipboard=true;
    click_at(app.result_copy_controls[0]);assert(clipboard==row(10,10,3,true)&&app.result_uncopied_count==4);
    assert(!app.result_copies.last_copy().empty());
    click_at(app.result_copy_controls[1]);
    assert(clipboard==row(50,50,5)+second+row(100,100,2)+row(2,2,1)&&app.result_uncopied_count==0);
    const auto palette=ecmgui_v2::ui::palette(true);
    assert(has_color(ImGui::GetColorU32(palette.success_bg)));
    assert(read(workflow).find("results_last_copy_utc = \""+app.result_copies.last_copy()+"\"")!=std::string::npos);
    assert(app.config.dirty()&&app.config.get("shared","workflow_worker_count")=="2");
    assert(read(workflow).find("worker_count = 1")!=std::string::npos);
    ecmgui_v2::UiState restarted_state;
    ecmgui_v2::App restarted(restarted_state,stdout);
    assert(restarted.load_config(workflow.u8string()));
    assert(restarted.result_copies.last_copy()==app.result_copies.last_copy());
    for(const auto& key:app.result_copy_keys)assert(restarted.result_copies.contains(key));
    assert(read(a)==first&&read(b)==second+row(50,50,5));
    restarted.results.set_paths({a.u8string(),b.u8string()});
    write(b,read(b)+row(60,60,6));
    assert(restarted.results.refresh());
    draw_frame(restarted);
    assert(restarted.result_uncopied_count==1);
    ImGui::DestroyContext(context);
    std::cout<<"{\"result_sort_cycle\":true,\"sort_survives_refresh\":true,\"factored_row_both_themes\":true,\"copy_selected_then_new\":true,\"copy_history_restored\":true}\n";
}
static void stage2_main_config(const fs::path& root,const fs::path& combined) {
    assert(!fs::exists(root));
    fs::create_directories(root);
    const auto config=root/"ecm.toml";
    write(config,"config_version = 4\n[shared]\nworker_count = 3\nworktodo_file = \"worktodo.txt\"\n"
        "[stage2]\ndevice = \"gpu\"\nstage2_gpu_device = 0\n[gui]\nstage2_exe = \""+
        combined.generic_u8string()+"\"\n[worker.3]\ndevice = \"cPu\"\n");
    write(root/"worktodo.txt","[Worker #1]\nECMSTAGE2N=first.save,300\n"
        "[Worker #2]\n[Worker #3]\nECMSTAGE2N=third.save,300\n");
    ecmgui_v2::UiState state;
    ecmgui_v2::App app(state,stdout);
    app.initialize(config.u8string(),true);
    const auto& workers=app.controller.workers();
    assert(workers.size()==3&&workers[2].stage2);
    assert(fs::equivalent(fs::u8path(workers[2].exe),combined));
    assert(workers[2].execution_target=="cpu"&&!workers[2].duplicate_device);
    assert(workers[0].execution_target=="gpu");
}
int main(int argc,char** argv){
    if(argc==4&&std::string_view(argv[1])=="--stage2-main-config") {
        stage2_main_config(fs::absolute(fs::u8path(argv[2])),fs::absolute(fs::u8path(argv[3])));
        return 0;
    }
    pause_schedule_rules();
    {ecmgui::WorkerView worker;worker.stage2=true;auto& status=worker.stage2_status;
     const auto apply=[&](const char* line){status.apply(ecmgui::parse_line(line).stage2);};
     apply("stage2_plan: save=f.save requested=5 available=5 completed=2 B2=1000 D=210");
     assert(ecmgui_v2::ui::worker_progress(worker,false).speed=="--");
     apply("curve_start: 3/5 record=3 sigma=17 B1=20");
     apply("stage2_full_wall: init=1 main=9 total=10");apply("curve_done: 3/5 record=3 wall=100");
     assert(status.progress(false).s_per_curve==10&&status.progress(false).eta_s==20);
     apply("curve_start: 4/5 record=4 sigma=18 B1=20");assert(status.progress(false).s_per_curve==10);
     apply("curve_done: 4/5 record=4 wall=30");
     assert(status.progress().s_per_curve==20&&status.progress(false).s_per_curve==30);
     auto shown=ecmgui_v2::ui::worker_progress(worker,false);assert(shown.speed=="30.00"&&shown.eta=="30 s");
     apply("curve_start: 5/5 record=5 sigma=19 B1=20");apply("curve_done: 5/5 record=5 wall=0");
     assert(status.progress(false).s_per_curve==0&&ecmgui_v2::ui::worker_progress(worker,false).speed=="--");
     apply("stage2_plan: save=next.save requested=5 available=5 completed=0 B2=1000 D=210");
     assert(status.progress(false).s_per_curve==0&&status.measured_curves==0);}
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
    workflow_errors(root);
    stage1_details(root);
    if(argc>2){generator_while_running(root,fs::absolute(fs::u8path(argv[2])));stage1_ntt_progress(root,fs::absolute(fs::u8path(argv[2])));}
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
    write(cfg,"# preserve comment\nconfig_version=4\n[shared]\nworktodo_file=\"worktodo.txt\"\nworker_count=2\n[stage1]\ntarget=\"cpu\"\n[gui]\nfont_size=\"auto\"\n[worker.1]\nautostart=false\n[worker.2]\nautostart=false\n");
    write(queue,"# queue preserved\n[Worker #1]\n[Worker #2]\n");
    ecmgui_v2::UiState state;ecmgui_v2::App app(state,stdout);app.initialize(cfg.u8string(),true);assert(app.ready);
    const auto original=read(cfg);
    std::strcpy(app.gen_input,"ECM2=1,2,127,-1,100,1,0\n");app.gen_options.use_recommended=false;app.gen_options.curves_fixed=4;app.gen_worker=2;
    assert(app.preview_generator());assert(app.gen_preview.segments.size()==1&&app.gen_preview.segments.front().worker==2);
    std::strcpy(app.gen_b1,"200");assert(!app.apply_generator());assert(read(queue)=="# queue preserved\n[Worker #1]\n[Worker #2]\n");
    assert(app.preview_generator());write(queue,read(queue)+"# external edit\n");assert(!app.apply_generator());
    assert(app.preview_generator());assert(app.apply_generator());assert(!app.apply_generator());
    const auto appended=read(queue);assert(appended.find("ECMSTAGE1=")>appended.find("[Worker #2]"));
    app.gen_options.curve_type="m3";assert(!app.preview_generator());app.gen_options.curve_type="m0";
    assert(!app.edit_business("gui","font_size","20"));assert(!app.edit_business("stage1","not_known","4"));
    assert(app.edit_business("stage2","b2_ratio","20"));assert(!app.start(1));
    assert(app.preview_generator());assert(app.apply_generator());
    assert(app.config.dirty()&&read(cfg)==original);assert(app.save_business());
    assert(read(cfg).find("# preserve comment")!=std::string::npos&&read(cfg).find("font_size=\"auto\"")!=std::string::npos);
    assert(app.edit_business("stage2","b2_ratio","30"));write(cfg,read(cfg)+"# changed externally\n");assert(!app.save_business());assert(app.load_config(cfg.u8string()));
    {
        const auto v4_path = root / "workflow-v4.toml";
        write(v4_path, "config_version = 4\n[shared]\nqueue_order = \"file_order\"\n"
                       "[stage1]\nstage1_gpu_device = 0\n"
                       "[stage2]\nstage2_gpu_device = 1\n"
                       "[worker.2]\nautostart = false\nstage2_gpu_device = 2\n");
        ecmgui_v2::UiState preferences;
        ecmgui_v2::App v4(preferences, stdout);
        assert(v4.load_config(v4_path.u8string()));
        assert(v4.edit_business("shared", "queue_order", "stage1_first"));
        assert(v4.edit_business("worker.2", "stage2_gpu_device", "3"));
        assert(v4.edit_business("worker.2", "autostart", "true"));
        assert(!v4.edit_business("shared", "queue_order", "unknown"));
        assert(v4.save_business());
        const auto saved = read(v4_path);
        assert(saved.find("queue_order = \"stage1_first\"") != saved.npos &&
               saved.find("stage2_gpu_device = 3") != saved.npos &&
               saved.find("autostart = true") != saved.npos);
        const auto first = ecmgui::preview_stage1(ecm_config::preview_entries(v4.config.lines(), 2));
        const auto second = ecmgui::preview_stage2(ecm_config::preview_entries(v4.config.lines(), 2));
        assert(first.device == 0 && second.device == 3);
    }
    using namespace ecmgui::probability_ui;Request request;const auto forward=calculate(request);assert(forward.error.empty());
    auto inverse=request;inverse.solve=Solve::Curves;inverse.target=.75;const auto solved=calculate(inverse);assert(solved.error.empty());assert(cumulative(solved.probability.selected,solved.parameters.curves)>=.75);
    const auto csv=curve_csv(forward);request.parameters.b1=10;assert(curve_csv(forward)==csv);
    {auto* context=ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;state.ui_theme="light";
     assert(app.edit_business("stage2","b2_ratio","40")&&app.save_layout_config());
     const auto saved_config=read(cfg);
     assert(!app.config.dirty()&&saved_config.find("b2_ratio = 40")!=saved_config.npos&&
            saved_config.find("theme = \"light\"")!=saved_config.npos);
     assert(app.save_layout_config()&&read(cfg)==saved_config); // Clean workflow is not rewritten.
     assert(app.edit_business("stage2","b2_ratio","50"));write(cfg,saved_config+"# external save conflict\n");
     assert(!app.save_layout_config()&&read(cfg)==saved_config+"# external save conflict\n"&&app.config.dirty());
     state.ui_theme="dark";
     assert(app.persist_preferences()&&app.config.dirty());
     const auto merged=read(cfg);
     assert(merged.find("# external save conflict")!=merged.npos&&merged.find("theme = \"dark\"")!=merged.npos&&
            merged.find("b2_ratio = 40")!=merged.npos&&merged.find("b2_ratio = 50")==merged.npos);
     assert(app.load_config(cfg.u8string())&&state.ui_theme=="dark");
     ImGui::DestroyContext(context);}
    assert(original.find("font_size=\"auto\"")!=std::string::npos);
    {const auto path=root/"gp-config.toml";write(path,"# preserve GP comment\nconfig_version=4\n[stage2]\nb2_ratio=10\n[worker.1]\nautostart=false\n");
     ecmgui_v2::UiState preferences;ecmgui_v2::App gp_app(preferences,stdout);assert(gp_app.load_config(path.u8string()));
     assert(gp_app.edit_business("stage2","b2_ratio","25"));gp_app.group_order_input.gp="C:/GP Folder/gp.exe";assert(gp_app.save_group_order_gp());
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
    result_table_contract(root);
    if(argc==3){
        const auto fixture=fs::absolute(argv[2]);
        no_protocol_probe(root,fixture);task_watch_dispatch(root,fixture);scheduled_worker_dispatch(root,fixture);worker_breaker_recovery(root,fixture);
        stage2_raw_times(root,fixture);stage2_paired_views(root,fixture);
    }
    std::cout<<"{\"passed\":true,\"generator_target\":2,\"stale_preview_refused\":true,\"mtime_refused\":true,\"cpu_m3_refused\":true,\"legacy_ui_preserved\":true,\"external_config_refused\":true,\"probability_snapshot\":true,\"order_mouse_selection_clipboard_readonly\":true}\n";
}
