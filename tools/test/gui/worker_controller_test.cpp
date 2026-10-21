#include "worker_controller.h"
#include "toml_file.h"
#include "../../../src/ecm/application/configuration/generated/ecm_config_generated.h"
#include <windows.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
namespace fs=std::filesystem;
using namespace ecmgui;
static int checks=0;
static void check(bool yes,const char *message){if(!yes)throw std::runtime_error(message);++checks;}
static std::string quoted(const std::string &s){std::ostringstream out;out<<std::quoted(s);return out.str();}
static std::string value(int argc,char **argv,const char *key){for(int i=1;i+1<argc;++i)if(std::string(argv[i])==key)return argv[i+1];return {};}
static bool contains(const std::vector<std::string> &lines,const std::string &text){for(const auto &v:lines)if(v.find(text)!=v.npos)return true;return false;}
static int child(int argc,char **argv) {
    std::cout.setf(std::ios::unitbuf);
    const auto scenario=value(argc,argv,"--controller-fixture");
    const auto stop1=value(argc,argv,"--stage1-stop-handle"),stop2=value(argc,argv,"--stage2-stop-handle");
    const bool stage2=!stop2.empty();
    std::cout<<"ECM-ELF 0.0.0 ("<<(stage2?"Stage2":"Stage1")<<")\n";
    if(scenario=="crash")return 9;
    if(scenario=="old"){std::cout<<"No input number on stdin\n";return 1;}
    if(scenario=="queue-step") {
        const auto queue=fs::u8path(value(argc,argv,"--queue"));std::ifstream in(queue);std::string first,rest,line;
        std::getline(in,first);while(std::getline(in,line))rest+=line+'\n';in.close();
        std::cout<<"START: "<<first<<'\n';
        if(!stage2)std::cout<<"factor[0]=101 curve=0 sigma=26 param=0 curve_model=mont execution_target=gpu\n";
        {std::ofstream out(queue);out<<rest;}
        if(stage2)std::cout<<"stage2_complete: curves_this_run=1 invalid_tasks=0\n";
        else std::cout<<"===== queue done, 1 task(s) processed =====\n";
        return 0;
    }
    if(scenario=="resumed")std::cout<<"Resuming from checkpoint: 23.8% complete (s_partial=1/4)\n";
    else std::cout<<"START: fixture task\nGPU: [====>   ] 42.0%  100, +50 bits (~1.25 s/curve) elapsed 1.0s remaining 1.0s\n";
    if(scenario=="checkpoint" || scenario=="stage2-stop") {
        const auto token=stage2?stop2:stop1;
        HANDLE event=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::stoull(token)));
        if(WaitForSingleObject(event,10000)!=WAIT_OBJECT_0)return 7;
        if(!stage2){{std::ofstream out(".m0_s1_m127_20.ckpt");out<<"fixture";}std::cout<<"Checkpoint saved: fixture\n";}
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        if(stage2)std::cout<<"stage2_stopped: curves_this_run=1 invalid_tasks=0\n";
        return 1;
    }
    if(scenario=="ok") {std::cout<<"===== queue done, 0 task(s) processed =====\n";return 0;}
    for(;;)std::this_thread::sleep_for(std::chrono::milliseconds(20));
}
struct Fixture {
    fs::path directory,queue,config;
    TomlFile ini;
    WorkerController controller{ini};
    WorkerSettings settings;
    std::vector<std::string> trace;
    std::vector<WorkerNotice> notices;
    Fixture(const fs::path &root,const fs::path &self,const std::string &name,const std::string &scenario,bool stage2=false,int autostart=0):directory(root/name) {
        fs::create_directories(directory/"bin");
        const auto exe=directory/"bin/fixture.exe";fs::copy_file(self,exe);
        queue=directory/"worktodo.txt";{std::ofstream out(queue);out<<(stage2?"ECMSTAGE2=fixture\n":"ECMSTAGE1=fixture\n");}
        config=directory/"ecm.toml";
        const auto args="--controller-fixture "+scenario+" --queue \""+queue.string()+"\"";
        {std::ofstream out(config);out<<"[shared]\nworktodo="<<quoted(queue.string())<<"\n[workers.1]\nautostart="<<autostart<<"\nextra_args="<<quoted(args)<<"\n";}
        std::string error;const bool loaded=ini.load(config.string(),error);check(loaded,error.c_str());
        settings={config.string(),exe.string(),exe.string(),"normal","ask",1,3000};
        controller.set_sinks([this](const std::string &line){trace.push_back(line);},[this](const WorkerNotice &notice){notices.push_back(notice);});
        controller.configure(settings);
    }
    void wait(const std::function<bool()> &done,int milliseconds=5000) {
        const auto deadline=mono_ms()+milliseconds;
        while(mono_ms()<deadline){controller.tick();if(done())return;std::this_thread::sleep_for(std::chrono::milliseconds(5));}
        throw std::runtime_error("Controller wait timed out");
    }
    const WorkerView &view()const{return controller.workers().front();}
    ~Fixture(){std::ofstream out(directory/"trace.log");for(const auto &line:trace)out<<line<<'\n';}
};
static int real_workers(const fs::path &config,const std::string &stage1,const std::string &stage2) {
    TomlFile ini;std::string error;const bool loaded=ini.load(config.string(),error);check(loaded,error.c_str());
    WorkerController controller(ini);std::vector<std::string> trace;
    controller.set_sinks([&](const std::string &line){trace.push_back(line);std::cout<<line<<std::endl;},{});
    controller.configure({config.string(),stage1,stage2,"normal","stop",1,5000});
    const auto cfg=ecm_config::read_stage1(ecm_config::cli_entries(ini.lines(),1,true));
    const auto queue=fs::u8path(cfg.worktodo);controller.start_all();const auto deadline=mono_ms()+60000;
    for(;;) {
        controller.tick();const auto &w=controller.workers().front();
        if(w.state()==WorkerRunState::Error || w.state()==WorkerRunState::Restarting) {
            for(const auto &line:w.raw)std::cerr<<line<<'\n';
            throw std::runtime_error("actual worker failed: "+w.state_text());
        }
        if(w.state()==WorkerRunState::QueueEmpty && fs::file_size(queue)==0)break;
        check(mono_ms()<deadline,"actual workers timed out");std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto &w=controller.workers().front();
    for(const auto &line:w.raw)std::cout<<"captured: "<<line<<'\n';
    check(w.stage2 && contains(trace,"dispatch ECMSTAGE2"),"actual mixed queue did not switch stages");
    check(w.application_version.rfind("ECM-ELF ",0)==0,"actual worker banner missing");
    std::cout<<"{\"passed\":true,\"actual_workers\":2,\"gui_window\":false,\"queue_empty\":true,\"stage1_to_stage2\":true}\n";
    return 0;
}
int main(int argc,char **argv) {
    try {
        if(!value(argc,argv,"--worker").empty())return child(argc,argv);
        if(argc==5 && std::string(argv[1])=="--real")return real_workers(fs::absolute(argv[2]),argv[3],argv[4]);
        check(argc==2,"expected fresh output directory");const auto root=fs::absolute(argv[1]),self=fs::absolute(argv[0]);
        check(!fs::exists(root),"output must be fresh");fs::create_directories(root);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<WorkerController &>().workers()[0])>>);
        {
            Fixture f(root,self,"autostart","ok",false,1);
            f.wait([&]{return f.view().state()==WorkerRunState::QueueEmpty;});
            const auto size=f.trace.size();for(int i=0;i<5;++i)f.controller.tick();
            check(f.trace.size()==size,"autostart repeated after queue completion");
            check(contains(f.trace,"autostart pid="),"autostart trace missing");
            check(f.view().progress_lines==1 && f.view().progress.valid && f.view().hist_s_per_curve.size()==1,"progress merge changed");
            check(f.view().application_version=="ECM-ELF 0.0.0","worker version lost");
            check(f.view().working_dir==(f.directory/"bin").string(),"Stage1 cwd changed");
        }
        {
            Fixture f(root,self,"checkpoint","checkpoint");check(f.controller.start(1),"start failed");
            f.wait([&]{return !f.view().task.empty();});
            bool rejected=false;try{f.controller.configure(f.settings);}catch(const std::runtime_error &){rejected=true;}
            check(rejected && f.view().running(),"reconfiguration changed active ownership");
            f.controller.request_stop(1);
            f.wait([&]{return contains(f.trace,"checkpoint file updated");});
            check(f.view().running(),"checkpoint notice killed the worker");
            f.wait([&]{return !f.view().stop_requested && !f.view().running();});
            check(f.view().state()==WorkerRunState::Stopped && !contains(f.trace,"stop timeout"),"normal checkpoint exit changed");
        }
        {
            Fixture f(root,self,"stage2","stage2-stop",true);check(f.controller.start(1),"Stage2 start failed");
            f.wait([&]{return !f.view().task.empty();});
            check(f.view().stage2 && f.view().working_dir==f.directory.string(),"Stage2 cwd/dispatch changed");
            check(f.view().command_line().find("--gui-events")!=std::string::npos,"Stage2 events missing");
            f.controller.request_stop(1);f.wait([&]{return !f.view().stop_requested && !f.view().running();});
            check(contains(f.trace,"finishing current Stage2 curve") && !contains(f.trace,"stop timeout"),"Stage2 graceful stop changed");
        }
        {
            Fixture f(root,self,"timeout","hang");f.settings.graceful_stop_ms=100;f.controller.configure(f.settings);
            check(f.controller.start(1),"hang start failed");f.controller.request_stop(1);
            f.wait([&]{return !f.view().running();});check(contains(f.trace,"stop timeout"),"stop deadline not enforced");
        }
        {
            Fixture f(root,self,"close","hang");check(f.controller.start(1),"close start failed");
            check(!f.controller.request_close() && f.controller.exit_phase()==WorkerController::ExitPhase::Confirm,"close confirmation lost");
            f.controller.cancel_close();check(f.controller.exit_phase()==WorkerController::ExitPhase::Idle && f.view().running(),"cancel killed worker");
            f.controller.force_close();check(f.controller.quit_requested() && !f.view().running(),"force close failed");
        }
        {
            Fixture f(root,self,"graceful-close","checkpoint");f.settings.exit_confirm="stop";f.controller.configure(f.settings);
            check(f.controller.start(1),"graceful close start failed");f.wait([&]{return !f.view().task.empty();});
            check(!f.controller.request_close(),"graceful close returned early");f.wait([&]{return f.controller.quit_requested();});
            check(!f.view().running() && !contains(f.trace,"stop timeout"),"graceful close did not wait for exit");
        }
        {
            Fixture f(root,self,"restart","crash");check(f.controller.start(1),"crash start failed");
            f.wait([&]{return f.view().state()==WorkerRunState::Restarting;});
            bool rejected=false;try{f.controller.configure(f.settings);}catch(const std::runtime_error &){rejected=true;}
            check(rejected,"restarting worker reconfigured");f.controller.request_stop(1);
            check(f.view().state()==WorkerRunState::Stopped && !f.controller.busy(),"restart cancellation changed");
        }
        {
            Fixture f(root,self,"resumed","resumed");check(f.controller.start(1),"resume start failed");
            f.wait([&]{return f.view().resume_pct>0;});check(f.view().resume_pct==23.8 && !f.view().progress.valid,"resume projection changed");f.controller.force_close();
        }
        {
            Fixture f(root,self,"old","old");check(f.controller.start(1),"old start failed");
            f.wait([&]{return f.view().old_driver;});check(!f.notices.empty() && f.notices.back().kind==WorkerNotice::Kind::OldDriver,"old driver notice lost");f.controller.force_close();
        }
        {
            Fixture f(root,self,"mixed","queue-step");{std::ofstream out(f.queue);out<<"ECMSTAGE1=first\nECMSTAGE2=second\nECMSTAGE1=third\n";}
            f.controller.start_all();f.wait([&]{return f.view().state()==WorkerRunState::QueueEmpty && fs::file_size(f.queue)==0;});
            check(contains(f.trace,"dispatch ECMSTAGE2") && contains(f.trace,"dispatch ECMSTAGE1"),"mixed queue phase switch lost");
            check(f.view().hits==2 && f.view().tasks_done==0 && !f.view().events.empty(),"retained state changed across phases");
            int changed=0;for(const auto &n:f.notices)changed+=n.kind==WorkerNotice::Kind::StageChanged;
            check(changed==2,"duplicate mixed dispatch");
        }
        std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"scenarios\":10,\"gpu_devices\":0}\n";return 0;
    }catch(const std::exception &e){std::cerr<<"Controller test: "<<e.what()<<'\n';return 1;}
}
