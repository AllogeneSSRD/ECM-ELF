#define main fixture_main
#include "../gui/fake_worker.cpp"
#undef main
#include "../../../src/ecm/storage/queue/owner_lease.h"
#include "../../../src/ecm/application/gui/toml_file.h"
#include "../../../src/ecm/application/configuration/generated/ecm_config_generated.h"
int main(int argc, char** argv) {
    g_argc = argc; g_argv = argv;
    std::string config; int worker = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gui-queue-protocol") { std::fputs(ecm_queue::owner_protocol, stdout); return 0; }
        if (arg == "--config" && i + 1 < argc) config = argv[++i];
        else if (arg == "--worker" && i + 1 < argc) worker = std::stoi(argv[++i]);
    }
    try {
        ecm_queue::OwnerLease lease;
        if (!config.empty()) {
            ecmgui::TomlFile file; std::string error;
            if (!file.load(config, error)) throw std::runtime_error(error);
            const auto value = ecm_config::read_stage1(ecm_config::cli_entries(file.lines(), worker, true));
            lease.acquire((std::filesystem::u8path(config).parent_path() / std::filesystem::u8path(value.worktodo)).u8string(), worker);
            std::printf("queue_owner: acquired worker=%d\n", worker); std::fflush(stdout);
        }
        if (arg_value("--scenario") == "v2-checkpoint-stop") {
            HANDLE stop = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::strtoull(arg_value("--stage1-stop-handle").c_str(),nullptr,10)));
            DWORD flags = 0; if (!stop || !GetHandleInformation(stop,&flags)) return 4;
            out("START: waiting for v2 checkpoint stop\n");
            if (WaitForSingleObject(stop,30000) != WAIT_OBJECT_0) return 5;
            out("Checkpoint saved: v2 GUI stop event\n"); return 1;
        }
        if(arg_value("--scenario")=="v2-exit-wait"){
            out("START: %s\n",arg_value("--stage2-stop-handle").empty()?"ECMSTAGE1=1,2,127,-1,100,1":"ECMSTAGE2=1,2,137,-1,\"ed_m137.save\",0,0,10");
            HANDLE stop=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::strtoull(arg_value("--stage1-stop-handle",arg_value("--stage2-stop-handle")).c_str(),nullptr,10)));
            if(!stop||WaitForSingleObject(stop,30000)!=WAIT_OBJECT_0)return 5;
            Sleep(static_cast<DWORD>(std::stoul(arg_value("--stop-delay","1200"))));
            out("Checkpoint saved: v2 exit wait\n");return 1;
        }
        if(arg_value("--scenario")=="v2-performance"){
            HANDLE stop=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::strtoull(arg_value("--stage1-stop-handle").c_str(),nullptr,10)));
            out("START: ECMSTAGE1=1,2,127,-1,100,1\n");
            const int count=std::atoi(arg_value("--lines","4000").c_str());
            for(int i=0;i<count;++i)out("Checkpoint: isolated performance fixture %d\n",i);
            if(WaitForSingleObject(stop,60000)!=WAIT_OBJECT_0)return 5;
            out("Checkpoint saved: v2 performance stop\n");return 1;
        }
        if(arg_value("--scenario")=="v2-queue-switch"){
            const auto marker=std::filesystem::u8path(arg_value("--marker"));
            const auto deadline=GetTickCount64()+15000;
            while(!std::filesystem::exists(marker)&&GetTickCount64()<deadline)Sleep(10);
            if(!std::filesystem::exists(marker))return 5;
            out("===== queue done, 1 task(s) processed =====\n");return 0;
        }
        if(arg_value("--scenario")=="v2-presentation"){
            out("ECM-ELF 0.49.1 (Stage1 fixture)\n");
            out("START: ECMSTAGE1=1,2,%d,-1,100,1,\"1234567\"\n",worker==1?127:5237);
            Sleep(500);
            for(double speed:{1.75,2.25,1.85,2.10,1.95,2.0}){out("stage1: [====> ] 25.0%%  25/100 (~%.2f s/curve) elapsed 50.0s ETA 150.0s\n",speed);Sleep(300);}
            for(int i=0;i<80;++i){out("Checkpoint saved: worker=%d event=%d\n",worker,i);out("RAW ONLY worker=%d sample=%d\n",worker,i);}
            HANDLE stop=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::strtoull(arg_value("--stage1-stop-handle").c_str(),nullptr,10)));
            if(!stop||WaitForSingleObject(stop,60000)!=WAIT_OBJECT_0)return 5;
            out("Checkpoint saved: presentation stop worker=%d\n",worker);return 1;
        }
        if(arg_value("--scenario")=="v2-presentation-stage2"){
            out("ECM-ELF 0.49.1 (Stage2 fixture)\n");
            out("START: ECMSTAGE2=1,2,137,-1,\"ed_m137.save\",0,0,10\n");
            out("stage2_plan: save=ed_m137.save requested=10 available=10 completed=2 B2=40000 D=210\n");
            out("curve_start: 3/10 record=3 sigma=17 B1=20 checksum=verified curve_model=ed\n");
            out("real_shape: B1=260000000 B2=1300000000000 S_bits=4283 D=1381380 P=126720\n");
            out("stage2_phase: Giant points previous=0 elapsed=4\n");
            out("batched_progress: batch=2/8 t=4 left=12\n");Sleep(300);return 0;
        }
        return fixture_main(argc, argv);
    } catch (const std::exception& error) { std::fprintf(stderr, "FATAL: %s\n", error.what()); return 23; }
}
