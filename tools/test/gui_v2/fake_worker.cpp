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
        return fixture_main(argc, argv);
    } catch (const std::exception& error) { std::fprintf(stderr, "FATAL: %s\n", error.what()); return 23; }
}
