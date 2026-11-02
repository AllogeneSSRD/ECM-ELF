#include "../../../src/ecm/application/gui_v2/version.h"
#include "../../../include/ecm/platform/program_version.h"
#include "../../../src/ecm/application/gui_v2/generated/business_options.h"
#include "../../../src/ecm/application/gui/toml_file.h"
#include "../../../src/ecm/application/configuration/generated/ecm_config_generated.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool valid,const char* message) {
    if(!valid)throw std::runtime_error(message);
}
int main(int argc,char** argv) {
    try {
        if(argc==3 && std::string(argv[1])=="--read") {
            ecmgui::TomlFile config;std::string error;
            if(!config.load(argv[2],error))throw std::runtime_error(error);
            const auto gui=ecm_config::read_gui(ecm_config::section_entries(config.lines(),"gui"));
            std::cout<<"Workflow accepted; workers="<<gui.NumWorkers<<"; compute_started=0\n";
            return 0;
        }
        require(argc==2,"need isolated output directory or --read FILE");
        require(std::string(ecmgui_v2::version)==ecm_program::version,"GUI version differs from public program version");
        for(const auto* key:{"stage2_baby_backend","stage2_giant_backend","stage2_b2_round_chain","stage2_tune_profile"}) {
            bool found=false;
            for(const auto& option:ecmgui_v2::business_options)
                if(std::string(option.table)=="stage2" && std::string(option.key)==key)found=true;
            require(found,"current Stage2 option missing from GUI metadata");
        }
        const auto out=std::filesystem::path(argv[1]);std::filesystem::create_directories(out);
        const auto path=out/"ecm.toml";
        ecmgui::TomlFile config;config.reset(path.u8string());
        config.set("gui","NumWorkers","2");
        config.set("stage2","stage2_baby_backend","auto");
        config.set("stage2","stage2_giant_backend","auto");
        config.set("stage2","stage2_groot_balanced","auto");
        config.set("stage2","stage2_b2_round_chain","1");
        config.set("workers.1.stage2","stage2_tune_profile","profiles/gpu0.toml");
        config.set("workers.2.stage2","stage2_tune_profile","profiles/gpu1.toml");
        config.set("workers.2.stage2","stage2_giant_backend","coop16");
        std::string error;if(!config.save(error))throw std::runtime_error(error);
        ecmgui::TomlFile loaded;if(!loaded.load(path.u8string(),error))throw std::runtime_error(error);
        const auto first=ecm_config::read_stage2(ecm_config::preview_entries(loaded.lines(),1));
        const auto second=ecm_config::read_stage2(ecm_config::preview_entries(loaded.lines(),2));
        require(first.baby_backend=="auto" && first.giant_backend=="auto" && second.giant_backend=="coop16" && first.groot_balanced=="auto" && first.b2_round_chain,
                "Stage2 options changed during GUI edit/save/reload");
        require(first.tune_profile=="profiles/gpu0.toml" && second.tune_profile=="profiles/gpu1.toml","worker profile overrides lost");
        const auto future=out/"future-compute.toml";
        {std::ofstream f(future);f<<"[gui]\nNumWorkers=2\n[stage1]\nfuture_stage1=true\ncurve=\"future\"\n"
            "[stage2]\nfuture_stage2=7\nstage2_giant_backend=\"future\"\n[workers.2.stage2]\nfuture_worker_value=42\n";}
        if(!loaded.load(future.u8string(),error))throw std::runtime_error("GUI rejected another executable's parameters: "+error);
        const auto invalid=out/"wrong-table.toml";
        {std::ofstream f(invalid);f<<"[workers.1.stage1]\nstage2_tune_profile=\"profiles/gpu0.toml\"\n";}
        require(loaded.load(invalid.u8string(),error),"GUI must defer a misplaced compute key to its worker");
        std::vector<ecm_config::TomlLine> strict;
        bool rejected=false;
        try { ecm_config::read_toml(invalid.u8string(),strict,ecm_config::TomlReadMode::Stage1); }
        catch(const std::exception& e) { rejected=std::string(e.what()).find("stage2_tune_profile")!=std::string::npos; }
        require(rejected,"Stage1 must reject a misplaced key in its own table");
        require(loaded.load(future.u8string(),error),"future config reload failed");
        const auto preview=ecmgui::preview_stage2(ecm_config::preview_entries(loaded.lines(),2));
        require(preview.giant_backend=="auto","unsupported compute enum blocked GUI preview");
        loaded.set_int("gui","NumWorkers",3);require(loaded.save(error),"GUI save failed");
        std::ifstream saved(future);const std::string body((std::istreambuf_iterator<char>(saved)),{});
        require(body.find("future_stage1=true")!=body.npos && body.find("stage2_giant_backend=\"future\"")!=body.npos &&
                body.find("future_worker_value=42")!=body.npos,"GUI save removed or rewrote deferred parameters");
        const auto own=out/"invalid-owned.toml";
        for(const auto* input:{"[shared]\nb2_ratio=-1\n","[gui]\nNumWorkers=\"future\"\n","[workers.2]\nfuture_owned_key=true\n"}) {
            {std::ofstream f(own);f<<input;}
            require(!loaded.load(own.u8string(),error) && error.find("INPUT[config]")!=error.npos,
                    "GUI silently defaulted an invalid shared/owned field");
        }
        std::cout<<"GUI identity and current Stage2 configuration: PASS; compute_started=0\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
