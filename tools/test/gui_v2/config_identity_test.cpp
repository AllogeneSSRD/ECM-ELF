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
        const auto first=ecm_config::read_stage2(ecm_config::cli_entries(loaded.lines(),1));
        const auto second=ecm_config::read_stage2(ecm_config::cli_entries(loaded.lines(),2));
        require(first.baby_backend=="auto" && first.giant_backend=="auto" && second.giant_backend=="coop16" && first.groot_balanced=="auto" && first.b2_round_chain,
                "Stage2 options changed during GUI edit/save/reload");
        require(first.tune_profile=="profiles/gpu0.toml" && second.tune_profile=="profiles/gpu1.toml","worker profile overrides lost");
        const auto invalid=out/"wrong-table.toml";
        {std::ofstream f(invalid);f<<"[workers.1.stage1]\nstage2_tune_profile=\"profiles/gpu0.toml\"\n";}
        require(!loaded.load(invalid.u8string(),error) && error.find("stage2_tune_profile")!=error.npos,
                "misplaced Stage2 key accepted or unexplained");
        std::cout<<"GUI identity and current Stage2 configuration: PASS; compute_started=0\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
