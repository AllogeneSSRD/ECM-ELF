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
static std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
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
        config.set("workers.1.shared","shared_tune_file","profiles/gpu0.toml");
        config.set("workers.2.shared","shared_tune_file","profiles/gpu1.toml");
        config.set("workers.2.stage2","stage2_giant_backend","coop16");
        std::string error;if(!config.save(error))throw std::runtime_error(error);
        const auto generated = read_text(path);
        require(generated.find("config_version = 3") == 0 &&
                generated.find("[worker.2]") != generated.npos &&
                generated.find("shared.tune_file = \"profiles/gpu1.toml\"") != generated.npos,
                "new GUI configuration does not use the v3 layout");
        ecmgui::TomlFile loaded;if(!loaded.load(path.u8string(),error))throw std::runtime_error(error);
        require(loaded.config_version() == 3, "GUI failed to recognize v3");
        const auto first=ecm_config::read_stage2(ecm_config::preview_entries(loaded.lines(),1));
        const auto second=ecm_config::read_stage2(ecm_config::preview_entries(loaded.lines(),2));
        require(first.baby_backend=="auto" && first.giant_backend=="auto" && second.giant_backend=="coop16" && first.groot_balanced=="auto" && first.b2_round_chain,
                "Stage2 options changed during GUI edit/save/reload");
        require(first.shared_tune_file=="profiles/gpu0.toml" && second.shared_tune_file=="profiles/gpu1.toml","shared worker profile overrides lost");

        const auto mixed = out / "v2-editor.toml";
        const std::string original =
            "# Retain configuration format and comments\nconfig_version = 2\n"
            "[shared]\nqueue_order = \"file_order\"\n"
            "[stage1]\ngpu_device = 0\n"
            "[stage1.handoff]\n  enabled  = true  # keep this comment\n"
            "[stage2]\ngpu_device = 1\n"
            "[worker.2]\nautostart = true\nstage2.gpu_device = 3\n"
            "[worker.2.stage1]\ngpu_device = 4\n";
        {std::ofstream f(mixed);f<<original;}
        if(!loaded.load(mixed.u8string(),error))throw std::runtime_error(error);
        require(read_text(mixed) == original, "loading v2 rewrote the file");
        require(loaded.display_value("shared","priority") == "file_order" &&
                loaded.binding_value("shared","priority","stage1_first") == "1",
                "GUI enum edit did not round-trip through stable bindings");
        require(loaded.display_value("workers.2","autostart") == "true",
                "GUI did not display the v2 worker boolean");
        loaded.set("stage1","auto_stage2",loaded.binding_value("stage1","auto_stage2","false"));
        loaded.set("stage1","auto_stage2_worktodo","other#queue.txt");
        loaded.set("workers.2.stage1","device","5");
        if(!loaded.save(error))throw std::runtime_error(error);
        const auto edited = read_text(mixed);
        require(edited.find("  enabled  = false  # keep this comment") != edited.npos &&
                edited.find("worktodo_file = \"other#queue.txt\"") != edited.npos &&
                edited.find("handoff.worktodo_file") == edited.npos,
                "GUI did not preserve the existing subsection and inline comment");
        if(!loaded.load(mixed.u8string(),error))throw std::runtime_error(error);
        const auto worker1 = ecm_config::preview_entries(loaded.lines(),1);
        const auto worker2 = ecm_config::preview_entries(loaded.lines(),2);
        require(ecm_config::read_stage1(worker1).device == 0 &&
                ecm_config::read_stage1(worker2).device == 5 &&
                ecm_config::read_stage2(worker2).device == 3,
                "GUI merged identically named Stage1 and Stage2 keys");
        const auto root_worker = out / "v2-root-worker.toml";
        {std::ofstream f(root_worker);f<<"config_version=2\n[worker]\n2.autostart=false\n";}
        if(!loaded.load(root_worker.u8string(),error))throw std::runtime_error(error);
        loaded.set("workers.2.stage1", "device", "7");
        if(!loaded.save(error))throw std::runtime_error(error);
        require(read_text(root_worker).find("2.stage1.gpu_device = 7") != std::string::npos,
                "GUI redefined a worker created by a dotted key");
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
