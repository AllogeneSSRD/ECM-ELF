#include "platform.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("fresh output directory required");
        const std::filesystem::path dir=argv[1];
        if(std::filesystem::exists(dir))throw std::runtime_error("output already exists");
        std::filesystem::create_directories(dir);std::string found;
        for(const auto *name:{".ecm_ckpt_347_ab_cd.dat",".e_s1_m347_1000.ckpt",".m0_s1_m347_1000.ckpt",
                              ".m2_s1_m347_1000.ckpt",".m3_s1_m347_1000.ckpt"}) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            {std::ofstream f(dir/name);f<<"fixture";}
            if(!ecmgui::newest_checkpoint_mtime(dir.string(),&found) || found!=name)
                throw std::runtime_error("primary checkpoint name was not detected");
        }
        for(const auto *name:{".m3_s1_m347_1000.bak.ckpt",".e_s1_m347_1000.ckpt.new",".ecm_ckpt_stage2_347_x.dat"}) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            {std::ofstream f(dir/name);f<<"excluded";}
            ecmgui::newest_checkpoint_mtime(dir.string(),&found);
            if(found!=".m3_s1_m347_1000.ckpt")throw std::runtime_error("non-primary checkpoint was detected");
        }
        std::cout<<"gui_checkpoint_scan=PASS primaries=5 exclusions=3\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
