#include "../../src/ecm/application/configuration/generated/ecm_config_generated.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void expect(bool valid,const char *message) {
    if(!valid)throw std::runtime_error(message);
}
static void write(const std::filesystem::path &path,const std::string &body) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out<<body;
    expect(bool(out),"cannot write fixture");
}
static void rejected(const std::filesystem::path &path,const std::string &body) {
    write(path,body);
    std::vector<ecm_config::TomlLine> lines;
    try {
        ecm_config::read_toml(path.string(),lines);
        (void)ecm_config::read_stage1(ecm_config::cli_entries(lines,2));
        (void)ecm_config::read_stage2(ecm_config::cli_entries(lines,2));
    }
    catch(const std::exception &) {return;}
    throw std::runtime_error("invalid TOML was accepted: "+body);
}
int main(int argc,char **argv) {
    try {
        expect(argc==3,"usage: fixture DEFAULT_TOML TEMP_TOML");
        std::vector<ecm_config::TomlLine> lines;
        expect(ecm_config::read_toml(argv[1],lines),"default TOML unavailable");
        const auto defaults1=ecm_config::read_stage1(ecm_config::cli_entries(lines,1));
        const auto defaults2=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
        expect(defaults1.gpu_param==3,
               "Stage1 generated default differs from TOML");
        expect(defaults2.device==-1,
               "Stage2 inheritance sentinel differs from TOML");
        expect(!defaults1.log_file_explicit&&!defaults1.has_submission&&!defaults1.has_results&&
               !defaults2.has_submission&&!defaults2.has_finished&&!defaults2.has_log&&
               !defaults2.has_owner&&!defaults2.has_pack_zero,
               "template disabled an implicit worker or environment default");
        const std::filesystem::path temp=argv[2];
        write(temp,"[shared]\nworktodo = 'queue#1.txt' # comment\n"
                   "[stage1]\ndevice = 0\n[workers.2]\ngpucurves = 16\n"
                   "[workers.2.stage1]\ndevice = 1\n");
        expect(ecm_config::read_toml(temp.string(),lines),"worker TOML unavailable");
        const auto one=ecm_config::read_stage1(ecm_config::cli_entries(lines,1));
        const auto two=ecm_config::read_stage1(ecm_config::cli_entries(lines,2));
        expect(one.device==0&&two.device==1,"worker override precedence failed");
        expect(one.worktodo=="queue#1.txt"&&two.worktodo=="queue#1.txt",
               "shared TOML string/comment parse failed");
        expect(ecm_config::read_worker(ecm_config::section_entries(lines,"workers.2")).gpucurves==16,
               "GUI worker table failed");
        rejected(temp,"[Worker #2]\ndevice = 1\n");
        rejected(temp,"device = 1\n");
        rejected(temp,"[stage1]\ndevice = 0\ndevice = 1\n");
        rejected(temp,"[stage1]\ndevice = '0'\n");
        rejected(temp,"[stage2]\nstage2_pack_zero = 1\n");
        rejected(temp,"[workers.2.stage1]\nstage2_device = 1\n");
        rejected(temp,"[stage1]\nsigma = 18446744073709551616\n");
        std::filesystem::remove(temp);
        std::cout<<"TOML parser and generated defaults: PASS\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
