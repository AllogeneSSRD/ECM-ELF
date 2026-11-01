#include "../../src/ecm/application/configuration/generated/ecm_config_generated.h"
#include "../../src/ecm/application/gui/toml_file.h"
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
static void rejected(const std::filesystem::path &path,const std::string &body,
                     ecm_config::TomlReadMode mode=ecm_config::TomlReadMode::Strict) {
    write(path,body);
    std::vector<ecm_config::TomlLine> lines;
    try {
        ecm_config::read_toml(path.string(),lines,mode);
        (void)ecm_config::read_stage1(ecm_config::cli_entries(lines,2));
        (void)ecm_config::read_stage2(ecm_config::cli_entries(lines,2));
    }
    catch(const std::exception &) {return;}
    throw std::runtime_error("invalid TOML was accepted: "+body);
}
static void boolean_diagnostic(const std::filesystem::path &path, bool worker) {
    write(path, std::string(worker ? "[workers.2.stage2]" : "[stage2]") +
                "\nstage2_giant_tail_balance = 1\n");
    std::vector<ecm_config::TomlLine> lines;
    std::string diagnostic;
    try { ecm_config::read_toml(path.string(), lines); }
    catch (const std::exception &e) { diagnostic = e.what(); }
    const auto expected = "INPUT[config]\nfile=" + path.string() + ":2\nworker=" +
        (worker ? "2" : "0") + "\nkey=stage2_giant_tail_balance\nvalue=1\n" +
        "wrong TOML value type for stage2_giant_tail_balance\nexpected bool [false|true]";
    expect(diagnostic == expected, "TOML diagnostic must put each field on a separate line");
}
int main(int argc,char **argv) {
    try {
        expect(argc==3,"usage: fixture DEFAULT_TOML TEMP_TOML");
        const auto *tail_meta=ecm_config::toml_option("stage2","stage2_giant_tail_balance");
        expect(tail_meta && std::string(tail_meta->type)=="bool" && std::string(tail_meta->domain)=="[false|true]",
            "boolean error hint advertises non-TOML literals");
        std::vector<ecm_config::TomlLine> lines;
        expect(ecm_config::read_toml(argv[1],lines),"default TOML unavailable");
        const auto defaults1=ecm_config::read_stage1(ecm_config::cli_entries(lines,1));
        const auto defaults2=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
        expect(defaults2.groot_balanced=="auto", "G-root layout default must be automatic");
        expect(defaults1.gpu_param==3,
               "Stage1 generated default differs from TOML");
        expect(defaults1.progress_average_seconds==0,"speed averaging must default to off");
        expect(defaults2.device==-1,
               "Stage2 inheritance sentinel differs from TOML");
        expect(!defaults1.log_file_explicit&&!defaults1.has_submission&&!defaults1.has_results&&
               !defaults2.has_submission&&!defaults2.has_finished&&!defaults2.has_log&&
               !defaults2.has_owner&&!defaults2.has_pack_zero,
               "template disabled an implicit worker or environment default");
        const std::filesystem::path temp=argv[2];
        for (const auto *value : {"\"auto\"", "true", "false"}) {
            write(temp, std::string("[stage2]\nstage2_groot_balanced=")+value+
                        "\n[workers.2.stage2]\nstage2_groot_balanced=false\n");
            expect(ecm_config::read_toml(temp.string(),lines), "G-root tri-state TOML unavailable");
            const auto global=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
            expect(global.groot_balanced==(value==std::string("\"auto\"")?"auto":value),
                   "G-root tri-state value lost");
            expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,2)).groot_balanced=="false",
                   "G-root worker override lost");
        }
        for (const auto *value : {"1", "0", "\"invalid\""})
            rejected(temp,std::string("[stage2]\nstage2_groot_balanced=")+value+"\n");
        write(temp,"# preserve this comment\n[stage2]\nstage2_groot_balanced=\"auto\"\n");
        ecmgui::TomlFile editor;std::string editor_error;
        expect(editor.load(temp.string(),editor_error),"GUI editor cannot load tri-state option");
        for (const auto *value : {"true", "false", "auto"}) {
            editor.set("stage2","stage2_groot_balanced",value);
            expect(editor.save(editor_error),"GUI editor cannot save tri-state option");
            expect(ecm_config::read_toml(temp.string(),lines),"GUI wrote invalid tri-state TOML");
            expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,1)).groot_balanced==value,
                   "GUI tri-state roundtrip changed value");
            for(const auto &line:lines)if(line.key=="stage2_groot_balanced") {
                char kind=0;ecm_config::toml_value(line.raw.substr(line.raw.find('=')+1),kind);
                expect(kind==(std::string(value)=="auto"?'s':'b'),"GUI wrote the wrong TOML scalar type");
            }
            expect(lines.front().raw=="# preserve this comment","GUI changed an unrelated source line");
        }
        boolean_diagnostic(temp, false);
        boolean_diagnostic(temp, true);
        write(temp, "[stage2]\nstage2_giant_tail_balance = true\nstage2_b2_round_chain = false\n");
        expect(ecm_config::read_toml(temp.string(), lines), "TOML boolean literals unavailable");
        const auto booleans = ecm_config::read_stage2(ecm_config::cli_entries(lines, 2));
        expect(booleans.giant_tail_balance && !booleans.b2_round_chain,
               "TOML boolean values lost during projection");
        for(const auto *value:{"0","0.25","5","60"}) {
            write(temp,std::string("[stage1]\nprogress_average_seconds=")+value+
                       "\n[workers.2.stage1]\nprogress_average_seconds=2.5\n");
            expect(ecm_config::read_toml(temp.string(),lines),"speed-window TOML unavailable");
            expect(ecm_config::read_stage1(ecm_config::cli_entries(lines,1)).progress_average_seconds==std::stod(value),
                   "speed-window value lost");
            expect(ecm_config::read_stage1(ecm_config::cli_entries(lines,2)).progress_average_seconds==2.5,
                   "worker speed-window override lost");
        }
        for(const auto *value:{"-0.01","60.01","nan","inf","'5'"})
            rejected(temp,std::string("[stage1]\nprogress_average_seconds=")+value+"\n");
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
        rejected(temp,"[gui]\nfuture_style = true\n");
        rejected(temp,"[gui]\ngpu_clock_ranges = 123\n");
        write(temp,"[shared]\nworktodo='compute.txt'\n[gui]\nfuture_style=true\n"
                   "NumWorkers=3\ngpu_clock_ranges=123\n[workers.2]\nfuture_label='unused'\n"
                   "[workers.2.stage1]\ndevice=1\n");
        std::vector<std::string> ignored_gui;
        expect(ecm_config::read_toml(temp.string(),lines,ecm_config::TomlReadMode::Computation,&ignored_gui),"compute TOML unavailable");
        expect(ignored_gui.size()==2&&ignored_gui[0].find("gui.future_style")!=std::string::npos&&
               ignored_gui[1].find("workers.2.future_label")!=std::string::npos,"unknown GUI warnings missing");
        const auto compute=ecm_config::cli_entries(lines,2);
        expect(compute.size()==2&&ecm_config::read_stage1(compute).device==1&&
               ecm_config::read_stage2(compute).worktodo=="compute.txt","GUI data leaked into compute projection");
        const auto routing=ecm_config::section_entries(lines,"gui");
        expect(routing.size()==1&&ecm_config::read_gui(routing).NumWorkers==3,
               "required Stage1 routing worker count was lost");
        expect(ecm_config::section_entries(lines,"workers.2").empty(),"GUI worker values were retained in computation mode");
        for(const auto *invalid:{"[stage1]\nfuture_compute=true\n","[stage1]\ndevice='0'\n",
                 "[future]\nvalue=true\n","[gui]\nfuture_style='unterminated\n",
                 "[gui]\nNumWorkers='3'\n",
                 "[gui]\nfuture_style=true\nfuture_style=false\n"})
            rejected(temp,invalid,ecm_config::TomlReadMode::Computation);
        std::filesystem::remove(temp);
        std::filesystem::remove(temp.string()+".bak");
        std::cout<<"TOML parser and generated defaults: PASS\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
