#include "../../src/ecm/application/configuration/generated/ecm_config_generated.h"
#include "../../src/ecm/application/gui/toml_file.h"
#include "../../src/io/config.h"
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
static void edwards_doubling_config(const std::filesystem::path& path) {
    write(path, "config_version=3\n[stage1]\nedwards_doubling=0\n"
                "[worker.2]\nstage1.edwards_doubling=1\n");
    ParsedConfig parsed;
    expect(read_config(path.string(), parsed), "valid Edwards doubling config rejected");
    EcmQueueConfig cfg;
    project_stage1_config(parsed, 1, cfg);
    expect(cfg.edwards_doubling == 0 && cfg.config_keys.count("edwards_doubling"),
           "Edwards HWCD config not inherited");
    project_stage1_config(parsed, 2, cfg);
    expect(cfg.edwards_doubling == 1, "Edwards worker override not applied");
    for (const auto* value : {"-1", "2", "'1'", "1.0", "true"})
        rejected(path, std::string("config_version=3\n[stage1]\nedwards_doubling=") + value + "\n");
}

static void boolean_diagnostic(const std::filesystem::path &path, bool worker) {
    write(path, std::string(worker ? "[workers.2.stage2]" : "[stage2]") +
                "\nstage2_giant_tail_balance = 1.0\n");
    std::vector<ecm_config::TomlLine> lines;
    std::string diagnostic;
    try { ecm_config::read_toml(path.string(), lines); }
    catch (const std::exception &e) { diagnostic = e.what(); }
    const auto expected = "INPUT[config]\nfile=" + path.string() + ":2\nworker=" +
        (worker ? "2" : "0") + "\nkey=stage2_giant_tail_balance\nvalue=1.0\n" +
        "wrong TOML value type for stage2_giant_tail_balance\nexpected bool [false|true|0|1]";
    expect(diagnostic == expected, "TOML diagnostic must put each field on a separate line");
}
static size_t boolean_aliases(const std::filesystem::path &path) {
    size_t checks=0;
    for(const auto &option:ecm_config::toml_options) {
        const std::string type=option.type,table=option.table;
        if(type!="bool" && type!="bool|string")continue;
        for(const bool worker:{false,true}) {
            if(worker && table=="gui")continue;
            const auto section=worker?"workers.2."+table:table;
            for(const auto *value:{"false","true","0","1","+0","+1","-0"}) {
                const std::string expected=(std::string(value)=="true" || std::string(value)=="1" ||
                    std::string(value)=="+1")?"true":"false";
                const auto body="["+section+"]\n"+option.key+"="+value+" # keep numeric spelling\n";
                write(path,body);std::vector<ecm_config::TomlLine> lines;
                expect(ecm_config::read_toml(path.string(),lines),"boolean alias TOML unavailable");
                expect(lines.back().value==expected && lines.back().raw.find(value)!=std::string::npos,
                       "boolean alias lost its value or original source spelling");
                const auto values=ecm_config::cli_entries(lines,worker?2:1);
                (void)ecm_config::read_stage1(values);(void)ecm_config::read_stage2(values);
                if(table=="gui")(void)ecm_config::read_gui(ecm_config::section_entries(lines,table));
                else {
                    expect(ecm_config::read_toml(path.string(),lines,ecm_config::TomlReadMode::Computation),
                           "computation reader rejected a boolean alias");
                    expect(lines.back().value==expected,"computation reader changed a boolean alias");
                }
                ++checks;
            }
            for(const auto *value:{"2","-1","1.0","0.0","[1]","yes"}) {
                rejected(path,"["+section+"]\n"+option.key+"="+value+"\n");++checks;
            }
        }
    }
    expect(checks>0,"schema has no boolean options");return checks;
}
static void role_isolation(const std::filesystem::path& path) {
    using Mode=ecm_config::TomlReadMode;
    std::vector<ecm_config::TomlLine> lines;
    for(const auto& table:{"stage2","workers.2.stage2"}) {
        write(path,"[shared]\nworktodo=\"queue.txt\"\n["+std::string(table)+
            "]\nfuture_key=[\"a\",\"b\"]\nstage2_giant_backend=42\nstage2_finished=\"receiver.finished\"\n");
        ParsedConfig parsed;expect(read_config(path.string(),parsed),"Stage1 validated foreign Stage2");
        expect(ecm_config::read_stage1(ecm_config::cli_entries(parsed.lines,2)).worktodo=="queue.txt","shared routing lost");
        const auto receiver=ecm_config::read_stage2(ecm_config::selected_entries(parsed.lines,2,
            {"worktodo","stage2_finished"},path.string()));
        expect(receiver.finished=="receiver.finished","Stage1 handoff lost receiver path");
        rejected(path,"["+std::string(table)+"]\nfuture_key=1\n",Mode::Stage2);
    }
    for(const auto& table:{"stage1","workers.2.stage1"}) {
        write(path,"[shared]\nworktodo=\"queue.txt\"\n["+std::string(table)+
            "]\nfuture_key=true\ncurve=\"future\"\nsigma=\"future\"\n[gui]\nNumWorkers=\"future\"\n");
        EcmQueueConfig inherited;
        expect(ecm_queue_config_load(path.string(),2,inherited),"Stage2 compatibility loader validated foreign Stage1");
        expect(inherited.worktodo=="queue.txt","Stage2 shared worktodo lost");
        expect(ecm_config::read_toml(path.string(),lines,Mode::Stage2),"Stage2 parser rejected foreign table");
        expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,2)).worktodo=="queue.txt","Stage2 projection failed");
        rejected(path,"["+std::string(table)+"]\nfuture_key=true\n",Mode::Stage1);
    }
    write(path,"[stage1]\ndevice=1\ntmp_dir=\"saves\"\nlcm=\"12\"\nfuture_key=true\n"
        "[workers.2.stage1]\ndevice=0\n[workers.2.stage2]\nstage2_device=1\n");
    EcmQueueConfig inherited;expect(ecm_queue_config_load(path.string(),2,inherited),"Stage2 inheritance failed");
    expect(inherited.device==0 && inherited.tmp_dir=="saves" && inherited.scalar_construction=="12","Stage2 legacy defaults changed");
    write(path,"[stage1]\ndevice=\"future\"\n[workers.2.stage1]\ndevice=1\n"
        "[workers.3.stage1]\ndevice=\"future\"\n");
    expect(ecm_queue_config_load(path.string(),2,inherited) && inherited.device==1,"unused foreign worker/default blocked inheritance");
    write(path,"[stage2]\nstage2_giant_backend=\"future\"\n");
    expect(ecm_config::read_toml(path.string(),lines,Mode::Stage2),"enum parse failed before consumer");
    bool bad=false;try{ecm_config::read_stage2(ecm_config::cli_entries(lines,2));}catch(...){bad=true;}
    expect(bad,"Stage2 accepted unknown own enum");
    write(path,"[stage1]\ndevice=\"future\"\ncurve=\"future\"\nlcm=\"future\"\n");
    ParsedConfig own_stage1;expect(!read_config(path.string(),own_stage1),"Stage1 ignored its own bad input");
    expect(ecm_queue_config_load(path.string(),2,inherited,{}),"unused legacy defaults blocked Stage2");
    for(auto mode:{Mode::Stage1,Mode::Stage2,Mode::Gui}) {
        rejected(path,"[shared]\nfuture_shared=true\n",mode);
        rejected(path,"[shared]\nworktodo=123\n",mode);
        rejected(path,"[stage1]\nfuture_key=\"unterminated\n",mode);
        rejected(path,"[stage2]\nfuture_key=1\nfuture_key=2\n",mode);
    }
    write(path,"[gui]\nNumWorkers=\"future\"\n");
    ParsedConfig invalid_routing;expect(!read_config(path.string(),invalid_routing),"Stage1 routing count must remain checked");
    write(path,"[stage2]\nstage2_finished=123\n");
    ParsedConfig parsed;expect(read_config(path.string(),parsed),"unused receiver path blocked Stage1");
    bad=false;try{ecm_config::selected_entries(parsed.lines,2,{"stage2_finished"},path.string());}catch(...){bad=true;}
    expect(bad,"consumed Stage2 receiver path type was not checked");
}

static void versioned_paths(const std::filesystem::path& path) {
    using Mode = ecm_config::TomlReadMode;
    const std::string old =
        "[shared]\nworktodo='queue.txt'\npriority=2\nskip=1\n"
        "[stage1]\ndevice=1\nauto_stage2=1\nntt_length=1024\n"
        "[stage2]\nstage2_device=2\nstage2_parallel='auto'\nstage2_finished=''\n"
        "[gui]\nNumWorkers=2\n[workers.2]\nautostart=1\n"
        "[workers.2.stage1]\ndevice=3\n[workers.2.stage2]\nstage2_device=4\n";
    const std::string current =
        "config_version=2 # Business config version\n"
        "[shared]\nworktodo_file='queue.txt'\nqueue_order='stage2_first'\nqueue_skip='stage1'\n"
        "[stage1]\ngpu_device=1\nhandoff.enabled=true\nntt.length=1024\n"
        "[stage2]\ngpu_device=2\ncuda.curve_concurrency='auto'\nfinished_file=''\n"
        "[gui]\nworker_count=2\n[worker.2]\nautostart=true\n"
        "stage1.gpu_device=3\nstage2.gpu_device=4\n";

    const auto validate_projection = [&](const std::string& body, int version) {
        write(path,body);
        std::vector<ecm_config::TomlLine> lines;
        int parsed_version = 0;
        expect(ecm_config::read_toml(path.string(),lines,Mode::Strict,nullptr,&parsed_version) &&
                   parsed_version==version,"wrong TOML version");
        for (const int worker : {1,2}) {
            const auto entries = ecm_config::cli_entries(lines,worker);
            const auto stage1 = ecm_config::read_stage1(entries);
            const auto stage2 = ecm_config::read_stage2(entries);
            expect(stage1.device==(worker==1?1:3) && stage2.device==(worker==1?2:4),
                   "same-name fields crossed stage/worker scopes");
            expect(stage1.auto_stage2==1 && stage1.ntt_length==1024 && stage2.parallel==0 &&
                       stage1.worktodo_priority==2 && stage2.worktodo_skip==1,
                   "v1 and v2 effective values differ");
            expect(std::find(entries.begin(),entries.end(),std::make_pair(
                std::string("stage2_finished"),std::string{}))!=entries.end(),
                "explicit empty value became absent");
        }
        expect(ecm_config::read_gui(ecm_config::section_entries(lines,"gui")).NumWorkers==2,
               "GUI worker count lost");
        expect(ecm_config::read_worker(ecm_config::section_entries(lines,"workers.2")).autostart==1,
               "worker launch flag lost");
        ParsedConfig parsed;
        expect(read_config(path.string(),parsed),"Stage1 did not read versioned config");
        EcmQueueConfig stage1;
        project_stage1_config(parsed,2,stage1);
        expect(stage1.auto_stage2==1 && stage1.device==3,"Stage1 typed projection changed");
    };
    validate_projection(old,1);
    validate_projection("config_version=1\n"+old,1);
    validate_projection(current,2);

    auto subsection = current;
    const std::string prior = "handoff.enabled=true\nntt.length=1024\n";
    subsection.replace(subsection.find(prior),prior.size(),
                       "ntt.length=1024\n[stage1.handoff]\nenabled=true\n");
    validate_projection(subsection,2);

    for (const auto* invalid : {
            "config_version='2'\n", "config_version=2.0\n", "config_version=4\n",
            "config_version=2\nconfig_version=1\n", "[stage1]\nconfig_version=2\n",
            "config_version=2\n[stage1]\nhandoff.enabled=true\n[stage1.handoff]\nworkers='2'\n",
            "config_version=2\n[stage1.handoff]\nworkers='2'\n[stage1]\nhandoff.enabled=true\n",
            "config_version=2\n[stage1]\nhandoff=true\nhandoff.enabled=true\n",
            "config_version=2\n[stage1]\n'handoff.enabled'=true\n",
            "config_version=2\n[stage1]\nhandoff.enabled=1\n",
            "config_version=2\n[stage1]\nauto_stage2=1\n",
            "config_version=2\n[worker.0]\nautostart=true\n",
            "config_version=2\n[worker.1000001]\nautostart=true\n"})
        rejected(path,invalid);

    write(path,"config_version=2\n[stage1]\ngpu_device=7\n[stage2]\n"
               "finished_file='receiver'\nfuture.values=['a','b']\n"
               "stage2_finished='legacy_name_in_foreign_table'\n");
    std::vector<ecm_config::TomlLine> lines;
    expect(ecm_config::read_toml(path.string(),lines,Mode::Stage1),
           "Stage1 rejected future foreign Stage2 option");
    const auto fields = ecm_config::selected_entries(lines,1,{"stage2_finished"},path.string());
    expect(ecm_config::read_stage2(fields).finished=="receiver",
           "foreign alias shadowed the recognized Stage2 handoff path");
    expect(lines.back().raw=="stage2_finished='legacy_name_in_foreign_table'" &&
               !lines.back().recognized && lines.back().deferred,"foreign value source lost");
}

static void cpu_slice_config(const std::filesystem::path& path) {
    write(path, "config_version=3\n[stage1]\n"
                "cpu_implementation='simd[:6144], scalar[:4096], gwnum[4096:], auto[]'\n"
                "[worker.2]\nstage1.cpu_implementation=''\n");
    ParsedConfig parsed;
    expect(read_config(path.string(), parsed), "Stage1 did not read v3 CPU policy");
    EcmQueueConfig stage1;
    project_stage1_config(parsed, 1, stage1);
    expect(stage1.selection.cpu_implementation_rules.size() == 4 &&
               stage1.selection.cpu_implementation_rules.front().end_bit == 6144,
           "Stage1 lost global CPU policy rules");
    project_stage1_config(parsed, 2, stage1);
    expect(stage1.selection.cpu_implementation_rules.empty() &&
               stage1.selection.cpu_implementation == ecm_stage1::CpuImplementation::Auto,
           "Stage1 did not clear inherited CPU policy with explicit empty override");
    write(path, "config_version=3\n[stage1]\ncpu_implementation='simd[3:2]'\n");
    expect(!read_config(path.string(), parsed), "Stage1 accepted invalid CPU slice");
    write(path, "config_version=2\n[stage1]\ncpu_implementation='simd[:6144], auto[]'\n");
    expect(!read_config(path.string(), parsed), "Stage1 accepted a v3 CPU policy in a v2 file");
}

static void v3_shared_config(const std::filesystem::path& path) {
    write(path, "config_version=3\n[shared]\nlog_file='combined.log'\nfinished_file='done.txt'\n"
                "log_file_max_KB=1000\ntune_file='shared.toml'\n"
                "[stage1]\nparam='m0'\n"
                "[stage2]\nb2_ratio=20\n[worker.2]\nshared.tune_file='worker.toml'\n"
                "stage2.b2_ratio=30\n[worker.3]\nshared.tune_file=''\n");
    ParsedConfig parsed;
    expect(read_config(path.string(), parsed), "v3 Stage1 config unavailable");
    EcmQueueConfig cfg;
    project_stage1_config(parsed, 1, cfg);
    expect(cfg.param == "m0" && cfg.shared_log_file == "combined.log" &&
           cfg.shared_finished_file == "done.txt" && cfg.log_file_max_kb == 1000 &&
           cfg.ntt_tune_file == "shared.toml" && cfg.handoff_mode == "ecm-elf", "v3 global fields lost");
    project_stage1_config(parsed, 2, cfg);
    expect(cfg.ntt_tune_file == "worker.toml", "worker tune did not override global Stage1 path");
    std::vector<ecm_config::TomlLine> lines;
    expect(ecm_config::read_toml(path.string(), lines, ecm_config::TomlReadMode::Stage2), "v3 Stage2 config unavailable");
    expect(ecm_config::read_stage2(ecm_config::cli_entries(lines, 2)).shared_tune_file == cfg.ntt_tune_file,
           "Stage1 and Stage2 resolved different worker tune files");
    project_stage1_config(parsed, 3, cfg);
    expect(cfg.ntt_tune_file.empty() &&
           ecm_config::read_stage2(ecm_config::cli_entries(lines, 3)).shared_tune_file.empty(),
           "empty worker tune path was lost");
    expect(ecm_queue_config_load(path.string(), 2, cfg) && cfg.b2_ratio == 30, "worker B2 ratio was lost");
    rejected(path, "config_version=3\n[worker.2]\nshared.log_file='invalid.log'\n");
    for (const auto* wrong : {"[stage1]\ntune_file='wrong.toml'\n", "[stage2]\ntune_file='wrong.toml'\n",
                              "[worker.2]\nstage1.tune_file='wrong.toml'\n",
                              "[worker.2.stage2]\ntune_file='wrong.toml'\n"}) {
        const auto source = std::string("config_version=3\n") + wrong;
        rejected(path, source, ecm_config::TomlReadMode::Stage1);
        rejected(path, source, ecm_config::TomlReadMode::Stage2);
        rejected(path, source, ecm_config::TomlReadMode::Gui);
    }
}

int main(int argc,char **argv) {
    try {
        expect(argc==3,"usage: fixture DEFAULT_TOML TEMP_TOML");
        versioned_paths(argv[2]);
        cpu_slice_config(argv[2]);
        v3_shared_config(argv[2]);
        edwards_doubling_config(argv[2]);
        const auto *tail_meta=ecm_config::toml_option("stage2","stage2_giant_tail_balance");
        expect(tail_meta && std::string(tail_meta->type)=="bool" && std::string(tail_meta->domain)=="[false|true|0|1]",
            "boolean error hint omits accepted aliases");
        std::vector<ecm_config::TomlLine> lines;
        expect(ecm_config::read_toml(argv[1],lines),"default TOML unavailable");
        const auto defaults1=ecm_config::read_stage1(ecm_config::cli_entries(lines,1));
        const auto defaults2=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
        expect(defaults2.groot_balanced=="auto", "G-root layout default must be automatic");
        expect(defaults1.gpu_param==3,
               "Stage1 generated default differs from TOML");
        expect(defaults1.progress_average_seconds==0,"speed averaging must default to off");
        expect(defaults1.edwards_doubling==1, "Edwards doubling must default to 2XY");
        expect(defaults2.device==-1,
               "Stage2 inheritance sentinel differs from TOML");
        expect(!defaults1.log_file_explicit&&!defaults1.has_submission&&!defaults1.has_results&&
               !defaults2.has_submission&&!defaults2.has_finished&&!defaults2.has_log&&
               !defaults2.has_owner&&!defaults2.has_pack_zero,
               "template disabled an implicit worker or environment default");
        const std::filesystem::path temp=argv[2];
        role_isolation(temp);
        const auto alias_checks=boolean_aliases(temp);
        for (const auto *value : {"\"auto\"", "true", "false", "1", "0"}) {
            write(temp, std::string("[stage2]\nstage2_groot_balanced=")+value+
                        "\n[workers.2.stage2]\nstage2_groot_balanced=0\n");
            expect(ecm_config::read_toml(temp.string(),lines), "G-root tri-state TOML unavailable");
            const auto global=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
            const auto canonical=value==std::string("\"auto\"")?std::string("auto"):
                ecm_config::automatic_boolean_text(value,"groot fixture");
            expect(global.groot_balanced==canonical,
                   "G-root tri-state value lost");
            expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,2)).groot_balanced=="false",
                   "G-root worker override lost");
        }
        for (const auto *value : {"2", "1.0", "\"invalid\""})
            rejected(temp,std::string("[stage2]\nstage2_groot_balanced=")+value+"\n");
        write(temp,"# preserve this comment\n[stage2]\nstage2_groot_balanced=\"auto\"\n");
        ecmgui::TomlFile editor;std::string editor_error;
        expect(editor.load(temp.string(),editor_error),"GUI editor cannot load tri-state option");
        for (const auto *value : {"true", "false", "1", "0", "auto"}) {
            editor.set("stage2","stage2_groot_balanced",value);
            const auto canonical=ecm_config::automatic_boolean_text(value,"groot fixture");
            expect(editor.get("stage2","stage2_groot_balanced")==canonical,"GUI kept a noncanonical boolean alias");
            expect(editor.save(editor_error),"GUI editor cannot save tri-state option");
            expect(ecm_config::read_toml(temp.string(),lines),"GUI wrote invalid tri-state TOML");
            expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,1)).groot_balanced==canonical,
                   "GUI tri-state roundtrip changed value");
            for(const auto &line:lines)if(line.key=="stage2_groot_balanced") {
                char kind=0;ecm_config::toml_value(line.raw.substr(line.raw.find('=')+1),kind);
                expect(kind==(std::string(value)=="auto"?'s':'b'),"GUI wrote the wrong TOML scalar type");
            }
            expect(lines.front().raw=="# preserve this comment","GUI changed an unrelated source line");
        }
        write(temp,"# keep this spelling\n[stage2]\nstage2_giant_tail_balance = 1 # enabled\n");
        expect(editor.load(temp.string(),editor_error),"GUI cannot load numeric boolean");
        expect(editor.get("stage2","stage2_giant_tail_balance")=="true","GUI loaded 1 as false");
        editor.set("stage2","stage2_giant_tail_balance","true");
        expect(editor.lines().back().raw=="stage2_giant_tail_balance = 1 # enabled",
               "GUI rewrote a semantically unchanged numeric boolean");
        for(const auto *value:{"0","1","false","true"}) {
            editor.set("stage2","stage2_giant_tail_balance",value);
            expect(editor.save(editor_error),"GUI cannot save numeric boolean");
            expect(ecm_config::read_toml(temp.string(),lines),"GUI wrote invalid boolean");
            const auto settings=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
            expect(settings.giant_tail_balance==ecm_config::boolean(value,"tail fixture"),"GUI boolean alias changed value");
        }
        bool invalid_editor=false;
        try {editor.set("stage2","stage2_giant_tail_balance","2");}
        catch(const std::exception &){invalid_editor=true;}
        expect(invalid_editor,"GUI silently converted an invalid boolean to false");
        boolean_diagnostic(temp, false);
        boolean_diagnostic(temp, true);
        write(temp, "[stage2]\nstage2_giant_tail_balance = true\nstage2_b2_round_chain = false\n");
        expect(ecm_config::read_toml(temp.string(), lines), "TOML boolean literals unavailable");
        const auto booleans = ecm_config::read_stage2(ecm_config::cli_entries(lines, 2));
        expect(booleans.giant_tail_balance && !booleans.b2_round_chain,
               "TOML boolean values lost during projection");
        for(const auto *baby:{"auto","scalar","coop32"})for(const auto *giant:{"auto","scalar","coop16","coop32"}) {
            write(temp,std::string("[stage2]\nstage2_baby_backend=\"")+baby+"\"\nstage2_giant_backend=\""+giant+
                "\"\n[workers.2.stage2]\nstage2_baby_backend=\"coop32\"\nstage2_giant_backend=\"coop16\"\n");
            expect(ecm_config::read_toml(temp.string(),lines),"point backend TOML unavailable");
            const auto global=ecm_config::read_stage2(ecm_config::cli_entries(lines,1));
            const auto worker=ecm_config::read_stage2(ecm_config::cli_entries(lines,2));
            expect(global.has_baby_backend && global.has_giant_backend && global.baby_backend==baby && global.giant_backend==giant,
                "point backend values lost");
            expect(worker.baby_backend=="coop32" && worker.giant_backend=="coop16","point backend worker override lost");
        }
        for(const auto *value:{"\"coop16\"","\"wrong\"","32"})
            rejected(temp,std::string("[stage2]\nstage2_baby_backend=")+value+"\n");
        for(const auto *value:{"\"coop8\"","\"wrong\"","32"})
            rejected(temp,std::string("[stage2]\nstage2_giant_backend=")+value+"\n");
        write(temp,"[stage1]\nstage1_backend=\"opencl\"\n[stage2]\nstage2_backend=\"cuda\"\n"
            "[workers.2.stage2]\nstage2_backend=\"opencl\"\n");
        expect(ecm_config::read_toml(temp.string(),lines),"GPU backend TOML unavailable");
        expect(ecm_config::read_stage1(ecm_config::cli_entries(lines,2)).gpu_backend=="opencl" &&
            ecm_config::read_stage2(ecm_config::cli_entries(lines,1)).gpu_backend=="cuda" &&
            ecm_config::read_stage2(ecm_config::cli_entries(lines,2)).gpu_backend=="opencl",
            "stage/worker GPU backend selection leaked");
        rejected(temp,"[stage2]\nstage2_backend=\"cpu\"\n");
        write(temp,"[stage2]\nstage2_tune_profile=\"profiles/shared.toml\"\n"
            "[workers.1.stage2]\nstage2_tune_profile=\"profiles/gpu0.toml\"\n"
            "[workers.2.stage2]\nstage2_tune_profile=\"profiles/gpu1.toml\"\n");
        expect(ecm_config::read_toml(temp.string(),lines),"worker tune paths unavailable");
        expect(ecm_config::read_stage2(ecm_config::cli_entries(lines,1)).tune_profile=="profiles/gpu0.toml" &&
            ecm_config::read_stage2(ecm_config::cli_entries(lines,2)).tune_profile=="profiles/gpu1.toml" &&
            ecm_config::read_stage2(ecm_config::cli_entries(lines,3)).tune_profile=="profiles/shared.toml",
            "worker tune profiles leaked or lost global fallback");
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
        rejected(temp,"[stage1]\ndevice = true\n");
        rejected(temp,"[stage1]\ncurve = 1\n");
        rejected(temp,"[stage2]\nstage2_pack_zero = \"1\"\n");
        rejected(temp,"[stage2]\nstage2_pack_zero = 2\n");
        rejected(temp,"[stage2]\nstage2_pack_zero = 2\n[workers.2.stage2]\nstage2_pack_zero = 0\n");
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
        std::cout<<"TOML parser and generated defaults: PASS; boolean alias checks="<<alias_checks<<"\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
