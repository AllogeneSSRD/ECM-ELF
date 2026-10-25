#include "io/worktodo.h"
#include "sync/files.h"
#include "sync/p95.h"
#include "ecm/storage/save/publication.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
namespace fs=std::filesystem;
static void check(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
static void put(const fs::path &p,const std::string &s){ecm_sync::write_atomic(p,s);}
static std::string get(const fs::path &p){return ecm_sync::read_file(p);}
static std::string point(unsigned sigma){return "METHOD=ECM; SIGMA="+std::to_string(sigma)+"; B1=100; N=127; X=0x1; CHECKSUM="+std::to_string(12700ull*sigma)+"; PROGRAM=fixture;\n";}
int main(int argc,char **argv)try {
    check(argc==2,"supply a fresh evidence directory");const auto root=fs::absolute(fs::u8path(argv[1]));
    check(!fs::exists(root),"evidence directory must be fresh");fs::create_directories(root);
    std::string error,line;Ecm2Task ecm;EcmStage2Task task;
    check(!ecm_parse_ecm2_line("ECM=1,2,7,-1,100,wrong,3",ecm,error) && error.find("B2")!=error.npos,"B2 diagnostic");
    check(!ecm_parse_ecm2_line("ECM=1,2,7,-1,100,0,-1",ecm,error),"negative curves accepted");
    check(!ecm_parse_ecm2_line("ECM=1,2,7,-1,100,0,3,18446744073709551616",ecm,error),"sigma overflow accepted");
    check(ecm_parse_ecm2_line("ECM=1,2,7,-1,100,0,3,0",ecm,error) && !ecm.has_sigma,"worktodo sigma zero semantics");
    check(ecm_parse_ecm2_line("ECM=1,2,7,-1,100",ecm,error) && !ecm.has_curves && ecm.curves_to_run==0,"parser applied omitted curve default");
    check(!ecm_parse_ecm2_line("ECM=1,2,7,-1,100,0,3,1,\"broken",ecm,error),"unterminated CSV accepted");
    const std::string row="ECMSTAGE1=1,2,7,-1,\"m0_name,part_100.save\",1e4,0,3,\"7,31\"";
    check(ecm_rewrite_stage2_range(row,150,2,line,error),"range rewrite");
    check(ecm_parse_stage1_line(line,task,error) && task.skip_curves==150 && task.curves_to_run==2 && task.save_name=="m0_name,part_100.save" && task.factors.size()==2,"CSV range preservation");
    check(!ecm_parse_stage1_line("ECMSTAGE1=1,2,7,-1,m0_100.save,0,invalid,3",task,error),"invalid skip accepted");
    const auto queue=root/"worktodo.txt";put(queue,"# keep\n[Worker #1]\n"+row+"\n[Worker #2]\n"+row+"\n");
    // Reorder/insert future work while the current task is running.
    put(queue,"# changed by external writer\n[Worker #1]\nECM=1,2,11,-1,100\n"+row+"\n"+row+"\n[Worker #2]\n"+row+"\n");
    check(ecm_worktodo_advance(queue.string(),1,row,WorktodoAction::Replace,line),"first identical row matching");
    auto text=get(queue);check(text.find(line)<text.find(row),"wrong duplicate replaced");
    check(text.find("[Worker #2]\n"+row)!=text.npos && text.find("ECM=1,2,11,-1,100")!=text.npos,"foreign/future rows changed");
    const auto receipt=root/"completion";CompletedWorktodo completed{line,line,"",(root/"finished.txt").string(),"test.ckpt"};
    check(ecm_worktodo_store_completion(receipt.string(),completed,error),"store completion");
    check(ecm_worktodo_commit_completion(receipt.string(),queue.string(),1,completed,error),"commit completion");
    const auto advanced=get(queue),finished=get(root/"finished.txt");
    check(ecm_worktodo_commit_completion(receipt.string(),queue.string(),1,completed,error),"resume committed completion");
    check(get(queue)==advanced && get(root/"finished.txt")==finished,"completion replay consumed another duplicate");
    put(queue,advanced+"# edit after crash\n");
    check(!ecm_worktodo_commit_completion(receipt.string(),queue.string(),1,completed,error),"ambiguous recovery must stop");
    const auto save=root/"m0_m7_100.save";put(save,"# old history\n"+point(10)+point(11));
    ecm_save::Range range;
    {ecm_save::Capture capture;check(ecm_save::publish(save.string(),point(21)+point(20)+point(22),true),"publish append");range=capture.range;}
    check(range.valid && range.skip==2 && range.records.size()==3 && ecm_save::record_count(save.string())==5,"actual batch range");
    const auto snapshot=ecm_save::verified_snapshot(range);
    const auto original_range=range;
    {ecm_save::Capture capture;check(ecm_save::publish(save.string(),point(22)+point(20)+point(21),true),"deduplicated recovery");check(capture.range.skip==2 && capture.range.records==range.records,"recovered range differs");}
    check(get(save)==snapshot,"recovery rewrote history");
    check(!ecm_save::publish(save.string(),point(11)+point(20)+point(30),true) && get(save)==snapshot,"noncontiguous batch must not publish");
    check(ecm_save::record_count((root/"absent.save").string())==0,"missing save count");
    put(root/"opaque.jsonl","{\"sigma\":9}\n");
    check(ecm_save::publish((root/"opaque.jsonl").string(),"{\"sigma\":3}\n",true),"opaque publisher compatibility");
    P95TransferConfig p95;p95.worktodo_path=(root/"p95/worktodo.txt").string();p95.sync_dir=(root/"p95/saves").string();
    p95.pending_path=(root/"p95_pending.txt").string();p95.sent_ledger_path=(root/"sent.txt").string();p95.recover_lost=false;
    fs::create_directories(fs::u8path(p95.sync_dir));const auto target=fs::u8path(p95.sync_dir)/save.filename();fs::create_directory(target);
    std::string handoff="ECMSTAGE2=1,2,7,-1,\""+save.string()+"\",10000,2,3";
    const auto original_handoff=handoff;
    check(ecm_sync::park_p95(p95,handoff,range,error),"persist P95 responsibility");
    check(!ecm_sync::flush_p95(p95,error) && !fs::exists(root/"p95/worktodo.add"),"file failure exposed P95 task");
    fs::remove(target);
    auto changed_configuration=p95;changed_configuration.worktodo_path=(root/"other-p95/worktodo.txt").string();
    check(ecm_sync::flush_p95(changed_configuration,error),error.c_str());
    check(!fs::exists(root/"other-p95/worktodo.add"),"restart rerouted a pending handoff using changed config");
    check(get(target)==snapshot,"P95 save is not the validated version");
    const auto add=get(root/"p95/worktodo.add");
    check(ecm_parse_stage2_line(add,task,error) && fs::equivalent(fs::u8path(task.save_name),target),"P95 task points at local save");
    check(ecm_sync::flush_p95(p95,error) && get(root/"p95/worktodo.add")==add,"P95 replay duplicated task");
    // A later batch extends the target; an old pending snapshot cannot truncate it.
    {ecm_save::Capture capture;check(ecm_save::publish(save.string(),point(30),true),"second batch");range=capture.range;}
    handoff="ECMSTAGE2=1,2,7,-1,\""+save.string()+"\",10000,5,1";
    check(ecm_sync::park_p95(p95,handoff,range,error) && ecm_sync::flush_p95(p95,error),"append-only P95 update");
    check(get(target).compare(0,snapshot.size(),snapshot)==0,"published P95 prefix changed");
    const auto delivered=get(root/"p95/worktodo.add");
    check(ecm_sync::park_p95(changed_configuration,original_handoff,original_range,error) && ecm_sync::flush_p95(changed_configuration,error),"replay after concurrent save growth");
    check(get(root/"p95/worktodo.add")==delivered && !fs::exists(root/"other-p95/worktodo.add"),"save growth/config change duplicated an old batch");
    const auto damaged=root/"damaged_100.save";auto damaged_point=point(99);
    const auto checksum_at=damaged_point.find("CHECKSUM="),checksum_end=damaged_point.find(';',checksum_at);
    damaged_point.replace(checksum_at,checksum_end-checksum_at,"CHECKSUM=0");put(damaged,damaged_point);
    const auto bad_range=ecm_save::inspect_range(damaged.string(),0,1);
    const auto bad_line="ECMSTAGE2=1,2,7,-1,\""+damaged.string()+"\",10000,0,1";
    check(!ecm_sync::park_p95(p95,bad_line,bad_range,error) && error.find("CHECKSUM")!=error.npos,"corrupt legacy checksum accepted for P95");
    check(get(root/"p95/worktodo.add")==delivered,"corrupt save exposed P95 work");
    const auto backup=root/"backup";
    ecm_sync::backup_saves(root.string(),"",backup.string(),true,0);
    check(get(backup/save.filename())==get(save),"second backup destination lost save bytes");
    check(!fs::exists(backup/"opaque.jsonl"),"backup copied a non-save file");
    const auto blocked=root/"blocked_backup";put(blocked,"not a directory");
    const auto fallback=root/"fallback_backup";
    ecm_sync::backup_saves(root.string(),blocked.string(),fallback.string(),true,0);
    check(get(fallback/save.filename())==get(save),"first backup failure blocked the second destination");
    std::cout<<"host_io_contract=passed parser, live_queue, completion_recovery, actual_range, deduplication, file_before_task, retry, stable_prefix, backup\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
