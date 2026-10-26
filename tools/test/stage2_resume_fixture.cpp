#include "ecm/platform/fingerprint.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "ecm/io/stage2_checkpoint.h"
#include "../../src/ecm/platform/compute_log.h"
#include <iostream>

int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("provide fresh output directory");
        const std::filesystem::path dir=argv[1];std::filesystem::create_directories(dir);
        unsigned checks=0;
        auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("resume fixture assertion failed: "+std::to_string(checks));};
        check(ecm_stage2::sha256_text("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        stage2_queue::Task a{"AID-A","abcdef",std::string(64,'b'),336,144,44000000000ull,false};
        auto b=a;b.aid="AID-B";check(a.id()!=b.id());
        auto changed=a;changed.b2*=2;check(changed.id()==a.id());
        changed=a;changed.total=300;check(changed.id()==a.id());
        changed=a;++changed.skip;check(changed.id()!=a.id());
        changed=a;changed.save_sha256=std::string(64,'c');check(changed.id()!=a.id());
        const auto path=a.path(dir,11694);check(path.filename().string().find(".ecm_ckpt_stage2_11694_")==0);
        std::ostringstream legacy_identity;
        legacy_identity<<std::quoted("row A")<<' '<<std::quoted("old queue path")<<' '<<std::string(64,'d')<<' '
            <<std::quoted("old save path")<<' '<<a.save_sha256<<' '<<std::quoted((dir/"old_results.jsonl").string())
            <<" 336 480 44000000000 0 0 0 1 6400 64 960 1 0 1";
        const auto legacy=dir/"worktodo.txt.progress";
        {std::ofstream out(legacy);out<<"1 3 144 "<<std::quoted(std::string(32,'a'))<<' '
            <<std::quoted(std::string(32,'e'))<<' '<<std::quoted(legacy_identity.str())<<'\n';}
        stage2_queue::State state;
        check(!stage2_queue::import_legacy(legacy,b,[&](const std::string &s){return b.aid=="AID-A" && s=="row A";},state));
        check(stage2_queue::import_legacy(legacy,a,[](const std::string &s){return s=="row A";},state));
        check(state.done==3 && state.total==144 && state.pending==std::string(32,'e'));
        check(state.pending_results==(dir/"old_results.jsonl").string());
        state.save(path);stage2_queue::State loaded;check(loaded.load(path));
        check(loaded.done==3 && loaded.identity==a.id() && loaded.version==3);
        check(loaded.publication_task==a.legacy_id() && loaded.b2==a.b2);
        check(std::filesystem::file_size(path)<1024);
        const auto receipt=dir/"old_results.jsonl";
        {std::ofstream out(receipt,std::ios::binary);out<<"{\"queue_receipt\":\""<<loaded.pending<<"\",\"clean\":true}";}
        check(!stage2_queue::result_written(receipt,loaded.pending));
        {std::ofstream out(receipt,std::ios::binary|std::ios::app);out<<'\n';}
        check(stage2_queue::result_written(receipt,loaded.pending));
        ++loaded.done;loaded.pending.clear();loaded.pending_results.clear();loaded.save(path);
        stage2_queue::State again;check(again.load(path) && again.done==4 && again.pending.empty());
        {std::ofstream out(legacy);out<<"1 144 144 \"x\" \"pending\" \"bad\"\n";}
        bool refused=false;try{stage2_queue::State invalid;invalid.load(legacy);}catch(const std::runtime_error &){refused=true;}
        check(refused);
        // Requested policy is mutable, while input and publication ownership stay stable.
        auto policy=a;policy.skip=0;policy.total=960;policy.b2=10000;
        stage2_queue::State running;running.reconfigure(policy);running.done=150;
        const auto run=running.run,publisher=running.publication_task;
        const auto policy_dir=dir/"policy";running.save(policy.path(policy_dir,11694));
        auto requested=policy;requested.total=300;requested.b2=20000;
        stage2_queue::State resumed;
        check(stage2_queue::load_task(policy_dir,11694,requested,resumed)==policy.path(policy_dir,11694));
        resumed.reconfigure(requested);
        check(resumed.done==150 && resumed.total==300 && resumed.b2==20000 && resumed.run==run && resumed.publication_task==publisher);
        requested.total=100;resumed.reconfigure(requested);
        check(resumed.done==150 && resumed.total==150 && resumed.pending.empty());
        resumed.save(requested.path(policy_dir,11694));
        stage2_queue::State terminal;check(terminal.load(requested.path(policy_dir,11694)) && terminal.done==150 && terminal.total==150);
        terminal.reconfigure(requested);check(terminal.done==150 && terminal.total==150 && terminal.publication_task==publisher);
        requested.total=1200;terminal.reconfigure(requested);check(terminal.total==1200 && terminal.done==150);
        requested.b2=0;requested.auto_b2=true;terminal.reconfigure(requested);check(terminal.auto_b2 && !terminal.b2);
        terminal.pending="not-yet-confirmed";
        auto changed_policy=requested;changed_policy.total=100;
        refused=false;try{terminal.reconfigure(changed_policy);}catch(const std::runtime_error &){refused=true;}
        check(refused && terminal.total==1200 && terminal.pending=="not-yet-confirmed");
        // A complete receipt is acknowledged before applying a smaller target.
        {std::ofstream out(receipt,std::ios::binary);out<<"{\"queue_receipt\":\"not-yet-confirmed\"}\n";}
        check(stage2_queue::result_written(receipt,terminal.pending));
        ++terminal.done;terminal.pending.clear();terminal.reconfigure(changed_policy);
        check(terminal.done==151 && terminal.total==151);
        auto write_v2=[&](const std::filesystem::path &folder,const stage2_queue::Task &task) {
            std::filesystem::create_directories(folder);
            const auto p=folder/(".ecm_ckpt_stage2_11694_"+task.legacy_id()+".dat");
            std::ofstream out(p);out<<"2 150 "<<task.total<<" \"legacy-run\" \"\" "<<std::quoted(task.legacy_id())<<" \"\"\n";
            return p;
        };
        const auto v2_dir=dir/"v2_changed_bounds";const auto old_path=write_v2(v2_dir,policy);
        requested=policy;requested.total=300;requested.b2=20000;unsigned bound_reads=0;
        stage2_queue::State converted;
        check(stage2_queue::load_task(v2_dir,11694,requested,converted,[&]{++bound_reads;return std::vector<uint64_t>{10000};})==old_path);
        check(bound_reads==1 && converted.done==150 && converted.total==960 && converted.b2==10000 && converted.publication_task==policy.legacy_id());
        converted.reconfigure(requested);converted.save(requested.path(v2_dir,11694));
        check(converted.total==300 && converted.b2==20000 && converted.publication_task==policy.legacy_id());
        stage2_queue::State fast;
        check(stage2_queue::load_task(v2_dir,11694,requested,fast,[&]()->std::vector<uint64_t>{throw std::runtime_error("unexpected legacy scan");})==old_path);
        check(fast.done==150 && fast.b2==20000);
        std::filesystem::rename(old_path,std::filesystem::path(old_path.string()+".migrated.fixture"));
        check(stage2_queue::load_task(v2_dir,11694,requested,fast)==requested.path(v2_dir,11694));
        const auto v2_count_dir=dir/"v2_changed_count";write_v2(v2_count_dir,policy);
        auto smaller=policy;smaller.total=100;stage2_queue::State overrun;
        check(!stage2_queue::load_task(v2_count_dir,11694,smaller,overrun).empty());
        overrun.reconfigure(smaller);check(overrun.done==150 && overrun.total==150);
        const auto ambiguous_dir=dir/"ambiguous";write_v2(ambiguous_dir,policy);
        auto other=policy;other.total=500;other.b2=30000;write_v2(ambiguous_dir,other);
        refused=false;try{stage2_queue::State ambiguous;stage2_queue::load_task(ambiguous_dir,11694,requested,ambiguous,[]{return std::vector<uint64_t>{10000,30000};});}
        catch(const std::runtime_error &){refused=true;}check(refused);
        auto unrelated=requested;unrelated.save_sha256=std::string(64,'c');stage2_queue::State empty;
        check(stage2_queue::load_task(v2_count_dir,11694,unrelated,empty).empty() && empty.identity.empty());
        stage2_log::level=stage2_log::phases;
        stage2_log::phase("Read and validate Stage1 point");
        {stage2_log::PlanningScope quiet;stage2_log::phase("Plan Stage2 bounds and memory");
            stage2_log::print(stage2_log::phases,"candidate information must stay quiet\n");
            {stage2_log::PlanningScope nested;stage2_log::phase("Plan Stage2 bounds and memory");}}
        check(stage2_log::level==stage2_log::phases && stage2_log::phase_silence==0);
        stage2_log::phase("Plan Stage2 bounds and memory");
        std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"gpu_curves\":0}\n";
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
