#include "ecm/platform/fingerprint.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "../../src/ecm/storage/queue/ecm_stage2_queue_state.h"
#include "../../src/ecm/platform/legacy/ecm_stage2_logging.h"
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
        auto changed=a;changed.b2*=2;check(changed.id()!=a.id());
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
        check(loaded.done==3 && loaded.identity==a.id() && loaded.version==2);
        check(std::filesystem::file_size(path)<512);
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
