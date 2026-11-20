#include "io/stage1_checkpoint.h"
#include "ecm/io/results.h"
#include "ecm/io/stage2_checkpoint.h"
#include "ecm/platform/fingerprint.h"
#include "ecm/io/save.h"
#include "io/json.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace fs=std::filesystem;
int main(int argc,char **argv) {
    try {
        if(argc!=7)throw std::runtime_error("out save schema done total B2 required");
        const fs::path out=fs::absolute(argv[1]);if(fs::exists(out))throw std::runtime_error("fresh output required");
        fs::create_directories(out);const auto save=out/"e_m521_1000.save";fs::copy_file(argv[2],save);
        const std::string schema=argv[3];const uint64_t done=std::stoull(argv[4]),total=std::stoull(argv[5]),b2=std::stoull(argv[6]);
        ecm_edwards::Integer n;mpz_ui_pow_ui(n.v,2,521);mpz_sub_ui(n.v,n.v,1);
        stage2_queue::Task task;task.n=ecm_edwards::hex(n.v);task.save_sha256=ecm_stage2::sha256_file(save);task.total=total;task.b2=b2;
        task.b1=1000;
        const auto publisher=schema=="v1" || schema=="v2"?task.legacy_id():task.id();
        std::ifstream in(save);std::ofstream internal(out/"stage2_results.jsonl",std::ios::binary);std::string line;
        const bool pending=schema=="pending" || schema=="unconfirmed" || schema=="torn_internal";
        const auto receipt=std::string(32,'e');
        const auto count=done+(schema=="v4"?2:schema=="pending");
        stage2_queue::State current;current.version=5;current.reconfigure(task);
        current.submission_path=(out/"results.json.txt").u8string();
        for(uint64_t i=0;i<count;++i) {
            if(!std::getline(in,line))throw std::runtime_error("insufficient save rows");
            if(schema=="v4" && i==done)continue;
            const auto at=line.find("SIGMA=");if(at==line.npos)throw std::runtime_error("missing sigma");
            ecm_results::Submission r;r.exponent=521;r.b1=1000;r.b2=b2;r.sigma=std::stoull(line.substr(at+6));
            r.when=ecm_results::utc_timestamp();
            const auto report=ecm_results::json::parse(ecm_results::submission_json(r,n.v)).dump();
            current.record_outcome(i,report,ecm_stage2::sha256_text(task.n+"|"+std::to_string(r.sigma)+"|1000|"+std::to_string(b2)+"|ed"));
            internal<<ecm_results::internal_json(r,n.v,0,"checkpoint-override-fixture",
                ",\"queue_receipt\":"+ecm_results::quoted(i==done?receipt:"seed-"+std::to_string(i))+
                ",\"stage2-engine\":{\"seeded_resume_fixture\":true}");
        }
        internal.close();if(!internal)throw std::runtime_error("cannot write seeded internal results");
        if(schema=="sealed") {
            current.final_nf=ecm_results::aggregate_nf(current.latest_nf,current.nf_keys.size());
            std::ofstream exported(out/"results.json.txt",std::ios::binary);
            exported<<current.final_nf<<'\n';
        }
        const auto progress_dir=out/"progress";fs::create_directories(progress_dir);
        fs::path checkpoint;
        if(schema=="v1") {
            checkpoint=out/"worktodo.txt.progress";
            std::ostringstream identity;
            identity<<std::quoted("ECMSTAGE2=1,2,521,-1,\"e_m521_1000.save\",10000,0,960")
                <<" \"\" \"\" \"\" "<<task.save_sha256<<' '<<std::quoted((out/"stage2_results.jsonl").u8string())
                <<" 0 "<<total<<' '<<b2<<" 0 0";
            std::ofstream state(checkpoint);
            state<<"1 "<<done<<' '<<total<<" \"legacy-fixture-run\" \"\" "<<std::quoted(identity.str())<<"\n";
        } else if(schema=="v2") {
            checkpoint=progress_dir/(".ecm_ckpt_stage2_521_"+task.legacy_id()+".dat");
            std::ofstream state(checkpoint);state<<"2 "<<done<<' '<<total<<" \"legacy-fixture-run\" \"\" "<<std::quoted(task.legacy_id())<<" \"\"\n";
        } else {
            stage2_queue::State state;state.version=3;state.reconfigure(task);state.done=done;
            if(pending){state.pending=receipt;state.pending_results=(out/"stage2_results.jsonl").string();}
            if(schema=="v4")state.reserved={{done,false,receipt,(out/"stage2_results.jsonl").u8string()},
                {done+1,true,"second",(out/"stage2_results.jsonl").u8string()}};
            checkpoint=task.legacy_path(progress_dir,521);
            if(schema=="v5" || schema=="sealed") {
                current.done=done;
                if(schema=="sealed")current.total=done;
                current.save(checkpoint);
            } else state.save(checkpoint);
        }
        std::cout<<"{\"publication_task\":"<<ecm_results::quoted(publisher)<<",\"checkpoint\":"<<ecm_results::quoted(checkpoint.u8string())<<"}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
