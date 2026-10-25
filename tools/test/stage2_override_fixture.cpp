#include "io/stage1_checkpoint.h"
#include "ecm/io/results.h"
#include "ecm/io/stage2_checkpoint.h"
#include "ecm/platform/fingerprint.h"
#include "ecm/io/save.h"
#include <fstream>
#include <iomanip>
#include <iostream>
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
        const auto publisher=schema=="v2"?task.legacy_id():task.id();
        std::ifstream in(save);std::ofstream internal(out/"stage2_results.jsonl",std::ios::binary);std::string line;
        const bool pending=schema=="pending" || schema=="unconfirmed" || schema=="torn_internal";
        const auto receipt=std::string(32,'e');const auto count=done+(schema=="pending" || schema=="torn_internal");
        for(uint64_t i=0;i<count;++i) {
            if(!std::getline(in,line))throw std::runtime_error("insufficient save rows");
            const auto at=line.find("SIGMA=");if(at==line.npos)throw std::runtime_error("missing sigma");
            ecm_results::Submission r;r.exponent=521;r.b1=1000;r.b2=b2;r.sigma=std::stoull(line.substr(at+6));
            r.publication_task=publisher;r.publication_order=i+1;r.save_name=save.stem().u8string();
            r.when=ecm_results::utc_timestamp();
            ecm_results::publish_submission(out/"results.json.txt",r,n.v,false);
            if(schema=="torn_internal" && i==done)continue;
            internal<<ecm_results::internal_json(r,n.v,0,"checkpoint-override-fixture",
                ",\"queue_receipt\":"+ecm_results::quoted(i==done?receipt:"seed-"+std::to_string(i))+
                ",\"stage2-engine\":{\"seeded_resume_fixture\":true}");
        }
        internal.close();if(!internal)throw std::runtime_error("cannot write seeded internal results");
        const auto progress_dir=out/"progress";fs::create_directories(progress_dir);
        fs::path checkpoint;
        if(schema=="v2") {
            checkpoint=progress_dir/(".ecm_ckpt_stage2_521_"+task.legacy_id()+".dat");
            std::ofstream state(checkpoint);state<<"2 "<<done<<' '<<total<<" \"legacy-fixture-run\" \"\" "<<std::quoted(task.legacy_id())<<" \"\"\n";
        } else {
            stage2_queue::State state;state.reconfigure(task);state.done=done;
            if(pending){state.pending=receipt;state.pending_results=(out/"stage2_results.jsonl").string();}
            checkpoint=task.path(progress_dir,521);state.save(checkpoint);
        }
        std::cout<<"{\"publication_task\":"<<ecm_results::quoted(publisher)<<",\"checkpoint\":"<<ecm_results::quoted(checkpoint.u8string())<<"}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
