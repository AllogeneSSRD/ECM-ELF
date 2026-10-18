#include "../../src/core/ecm_submission_writer.h"
#include "../../src/core/ecm_internal_result.h"
#include "../../src/core/ecm_stage2_queue_state.h"
#include <iostream>
static void check(bool ok){if(!ok)throw std::runtime_error("result lifecycle assertion failed");}
int main(int argc,char **argv) {
    try {
        namespace r=ecm_results;namespace fs=std::filesystem;
        if(argc==3 && std::string(argv[1])=="--export-internal"){r::refresh_internal(argv[2]);return 0;}
        if(argc!=2)return 2;const fs::path out=argv[1];fs::create_directories(out);
        ecm_edwards::Integer n;mpz_ui_pow_ui(n.v,2,127);mpz_sub_ui(n.v,n.v,1);
        r::Submission value;value.exponent=127;value.b1=20;value.b2=20;value.stage1_only=true;
        const unsigned sigmas[]={900,70,1000};
        for(unsigned i=0;i<3;++i) {
            value.sigma=sigmas[i];const auto key="synthetic|ed|"+std::to_string(value.sigma)+"|20|stage1";
            r::publish_internal(out/"internal.jsonl",key,r::internal_json(value,n.v,0,"fixture",
                ",\"stage1-engine\":{\"curve\":"+std::to_string(i)+"}"),false);
        }
        r::refresh_internal(out/"internal.jsonl");
        for(unsigned i=0;i<3;++i){value.sigma=sigmas[i];r::publish_submission(out/"stage1.json.txt",value,n.v,false);}
        check(!fs::exists(out/"stage1.json.txt"));
        r::seal_submissions(out/"stage1.json.txt");
        value.stage1_only=false;value.b2=2000;
        for(unsigned i=0;i<3;++i){value.sigma=sigmas[i];r::publish_submission(out/"stage2.json.txt",value,n.v,false);}
        check(!fs::exists(out/"stage2.json.txt"));
        r::seal_submissions(out/"stage2.json.txt");
        // Whole-task NF publication, even if a different task has pending work.
        value.publication_task="task-a";value.sigma=1;r::publish_submission(out/"scoped.json.txt",value,n.v);
        value.publication_task="task-b";value.sigma=2;r::publish_submission(out/"scoped.json.txt",value,n.v);
        r::seal_submissions(out/"scoped.json.txt",0,"task-a");
        {std::ifstream file(out/"scoped.json.txt");std::string line;check(bool(std::getline(file,line)));check(r::json::parse(line).get("curves").text()=="1");check(!std::getline(file,line));}
        // A factor is an immediate individual submission, without sealing NF.
        ecm_edwards::Integer composite;mpz_ui_pow_ui(composite.v,2,29);mpz_sub_ui(composite.v,composite.v,1);
        auto factor=value;factor.exponent=29;factor.stage=1;factor.d=0;factor.poly_size=0;factor.factors={"233"};
        r::publish_submission(out/"factor.json.txt",factor,composite.v);
        check(fs::exists(out/"factor.json.txt"));
        // Migration is atomic and order survives export/replay from the new path.
        fs::rename(r::submission_ledger(out/"internal.jsonl"),r::legacy_submission_ledger(out/"internal.jsonl"));
        r::atomic_text(r::legacy_submission_ledger(out/"internal.jsonl")/"format","ECM_INTERNAL_LEDGER_V1\n");
        r::refresh_internal(out/"internal.jsonl");check(!fs::exists(r::legacy_submission_ledger(out/"internal.jsonl")));
        r::publish_internal(out/"internal.jsonl","synthetic|ed|70|20|stage1","{}",true);
        // Two histories must never be merged or silently replace each other.
        const auto conflict=out/"conflict.jsonl";r::atomic_text(conflict,"unchanged\n");
        for(const auto &dir:{r::submission_ledger(conflict),r::legacy_submission_ledger(conflict)}) {
            fs::create_directories(dir);r::atomic_text(dir/"format","ECM_INTERNAL_LEDGER_V2\n");
        }
        bool refused=false;try{r::refresh_internal(conflict);}catch(const std::runtime_error &){refused=true;}
        check(refused);{std::ifstream file(conflict);std::string line;std::getline(file,line);check(line=="unchanged");}
        // Receipt detection is independent of member order and rejects torn JSON.
        const auto receipt=out/"receipt.jsonl";
        r::atomic_text(receipt,"{\"status\":\"NF\",\"queue_receipt\":\"ready\"}\n{\"queue_receipt\":\"torn\"}");
        check(stage2_queue::result_written(receipt,"ready"));check(!stage2_queue::result_written(receipt,"torn"));
        r::atomic_text(receipt,"{\"queue_receipt\":\"broken\",}\n");check(!stage2_queue::result_written(receipt,"broken"));
        for(const auto &invalid:{"{\"x\":1,\"x\":2}","{\"x\":01}","[true,]","{\"x\":\"\\uD800\"}"}) {
            bool rejected=false;try{(void)r::json::parse(invalid);}catch(const std::runtime_error &){rejected=true;}check(rejected);
        }
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
