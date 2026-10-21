#include "../../src/ecm/core/curve/ecm_edwards_curve.h"
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <filesystem>
static std::filesystem::path data_ledger(const std::filesystem::path &p){return p.parent_path()/"data"/(p.filename().string()+".ecm-ledger");}
static std::filesystem::path legacy_ledger(const std::filesystem::path &p){return std::filesystem::path(p.string()+".ecm-ledger");}
static void fixture_text(const std::filesystem::path &p,const std::string &text){std::ofstream out(p,std::ios::binary|std::ios::trunc);out<<text;out.close();if(!out)throw std::runtime_error("cannot write fixture input");}
#include "ecm/storage/results/publication.h"
#include "ecm/storage/queue/stage2_progress.h"
#include "../../src/ecm/storage/results/ecm_result_json.h"
#include <iostream>
static void check(bool ok){if(!ok)throw std::runtime_error("result lifecycle assertion failed");}
static std::string read_bytes(const std::filesystem::path &path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot read fixture output");
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
int main(int argc,char **argv) {
    try {
        namespace r=ecm_results;namespace fs=std::filesystem;
        if(argc==3 && std::string(argv[1])=="--export-internal"){r::refresh_internal(argv[2]);return 0;}
        if(argc==3 && std::string(argv[1])=="--export-submission"){r::refresh_submission_if_owned(argv[2]);return 0;}
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
        // Stage2 task aggregation ignores per-curve geometry and uses the final
        // completed curve, regardless of receipt hash order.
        value.publication_task="geometry-task";value.b2=2000;value.d=30030;value.poly_size=96;
        for(unsigned i=0;i<3;++i) {
            value.sigma=sigmas[i];value.publication_order=i+1;
            value.when="2026-10-20 12:00:0"+std::to_string(i);
            value.d=30030+2*i;value.poly_size=96+32*i;
            r::publish_submission(out/"geometry.json.txt",value,n.v);
        }
        check(!fs::exists(out/"geometry.json.txt"));
        r::seal_submissions(out/"geometry.json.txt",2,"geometry-task",true);
        {std::ifstream file(out/"geometry.json.txt");std::string line;check(bool(std::getline(file,line)));
            const auto row=r::json::parse(line);check(row.get("curves").text()=="3" && row.get("d").text()=="30034" &&
                row.get("poly-size").text()=="160" && row.get("timestamp").text()=="2026-10-20 12:00:02");
            check(!std::getline(file,line));}
        // A factor is an immediate individual submission, without sealing NF.
        ecm_edwards::Integer composite;mpz_ui_pow_ui(composite.v,2,29);mpz_sub_ui(composite.v,composite.v,1);
        auto factor=value;factor.exponent=29;factor.stage=1;factor.d=0;factor.poly_size=0;factor.factors={"233"};
        r::publish_submission(out/"factor.json.txt",factor,composite.v);
        check(fs::exists(out/"factor.json.txt"));
        auto mixed=factor;mixed.publication_task="mixed-task";
        r::publish_submission(out/"mixed.json.txt",mixed,composite.v);
        {std::ifstream file(out/"mixed.json.txt");std::string line;check(bool(std::getline(file,line)));
            check(r::json::parse(line).get("status").text()=="F" && !std::getline(file,line));}
        mixed.factors.clear();mixed.stage=0;mixed.sigma=999;mixed.publication_order=2;
        mixed.d=30030;mixed.poly_size=96;
        r::publish_submission(out/"mixed.json.txt",mixed,composite.v);
        r::seal_submissions(out/"mixed.json.txt",2,"mixed-task",true);
        {std::ifstream file(out/"mixed.json.txt");std::string line;
            check(bool(std::getline(file,line)) && r::json::parse(line).get("status").text()=="F");
            check(bool(std::getline(file,line)) && r::json::parse(line).get("status").text()=="NF");}
        // Both stages preserve every published byte and append across tasks,
        // even when result timestamps run backwards. A hard link detects replace.
        for(int stage:{1,2}) {
            const auto path=out/("append"+std::to_string(stage)+".json.txt");
            const auto alias=out/("append"+std::to_string(stage)+".linked.txt");
            auto next=value;next.stage1_only=stage==1;next.b2=stage==1?20:2000;
            if(stage==1){next.d=0;next.poly_size=0;}
            for(unsigned i=0;i<3;++i) {
                const auto before=fs::exists(path)?read_bytes(path):std::string{};
                next.sigma=100+i;next.publication_task="append-task-"+std::to_string(i);
                next.when="2026-10-21 12:00:0"+std::to_string(2-i);
                r::publish_submission(path,next,n.v);
                r::seal_submissions(path,stage,next.publication_task,true);
                const auto after=read_bytes(path);check(after.compare(0,before.size(),before)==0);
                if(i==0)fs::create_hard_link(path,alias);
                check(read_bytes(alias)==after);
                r::publish_submission(path,next,n.v);
                r::seal_submissions(path,stage,next.publication_task,true);
                r::refresh_submission_if_owned(path);check(read_bytes(path)==after);
            }
        }
        // NF, then F, then NF remains interleaved in publication order.
        auto alternating=mixed;alternating.publication_task="alternating-a";alternating.sigma=1100;
        const auto alternating_path=out/"alternating.json.txt";
        r::publish_submission(alternating_path,alternating,composite.v);
        r::seal_submissions(alternating_path,2,alternating.publication_task,true);
        alternating.factors={"233"};alternating.stage=1;alternating.sigma=1101;
        alternating.d=0;alternating.poly_size=0;
        r::publish_submission(alternating_path,alternating,composite.v);
        alternating.factors.clear();alternating.stage=0;alternating.sigma=1102;
        alternating.d=30030;alternating.poly_size=96;
        alternating.publication_task="alternating-b";alternating.when="2026-10-20 11:00:00";
        r::publish_submission(alternating_path,alternating,composite.v);
        r::seal_submissions(alternating_path,2,alternating.publication_task,true);
        // Migration is atomic and order survives export/replay from the new path.
        fs::rename(data_ledger(out/"internal.jsonl"),legacy_ledger(out/"internal.jsonl"));
        fixture_text(legacy_ledger(out/"internal.jsonl")/"format","ECM_INTERNAL_LEDGER_V1\n");
        r::refresh_internal(out/"internal.jsonl");check(!fs::exists(legacy_ledger(out/"internal.jsonl")));
        r::publish_internal(out/"internal.jsonl","synthetic|ed|70|20|stage1","{}",true);
        // Two histories must never be merged or silently replace each other.
        const auto conflict=out/"conflict.jsonl";fixture_text(conflict,"unchanged\n");
        for(const auto &dir:{data_ledger(conflict),legacy_ledger(conflict)}) {
            fs::create_directories(dir);fixture_text(dir/"format","ECM_INTERNAL_LEDGER_V2\n");
        }
        bool refused=false;try{r::refresh_internal(conflict);}catch(const std::runtime_error &){refused=true;}
        check(refused);{std::ifstream file(conflict);std::string line;std::getline(file,line);check(line=="unchanged");}
        // Receipt detection is independent of member order and rejects torn JSON.
        const auto receipt=out/"receipt.jsonl";
        fixture_text(receipt,"{\"status\":\"NF\",\"queue_receipt\":\"ready\"}\n{\"queue_receipt\":\"torn\"}");
        check(stage2_queue::result_written(receipt,"ready"));check(!stage2_queue::result_written(receipt,"torn"));
        fixture_text(receipt,"{\"queue_receipt\":\"broken\",}\n");check(!stage2_queue::result_written(receipt,"broken"));
        for(const auto &invalid:{"{\"x\":1,\"x\":2}","{\"x\":01}","[true,]","{\"x\":\"\\uD800\"}"}) {
            bool rejected=false;try{(void)r::json::parse(invalid);}catch(const std::runtime_error &){rejected=true;}check(rejected);
        }
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
