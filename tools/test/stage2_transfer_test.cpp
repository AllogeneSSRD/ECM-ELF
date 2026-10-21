#include "../../src/ecm/storage/transfer/ecm_stage2_transfer.h"
#include "../../src/ecm/storage/queue/ecm_worktodo.h"
#include "../../src/ecm/storage/queue/ecm_worker_list.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<std::string> read_lines(const std::string &path) {
    std::ifstream in(path);std::vector<std::string> lines;std::string line;
    require(bool(in),"cannot read test queue");
    while(std::getline(in,line))lines.push_back(line);
    return lines;
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "pass a fresh test directory");
        const fs::path root = argv[1];
        fs::create_directories(root);
        std::vector<int> parsed;int ignored=0;
        require(ecm_workers::parse_list("3,1-2,2",1000000,parsed,ignored) &&
                parsed==std::vector<int>({3,1,2}) && ignored==0,"worker list/range parsing failed");
        require(ecm_workers::parse_list("3-1",1000000,parsed,ignored) && parsed.size()==3,
                "reversed range failed");
        require(!ecm_workers::parse_list("1,wrong",1000000,parsed,ignored),"invalid list accepted");
        require(ecm_workers::parse_list("1-9223372036854775807",8,parsed,ignored) &&
                parsed.size()==8 && ignored>0,"large range not bounded");
        const Stage2TransferConfig cfg{
            (root / "worktodo.txt").string(),
            (root / "stage2_finished.txt").string(),
            (root / "stage1.stage2_pending_worker1").string(), 2};
        const std::string line =
            "ECMSTAGE2=1,2,127,-1,\"C:\\stage1\\e_m127_1000.save\",100000,0,3";
        std::string error, current;
        size_t remaining = 0;
        require(ecm_stage2_transfer_park(cfg, line, error), "park failed");
        require(ecm_stage2_transfer_flush(cfg.pending, remaining, error) && remaining == 0,
                "first delivery failed");
        require(ecm_worktodo_first_line(cfg.worktodo, 2, current) && current == line,
                "wrong worker or line");
        require(!ecm_worktodo_first_line(cfg.worktodo, 1, current), "leaked into worker 1");

        // Simulate a crash after target publication but before pending acknowledgement.
        require(ecm_stage2_transfer_park(cfg, line, error), "second park failed");
        require(ecm_stage2_transfer_flush(cfg.pending, remaining, error), "active replay failed");
        require(ecm_worktodo_advance(cfg.worktodo, 2, line, WorktodoAction::Remove),
                "advance failed");
        require(!ecm_worktodo_first_line(cfg.worktodo, 2, current), "duplicate active task");

        require(ecm_append_text_line(cfg.finished, line), "finished append failed");
        require(ecm_stage2_transfer_park(cfg, line, error), "third park failed");
        require(ecm_stage2_transfer_flush(cfg.pending, remaining, error), "finished replay failed");
        require(!ecm_worktodo_first_line(cfg.worktodo, 2, current), "finished task requeued");

        const std::string error_line = line.substr(0, line.size() - 1) + "2";
        require(ecm_append_text_line(cfg.finished, "# " + error_line), "error receipt append failed");
        require(ecm_stage2_transfer_park(cfg, error_line, error), "error receipt park failed");
        require(ecm_stage2_transfer_flush(cfg.pending, remaining, error), "error receipt replay failed");
        require(!ecm_worktodo_first_line(cfg.worktodo, 2, current), "invalid task requeued");

        const Stage2TransferConfig blocked{
            (root / "blocked").string(), (root / "blocked.finished").string(),
            (root / "blocked.pending").string(), 3};
        const std::string retry_line = line.substr(0, line.size() - 1) + "2";
        fs::create_directory(blocked.worktodo);
        require(ecm_stage2_transfer_park(blocked, retry_line, error), "blocked park failed");
        require(!ecm_stage2_transfer_flush(blocked.pending, remaining, error) && remaining == 1,
                "delivery failure did not preserve pending task");
        fs::remove(blocked.worktodo);
        require(ecm_stage2_transfer_flush(blocked.pending, remaining, error) && remaining == 0,
                "retry after queue recovery failed");
        require(ecm_worktodo_first_line(blocked.worktodo, 3, current) && current == retry_line,
                "retry lost assignment");

        Stage2TransferConfig routed{(root/"routed.txt").string(),(root/"routed.finished").string(),
                                   (root/"routed.pending").string(),1};
        routed.workers={3,2};
        require(ecm_append_text_line(routed.worktodo,"[Worker #2]\n"+line+"\n[Worker #3]"),"seed queues failed");
        int receiver=0;
        require(ecm_stage2_transfer_park(routed,retry_line,error,&receiver) && receiver==3,
                "shortest candidate queue not selected");
        require(ecm_stage2_transfer_flush(routed.pending,remaining,error),"routed flush failed");
        const auto tie_line=retry_line+",\"3\"";
        require(ecm_stage2_transfer_park(routed,tie_line,error,&receiver) && receiver==2,
                "tie did not select lowest worker ID");
        require(ecm_stage2_transfer_flush(routed.pending,remaining,error),"tie flush failed");
        require(ecm_worktodo_first_line(routed.worktodo,3,current) && current==retry_line,
                "receiver task missing");

        Stage2TransferConfig pending_balance{(root/"new-target.txt").string(),(root/"new-target.finished").string(),
                                             (root/"balanced.pending").string(),4};
        pending_balance.auto_workers=true;pending_balance.configured_workers=3;
        require(ecm_stage2_transfer_park(pending_balance,line,error,&receiver) && receiver==1,
                "auto NumWorkers fallback failed");
        require(ecm_stage2_transfer_park(pending_balance,retry_line,error,&receiver) && receiver==2,
                "pending assignments excluded from candidate load");
        require(ecm_stage2_transfer_flush(pending_balance.pending,remaining,error),"balanced flush failed");
        pending_balance.worktodo=(root/"existing-sections.txt").string();
        require(ecm_append_text_line(pending_balance.worktodo,"[Worker #7]\n[Worker #3]"),"seed auto sections failed");
        require(ecm_stage2_transfer_park(pending_balance,line,error,&receiver) && receiver==3,
                "auto ignored empty existing sections");
        require(ecm_stage2_transfer_flush(pending_balance.pending,remaining,error),"auto sections flush failed");
        require(ecm_stage2_transfer_park(pending_balance,retry_line,error,&receiver) && receiver==7,
                "auto shortest existing section failed");
        require(ecm_stage2_transfer_flush(pending_balance.pending,remaining,error),"auto receiver flush failed");

        Stage2TransferConfig receipts{(root/"receipts.txt").string(),(root/"global.finished").string(),
                                      (root/"receipts.pending").string(),2};
        receipts.finished_by_worker[2]=(root/"worker2.finished").string();
        require(ecm_append_text_line(receipts.finished_by_worker[2],line),"seed receiver receipt failed");
        require(ecm_stage2_transfer_park(receipts,line,error) &&
                ecm_stage2_transfer_flush(receipts.pending,remaining,error),"receiver receipt retry failed");
        require(!ecm_worktodo_first_line(receipts.worktodo,2,current),"receiver-specific receipt ignored");
        receipts.finished_by_worker[2].clear();
        require(!ecm_stage2_transfer_park(receipts,retry_line,error),"empty receiver receipt path accepted");

        // Real handoffs reuse the destination section and retain task FIFO.
        const std::string first="ECMSTAGE2=1,2,127,-1,\"first.save\",100000,0,1";
        const std::string second="ECMSTAGE2=1,2,127,-1,\"second.save\",100000,0,1";
        const std::string third="ECMSTAGE2=1,2,127,-1,\"third.save\",100000,0,1";
        auto layout_case=[&](const char *name,int worker,const std::string &seed,
                             const std::vector<std::string> &expected) {
            Stage2TransferConfig target{(root/(std::string(name)+".txt")).string(),
                (root/(std::string(name)+".finished")).string(),
                (root/(std::string(name)+".pending")).string(),worker};
            if(!seed.empty())require(ecm_append_text_line(target.worktodo,seed),"seed layout failed");
            require(ecm_stage2_transfer_park(target,third,error),"layout park failed");
            require(ecm_stage2_transfer_flush(target.pending,remaining,error),"layout flush failed");
            require(read_lines(target.worktodo)==expected,name);
            require(ecm_stage2_transfer_park(target,third,error) &&
                    ecm_stage2_transfer_flush(target.pending,remaining,error),"layout replay failed");
            require(read_lines(target.worktodo)==expected,"layout replay changed queue");
        };
        layout_case("reuse_empty_section",2,"# keep\n[Worker #2]",
                    {"# keep","[Worker #2]",third});
        layout_case("reuse_middle_section",2,"[Worker #1]\n"+first+"\n[Worker #2]\n"+second+"\n# receiver note\n[Worker #3]\n"+first,
                    {"[Worker #1]",first,"[Worker #2]",second,"# receiver note",third,"[Worker #3]",first});
        layout_case("compact_production_shape",2,"[Worker #1]\n"+first+"\n[Worker #2]\n[Worker #2]\n"+second+"\n# keep comment\n[Worker #2]",
                    {"[Worker #1]",first,"[Worker #2]",second,"# keep comment",third});
        layout_case("keep_distinct_sections",2,"[Worker #2]\n"+first+"\n[Worker #3]\n"+first+"\n[Worker #2]\n"+second,
                    {"[Worker #2]",first,"[Worker #3]",first,"[Worker #2]",second,third});
        layout_case("keep_foreign_boundary",2,"[Worker #2]\n[Other]\n[Worker #2]",
                    {"[Worker #2]","[Other]","[Worker #2]",third});
        layout_case("implicit_worker_one",1,"# keep\n"+first,
                    {"# keep",first,third});
        layout_case("insert_implicit_worker_one",1,"# before header\n[Worker #2]\n"+first,
                    {"# before header",third,"[Worker #2]",first});
        layout_case("create_missing_section",4,"[Worker #2]\n"+first,
                    {"[Worker #2]",first,"[Worker #4]",third});

        Stage2TransferConfig repeated{(root/"repeated.txt").string(),(root/"repeated.finished").string(),
                                      (root/"repeated.pending").string(),2};
        for(const auto &task:{first,second,third}) {
            require(ecm_stage2_transfer_park(repeated,task,error) &&
                    ecm_stage2_transfer_flush(repeated.pending,remaining,error),"repeated handoff failed");
            require(ecm_worktodo_advance(repeated.worktodo,2,task,WorktodoAction::Remove),"repeated consume failed");
        }
        require(read_lines(repeated.worktodo)==std::vector<std::string>{"[Worker #2]"},
                "consumed handoffs accumulated empty headers");
        std::cout << "stage2 transfer recovery OK\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
