#include "ecm_stage2_transfer.h"
#include "ecm_worktodo.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "pass a fresh test directory");
        const fs::path root = argv[1];
        fs::create_directories(root);
        const Stage2TransferConfig cfg{
            (root / "stage2_worktodo.txt").string(),
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
        std::cout << "stage2 transfer recovery OK\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
