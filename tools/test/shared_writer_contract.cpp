#include "ecm/application/gui/worktodo_gen.h"
#include "ecm/integer.h"
#include "ecm/io/results.h"
#include "ecm/service/stage2/input.h"
#include "io/files.h"
#include "io/json.h"
#include "io/prime95.h"
#include "io/worktodo.h"
#include "sync/p95.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

namespace fs = std::filesystem;
namespace r = ecm_results;
static const std::string old_task = "ECMSTAGE1=1,2,127,-1,\"old.save\",10000,0,1";
static const std::string new_task = "ECMSTAGE1=1,2,127,-1,\"new.save\",10000,0,1";
static void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}
static void put(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    require(bool(out), "cannot write fixture");
}
static std::string get(const fs::path& path) {
    return fs::exists(path) ? ecm_io::read_file(path) : std::string{};
}
static std::string stage2_row(mpz_srcptr n) {
    r::Submission s;
    s.exponent = 29;
    s.b1 = 20;
    s.b2 = 2000;
    s.sigma = 200;
    s.when = "2026-10-26 12:00:00";
    return r::internal_json(s, n, 0, "writer-contract",
                            ",\"stage2-engine\":{\"name\":\"contract\"}");
}
static void stage1(const fs::path& root, mpz_srcptr n) {
    r::Submission s;
    s.exponent = 29;
    s.b1 = s.b2 = 20;
    s.sigma = 100;
    s.stage1_only = true;
    s.when = "2026-10-26 12:00:00";
    std::vector<r::Stage1BatchRecord> batch;
    batch.push_back({s, r::internal_json(s, n, 0, "writer-contract",
                                         ",\"stage1-engine\":{\"coverage\":\"B1\"}")});
    r::publish_stage1_batch(root / "internal.jsonl", root / "submission.txt", std::move(batch), n,
                            "writer-contract");
}
static int child(const std::string& mode, const fs::path& root, const std::string& events) {
    const auto target =
        root / (mode == "gui" || mode == "advance"                    ? "queue.txt"
                : mode == "stage1" || mode == "cpu" || mode == "cuda" ? "internal.jsonl"
                                                                      : "worktodo.add");
    SetEnvironmentVariableA("ECM_TEST_WRITE_TARGET", target.string().c_str());
    SetEnvironmentVariableA("ECM_TEST_WRITE_EVENTS", events.c_str());
    ecm_edwards::Integer n;
    mpz_set_ui(n.v, 1);
    mpz_mul_2exp(n.v, n.v, 29);
    mpz_sub_ui(n.v, n.v, 1);
    try {
        if (mode == "hold" || mode == "try_lock") {
            ecm_io::FileLock lock((root / "probe").string(), ecm_io::FileLockDomain::Publication,
                                  200);
            if (mode == "hold") {
                HANDLE ready = OpenEventA(EVENT_MODIFY_STATE, FALSE, (events + ".ready").c_str());
                HANDLE go = OpenEventA(SYNCHRONIZE, FALSE, (events + ".go").c_str());
                require(ready && go, "lease events");
                SetEvent(ready);
                WaitForSingleObject(go, 10000);
                CloseHandle(ready);
                CloseHandle(go);
            }
        } else if (mode == "feeder") {
            std::string error;
            require(p95_write_worktodo_add(target.string(),
                                           {{1, "ECMSTAGE2=1,2,127,-1,\"first.save\",10000,0,1"}},
                                           true, error),
                    error);
        } else if (mode == "handoff") {
            P95TransferConfig cfg;
            cfg.worktodo_path = (root / "worktodo.txt").string();
            cfg.pending_path =
                (root / (events.find(".a") != events.npos ? "pending-a" : "pending-b")).string();
            cfg.add_workers = "1";
            cfg.recover_lost = false;
            const auto name = events.find(".a") != events.npos ? "first.save" : "second.save";
            const auto result = p95_transfer_deliver(cfg, "ECMSTAGE2=1,2,127,-1,\"" +
                                                              std::string(name) + "\",10000,0,1");
            require(result.ok, result.error);
        } else if (mode == "gui") {
            ecmgui::FileStamp stamp;
            require(ecmgui::stamp_file(target.string(), stamp), "stamp");
            std::string error;
            require(ecmgui::apply_append(target.string(), "[Worker #1]\n" + new_task + "\n", stamp,
                                         error),
                    error);
        } else if (mode == "advance") {
            require(ecm_worktodo_advance(target.string(), 2, old_task, WorktodoAction::Remove),
                    "advance");
        } else if (mode == "stage1")
            stage1(root, n.v);
        else if (mode == "cpu")
            r::append_text_record(target, stage2_row(n.v));
        else if (mode == "cuda")
            s2prod::append(target, stage2_row(n.v));
        else
            throw std::runtime_error("unknown child mode");
        return 0;
    } catch (const std::exception& e) {
        put(root / (mode + ".error"), e.what());
        return 3;
    }
}
struct Process {
    HANDLE value = nullptr;
    Process(const fs::path& exe, const std::string& mode, const fs::path& root,
            const std::string& events) {
        auto command = L"\"" + exe.wstring() + L"\" --child " +
                       std::wstring(mode.begin(), mode.end()) + L" \"" + root.wstring() + L"\" " +
                       std::wstring(events.begin(), events.end());
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION info{};
        require(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                               nullptr, nullptr, &startup, &info) != 0,
                "CreateProcess");
        value = info.hProcess;
        CloseHandle(info.hThread);
    }
    ~Process() {
        if (value) {
            if (WaitForSingleObject(value, 0) == WAIT_TIMEOUT) {
                TerminateProcess(value, 99);
                WaitForSingleObject(value, 2000);
            }
            CloseHandle(value);
        }
    }
    DWORD wait() {
        require(WaitForSingleObject(value, 10000) == WAIT_OBJECT_0, "child timed out");
        DWORD code = 0;
        GetExitCodeProcess(value, &code);
        return code;
    }
};
struct Events {
    std::string prefix;
    HANDLE ready = nullptr, go = nullptr;
    Events(const std::string& name, bool release) : prefix(name) {
        ready = CreateEventA(nullptr, TRUE, FALSE, (name + ".ready").c_str());
        go = CreateEventA(nullptr, TRUE, release, (name + ".go").c_str());
        require(ready && go, "CreateEvent");
    }
    ~Events() {
        CloseHandle(ready);
        CloseHandle(go);
    }
};
static bool race(const fs::path& exe, const fs::path& dir, const std::string& first,
                 const std::string& second, bool age = false) {
    fs::create_directories(dir);
    if (first == "gui")
        put(dir / "queue.txt", "[Worker #1]\n# editable\n[Worker #2]\n" + old_task + "\n");
    const auto prefix =
        "Local\\ECMWriter" + std::to_string(GetCurrentProcessId()) + "_" + dir.filename().string();
    Events a(prefix + ".a", false), b(prefix + ".b", true);
    Process one(exe, first, dir, a.prefix);
    require(WaitForSingleObject(a.ready, 5000) == WAIT_OBJECT_0,
            "first writer did not reach real publication");
    if (age && fs::exists(dir / "worktodo.add.lock")) {
        HANDLE marker = CreateFileW((dir / "worktodo.add.lock").c_str(), FILE_WRITE_ATTRIBUTES,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, 0, nullptr);
        require(marker != INVALID_HANDLE_VALUE, "age legacy marker");
        FILETIME now;
        GetSystemTimeAsFileTime(&now);
        ULARGE_INTEGER t;
        t.LowPart = now.dwLowDateTime;
        t.HighPart = now.dwHighDateTime;
        t.QuadPart -= 1200000000ULL;
        FILETIME old{t.LowPart, t.HighPart};
        require(SetFileTime(marker, nullptr, nullptr, &old) != 0, "SetFileTime");
        CloseHandle(marker);
    }
    Process two(exe, second, dir, b.prefix);
    // A production gate blocks B before its publication point. The release is
    // deterministic; this bounded observation only distinguishes blocked B.
    const bool interleaved = WaitForSingleObject(b.ready, 500) == WAIT_OBJECT_0 ||
                             WaitForSingleObject(two.value, 0) == WAIT_OBJECT_0;
    if (interleaved)
        two.wait();
    SetEvent(a.go);
    const auto first_code = one.wait(), second_code = two.wait();
    const auto bytes = get(dir / (first == "gui"      ? "queue.txt"
                                  : first == "stage1" ? "internal.jsonl"
                                                      : "worktodo.add"));
    const bool kept =
        first == "gui" ? bytes.find(old_task) == bytes.npos && bytes.find(new_task) != bytes.npos
        : first == "stage1"
            ? bytes.find("stage1-engine") != bytes.npos && bytes.find("stage2-engine") != bytes.npos
            : bytes.find("first.save") != bytes.npos && bytes.find("second.save") != bytes.npos;
    const bool passed = !interleaved && first_code == 0 && second_code == 0 && kept;
    std::cout << "{\"case\":\"" << dir.filename().string()
              << "\",\"passed\":" << (passed ? "true" : "false")
              << ",\"interleaved\":" << (interleaved ? "true" : "false")
              << ",\"first_exit\":" << first_code << ",\"second_exit\":" << second_code
              << ",\"records_retained\":" << (kept ? "true" : "false") << "}\n";
    return passed;
}
static bool marker_failure(const fs::path& exe, const fs::path& root) {
    fs::create_directories(root);
    const auto marker = root / "worktodo.add.lock";
    put(marker, "retired marker");
    HANDLE file =
        CreateFileW(marker.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr, OPEN_EXISTING, 0, nullptr);
    require(file != INVALID_HANDLE_VALUE, "marker fixture");
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    ULARGE_INTEGER t;
    t.LowPart = now.dwLowDateTime;
    t.HighPart = now.dwHighDateTime;
    t.QuadPart -= 1200000000ULL;
    FILETIME old{t.LowPart, t.HighPart};
    SetFileTime(file, nullptr, nullptr, &old);
    CloseHandle(file);
    require(SetFileAttributesW(marker.c_str(), FILE_ATTRIBUTE_READONLY) != 0, "readonly marker");
    Events events("Local\\ECMMarker" + std::to_string(GetCurrentProcessId()) + ".b", true);
    Process writer(exe, "handoff", root, events.prefix);
    const bool completed = WaitForSingleObject(writer.value, 3800) == WAIT_OBJECT_0;
    const bool passed = completed && writer.wait() == 0;
    std::cout << "{\"case\":\"retired_marker_delete_failure\",\"passed\":"
              << (passed ? "true" : "false")
              << ",\"completed_within_budget\":" << (completed ? "true" : "false") << "}\n";
    return passed;
}
static bool kernel_lifetime(const fs::path& exe, const fs::path& root) {
    fs::create_directories(root / "alias");
    Events events("Local\\ECMKernel" + std::to_string(GetCurrentProcessId()) + ".a", false);
    Process owner(exe, "hold", root, events.prefix);
    require(WaitForSingleObject(events.ready, 3000) == WAIT_OBJECT_0, "owner ready");
    const auto start = GetTickCount64();
    Process busy(exe, "try_lock", root / "alias" / "..", events.prefix);
    require(busy.wait() == 3, "alias bypassed gate");
    const auto elapsed = GetTickCount64() - start;
    require(elapsed >= 150 && elapsed < 1500, "lock timeout was not bounded");
    TerminateProcess(owner.value, 99);
    owner.wait();
    Process retry(exe, "try_lock", root, events.prefix);
    require(retry.wait() == 0, "killed owner retained lock");
    require(!fs::exists(root / "probe.publish.lock"), "Windows lock carrier was not removed");
    put(root / "probe", "identity");
    fs::create_hard_link(root / "probe", root / "linked");
    bool rejected = false;
    try {
        ecm_io::FileLock lock((root / "linked").string());
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "hard-link writer alias was accepted");
    std::cout << "{\"case\":\"kernel_lifetime_alias_timeout\",\"passed\":true,\"timeout_ms\":"
              << elapsed << "}\n";
    return true;
}
int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--child")
        return child(argv[2], fs::absolute(argv[3]), argv[4]);
    try {
        if (argc == 4 && std::string(argv[1]) == "--legacy-marker")
            return marker_failure(fs::absolute(argv[2]), fs::absolute(argv[3])) ? 0 : 1;
        require(argc == 2, "fresh evidence directory required");
        const auto root = fs::absolute(argv[1]);
        require(!fs::exists(root), "fresh evidence directory required");
        fs::create_directories(root);
        const auto exe = fs::absolute(argv[0]);
        bool passed = true;
        passed = race(exe, root / "p95_feeder", "feeder", "handoff") && passed;
        passed = race(exe, root / "p95_stale", "handoff", "handoff", true) && passed;
        passed = race(exe, root / "gui_edit", "gui", "advance") && passed;
        passed = race(exe, root / "result_cpu", "stage1", "cpu") && passed;
        passed = race(exe, root / "result_cuda", "stage1", "cuda") && passed;
        passed = marker_failure(exe, root / "marker_failure") && passed;
        passed = kernel_lifetime(exe, root / "kernel") && passed;
        const auto mixed = root / "mixed";
        fs::create_directory(mixed);
        ecm_edwards::Integer n;
        mpz_set_ui(n.v, 1);
        mpz_mul_2exp(n.v, n.v, 29);
        mpz_sub_ui(n.v, n.v, 1);
        r::append_text_record(mixed / "internal.jsonl", stage2_row(n.v));
        bool accepted = true;
        try {
            stage1(mixed, n.v);
            stage1(mixed, n.v);
            r::check_stage1_outputs(mixed / "internal.jsonl", mixed / "submission.txt");
        } catch (const std::exception& e) {
            accepted = false;
            put(mixed / "error.txt", e.what());
        }
        std::cout << "{\"case\":\"mixed_history\",\"passed\":" << (accepted ? "true" : "false")
                  << "}\n";
        auto row = r::json::parse(stage2_row(n.v));
        row.set("b1", r::json::Value::number("99"));
        put(mixed / "internal.jsonl", row.dump() + "\n");
        bool corrupt = false;
        try {
            stage1(mixed, n.v);
        } catch (const std::exception&) {
            corrupt = true;
        }
        require(corrupt, "damaged Stage2 checksum accepted");
        row = r::json::parse(stage2_row(n.v));
        row.set("stage1-engine", r::json::parse("{}"));
        put(mixed / "internal.jsonl", row.dump() + "\n");
        bool ambiguous = false;
        try {
            stage1(mixed, n.v);
        } catch (const std::exception&) {
            ambiguous = true;
        }
        require(ambiguous, "ambiguous engine markers accepted");
        const auto initial = root / "stage2_initial";
        fs::create_directories(initial);
        stage1(initial, n.v);
        r::check_stage2_outputs(initial / "internal.jsonl", initial / "stage2_submission.txt",
                                false);
        r::append_text_record(initial / "internal.jsonl", stage2_row(n.v));
        bool missing_ledger = false;
        try {
            r::check_stage2_outputs(initial / "internal.jsonl", initial / "stage2_submission.txt",
                                    false);
        } catch (const std::exception&) {
            missing_ledger = true;
        }
        require(missing_ledger, "missing Stage2 ledger accepted after Stage2 output");
        bool case_alias = false;
        const auto alias_start = GetTickCount64();
        try {
            r::check_stage1_outputs(root / "Fresh.jsonl", root / "fresh.JSONL");
        } catch (const std::exception&) {
            case_alias = true;
        }
        require(case_alias && GetTickCount64() - alias_start < 1000,
                "fresh case alias acquired the same gate twice");
        P95TransferConfig uncertain;
        uncertain.worktodo_path = (root / "uncertain" / "worktodo.txt").string();
        fs::create_directories(root / "uncertain");
        uncertain.pending_path = (root / "uncertain" / "pending").string();
        uncertain.add_workers = "1";
        uncertain.recover_lost = false;
        SetEnvironmentVariableA("ECM_TEST_AFTER_REPLACE", "1");
        const auto maybe =
            p95_transfer_deliver(uncertain, "ECMSTAGE2=1,2,127,-1,\"visible.save\",10000,0,1");
        SetEnvironmentVariableA("ECM_TEST_AFTER_REPLACE", nullptr);
        require(!maybe.ok && maybe.publication_uncertain &&
                    get(root / "uncertain" / "worktodo.add").find("visible.save") !=
                        std::string::npos,
                "visible replacement failure was treated as an uncommitted write");
        std::cout << "{\"case\":\"mixed_history_validation\",\"passed\":true}\n";
        return passed && accepted ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
