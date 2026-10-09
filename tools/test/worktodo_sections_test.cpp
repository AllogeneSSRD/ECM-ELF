// Self-checking unit test for the SECTION support of ecm_worktodo (D2, see
// docs/usage/STAGE1.md): the [Worker #N] header syntax, the section-aware
// first_line / advance pair, and list_workers.
//
// Compile (from the repo root):
//   tools\build_tool.bat tools\test\worktodo_sections_test.cpp src\core\ecm_worktodo.cpp
// Exit code: 0 = every check passed, 1 = at least one failed.
//
// Why a separate test from ecm_worktodo_test.cpp: that one is a print-only smoke
// test for the line parsers (no assertions, always exits 0). This one must FAIL
// loudly, because a broken section walk shows up as "the queue eats another
// worker's task" -- which is silent otherwise.

#include "../../src/core/ecm_worktodo.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string &what, const std::string &detail = "") {
    if (ok) {
        g_pass++;
        printf("  [ok]   %s\n", what.c_str());
    } else {
        g_fail++;
        printf("  [FAIL] %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ",
               detail.c_str());
    }
}

static void write_file(const std::string &path, const std::vector<std::string> &lines) {
    std::ofstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
    for (const std::string &l : lines) {
        out << l << "\n";
    }
}

static std::vector<std::string> read_file(const std::string &path) {
    std::vector<std::string> out;
    std::ifstream in(path, std::ios::in | std::ios::binary);
    std::string l;
    while (std::getline(in, l)) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        out.push_back(l);
    }
    return out;
}

static bool contains(const std::vector<std::string> &v, const std::string &want) {
    for (const std::string &s : v) {
        if (s == want) return true;
    }
    return false;
}

static void test_header_syntax() {
    printf("[1] [Worker #N] header syntax\n");
    bool bracket = false;
    check(ecm_worktodo_parse_worker_header("[Worker #1]", &bracket) == 1, "[Worker #1] -> 1");
    check(ecm_worktodo_parse_worker_header("[worker#7]", &bracket) == 7, "[worker#7] -> 7 (case)");
    check(ecm_worktodo_parse_worker_header("[ WORKER  # 12 ]", &bracket) == 12,
          "[ WORKER  # 12 ] -> 12 (spaces)");
    check(ecm_worktodo_parse_worker_header("[Worker #0]", &bracket) == 0, "[Worker #0] -> 0");
    check(ecm_worktodo_parse_worker_header("[Main]", &bracket) == 0 && bracket,
          "[Main] -> 0 but flagged as a bracket line");
    check(ecm_worktodo_parse_worker_header("[Worker x]", &bracket) == 0, "[Worker x] -> 0");
    check(ecm_worktodo_parse_worker_header("ECMSTAGE2=1,2,991,-1,x,0,0,3", &bracket) == 0 && !bracket,
          "task line -> 0 and not a bracket line");
}

// The fixture: two sections, comments, blank lines and a trailing no-section tail.
static const char *kLine1 = "ECMSTAGE2=1,2,991,-1,m991_1e6.save,0,0,3";
static const char *kLine2 = "ECMSTAGE2=1,2,4003,-1,m4003_1e6.save,0,0,3";

static void test_first_line_and_advance() {
    printf("[2] section-aware first_line / advance\n");
    const std::string path = "worktodo_sections_test.tmp";
    write_file(path, {
        "# leading comment",
        kLine1,                       // before any header -> worker 1
        "",
        "[Worker #2]",
        kLine2,
        "# trailing comment",
    });

    std::string line;
    check(ecm_worktodo_first_line(path, 1, line) && line == kLine1,
          "worker 1 gets the unprefixed line", line);
    line.clear();
    check(ecm_worktodo_first_line(path, 2, line) && line == kLine2,
          "worker 2 gets its own section", line);
    check(!ecm_worktodo_first_line(path, 3, line), "worker 3 has no task line");

    // Worker 2 must not touch worker 1's line, headers or comments.
    check(ecm_worktodo_advance(path, 2, kLine2, WorktodoAction::MarkError),
          "advance(worker 2, MarkError) rewrote the file");
    std::vector<std::string> after = read_file(path);
    check(contains(after, std::string("# ERROR ") + kLine2), "worker 2's line is marked", "");
    check(contains(after, kLine1), "worker 1's line is untouched");
    check(contains(after, "[Worker #2]"), "the section header survives");
    check(contains(after, "# leading comment") && contains(after, "# trailing comment"),
          "comments survive");
    check(after.size() == 6, "line count unchanged (only the text of one line changed)");

    // Now worker 1 removes its line: only that line disappears.
    check(ecm_worktodo_advance(path, 1, kLine1, WorktodoAction::Remove),
          "advance(worker 1, Remove) rewrote the file");
    after = read_file(path);
    check(!contains(after, kLine1), "worker 1's line is gone");
    check(contains(after, std::string("# ERROR ") + kLine2), "worker 2's marked line stays");
    check(!ecm_worktodo_first_line(path, 1, line), "worker 1 is now empty");
    line.clear();
    check(!ecm_worktodo_first_line(path, 2, line),
          "worker 2 sees no task either (already marked)");

    // Legacy call shape: worker <= 0 ignores sections (whole file = one queue).
    write_file(path, {kLine1, "[Worker #2]", kLine2});
    line.clear();
    check(ecm_worktodo_first_line(path, 0, line) && line == kLine1,
          "worker 0 (legacy) reads the first task line of the whole file", line);

    // A sectionless file behaves exactly as before for worker 1.
    write_file(path, {"# only", kLine1});
    line.clear();
    check(ecm_worktodo_first_line(path, 1, line) && line == kLine1,
          "sectionless file: worker 1 reads the plain line", line);

    std::remove(path.c_str());
}

static void test_list_workers() {
    printf("[3] list_workers\n");
    const std::string path = "worktodo_sections_test.tmp";
    std::vector<uint32_t> workers;
    std::string err;

    write_file(path, {kLine1, "[Worker #2]", kLine2, "[Worker #7]", kLine2});
    check(ecm_worktodo_list_workers(path, workers, err) && workers.size() == 3 &&
              workers[0] == 1 && workers[1] == 2 && workers[2] == 7,
          "sections 1, 2, 7 reported in ascending order");

    write_file(path, {kLine1, kLine2});
    check(ecm_worktodo_list_workers(path, workers, err) && workers.size() == 1 &&
              workers[0] == 1,
          "sectionless file reports worker 1 only");

    write_file(path, {"# nothing to do", "", "[Worker #4]"});
    check(ecm_worktodo_list_workers(path, workers, err) && workers.empty(),
          "headers without task lines report no worker");

    check(!ecm_worktodo_list_workers("does_not_exist.tmp", workers, err) && !err.empty(),
          "missing file -> false + error text");
    std::remove(path.c_str());
}

int main() {
    test_header_syntax();
    test_first_line_and_advance();
    test_list_workers();
    printf("\npassed: %d   failed: %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
