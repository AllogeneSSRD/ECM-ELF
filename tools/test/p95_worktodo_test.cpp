// p95_worktodo_test.cpp — unit tests for Prime95 worktodo/prime.txt helpers.
//
// Build:
//   cl /O2 /EHsc /utf-8 -Isrc/core tools/p95_worktodo_test.cpp src/core/p95_worktodo.cpp
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "p95_worktodo.h"

static int g_fail = 0;
static int g_pass = 0;

static void check(bool cond, const std::string &what) {
    if (cond) { g_pass++; std::printf("[OK ] %s\n", what.c_str()); }
    else { g_fail++; std::printf("[FAIL] %s\n", what.c_str()); }
}

static std::string tmp_path(const char *name) {
    std::string p = "p95test_";
    p += name;
    return p;
}

static void write_file(const std::string &path, const std::string &content) {
    std::ofstream o(path, std::ios::trunc);
    o << content;
}

static std::string read_file(const std::string &path) {
    std::ifstream i(path);
    std::string s((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
    return s;
}

int main() {
    // ---- 1. worktodo.txt with worker sections -----------------------------
    const std::string wt = tmp_path("worktodo.txt");
    write_file(wt,
        "[Worker #1]\n"
        "ECM=1,2,991,-1,1000000,0,1,105413044550089\n"
        "\n"
        "[Worker #2]\n"
        "# commented out\n"
        "\n"
        "[Worker #3]\n"
        "ECM=B7D7B831040C5F71CC3C09FF97602611,1,2,12323,-1,55000000,0,664,\"41208113,62107921\"\n"
        "\n"
        "[Worker #4]\n"
        ";;MOVED;;[Worker #5]\n"
        "Pminus1=N/A,1,2,550169,-1,66427649,0,68,\"3444032551901327\"\n");

    std::vector<P95WorkerSection> secs;
    std::string err;
    check(p95_read_worktodo(wt, secs, err), "read worktodo.txt");
    check(secs.size() == 5, "5 sections parsed (got " + std::to_string(secs.size()) + ")");
    if (secs.size() == 5) {
        check(secs[0].worker == 1 && secs[1].worker == 2 && secs[2].worker == 3 &&
              secs[3].worker == 4 && secs[4].worker == 5,
              "worker numbers 1,2,3,4,5");
        check(p95_count_active(secs[0]) == 1, "worker1 active=1");
        check(p95_count_handoff(secs[0]) == 1, "worker1 handoff=1 (curves=1 + sigma)");
        check(p95_count_active(secs[1]) == 0, "worker2 empty (comment ignored)");
        check(p95_count_active(secs[2]) == 1, "worker3 active=1");
        check(p95_count_handoff(secs[2]) == 0, "worker3 handoff=0 (curves=664, no sigma)");
        check(p95_count_active(secs[3]) == 0, "worker4 empty");
        check(secs[4].worker == 5 && p95_count_active(secs[4]) == 1,
              ";;MOVED;;[Worker #5] header recognised");
    }
    check(p95_total_handoff(secs) == 1, "total handoff = 1");

    // ---- 2. handoff classification edge cases -----------------------------
    check(p95_line_is_handoff("ECM=1,2,991,-1,1000000,0,1,105413044550089"),
          "handoff: no AID");
    check(p95_line_is_handoff("ECM=N/A,1,2,991,-1,1000000,0,1,105413044550089"),
          "handoff: AID=N/A");
    check(p95_line_is_handoff("ECM2=1,2,677,-1,1000000,0,1,6581585141005897"),
          "handoff: ECM2= prefix");
    check(p95_line_is_handoff(
              "ECM=B7D7B831040C5F71CC3C09FF97602611,1,2,12323,-1,55000000,0,1,664,\"1\""),
          "handoff: 32-hex AID + curves=1 + sigma");
    check(!p95_line_is_handoff("ECM=1,2,991,-1,1000000,0,100"),
          "not handoff: no sigma, curves=100");
    check(!p95_line_is_handoff("ECM=AID,1,2,12323,-1,55000000,0,664,\"1,2\""),
          "not handoff: curves=664 (fresh multi-curve assignment)");
    check(!p95_line_is_handoff("ECM=1,2,991,-1,1000000,0,4,\"8218291649\""),
          "not handoff: curves=4 + quoted factors (factors must not shift positions)");
    check(p95_line_is_handoff("ECM=1,2,991,-1,1000000,0,1,105413044550089,\"8218291649\""),
          "handoff: sigma + quoted factors");
    check(!p95_line_is_handoff("ECMSTAGE2=1,2,5153,-1,\"m5153.save\",0,0,960"),
          "not handoff: ECMSTAGE2=");
    check(!p95_line_is_handoff("# ECM=1,2,991,-1,1000000,0,1,5"), "not handoff: commented");
    check(!p95_line_is_handoff("Pminus1=N/A,1,2,550169,-1,66427649,0,68"),
          "not handoff: Pminus1");

    // ---- 3. worktodo.add merge (append keeps existing sections) -----------
    const std::string add = tmp_path("worktodo.add");
    remove(add.c_str());
    std::vector<std::pair<int, std::string>> a1;
    a1.push_back(std::make_pair(2, std::string("ECM=1,2,991,-1,1000000,0,1,111")));
    check(p95_write_worktodo_add(add, a1, /*append=*/true, err),
          "write worktodo.add (create)");
    std::vector<P95WorkerSection> got;
    check(p95_read_worktodo(add, got, err) && got.size() == 1 && got[0].worker == 2,
          "created [Worker #2] section");
    check(read_file(add) == "[Worker #2]\nECM=1,2,991,-1,1000000,0,1,111\n",
          "worktodo.add content exact");

    std::vector<std::pair<int, std::string>> a2;
    a2.push_back(std::make_pair(2, std::string("ECM=1,2,991,-1,1000000,0,1,222")));
    a2.push_back(std::make_pair(1, std::string("ECM=1,2,991,-1,1000000,0,1,333")));
    check(p95_write_worktodo_add(add, a2, /*append=*/true, err),
          "write worktodo.add (append)");
    check(read_file(add) ==
              "[Worker #1]\nECM=1,2,991,-1,1000000,0,1,333\n"
              "\n"
              "[Worker #2]\nECM=1,2,991,-1,1000000,0,1,111\n"
              "ECM=1,2,991,-1,1000000,0,1,222\n",
          "append merged into existing worker + ascending order");

    // append=false truncates
    check(p95_write_worktodo_add(add, a1, /*append=*/false, err),
          "write worktodo.add (truncate)");
    check(read_file(add) == "[Worker #2]\nECM=1,2,991,-1,1000000,0,1,111\n",
          "truncate keeps only new assignment");

    // ---- 4. prime.txt keys -----------------------------------------------
    const std::string prime = tmp_path("prime.txt");
    write_file(prime,
        "# comment\n"
        "NumWorkers=6\n"
        "Memory=8192 during 7:30-23:30 else 8192\n"
        "MaxHighMemWorkers=3\n"
        "\n"
        "[Internals]\n"
        "MaxHighMemWorkers=99\n");
    long long v = -1;
    check(p95_read_prime_int(prime, "MaxHighMemWorkers", v) && v == 3,
          "MaxHighMemWorkers=3 from prime.txt (got " + std::to_string(v) + ")");
    check(p95_read_prime_int(prime, "NumWorkers", v) && v == 6, "NumWorkers=6");
    check(!p95_read_prime_int(prime, "NoSuchKey", v), "missing key returns false");
    check(!p95_read_prime_int(tmp_path("does_not_exist.txt"), "X", v),
          "missing prime.txt returns false");

    // ---- 5. misc ---------------------------------------------------------
    check(p95_ecm_save_name(991) == "e0000991", "save name e0000991");
    check(p95_ecm_save_name(12345678) == "e12345678", "save name e12345678");

    remove(wt.c_str());
    remove(add.c_str());
    remove(prime.c_str());

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
