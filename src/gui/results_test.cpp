// Unit test for the results store (milestone M5, docs/DEV_ECM_GUI.md section 9).
// Build: cmake --build <dir> --target ecm_gui_results_test
// Exit code: 0 = every check passed.
//
// What it pins down (all of it is a promise the GUI makes to the user):
//   * results.json.txt is APPEND-ONLY JSONL: one object per hit, fields aligned with
//     prime95's results.json.txt;
//   * results.txt merges every hit of the same factor into ONE line with the curve
//     and sigma lists (measured need: on M677/B1=1e6 five of eight curves report the
//     same 31-bit factor);
//   * results.txt is reproducible from the JSONL alone (rebuild must give the same
//     text), which is what makes rewriting it safe;
//   * the save-name contract parsing (m{n}_{b1}.save) and the fallback exponent from
//     a worktodo N expression.

#include "results.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ecmgui;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string &what, const std::string &detail = std::string()) {
    if (ok) {
        ++g_pass;
        std::printf("  [ok]   %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ",
                    detail.c_str());
    }
}

std::string read_all(const std::string &path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t count_lines(const std::string &text, const char *needle) {
    std::size_t n = 0, pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += std::strlen(needle);
    }
    return n;
}

HitRecord make_hit(const std::string &factor, int curve, unsigned long long sigma,
                   const std::string &save, int worker) {
    HitRecord r;
    r.factor = factor;
    r.curve = curve;
    r.sigma = sigma;
    r.has_sigma = true;
    r.param = 3;
    r.method = "gpu";
    r.save = save;
    r.worker = worker;
    r.device = 0;
    r.task = "ECMSTAGE2=1,2,677,-1,\"m677_1e6.save\",0,0,8";
    int exponent = 0;
    std::string b1;
    if (ResultsStore::parse_save_name(save, &exponent, &b1)) {
        r.exponent = exponent;
        r.b1_text = b1;
        r.b1 = std::atof(b1.c_str());
    }
    r.timestamp = "2026-09-28T20:00:00Z";
    return r;
}

} // namespace

int main() {
    const std::string json = "results_test.json.txt";
    const std::string txt = "results_test.txt";
    std::remove(json.c_str());
    std::remove(txt.c_str());

    std::printf("[1] save-name and expression parsing\n");
    {
        int exp = 0;
        std::string b1;
        check(ResultsStore::parse_save_name("m677_1e6.save", &exp, &b1) && exp == 677 && b1 == "1e6",
              "m{n}_{b1}.save parsed", std::to_string(exp) + "/" + b1);
        check(ResultsStore::parse_save_name("m5351_110e6.save", &exp, &b1) && exp == 5351 &&
                  b1 == "110e6", "bigger exponent + 110e6", std::to_string(exp) + "/" + b1);
        check(!ResultsStore::parse_save_name("m991_1e4_w1.save", &exp, &b1),
              "a name breaking the contract is rejected (B1 must be the last _ token)");
        check(ResultsStore::exponent_from_n_expr("(1*2^677-1)") == 677, "exponent from N",
              std::to_string(ResultsStore::exponent_from_n_expr("(1*2^677-1)")));
        check(ResultsStore::exponent_from_n_expr("(2^991-1)/(8218291649)") == 991,
              "exponent from a divided N");
    }

    std::printf("[2] a single hit lands in both files\n");
    {
        ResultsStore st;
        std::string err;
        check(st.init(json, txt, err), "init on a fresh sandbox", err);
        check(st.add(make_hit("1943118631", 3, 1824417927ull, "m677_1e6.save", 1), err),
              "add a hit", err);
        check(st.flush(err), "flush", err);
        const std::string j = read_all(json);
        const std::string t = read_all(txt);
        check(count_lines(j, "\"status\":\"F\"") == 1, "one JSONL object", j);
        check(j.find("\"factors\":[\"1943118631\"]") != std::string::npos, "factor in the JSONL");
        check(j.find("\"exponent\":677") != std::string::npos, "exponent in the JSONL");
        check(j.find("\"sigma\":1824417927") != std::string::npos, "sigma in the JSONL");
        check(j.find("\"curve\":3") != std::string::npos, "curve in the JSONL");
        check(j.find("\"worktype\":\"ECM\"") != std::string::npos, "worktype (prime95 field name)");
        check(t.find("M677 has a factor: 1943118631") != std::string::npos,
              "results.txt uses the prime95-shaped line", t);
        check(t.find("curves 3") != std::string::npos, "curve list");
        check(t.find("Sigmas=[1824417927]") != std::string::npos, "sigma list");
        check(t.find("hits=1") != std::string::npos, "hit count");
        check(count_lines(t, "has a factor") == 1, "exactly one merged line");
    }

    std::printf("[3] repeated hits of one factor merge into a single line\n");
    {
        ResultsStore st;
        std::string err;
        st.init(json, txt, err);
        // Two more curves of the same batch hit the same factor.
        check(st.add(make_hit("1943118631", 4, 1824417928ull, "m677_1e6.save", 1), err), "add #2", err);
        check(st.add(make_hit("1943118631", 9, 1824417933ull, "m677_1e6.save", 2), err), "add #3", err);
        check(st.add(make_hit("1943118631", 4, 1824417928ull, "m677_1e6.save", 1), err),
              "add a duplicate (curve+sigma seen before)", err);
        check(st.flush(err), "flush", err);
        const std::string j = read_all(json);
        const std::string t = read_all(txt);
        check(count_lines(j, "\"status\":\"F\"") == 4, "the JSONL keeps every hit (append-only)",
              std::to_string(count_lines(j, "\"status\":\"F\"")));
        check(count_lines(t, "has a factor") == 1, "still one merged line");
        check(t.find("curves 3,4,9") != std::string::npos, "curves merged in order", t);
        check(t.find("Sigmas=[1824417927,1824417928,1824417933]") != std::string::npos,
              "sigmas merged, duplicates dropped", t);
        check(t.find("hits=4") != std::string::npos, "hit counter includes duplicates", t);
        check(st.hit_count() == 4, "store counts 4 hits", std::to_string(st.hit_count()));
        check(st.factors().size() == 1, "one distinct factor");
    }

    std::printf("[4] a different factor gets its own line\n");
    {
        ResultsStore st;
        std::string err;
        st.init(json, txt, err);
        check(st.add(make_hit("1032053878207116276718209860111", 8, 1824417932ull,
                              "m677_1e6.save", 1), err), "add a second factor", err);
        check(st.flush(err), "flush", err);
        const std::string t = read_all(txt);
        check(count_lines(t, "has a factor") == 2, "two merged lines");
        check(t.find("M677 has a factor: 1032053878207116276718209860111") != std::string::npos,
              "the second factor is there");
    }

    std::printf("[5] the merged table is reproducible from the JSONL\n");
    {
        const std::string live = read_all(txt);
        ResultsStore st;
        std::string err;
        check(st.init(json, txt, err), "re-open the same sandbox", err);
        check(st.rebuild_from_jsonl(err), "rebuild_from_jsonl", err);
        const std::string rebuilt = read_all(txt);
        // Only the "updated <utc>" header differs between the two writes.
        const auto strip_header = [](std::string s) {
            const std::size_t p = s.find("# updated");
            if (p == std::string::npos) return s;
            const std::size_t e = s.find('\n', p);
            return s.substr(0, p) + s.substr(e == std::string::npos ? s.size() : e);
        };
        check(strip_header(live) == strip_header(rebuilt), "rebuild gives the same table",
              "live=" + std::to_string(strip_header(live).size()) +
                  " rebuilt=" + std::to_string(strip_header(rebuilt).size()));
        check(st.hit_count() == 5, "a restart replays the JSONL (5 hits)",
              std::to_string(st.hit_count()));
        check(st.factors().size() == 2, "and rebuilds both factors");
    }

    std::printf("[6] the merged line format\n");
    {
        MergedFactor f;
        f.factor = "1943118631";
        f.bits = 31;
        f.exponent = 677;
        f.b1_text = "1e6";
        f.param = 3;
        f.method = "gpu";
        f.curves = {0, 1, 2};
        f.sigmas = {100, 101, 102};
        f.hits = 3;
        const std::string line = ResultsStore::format_line(f);
        check(line == "M677 has a factor: 1943118631 (ECM curves 0,1,2, B1=1e6, param 3, gpu, "
                      "Sigmas=[100,101,102], hits=3)",
              "exact shape", line);
    }

    std::printf("[7] damage tolerance\n");
    {
        // A truncated last line (crash while appending) must not break the replay.
        {
            std::ofstream out(json, std::ios::out | std::ios::app | std::ios::binary);
            out << "{\"status\":\"F\", \"factors\":[\"999";
        }
        ResultsStore st;
        std::string err;
        check(st.init(json, txt, err), "init with a truncated tail", err);
        check(st.hit_count() == 5, "the truncated line is ignored, the good ones are kept",
              std::to_string(st.hit_count()));
    }

    std::remove(json.c_str());
    std::remove(txt.c_str());
    std::printf("\npassed: %d   failed: %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
