// Standalone smoke test for the Prime95 ECM2= worktodo parser.
// Compile: cl /EHsc /I <gmp>/include /I src/core ecm_worktodo_test.cpp src/core/ecm_worktodo.cpp gmp.lib
#include "../src/core/ecm_worktodo.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void test_ecm2(const std::string &line) {
    Ecm2Task t;
    std::string err;
    bool ok = ecm_parse_ecm2_line(line, t, err);
    printf("[%s] %s\n", ok ? "OK " : "FAIL", line.c_str());
    if (!ok) { printf("    err=%s\n", err.c_str()); return; }
    printf("    aid=%s fft2=%s k=%s b=%s n=%lu c=%s B1=%g B2=%g curves=%u sigma_fixed=%d sigma=%llu factors=%zu\n",
           t.aid.c_str(), t.fft2.c_str(), t.k.c_str(), t.b.c_str(), (unsigned long)t.n,
           t.c.c_str(), t.B1, t.B2, (unsigned)t.curves_to_run,
           (int)t.has_sigma, (unsigned long long)t.sigma, t.factors.size());
    mpz_t N;
    mpz_init(N);
    if (ecm_compute_ecm2_n(t, N, err)) {
        char *s = mpz_get_str(nullptr, 10, N);
        printf("    N=%s\n", s);
        free(s);
    } else {
        printf("    N-err=%s\n", err.c_str());
    }
    mpz_clear(N);
}

static void test_stage2(const std::string &line) {
    EcmStage2Task t;
    std::string err;
    bool ok = ecm_parse_stage2_line(line, t, err);
    printf("[%s] %s\n", ok ? "OK " : "FAIL", line.c_str());
    if (ok) {
        printf("    aid=%s k=%s b=%s n=%lu c=%s save=%s curves=%u factors=%zu\n",
               t.aid.c_str(), t.k.c_str(), t.b.c_str(), (unsigned long)t.n,
               t.c.c_str(), t.save_name.c_str(), (unsigned)t.curves_to_run, t.factors.size());
    } else {
        printf("    err=%s\n", err.c_str());
    }
}

int main() {
    test_ecm2("ECM2=1,2,991,-1,1000000,0,1,105413044550089");
    test_ecm2("ECM=1,2,991,-1,1000000,0,1,105413044550089");   // ECM= 与 ECM2= 等价
    test_ecm2("ECM=1,2,991,-1,1000000");                        // 缺 B2/curves/sigma → 默认 0/100/随机
    test_ecm2("ECM=1,2,991,-1,1000000,0,1,105413044550089,\"8218291649\"");   // 单因子 quoted
    test_ecm2("ECM=N/A,1,2,991,-1,1000000,0,1");
    test_ecm2("ECM=AID123,FFT2=192K,1,2,991,-1,1000000,0,1,105413044550089,\"1943118631,8218291649\"");
    test_ecm2("ECM=1,2,4003,-1,1000000,0,1,2027329164697536");
    test_stage2("ECMSTAGE2=1,2,991,-1,m991_1e6.save,0,0,3");
    return 0;
}
