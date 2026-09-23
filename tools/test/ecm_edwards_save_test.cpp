// ecm_edwards_save_test.cpp — MIDSTAGE/STAGE1 存档读写单测.
// 1) 用 e0000347 的参数写 MIDSTAGE, 与真实文件字节级比对.
// 2) 写→读回, 校验 Qx/Qz 与 checksum.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <gmp.h>
#include "ecm_edwards_save.h"

static bool read_file_bytes(const std::string &path, std::vector<unsigned char> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

int main() {
    mpz_t Qx, Qz, rx, rz;
    mpz_inits(Qx, Qz, rx, rz, NULL);
    mpz_set_str(Qx, "85183067367971599076486525774441605347591666837158980405258912417697054795733873485227501504996375824397", 10);
    mpz_set_str(Qz, "152775196292170372149790400447947966778233322567163432737639337868938246891692081507753868643328930052391", 10);

    ecm_save_common cm;
    cm.k = 1.0;
    cm.b = 2;
    cm.n = 347;
    cm.c = -1;
    cm.curve = 1;
    cm.B1 = 1000000;
    cm.B2 = 100000000;
    cm.sigma = 20260922;

    // 输出到 tools/test/fixtures/ (派生文件; 供与 Prime95 存档人工比对)
    const std::string out_path = "D:/code/MPA-OpenCl/tools/test/fixtures/_test_e0000347.save";
    if (!ecm_edwards_write_midstage(out_path, cm, Qx, Qz)) {
        printf("WRITE FAILED\n");
        return 1;
    }

    // 1) 字节级比对真实 e0000347
    std::vector<unsigned char> mine, ref;
    read_file_bytes(out_path, mine);
    if (read_file_bytes("D:/code/GIMPS/p95v3104b05.win64/e0000347", ref)) {
        bool same = (mine.size() == ref.size()) && (memcmp(mine.data(), ref.data(), mine.size()) == 0);
        printf("byte-identical vs e0000347: %s (mine=%zu ref=%zu)\n", same ? "YES" : "NO", mine.size(), ref.size());
        if (!same) {
            size_t n = mine.size() < ref.size() ? mine.size() : ref.size();
            for (size_t i = 0; i < n; i++) {
                if (mine[i] != ref[i]) { printf("  first diff at offset 0x%zx: mine=%02x ref=%02x\n", i, mine[i], ref[i]); break; }
            }
        }
    } else {
        printf("(reference e0000347 not found, skip byte compare)\n");
    }

    // 2) 写→读回
    ecm_save_common rc;
    if (!ecm_edwards_read_midstage(out_path, rc, rx, rz)) {
        printf("READ FAILED\n");
        return 1;
    }
    printf("roundtrip: n=%u B1=%llu B2=%llu sigma=%llu curve=%u\n",
           rc.n, (unsigned long long)rc.B1, (unsigned long long)rc.B2,
           (unsigned long long)rc.sigma, rc.curve);
    printf("Qx match: %s\n", mpz_cmp(rx, Qx) == 0 ? "YES" : "NO");
    printf("Qz match: %s\n", mpz_cmp(rz, Qz) == 0 ? "YES" : "NO");

    // 3) STAGE1 写→读回 (用假值, 只测格式/checksum 往返)
    mpz_t dx, dy, ex, ey, ez, rdx, rdy, rex, rey, rez;
    mpz_inits(dx, dy, ex, ey, ez, rdx, rdy, rex, rey, rez, NULL);
    mpz_set_str(dx, "104556554552488554205787423168183375242196345094436345346860667677933248762857395662015551685902788651774", 10);
    mpz_set_str(dy, "9870953467125125219696581814346914227832422779618282985357731528541139318585858019072024232322472929060", 10);
    mpz_set_str(ex, "123456789", 10);
    mpz_set_str(ey, "987654321", 10);
    mpz_set_str(ez, "555", 10);

    const std::string s1_path = "D:/code/MPA-OpenCl/tools/test/fixtures/_test_stage1.save";
    if (!ecm_edwards_write_stage1(s1_path, cm, 2, 1442105, 12345, 64, dx, dy, ex, ey, ez)) {
        printf("STAGE1 WRITE FAILED\n");
        return 1;
    }
    ecm_save_common rc2;
    uint64_t sp; uint32_t ebs, bn, ds;
    if (!ecm_edwards_read_stage1(s1_path, rc2, &sp, &ebs, &bn, &ds, rdx, rdy, rex, rey, rez)) {
        printf("STAGE1 READ FAILED\n");
        return 1;
    }
    printf("stage1 roundtrip: start_prime=%llu expbuf=%u bitnum=%u dictsize=%u\n",
           (unsigned long long)sp, ebs, bn, ds);
    printf("dict_x match: %s, e_x match: %s\n",
           mpz_cmp(rdx, dx) == 0 ? "YES" : "NO", mpz_cmp(rex, ex) == 0 ? "YES" : "NO");

    mpz_clears(Qx, Qz, rx, rz, dx, dy, ex, ey, ez, rdx, rdy, rex, rey, rez, NULL);
    return 0;
}
