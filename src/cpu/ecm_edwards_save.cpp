#include "ecm_edwards_save.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr uint32_t ECM_MAGICNUM = 0x1725bcd9u;
constexpr uint32_t ECM_VERSION = 6u;
constexpr uint32_t CHECKSUM_OFFSET = 48u;

// ---------- LE write helpers (sum != NULL 时累加 checksum) ----------

void put_u32(std::ofstream &o, uint32_t v, uint32_t *sum) {
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                          (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    o.write(reinterpret_cast<const char *>(b), 4);
    if (sum) *sum += v;
}
void put_i32(std::ofstream &o, int32_t v, uint32_t *sum) {
    put_u32(o, static_cast<uint32_t>(v), sum);
}
void put_u64(std::ofstream &o, uint64_t v, uint32_t *sum) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = (unsigned char)(v >> (8 * i));
    o.write(reinterpret_cast<const char *>(b), 8);
    if (sum) *sum += (uint32_t)((v >> 32) + v);
}
void put_double(std::ofstream &o, double v) {
    o.write(reinterpret_cast<const char *>(&v), 8);
}
void put_bytes(std::ofstream &o, const char *p, size_t n) {
    if (n) o.write(p, static_cast<std::streamsize>(n));
}

// ---------- LE read helpers ----------

bool get_u32(std::ifstream &in, uint32_t *v, uint32_t *sum) {
    unsigned char b[4];
    in.read(reinterpret_cast<char *>(b), 4);
    if (!in) return false;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    if (sum) *sum += *v;
    return true;
}
bool get_i32(std::ifstream &in, int32_t *v, uint32_t *sum) {
    uint32_t t;
    if (!get_u32(in, &t, sum)) return false;
    *v = (int32_t)t;
    return true;
}
bool get_u64(std::ifstream &in, uint64_t *v, uint32_t *sum) {
    unsigned char b[8];
    in.read(reinterpret_cast<char *>(b), 8);
    if (!in) return false;
    *v = 0;
    for (int i = 0; i < 8; ++i) *v |= ((uint64_t)b[i]) << (8 * i);
    if (sum) *sum += (uint32_t)((*v >> 32) + *v);
    return true;
}
bool get_double(std::ifstream &in, double *v) {
    in.read(reinterpret_cast<char *>(v), 8);
    return static_cast<bool>(in);
}
bool get_bytes(std::ifstream &in, char *p, size_t n) {
    if (n) in.read(p, static_cast<std::streamsize>(n));
    return static_cast<bool>(in);
}

// ---------- giant: 4-byte limb count + 32-bit LE limbs ----------
// Prime95 write_giant 的 checksum 累加 = len(经 write_long) + Σlimbs + len(显式二次).

bool put_giant(std::ofstream &o, const mpz_t v, uint32_t *sum) {
    const size_t bits = mpz_sizeinbase(v, 2);
    size_t count = (bits + 31) / 32;
    if (count == 0) count = 1;
    std::vector<uint32_t> limbs(count, 0);
    size_t written = 0;
    mpz_export(limbs.data(), &written, -1, sizeof(uint32_t), 0, 0, v);

    put_u32(o, static_cast<uint32_t>(count), sum);      // len (write_long)
    for (size_t i = 0; i < count; ++i) {
        const uint32_t limb = (i < written) ? limbs[i] : 0;
        unsigned char b[4] = {(unsigned char)limb, (unsigned char)(limb >> 8),
                              (unsigned char)(limb >> 16), (unsigned char)(limb >> 24)};
        o.write(reinterpret_cast<const char *>(b), 4);
        if (sum) *sum += limb;
    }
    if (sum) *sum += static_cast<uint32_t>(count);      // 显式二次 len
    return static_cast<bool>(o);
}

bool get_giant(std::ifstream &in, mpz_t v, uint32_t *sum) {
    uint32_t count = 0;
    if (!get_u32(in, &count, sum)) return false;        // len (read_long)
    if (count == 0) return false;
    std::vector<uint32_t> limbs(count);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char b[4];
        in.read(reinterpret_cast<char *>(b), 4);
        if (!in) return false;
        limbs[i] = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
        if (sum) *sum += limbs[i];
    }
    mpz_import(v, count, -1, sizeof(uint32_t), 0, 0, limbs.data());
    if (sum) *sum += count;                             // 显式二次 len
    return true;
}

void put_header(std::ofstream &o, const ecm_save_common &cm, const char *stage) {
    put_u32(o, ECM_MAGICNUM, nullptr);
    put_u32(o, ECM_VERSION, nullptr);
    put_double(o, cm.k);
    put_u32(o, cm.b, nullptr);
    put_u32(o, cm.n, nullptr);
    put_i32(o, cm.c, nullptr);
    put_bytes(o, stage, 10);
    put_bytes(o, "\0", 1);
    put_bytes(o, "\0", 1);
    put_double(o, cm.pct_complete);
    put_u32(o, 0, nullptr);   // checksum 占位 (offset 48)
}

void put_checksum(std::ofstream &o, uint32_t sum) {
    o.seekp(CHECKSUM_OFFSET, std::ios::beg);
    put_u32(o, sum, nullptr);
}

bool get_header(std::ifstream &in, ecm_save_common &cm) {
    uint32_t magic, version;
    char stage[10], pad[2];
    double pct;
    if (!get_u32(in, &magic, nullptr)) return false;
    if (magic != ECM_MAGICNUM) return false;
    if (!get_u32(in, &version, nullptr)) return false;
    if (version == 0 || version > ECM_VERSION) return false;
    if (!get_double(in, &cm.k)) return false;
    if (!get_u32(in, &cm.b, nullptr)) return false;
    if (!get_u32(in, &cm.n, nullptr)) return false;
    if (!get_i32(in, &cm.c, nullptr)) return false;
    if (!get_bytes(in, stage, 10)) return false;
    if (!get_bytes(in, pad, 2)) return false;
    if (!get_double(in, &pct)) return false;
    memcpy(cm.stage, stage, 10);
    cm.stage[9] = 0;
    cm.pct_complete = pct;
    return true;
}

// 读回文件里存的 checksum 值 (offset 48), 供校验.
bool get_file_checksum(std::ifstream &in, uint32_t *file_sum) {
    in.seekg(CHECKSUM_OFFSET, std::ios::beg);
    return get_u32(in, file_sum, nullptr);
}

void make_stage_str(char out[10], uint32_t curve, char stage_no) {
    memset(out, 0, 10);
    snprintf(out, 10, "C%uS%c", curve, stage_no);
}

} // namespace

// ---------------------------------------------------------------------------
// 仅读头部 (供覆盖冲突判定)
// ---------------------------------------------------------------------------

bool ecm_save_read_header(const std::string &path, ecm_save_common &cm, uint32_t *state) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    if (!get_header(in, cm)) return false;

    // 数据区布局: curve(4) @52, average_B2(8) @56, state(4) @64,
    //              sigma(8) @68, B(8) @76, C(8) @84, ...
    in.seekg(CHECKSUM_OFFSET + 4, std::ios::beg);
    uint32_t curve = 0, st = 0;
    uint64_t avg_b2 = 0, sigma = 0, b1 = 0, b2 = 0;
    if (!get_u32(in, &curve, nullptr)) return false;
    if (!get_u64(in, &avg_b2, nullptr)) return false;
    if (!get_u32(in, &st, nullptr)) return false;
    if (!get_u64(in, &sigma, nullptr)) return false;
    if (!get_u64(in, &b1, nullptr)) return false;
    if (!get_u64(in, &b2, nullptr)) return false;

    cm.curve = curve;
    cm.sigma = sigma;
    cm.B1 = b1;
    cm.B2 = b2;
    (void)avg_b2;
    if (state) *state = st;
    return true;
}

// ---------------------------------------------------------------------------
// MIDSTAGE
// ---------------------------------------------------------------------------

bool ecm_edwards_write_midstage(const std::string &path, const ecm_save_common &cm,
                                const mpz_t Qx, const mpz_t Qz) {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    if (!o) return false;

    char stage[10];
    make_stage_str(stage, cm.curve, '2');
    put_header(o, cm, stage);

    uint32_t sum = 0;
    put_u32(o, cm.curve, &sum);
    put_u64(o, 0, nullptr);             // average_B2 (不进 checksum)
    put_u32(o, 2, &sum);                // state = MIDSTAGE
    put_u64(o, cm.sigma, nullptr);      // sigma (不进 checksum)
    put_u64(o, cm.B1, &sum);            // B
    put_u64(o, cm.B2, &sum);            // C
    put_u32(o, 0, &sum);                // sigma_type = 0 (Atkin-Morain)
    put_u32(o, 1, &sum);                // montg_stage1 = 1
    put_giant(o, Qx, &sum);
    put_i32(o, 1, &sum);                // Qz flag
    put_giant(o, Qz, &sum);
    put_i32(o, 0, &sum);                // gg flag
    put_checksum(o, sum);
    o.close();
    return !o.fail();
}

bool ecm_edwards_read_midstage(const std::string &path, ecm_save_common &cm,
                               mpz_t Qx, mpz_t Qz) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    if (!get_header(in, cm)) return false;

    uint32_t file_sum = 0;
    if (!get_file_checksum(in, &file_sum)) return false;
    in.seekg(CHECKSUM_OFFSET + 4, std::ios::beg);   // 回到 data 区起点

    uint32_t sum = 0;
    uint32_t curve, state, sigma_type, montg_stage1;
    uint64_t avg_b2, sigma, B1, B2;
    if (!get_u32(in, &curve, &sum)) return false;
    if (!get_u64(in, &avg_b2, nullptr)) return false;
    if (!get_u32(in, &state, &sum)) return false;
    if (!get_u64(in, &sigma, nullptr)) return false;
    if (!get_u64(in, &B1, &sum)) return false;
    if (!get_u64(in, &B2, &sum)) return false;
    if (!get_u32(in, &sigma_type, &sum)) return false;
    if (!get_u32(in, &montg_stage1, &sum)) return false;

    cm.curve = curve;
    cm.sigma = sigma;
    cm.B1 = B1;
    cm.B2 = B2;
    (void)avg_b2;
    (void)sigma_type;
    (void)montg_stage1;

    if (state != 2) return false;   // 本读函数仅支持 MIDSTAGE (验证用)

    if (!get_giant(in, Qx, &sum)) return false;
    int32_t qz_flag = 0;
    if (!get_i32(in, &qz_flag, &sum)) return false;
    if (qz_flag) {
        if (!get_giant(in, Qz, &sum)) return false;
    } else {
        mpz_set_ui(Qz, 0);
    }
    int32_t gg_flag = 0;
    if (!get_i32(in, &gg_flag, &sum)) return false;
    (void)gg_flag;

    return sum == file_sum;
}

// ---------------------------------------------------------------------------
// STAGE1 (Edwards 中途 checkpoint)
// ---------------------------------------------------------------------------

bool ecm_edwards_write_stage1(const std::string &path, const ecm_save_common &cm,
                              uint64_t stage1_start_prime, uint32_t exp_buffer_size,
                              uint32_t bitnum, uint32_t dict_size,
                              const mpz_t dict_x, const mpz_t dict_y,
                              const mpz_t e_x, const mpz_t e_y, const mpz_t e_z) {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    if (!o) return false;

    char stage[10];
    make_stage_str(stage, cm.curve, '1');
    put_header(o, cm, stage);

    uint32_t sum = 0;
    put_u32(o, cm.curve, &sum);
    put_u64(o, 0, nullptr);             // average_B2
    put_u32(o, 1, &sum);                // state = STAGE1
    put_u64(o, cm.sigma, nullptr);
    put_u64(o, cm.B1, &sum);            // B
    put_u64(o, cm.B2, &sum);            // C
    put_u32(o, 0, &sum);                // sigma_type = 0
    put_u32(o, 0, &sum);                // montg_stage1 = 0 (Edwards)
    put_u64(o, stage1_start_prime, &sum);
    put_u32(o, exp_buffer_size, &sum);
    put_u32(o, bitnum, &sum);
    put_u32(o, dict_size, &sum);
    put_giant(o, dict_x, &sum);
    put_giant(o, dict_y, &sum);
    put_giant(o, e_x, &sum);
    put_giant(o, e_y, &sum);
    put_giant(o, e_z, &sum);
    put_checksum(o, sum);
    o.close();
    return !o.fail();
}

bool ecm_edwards_read_stage1(const std::string &path, ecm_save_common &cm,
                             uint64_t *stage1_start_prime, uint32_t *exp_buffer_size,
                             uint32_t *bitnum, uint32_t *dict_size,
                             mpz_t dict_x, mpz_t dict_y,
                             mpz_t e_x, mpz_t e_y, mpz_t e_z) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    if (!get_header(in, cm)) return false;

    uint32_t file_sum = 0;
    if (!get_file_checksum(in, &file_sum)) return false;
    in.seekg(CHECKSUM_OFFSET + 4, std::ios::beg);

    uint32_t sum = 0;
    uint32_t curve, state, sigma_type, montg_stage1;
    uint64_t avg_b2, sigma, B1, B2;
    if (!get_u32(in, &curve, &sum)) return false;
    if (!get_u64(in, &avg_b2, nullptr)) return false;
    if (!get_u32(in, &state, &sum)) return false;
    if (!get_u64(in, &sigma, nullptr)) return false;
    if (!get_u64(in, &B1, &sum)) return false;
    if (!get_u64(in, &B2, &sum)) return false;
    if (!get_u32(in, &sigma_type, &sum)) return false;
    if (!get_u32(in, &montg_stage1, &sum)) return false;

    cm.curve = curve;
    cm.sigma = sigma;
    cm.B1 = B1;
    cm.B2 = B2;
    (void)avg_b2;
    (void)sigma_type;
    (void)montg_stage1;

    if (state != 1) return false;

    if (!get_u64(in, stage1_start_prime, &sum)) return false;
    if (!get_u32(in, exp_buffer_size, &sum)) return false;
    if (!get_u32(in, bitnum, &sum)) return false;
    if (!get_u32(in, dict_size, &sum)) return false;
    if (!get_giant(in, dict_x, &sum)) return false;
    if (!get_giant(in, dict_y, &sum)) return false;
    if (!get_giant(in, e_x, &sum)) return false;
    if (!get_giant(in, e_y, &sum)) return false;
    if (!get_giant(in, e_z, &sum)) return false;

    return sum == file_sum;
}
