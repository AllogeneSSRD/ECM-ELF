/* ecm_mont_ckpt.cpp -- see ecm_mont_ckpt.h for the format contract. */
#include "ecm_mont_ckpt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

/* FNV-1a 64 over the bytes written so far.  The reader recomputes it over every
   byte preceding the CHECKSUM line, so a truncated or half-flushed file is
   rejected instead of being resumed from garbage. */
#define CK_FNV_OFFSET 1469598103934665603ull
#define CK_FNV_PRIME  1099511628211ull

static uint64_t ck_fnv(uint64_t h, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= CK_FNV_PRIME;
    }
    return h;
}

void mont_ckpt_init(mont_ckpt_t *ck)
{
    memset(ck, 0, sizeof(*ck));
    ck->status = MONT_CKPT_INFLIGHT;
    mpz_inits(ck->X0, ck->Z0, ck->X1, ck->Z1, ck->xout, ck->factor, NULL);
    ck->field[0] = '\0';
}

void mont_ckpt_clear(mont_ckpt_t *ck)
{
    mpz_clears(ck->X0, ck->Z0, ck->X1, ck->Z1, ck->xout, ck->factor, NULL);
}

void mont_ckpt_ident(mont_ckpt_t *ck, uint32_t curve, uint64_t sigma, double B1,
                     int torsion, size_t sbits, const char *field)
{
    ck->curve = curve;
    ck->sigma = sigma;
    ck->B1 = B1;
    ck->torsion = torsion;
    ck->sbits = sbits;
    snprintf(ck->field, sizeof(ck->field), "%s", field ? field : "");
}

std::string mont_ckpt_path(const std::string &stem, uint32_t curve)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "_c%07u.ckpt", curve);
    return stem + buf;
}

/* --------------------------------------------------------------------------
 * writer
 * ------------------------------------------------------------------------ */
static void w_line(FILE *f, uint64_t *sum, const char *s)
{
    /* The reader accumulates every byte it consumed, newline included, so the
       newline has to be part of the writer's sum as well. */
    *sum = ck_fnv(*sum, s, strlen(s));
    *sum = ck_fnv(*sum, "\n", 1);
    fputs(s, f);
    fputc('\n', f);
}

static void w_key(FILE *f, uint64_t *sum, const char *key, const char *val)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s=", key);
    *sum = ck_fnv(*sum, buf, strlen(buf));
    fputs(buf, f);
    *sum = ck_fnv(*sum, val, strlen(val));
    fputs(val, f);
    *sum = ck_fnv(*sum, "\n", 1);
    fputc('\n', f);
}

static void w_u64(FILE *f, uint64_t *sum, const char *key, unsigned long long v)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%llu", v);
    w_key(f, sum, key, buf);
}

static void w_mpz(FILE *f, uint64_t *sum, const char *key, const mpz_t z)
{
    char *s = mpz_get_str(NULL, 16, z);   /* NULL: let GMP allocate (may be huge) */
    w_key(f, sum, key, s ? s : "0");
    if (s) free(s);
}

int mont_ckpt_write(const std::string &path, const mpz_t N, const mont_ckpt_t *ck)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return -1;
    uint64_t sum = CK_FNV_OFFSET;

    char head[48];
    snprintf(head, sizeof(head), "MPA-MONT-CKPT %d", MONT_CKPT_VERSION);
    w_line(f, &sum, head);
    w_mpz(f, &sum, "N", N);
    {
        char b1[64];
        snprintf(b1, sizeof(b1), "%.0f", ck->B1);
        w_key(f, &sum, "B1", b1);
    }
    w_u64(f, &sum, "TORSION", (unsigned long long)ck->torsion);
    w_u64(f, &sum, "SBITS", (unsigned long long)ck->sbits);
    w_u64(f, &sum, "CURVE", (unsigned long long)ck->curve);
    w_u64(f, &sum, "SIGMA", (unsigned long long)ck->sigma);
    w_key(f, &sum, "FIELD", ck->field);
    w_key(f, &sum, "STATUS", ck->status == MONT_CKPT_DONE ? "DONE" : "INFLIGHT");
    w_u64(f, &sum, "BITNUM", (unsigned long long)ck->bitnum);

    if (ck->status == MONT_CKPT_DONE) {
        w_u64(f, &sum, "HIT", (unsigned long long)(ck->hit ? 1 : 0));
        if (ck->hit) w_mpz(f, &sum, "FACTOR", ck->factor);
        else         w_mpz(f, &sum, "XOUT", ck->xout);
    } else {
        w_mpz(f, &sum, "X0", ck->X0);
        w_mpz(f, &sum, "Z0", ck->Z0);
        w_mpz(f, &sum, "X1", ck->X1);
        w_mpz(f, &sum, "Z1", ck->Z1);
    }

    {
        char buf[32];
        snprintf(buf, sizeof(buf), "CHECKSUM=%016llx", (unsigned long long)sum);
        fputs(buf, f);            /* not part of the checksum itself */
        fputc('\n', f);
    }
    fputs("END\n", f);
    const int rc = (fclose(f) == 0) ? 0 : -1;
    return rc;
}

/* --------------------------------------------------------------------------
 * reader
 * ------------------------------------------------------------------------ */
static void set_hex(mpz_t z, const std::string &v)
{
    if (v.empty()) mpz_set_ui(z, 0);
    else if (mpz_set_str(z, v.c_str(), 16) != 0) mpz_set_ui(z, 0);
}

int mont_ckpt_read(const std::string &path, const mpz_t N, double B1, int torsion,
                   size_t sbits, mont_ckpt_t *ck, std::string *why)
{
    std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
    if (!in) return 0;

    uint64_t sum = CK_FNV_OFFSET;
    bool seen_sum = false, sum_ok = false, saw_end = false, saw_magic = false;
    bool have_state = false, have_result = false;
    std::string line;
    bool bad = false;
    std::string reason;

    while (std::getline(in, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.empty()) continue;

        if (line.compare(0, 9, "CHECKSUM=") == 0) {
            const unsigned long long want = strtoull(line.c_str() + 9, NULL, 16);
            sum_ok = (want == sum);
            seen_sum = true;
            continue;
        }
        sum = ck_fnv(sum, line.data(), line.size());
        sum = ck_fnv(sum, "\n", 1);

        if (line == "END") { saw_end = true; break; }
        if (line.compare(0, 14, "MPA-MONT-CKPT ") == 0) {
            const int ver = atoi(line.c_str() + 14);
            if (ver != MONT_CKPT_VERSION) {
                bad = true;
                reason = "checkpoint version mismatch";
            }
            saw_magic = true;
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            bad = true;
            reason = "malformed line";
            break;
        }
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);

        if (key == "N") {
            mpz_t t;
            mpz_init(t);
            set_hex(t, val);
            if (mpz_cmp(t, N) != 0) {
                bad = true;
                reason = "different N";
            }
            mpz_clear(t);
        } else if (key == "B1") {
            if (atof(val.c_str()) != B1) {
                bad = true;
                reason = "different B1";
            }
        } else if (key == "TORSION") {
            if (atoi(val.c_str()) != torsion) {
                bad = true;
                reason = "different torsion (exponent convention)";
            }
        } else if (key == "SBITS") {
            if ((size_t)strtoull(val.c_str(), NULL, 10) != sbits) {
                bad = true;
                reason = "different s_bits (B1 exponent)";
            }
        } else if (key == "CURVE") {
            ck->curve = (uint32_t)strtoul(val.c_str(), NULL, 10);
        } else if (key == "SIGMA") {
            ck->sigma = strtoull(val.c_str(), NULL, 10);
        } else if (key == "FIELD") {
            snprintf(ck->field, sizeof(ck->field), "%s", val.c_str());
        } else if (key == "STATUS") {
            ck->status = (val == "DONE") ? MONT_CKPT_DONE : MONT_CKPT_INFLIGHT;
        } else if (key == "BITNUM") {
            ck->bitnum = (size_t)strtoull(val.c_str(), NULL, 10);
        } else if (key == "HIT") {
            ck->hit = atoi(val.c_str()) ? 1 : 0;
        } else if (key == "X0") {
            set_hex(ck->X0, val); have_state = true;
        } else if (key == "Z0") {
            set_hex(ck->Z0, val); have_state = true;
        } else if (key == "X1") {
            set_hex(ck->X1, val); have_state = true;
        } else if (key == "Z1") {
            set_hex(ck->Z1, val); have_state = true;
        } else if (key == "XOUT") {
            set_hex(ck->xout, val); have_result = true;
        } else if (key == "FACTOR") {
            set_hex(ck->factor, val); have_result = true;
        }
        if (bad) break;
    }

    if (bad) { if (why) *why = reason; return 0; }
    if (!saw_magic) { if (why) *why = "not a mont checkpoint"; return 0; }
    if (!saw_end) { if (why) *why = "truncated (no END)"; return 0; }
    if (!seen_sum || !sum_ok) { if (why) *why = "checksum mismatch"; return 0; }
    if (ck->status == MONT_CKPT_DONE) {
        if (!have_result) { if (why) *why = "DONE without a result"; return 0; }
    } else if (ck->bitnum > 0 && !have_state) {
        /* bitnum == 0 with no state is legal: it is the seed record that pins the
           curve's sigma before any ladder work has happened (see the driver). */
        if (why) *why = "INFLIGHT without a mid-ladder state";
        return 0;
    }
    return 1;
}

void mont_ckpt_remove(const std::string &path)
{
    remove(path.c_str());
}
