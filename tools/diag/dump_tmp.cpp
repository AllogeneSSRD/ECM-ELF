/* dump_tmp.cpp -- print the (Qx, Qz) that a stage-1 run wrote into a .tmp file.
 * Uses the driver's own reader (ecm_edwards_save), so the parse is exactly the
 * one the pipeline uses for stage-2 handoff.  Everything goes through printf with
 * mpz_get_str: mixing gmp_printf and printf interleaves out of order on Windows.
 *   usage: dump_tmp <file.tmp> [more.tmp ...]
 */
#include "ecm_edwards_save.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <file.tmp> [...]\n", argv[0]); return 2; }
    for (int i = 1; i < argc; i++) {
        ecm_save_common cm;
        mpz_t Qx, Qz;
        mpz_inits(Qx, Qz, NULL);
        if (!ecm_edwards_read_midstage(argv[i], cm, Qx, Qz)) {
            printf("%s : READ FAILED\n", argv[i]);
        } else {
            char *sx = mpz_get_str(NULL, 16, Qx);
            char *sz = mpz_get_str(NULL, 16, Qz);
            printf("%s\n  curve=%u sigma=%llu B1=%llu B2=%llu\n  Qx=%s\n  Qz=%s\n",
                   argv[i], cm.curve, (unsigned long long)cm.sigma,
                   (unsigned long long)cm.B1, (unsigned long long)cm.B2,
                   sx ? sx : "?", sz ? sz : "?");
            free(sx); free(sz);
        }
        mpz_clears(Qx, Qz, NULL);
        fflush(stdout);
    }
    return 0;
}
