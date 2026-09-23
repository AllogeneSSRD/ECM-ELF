#include <stdio.h>
#include <gmp.h>
int main(void) {
    printf("mp_bits_per_limb = %d\n", (int)GMP_NUMB_BITS);
    printf("sizeof(mp_limb_t) = %d\n", (int)sizeof(mp_limb_t));
    printf("GMP_NUMB_MAX = %llu\n", (unsigned long long)GMP_NUMB_MAX);
    return 0;
}
