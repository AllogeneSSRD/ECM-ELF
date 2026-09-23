// cpuid_probe.c — report AVX/AVX-512 feature bits relevant to batched modmul.
// Build: cl /O2 tools/bench/cpuid_probe.c  (or gcc -O2)
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) { __cpuidex((int *)r, (int)leaf, (int)sub); }
static uint64_t xgetbv0(void) { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
static uint64_t xgetbv0(void) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}
#endif

static void bit(const char *name, uint32_t reg, int b) {
    printf("  %-24s %s\n", name, (reg >> b) & 1u ? "yes" : "no");
}

int main(void) {
    uint32_t r[4] = {0, 0, 0, 0};
    char vendor[13] = {0};
    cpuid(0, 0, r);
    memcpy(vendor, &r[1], 4);
    memcpy(vendor + 4, &r[3], 4);
    memcpy(vendor + 8, &r[2], 4);
    printf("vendor: %s\n", vendor);
    cpuid(1, 0, r);
    printf("family/model: %u/%u\n", ((r[0] >> 8) & 0xF) + ((r[0] >> 20) & 0xFF),
           ((r[0] >> 4) & 0xF) + (((r[0] >> 16) & 0xF) << 4));
    bit("OSXSAVE", r[2], 27);
    bit("AVX", r[2], 28);

    uint64_t xcr0 = xgetbv0();
    printf("XCR0 = 0x%llx  (SSE=%d AVX=%d opmask=%d ZMM_hi256=%d Hi16_ZMM=%d)\n",
           (unsigned long long)xcr0, (int)(xcr0 & 1), (int)((xcr0 >> 1) & 1), (int)((xcr0 >> 5) & 1),
           (int)((xcr0 >> 6) & 1), (int)((xcr0 >> 7) & 1));

    cpuid(7, 0, r);
    printf("leaf 7.0 EBX:\n");
    bit("BMI1", r[1], 3);
    bit("BMI2", r[1], 8);
    bit("AVX512F", r[1], 16);
    bit("AVX512DQ", r[1], 17);
    bit("AVX512IFMA", r[1], 21);
    bit("AVX512PF", r[1], 26);
    bit("AVX512ER", r[1], 27);
    bit("AVX512CD", r[1], 28);
    bit("AVX512BW", r[1], 30);
    bit("AVX512VL", r[1], 31);
    printf("leaf 7.0 ECX:\n");
    bit("AVX512VBMI", r[2], 1);
    bit("AVX512VBMI2", r[2], 6);
    bit("GFNI", r[2], 8);
    bit("VAES", r[2], 9);
    bit("VPCLMULQDQ", r[2], 10);
    bit("AVX512VNNI", r[2], 11);
    bit("AVX512BITALG", r[2], 12);
    bit("AVX512VPOPCNTDQ", r[2], 14);
    printf("leaf 7.0 EDX:\n");
    bit("AVX5124VNNIW", r[3], 2);
    bit("AVX5124FMAPS", r[3], 3);
    return 0;
}
