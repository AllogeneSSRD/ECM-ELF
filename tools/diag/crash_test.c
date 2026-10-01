/* crash_test.c -- deliberately dereferences a null pointer, to verify that the machine no
   longer shows a Windows Error Reporting dialog (which used to hang the parent cmd.exe and
   cost the test harness tens of minutes per crash).  Build: see build_crash_test.ps1. */
#include <stdio.h>
int main(void) {
    volatile int *p = (int *)0;
    printf("crash_test: about to dereference NULL\n");
    fflush(stdout);
    *p = 1;                      /* access violation -> 0xC0000005 */
    printf("crash_test: NOT REACHED\n");
    return 0;
}