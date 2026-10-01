/* sleep_test.c -- blocks for N seconds (default 60).  Used to verify that
   tools/test/run_with_timeout.ps1 really kills a hung process tree. */
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
int main(int argc, char **argv) {
    int secs = (argc > 1) ? atoi(argv[1]) : 60;
    printf("sleep_test: sleeping %d s\n");
    fflush(stdout);
#ifdef _WIN32
    Sleep((DWORD)secs * 1000);
#endif
    printf("sleep_test: done\n");
    return 0;
}