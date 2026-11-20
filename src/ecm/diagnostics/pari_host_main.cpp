#include "pari_host.h"

int wmain(int argc, wchar_t** argv) {
    return argc == 2 ? ecm_diagnostics::run_pari_request(argv[1]) : 2;
}
