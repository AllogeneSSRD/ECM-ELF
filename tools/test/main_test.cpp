#include "ecm/backend/opencl/runtime/device.h"
#include "ecm/backend/opencl/runtime/cgbn_opencl.h"
#include "opencl_low_risk_tests.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }


int main() {
    probePlatforms();

    bool ok = runOpenClLowRiskOperatorTests();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
