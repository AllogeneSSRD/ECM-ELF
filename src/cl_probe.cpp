#include "cl_probe.h"
#include "ecm/application/configuration/runtime.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Print device information.
void printDeviceInfo(cl_device_id device) {
    cl_int err;

    char buffer[1024];

    // Device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
    std::cout << "    Device Name: " << buffer << std::endl;

    // Device type
    cl_device_type devType;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(devType), &devType, NULL);
    std::cout << "    Device Type: "
              << ((devType & CL_DEVICE_TYPE_GPU) ? "GPU" :
                  (devType & CL_DEVICE_TYPE_CPU) ? "CPU" : "Other")
              << std::endl;

    // Compute unit count
    cl_uint computeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(computeUnits), &computeUnits, NULL);
    std::cout << "    Compute Units: " << computeUnits << std::endl;

    // Max frequency
    cl_uint freq;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(freq), &freq, NULL);
    std::cout << "    Clock Frequency: " << freq << " MHz" << std::endl;

    // Work-group dimensions
    cl_uint dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(dims), &dims, NULL);
    std::cout << "    Work Item Dimensions: " << dims << std::endl;

    // Max work-item sizes per dimension
    std::vector<size_t> sizes(dims);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(size_t) * dims, sizes.data(), NULL);

    std::cout << "    Max Work Item Sizes: ";
    for (auto s : sizes) std::cout << s << " ";
    std::cout << std::endl;

    // Max work-group size
    size_t wgSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(wgSize), &wgSize, NULL);
    std::cout << "    Max Work Group Size: " << wgSize << std::endl;

    // Global memory
    cl_ulong globalMem;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(globalMem), &globalMem, NULL);
    std::cout << "    Global Memory: " << globalMem / (1024 * 1024) << " MB" << std::endl;

    // Local memory
    cl_ulong localMem;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(localMem), &localMem, NULL);
    std::cout << "    Local Memory: " << localMem << " Bytes" << std::endl;
}

// Interactive device selection by index, returns selected cl_device_id.
cl_device_id chooseDeviceInteractive() {
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL(err, "Failed to get platform count");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    CHECK_CL(err, "Failed to get platforms");

    struct Entry { cl_platform_id platform; cl_device_id device; std::string platformName; std::string deviceName; };
    std::vector<Entry> entries;

    char pname[1024];
    char dname[1024];

    for (cl_uint i = 0; i < numPlatforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND) continue;
        CHECK_CL(err, "Failed to get device count");
        if (numDevices == 0) continue;

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        CHECK_CL(err, "Failed to get devices");

        for (cl_uint j = 0; j < numDevices; j++) {
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            entries.push_back({platforms[i], devices[j], std::string(pname), std::string(dname)});
        }
    }

    if (entries.empty()) {
        std::cerr << "No OpenCL devices found." << std::endl;
        return NULL;
    }

    std::cout << "Available devices:\n";
    for (size_t idx = 0; idx < entries.size(); ++idx) {
        std::cout << "  [" << idx << "] " << entries[idx].platformName << " - " << entries[idx].deviceName << std::endl;
    }

    int sel = -1;
    std::string line;
    while (true) {
        std::cout << "Select device index: ";
        if (!std::getline(std::cin, line)) { std::cerr << "Input error" << std::endl; return NULL; }
        try { sel = std::stoi(line); }
        catch (...) { std::cout << "Invalid input, please enter a number." << std::endl; continue; }
        if (sel < 0 || static_cast<size_t>(sel) >= entries.size()) {
            std::cout << "Index out of range." << std::endl; continue;
        }
        break;
    }

    cl_device_id selDev = entries[sel].device;
    char nameBuf[1024];
    char verBuf[1024];
    clGetDeviceInfo(selDev, CL_DEVICE_NAME, sizeof(nameBuf), nameBuf, NULL);
    clGetDeviceInfo(selDev, CL_DEVICE_VERSION, sizeof(verBuf), verBuf, NULL);
    std::cout << "Device Name: " << nameBuf << std::endl;
    std::cout << "Version: " << verBuf << std::endl;
    return selDev;
}

// Main probe function.
void probePlatforms() {
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL(err, "Failed to get platform count");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    CHECK_CL(err, "Failed to get platforms");

    std::cout << "Number of platforms: " << numPlatforms << std::endl;

    for (cl_uint i = 0; i < numPlatforms; i++) {
        char buffer[1024];

        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          sizeof(buffer), buffer, NULL);

        std::cout << "\nPlatform " << i << ": " << buffer << std::endl;

        // Enumerate devices
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                             0, NULL, &numDevices);
        CHECK_CL(err, "Failed to get device count");

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                             numDevices, devices.data(), NULL);
        CHECK_CL(err, "Failed to get devices");

        for (cl_uint j = 0; j < numDevices; j++) {
            std::cout << "\n  Device " << j << ":\n";
            printDeviceInfo(devices[j]);
        }
    }
}

int listOpenclDevices() {
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 0;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enumerate OpenCL platforms, err=" << err << std::endl;
        return 0;
    }

    int index = 0;
    std::cout << "Available OpenCL devices:" << std::endl;
    for (cl_uint i = 0; i < numPlatforms; i++) {
        char pname[1024] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND || numDevices == 0) continue;
        if (err != CL_SUCCESS) continue;

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        if (err != CL_SUCCESS) continue;

        for (cl_uint j = 0; j < numDevices; j++) {
            char dname[1024] = {0};
            char dver[256] = {0};
            cl_device_type dtype = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(dver), dver, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
            const char *kind = (dtype & CL_DEVICE_TYPE_GPU) ? "GPU"
                               : (dtype & CL_DEVICE_TYPE_CPU) ? "CPU"
                                                              : "OTHER";
            std::cout << "  [" << index << "] " << pname << " | " << dname
                      << " | " << kind << " | " << dver << std::endl;
            ++index;
        }
    }
    if (index == 0) {
        std::cerr << "No OpenCL devices found." << std::endl;
    }
    return index;
}

bool configureOpenclDeviceIndex(int deviceIndex, bool printDevices) {
    if (printDevices) {
        int count = listOpenclDevices();
        if (count == 0) return false;
        if (deviceIndex < 0 || deviceIndex >= count) {
            std::cerr << "Invalid -d index: " << deviceIndex << ", available range: [0, "
                      << (count - 1) << "]" << std::endl;
            return false;
        }
    }
    ecm_runtime_config().device_index = deviceIndex;
    return true;
}

int findFirstAmdGpuDeviceIndex(bool printDevices) {
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        if (printDevices) std::cerr << "No OpenCL platforms found." << std::endl;
        return -1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        if (printDevices) std::cerr << "Failed to enumerate OpenCL platforms, err=" << err << std::endl;
        return -1;
    }

    int index = 0;
    if (printDevices) std::cout << "Available OpenCL devices:" << std::endl;
    for (cl_uint i = 0; i < numPlatforms; i++) {
        char pname[1024] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND || numDevices == 0) continue;
        if (err != CL_SUCCESS) continue;

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        if (err != CL_SUCCESS) continue;

        for (cl_uint j = 0; j < numDevices; j++) {
            char dname[1024] = {0};
            char dver[256] = {0};
            cl_device_type dtype = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(dver), dver, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
            const char *kind = (dtype & CL_DEVICE_TYPE_GPU) ? "GPU"
                               : (dtype & CL_DEVICE_TYPE_CPU) ? "CPU"
                                                              : "OTHER";
            if (printDevices) {
                std::cout << "  [" << index << "] " << pname << " | " << dname
                          << " | " << kind << " | " << dver << std::endl;
            }

            std::string sp = pname;
            std::string sd = dname;
            std::transform(sp.begin(), sp.end(), sp.begin(), [](unsigned char c){ return (char)std::tolower(c); });
            std::transform(sd.begin(), sd.end(), sd.begin(), [](unsigned char c){ return (char)std::tolower(c); });
            if ((dtype & CL_DEVICE_TYPE_GPU) &&
                (sp.find("amd") != std::string::npos || sd.find("amd") != std::string::npos ||
                 sd.find("gfx") != std::string::npos || sd.find("radeon") != std::string::npos)) {
                return index;
            }
            ++index;
        }
    }
    return -1;
}

bool configureFirstAmdGpuDevice(bool printDevices) {
    int idx = findFirstAmdGpuDeviceIndex(printDevices);
    if (idx < 0) {
        std::cerr << "No AMD GPU OpenCL device found." << std::endl;
        return false;
    }
    return configureOpenclDeviceIndex(idx, false);
}
