#include "../../../src/ecm/storage/queue/owner_lease.h"
#include <chrono>
#include <cstdio>
#include <thread>
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    try {
        ecm_queue::OwnerLease owner(argv[1], std::stoi(argv[2]));
        std::puts("ACQUIRED"); std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(argv[3])));
        return 0;
    } catch (const std::exception& error) { std::puts(error.what()); return 23; }
}
