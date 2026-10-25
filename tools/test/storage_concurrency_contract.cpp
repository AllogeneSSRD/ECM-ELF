// Exercise the production cache/outbox implementations with deterministic
// scheduling at their existing C-file and queue-publication calls.
#include "io/worktodo.h"
#include "sync/files.h"
#include <windows.h>
#include <gmp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace schedule {
using namespace std::chrono_literals;
std::mutex mutex;
std::condition_variable changed;
thread_local int writer = 0;
bool first_closed = false, second_started = false, second_opened = false;
bool observed = false, queue_entered = false, park_started = false;

FILE* open(const char* path, const char* mode) {
    FILE* file = std::fopen(path, mode);
    if (file && mode[0] == 'w' && writer == 2) {
        std::unique_lock<std::mutex> lock(mutex);
        second_opened = true;
        changed.notify_all();
        changed.wait(lock, [] { return observed; });
    }
    return file;
}

int close(FILE* file) {
    const int result = std::fclose(file);
    if (writer == 1) {
        std::unique_lock<std::mutex> lock(mutex);
        first_closed = true;
        changed.notify_all();
        changed.wait(lock, [] { return second_started; });
        // A serialized second writer cannot open until the first returns.
        changed.wait_for(lock, 250ms, [] { return second_opened; });
    }
    return result;
}

bool append(const std::string& path, const std::string& finished, int worker,
            const std::string& line, std::string& error) {
    if (line.find("first.save") != std::string::npos) {
        std::unique_lock<std::mutex> lock(mutex);
        queue_entered = true;
        changed.notify_all();
        changed.wait(lock, [] { return park_started; });
        // Allow an unprotected park to finish while flush owns its snapshot.
        changed.wait_for(lock, 250ms, [] { return observed; });
    }
    return ecm_worktodo_append_unique(path, finished, worker, line, error);
}
} // namespace schedule

#define fopen schedule::open
#define fclose schedule::close
#include "../../src/ecm/storage/cache/ecm_stage1_exp_cache.cpp"
#undef fopen
#undef fclose
#define ecm_worktodo_append_unique schedule::append
#include "../../src/ecm/storage/transfer/ecm_stage2_transfer.cpp"
#undef ecm_worktodo_append_unique

static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

static void cache_race(const std::filesystem::path& root) {
    mpz_t value, loaded;
    mpz_inits(value, loaded, nullptr);
    require(ecm_build_lcm_exponent(value, 1000000, 1), "build test exponent");
    const auto path = (root / "cache.bin").string();
    auto first = std::async(std::launch::async, [&] {
        schedule::writer = 1;
        return ecm_exp_cache_save(path, value, 1000000, 1, nullptr);
    });
    auto second = std::async(std::launch::async, [&] {
        {
            std::unique_lock<std::mutex> lock(schedule::mutex);
            schedule::changed.wait(lock, [] { return schedule::first_closed; });
            schedule::second_started = true;
            schedule::changed.notify_all();
        }
        schedule::writer = 2;
        return ecm_exp_cache_save(path, value, 1000000, 1, nullptr);
    });
    const bool first_saved = first.get();
    std::string error;
    const bool valid =
        ecm_exp_cache_load(path, loaded, 1000000, 1, &error) && mpz_cmp(value, loaded) == 0;
    {
        std::lock_guard<std::mutex> lock(schedule::mutex);
        schedule::observed = true;
        schedule::changed.notify_all();
    }
    const bool second_saved = second.get();
    mpz_clears(value, loaded, nullptr);
    require(first_saved, "first cache publication failed");
    require(valid, "published cache was truncated by another writer");
    require(second_saved, "second cache publication failed");
}

static void cache_bounds(const std::filesystem::path& root) {
    mpz_t value;
    mpz_init(value);
    require(ecm_build_lcm_exponent(value, 1000000, 1), "build bounds exponent");
    const auto path = (root / "bounds.bin").string();
    require(ecm_exp_cache_save(path, value, 1000000, 1, nullptr), "seed bounds cache");
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(40);
    const uint64_t excessive = UINT64_MAX;
    file.write(reinterpret_cast<const char*>(&excessive), sizeof(excessive));
    file.close();
    bool rejected = false;
    try {
        rejected = !ecm_exp_cache_load(path, value, 1000000, 1, nullptr);
    } catch (...) {
        mpz_clear(value);
        throw std::runtime_error("corrupt cache length escaped as allocation exception");
    }
    mpz_clear(value);
    require(rejected, "corrupt cache length accepted");
}

static void outbox_race(const std::filesystem::path& root) {
    Stage2TransferConfig cfg{(root / "queue.txt").string(), (root / "finished.txt").string(),
                             (root / "outbox").string(), 1};
    const std::string first = "ECMSTAGE2=1,2,127,-1,\"first.save\",100000,0,1";
    const std::string second = "ECMSTAGE2=1,2,127,-1,\"second.save\",100000,0,1";
    std::string error;
    require(ecm_stage2_transfer_park(cfg, first, error), "seed outbox");
    auto flush = std::async(std::launch::async, [&] {
        size_t remaining = 0;
        std::string detail;
        return ecm_stage2_transfer_flush(cfg.pending, remaining, detail);
    });
    auto park = std::async(std::launch::async, [&] {
        {
            std::unique_lock<std::mutex> lock(schedule::mutex);
            schedule::changed.wait(lock, [] { return schedule::queue_entered; });
            schedule::park_started = true;
            schedule::changed.notify_all();
        }
        std::string detail;
        const bool ok = ecm_stage2_transfer_park(cfg, second, detail);
        std::lock_guard<std::mutex> lock(schedule::mutex);
        schedule::observed = true;
        schedule::changed.notify_all();
        return ok;
    });
    require(flush.get(), "flush failed");
    require(park.get(), "concurrent park failed");
    size_t remaining = 0;
    require(ecm_stage2_transfer_flush(cfg.pending, remaining, error), "retry outbox");
    const auto bytes = ecm_sync::read_file(std::filesystem::u8path(cfg.worktodo));
    require(bytes.find(first) != std::string::npos, "first handoff missing");
    require(bytes.find(second) != std::string::npos, "flush discarded concurrent parked handoff");
}

static void outbox_failures(const std::filesystem::path& root) {
    Stage2TransferConfig cfg{(root / "queue.txt").string(), (root / "finished.txt").string(),
                             (root / "outbox").string(), 1};
    const std::string line = "ECMSTAGE2=1,2,127,-1,\"retry.save\",100000,0,1";
    const auto blocked = cfg.pending + ".publish.tmp";
    std::string error;
    std::filesystem::create_directory(blocked);
    require(!ecm_stage2_transfer_park(cfg, line, error), "failed park reported success");
    require(!std::filesystem::exists(cfg.pending), "failed park created outbox");
    std::filesystem::remove(blocked);
    require(ecm_stage2_transfer_park(cfg, line, error), "park retry failed");
    const auto original = ecm_sync::read_file(cfg.pending);
    std::filesystem::create_directory(blocked);
    size_t remaining = 0;
    require(!ecm_stage2_transfer_flush(cfg.pending, remaining, error),
            "failed acknowledgement reported success");
    require(remaining == 1 && ecm_sync::read_file(cfg.pending) == original,
            "acknowledgement failure lost responsibility");
    std::filesystem::remove(blocked);
    require(ecm_stage2_transfer_flush(cfg.pending, remaining, error) && remaining == 0,
            "acknowledgement retry failed");
    const auto queue = ecm_sync::read_file(cfg.worktodo);
    const auto first = queue.find(line);
    require(first != std::string::npos && queue.find(line, first + 1) == std::string::npos,
            "acknowledgement retry duplicated assignment");
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "pass case and fresh directory");
        const std::filesystem::path root = argv[2];
        std::filesystem::create_directories(root);
        const std::string mode = argv[1];
        if (mode == "cache-race")
            cache_race(root);
        else if (mode == "cache-bounds")
            cache_bounds(root);
        else if (mode == "outbox-race")
            outbox_race(root);
        else if (mode == "outbox-failures")
            outbox_failures(root);
        else
            throw std::runtime_error("unknown case");
        std::cout << mode << ": PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
