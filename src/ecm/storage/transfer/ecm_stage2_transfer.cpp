#include "ecm_stage2_transfer.h"
#include "io/worktodo.h"
#include "sync/files.h"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {
struct Pending {
    std::string worktodo, finished, line;
    int worker = 0;
};

std::string encode(const Pending& p) {
    std::ostringstream out;
    out << std::quoted(p.worktodo) << ' ' << std::quoted(p.finished) << ' ' << p.worker << ' '
        << std::quoted(p.line);
    return out.str();
}

bool decode(const std::string& raw, Pending& p) {
    std::istringstream in(raw);
    in >> std::quoted(p.worktodo) >> std::quoted(p.finished) >> p.worker >> std::quoted(p.line);
    in >> std::ws;
    return !in.fail() && in.eof() && p.worker > 0 && !p.worktodo.empty() && !p.finished.empty() &&
           p.line.compare(0, 10, "ECMSTAGE2=") == 0;
}

std::string read_pending(const std::string& path) {
    const auto file = std::filesystem::u8path(path);
    return std::filesystem::exists(file) ? ecm_sync::read_file(file) : std::string{};
}

std::vector<Pending> parse_pending(const std::string& bytes, const std::string& path) {
    std::vector<Pending> entries;
    std::istringstream input(bytes);
    std::string line;
    while (std::getline(input, line)) {
        Pending item;
        if (!decode(line, item))
            throw std::runtime_error("invalid Stage2 pending entry; retained: " + path);
        entries.push_back(std::move(item));
    }
    return entries;
}
} // namespace

bool ecm_stage2_transfer_park(const Stage2TransferConfig& cfg, const std::string& line,
                              std::string& error, int* receiver) {
    error.clear();
    if (cfg.worktodo.empty() || cfg.finished.empty() || cfg.pending.empty() || cfg.worker < 1 ||
        cfg.worker > 1000000 || line.compare(0, 10, "ECMSTAGE2=") != 0 ||
        line.find_first_of("\r\n") != std::string::npos) {
        error = "invalid Stage2 transfer";
        return false;
    }
    try {
        const auto file = std::filesystem::u8path(cfg.pending);
        if (!file.parent_path().empty())
            std::filesystem::create_directories(file.parent_path());
        // Park and flush share the outbox lock. Take it before choosing a receiver
        // so outstanding assignments participate in the same load snapshot.
        ecm_sync::FileLock lock(cfg.pending);
        auto bytes = read_pending(cfg.pending);
        const auto entries = parse_pending(bytes, cfg.pending);
        int selected = cfg.worker;
        auto workers = cfg.workers;
        if (cfg.auto_workers || !workers.empty()) {
            std::map<int, size_t> loads;
            std::string load_error;
            // An unavailable target still permits parking; retry will deliver later.
            (void)ecm_worktodo_worker_loads(cfg.worktodo, loads, load_error);
            if (cfg.auto_workers) {
                workers.clear();
                for (const auto& load : loads)
                    workers.push_back(load.first);
                if (cfg.configured_workers < 1 || cfg.configured_workers > 1000000) {
                    error = "invalid configured Stage2 worker count";
                    return false;
                }
                if (workers.empty())
                    for (int i = 1; i <= cfg.configured_workers; ++i)
                        workers.push_back(i);
            }
            std::sort(workers.begin(), workers.end());
            if (workers.empty() || workers.front() < 1 || workers.back() > 1000000) {
                error = "invalid Stage2 receiver worker list";
                return false;
            }
            // Include this source's undelivered assignments in the load snapshot.
            for (const auto& old : entries)
                if (old.worktodo == cfg.worktodo)
                    ++loads[old.worker];
            const auto best = std::min_element(workers.begin(), workers.end(),
                                               [&](int a, int b) { return loads[a] < loads[b]; });
            selected = *best;
        }
        const auto finished = cfg.finished_by_worker.find(selected);
        const auto& finished_path =
            finished == cfg.finished_by_worker.end() ? cfg.finished : finished->second;
        if (finished_path.empty()) {
            error = "empty Stage2 receiver finished path";
            return false;
        }
        const std::string entry = encode({cfg.worktodo, finished_path, line, selected});
        if (!bytes.empty() && bytes.back() != '\n')
            bytes += '\n';
        bytes += entry + '\n';
        // The source queue may advance only after its handoff responsibility is
        // durable, rather than merely flushed from the C++ stream buffer.
        ecm_sync::write_atomic(file, bytes);
        if (receiver)
            *receiver = selected;
        return true;
    } catch (const std::exception& failure) {
        error = "cannot park Stage2 transfer (" + cfg.pending + "): " + failure.what();
        return false;
    }
}

bool ecm_stage2_transfer_flush(const std::string& pending, size_t& remaining, std::string& error) {
    remaining = 0;
    error.clear();
    try {
        const auto file = std::filesystem::u8path(pending);
        // A missing parent cannot contain an outbox or its lock yet.
        if (!file.parent_path().empty() && !std::filesystem::exists(file.parent_path()))
            return true;
        // Lock order: outbox, then the target's short queue edit lock. Neither
        // is held while computing a Stage1 batch.
        ecm_sync::FileLock lock(pending);
        const auto entries = parse_pending(read_pending(pending), pending);
        remaining = entries.size();
        size_t consumed = 0;
        for (; consumed < entries.size(); ++consumed) {
            const auto& entry = entries[consumed];
            if (!ecm_worktodo_append_unique(entry.worktodo, entry.finished, entry.worker,
                                            entry.line, error))
                break;
        }
        if (consumed == 0)
            return remaining == 0;
        std::string bytes;
        for (size_t i = consumed; i < entries.size(); ++i)
            bytes += encode(entries[i]) + '\n';
        // If acknowledgement fails, leave all original entries for idempotent
        // replay against the target's active/finished task checks.
        ecm_sync::write_atomic(file, bytes);
        remaining -= consumed;
        return remaining == 0;
    } catch (const std::exception& failure) {
        error = "cannot flush Stage2 transfer (" + pending + "): " + failure.what();
        return false;
    }
}
