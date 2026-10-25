#include "p95.h"
#include "files.h"
#include "io/worktodo.h"
#include "ecm/storage/save/publication.h"
#include "ecm/storage/queue/p95_worktodo.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ecm_sync {
namespace {
uint64_t checksum(const std::string& bytes, size_t prefix_size) {
    uint64_t value = 14695981039346656037ull;
    for (size_t i = 0; i < prefix_size; ++i) {
        value ^= static_cast<unsigned char>(bytes[i]);
        value *= 1099511628211ull;
    }
    return value;
}

std::filesystem::path pending_dir(const P95TransferConfig& cfg) {
    return std::filesystem::u8path(cfg.pending_path + ".handoff");
}

struct Intent {
    std::string source;
    std::string target;
    std::string line;
    std::string source_assignment;
    uint64_t size = 0;
    uint64_t sum = 0;
    uint64_t range_hash = 0;
    P95TransferConfig delivery;
};

// Field order is the on-disk version 3 protocol. Keep the frozen destination and
// routing policy when replaying an intent under a newer worker configuration.
std::string encode(const Intent& intent) {
    std::ostringstream out;
    out << "ECM-P95-HANDOFF 3\n"
        << intent.size << ' ' << intent.sum << ' ' << intent.range_hash << '\n'
        << std::quoted(intent.source) << '\n'
        << std::quoted(intent.target) << '\n'
        << std::quoted(intent.line) << '\n'
        << std::quoted(intent.source_assignment) << '\n'
        << std::quoted(intent.delivery.worktodo_path) << '\n'
        << std::quoted(intent.delivery.add_workers) << '\n'
        << std::quoted(intent.delivery.prime_log_path) << '\n'
        << std::quoted(intent.delivery.results_txt_path) << '\n'
        << std::quoted(intent.delivery.sent_ledger_path) << '\n'
        << intent.delivery.keep_aid << ' ' << intent.delivery.recover_lost << '\n';
    return out.str();
}

Intent decode(const std::string& bytes) {
    std::istringstream in(bytes);
    std::string magic;
    std::getline(in, magic);
    Intent intent;
    if (magic != "ECM-P95-HANDOFF 3" ||
        !(in >> intent.size >> intent.sum >> intent.range_hash >> std::quoted(intent.source) >>
          std::quoted(intent.target) >> std::quoted(intent.line) >>
          std::quoted(intent.source_assignment) >> std::quoted(intent.delivery.worktodo_path) >>
          std::quoted(intent.delivery.add_workers) >> std::quoted(intent.delivery.prime_log_path) >>
          std::quoted(intent.delivery.results_txt_path) >>
          std::quoted(intent.delivery.sent_ledger_path) >> intent.delivery.keep_aid >>
          intent.delivery.recover_lost))
        throw std::runtime_error("invalid P95 handoff intent");
    return intent;
}

bool contains_line(const std::string& file, const std::string& line,
                   const std::string& prefix = {}) {
    if (file.empty() || !std::filesystem::exists(std::filesystem::u8path(file)))
        return false;
    std::istringstream in(read_file(std::filesystem::u8path(file)));
    std::string actual;
    while (std::getline(in, actual)) {
        if (!actual.empty() && actual.back() == '\r')
            actual.pop_back();
        if (actual == prefix + line)
            return true;
    }
    return false;
}

void validate_ready(const std::string& line) {
    EcmStage2Task task;
    std::string error;
    if (!ecm_parse_stage2_line(line, task, error))
        throw std::runtime_error("pending P95 task is not a saved Stage2 task: " + error);
    if (!std::filesystem::u8path(task.save_name).is_absolute())
        throw std::runtime_error("pending P95 save must use an absolute path: " + task.save_name);
    const auto count = ecm_save::record_count(task.save_name);
    if (task.skip_curves > count || task.curves_to_run > count - task.skip_curves)
        throw std::runtime_error("pending P95 save does not contain its referenced curve range: " +
                                 task.save_name);
    const auto range =
        ecm_save::inspect_range(task.save_name, task.skip_curves, task.curves_to_run);
    mpz_t modulus;
    mpz_init(modulus);
    try {
        if (!ecm_compute_stage2_n(task, modulus, error))
            throw std::runtime_error(error);
        ecm_save::validate_p95_batch(range, modulus);
    } catch (...) {
        mpz_clear(modulus);
        throw;
    }
    mpz_clear(modulus);
}
} // namespace

bool park_p95(const P95TransferConfig& cfg, const std::string& line, const ecm_save::Range& range,
              std::string& error) {
    if (cfg.worktodo_path.empty())
        return true;
    try {
        if (cfg.sync_dir.empty())
            throw std::runtime_error("p95_sync_dir must name the save destination");
        const auto bytes = ecm_save::verified_snapshot(range);
        // Validate the local saved-point assignment before taking durable
        // delivery responsibility or exposing any external work line.
        validate_ready(line);

        Intent intent;
        intent.source = range.path;
        intent.delivery = cfg;
        intent.source_assignment = line;
        std::string identities;
        for (const auto& record : range.records)
            identities += record + '\n';
        intent.range_hash = checksum(identities, identities.size());
        intent.target = std::filesystem::absolute(std::filesystem::u8path(cfg.sync_dir) /
                                                  std::filesystem::u8path(range.path).filename())
                            .lexically_normal()
                            .u8string();
        std::string rewritten;
        if (!ecm_rewrite_stage2_save(line, intent.target, rewritten, error))
            return false;
        intent.line = rewritten;
        intent.size = bytes.size();
        intent.sum = checksum(bytes, bytes.size());
        const auto encoded = encode(intent);

        const auto folder = pending_dir(cfg);
        std::filesystem::create_directories(folder);
        FileLock lock(cfg.pending_path + ".handoff");
        // Identity excludes later source suffixes and delivery configuration:
        // retrying this batch must reuse its first persisted destination.
        const auto logical = intent.source + '\n' + intent.source_assignment + '\n' +
                             std::to_string(intent.range_hash);
        const auto path = folder / (std::to_string(checksum(logical, logical.size())) + ".intent");
        auto done = path;
        done += ".done";
        for (const auto& existing : {done, path}) {
            if (!std::filesystem::exists(existing))
                continue;

            const auto previous = decode(read_file(existing));
            if (previous.source != intent.source ||
                previous.source_assignment != intent.source_assignment ||
                previous.range_hash != intent.range_hash || bytes.size() < previous.size ||
                checksum(bytes, size_t(previous.size)) != previous.sum)
                throw std::runtime_error("P95 receipt/source identity conflict");
            return true; // Keep the original destination and responsibility frozen.
        }

        write_atomic(path, encoded);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool flush_p95(const P95TransferConfig& cfg, std::string& error) {
    if (cfg.worktodo_path.empty())
        return true;
    try {
        // Import old ready-line pending files without dropping them. Native ECM
        // lines cannot prove a saved range and require operator correction.
        if (std::filesystem::exists(std::filesystem::u8path(cfg.pending_path))) {
            const auto legacy = read_file(std::filesystem::u8path(cfg.pending_path));
            std::istringstream in(legacy);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty())
                    continue;

                validate_ready(line);
                EcmStage2Task task;
                std::string ignored;
                (void)ecm_parse_stage2_line(line, task, ignored);
                const auto range =
                    ecm_save::inspect_range(task.save_name, task.skip_curves, task.curves_to_run);
                if (!park_p95(cfg, line, range, error))
                    throw std::runtime_error(error);
            }
            if (!legacy.empty())
                write_atomic(std::filesystem::u8path(cfg.pending_path), "");
        }
        const auto folder = pending_dir(cfg);
        if (!std::filesystem::exists(folder))
            return true;

        FileLock delivery(cfg.pending_path + ".handoff");
        std::vector<std::filesystem::path> intents;
        for (const auto& entry : std::filesystem::directory_iterator(folder))
            if (entry.path().extension() == ".intent")
                intents.push_back(entry.path());
        std::sort(intents.begin(), intents.end());

        for (const auto& path : intents) {
            auto done = path;
            done += ".done";
            if (std::filesystem::exists(done)) {
                std::filesystem::remove(path);
                continue;
            }
            const auto encoded = read_file(path);
            const auto intent = decode(encoded);
            const auto source = std::filesystem::u8path(intent.source),
                       target = std::filesystem::u8path(intent.target);
            std::string bytes;
            {
                FileLock lock(intent.source);
                bytes = read_file(source);
            }
            // A source may grow after Stage1 completes; its recorded prefix must
            // remain unchanged. Release its lock before locking the destination.
            if (bytes.size() < intent.size || checksum(bytes, size_t(intent.size)) != intent.sum)
                throw std::runtime_error("P95 source save changed before delivery: " +
                                         intent.source);
            std::filesystem::create_directories(target.parent_path());
            {
                FileLock lock(intent.target);
                const auto existing =
                    std::filesystem::exists(target) ? read_file(target) : std::string{};
                const auto common = (std::min)(existing.size(), bytes.size());
                if (existing.compare(0, common, bytes, 0, common) != 0)
                    throw std::runtime_error(
                        "P95 destination save prefix conflicts; refusing to replace history: " +
                        intent.target);
                if (existing.size() < bytes.size())
                    write_atomic(target, bytes);
            }
            validate_ready(intent.line);

            // Legacy pending/recovery must also name already usable saves. Never
            // release a new add file while an older line references missing data.
            if (std::filesystem::exists(std::filesystem::u8path(cfg.pending_path))) {
                std::istringstream in(read_file(std::filesystem::u8path(cfg.pending_path)));
                std::string line;
                while (std::getline(in, line))
                    if (!line.empty())
                        validate_ready(line);
            }
            for (const auto& line : p95_transfer_recover_lines(intent.delivery))
                validate_ready(line);
            auto inflight = path;
            inflight += ".sending";
            const auto add = p95_transfer_add_path(intent.delivery.worktodo_path);
            if (std::filesystem::exists(inflight)) {
                // A previous process may have sent the task before crashing.
                // Without evidence of delivery, stop instead of sending twice.
                const bool known =
                    contains_line(add, intent.line) ||
                    contains_line(intent.delivery.worktodo_path, intent.line) ||
                    contains_line(intent.delivery.sent_ledger_path, intent.line, "sent ") ||
                    contains_line(intent.delivery.sent_ledger_path, p95_strip_aid(intent.line),
                                  "sent ");
                if (!known)
                    throw std::runtime_error("P95 delivery acknowledgement is ambiguous; retained "
                                             "intent requires inspection: " +
                                             path.u8string());
            } else {
                write_atomic(inflight, encoded);
                // Each intent owns its low-level retry file; another batch must
                // never flush an assignment whose save it has not validated.
                auto delivery_config = intent.delivery;
                delivery_config.pending_path = path.u8string() + ".ready";
                const auto result = p95_transfer_deliver(delivery_config, intent.line);
                if (!result.ok) {
                    std::filesystem::remove(inflight);
                    throw std::runtime_error(result.error);
                }
                ecm_ts_fprintf(stdout,
                               "p95_add: %s worker=%d added=%zu pending_delivered=%zu "
                               "recovered=%zu aid=%s file=\"%s\" note=\"%s\"\n",
                               result.note.empty() ? "ok" : "warn", result.worker, result.added,
                               result.pending_delivered, result.recovered,
                               result.aid_dropped ? "dropped" : "kept", result.add_path.c_str(),
                               p95_transfer_quote(result.note).c_str());
            }
            // Retain the durable acknowledgement for deduplication before
            // removing responsibility and the ambiguous-send marker.
            write_atomic(done, encoded);
            std::filesystem::remove(path);
            std::filesystem::remove(inflight);
        }
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

size_t pending_p95_count(const P95TransferConfig& cfg) {
    size_t count = p95_transfer_pending_count(cfg.pending_path);
    std::error_code error;
    const auto folder = pending_dir(cfg);
    if (!std::filesystem::exists(folder, error))
        return count;
    for (std::filesystem::directory_iterator it(folder, error), end; !error && it != end;
         it.increment(error))
        if (it->path().extension() == ".intent")
            ++count;
    return count;
}
} // namespace ecm_sync
