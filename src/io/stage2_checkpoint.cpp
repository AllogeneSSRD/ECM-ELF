#include "ecm/io/stage2_checkpoint.h"
#include "ecm/platform/fingerprint.h"
#include "io/json.h"
#include <windows.h>
#include <fstream>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <set>

namespace stage2_queue {
struct LegacyIdentity {
    std::string task, queue, queue_sha256, save, save_sha256, results;
    uint64_t skip = 0, requested = 0, b2 = 0, d = 0;
    bool auto_b2 = false;
    explicit LegacyIdentity(const std::string& text);
};

bool import_legacy(const fs::path& path, const Task& task,
                   const std::function<bool(const std::string&)>& match, State& out,
                   const std::function<std::vector<uint64_t>()>& legacy_bounds) {
    State old;
    if (!old.load(path))
        return false;
    if (old.version == 2) {
        auto previous = task;
        previous.total = old.total;
        std::vector<uint64_t> bounds{task.b2, 0};
        auto matches = [&](uint64_t b2) {
            previous.b2 = b2;
            previous.auto_b2 = !b2;
            return previous.legacy_id() == old.identity;
        };
        auto found = std::find_if(bounds.begin(), bounds.end(), matches);
        if (found == bounds.end() && legacy_bounds) {
            const auto extra = legacy_bounds();
            bounds.insert(bounds.end(), extra.begin(), extra.end());
            found = std::find_if(bounds.begin(), bounds.end(), matches);
        }
        if (found == bounds.end())
            return false;
        old.publication_task = old.identity;
        old.b2 = *found;
        old.auto_b2 = !*found;
    } else if (old.version >= 3) {
        if (old.identity != task.id())
            return false;
    } else {
        const LegacyIdentity identity(old.identity);
        if (identity.save_sha256 != task.save_sha256 || identity.skip != task.skip ||
            !match(identity.task))
            return false;
        old.pending_results = identity.results;
        auto previous = task;
        previous.total = old.total;
        previous.b2 = identity.b2;
        previous.auto_b2 = identity.auto_b2;
        old.publication_task = previous.legacy_id();
        old.b2 = identity.b2;
        old.auto_b2 = identity.auto_b2 && !identity.b2;
    }
    old.identity = task.id();
    out = std::move(old);
    return true;
}

bool State::load(const fs::path& path) {
    if (!fs::exists(path))
        return false;
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read Stage2 progress: " + path.u8string());
    std::string bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.compare(0, 2, "5 ") == 0) {
        constexpr size_t digest_line = 65;
        if (bytes.size() <= digest_line || bytes.back() != '\n' ||
            bytes[bytes.size() - digest_line - 1] != '\n' ||
            ecm_stage2::sha256_text(bytes.substr(0, bytes.size() - digest_line)) !=
                bytes.substr(bytes.size() - digest_line, digest_line - 1))
            throw std::runtime_error("Stage2 checkpoint checksum mismatch: " + path.u8string());
        bytes.resize(bytes.size() - digest_line);
    }
    std::istringstream in(bytes);
    unsigned schema = 0;
    if (!(in >> schema >> done >> total >> std::quoted(run) >> std::quoted(pending) >>
          std::quoted(identity)) ||
        (schema < 1 || schema > 5) || done > total ||
        (done == total && !pending.empty()) || run.empty())
        throw std::runtime_error("invalid Stage2 progress file; inspect: " + path.string());
    version = schema;
    pending_results.clear();
    publication_task.clear();
    b2 = 0;
    auto_b2 = false;
    reserved.clear();
    submission_path.clear();
    latest_nf.clear();
    final_nf.clear();
    latest_nf_position = 0;
    nf_keys.clear();
    factors.clear();
    if (schema >= 2 && !(in >> std::quoted(pending_results)))
        throw std::runtime_error("missing Stage2 receipt path");
    if (schema >= 2 &&
        (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != identity.npos))
        throw std::runtime_error("invalid Stage2 task ID");
    if (schema >= 3) {
        unsigned mode = 0;
        if (!(in >> std::quoted(publication_task) >> b2 >> mode) || mode > 1 || (mode && b2) ||
            publication_task.size() != 64 ||
            publication_task.find_first_not_of("0123456789abcdef") != publication_task.npos)
            throw std::runtime_error("invalid Stage2 execution policy or publication task");
        auto_b2 = mode != 0;
    }
    if (schema == 4) {
        uint64_t count = 0;
        if (!pending.empty() || !pending_results.empty() || !(in >> count) || !count || count > total - done)
            throw std::runtime_error("invalid Stage2 reserved curves");
        std::set<std::string> receipts;
        for (uint64_t i = 0; i < count; ++i) {
            Reservation entry;
            unsigned completed = 0;
            if (!(in >> entry.position >> completed >> std::quoted(entry.receipt) >> std::quoted(entry.results)) ||
                entry.position != done + i || completed > 1 || (i == 0 && completed) ||
                entry.receipt.empty() || entry.results.empty() || !receipts.insert(entry.receipt).second)
                throw std::runtime_error("invalid Stage2 reserved curve");
            entry.completed = completed != 0;
            reserved.push_back(std::move(entry));
        }
    }
    if (schema == 5) {
        uint64_t count = 0, nf = 0, hits = 0;
        if (bytes.compare(0, 2, "5 ") || !pending.empty() || !pending_results.empty() ||
            !(in >> std::quoted(submission_path) >> latest_nf_position >> std::quoted(latest_nf) >>
              std::quoted(final_nf) >> count >> nf >> hits) ||
            count > total - done || nf > total || hits > total - nf ||
            bool(nf) != !latest_nf.empty() || bool(nf) != bool(latest_nf_position) ||
            latest_nf_position > done + count || (!final_nf.empty() && (!nf || done != total)))
            throw std::runtime_error("invalid Stage2 checkpoint results");
        for (uint64_t i = 0; i < count; ++i) {
            unsigned completed = 0;
            if (!(in >> completed) || completed > 1 || (!i && completed))
                throw std::runtime_error("invalid Stage2 checkpoint reservation");
            reserved.push_back({done + i, completed != 0, {}, {}});
        }
        const uint64_t completed = done + std::count_if(reserved.begin(), reserved.end(),
            [](const Reservation& entry) { return entry.completed; });
        if (nf + hits > completed ||
            (latest_nf_position > done && !reserved[latest_nf_position - done - 1].completed))
            throw std::runtime_error("Stage2 results exceed completed checkpoint progress");
        for (uint64_t i = 0; i < nf; ++i) {
            std::string key;
            if (!(in >> key) || key.size() != 64 ||
                key.find_first_not_of("0123456789abcdef") != key.npos ||
                !nf_keys.insert(std::move(key)).second)
                throw std::runtime_error("invalid Stage2 checkpoint NF key");
        }
        for (uint64_t i = 0; i < hits; ++i) {
            std::string factor;
            if (!(in >> std::quoted(factor)) || factor.empty())
                throw std::runtime_error("invalid Stage2 checkpoint factor");
            factors.push_back(std::move(factor));
        }
    }
    in >> std::ws;
    if (!in.eof())
        throw std::runtime_error("trailing data in Stage2 progress file: " + path.string());
    return true;
}

void State::save(const fs::path& path) const {
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    const fs::path temp(path.string() + ".tmp." + std::to_string(GetCurrentProcessId()));
    std::ostringstream out;
    const bool multiple = reserved.size() > 1;
    const auto &receipt = reserved.size() == 1 ? reserved.front().receipt : pending;
    const auto &results = reserved.size() == 1 ? reserved.front().results : pending_results;
    const bool self_contained = version == 5;
    out << (self_contained ? "5 " : multiple ? "4 " : "3 ") << done << ' ' << total << ' ' << std::quoted(run) << ' '
        << std::quoted(self_contained ? std::string{} : receipt) << ' ' << std::quoted(identity) << ' '
        << std::quoted(self_contained ? std::string{} : results) << ' '
        << std::quoted(publication_task) << ' ' << b2 << ' ' << unsigned(auto_b2);
    if (self_contained) {
        out << ' ' << std::quoted(submission_path) << ' ' << latest_nf_position << ' '
            << std::quoted(latest_nf) << ' ' << std::quoted(final_nf) << ' '
            << reserved.size() << ' ' << nf_keys.size() << ' ' << factors.size();
        for (const auto& entry : reserved)
            out << ' ' << unsigned(entry.completed);
        for (const auto& key : nf_keys)
            out << '\n' << key;
        for (const auto& factor : factors)
            out << '\n' << std::quoted(factor);
    } else if (multiple) {
        out << ' ' << reserved.size();
        for (const auto& entry : reserved)
            out << '\n' << entry.position << ' ' << unsigned(entry.completed) << ' '
                << std::quoted(entry.receipt) << ' ' << std::quoted(entry.results);
    }
    out << '\n';
    auto bytes = out.str();
    if (self_contained)
        bytes += ecm_stage2::sha256_text(bytes) + '\n';
    HANDLE file = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot create Stage2 progress file");
    DWORD written = 0;
    const bool ok =
        bytes.size() <= MAXDWORD &&
        WriteFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) &&
        written == bytes.size() && FlushFileBuffers(file);
    CloseHandle(file);
    if (!ok || !MoveFileExW(temp.c_str(), path.c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish Stage2 progress; previous state retained: " +
                                 path.string());
}

std::string Task::legacy_id() const {
    std::ostringstream out;
    out << "stage2-curves-v2 " << std::quoted(aid) << ' ' << std::quoted(n) << ' ' << save_sha256
        << ' ' << skip << ' ' << total << ' ' << b2 << ' ' << (auto_b2 && !b2);
    return ecm_stage2::sha256_text(out.str());
}

std::string Task::id() const {
    std::ostringstream out;
    out << "stage2-curves-v3 " << std::quoted(aid) << ' ' << std::quoted(n) << ' ' << save_sha256
        << ' ' << skip;
    return ecm_stage2::sha256_text(out.str());
}

void State::reconfigure(const Task& task) {
    const auto key = task.id();
    if (!identity.empty() && identity != key)
        throw std::runtime_error("Stage2 checkpoint input identity mismatch");
    const bool mode = task.auto_b2 && !task.b2;
    if (!final_nf.empty() && (total != (std::max)(done, task.total) || b2 != task.b2 || auto_b2 != mode))
        throw std::runtime_error("Stage2 has a staged NF report; finish publication under its original policy");
    if (version == 5 && !reserved.empty() && task.total < done + reserved.size())
        throw std::runtime_error("Stage2 has completed curves beyond a gap; fill the gap before shrinking the task");
    if (version < 5 && (!pending.empty() || !reserved.empty()) &&
        (total != task.total || b2 != task.b2 || auto_b2 != mode))
        throw std::runtime_error("Stage2 has an unconfirmed curve; recover its original B2 and "
                                 "curve count before changing the task");
    if (identity.empty()) {
        identity = key;
        publication_task = key;
        run = token();
    }
    total = (std::max)(done, task.total);
    b2 = task.b2;
    auto_b2 = mode;
}

fs::path load_task(const fs::path& directory, size_t bits, const Task& task, State& out,
                   const std::function<std::vector<uint64_t>()>& legacy_bounds,
                   fs::path* destination) {
    const auto base = task.path(directory, bits);
    if (destination)
        *destination = base;
    if (!fs::exists(directory))
        return {};
    const auto stem = base.stem().u8string();
    const auto prefix = ".ecm_ckpt_stage2_" + std::to_string(bits) + '_';
    std::set<fs::path> occupied;
    fs::path current_path;
    const auto identity = task.id();
    std::vector<fs::path> candidates;
    for (const auto& file : fs::directory_iterator(directory)) {
        if (file.path().extension() != ".dat")
            continue;
        const auto name = file.path().stem().u8string();
        if (name == stem || name.compare(0, stem.size()+1, stem+'-') == 0) {
            occupied.insert(file.path().filename());
            State state;
            if (!state.load(file.path()) || state.version != 5)
                throw std::runtime_error("invalid Stage2 B1 checkpoint: " + file.path().u8string());
            if (state.identity == identity) {
                if (!current_path.empty())
                    throw std::runtime_error("multiple Stage2 B1 checkpoints match this input");
                current_path = file.path();
                out = std::move(state);
            }
        } else if (name.size() == prefix.size()+64 && name.compare(0, prefix.size(), prefix) == 0 &&
                   name.find_first_not_of("0123456789abcdef", prefix.size()) == name.npos) {
            candidates.push_back(file.path());
        }
    }
    if (destination) {
        unsigned suffix = 0;
        while (occupied.count(destination->filename()))
            *destination = task.path(directory, bits, ++suffix);
    }
    const auto old_path = task.legacy_path(directory,bits);
    if (!current_path.empty()) {
        if (destination)
            *destination = current_path;
        // A previous start may have written the new file just before archiving.
        for (const auto& old : {old_path,directory/fs::u8path(prefix+out.publication_task+".dat")}) {
            State previous;
            if (previous.load(old) && previous.run == out.run)
                return old;
        }
        return current_path;
    }
    if (out.load(old_path)) {
        if (out.version < 3 || out.identity != identity)
            throw std::runtime_error("Stage2 legacy checkpoint input identity mismatch");
        return old_path;
    }
    std::optional<std::vector<uint64_t>> cached;
    const auto bounds = [&]() -> const std::vector<uint64_t>& {
        if (!cached)
            cached = legacy_bounds ? legacy_bounds() : std::vector<uint64_t>{};
        return *cached;
    };
    fs::path source;
    for (const auto& candidate : candidates) {
        State old;
        if (!old.load(candidate) || old.version != 2 ||
            !import_legacy(candidate,task,{},old,bounds))
            continue;
        if (!source.empty())
            throw std::runtime_error("multiple legacy Stage2 checkpoints match this input; "
                                     "preserve them and resolve the duplicate tasks");
        source = candidate;
        out = std::move(old);
    }
    return source;
}

fs::path Task::path(const fs::path& directory, size_t bits, unsigned suffix) const {
    return directory / (".ecm_ckpt_stage2_" + std::to_string(bits) + '_' + std::to_string(b1) +
                        (suffix ? "-" + std::to_string(suffix) : "") + ".dat");
}

fs::path Task::legacy_path(const fs::path& directory, size_t bits) const {
    return directory / (".ecm_ckpt_stage2_" + std::to_string(bits) + '_' + id() + ".dat");
}

std::string token() {
    std::random_device source;
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i)
        out << std::setw(8) << source();
    return out.str();
}

void State::advance_completed() {
    const auto first_unfinished = std::find_if(reserved.begin(),reserved.end(),
        [](const Reservation& entry) {return !entry.completed;});
    done += first_unfinished - reserved.begin();
    reserved.erase(reserved.begin(),first_unfinished);
}

void State::complete(Reservation& entry, const std::string& outcome) {
    const auto message = ecm_results::json::parse(outcome);
    if (entry.completed)
        throw std::runtime_error("duplicate Stage2 checkpoint outcome");
    record_outcome(entry.position, message.get("report").dump(), message.get("key").text());
    entry.completed = true;
    advance_completed();
}

void State::record_outcome(uint64_t position, const std::string& report, std::string key) {
    const auto row = ecm_results::json::parse(report);
    const auto status = row.get("status").text();
    if (row.get("curves").text() != "1" ||
        (status != "NF" && status != "F"))
        throw std::runtime_error("invalid Stage2 checkpoint outcome");
    if (status == "F") {
        factors.push_back(report);
    } else {
        if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != key.npos)
            throw std::runtime_error("invalid Stage2 checkpoint outcome key");
        if (nf_keys.insert(std::move(key)).second && position + 1 > latest_nf_position) {
            latest_nf_position = position + 1;
            latest_nf = report;
        }
    }
}

bool result_written(const fs::path& path, const std::string& receipt) {
    if (receipt.empty() || !fs::exists(path))
        return false;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot inspect Stage2 result receipts");

    std::string line;
    while (std::getline(in, line)) {
        // An unterminated last line may be a torn append. Never acknowledge it.
        if (in.eof())
            break;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        try {
            const auto row = ecm_results::json::parse(line);
            if (row.type == 'o' && row.get("queue_receipt").text() == receipt)
                return true;
        } catch (
            const std::runtime_error&) { /* A torn/invalid append is never a durable receipt. */
        }
    }
    if (in.bad())
        throw std::runtime_error("cannot read Stage2 result receipts");
    return false;
}

bool finished_written(const fs::path& path, const std::string& run, const std::string& task) {
    if (!fs::exists(path))
        return false;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot inspect Stage2 finished receipts");
    const std::string marker = "# stage2_task_id=" + run;
    std::string previous, line;
    while (std::getline(in, line)) {
        if (in.eof())
            break;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (previous == marker && line == task)
            return true;
        previous = line;
    }
    if (in.bad())
        throw std::runtime_error("cannot read Stage2 finished receipts");
    return false;
}

LegacyIdentity::LegacyIdentity(const std::string& text) {
    std::istringstream in(text);
    if (!(in >> std::quoted(task) >> std::quoted(queue) >> queue_sha256 >> std::quoted(save) >>
          save_sha256 >> std::quoted(results) >> skip >> requested >> b2 >> d >> auto_b2))
        throw std::runtime_error("cannot parse legacy Stage2 task identity");
}
} // namespace stage2_queue
