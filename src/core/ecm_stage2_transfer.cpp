#include "ecm_stage2_transfer.h"
#include "ecm_worktodo.h"

#include <filesystem>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
struct Pending {
    std::string worktodo, finished, line;
    int worker = 0;
};

std::string encode(const Pending &p) {
    std::ostringstream out;
    out << std::quoted(p.worktodo) << ' ' << std::quoted(p.finished) << ' '
        << p.worker << ' ' << std::quoted(p.line);
    return out.str();
}

bool decode(const std::string &raw, Pending &p) {
    std::istringstream in(raw);
    in >> std::quoted(p.worktodo) >> std::quoted(p.finished) >> p.worker >> std::quoted(p.line);
    in >> std::ws;
    return !in.fail() && in.eof() && p.worker > 0 && !p.worktodo.empty() &&
           !p.finished.empty() && p.line.compare(0, 10, "ECMSTAGE2=") == 0;
}

bool replace(const std::string &source, const std::string &target) {
#ifdef _WIN32
    return MoveFileExW(std::filesystem::u8path(source).c_str(), std::filesystem::u8path(target).c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(source.c_str(), target.c_str()) == 0;
#endif
}
}

bool ecm_stage2_transfer_park(const Stage2TransferConfig &cfg,
                              const std::string &line, std::string &error) {
    error.clear();
    if (cfg.worktodo.empty() || cfg.finished.empty() || cfg.pending.empty() ||
        cfg.worker < 1 || cfg.worker > 1000000 || line.compare(0, 10, "ECMSTAGE2=") != 0 ||
        line.find_first_of("\r\n") != std::string::npos) {
        error = "invalid Edwards Stage2 transfer";
        return false;
    }
    const std::string entry = encode({cfg.worktodo, cfg.finished, line, cfg.worker});
    std::ofstream out(std::filesystem::u8path(cfg.pending), std::ios::binary | std::ios::app);
    if (out) {
        out << entry << '\n';
        out.flush();
        out.close();
    }
    if (!out) {
        error = "cannot append Edwards Stage2 pending transfer: " + cfg.pending;
        return false;
    }
    return true;
}

bool ecm_stage2_transfer_flush(const std::string &pending, size_t &remaining,
                               std::string &error) {
    remaining = 0;
    error.clear();
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::u8path(pending), ec)) {
        if (ec) error = "cannot inspect Stage2 pending file: " + ec.message();
        return !ec;
    }
    std::ifstream in(std::filesystem::u8path(pending), std::ios::binary);
    if (!in) { error = "cannot read Stage2 pending file: " + pending; return false; }
    std::vector<Pending> entries;
    std::string raw;
    while (std::getline(in, raw)) {
        Pending item;
        if (!decode(raw, item)) {
            error = "invalid Stage2 pending entry; retained: " + pending;
            return false;
        }
        entries.push_back(std::move(item));
    }
    if (in.bad()) { error = "cannot finish reading Stage2 pending file"; return false; }
    in.close();
    size_t consumed = 0;
    for (; consumed < entries.size(); ++consumed) {
        if (!ecm_worktodo_append_unique(entries[consumed].worktodo, entries[consumed].finished,
                                        entries[consumed].worker, entries[consumed].line, error)) break;
    }
    remaining = entries.size() - consumed;
    if (consumed == 0) return remaining == 0;
    const std::string tmp = pending + ".new";
    std::ofstream out(std::filesystem::u8path(tmp), std::ios::binary | std::ios::trunc);
    if (!out) { error = "cannot write Stage2 pending temp file"; return false; }
    for (size_t i = consumed; i < entries.size(); ++i) out << encode(entries[i]) << '\n';
    out.close();
    if (!out || !replace(tmp, pending)) {
        error = "cannot update Stage2 pending file";
        return false;
    }
    return remaining == 0;
}
