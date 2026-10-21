#include "ecm_stage2_transfer.h"
#include "../queue/ecm_worktodo.h"
#include <algorithm>

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
                              const std::string &line, std::string &error,int *receiver) {
    error.clear();
    if (cfg.worktodo.empty() || cfg.finished.empty() || cfg.pending.empty() ||
        cfg.worker < 1 || cfg.worker > 1000000 || line.compare(0, 10, "ECMSTAGE2=") != 0 ||
        line.find_first_of("\r\n") != std::string::npos) {
        error = "invalid Stage2 transfer";
        return false;
    }
    int selected=cfg.worker;
    auto workers=cfg.workers;
    if(cfg.auto_workers || !workers.empty()) {
        std::map<int,size_t> loads;std::string load_error;
        // An unavailable target still permits parking; retry will deliver later.
        (void)ecm_worktodo_worker_loads(cfg.worktodo,loads,load_error);
        if(cfg.auto_workers) {
            workers.clear();for(const auto &load:loads)workers.push_back(load.first);
            if(cfg.configured_workers<1 || cfg.configured_workers>1000000) {
                error="invalid configured Stage2 worker count";return false;
            }
            if(workers.empty())for(int i=1;i<=cfg.configured_workers;++i)workers.push_back(i);
        }
        std::sort(workers.begin(),workers.end());
        if(workers.empty() || workers.front()<1 || workers.back()>1000000) {
            error="invalid Stage2 receiver worker list";return false;
        }
        // Include this source's undelivered assignments in the load snapshot.
        std::ifstream parked(std::filesystem::u8path(cfg.pending),std::ios::binary);
        std::string raw;while(std::getline(parked,raw)) {
            Pending old;if(decode(raw,old)&&old.worktodo==cfg.worktodo)++loads[old.worker];
        }
        const auto best=std::min_element(workers.begin(),workers.end(),[&](int a,int b){return loads[a]<loads[b];});
        selected=*best;
    }
    if(receiver)*receiver=selected;
    const auto finished=cfg.finished_by_worker.find(selected);
    const auto &finished_path=finished==cfg.finished_by_worker.end()?cfg.finished:finished->second;
    if(finished_path.empty()){error="empty Stage2 receiver finished path";return false;}
    const std::string entry = encode({cfg.worktodo, finished_path, line, selected});
    std::ofstream out(std::filesystem::u8path(cfg.pending), std::ios::binary | std::ios::app);
    if (out) {
        out << entry << '\n';
        out.flush();
        out.close();
    }
    if (!out) {
        error = "cannot append Stage2 pending transfer: " + cfg.pending;
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
