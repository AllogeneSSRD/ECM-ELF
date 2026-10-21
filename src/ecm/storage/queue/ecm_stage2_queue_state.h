#pragma once
#include "ecm/storage/queue/stage2_progress.h"
#include <utility>
namespace stage2_queue {
struct LegacyIdentity {
    std::string task,queue,queue_sha256,save,save_sha256,results;
    uint64_t skip=0,requested=0,b2=0,d=0;
    bool auto_b2=false;
    explicit LegacyIdentity(const std::string &text);
};
// Task parsing/N reconstruction is supplied by the production queue parser.
// Ordering, queue path and unrelated settings never participate in this match.
template<class MatchTask> bool import_legacy(const fs::path &path,const Task &task,MatchTask match,State &out) {
    State old;if(!old.load(path))return false;
    if(old.version==2) {
        if(old.identity!=task.id())return false;
    } else {
        const LegacyIdentity identity(old.identity);
        if(old.total!=task.total || identity.save_sha256!=task.save_sha256 || identity.skip!=task.skip ||
           identity.b2!=task.b2 || (identity.auto_b2 && !identity.b2)!=(task.auto_b2 && !task.b2) ||
           !match(identity.task))return false;
        old.pending_results=old.pending.empty()?std::string{}:identity.results;
    }
    old.identity=task.id();old.version=2;out=std::move(old);return true;
}
}
