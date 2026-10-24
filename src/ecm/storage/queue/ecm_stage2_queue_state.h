#pragma once
#include "ecm/storage/queue/stage2_progress.h"
#include <algorithm>
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
template<class MatchTask> bool import_legacy(const fs::path &path,const Task &task,MatchTask match,State &out,
    const std::function<std::vector<uint64_t>()> &legacy_bounds={}) {
    State old;if(!old.load(path))return false;
    if(old.version==2) {
        auto previous=task;previous.total=old.total;
        std::vector<uint64_t> bounds{task.b2,0};
        auto matches=[&](uint64_t b2){previous.b2=b2;previous.auto_b2=!b2;return previous.legacy_id()==old.identity;};
        auto found=std::find_if(bounds.begin(),bounds.end(),matches);
        if(found==bounds.end() && legacy_bounds) {const auto extra=legacy_bounds();bounds.insert(bounds.end(),extra.begin(),extra.end());found=std::find_if(bounds.begin(),bounds.end(),matches);}
        if(found==bounds.end())return false;
        old.publication_task=old.identity;old.b2=*found;old.auto_b2=!*found;
    } else if(old.version==3) {
        if(old.identity!=task.id())return false;
    } else {
        const LegacyIdentity identity(old.identity);
        if(identity.save_sha256!=task.save_sha256 || identity.skip!=task.skip ||
           !match(identity.task))return false;
        old.pending_results=old.pending.empty()?std::string{}:identity.results;
        auto previous=task;previous.total=old.total;previous.b2=identity.b2;previous.auto_b2=identity.auto_b2;
        old.publication_task=previous.legacy_id();old.b2=identity.b2;old.auto_b2=identity.auto_b2 && !identity.b2;
    }
    old.identity=task.id();old.version=3;out=std::move(old);return true;
}
}
