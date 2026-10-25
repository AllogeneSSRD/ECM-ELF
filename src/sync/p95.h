#pragma once

#include "ecm/storage/save/range.h"
#include "ecm/storage/transfer/p95_transfer.h"

namespace ecm_sync {
// Persist responsibility before advancing the local queue. No task is published
// here. Retry validates the recorded source version before copying its save.
bool park_p95(const P95TransferConfig&, const std::string& stage2_line, const ecm_save::Range&,
              std::string& error);

// Publish the validated save first, then its worktodo.add assignment. Failures
// retain responsibility for retry; an ambiguous send is never blindly replayed.
bool flush_p95(const P95TransferConfig&, std::string& error);
size_t pending_p95_count(const P95TransferConfig&);
} // namespace ecm_sync
