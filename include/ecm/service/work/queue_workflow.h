#pragma once
#include "ecm/service/stage1/task.h"
#include <functional>
#include <string>
struct P95TransferConfig;
struct Stage2TransferConfig;
namespace ecm_stage1 {
struct QueuePaths {
    std::string worktodo,finished,exe_dir,sync1,sync2;
    int worker=1;
    bool full_sync=false;
};
struct QueueItem {
    // Owns N, strings and options. Transfer pointers borrow application state
    // for the synchronous workflow; the source must not retain this item.
    mpz_t modulus;
    double b1=0,b2=0;
    uint32_t curves=0;
    std::string save_file,expression,aid,ticket,published_line,stage2_line,auto_stage2_line;
    Stage1Options options;
    long long marker=0;
    const P95TransferConfig *p95=nullptr;
    const Stage2TransferConfig *stage2=nullptr;
    QueueItem();
    ~QueueItem();
    QueueItem(const QueueItem &)=delete;
    QueueItem &operator=(const QueueItem &)=delete;
};
// MarkError advances only the original ticket; Error preserves it and stops.
enum class QueueReadStatus { Ready, Empty, Error, MarkError };
using NextQueueTask=std::function<QueueReadStatus(QueueItem &)>;
// Source inputs are already normalized; task and source completion each have one owner.
int run_queue_workflow(const QueuePaths &,const NextQueueTask &,const TaskDependencies &);
}
