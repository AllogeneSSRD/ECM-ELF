#include "ecm/service/work/queue_workflow.h"
#include "../stage1/legacy_execution.h"
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "../../storage/transfer/ecm_stage2_transfer.h"
#include "../../storage/transfer/p95_transfer.h"
#include "ecm/platform/logging.h"
#include <cstdlib>
#include <iostream>
namespace ecm_stage1 {
QueueItem::QueueItem(){mpz_init(modulus);}
QueueItem::~QueueItem(){mpz_clear(modulus);}
static bool run_one(const QueuePaths &paths,const QueueItem &item,const TaskDependencies &tasks,int *processed) {
    const auto N=item.modulus;
    const double B1=item.b1,B2=item.b2;
    const auto curves=item.curves;
    const auto &save_name=item.save_file,&n_expr=item.expression,&aid=item.aid,&line=item.ticket;
    const auto &opt=item.options;
    const auto &published_line=item.published_line,&stage2_line=item.stage2_line,&auto_stage2_line=item.auto_stage2_line;
    const auto *p95=item.p95;const auto *stage2=item.stage2;
    const auto &worktodo_path=paths.worktodo,&finished_path=paths.finished;
    const auto &exe_dir=paths.exe_dir,&sync1=paths.sync1,&sync2=paths.sync2;
    const auto worker=paths.worker;const auto full_sync=paths.full_sync;const auto marker=item.marker;
    auto result=run_stage1_task({N,B1,B2,curves,save_name,/*append_save=*/true,n_expr,opt},tasks);

    const bool has_factors = result.has_factors();

    if (result.failure_source()==Stage1FailureSource::Preparation ||
        result.failure_source()==Stage1FailureSource::Publication) {
        ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged.\n",
                       result.failure_source()==Stage1FailureSource::Publication ? "Stage1 save/result publication failed" :
                                               "backend prepare failed");
        return false;
    }

    bool found_factor = false;
    /* D3 (docs/usage/GUI.md): the hit line carries everything a consumer needs to
       build a results record, because each back-end prints its own, mutually
       incompatible hit line ("curve i sigma=M -> factor found" from the CPU paths,
       "GPU: factor ... with curve c (sigma p:s)" from the GPU ones). Fields: the
       factor, the curve index inside this batch, the EXACT per-curve sigma
       (result.sigma(), when the path knows it), the parametrization, which engine ran,
       and the save-file name (generated for GPU ECM=/ECM2= tasks). */
    const char *curve_token = ecm_stage1::curve_model_name(opt.execution.selection.curve);
    const char *target_token = ecm_stage1::execution_target_name(opt.execution.selection.target);
    if (has_factors) {
        for (uint32_t i = 0; i < result.curves(); ++i) {
            if (result.found(i) != ECM_NO_FACTOR_FOUND) {
                char *fs = mpz_get_str(nullptr, 10, result.factor(i));
                std::cout << "factor[" << i << "]=" << (fs ? fs : "?") << " curve=" << i;
                if (result.has_sigmas()) {
                    std::cout << " sigma=" << result.sigma(i);
                }
                if(opt.execution.selection.curve==ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                                                   ? opt.execution.selection.gpu_param : 0);
                std::cout << " curve_model=" << curve_token << " execution_target=" << target_token
                          << " save=" << save_name << "\n";
                free(fs);
                found_factor = true;
            }
        }
    }
    if (found_factor) {
        ecm_ts_fprintf(stdout, "FACTOR FOUND aid=%s task=%s\n",
                       aid.empty() ? "N/A" : aid.c_str(), line.c_str());
    }


    if (result.status()==Stage1TaskStatus::Stopped) {
        ecm_ts_fprintf(stdout, "STOP: Stage1 interrupted; worktodo unchanged: %s\n",
                       line.c_str());
        return false;
    }

    if (result.status()==Stage1TaskStatus::Failed) {
        ecm_ts_fprintf(stderr, "ERROR: stage1 failed for task: %s\n", line.c_str());
        // Marking the line is what stops the queue from re-running it: if that write
        // fails, abort instead of spinning on the same task forever.
        if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot mark the failed line in %s; aborting\n",
                           worktodo_path.c_str());
            return false;
        }
        return true;
    }

    const auto &completed_line=published_line.empty()?line:published_line;
    int stage2_receiver=stage2?stage2->worker:worker;
    if (stage2 != nullptr && !found_factor) {
        std::string transfer_error;
        if (!ecm_stage2_transfer_park(*stage2, stage2_line, transfer_error,&stage2_receiver)) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", transfer_error.c_str());
            return false;
        }
    }
    if (!ecm_append_text_line(finished_path, completed_line)) {
        ecm_ts_fprintf(stderr, "ERROR: cannot append to %s\n", finished_path.c_str());
        if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot mark the failed line in %s; aborting\n",
                           worktodo_path.c_str());
            return false;
        }
        return true;
    }
    // A Stage1 hit completes this factor search. Its saved point carries a
    // factor, so Stage2 cannot use it as a clean curve.
    const bool queue_stage2 = !found_factor && !auto_stage2_line.empty();
    if (!ecm_worktodo_advance(worktodo_path, worker, line,
            queue_stage2 ? WorktodoAction::Replace : WorktodoAction::Remove,
            queue_stage2 ? auto_stage2_line : std::string{})) {
        // The task itself succeeded (its .save is on disk), but the queue cannot drop
        // the line. Abort: continuing would re-run it and append a second time.
        ecm_ts_fprintf(stderr, "ERROR: cannot advance the finished line in %s; aborting "
                               "(the task's .save is already written)\n",
                       worktodo_path.c_str());
        return false;
    }

    ecm_sync_save_files(exe_dir, sync1, sync2, full_sync, marker);
    if (!opt.persistence.tmp_dir.empty() && opt.persistence.tmp_dir != exe_dir)
        ecm_sync_save_files(opt.persistence.tmp_dir, sync1, sync2, full_sync, marker);

    if (stage2 != nullptr && !found_factor) {
        size_t remaining = 0;
        std::string transfer_error;
        const bool delivered = ecm_stage2_transfer_flush(stage2->pending, remaining, transfer_error);
        if (delivered)
            ecm_ts_fprintf(stdout, "stage2_transfer: ok worker=%d file=\"%s\"\n",
                           stage2_receiver, stage2->worktodo.c_str());
        else
            ecm_ts_fprintf(stdout, "stage2_transfer: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                           stage2_receiver, remaining, stage2->worktodo.c_str(),
                           p95_transfer_quote(transfer_error).c_str());
    }

    /* ── Prime95 handoff (docs/usage/GUI.md) ──────────────────────────────────
       The task is finished and its .save is synchronized, so Prime95's stage 2 can take
       it from worktodo.add. AID, bounds and known factors are preserved; an ECMSTAGE2
       save token names the actual curve-prefixed file. Other fields remain verbatim.
       and a handoff failure NEVER fails the task: the line is parked and re-delivered
       with the next successful delivery, and the GUI shows a red notice meanwhile.
       Supported handoff remains available after local result publication. */
    if (p95 != nullptr && !p95->worktodo_path.empty()) {
        std::string p95_line=completed_line;
        if(p95_line.compare(0,10,"ECMSTAGE1=")==0)p95_line.replace(0,10,"ECMSTAGE2=");
        const P95TransferResult tr = p95_transfer_deliver(*p95, p95_line);
        if (!tr.ok) {
            ecm_ts_fprintf(stdout,
                           "p95_add: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                           tr.worker, tr.pending_left, tr.add_path.c_str(),
                           p95_transfer_quote(tr.error).c_str());
        } else if (!tr.note.empty()) {
            ecm_ts_fprintf(stdout,
                           "p95_add: warn worker=%d added=%zu pending_delivered=%zu recovered=%zu "
                           "aid=%s file=\"%s\" note=\"%s\"\n",
                           tr.worker, tr.added, tr.pending_delivered, tr.recovered,
                           tr.aid_dropped ? "dropped" : "kept", tr.add_path.c_str(),
                           p95_transfer_quote(tr.note).c_str());
        } else {
            ecm_ts_fprintf(stdout,
                           "p95_add: ok worker=%d added=%zu pending_delivered=%zu recovered=%zu "
                           "aid=%s file=\"%s\"\n",
                           tr.worker, tr.added, tr.pending_delivered, tr.recovered,
                           tr.aid_dropped ? "dropped" : "kept", tr.add_path.c_str());
        }
    }

    if (processed) {
        (*processed)++;
    }
    return true;
}

int run_queue_workflow(const QueuePaths &paths,const NextQueueTask &next_task,const TaskDependencies &tasks) {
    int processed=0;
    for(;;) {
        if(ecm_stage1_stop_requested()) {
            ecm_ts_fprintf(stdout,"STOP: queue interrupted; remaining worktodo unchanged\n");
            return 1;
        }
        QueueItem item;
        const auto read=next_task(item);
        if(read==QueueReadStatus::Error)return 1;
        if(read==QueueReadStatus::Empty)break;
        if(read==QueueReadStatus::MarkError) {
            // Preserve the current input-error policy: a failed mark ends this loop with 0.
            if(!ecm_worktodo_advance(paths.worktodo,paths.worker,item.ticket,WorktodoAction::MarkError))break;
            continue;
        }
        if(!run_one(paths,item,tasks,&processed))return 1;
    }
    ecm_ts_fprintf(stdout,"===== queue done, %d task(s) processed =====\n",processed);
    return 0;
}
} // namespace ecm_stage1
