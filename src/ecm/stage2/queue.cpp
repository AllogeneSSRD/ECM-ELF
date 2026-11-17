#include "../../io/files.h"
// Backend-independent worktodo ownership, checkpoint recovery and publication.
#include "task.h"
#include "backend.h"
#include "input.h"
#include "process.h"
#include "../platform/compute_log.h"
#include "modulus.h"
#include "ecm/io/stage2_checkpoint.h"
#include "ecm/platform/fingerprint.h"
#include "ecm/platform/log_file.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/io/results.h"
#include "../../io/json.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
namespace s2prod {
namespace {
fs::path fatal_log;
int fatal_log_worker=1;
uint64_t fatal_log_max_kb=0;

void append_log(const fs::path &path,const std::string &message,int worker,uint64_t max_kb) {
    if(max_kb)ecm_append_worker_log(path,worker,max_kb,message.empty()?"\n":message);
    else append(path,ecm_format_worker_log_lines(message,worker));
}

void append_finished(const fs::path &path,const std::string &marker,
                     const std::string &task,const std::string &run={}) {
    // Stage1 uses this same domain for its finished file replacement. Keep
    // the Stage2 marker check and append within one lock for both stages.
    if(path.has_parent_path())fs::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string(),ecm_io::FileLockDomain::QueueEdit);
    if(run.empty() || !stage2_queue::finished_written(path,run,task))
        append(path,marker+'\n'+task);
}
}

void append_fatal_log(const std::string& message) {
    if(!fatal_log.empty())append_log(fatal_log,message,fatal_log_worker,fatal_log_max_kb);
}
namespace {
std::string checkpoint_nf_key(const Record& r, const std::string& b2) {
    return ecm_stage2::sha256_text(r.n+"|"+std::to_string(r.sigma)+"|"+
        std::to_string(r.b1)+"|"+b2+(r.edwards?"|ed":"|mont0"));
}

void migrate_queue_outcomes(stage2_queue::State& state, const fs::path& save, uint64_t skip,
                            const fs::path& results, const fs::path& submission, mpz_srcptr n) {
    if(!state.pending.empty())
        state.reserved.push_back({state.done,false,std::move(state.pending),state.pending_results});
    const uint64_t extent=state.done+state.reserved.size();
    if(extent) {
        const auto points=records(save,skip,extent);
        if(points.size()!=extent)
            throw std::runtime_error("legacy Stage2 progress exceeds available save records");
        const auto receipts=ecm_results::read_legacy_stage2_curves(submission,state.publication_task,n);
        std::multimap<std::pair<uint64_t,uint64_t>,const ecm_results::LegacyStage2Curve*> by_point;
        for(const auto& receipt:receipts)
            by_point.emplace(std::make_pair(receipt.sigma,receipt.b1),&receipt);
        // An owned result contains the original B2 and reporting geometry.
        // When no such receipt exists, the old JSONL can disambiguate a
        // duplicate whose first submission owner was another task.
        std::map<fs::path,std::vector<ecm_results::json::Value>> logs;
        for(uint64_t i=0;i<extent;++i) {
            const auto& point=points[i];
            const auto* reservation=i<state.done?nullptr:&state.reserved[i-state.done];
            const bool confirmed=!reservation || reservation->completed;
            const auto [first,last]=by_point.equal_range({point.sigma,point.b1});
            const ecm_results::LegacyStage2Curve* match=nullptr;
            bool ambiguous=false,has_receipt=false;
            for(auto it=first;it!=last;++it) {
                const auto* entry=it->second;
                if(!entry->owned || entry->edwards!=point.edwards ||
                   (entry->order && entry->order!=point.index))
                    continue;
                if(match && match->report!=entry->report)ambiguous=true;
                match=entry;
            }
            if(!match || ambiguous) {
                const auto log_path=reservation && !reservation->results.empty()?fs::u8path(reservation->results):
                    state.pending_results.empty()?results:fs::u8path(state.pending_results);
                auto [log,fresh]=logs.try_emplace(log_path);
                if(fresh && fs::exists(log_path)) {
                    std::ifstream stream(log_path,std::ios::binary);
                    if(!stream)throw std::runtime_error("cannot read legacy Stage2 JSONL");
                    std::string line;
                    while(std::getline(stream,line) && !stream.eof()) {
                        try {log->second.push_back(ecm_results::json::parse(line));}
                        catch(const std::runtime_error&) {}
                    }
                    if(stream.bad())throw std::runtime_error("cannot read legacy Stage2 JSONL");
                }
                for(const auto& row:log->second) {
                    const auto& engine=row.get("stage2-engine");
                    if(reservation && !reservation->receipt.empty() &&
                       row.get("queue_receipt").text()!=reservation->receipt)
                        continue;
                    if(engine.get("record").text()!=std::to_string(point.index) ||
                       engine.get("record_hash").text()!=std::to_string(point.hash))
                        continue;
                    const auto sigma=point.edwards?row.get("Edwards").get("sigma").text():row.get("sigma").text();
                    if(row.has("Edwards")!=point.edwards || sigma!=std::to_string(point.sigma) ||
                       row.get("b1").text()!=std::to_string(point.b1))
                        continue;
                    if(reservation && !reservation->receipt.empty())
                        has_receipt=true;
                    match=nullptr;
                    ambiguous=false;
                    for(auto it=first;it!=last;++it) {
                        const auto* entry=it->second;
                        if(entry->edwards!=point.edwards)continue;
                        const auto public_row=ecm_results::json::parse(entry->report);
                        if(row.get("status").text()!=public_row.get("status").text() ||
                           row.get("factors").dump()!=public_row.get("factors").dump() ||
                           (public_row.get("stage").text()!="1" &&
                            row.get("b2").text()!=std::to_string(entry->b2)))
                            continue;
                        if(match && match->report!=entry->report)ambiguous=true;
                        match=entry;
                    }
                    if(match && !ambiguous)break;
                }
            }
            if(!match && !ambiguous) {
                // A verified factor can reach the ledger before the old
                // worker's JSONL append. Preserve it using its curve identity.
                for(auto it=first;it!=last;++it) {
                    const auto* entry=it->second;
                    if(entry->edwards!=point.edwards ||
                       ecm_results::json::parse(entry->report).get("status").text()!="F")
                        continue;
                    if(match && match->report!=entry->report) {
                        match=nullptr;
                        break;
                    }
                    match=entry;
                }
                if(match && !confirmed && match->b2 && match->b2!=state.b2) {
                    // The old policy did not uniquely identify this attempt.
                    match=nullptr;
                }
            }
            if(!match && !ambiguous && confirmed) {
                // An earlier position in this save may already own the same
                // mathematical receipt. The old worker deduplicated it.
                for(auto it=first;it!=last;++it) {
                    const auto* entry=it->second;
                    if(!entry->owned || entry->edwards!=point.edwards ||
                       !entry->order || entry->order>=point.index || entry->b2!=state.b2)
                        continue;
                    if(match && match->report!=entry->report)ambiguous=true;
                    match=entry;
                }
            }
            if(ambiguous || (!match && (confirmed || has_receipt)))
                throw std::runtime_error("cannot verify legacy Stage2 outcome for save record "+
                    std::to_string(point.index)+"; retain the old checkpoint, ledger and JSONL");
            if(!match)continue;
            const bool factor=ecm_results::json::parse(match->report).get("status").text()=="F";
            if(factor || (match->owned && !match->sealed))
                state.record_outcome(i,match->report,checkpoint_nf_key(point,std::to_string(match->b2)));
            if(reservation)state.reserved[i-state.done].completed=true;
        }
    }
    state.advance_completed();
    state.pending.clear();
    state.pending_results.clear();
    for(auto& entry:state.reserved) {
        entry.receipt.clear();
        entry.results.clear();
    }
    state.version=5;
}

} // namespace

void CurveQueue::commit(uint64_t position,const std::string &outcome) const {
    const auto hits=state.factors.size();
    state.complete(state.reserved.at(position-state.done),outcome);
    state.save(progress);
    if(state.factors.size()>hits)
        ecm_results::publish_stage2_checkpoint(fs::u8path(state.submission_path),{state.factors.back()});
}

std::string checkpoint_outcome(const Record &record,uint64_t b2,const std::string &report) {
    return "{\"key\":"+json_string(checkpoint_nf_key(record,std::to_string(b2)))+
           ",\"report\":"+ecm_results::json::parse(report).dump()+"}";
}

int run_task_queue(const TaskEngine &engine,Options worker_options,const TaskPaths &paths,const QueuePolicy &cfg,const TaskDefaults &defaults) {
    const auto &startup=worker_options;
    const auto &[cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir,explicit_results]=paths;
    const int device=startup.device;const uint64_t d=startup.d;
    const bool queue = startup.save.empty();
    ecm_queue::OwnerLease owner;
    if (queue) owner.acquire(worktodo.u8string(), startup.worker);
    if (queue && startup.selection) throw std::runtime_error("queue curve selection comes from the Stage2 save; use --save for overrides");
    Handle lock;
    if (queue) {
        if (!fs::exists(worktodo)) throw std::runtime_error("worktodo not found: " + worktodo.string());
        if (!startup.dry && !startup.plan_only) {
            // Each worker section has one Stage2 consumer. The shared queue edit
            // lock serializes row rewrites across different workers.
            fs::path lockpath(worktodo.string() + ".worker" + std::to_string(startup.worker) + ".stage2.lock");
            lock.value = CreateFileW(lockpath.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
            if (lock.value == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot lock worktodo worker (another process may own it)");
        }
    }
    auto same_path=[](const fs::path &a,const fs::path &b) {
        if(a.empty()||b.empty())return false;
        std::error_code error;
        if(fs::equivalent(a,b,error)&&!error)return true;
        return upper(a.lexically_normal().string())==upper(b.lexically_normal().string());
    };
    // A common TOML may share device/save defaults, never writable queue state.
    std::vector<fs::path> writable={worktodo,finished,results,log,legacy_progress,fs::u8path(startup.debug_file),fs::u8path(startup.submission)};
    if(startup.parallel!=ParallelMode::serial && startup.debug_log)for(const char *suffix:{".slot1",".slot2"})
        writable.emplace_back(fs::u8path(startup.debug_file+suffix));
    for(size_t i=0;i<writable.size();++i) {
        if(!startup.tune_profile.empty() && same_path(writable[i],fs::path(startup.tune_profile)))throw std::runtime_error("output path conflicts with measured tune profile");
        if(!startup.stage1_profile.empty() && same_path(writable[i],fs::path(startup.stage1_profile)))throw std::runtime_error("output path conflicts with Stage1 tune profile");
        if(!startup.stage1_csv.empty() && same_path(writable[i],fs::path(startup.stage1_csv)))throw std::runtime_error("output path conflicts with Stage1 cost CSV");
        if(same_path(writable[i],ini)||same_path(writable[i],executable()))throw std::runtime_error("output path conflicts with configuration or executable");
        for(size_t j=0;j<i;++j)if(same_path(writable[i],writable[j]))throw std::runtime_error("queue, finished, progress, result and log paths must be distinct");
    }

    if(queue && !cfg.shared_finished && same_path(finished,absolute_from(executable().parent_path(),cfg.finished)))
        throw std::runtime_error("Stage1 and Stage2 finished records must use distinct paths");
    auto report=[&](const std::string &message,bool error=false) {
        (error?std::cerr:std::cout)<<message<<std::endl;
        if(!startup.dry&&!startup.plan_only&&!log.empty())
            append_log(log,message,startup.worker,startup.log_max_kb);
    };
    report(std::string(ecm_program::name)+" "+ecm_program::version+" ("+engine.name+")");
    if(!SetConsoleCtrlHandler(console_control,TRUE))throw std::runtime_error("cannot install console stop handler");
    struct ControlGuard { ~ControlGuard(){SetConsoleCtrlHandler(console_control,FALSE);} } control_guard;
    if(engine.begin)engine.begin();
    uint64_t completed=0,failed_tasks=0;
    for (;;) {
        poll_inherited_stop();
        fatal_log.clear();
        if(stop_requests)break;
        // Freeze this task's inherited options before applying its worktodo row.
        // Only runtime tuning feedback is carried from the preceding task.
        Options o=startup;
        fs::path save;
        std::string task_line,expected_n;
        QueueSelection fields;
        std::vector<Record> plan;
        uint64_t skip=o.skip,count=o.curves,b2=o.b2?o.b2:defaults.b2;
        std::string_view b2_source=o.b2?"cli":defaults.b2?"config":o.auto_b2?"auto":"unset";
        if(queue) {
            std::string read_error;
            const auto read = ecm_worktodo_read_next_line(worktodo.string(), o.worker,
                {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage2,
                task_line, read_error);
            if(read == WorktodoReadResult::Error)throw std::runtime_error(read_error);
            if(read == WorktodoReadResult::Empty)break;
        }
        try {
            if(queue) {
                fields=queue_fields(task_line);
                skip=fields.skip;count=fields.count;
                if(!o.b2&&fields.b2) {
                    b2=fields.b2;
                    b2_source="worktodo";
                }
                if(!fields.task.modulus_from_save)expected_n=queue_modulus(fields.task,nullptr);
                save=absolute_from(tmp,fields.task.save_name);
                o.submission_exponent=fields.task.k=="1" && fields.task.b=="2" && fields.task.c=="-1"?fields.task.n:0;
                o.submission_known=fields.task.factors;o.submission_aid=fields.task.aid;
                o.inferred_carrier=o.auto_carrier && fields.task.k=="1" && fields.task.b=="2" && fields.task.c=="-1" &&
                    fields.task.n>=2 && fields.task.n<=ecm_stage2::max_input_bits?unsigned(fields.task.n):0;
            } else save=absolute_from(cwd,o.save);
            for(const auto &path:writable)if(same_path(save,path))throw std::runtime_error("save path conflicts with writable queue/log/result state");
            if(!o.dry&&!o.plan_only){fatal_log=log;fatal_log_worker=o.worker;fatal_log_max_kb=o.log_max_kb;}
            if(o.auto_b2&&!b2&&o.cost_profile.empty()&&o.tune_profile.empty())throw std::runtime_error("Auto B2 requires --tune-profile or --cost-profile FILE");
            if(o.carrier_exponent && o.auto_b2 && !b2 && o.tune_profile.empty())throw std::runtime_error("Mersenne carrier Auto B2 requires full ECM tune data");
            plan=records(save,skip,count);
            if(queue && fields.task.modulus_from_save)
                expected_n=queue_modulus(fields.task,plan.empty()?nullptr:&plan.front());
            if(cfg.b2_ratio && !plan.empty()) {
                if(plan.front().b1 > (static_cast<uint64_t>(INT64_MAX)-8192)/cfg.b2_ratio)
                    throw std::runtime_error("b2_ratio multiplication exceeds Stage2 B2 range");
                b2=plan.front().b1*cfg.b2_ratio;
                b2_source="b2_ratio";
            }
            for(const auto &r:plan) {
                if(engine.validate)engine.validate(r,b2);
                if(r.edwards && o.auto_b2 && !b2)throw std::runtime_error("Edwards Stage2 currently requires an explicit B2");
                if((!(o.auto_b2&&!b2)&&b2<=r.b1)||b2>static_cast<uint64_t>(INT64_MAX)-8192) {
                    auto message=std::string("B2 must exceed every saved B1 and fit the engine's signed index range")+
                        " (source="+std::string(b2_source)+')';
                    if(b2_source=="worktodo")throw TaskInputError(message);
                    if(b2_source=="unset" && stage2_backend::opencl)
                        message+="; OpenCL Stage2 requires a nonzero B2 from --b2, worktodo, [stage2].b2 or b2_ratio";
                    throw std::runtime_error(message);
                }
                if(!expected_n.empty()&&r.n!=expected_n)
                    throw TaskInputError(fields.task.modulus_from_save ?
                        "ECMSTAGE2N selected save records must have the same N" : "worktodo N differs from save N");
                if(queue&&r.b1!=plan.front().b1)throw TaskInputError("queue saves must have the same B1");
                if(r.x=="0")throw TaskInputError("saved X=0 gives no usable Stage1 point");
                if(o.carrier_exponent) {
                    ecm_stage2::ModulusContext modulus;std::string error;
                    // An invalid command option must not mark a valid queue row as erroneous.
                    if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
                }
            }
        } catch(const TaskInputError &e) {
            if(!queue || o.dry || o.plan_only)throw;
            // Record the reason durably before commenting out the original row.
            report("ERROR: task input: "+std::string(e.what())+" | task="+task_line,true);
            append_finished(finished,"# ERROR reason="+std::string(e.what()),"# "+task_line);
            if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::MarkError))
                throw std::runtime_error("cannot mark invalid task; queue changed or rewrite failed");
            ++failed_tasks;
            if(o.once)break;
            continue;
        }
        if(count && plan.size()<count)
            report("WARNING: requested="+std::to_string(count)+" available="+std::to_string(plan.size())+
                   " after skip="+std::to_string(skip)+"; running available curves only.",true);
        if(plan.empty())report("WARNING: no available curves after skip; task will finish with zero curves.",true);
        stage2_queue::State state;
        const bool persist=queue&&!o.dry&&!o.plan_only;
        fs::path progress;
        Handle progress_lock;
        if(persist) {
            stage2_queue::Task task;
            task.aid=fields.task.aid;task.n=expected_n;
            task.save_sha256=ecm_stage2::sha256_file(save);task.skip=skip;task.total=plan.size();
            task.b1=plan.empty()?0:plan.front().b1;
            task.b2=b2;task.auto_b2=o.auto_b2&&!b2;
            Big target;mpz_set_str(target.z,expected_n.c_str(),16);
            const auto bits=mpz_sizeinbase(target.z,2);
            // Serialize filename allocation, then keep only the selected task lease.
            fs::create_directories(progress_dir);
            Handle name_lock;
            auto name_lockpath=task.path(progress_dir,bits);name_lockpath+=".name.lock";
            name_lock.value=CreateFileW(name_lockpath.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
            if(name_lock.value==INVALID_HANDLE_VALUE)throw std::runtime_error("another process is allocating this Stage2 checkpoint name");
            const auto legacy_bounds=[&] {return ecm_results::legacy_stage2_bounds(fs::u8path(o.submission));};
            auto source=stage2_queue::load_task(progress_dir,bits,task,state,legacy_bounds,&progress);
            for(const auto &path:writable)if(same_path(progress,path))throw std::runtime_error("task checkpoint conflicts with output state");
            if(same_path(progress,save)||same_path(progress,ini)||same_path(progress,executable())||
               (!o.tune_profile.empty()&&same_path(progress,fs::path(o.tune_profile))))
                throw std::runtime_error("task checkpoint conflicts with input");
            auto lockpath=progress;lockpath+=".lock";
            progress_lock.value=CreateFileW(lockpath.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
            if(progress_lock.value==INVALID_HANDLE_VALUE)
                throw std::runtime_error("another process owns this Stage2 task checkpoint");
            const auto match_legacy=[&](const std::string &line) {
                const auto old=queue_fields(line);Big n;std::string error;
                return old.task.aid==task.aid && ecm_compute_stage2_n(old.task,n.z,error) && number(n.z)==task.n;
            };
            if(source.empty() && stage2_queue::import_legacy(legacy_progress,task,match_legacy,state,legacy_bounds))
                source=legacy_progress;
            else if(source==progress && fs::exists(legacy_progress)) {
                stage2_queue::State previous;
                if(stage2_queue::import_legacy(legacy_progress,task,match_legacy,previous,legacy_bounds) &&
                   previous.run==state.run && previous.publication_task==state.publication_task)source=legacy_progress;
            }
            Handle legacy_lock;
            if(!source.empty() && source!=progress) {
                auto path=source;path+=".lock";
                legacy_lock.value=CreateFileW(path.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
                if(legacy_lock.value==INVALID_HANDLE_VALUE)
                    throw std::runtime_error("another process owns this legacy Stage2 checkpoint");
                if(state.version<5) {
                    const auto previous_run=state.run;
                    if(!stage2_queue::import_legacy(source,task,match_legacy,state,legacy_bounds) ||
                       previous_run!=state.run)
                        throw std::runtime_error("legacy Stage2 checkpoint changed during discovery");
                }
            }
            const auto previous_total=state.total,previous_b2=state.b2;
            const auto previous_auto=state.auto_b2;
            state.submission_path=ecm_io::file_identity(fs::u8path(o.submission));
            if(state.version<5)
                migrate_queue_outcomes(state,save,skip,results,fs::u8path(o.submission),target.z);
            while(!state.reserved.empty() && !state.reserved.back().completed)
                state.reserved.pop_back();
            // Commit migration before applying policy changes or retiring old receipts.
            if(!source.empty() && source!=progress) {
                state.save(progress);
                // Materialize old sealed batches before reporting that their
                // source can be removed. Pending NF now belongs to the ckpt.
                ecm_results::refresh_submission_if_owned(fs::u8path(o.submission));
                if(!state.nf_keys.empty())
                    ecm_results::retire_legacy_stage2_curves(fs::u8path(o.submission),state.publication_task,
                        [&](const std::string& key){return state.nf_keys.count(ecm_stage2::sha256_text(key))!=0;});
                auto archive=source;archive+=".migrated";
                if(!MoveFileExW(source.c_str(),archive.c_str(),MOVEFILE_WRITE_THROUGH))
                    throw std::runtime_error("cannot archive imported Stage2 checkpoint: "+source.u8string());
                report("stage2_resume_import: completed="+std::to_string(state.done)+" total="+std::to_string(state.total)+
                    " legacy_archived=1 checkpoint="+progress.u8string());
            }
            state.reconfigure(task);
            if(state.done>plan.size() && records(save,skip,state.done).size()!=state.done)
                throw std::runtime_error("Stage2 completed progress exceeds the available save records");
            state.save(progress);
            ecm_results::publish_stage2_checkpoint(fs::u8path(state.submission_path),state.factors);
            if(!source.empty() && (previous_total!=task.total || previous_b2!=task.b2 || previous_auto!=task.auto_b2))
                report("stage2_resume_override: previous_total="+std::to_string(previous_total)+" requested="+std::to_string(task.total)+
                    " completed="+std::to_string(state.done)+" effective_total="+std::to_string(state.total)+
                    " previous_B2="+std::to_string(previous_b2)+" B2="+std::to_string(task.b2)+
                    "; completed curves retain their actual bounds.",true);
            if(state.done>plan.size())report("WARNING: requested="+std::to_string(plan.size())+
                " is below already completed="+std::to_string(state.done)+
                "; no further curves will run; sealing all completed curves (factor hits reported separately).",true);
        }
        if(engine.prepare && state.done<plan.size())engine.prepare(o,plan[state.done]);
        const auto curve_results = persist && !explicit_results ? fs::path{} : results;
        if(stage2_log::enabled(stage2_log::curve)||o.gui_events||o.dry) {
            if(o.gui_events && !task_line.empty())std::cout<<"START: "<<task_line<<std::endl;
            std::ostringstream message;
            message << "stage2_plan: save=" << save.string() << " requested=" << (count?std::to_string(count):"all")
                    << " available=" << plan.size() << " completed=" << state.done << " remaining=" << (state.done<plan.size()?plan.size()-state.done:0)
                    << " B2=" << b2 << " D=" << d << " device=" << device << " worker=" << o.worker
                    << " B2_source=" << b2_source << " log=" << log.string() << " results=" << curve_results.string();
            if(persist)message<<" checkpoint="<<progress.string();
            report(message.str());
        }
        o.publication_task=persist?state.publication_task:std::to_string(ecm_io::fingerprint(ecm_stage2::sha256_file(save)+"|"+std::to_string(skip)+"|"+std::to_string(plan.size())+"|"+std::to_string(b2)));
        bool task_completed=true;
        if(!o.dry && !o.plan_only) {
            const QueueReport queue_report=report;
            const CurveQueue curve_queue{plan,save,curve_results,log,progress,defaults,state,completed,queue_report};
            task_completed=engine.run(curve_queue,o,b2,d);
        } else for(size_t i=static_cast<size_t>(state.done);i<plan.size();++i) {
            poll_inherited_stop();
            if(stop_requests){task_completed=false;break;}
            const auto &r=plan[i];
            o.runtime_remaining=plan.size()-i;
            if(stage2_log::enabled(stage2_log::curve)||o.dry)
                report("curve_start: "+std::to_string(i+1)+"/"+std::to_string(plan.size())+" record="+std::to_string(r.index)+
                       " sigma="+std::to_string(r.sigma)+" B1="+std::to_string(r.b1)+" checksum="+(r.checksum?"verified":"absent")+
                       " curve_model="+(r.edwards?std::string("ed"):std::string("mont"))+(r.edwards?"":" param=0"));
            if(o.dry)continue;
            if(o.plan_only) {
                engine.plan(o,r,b2,d);
            }
        }
        if(task_completed && !o.dry && !o.plan_only && !o.tune_child) {
            if(persist) {
                if(!state.nf_keys.empty())
                    ecm_results::publish_stage2_checkpoint(fs::u8path(state.submission_path),{},
                        state.latest_nf,state.nf_keys.size(),&state.final_nf,[&]{state.save(progress);});
            } else {
                ecm_results::check_stage2_outputs(results,fs::u8path(o.submission),persist && state.done>0);
                ecm_results::seal_submissions(fs::u8path(o.submission),2,o.publication_task,true);
            }
        }
        if(!queue||o.dry||o.plan_only)break;
        if(state.done<state.total)break;
        std::string current;
        std::string read_error;
        const auto read = ecm_worktodo_read_next_line(worktodo.string(), o.worker,
            {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage2,
            current, read_error);
        if(read == WorktodoReadResult::Error)throw std::runtime_error(read_error);
        if(read != WorktodoReadResult::Task||current!=task_line)
            throw std::runtime_error("task completed but worktodo changed; queue and progress retained");
        append_finished(finished,"# stage2_task_id="+state.run,task_line,state.run);
        if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::Remove))
            throw std::runtime_error("task completed but queue could not be advanced; progress retained");
        // A restart after queue advancement can ignore a complete stale state.
        if(!DeleteFileW(progress.c_str()) && GetLastError()!=ERROR_FILE_NOT_FOUND)
            throw std::runtime_error("task advanced but cannot remove completed progress file");
        report("task_done: requested="+(count?std::to_string(count):"all")+" available="+std::to_string(plan.size())+
               " completed="+std::to_string(state.done));
        worker_options.runtime_prepare_hints=std::move(o.runtime_prepare_hints);
        worker_options.runtime_route_hints=std::move(o.runtime_route_hints);
        if(o.once||stop_requests)break;
    }
    if(stage2_log::enabled(stage2_log::curve)||startup.dry)
        report(std::string(startup.plan_only?"plan_complete":startup.dry?"dry_run_complete":stop_requests?"stage2_stopped":"stage2_complete")+
               ": curves_this_run="+std::to_string(completed)+" invalid_tasks="+std::to_string(failed_tasks));
    return failed_tasks?1:0;
}
}
