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
#include <sstream>
#include <string_view>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
namespace s2prod {
namespace {
fs::path fatal_log;
int fatal_log_worker=1;
uint64_t fatal_log_max_kb=0;

class QueueFileLock {
#ifdef _WIN32
    Handle handle_;
#else
    int fd_ = -1;
#endif
public:
    QueueFileLock() = default;
    QueueFileLock(const QueueFileLock&) = delete;
    QueueFileLock& operator=(const QueueFileLock&) = delete;
    ~QueueFileLock() {
#ifndef _WIN32
        if(fd_ >= 0) close(fd_);
#endif
    }
    bool open(const fs::path &path, bool temporary = true) {
#ifdef _WIN32
        const DWORD flags = FILE_ATTRIBUTE_NORMAL | (temporary ? FILE_FLAG_DELETE_ON_CLOSE : 0);
        handle_.value = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, flags, nullptr);
        return handle_.value != INVALID_HANDLE_VALUE;
#else
        // Retain the lock inode so competing opens always lock the same file.
        (void)temporary;
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        return fd_ >= 0 && flock(fd_, LOCK_EX | LOCK_NB) == 0;
#endif
    }
};

class QueueStopHandler {
#ifndef _WIN32
    struct sigaction old_int_{}, old_term_{};
#endif
public:
    QueueStopHandler() {
#ifdef _WIN32
        if(!SetConsoleCtrlHandler(console_control, TRUE))
            throw std::runtime_error("cannot install console stop handler");
#else
        struct sigaction action{};
        action.sa_handler = console_control;
        sigemptyset(&action.sa_mask);
        sigaddset(&action.sa_mask, SIGINT);
        sigaddset(&action.sa_mask, SIGTERM);
        if(sigaction(SIGINT, &action, &old_int_) != 0)
            throw std::runtime_error("cannot install Stage2 SIGINT handler");
        if(sigaction(SIGTERM, &action, &old_term_) != 0) {
            sigaction(SIGINT, &old_int_, nullptr);
            throw std::runtime_error("cannot install Stage2 SIGTERM handler");
        }
#endif
    }
    ~QueueStopHandler() {
#ifdef _WIN32
        SetConsoleCtrlHandler(console_control, FALSE);
#else
        sigaction(SIGTERM, &old_term_, nullptr);
        sigaction(SIGINT, &old_int_, nullptr);
#endif
    }
};

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

void publish_outcomes(stage2_queue::State& state, const fs::path& progress, bool final = false) {
    const auto save = [&] { state.save(progress); };
    const auto path = fs::u8path(state.submission_path);
    while (state.published_factors < state.factors.size()) {
        const auto& factor = state.factors[state.published_factors];
        const auto key = ecm_results::factor_identity(factor);
        const bool duplicate = std::any_of(state.factors.begin(),
            state.factors.begin() + state.published_factors, [&](const auto& previous) {
                return ecm_results::factor_identity(previous) == key;
            });
        const auto acknowledge = [&] {
            ++state.published_factors;
            save();
        };
        if (duplicate && !state.append.pending)
            acknowledge();
        else
            ecm_results::append_checkpointed(path, factor + '\n', state.append, save,
                                             acknowledge, state.adopt_v5);
    }
    if (!final || state.nf_keys.empty() || state.nf_published)
        return;
    if (state.final_nf.empty()) {
        state.final_nf = ecm_results::aggregate_nf(state.latest_nf, state.nf_keys.size());
        save();
    }
    ecm_results::append_checkpointed(path, state.final_nf + '\n', state.append, save, [&] {
        state.nf_published = true;
        save();
    }, state.adopt_v5);
}

} // namespace

void CurveQueue::commit(uint64_t position,const std::string &outcome) const {
    state.complete(state.reserved.at(position-state.done),outcome);
    state.save(progress);
    publish_outcomes(state,progress);
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
    QueueFileLock lock;
    if (queue) {
        if (!fs::exists(worktodo)) throw std::runtime_error("worktodo not found: " + worktodo.string());
        if (!startup.dry && !startup.plan_only) {
            // Each worker section has one Stage2 consumer. The shared queue edit
            // lock serializes row rewrites across different workers.
            fs::path lockpath(worktodo.string() + ".worker" + std::to_string(startup.worker) + ".stage2.lock");
            if (!lock.open(lockpath, false)) throw std::runtime_error("cannot lock worktodo worker (another process may own it)");
        }
    }
    auto same_path=[](const fs::path &a,const fs::path &b) {
        if(a.empty()||b.empty())return false;
        std::error_code error;
        if(fs::equivalent(a,b,error)&&!error)return true;
#ifdef _WIN32
        return upper(a.lexically_normal().string())==upper(b.lexically_normal().string());
#else
        return a.lexically_normal()==b.lexically_normal();
#endif
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
    QueueStopHandler control_guard;
    bool began=false;
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
        std::size_t selected_records=0;
        uint64_t saved_b1=0;
        bool edwards_save=false;
        std::string_view b2_source=o.b2?"cli":defaults.b2?"config":o.auto_b2?"auto":"unset";
        if(queue) {
            std::string read_error;
            const auto read = ecm_worktodo_read_next_line(worktodo.string(), o.worker,
                {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage2,
                task_line, read_error);
            if(read == WorktodoReadResult::Error)throw std::runtime_error(read_error);
            if(startup.dispatch_row &&
               (read != WorktodoReadResult::Task || fingerprint(task_line)!=*startup.dispatch_row))
                return DispatchChanged;
            if(read == WorktodoReadResult::Empty)break;
        }
        try {
            if(queue) {
                fields=queue_fields(task_line,engine.input_max_bits);
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
            plan=records(save,skip,count,engine.input_max_bits,engine.edwards_preparation);
            if(!queue && plan.empty())
                throw TaskInputError("skip exceeds saved curve count or save contains no curves");
            selected_records=plan.size();
            if(!plan.empty()) {
                saved_b1=plan.front().b1;
                edwards_save=plan.front().edwards;
            }
            if(queue && fields.task.modulus_from_save)
                expected_n=queue_modulus(fields.task,plan.empty()?nullptr:&plan.front());
            if(cfg.b2_ratio && !plan.empty()) {
                if(plan.front().b1 > (static_cast<uint64_t>(INT64_MAX)-8192)/cfg.b2_ratio)
                    throw std::runtime_error("b2_ratio multiplication exceeds Stage2 B2 range");
                b2=plan.front().b1*cfg.b2_ratio;
                b2_source="b2_ratio";
            }
            for(const auto &r:plan) {
                if(!expected_n.empty()&&r.n!=expected_n)
                    throw TaskInputError(fields.task.modulus_from_save ?
                        "ECMSTAGE2N selected save records must have the same N" : "worktodo N differs from save N");
                if(queue&&r.b1!=plan.front().b1)throw TaskInputError("queue saves must have the same B1");
                if(r.skip_stage2)continue;
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
                if(r.x=="0")throw TaskInputError("saved X=0 gives no usable Stage1 point");
                if(o.carrier_exponent) {
                    ecm_stage2::ModulusContext modulus;std::string error;
                    // An invalid command option must not mark a valid queue row as erroneous.
                    if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
                }
            }
            plan.erase(std::remove_if(plan.begin(), plan.end(),
                [](const Record& record) { return record.skip_stage2; }), plan.end());
        } catch(const TaskInputError &e) {
            if(!queue || o.dry || o.plan_only)throw;
            // Record the reason durably before commenting out the original row.
            report("ERROR: task input: "+std::string(e.what())+" | task="+task_line,true);
            append_finished(finished,"# ERROR reason="+std::string(e.what()),"# "+task_line);
            if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::MarkError))
                throw std::runtime_error("cannot mark invalid task; queue changed or rewrite failed");
            ++failed_tasks;
            if(startup.dispatch_row)return stop_requests?1:DispatchInvalid;
            if(o.once)break;
            continue;
        }
        if(count && selected_records<count)
            report("WARNING: requested="+std::to_string(count)+" available="+std::to_string(selected_records)+
                   " after skip="+std::to_string(skip)+"; running available curves only.",true);
        if(plan.empty())report("WARNING: no available curves after skip; task will finish with zero curves.",true);
        stage2_queue::State state;
        const bool persist=!o.dry&&!o.plan_only;
        fs::path progress;
        QueueFileLock progress_lock;
        if(persist && queue) {
            stage2_queue::Task task;
            task.aid=fields.task.aid;task.n=expected_n;
            task.save_sha256=ecm_stage2::sha256_file(save);task.skip=skip;task.total=selected_records;
            task.b1=saved_b1;
            task.b2=b2;task.auto_b2=o.auto_b2&&!b2;
            Big target;mpz_set_str(target.z,expected_n.c_str(),16);
            const auto bits=mpz_sizeinbase(target.z,2);
            // Serialize filename allocation, then keep only the selected task lease.
            fs::create_directories(progress_dir);
            QueueFileLock name_lock;
            auto name_lockpath=task.path(progress_dir,bits);name_lockpath+=".name.lock";
            if(!name_lock.open(name_lockpath))throw std::runtime_error("another process is allocating this Stage2 checkpoint name");
            const auto match_legacy=[&](const std::string &line) {
                const auto old=queue_fields(line,engine.input_max_bits);Big n;std::string error;
                return old.task.aid==task.aid && ecm_compute_stage2_n(old.task,n.z,error) && number(n.z)==task.n;
            };
            auto source=stage2_queue::load_task(progress_dir,bits,task,state,{},&progress);
            if(source.empty() && stage2_queue::import_legacy(legacy_progress,task,match_legacy,state))
                source=legacy_progress;
            const bool skips_records=selected_records!=plan.size();
            if(!source.empty() && skips_records) {
                // Earlier workers counted saved factors as completed slots.
                // Finish their existing checkpoint with the original numbering.
                plan=records(save,skip,count,engine.input_max_bits,engine.edwards_preparation);
            } else if(source.empty() && edwards_save) {
                auto point_task=task;
                point_task.save_sha256=ecm_stage2::sha256_text("stage2-points-only|"+task.save_sha256);
                point_task.total=plan.size();
                fs::path point_progress;
                source=stage2_queue::load_task(progress_dir,bits,point_task,state,{},&point_progress);
                if(skips_records || !source.empty()) {
                    task=std::move(point_task);
                    progress=std::move(point_progress);
                }
            }
            for(const auto &path:writable)if(same_path(progress,path))throw std::runtime_error("task checkpoint conflicts with output state");
            if(same_path(progress,save)||same_path(progress,ini)||same_path(progress,executable())||
               (!o.tune_profile.empty()&&same_path(progress,fs::path(o.tune_profile))))
                throw std::runtime_error("task checkpoint conflicts with input");
            auto lockpath=progress;lockpath+=".lock";
            if(!progress_lock.open(lockpath))
                throw std::runtime_error("another process owns this Stage2 task checkpoint");
            if(source==progress && fs::exists(legacy_progress)) {
                stage2_queue::State previous;
                if(stage2_queue::import_legacy(legacy_progress,task,match_legacy,previous) &&
                   previous.run==state.run && previous.publication_task==state.publication_task)source=legacy_progress;
            }
            QueueFileLock legacy_lock;
            if(!source.empty() && source!=progress) {
                auto path=source;path+=".lock";
                if(!legacy_lock.open(path))
                    throw std::runtime_error("another process owns this legacy Stage2 checkpoint");
                if(state.version<5) {
                    const auto previous_run=state.run;
                    if(!stage2_queue::import_legacy(source,task,match_legacy,state) ||
                       previous_run!=state.run)
                        throw std::runtime_error("legacy Stage2 checkpoint changed during discovery");
                }
            }
            const auto previous_total=state.total,previous_b2=state.b2;
            const auto previous_auto=state.auto_b2;
            if(state.version<5 && (state.done || !state.pending.empty() || !state.reserved.empty()))
                throw std::runtime_error("Stage2 checkpoint predates self-contained outcomes; finish it with the version that created it");
            if(state.version<6) {
                state.adopt_v5=state.version==5;
                state.version=6;
            }
            while(!state.reserved.empty() && !state.reserved.back().completed)
                state.reserved.pop_back();
            // Commit migration before applying policy changes or retiring old receipts.
            if(!source.empty() && source!=progress) {
                state.save(progress);
                auto archive=source;archive+=".migrated";
#ifdef _WIN32
                if(!MoveFileExW(source.c_str(),archive.c_str(),MOVEFILE_WRITE_THROUGH))
#else
                if(fs::exists(archive) || !ecm_replace_file(source,archive))
#endif
                    throw std::runtime_error("cannot archive imported Stage2 checkpoint: "+source.u8string());
                report("stage2_resume_import: completed="+std::to_string(state.done)+" total="+std::to_string(state.total)+
                    " legacy_archived=1 checkpoint="+progress.u8string());
            }
            state.reconfigure(task);
            if(state.done>plan.size() &&
               records(save,skip,state.done,engine.input_max_bits,EdwardsPreparation::ValidateOnly).size()!=state.done)
                throw std::runtime_error("Stage2 completed progress exceeds the available save records");
            if(!source.empty() && (previous_total!=task.total || previous_b2!=task.b2 || previous_auto!=task.auto_b2))
                report("stage2_resume_override: previous_total="+std::to_string(previous_total)+" requested="+std::to_string(task.total)+
                    " completed="+std::to_string(state.done)+" effective_total="+std::to_string(state.total)+
                    " previous_B2="+std::to_string(previous_b2)+" B2="+std::to_string(task.b2)+
                    "; completed curves retain their actual bounds.",true);
            if(state.done>plan.size())report("WARNING: requested="+std::to_string(plan.size())+
                " is below already completed="+std::to_string(state.done)+
                "; no further curves will run; sealing all completed curves (factor hits reported separately).",true);
        } else if(persist) {
            stage2_queue::Task task;
            task.aid=o.submission_aid;
            task.skip=skip;
            task.total=plan.size();
            task.b2=b2;
            task.auto_b2=o.auto_b2&&!b2;
            auto identity=ecm_stage2::sha256_file(save)+"|"+std::to_string(selected_records)+"|"+
                std::to_string(b2)+"|"+std::to_string(task.auto_b2)+"|"+
                ecm_io::file_identity(fs::u8path(o.submission))+"|"+std::to_string(o.submission_exponent);
            for(const auto &known:o.submission_known)identity+="|"+known;
            task.save_sha256=ecm_stage2::sha256_text(identity);
            progress=progress_dir/(".ecm_ckpt_stage2_direct_"+task.id()+".dat");
            for(const auto &path:writable)if(same_path(progress,path))
                throw std::runtime_error("direct Stage2 checkpoint conflicts with output state");
            if(same_path(progress,save)||same_path(progress,ini)||same_path(progress,executable())||
               (!o.tune_profile.empty()&&same_path(progress,fs::path(o.tune_profile))))
                throw std::runtime_error("direct Stage2 checkpoint conflicts with input");
            fs::create_directories(progress_dir);
            auto lockpath=progress;lockpath+=".lock";
            if(!progress_lock.open(lockpath))
                throw std::runtime_error("another process owns this direct Stage2 checkpoint");
            state.load(progress);
            state.reconfigure(task);
        }
        if(persist) {
            const auto submission=ecm_io::file_identity(fs::u8path(o.submission));
            if(!state.submission_path.empty() && state.submission_path!=submission)
                throw std::runtime_error("Stage2 checkpoint belongs to a different submission path");
            state.submission_path=submission;
            state.save(progress);
            publish_outcomes(state,progress,!state.final_nf.empty());
            if(state.adopt_v5) {
                state.adopt_v5=false;
                state.save(progress);
            }
        }
        if(!began) {
            if(engine.begin)engine.begin();
            began=true;
        }
        if(engine.prepare && state.done<plan.size())engine.prepare(o,plan[state.done]);
        const auto curve_results = queue && persist && !explicit_results ? fs::path{} : results;
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
        if(!queue && o.dry && task_completed)
            report("validated "+std::to_string(plan.size())+" curves; B2="+std::to_string(b2));
        if(task_completed && persist && state.done==state.total)
            publish_outcomes(state,progress,true);
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
        std::error_code removal_error;
        fs::remove(progress, removal_error);
        if(removal_error && removal_error!=std::errc::no_such_file_or_directory)
            throw std::runtime_error("task advanced but cannot remove completed progress file");
        report("task_done: requested="+(count?std::to_string(count):"all")+" available="+std::to_string(plan.size())+
               " completed="+std::to_string(state.done));
        worker_options.runtime_prepare_hints=std::move(o.runtime_prepare_hints);
        worker_options.runtime_route_hints=std::move(o.runtime_route_hints);
        if(startup.dispatch_row)return stop_requests?0:DispatchAdvanced;
        if(o.once||stop_requests)break;
    }
    if(stage2_log::enabled(stage2_log::curve)||startup.dry)
        report(std::string(startup.plan_only?"plan_complete":startup.dry?"dry_run_complete":stop_requests?"stage2_stopped":"stage2_complete")+
               ": curves_this_run="+std::to_string(completed)+" invalid_tasks="+std::to_string(failed_tasks));
    return failed_tasks?1:!queue&&stop_requests?130:0;
}
}
