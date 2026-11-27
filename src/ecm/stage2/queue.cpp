#include "../../io/files.h"
#include "../../io/stage2_handoff.h"
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
#include <cerrno>
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

// A stable auxiliary file carries one OS byte-range lock per absolute record.
// Keep it open for the work window so an interrupted process releases its claims.
class CurveClaims {
#ifdef _WIN32
    Handle file_;
#else
    int file_ = -1;
#endif
public:
    explicit CurveClaims(fs::path path) {
        path += ".claims";
#ifdef _WIN32
        file_.value=CreateFileW(path.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,
                                nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file_.value==INVALID_HANDLE_VALUE)
#else
        file_=::open(path.c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0600);
        if(file_<0)
#endif
            throw std::runtime_error("cannot open Stage2 curve claims: "+path.u8string());
    }
    CurveClaims(const CurveClaims&) = delete;
    CurveClaims& operator=(const CurveClaims&) = delete;
    ~CurveClaims() {
#ifndef _WIN32
        close(file_);
#endif
    }
    bool claim(uint64_t record) {
        if(!record || record>INT64_MAX)throw std::runtime_error("Stage2 curve claim index is out of range");
#ifdef _WIN32
        OVERLAPPED range{};
        range.Offset=static_cast<DWORD>(record);
        range.OffsetHigh=static_cast<DWORD>(record>>32);
        if(LockFileEx(file_.value,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,1,0,&range))return true;
        if(GetLastError()==ERROR_LOCK_VIOLATION)return false;
#else
        struct flock range{};
        range.l_type=F_WRLCK;
        range.l_whence=SEEK_SET;
        range.l_start=static_cast<off_t>(record);
        range.l_len=1;
        if(fcntl(file_,F_SETLK,&range)==0)return true;
        if(errno==EACCES || errno==EAGAIN)return false;
#endif
        throw std::runtime_error("cannot claim Stage2 curve "+std::to_string(record));
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

void publish_shared(ecm_handoff::Ledger& ledger,const fs::path& progress) {
    const auto save=[&] { ledger.write(progress); };
    for(auto& entry:ledger.publications) {
        auto& item=entry.second;
        if(item.published)continue;
        ecm_results::append_checkpointed(fs::u8path(item.path),item.report,item.append,save,[&] {
            item.published=true;
            item.report.clear();
            save();
        });
    }
}

void store_shared_curve(const fs::path& progress,const std::string& submission,const Record& record,
                        const std::string& outcome) {
    const auto body=ecm_results::json::parse(outcome).get("report");
    const auto status=body.get("status").text();
    if((status!="NF" && status!="F") || body.get("curves").text()!="1")
        throw std::runtime_error("invalid Stage2 curve result");
    const auto report=body.dump();
    const ecm_io::FileLock lock(progress.u8string());
    auto ledger=ecm_handoff::Ledger::read(progress);
    auto& rows=ledger.curves[record.n];
    const auto lower=body.has("b2-min")?u64(body.get("b2-min").text(),"B2min"):record.b1;
    const auto upper=u64(body.get("b2").text(),"B2");
    if(!rows.emplace(record.index,ecm_handoff::CurveResult{record.hash,lower,upper,report}).second)
        throw std::runtime_error("duplicate committed Stage2 curve "+std::to_string(record.index));
    if(status=="F")
        ledger.publications.try_emplace("F|"+ecm_results::factor_identity(report),
                                        ecm_handoff::Publication{submission,report+'\n'});
    ledger.write(progress);
    publish_shared(ledger,progress);
}

std::string shared_nf(const ecm_handoff::Ledger& ledger,const std::vector<Record>& plan,
                      const EcmStage2Task& task) {
    struct Aggregate {
        uint64_t lower,upper,b1,count;
        std::string latest;
    };
    std::vector<Aggregate> aggregates;
    for(const auto& record:plan) {
        if(ledger.cancelled(record.n,record.index))continue;
        const auto& result=ledger.curves.at(record.n).at(record.index);
        if(ecm_results::json::parse(result.report).get("status").text()!="NF")continue;
        const auto compatible=[&](const Aggregate& a) {
            return (std::max)(a.lower,result.b2_min)<(std::min)(a.upper,result.b2);
        };
        auto same=std::find_if(aggregates.begin(),aggregates.end(),compatible);
        if(same==aggregates.end())
            aggregates.push_back({result.b2_min,result.b2,record.b1,1,result.report});
        else {
            same->lower=(std::max)(same->lower,result.b2_min);
            same->upper=(std::min)(same->upper,result.b2);
            ++same->count;
            same->latest=result.report;
        }
    }
    std::string reports;
    for(const auto& a:aggregates) {
        auto report=ecm_results::json::parse(a.latest);
        report.set("b2",ecm_results::json::Value::number(std::to_string(a.upper)));
        if(a.lower!=a.b1 || report.has("b2-min"))
            report.set("b2-min",ecm_results::json::Value::number(std::to_string(a.lower)));
        if(!task.aid.empty())report.set("aid",ecm_results::json::parse(json_string(task.aid)));
        reports+=ecm_results::aggregate_nf(report.dump(),a.count)+'\n';
    }
    return reports;
}

} // namespace

void CurveQueue::commit(uint64_t position,const std::string &outcome) const {
    if(commit_shared)commit_shared(plan.at(position),outcome);
    state.complete(state.reserved.at(position-state.done),outcome);
    if(commit_shared)return;
    state.save(progress);
    publish_outcomes(state,progress);
}

std::string checkpoint_outcome(const Record &record,uint64_t b2,const std::string &report,uint64_t b2_min) {
    const auto bound=(b2_min?std::to_string(b2_min)+"-":std::string{})+std::to_string(b2);
    return "{\"key\":"+json_string(checkpoint_nf_key(record,bound))+
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
    std::set<std::string> waiting_handoffs;
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
        stage2_queue::State state;
        fs::path progress;
        const bool persist=!o.dry&&!o.plan_only;
        bool close_only=false, may_cancel=false;
        const auto finish_row = [&](const std::string& run) {
            append_finished(finished,"# stage2_task_id="+run,task_line,run);
            if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::Remove))
                throw std::runtime_error("task completed but queue could not be advanced; progress retained");
        };
        uint64_t skip=o.skip,count=o.curves,b2=o.b2?o.b2:defaults.b2;
        std::size_t selected_records=0;
        uint64_t first_record_key=0;
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
        const auto defer = [&] {
            report("stage2_handoff_wait: skip="+std::to_string(skip)+" count="+std::to_string(count));
            return waiting_handoffs.insert(task_line).second &&
                ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::AppendToWorker,task_line);
        };
        try {
            if(queue) {
                fields=queue_fields(task_line,engine.input_max_bits);
                skip=fields.skip;count=fields.count;
                if(!o.b2&&fields.b2) {
                    b2=fields.b2;
                    o.b2_min=fields.b2_min;
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
            if(persist){fatal_log=log;fatal_log_worker=o.worker;fatal_log_max_kb=o.log_max_kb;}
            if(queue) {
                progress=ecm_handoff::checkpoint_path(save);
                for(const auto& path:writable)if(same_path(progress,path))
                    throw std::runtime_error("shared Stage2 checkpoint conflicts with output state");
                if(same_path(progress,ini)||same_path(progress,executable())||
                   (!o.tune_profile.empty()&&same_path(progress,fs::path(o.tune_profile))))
                    throw std::runtime_error("shared Stage2 checkpoint conflicts with input");
            }
            if(queue && !fields.task.partial && !fields.task.handoff_group.empty()) {
                const auto ledger=ecm_handoff::Ledger::read(progress);
                const auto group=ledger.producers.find(fields.task.handoff_group);
                if(group==ledger.producers.end() || group->second.status==ecm_handoff::Status::Open ||
                   group->second.modulus!=expected_n || skip!=group->second.saved_start ||
                   count!=group->second.prefix_count)
                    throw TaskInputError("final Stage2 range differs from the Stage1 handoff");
                close_only=group->second.status==ecm_handoff::Status::Factored;
            }
            if(fields.task.partial && persist && !fs::exists(save)) {
                const bool rotated=defer();
                if(startup.dispatch_row)return rotated?DispatchDeferred:0;
                if(rotated && !o.once)continue;
                break;
            }
            const auto projection=fields.task.project_cofactor?expected_n:std::string{};
            if(!close_only || count)
                plan=records(save,skip,count,engine.input_max_bits,engine.edwards_preparation,projection);
            if(fields.task.partial && plan.size()!=count) {
                if(!persist)throw TaskInputError("PART range exceeds available save records");
                const bool rotated=defer();
                if(startup.dispatch_row)return rotated?DispatchDeferred:0;
                if(rotated && !o.once)continue;
                break;
            }
            if(!queue && plan.empty())
                throw TaskInputError("skip exceeds saved curve count or save contains no curves");
            selected_records=plan.size();
            if(!plan.empty())first_record_key=plan.front().hash;
            if(queue && fields.task.modulus_from_save)
                expected_n=queue_modulus(fields.task,plan.empty()?nullptr:&plan.front());
            if(!b2 && cfg.b2_ratio && !plan.empty()) {
                if(plan.front().b1 > (static_cast<uint64_t>(INT64_MAX)-8192)/cfg.b2_ratio)
                    throw std::runtime_error("b2_ratio multiplication exceeds Stage2 B2 range");
                b2=plan.front().b1*cfg.b2_ratio;
                b2_source="b2_ratio";
            }
            if(!close_only && std::any_of(plan.begin(),plan.end(),[](const Record& r){ return !r.skip_stage2; })) {
                if(o.auto_b2&&!b2&&o.cost_profile.empty()&&o.tune_profile.empty())
                    throw std::runtime_error("Auto B2 requires --tune-profile or --cost-profile FILE");
                if(o.carrier_exponent && o.auto_b2 && !b2 && o.tune_profile.empty())
                    throw std::runtime_error("Mersenne carrier Auto B2 requires full ECM tune data");
            }
            for(const auto &r:plan) {
                if(!expected_n.empty()&&r.n!=expected_n)
                    throw TaskInputError(fields.task.modulus_from_save ?
                        "ECMSTAGE2N selected save records must have the same N" : "worktodo N differs from save N");
                if(queue&&r.b1!=plan.front().b1)throw TaskInputError("queue saves must have the same B1");
                if(close_only || r.skip_stage2)continue;
                if(engine.validate)engine.validate(r,b2);
                if(o.b2_min && (o.b2_min<r.b1 || b2<=o.b2_min)) {
                    const std::string message="Stage2 interval requires saved B1 <= B2min < B2";
                    if(b2_source=="worktodo")throw TaskInputError(message);
                    throw std::runtime_error(message);
                }
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
        QueueFileLock progress_lock;
        std::unique_ptr<CurveClaims> claims;
        if(persist && queue) {
            const auto row_key=ecm_stage2::sha256_text(ecm_io::file_identity(save)+"|"+expected_n+"|"+
                fields.task.aid+"|"+std::to_string(skip)+"|"+std::to_string(selected_records)+"|"+
                fields.task.handoff_group);
            bool active=false;
            uint64_t finished_curves=0;
            std::vector<Record> pending;
            {
                // A nonblocking record claim never waits while we hold the ledger writer lock.
                const ecm_io::FileLock lock(progress.u8string());
                auto ledger=ecm_handoff::Ledger::read(progress);
                publish_shared(ledger,progress);
                std::set<uint64_t> factors_verified;
                for(auto& record:plan) {
                    const auto& previous=ledger.curves[record.n];
                    const auto ready=previous.find(record.index);
                    if(ready!=previous.end()) {
                        if(ready->second.point_key!=record.hash)
                            throw std::runtime_error("completed Stage2 record changed: "+std::to_string(record.index));
                        ++finished_curves;
                        continue;
                    }
                    if(const auto* cancelled=ledger.cancelled(record.n,record.index)) {
                        if(cancelled->factor_record!=UINT64_MAX &&
                           factors_verified.insert(cancelled->factor_record).second) {
                            const auto factor=records(save,cancelled->factor_record,1,0,EdwardsPreparation::ValidateOnly);
                            if(factor.size()!=1 || !factor.front().factor || !factor.front().skip_stage2 ||
                               factor.front().n!=record.n)
                                throw std::runtime_error("saved factor differs from the Stage1 handoff");
                        }
                        if(fields.task.partial)continue;
                        // The final task waits for any old-modulus computation already in flight.
                        if(!claims)claims=std::make_unique<CurveClaims>(progress);
                        if(!claims->claim(record.index))active=true;
                        continue;
                    }
                    if(close_only)continue;
                    if(!claims)claims=std::make_unique<CurveClaims>(progress);
                    if(!claims->claim(record.index)) {
                        active=true;
                        continue;
                    }
                    if(!may_cancel)may_cancel=std::any_of(ledger.producers.begin(),ledger.producers.end(),[&](const auto& entry) {
                        const auto& producer=entry.second;
                        return producer.status==ecm_handoff::Status::Open && producer.modulus==record.n &&
                            record.index>producer.saved_start && record.index-producer.saved_start<=producer.saved_count;
                    });
                    pending.push_back(std::move(record));
                }
                if(pending.empty() && !active && !fields.task.partial && !close_only) {
                    const auto id="NF|"+row_key;
                    if(!ledger.publications.count(id)) {
                        auto batch=shared_nf(ledger,plan,fields.task);
                        if(!batch.empty()) {
                            ledger.publications.emplace(id,ecm_handoff::Publication{
                                ecm_io::file_identity(fs::u8path(o.submission)),std::move(batch)});
                            ledger.write(progress);
                            publish_shared(ledger,progress);
                        }
                    }
                }
            }
            if(pending.empty()) {
                if(active) {
                    const bool rotated=defer();
                    if(startup.dispatch_row)return rotated?DispatchDeferred:0;
                    if(rotated && !o.once)continue;
                    break;
                }
                waiting_handoffs.erase(task_line);
                finish_row(row_key);
                report(std::string(fields.task.partial?"stage2_part_done: ":"task_done: ")+
                       "requested="+(count?std::to_string(count):"all")+" available="+
                       std::to_string(plan.size())+" completed="+std::to_string(finished_curves));
                if(startup.dispatch_row)return DispatchAdvanced;
                if(o.once || stop_requests)break;
                continue;
            }
            plan=std::move(pending);
            state.total=plan.size();
        } else if(persist) {
            const auto save_identity = "save-position-v1|" + ecm_io::file_identity(save) + "|" +
                                       std::to_string(first_record_key);
            stage2_queue::Task task;
            task.aid=o.submission_aid;
            task.skip=skip;
            task.total=plan.size();
            task.b2=b2;
            task.b2_min=o.b2_min;
            task.auto_b2=o.auto_b2&&!b2;
            auto identity=save_identity+"|"+std::to_string(selected_records)+"|"+
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
        if(persist && !queue) {
            const auto submission=ecm_io::file_identity(fs::u8path(o.submission));
            if(!state.submission_path.empty() && state.submission_path!=submission)
                throw std::runtime_error("Stage2 checkpoint belongs to a different submission path");
            if(state.submission_path.empty())state.submission_path=submission;
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
            if(o.b2_min)message<<" B2min="<<o.b2_min;
            if(persist)message<<" checkpoint="<<progress.string();
            report(message.str());
        }
        bool task_completed=true;
        std::function<bool()> cancelled;
        if(may_cancel)cancelled=[&] {
            const auto ledger=ecm_handoff::Ledger::read(progress);
            return std::any_of(plan.begin(),plan.end(),[&](const auto& record) {
                return ledger.cancelled(record.n,record.index)!=nullptr;
            });
        };
        if(!o.dry && !o.plan_only) {
            const QueueReport queue_report=report;
            std::function<void(const Record&,const std::string&)> commit_shared;
            if(queue) {
                const auto submission=ecm_io::file_identity(fs::u8path(o.submission));
                commit_shared=[&,submission](const Record& record,const std::string& outcome) {
                    store_shared_curve(progress,submission,record,outcome);
                };
            }
            const CurveQueue curve_queue{plan,save,curve_results,log,progress,defaults,state,completed,
                                         queue_report,cancelled,commit_shared};
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
        if(task_completed && persist && !queue && state.done==state.total)
            publish_outcomes(state,progress,true);
        if(!queue||o.dry||o.plan_only)break;
        if(!task_completed && !(cancelled && cancelled()))break;
        worker_options.runtime_prepare_hints=std::move(o.runtime_prepare_hints);
        worker_options.runtime_route_hints=std::move(o.runtime_route_hints);
    }
    if(stage2_log::enabled(stage2_log::curve)||startup.dry)
        report(std::string(startup.plan_only?"plan_complete":startup.dry?"dry_run_complete":stop_requests?"stage2_stopped":"stage2_complete")+
               ": curves_this_run="+std::to_string(completed)+" invalid_tasks="+std::to_string(failed_tasks));
    return failed_tasks?1:!queue&&stop_requests?130:0;
}
}
