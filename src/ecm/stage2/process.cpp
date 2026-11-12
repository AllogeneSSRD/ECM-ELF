// Owns process/job/pipe lifetime, stop policy and streamed console projection.
#include "process.h"
#include "console.h"
#include "../platform/compute_log.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace s2prod {
namespace fs=std::filesystem;
volatile LONG stop_requests = 0;
HANDLE inherited_stop_event=nullptr;
fs::path executable() {
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (!n || n == buf.size()) throw std::runtime_error("cannot locate executable");
    return fs::path(std::wstring(buf.data(), n));
}
void poll_inherited_stop() {
    if(inherited_stop_event && WaitForSingleObject(inherited_stop_event,0)==WAIT_OBJECT_0)
        InterlockedCompareExchange(&stop_requests,1,0);
}
BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    InterlockedIncrement(&stop_requests);
    return TRUE;
}
unsigned queue_pipeline_level(unsigned arithmetic_bits,unsigned workers,bool factor_only) {
    const char *value=std::getenv("ECM_STAGE2_QUEUE_PIPELINE");
    if(value)return (value[0]=='1' || value[0]=='2') && !value[1]?unsigned(value[0]-'0'):0;
    const unsigned limit=workers==2 && !factor_only?2048:4096;
    return arithmetic_bits && arithmetic_bits<=limit?2:0;
}
bool early_gpu_handoff_requested() {
    const char *value=std::getenv("ECM_STAGE2_EARLY_GPU_HANDOFF");
    return !value || (value[0]=='1' && !value[1]);
}
std::wstring quote(const std::wstring &s) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : s) {
        if (c == L'\\') { ++slashes; continue; }
        out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0; out += c;
    }
    out.append(2 * slashes, L'\\');
    return out + L'"';
}
struct CurveProcess::Impl {
    using Clock = std::chrono::steady_clock;
    CurveProcessRequest request;
    Handle out_read,err_read,job,process,gpu_start,gpu_done,gpu_input;
    Clock::time_point begin,next;
    std::string phase="Starting curve",stdout_pending,stderr_pending,queue_sample;
    std::string curve_label;
    std::optional<std::string> gpu_feedback;
    bool heartbeat=false,stop_reported=false;
    stage2_console::Summary console_summary;

    explicit Impl(CurveProcessRequest value);
    void console_line(const std::string &text,bool error=false);
    void line(const std::string &text,bool error,std::ofstream &output);
    void drain(HANDLE pipe,std::string &pending,bool error,std::ofstream &output);
    std::optional<int> poll();
};

CurveProcess::Impl::Impl(CurveProcessRequest value)
    : request(std::move(value)),
      console_summary(request.batch,request.console_budget.arena,request.console_budget.owner_mb) {
    const auto &o=request;
    if(o.queue_curve)curve_label="[curve "+std::to_string(o.queue_curve)+"] ";
    const auto &save=o.save,&results=o.results,&log=o.log;
    const auto &r=o.record;
    const auto b2=o.b2,d=o.d,batch=o.batch;const int device=o.device;
    const fs::path exe = executable();
    std::wstring cmd = quote(exe.wstring()) + L" --curve-worker --save " + quote(save.wstring());
    auto arg = [&](const wchar_t *key, uint64_t n) { cmd += L" "; cmd += key; cmd += L" "; cmd += std::to_wstring(n); };
    SECURITY_ATTRIBUTES security{sizeof(security),nullptr,TRUE};
    if(o.queue_pipeline) {
        gpu_start.value=CreateEventW(&security,TRUE,FALSE,nullptr);
        gpu_done.value=CreateEventW(&security,TRUE,FALSE,nullptr);
        if(!gpu_start.value || !gpu_done.value)throw std::runtime_error("cannot create Stage2 pipeline events");
        arg(L"--queue-gpu-start",reinterpret_cast<uintptr_t>(gpu_start.value));
        arg(L"--queue-gpu-done",reinterpret_cast<uintptr_t>(gpu_done.value));
        if(o.tune_runtime) {
            gpu_input.value=CreateFileMappingW(INVALID_HANDLE_VALUE,&security,PAGE_READWRITE,0,queue_feedback_bytes,nullptr);
            if(!gpu_input.value)throw std::runtime_error("cannot allocate Stage2 pipeline feedback");
            arg(L"--queue-gpu-input",reinterpret_cast<uintptr_t>(gpu_input.value));
        }
    }
    arg(L"--record-offset", r.offset); arg(L"--record-hash", r.hash); arg(L"--record-index", r.index);
    arg(L"--b2", b2); arg(L"--d", d); arg(L"--device", device); arg(L"--worker", o.worker);
    if(o.has_carrier || o.carrier_exponent)arg(L"--carrier-exponent",o.carrier_exponent);
    if(!o.tune_profile.empty())cmd+=L" --tune-profile "+quote(fs::path(o.tune_profile).wstring());
    if(o.groot_layout>=0)arg(L"--groot-layout",uint64_t(o.groot_layout));
    else cmd+=L" --groot-layout auto"; // Preserve explicit auto over a forced TOML value.
    arg(L"--short-ntt-batch-tune",o.short_batch_tune?1:0);
    arg(L"--pack-zero",o.pack_zero?1:0);
    if(o.weighted_ntt)arg(L"--weighted-ntt",1);
    arg(L"--tune-runtime",o.tune_runtime?1:0);
    if(o.fixed_queue_route)cmd+=L" --fixed-queue-route";
    if(!o.queue_phase_gate.empty())cmd+=L" --queue-phase-gate "+quote(fs::path(o.queue_phase_gate).wstring());
    arg(L"--tune-runtime-samples",o.runtime_samples);
    arg(L"--tune-min-repeats",o.tune_min_repeats);
    arg(L"--runtime-remaining-curves",o.runtime_remaining);
    arg(L"--runtime-prepare-cache",o.runtime_prepare_cache?1:0);
    if(o.runtime_prepare_cache && !o.runtime_prepare_hints.empty())
        cmd+=L" --runtime-prepare-hints "+quote(std::wstring(o.runtime_prepare_hints.begin(),o.runtime_prepare_hints.end()));
    if(o.runtime_prepare_cache && !o.runtime_route_hints.empty())
        cmd+=L" --runtime-route-hints "+quote(std::wstring(o.runtime_route_hints.begin(),o.runtime_route_hints.end()));
    cmd+=L" --tune-ignore "+quote(fs::path(o.tune_ignore).wstring());
    if(!o.stage1_csv.empty())cmd+=L" --stage1-cost-csv "+quote(fs::path(o.stage1_csv).wstring());
    if(!o.condition_tag.empty())cmd+=L" --tune-condition-tag "+quote(fs::path(o.condition_tag).wstring());
    arg(L"--stage1-cost-mhz",o.stage1_mhz);arg(L"--target-factor-bits",o.factor_bits);
    arg(L"--inferred-carrier-exponent",o.inferred_carrier);
    arg(L"--auto-carrier",o.auto_carrier?1:0);
    cmd+=L" --prediction-policy "+quote(fs::path(o.prediction_policy).wstring());
    // Startup calibration belongs to the parent queue, never every curve child.
    arg(L"--short-calibration",0);
    arg(L"--batch-mb",batch);arg(L"--arena-mb",o.arena);arg(L"--owner-budget-mb",o.owner_mb);
    if(o.tune_child)cmd+=L" --ecm-tune-worker";
    arg(L"--log-level", static_cast<uint64_t>(o.log_level));
    arg(L"--ntt-counts",o.ntt_counts?1:0);
    arg(L"--relocation",o.relocation?1:0);
    arg(L"--giant-tail-balance",o.giant_tail_balance?1:0);
    arg(L"--b2-round-chain",o.b2_round_chain?1:0);
    cmd+=L" --giant-chain "+quote(std::wstring(o.giant_chain.begin(),o.giant_chain.end()));
    cmd+=L" --baby-backend "+quote(std::wstring(o.baby_backend.begin(),o.baby_backend.end()));
    cmd+=L" --giant-backend "+quote(std::wstring(o.giant_backend.begin(),o.giant_backend.end()));
    cmd+=L" --seed-backend "+quote(std::wstring(o.seed_backend.begin(),o.seed_backend.end()));
    cmd+=L" --reduce-backend "+quote(std::wstring(o.reduce_backend.begin(),o.reduce_backend.end()));
    cmd+=L" --reduce-threads "+std::to_wstring(o.reduce_threads)+L" --reduce-kernel "+std::to_wstring(o.reduce_kernel);
    cmd += L" --results " + quote(results.wstring());
    if(!o.submission.empty())cmd+=L" --submission "+quote(fs::path(o.submission).wstring());
    if(!o.submission_ini.empty())cmd+=L" --config "+quote(fs::path(o.submission_ini).wstring());
    if(o.submission_exponent)arg(L"--exponent",o.submission_exponent);
    for(const auto &factor:o.submission_known)cmd+=L" --known-factor "+quote(fs::path(factor).wstring());
    if(!o.submission_aid.empty())cmd+=L" --aid "+quote(fs::path(o.submission_aid).wstring());
    if (o.factorize_hits) {
        cmd += L" --factorize-hits --gp " + quote(fs::path(o.gp).wstring());
        arg(L"--factor-timeout", o.factor_timeout);
    }
    if (o.factor_only) cmd += L" --factor-only";
    if (o.auto_b2 && !b2) {
        auto real=[](double value){std::wostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)<<value;return out.str();};
        cmd += L" --auto-b2";
        if(!o.cost_profile.empty())cmd+=L" --cost-profile " + quote(fs::path(o.cost_profile).wstring());
        arg(L"--arena-mb",o.arena);arg(L"--owner-budget-mb",o.owner_mb);arg(L"--stage1-batch",o.stage1_batch);
        if(o.auto_min)arg(L"--auto-min-b2",o.auto_min);if(o.auto_max)arg(L"--auto-max-b2",o.auto_max);
        if(o.stage1_seconds)cmd+=L" --stage1-seconds-per-curve "+real(o.stage1_seconds);
        if(!o.stage1_profile.empty())cmd+=L" --stage1-tune-profile "+quote(fs::path(o.stage1_profile).wstring());
        cmd+=L" --lcm "+quote(fs::path(o.stage1_exponent=="choose12"?"12":"1").wstring());
        cmd+=L" --stage2-ratio-adjust "+real(o.ratio_adjust);
    }
    if(!o.publication_task.empty())cmd+=L" --publication-task "+quote(fs::path(o.publication_task).wstring());
    if(!o.receipt.empty())cmd+=L" --queue-receipt "+quote(fs::path(o.receipt).wstring());
    if(o.debug_log)cmd+=L" --debug-log-file "+quote(fs::path(o.debug_file).wstring());
    // Preserve readable batch statistics in the file even with a concise console.
    const int engine_level=o.engine_log_level;
    arg(L"--log-level",static_cast<uint64_t>(engine_level));
    if(!log.empty()) {
        fs::create_directories(log.parent_path());
        std::ofstream output(log,std::ios::binary|std::ios::app);
        if(!output)throw std::runtime_error("cannot open Stage2 log: "+log.string());
    }
    if(o.debug_log)fs::create_directories(fs::path(o.debug_file).parent_path());
    Handle out_write,err_write;
    if(!CreatePipe(&out_read.value,&out_write.value,&security,0) ||
       !CreatePipe(&err_read.value,&err_write.value,&security,0) ||
       !SetHandleInformation(out_read.value,HANDLE_FLAG_INHERIT,0) ||
       !SetHandleInformation(err_read.value,HANDLE_FLAG_INHERIT,0))
        throw std::runtime_error("cannot create curve output pipes");
    job.value=CreateJobObjectW(nullptr,nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!job.value || !SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))
        throw std::runtime_error("cannot create curve process job");
    STARTUPINFOW si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdOutput=out_write.value;si.hStdError=err_write.value;si.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    if(!CreateProcessW(exe.c_str(),cmd.data(),nullptr,nullptr,TRUE,CREATE_SUSPENDED|CREATE_NEW_PROCESS_GROUP,nullptr,nullptr,&si,&pi))
        throw std::runtime_error("cannot launch curve process: "+std::to_string(GetLastError()));
    Handle thread;process.value=pi.hProcess;thread.value=pi.hThread;
    if(!AssignProcessToJobObject(job.value,pi.hProcess)) {
        TerminateProcess(pi.hProcess,2);throw std::runtime_error("cannot bind curve process lifetime");
    }
    if(o.queue_pipeline && (!SetHandleInformation(gpu_start.value,HANDLE_FLAG_INHERIT,0) ||
                            !SetHandleInformation(gpu_done.value,HANDLE_FLAG_INHERIT,0) ||
                            (o.tune_runtime && !SetHandleInformation(gpu_input.value,HANDLE_FLAG_INHERIT,0))))
        throw std::runtime_error("cannot isolate Stage2 pipeline events");
    if(ResumeThread(pi.hThread)==static_cast<DWORD>(-1))throw std::runtime_error("cannot resume curve process");
    CloseHandle(out_write.value);out_write.value=INVALID_HANDLE_VALUE;
    CloseHandle(err_write.value);err_write.value=INVALID_HANDLE_VALUE;
    begin=Clock::now();
    next=begin+std::chrono::seconds(30);
}

void CurveProcess::Impl::console_line(const std::string &text,bool error) {
    if(heartbeat){std::cout<<'\n';heartbeat=false;}
    (error?std::cerr:std::cout)<<curve_label<<text<<std::endl;
}

void CurveProcess::Impl::line(const std::string &text,bool error,std::ofstream &output) {
    const auto &o=request;
    constexpr char feedback_prefix[]="stage2_queue_observation: ";
    if(!error && o.queue_pipeline && o.tune_runtime && text.rfind(feedback_prefix,0)==0) {
        gpu_feedback=text.substr(sizeof(feedback_prefix)-1);
        return;
    }
    constexpr char sample_prefix[]="stage2_parallel_curve: ";
    if(!error && o.fixed_queue_route && text.rfind(sample_prefix,0)==0)
        queue_sample=text.substr(sizeof(sample_prefix)-1);
    if(!error && o.runtime_prepare_cache && text.rfind("{\"type\":\"tune_selection\"",0)==0) {
        if(o.observe_selection)o.observe_selection(text);
    }
    const bool milestone=text.rfind("stage2_phase: ",0)==0;
    const bool result=text.rfind("stage2_result: ",0)==0;
    const bool detail=text.rfind("batched_progress:",0)==0||text.rfind("stage2_full_wall:",0)==0||
        text.rfind("stage2_modulus:",0)==0||text.rfind("real_shape:",0)==0;
    const bool tagged=milestone||result||detail||error||text.find("WARNING")!=text.npos;
    // Release the log between pumps so task milestones and both children can append.
    if(!o.log.empty()){
        if(!output.is_open())output.open(o.log,std::ios::binary|std::ios::app);
        // Structured benchmark records keep their existing framing.
        if(tagged)output<<curve_label;
        output<<text<<'\n';output.flush();if(!output)throw std::runtime_error("Stage2 log write failed");
    }
    if(o.tune_child)return; // Full diagnostic evidence stays in the per-trial file.
    const auto summary=error?stage2_console::Projection{}:console_summary.observe(text);
    if(stage2_log::enabled(stage2_log::phases))
        for(const auto &entry:summary.lines)console_line(entry);
    const bool tune_update=text.compare(0,21,"stage2_tune_runtime: ")==0;
    if(milestone){phase=text.substr(14);const auto timing=phase.find(" previous=");if(timing!=phase.npos)phase.resize(timing);}
    const bool gui_detail=o.gui_events&&(milestone||detail);
    if((!summary.replace||(gui_detail&&!stage2_log::enabled(stage2_log::phases))) &&
       (gui_detail || error || (stage2_log::enabled(stage2_log::phases)&&(milestone||tune_update)) ||
       ((stage2_log::enabled(stage2_log::curve)||o.gui_events)&&(result || text.rfind("stage2_relocation:",0)==0)) ||
       stage2_log::enabled(stage2_log::batches) ||
       (o.ntt_counts && (text.rfind("ntt_count_",0)==0 || text.rfind("ntt_length_counts:",0)==0 || text.rfind("ntt_work:",0)==0)) ||
       text.find("WARNING")!=text.npos)){
        if(tagged)console_line(text,error);
        else (error?std::cerr:std::cout)<<text<<std::endl;
    }
}

void CurveProcess::Impl::drain(HANDLE pipe,std::string &pending,bool error,std::ofstream &output) {
    DWORD available=0;
    while(PeekNamedPipe(pipe,nullptr,0,nullptr,&available,nullptr) && available) {
        char buffer[4096];DWORD got=0;
        if(!ReadFile(pipe,buffer,std::min<DWORD>(available,sizeof(buffer)),&got,nullptr) || !got)break;
        pending.append(buffer,got);
        size_t pos;
        while((pos=pending.find('\n'))!=std::string::npos) {
            std::string text=pending.substr(0,pos);pending.erase(0,pos+1);
            if(!text.empty()&&text.back()=='\r')text.pop_back();
            line(text,error,output);
        }
    }
}

std::optional<int> CurveProcess::Impl::poll() {
    const auto &o=request;
    std::ofstream output;
    poll_inherited_stop();
    if(stop_requests>=2) {
        TerminateJobObject(job.value,130);
        WaitForSingleObject(process.value,5000);
        throw std::runtime_error("immediate stop requested; incomplete curve will restart next time");
    }
    if(stop_requests && !stop_reported) {
        line("Stop requested: finishing active curves. Press Ctrl+C again to terminate immediately.",true,output);
        stop_reported=true;
    }
    drain(out_read.value,stdout_pending,false,output);drain(err_read.value,stderr_pending,true,output);
    const DWORD wait=WaitForSingleObject(process.value,0);
    if(wait==WAIT_TIMEOUT) {
        const auto now=Clock::now();
        if(now>=next && !o.tune_child && stage2_log::enabled(stage2_log::curve)) {
            if(o.queue_curve)console_line(phase+" | elapsed="+
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now-begin).count())+" s");
            else {
                std::cout<<'\r'<<"Curve "<<o.record.index<<": "<<phase<<" | elapsed="
                         <<std::chrono::duration_cast<std::chrono::seconds>(now-begin).count()<<" s     "<<std::flush;
                heartbeat=true;
            }
            next=now+std::chrono::seconds(30);
    }
        return std::nullopt;
        }
    if(wait!=WAIT_OBJECT_0)throw std::runtime_error("curve process wait failed");
    drain(out_read.value,stdout_pending,false,output);drain(err_read.value,stderr_pending,true,output);
    if(!stdout_pending.empty())line(stdout_pending,false,output);
    if(!stderr_pending.empty())line(stderr_pending,true,output);
    if(!o.tune_child && stage2_log::enabled(stage2_log::phases))
        for(const auto &entry:console_summary.finish().lines)console_line(entry);
    if(heartbeat)std::cout<<std::endl;
    DWORD code=1;
    if(!GetExitCodeProcess(process.value,&code))throw std::runtime_error("cannot read curve exit status");
    return code==0?0:(code==3?3:1);
}

CurveProcess::CurveProcess(CurveProcessRequest request):state_(std::make_unique<Impl>(std::move(request))) {}
CurveProcess::~CurveProcess() = default;
HANDLE CurveProcess::handle() const {return state_->process.value;}
HANDLE CurveProcess::gpu_done_handle() const {return state_->gpu_done.value;}
void CurveProcess::allow_gpu(const std::string &feedback) {
    if(!feedback.empty()) {
        if(feedback.size()>=queue_feedback_bytes)throw std::runtime_error("Stage2 pipeline feedback exceeds its transport");
        void *view=MapViewOfFile(state_->gpu_input.value,FILE_MAP_WRITE,0,0,queue_feedback_bytes);
        if(!view)throw std::runtime_error("cannot send Stage2 pipeline feedback");
        std::memcpy(view,feedback.c_str(),feedback.size()+1);
        UnmapViewOfFile(view);
    }
    if(!SetEvent(state_->gpu_start.value))throw std::runtime_error("cannot admit Stage2 pipeline worker");
}
bool CurveProcess::gpu_released() const {
    return (!state_->request.tune_runtime || state_->gpu_feedback.has_value()) &&
        WaitForSingleObject(state_->gpu_done.value,0)==WAIT_OBJECT_0;
}
const std::optional<std::string> &CurveProcess::gpu_feedback() const {return state_->gpu_feedback;}
std::optional<int> CurveProcess::poll() {return state_->poll();}
const std::string &CurveProcess::queue_sample() const {return state_->queue_sample;}

int run_curve_process(const CurveProcessRequest &request) {
    CurveProcess process(request);
    while(true) {
        if(const auto result=process.poll())return *result;
        if(WaitForSingleObject(process.handle(),50)==WAIT_FAILED)
            throw std::runtime_error("curve process wait failed");
    }
}
}
