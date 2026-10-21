// Owns process/job/pipe lifetime, stop policy and streamed console projection.
#include "stage2_curve_process.h"
#include "../../service/stage2/input.h"
#include "../../service/planning/stage2.h"
#include "../../service/planning/legacy/ecm_stage2_runtime_cache.h"
#include "../legacy/ecm_stage2_console.h"
#include "../legacy/ecm_stage2_logging.h"
#include "../../storage/results/ecm_result_json.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace s2prod {
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
int child_run(Options &o, const fs::path &save, const Record &r,
              uint64_t b2, uint64_t d, int device, const fs::path &results, const fs::path &log,
              uint64_t batch, const TaskDefaults &ini_settings) {
    const fs::path exe = executable();
    std::wstring cmd = quote(exe.wstring()) + L" --curve-worker --save " + quote(save.wstring());
    auto arg = [&](const wchar_t *key, uint64_t n) { cmd += L" "; cmd += key; cmd += L" "; cmd += std::to_wstring(n); };
    arg(L"--record-offset", r.offset); arg(L"--record-hash", r.hash); arg(L"--record-index", r.index);
    arg(L"--b2", b2); arg(L"--d", d); arg(L"--device", device); arg(L"--worker", o.worker);
    if(o.has_carrier || o.carrier_exponent)arg(L"--carrier-exponent",o.carrier_exponent);
    if(!o.tune_profile.empty())cmd+=L" --tune-profile "+quote(fs::path(o.tune_profile).wstring());
    arg(L"--short-ntt-batch-tune",o.short_batch_tune?1:0);
    arg(L"--pack-zero",o.pack_zero?1:0);
    arg(L"--tune-runtime",o.tune_runtime?1:0);
    arg(L"--tune-runtime-samples",o.runtime_samples);
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
    const int engine_level=ecm_cuda_stage2_default_log_level()==stage2_log::debug?stage2_log::debug:stage2_log::batches;
    arg(L"--log-level",static_cast<uint64_t>(engine_level));
    std::ofstream output;
    if(!log.empty()) {
        fs::create_directories(log.parent_path());
        output.open(log,std::ios::binary|std::ios::app);
        if(!output)throw std::runtime_error("cannot open Stage2 log: "+log.string());
    }
    if(o.debug_log)fs::create_directories(fs::path(o.debug_file).parent_path());
    SECURITY_ATTRIBUTES security{sizeof(security),nullptr,TRUE};
    Handle out_read,out_write,err_read,err_write;
    if(!CreatePipe(&out_read.value,&out_write.value,&security,0) ||
       !CreatePipe(&err_read.value,&err_write.value,&security,0) ||
       !SetHandleInformation(out_read.value,HANDLE_FLAG_INHERIT,0) ||
       !SetHandleInformation(err_read.value,HANDLE_FLAG_INHERIT,0))
        throw std::runtime_error("cannot create curve output pipes");
    Handle job;job.value=CreateJobObjectW(nullptr,nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!job.value || !SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))
        throw std::runtime_error("cannot create curve process job");
    STARTUPINFOW si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdOutput=out_write.value;si.hStdError=err_write.value;si.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    if(!CreateProcessW(exe.c_str(),cmd.data(),nullptr,nullptr,TRUE,CREATE_SUSPENDED|CREATE_NEW_PROCESS_GROUP,nullptr,nullptr,&si,&pi))
        throw std::runtime_error("cannot launch curve process: "+std::to_string(GetLastError()));
    Handle process,thread;process.value=pi.hProcess;thread.value=pi.hThread;
    if(!AssignProcessToJobObject(job.value,pi.hProcess)) {
        TerminateProcess(pi.hProcess,2);throw std::runtime_error("cannot bind curve process lifetime");
    }
    if(ResumeThread(pi.hThread)==static_cast<DWORD>(-1))throw std::runtime_error("cannot resume curve process");
    CloseHandle(out_write.value);out_write.value=INVALID_HANDLE_VALUE;
    CloseHandle(err_write.value);err_write.value=INVALID_HANDLE_VALUE;
    const auto begin=std::chrono::steady_clock::now();
    auto next=begin+std::chrono::seconds(30);
    std::string phase="Starting curve",stdout_pending,stderr_pending;
    bool heartbeat=false,stop_reported=false;
    stage2_console::Summary console_summary(batch,ini_settings.arena,ini_settings.owner_mb);
    auto console_line=[&](const std::string &text,bool error=false) {
        if(heartbeat){std::cout<<'\n';heartbeat=false;}
        (error?std::cerr:std::cout)<<text<<std::endl;
    };
    auto line=[&](const std::string &text,bool error) {
        if(!error && o.runtime_prepare_cache && text.rfind("{\"type\":\"tune_selection\"",0)==0) {
            try {
                const auto event=ecm_results::json::parse(text);
                ecm_stage2::tune::portable::RuntimePauseHints hints;
                hints.assign(event.get("runtime_prepare_hints").text());o.runtime_prepare_hints=hints.text();
                const auto route_text=event.get("runtime_route_hints").text();
                ecm_stage2::tune::portable::RuntimeRouteHints routes;
                if(open_runtime_routes(route_text,routes))o.runtime_route_hints=route_text;else o.runtime_route_hints.clear();
            }catch(const std::runtime_error &){o.runtime_prepare_hints.clear();o.runtime_route_hints.clear();}
        }
        if(output.is_open()){output<<text<<'\n';output.flush();if(!output)throw std::runtime_error("Stage2 log write failed");}
        if(o.tune_child)return; // Full diagnostic evidence stays in the per-trial file.
        const auto summary=error?stage2_console::Projection{}:console_summary.observe(text);
        if(stage2_log::enabled(stage2_log::phases))
            for(const auto &entry:summary.lines)console_line(entry);
        const bool milestone=text.compare(0,14,"stage2_phase: ")==0;
        const bool tune_update=text.compare(0,21,"stage2_tune_runtime: ")==0;
        const bool result=text.compare(0,15,"stage2_result: ")==0;
        if(milestone){phase=text.substr(14);const auto timing=phase.find(" previous=");if(timing!=phase.npos)phase.resize(timing);}
        if(!summary.replace && (error || (stage2_log::enabled(stage2_log::phases)&&(milestone||tune_update)) ||
           ((stage2_log::enabled(stage2_log::curve)||o.gui_events)&&(result || text.rfind("stage2_relocation:",0)==0)) ||
           stage2_log::enabled(stage2_log::batches) ||
           (o.ntt_counts && (text.rfind("ntt_count_",0)==0 || text.rfind("ntt_length_counts:",0)==0 || text.rfind("ntt_work:",0)==0)) ||
           text.find("WARNING")!=text.npos))console_line(text,error);
    };
    auto drain=[&](HANDLE pipe,std::string &pending,bool error) {
        DWORD available=0;
        while(PeekNamedPipe(pipe,nullptr,0,nullptr,&available,nullptr) && available) {
            char buffer[4096];DWORD got=0;
            if(!ReadFile(pipe,buffer,std::min<DWORD>(available,sizeof(buffer)),&got,nullptr) || !got)break;
            pending.append(buffer,got);
            size_t pos;
            while((pos=pending.find('\n'))!=std::string::npos) {
                std::string text=pending.substr(0,pos);pending.erase(0,pos+1);
                if(!text.empty()&&text.back()=='\r')text.pop_back();
                line(text,error);
            }
        }
    };
    DWORD wait=WAIT_TIMEOUT;
    for(;;) {
        poll_inherited_stop();
        if(stop_requests>=2) {
            TerminateJobObject(job.value,130);
            WaitForSingleObject(pi.hProcess,5000);
            throw std::runtime_error("immediate stop requested; incomplete curve will restart next time");
        }
        if(stop_requests && !stop_reported) {
            line("Stop requested: finishing current curve; press Ctrl+C again to terminate immediately.",true);
            stop_reported=true;
        }
        drain(out_read.value,stdout_pending,false);drain(err_read.value,stderr_pending,true);
        wait=WaitForSingleObject(pi.hProcess,50);
        if(wait==WAIT_OBJECT_0)break;
        if(wait!=WAIT_TIMEOUT)throw std::runtime_error("curve process wait failed");
        const auto now=std::chrono::steady_clock::now();
        if(now>=next && !o.tune_child && stage2_log::enabled(stage2_log::curve)) {
            std::cout<<'\r'<<"Curve "<<r.index<<": "<<phase<<" | elapsed="
                     <<static_cast<unsigned long long>(std::chrono::duration<double>(now-begin).count())<<" s     "<<std::flush;
            heartbeat=true;next=now+std::chrono::seconds(30);
        }
    }
    drain(out_read.value,stdout_pending,false);drain(err_read.value,stderr_pending,true);
    if(!stdout_pending.empty())line(stdout_pending,false);
    if(!stderr_pending.empty())line(stderr_pending,true);
    if(!o.tune_child && stage2_log::enabled(stage2_log::phases))
        for(const auto &entry:console_summary.finish().lines)console_line(entry);
    if(heartbeat)std::cout<<std::endl;
    DWORD code=1;
    if(!GetExitCodeProcess(pi.hProcess,&code))throw std::runtime_error("cannot read curve exit status");
    return code==0?0:(code==3?3:1);
}
}
