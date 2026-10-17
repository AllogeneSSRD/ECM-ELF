#pragma once
#include "ecm_result_submission.h"
#include "ecm_edwards_io.h"
#include "ecm_file_publication.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <chrono>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ecm_results {
namespace fs=std::filesystem;
inline void atomic_text(const fs::path &path,const std::string &text) {
    const auto temp=fs::path(path.string()+".new");
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot prepare submission output");
    out<<text;out.close();
    if(!out)throw std::runtime_error("cannot finish submission output write");
    if(!ecm_replace_file(temp,path))
#ifdef _WIN32
        throw std::runtime_error("cannot publish submission output; Windows error="+std::to_string(GetLastError()));
#else
        throw std::runtime_error("cannot publish submission output");
#endif
}
struct SubmissionLock {
#ifdef _WIN32
    HANDLE value=INVALID_HANDLE_VALUE;
#else
    int value=-1;
#endif
    explicit SubmissionLock(const fs::path &path) {
        if(!path.parent_path().empty())fs::create_directories(path.parent_path());
        const auto lock=fs::path(path.string()+".ecm-ledger.lock");
#ifdef _WIN32
        value=CreateFileW(lock.c_str(),GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
        if(value==INVALID_HANDLE_VALUE)throw std::runtime_error("another writer owns the submission output");
#else
        value=::open(lock.c_str(),O_CREAT|O_WRONLY,0600);
        if(value<0)throw std::runtime_error("cannot open result writer lock");
        if(::flock(value,LOCK_EX|LOCK_NB)){::close(value);value=-1;throw std::runtime_error("another writer owns the submission output");}
#endif
    }
    ~SubmissionLock(){
#ifdef _WIN32
        if(value!=INVALID_HANDLE_VALUE)CloseHandle(value);
#else
        if(value>=0){::flock(value,LOCK_UN);::close(value);}
#endif
    }
    SubmissionLock(const SubmissionLock&)=delete;
};
struct SubmissionEntry {std::string key,group,json;};
inline SubmissionEntry read_submission_entry(const fs::path &path) {
    std::ifstream in(path,std::ios::binary);
    std::string tag,key,group,json,checksum,extra;
    if(!std::getline(in,tag) || tag!="ECM_SUBMISSION_V1" || !std::getline(in,key) ||
       !std::getline(in,group) || !std::getline(in,json) || !std::getline(in,checksum) || std::getline(in,extra) || in.bad())
        throw std::runtime_error("damaged submission ledger entry");
    const auto body=tag+'\n'+key+'\n'+group+'\n'+json+'\n';
    if(checksum!=std::to_string(ecm_edwards::hash(body)))throw std::runtime_error("submission ledger checksum mismatch");
    return {key,group,json};
}
inline fs::path submission_ledger(const fs::path &path){return fs::path(path.string()+".ecm-ledger");}
inline std::string stage1_factor_key(const std::string &key) {
    std::istringstream in(key);std::vector<std::string> fields;std::string field;
    while(std::getline(in,field,'|'))fields.push_back(field);
    if(fields.size()<7 || fields[4]!="1" ||
       (fields[5]!="ed" && fields[5]!="mont0" && fields[5]!="mont2" && fields[5]!="mont3"))return key;
    // Stage1 factors have no completed B2. Reading their save in Stage2 must
    // retain the same receipt, regardless of the requested continuation bound.
    fields[3]="0";std::string result;
    for(const auto &value:fields){if(!result.empty())result+='|';result+=value;}return result;
}
inline void require_ledger_format(const fs::path &ledger,const std::string &format) {
    std::ifstream file(ledger/"format",std::ios::binary);std::string value,extra;
    if(!std::getline(file,value) || value!=format || std::getline(file,extra))
        throw std::runtime_error("result ledger has a different owner/format");
}
inline void render_submission(const fs::path &path,const fs::path &ledger) {
    std::map<std::string,std::string> batches;
    std::map<std::string,std::string> factors;
    std::map<std::string,fs::file_time_type> factor_times;
    for(const auto &file:fs::directory_iterator(ledger)) {
        if(file.path().extension()!=".curve" && file.path().extension()!=".batch")continue;
        const auto entry=read_submission_entry(file.path());
        if(file.path().extension()==".batch")batches.emplace(entry.key,entry.json);
        else if(entry.group=="F") {
            const auto key=stage1_factor_key(entry.key);const auto time=fs::last_write_time(file.path());
            if(!factors.count(key) || time<factor_times.at(key)){factors[key]=entry.json;factor_times[key]=time;}
        }
    }
    std::string output;
    for(const auto &pair:batches)output+=pair.second+'\n';
    for(const auto &pair:factors)output+=pair.second+'\n';
    atomic_text(path,output);
}
inline void seal_submissions(const fs::path &path) {
    const auto ledger=submission_ledger(path);if(!fs::exists(ledger/"format"))return;
    SubmissionLock lock(path);
    require_ledger_format(ledger,"ECM_SUBMISSION_LEDGER_V1");
    std::set<std::string> sealed,lines;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".batch") {
        const auto batch=read_submission_entry(file.path());lines.insert(batch.json);
        std::istringstream ids(batch.group);std::string id;
        while(std::getline(ids,id,','))if(!sealed.insert(id).second)throw std::runtime_error("NF curve occurs in two submission batches");
    }
    std::map<std::string,std::vector<std::pair<std::string,SubmissionEntry>>> pending;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve") {
        const auto id=file.path().stem().string();const auto entry=read_submission_entry(file.path());
        if(entry.group!="F" && !sealed.count(id))pending[entry.group].push_back({id,entry});
    }
    for(auto &group:pending) {
        auto &entries=group.second;std::sort(entries.begin(),entries.end(),[](const auto &a,const auto &b){return a.first<b.first;});
        std::string ids;for(const auto &entry:entries){if(!ids.empty())ids+=',';ids+=entry.first;}
        auto json=entries.back().second.json;const std::string count="\"curves\":1,";
        const auto at=json.find(count);if(at==json.npos)throw std::runtime_error("invalid NF ledger count");
        json.replace(at,count.size(),"\"curves\":"+std::to_string(entries.size())+",");
        const std::string time="\"timestamp\":\"";const auto ts=json.find(time);
        if(ts==json.npos)throw std::runtime_error("invalid NF timestamp");
        // Same-second batches can otherwise be byte-identical. Wait for a real
        // timestamp; never invent a future date or re-report cumulative counts.
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        for(;;) {
            json.replace(ts+time.size(),19,utc_timestamp());
            if(!lines.count(json))break;
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("system clock did not advance for NF report");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto key=std::to_string(ecm_edwards::hash(ids));
        const auto body="ECM_SUBMISSION_V1\n"+key+'\n'+ids+'\n'+json+'\n';
        atomic_text(ledger/(key+".batch"),body+std::to_string(ecm_edwards::hash(body))+'\n');
        lines.insert(json);
    }
    render_submission(path,ledger);
}
inline void refresh_submission_if_owned(const fs::path &path) {
    const auto ledger=submission_ledger(path);
    if(!fs::exists(ledger/"format"))return;
    require_ledger_format(ledger,"ECM_SUBMISSION_LEDGER_V1");
    SubmissionLock lock(path);render_submission(path,ledger);
}
// The caller holds SubmissionLock, including while establishing ownership.
inline fs::path prepare_submission_ledger(const fs::path &path) {
    const auto ledger=submission_ledger(path);
    if(!fs::exists(ledger/"format")) {
        // Never replace an unrelated Prime95/manual result file.
        if(fs::exists(path))throw std::runtime_error("submission file has no owned ledger; select a fresh output path");
        fs::create_directories(ledger);atomic_text(ledger/"format","ECM_SUBMISSION_LEDGER_V1\n");
    }
    require_ledger_format(ledger,"ECM_SUBMISSION_LEDGER_V1");
    return ledger;
}
inline void ensure_submission_output(const fs::path &path) {
    SubmissionLock lock(path);
    render_submission(path,prepare_submission_ledger(path));
}
inline void publish_submission(const fs::path &path,const Submission &r,const mpz_t n,bool render_now=true) {
    if(r.curves!=1)throw std::runtime_error("submission ledger accepts individual completed curves");
    auto line=submission_json(r,n);line.pop_back();
    SubmissionLock lock(path);const auto ledger=prepare_submission_ledger(path);
    auto key=ecm_edwards::hex(n)+"|"+std::to_string(r.sigma)+"|"+std::to_string(r.b1)+"|"+
        std::to_string(r.b2)+"|"+std::to_string(r.stage)+"|"+(r.edwards?"ed":"mont"+std::to_string(r.param));
    for(const auto &factor:r.factors)key+='|'+factor;
    key=stage1_factor_key(key);
    auto group=r;group.when="1970-01-01 00:00:00";
    const auto group_key=r.factors.empty()?submission_json(group,n):"F";
    const auto group_line=r.factors.empty()?group_key.substr(0,group_key.size()-1):group_key;
    const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
    if(r.stage==1 && !fs::exists(entry)) {
        // Preserve older receipts and their first owner; their filenames used
        // the continuation B2. Export also collapses existing legacy duplicates.
        for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve") {
            const auto previous=read_submission_entry(file.path());
            if(previous.group=="F" && stage1_factor_key(previous.key)==key) {
                if(render_now)render_submission(path,ledger);return;
            }
        }
    }
    if(fs::exists(entry)) {
        if(read_submission_entry(entry).key!=key)throw std::runtime_error("submission ledger identity collision");
    } else {
        const auto body="ECM_SUBMISSION_V1\n"+key+'\n'+group_line+'\n'+line+'\n';
        atomic_text(entry,body+std::to_string(ecm_edwards::hash(body))+'\n');
    }
    // Factors publish immediately. NF receipts are sealed at a normal task end;
    // immutable batches prevent cumulative counts being credited twice.
    if(render_now)render_submission(path,ledger);
}
// Internal records have independent ownership and durable per-curve receipts.
// Replaying a completed checkpoint after an output failure must not append a
// second result for the same mathematical work.
inline void publish_internal(const fs::path &path,const std::string &key,const std::string &json,bool render_now=true) {
    SubmissionLock lock(path);const auto ledger=submission_ledger(path);
    if(!fs::exists(ledger/"format")) {
        if(fs::exists(path))throw std::runtime_error("internal result file has no owned ledger; select a fresh output path");
        fs::create_directories(ledger);atomic_text(ledger/"format","ECM_INTERNAL_LEDGER_V1\n");
    }
    require_ledger_format(ledger,"ECM_INTERNAL_LEDGER_V1");
    const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
    auto line=json;if(!line.empty() && line.back()=='\n')line.pop_back();
    if(fs::exists(entry)) {
        if(read_submission_entry(entry).key!=key)throw std::runtime_error("internal result identity collision");
    } else {
        const auto body="ECM_SUBMISSION_V1\n"+key+"\nF\n"+line+'\n';
        atomic_text(entry,body+std::to_string(ecm_edwards::hash(body))+'\n');
    }
    if(render_now)render_submission(path,ledger);
}
inline void refresh_internal(const fs::path &path) {
    SubmissionLock lock(path);const auto ledger=submission_ledger(path);
    require_ledger_format(ledger,"ECM_INTERNAL_LEDGER_V1");render_submission(path,ledger);
}
} // namespace ecm_results
