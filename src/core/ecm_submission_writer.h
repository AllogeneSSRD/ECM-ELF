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
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace ecm_results {
namespace fs=std::filesystem;
inline void atomic_text(const fs::path &path,const std::string &text) {
    const auto temp=fs::path(path.string()+".new");
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot prepare submission output");
    out<<text;out.close();
    if(!out)throw std::runtime_error("cannot finish submission output write");
    if(!ecm_replace_file(temp,path))
        throw std::runtime_error("cannot publish submission output; Windows error="+std::to_string(GetLastError()));
}
struct SubmissionLock {
    HANDLE value=INVALID_HANDLE_VALUE;
    explicit SubmissionLock(const fs::path &path) {
        if(!path.parent_path().empty())fs::create_directories(path.parent_path());
        const auto lock=fs::path(path.string()+".ecm-ledger.lock");
        value=CreateFileW(lock.c_str(),GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
        if(value==INVALID_HANDLE_VALUE)throw std::runtime_error("another writer owns the submission output");
    }
    ~SubmissionLock(){if(value!=INVALID_HANDLE_VALUE)CloseHandle(value);}
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
inline void render_submission(const fs::path &path,const fs::path &ledger) {
    std::map<std::string,std::string> batches;
    std::map<std::string,std::string> factors;
    for(const auto &file:fs::directory_iterator(ledger)) {
        if(file.path().extension()!=".curve" && file.path().extension()!=".batch")continue;
        const auto entry=read_submission_entry(file.path());
        if(file.path().extension()==".batch")batches.emplace(entry.key,entry.json);
        else if(entry.group=="F")factors.emplace(entry.key,entry.json);
    }
    std::string output;
    for(const auto &pair:batches)output+=pair.second+'\n';
    for(const auto &pair:factors)output+=pair.second+'\n';
    atomic_text(path,output);
}
inline void seal_submissions(const fs::path &path) {
    const auto ledger=submission_ledger(path);if(!fs::exists(ledger/"format"))return;
    SubmissionLock lock(path);
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
        const auto deadline=GetTickCount64()+2000;
        for(;;) {
            json.replace(ts+time.size(),19,utc_timestamp());
            if(!lines.count(json))break;
            if(GetTickCount64()>deadline)throw std::runtime_error("system clock did not advance for NF report");
            Sleep(20);
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
    SubmissionLock lock(path);render_submission(path,ledger);
}
inline void publish_submission(const fs::path &path,const Submission &r,const mpz_t n) {
    if(r.curves!=1)throw std::runtime_error("submission ledger accepts individual completed curves");
    auto line=submission_json(r,n);line.pop_back();
    SubmissionLock lock(path);const auto ledger=submission_ledger(path);
    if(!fs::exists(ledger/"format")) {
        // Never replace an unrelated Prime95/manual result file.
        if(fs::exists(path))throw std::runtime_error("submission file has no owned ledger; select a fresh output path");
        fs::create_directories(ledger);atomic_text(ledger/"format","ECM_SUBMISSION_LEDGER_V1\n");
    }
    auto key=ecm_edwards::hex(n)+"|"+std::to_string(r.sigma)+"|"+std::to_string(r.b1)+"|"+
        std::to_string(r.b2)+"|"+std::to_string(r.stage)+"|"+(r.edwards?"ed":"mont0");
    for(const auto &factor:r.factors)key+='|'+factor;
    auto group=r;group.when="1970-01-01 00:00:00";
    const auto group_key=r.factors.empty()?submission_json(group,n):"F";
    const auto group_line=r.factors.empty()?group_key.substr(0,group_key.size()-1):group_key;
    const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
    if(fs::exists(entry)) {
        if(read_submission_entry(entry).key!=key)throw std::runtime_error("submission ledger identity collision");
    } else {
        const auto body="ECM_SUBMISSION_V1\n"+key+'\n'+group_line+'\n'+line+'\n';
        atomic_text(entry,body+std::to_string(ecm_edwards::hash(body))+'\n');
    }
    // Factors publish immediately. NF receipts are sealed at a normal task end;
    // immutable batches prevent cumulative counts being credited twice.
    render_submission(path,ledger);
}
} // namespace ecm_results
