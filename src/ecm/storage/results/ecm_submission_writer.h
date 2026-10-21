#pragma once
#include "ecm_result_submission.h"
#include "ecm_result_paths.h"
#include "../save/ecm_edwards_io.h"
#include "../../platform/ecm_file_publication.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
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
// The caller holds the output lock, shared with old binaries. Rename the whole
// private directory; never merge two histories or replace an existing target.
inline fs::path migrate_result_ledger(const fs::path &path) {
    const auto old=legacy_submission_ledger(path),next=submission_ledger(path);
    if(fs::exists(old)) {
        if(fs::exists(next))throw std::runtime_error("both old and data result ledgers exist; reconcile privately before continuing");
        if(!fs::is_regular_file(old/"format"))throw std::runtime_error("legacy result ledger has no owner marker");
        fs::create_directories(next.parent_path());fs::rename(old,next);
    }
    return next;
}
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
inline std::string read_submission_bytes(const fs::path &path) {
    if(!fs::exists(path))return {};
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot read submission output");
    std::string text{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
    if(in.bad())throw std::runtime_error("cannot finish reading submission output");
    return text;
}
// The caller holds SubmissionLock. The durable publication history is written
// before appending: a retry can finish a partial append without reordering or
// duplicating previously published bytes. Existing outputs seed the history.
inline void append_submissions(const fs::path &path,const fs::path &ledger) {
    struct Row {std::string id,json;fs::file_time_type time;};
    std::map<std::string,Row> rows;
    for(const auto &file:fs::directory_iterator(ledger)) {
        if(file.path().extension()!=".curve" && file.path().extension()!=".batch")continue;
        const auto entry=read_submission_entry(file.path());
        const bool batch=file.path().extension()==".batch";
        if(!batch && entry.group!="F")continue;
        const auto id=batch?"NF:"+entry.key:"F:"+stage1_factor_key(entry.key);
        const auto time=fs::last_write_time(file.path());
        const auto previous=rows.find(id);
        if(previous==rows.end() || time<previous->second.time)rows.insert_or_assign(id,Row{id,entry.json,time});
    }
    const auto actual=read_submission_bytes(path);
    const auto order_path=ledger/"publication.order";
    const bool has_order=fs::exists(order_path);
    std::string ordered=actual;
    if(has_order) {
        const auto record=read_submission_entry(order_path);
        const auto value=json::parse(record.json);
        if(record.key!="publication-order" || record.group!="ORDER" || value.type!='s')
            throw std::runtime_error("invalid submission publication history");
        ordered=value.text();
        if(actual.size()>ordered.size() || ordered.compare(0,actual.size(),actual)!=0)
            throw std::runtime_error("submission output differs from publication history");
    }
    std::set<std::string> published;
    std::istringstream existing(ordered);std::string line;
    while(std::getline(existing,line)) {
        if(!line.empty() && line.back()=='\r')line.pop_back();
        if(line.empty())continue;
        const auto value=json::parse(line);
        if(value.type!='o')throw std::runtime_error("invalid submission output row");
        published.insert(value.dump());
    }
    // Only unrecorded receipts need this recovery order. Normal publication
    // appends each newly sealed NF batch or immediate F under the same lock.
    std::vector<Row> pending;
    for(const auto &pair:rows)if(!published.count(json::parse(pair.second.json).dump()))pending.push_back(pair.second);
    std::sort(pending.begin(),pending.end(),[](const auto &a,const auto &b){
        if(a.time!=b.time)return a.time<b.time;return a.id<b.id;
    });
    for(const auto &row:pending) {
        if(!published.insert(json::parse(row.json).dump()).second)continue;
        if(!ordered.empty() && ordered.back()!='\n')ordered+='\n';
        ordered+=row.json+'\n';
    }
    if(!has_order || !pending.empty()) {
        const auto body="ECM_SUBMISSION_V1\npublication-order\nORDER\n"+json::quote(ordered)+'\n';
        atomic_text(order_path,body+std::to_string(ecm_edwards::hash(body))+'\n');
    }
    if(actual.size()!=ordered.size() || !fs::exists(path)) {
        std::ofstream out(path,std::ios::binary|std::ios::app);
        if(!out)throw std::runtime_error("cannot append submission output");
        out.write(ordered.data()+actual.size(),static_cast<std::streamsize>(ordered.size()-actual.size()));
        out.close();
        if(!out)throw std::runtime_error("cannot finish submission output append");
    }
}
inline uint64_t submission_order(const SubmissionEntry &entry) {
    const auto marker=entry.group.find(":O:");
    if(marker==std::string::npos)return 0;
    const auto start=marker+3,end=entry.group.find(':',start);
    if(end==std::string::npos || end==start ||
       entry.group.find_first_not_of("0123456789",start)!=end)return 0;
    return std::stoull(entry.group.substr(start,end-start));
}
inline void seal_submissions(const fs::path &path,int stage_filter=0,const std::string &task={},bool whole_task=false) {
    SubmissionLock lock(path);const auto ledger=migrate_result_ledger(path);if(!fs::exists(ledger/"format"))return;
    require_ledger_format(ledger,"ECM_SUBMISSION_LEDGER_V1");
    std::set<std::string> sealed,lines;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".batch") {
        const auto batch=read_submission_entry(file.path());lines.insert(batch.json);
        std::istringstream ids(batch.group);std::string id;
        while(std::getline(ids,id,','))if(!sealed.insert(id).second)throw std::runtime_error("NF curve occurs in two submission batches");
    }
    struct Pending {std::string id;SubmissionEntry entry;fs::file_time_type time;};
    std::map<std::string,std::vector<Pending>> pending;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve") {
        const auto id=file.path().stem().string();const auto entry=read_submission_entry(file.path());
        if(entry.group=="F" || sealed.count(id))continue;
        if(!task.empty() && entry.group.compare(0,task.size()+3,"T:"+task+":"))continue;
        const auto row=json::parse(entry.json);
        const bool stage1=row.get("stage").text()=="1" || row.get("b1").text()==row.get("b2").text();
        if(stage_filter && stage1!=(stage_filter==1))continue;
        std::string group=entry.group;
        if(whole_task && group.compare(0,2,"T:")==0) {
            const auto end=group.find(':',2);
            if(end!=std::string::npos)group.resize(end);
        }
        pending[group].push_back({id,entry,fs::last_write_time(file.path())});
    }
    for(auto &group:pending) {
        auto &entries=group.second;
        const auto latest=std::max_element(entries.begin(),entries.end(),[](const auto &a,const auto &b){
            const auto ao=submission_order(a.entry),bo=submission_order(b.entry);
            if(ao!=bo)return ao<bo;
            if(a.time!=b.time)return a.time<b.time;
            return a.id<b.id;
        });
        auto json=latest->entry.json;
        std::sort(entries.begin(),entries.end(),[](const auto &a,const auto &b){return a.id<b.id;});
        std::string ids;for(const auto &entry:entries){if(!ids.empty())ids+=',';ids+=entry.id;}
        {
            auto row=json::parse(json);row.set("curves",json::Value::number(std::to_string(entries.size())));
            // Only real UTC timestamps are used to distinguish equal batches.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            bool keep_completion_time=whole_task;
            for(;;) {
                if(!keep_completion_time)row.set("timestamp",json::Value::string(utc_timestamp()));
                keep_completion_time=false;
                json=with_result_checksum(row.dump());
                if(!lines.count(json))break;
                if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("system clock did not advance for NF report");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        const auto key=std::to_string(ecm_edwards::hash(ids));
        const auto body="ECM_SUBMISSION_V1\n"+key+'\n'+ids+'\n'+json+'\n';
        atomic_text(ledger/(key+".batch"),body+std::to_string(ecm_edwards::hash(body))+'\n');
        lines.insert(json);
        append_submissions(path,ledger);
    }
    append_submissions(path,ledger);
}
inline void refresh_submission_if_owned(const fs::path &path) {
    SubmissionLock lock(path);const auto ledger=migrate_result_ledger(path);
    if(!fs::exists(ledger/"format"))return;
    require_ledger_format(ledger,"ECM_SUBMISSION_LEDGER_V1");append_submissions(path,ledger);
}
// The caller holds SubmissionLock, including while establishing ownership.
inline fs::path prepare_submission_ledger(const fs::path &path) {
    const auto ledger=migrate_result_ledger(path);
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
    append_submissions(path,prepare_submission_ledger(path));
}
inline void publish_submission(const fs::path &path,const Submission &r,const mpz_t n,bool render_now=false) {
    if(r.curves!=1)throw std::runtime_error("submission ledger accepts individual completed curves");
    auto line=submission_json(r,n);line.pop_back();
    SubmissionLock lock(path);const auto ledger=prepare_submission_ledger(path);
    auto key=ecm_edwards::hex(n)+"|"+std::to_string(r.sigma)+"|"+std::to_string(r.b1)+"|"+
        std::to_string(r.b2)+"|"+std::to_string(r.stage)+"|"+(r.edwards?"ed":"mont"+std::to_string(r.param));
    for(const auto &factor:r.factors)key+='|'+factor;
    key=stage1_factor_key(key);
    auto group=r;group.when="1970-01-01 00:00:00";
    const auto group_key=r.factors.empty()?"T:"+r.publication_task+":"+
        (r.publication_order?"O:"+std::to_string(r.publication_order)+":":std::string{})+submission_json(group,n):"F";
    const auto group_line=r.factors.empty()?group_key.substr(0,group_key.size()-1):group_key;
    const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
    if(r.stage==1 && !fs::exists(entry)) {
        // Preserve older receipts and their first owner; their filenames used
        // the continuation B2. Export also collapses existing legacy duplicates.
        for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve") {
            const auto previous=read_submission_entry(file.path());
            if(previous.group=="F" && stage1_factor_key(previous.key)==key) {
                if(render_now || !r.factors.empty())append_submissions(path,ledger);return;
            }
        }
    }
    if(fs::exists(entry)) {
        if(read_submission_entry(entry).key!=key)throw std::runtime_error("submission ledger identity collision");
    } else {
        const auto body="ECM_SUBMISSION_V1\n"+key+'\n'+group_line+'\n'+line+'\n';
        atomic_text(entry,body+std::to_string(ecm_edwards::hash(body))+'\n');
    }
    // Factors publish immediately; NF waits for whole-task completion.
    // Immutable batches prevent cumulative counts being credited twice.
    if(render_now || !r.factors.empty())append_submissions(path,ledger);
}
// Internal receipts carry a monotone export order, independent of sigma and
// directory iteration. V1 receipts retain their original creation/write order.
inline void render_internal(const fs::path &path,const fs::path &ledger) {
    struct Row {SubmissionEntry entry;fs::file_time_type time;};std::vector<Row> rows;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve")
        rows.push_back({read_submission_entry(file.path()),fs::last_write_time(file.path())});
    std::sort(rows.begin(),rows.end(),[](const auto &a,const auto &b){
        const bool ai=a.entry.group.compare(0,2,"I:")==0,bi=b.entry.group.compare(0,2,"I:")==0;
        if(ai!=bi)return !ai;
        if(ai)return std::stoull(a.entry.group.substr(2))<std::stoull(b.entry.group.substr(2));
        if(a.time!=b.time)return a.time<b.time;return a.entry.key<b.entry.key;
    });
    std::string output;for(const auto &r:rows)output+=r.entry.json+'\n';atomic_text(path,output);
}
inline fs::path prepare_internal_ledger(const fs::path &path) {
    const auto ledger=migrate_result_ledger(path);
    if(!fs::exists(ledger/"format")) {
        if(fs::exists(path))throw std::runtime_error("internal result file has no owned ledger; select a fresh output path");
        fs::create_directories(ledger);atomic_text(ledger/"format","ECM_INTERNAL_LEDGER_V2\n");
    }
    std::ifstream in(ledger/"format",std::ios::binary);std::string tag;std::getline(in,tag);in.close();
    if(tag=="ECM_INTERNAL_LEDGER_V1") {require_ledger_format(ledger,tag);atomic_text(ledger/"format","ECM_INTERNAL_LEDGER_V2\n");}
    else require_ledger_format(ledger,"ECM_INTERNAL_LEDGER_V2");
    return ledger;
}
inline void publish_internal_batch(const fs::path &path,const std::vector<std::pair<std::string,std::string>> &records,bool render_now=true) {
    SubmissionLock lock(path);const auto ledger=prepare_internal_ledger(path);uint64_t next=0;
    for(const auto &file:fs::directory_iterator(ledger))if(file.path().extension()==".curve") {
        const auto entry=read_submission_entry(file.path());
        if(entry.group.compare(0,2,"I:")==0)next=(std::max)(next,uint64_t(std::stoull(entry.group.substr(2)))+uint64_t(1));else ++next;
    }
    for(const auto &record:records) {
        const auto &key=record.first;auto line=record.second;if(!line.empty()&&line.back()=='\n')line.pop_back();
        const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
        if(fs::exists(entry)) {if(read_submission_entry(entry).key!=key)throw std::runtime_error("internal result identity collision");}
        else {const auto body="ECM_SUBMISSION_V1\n"+key+"\nI:"+std::to_string(next++)+'\n'+line+'\n';
            atomic_text(entry,body+std::to_string(ecm_edwards::hash(body))+'\n');}
    }
    if(render_now)render_internal(path,ledger);
}
inline void publish_internal(const fs::path &path,const std::string &key,const std::string &json,bool render_now=true) {
    publish_internal_batch(path,{{key,json}},render_now);
}
inline void refresh_internal(const fs::path &path) {
    SubmissionLock lock(path);const auto ledger=prepare_internal_ledger(path);render_internal(path,ledger);
}
} // namespace ecm_results
