#pragma once
#include <windows.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include "ecm_stage2_fingerprint.h"

namespace stage2_queue {
namespace fs=std::filesystem;
inline std::string token() {
    std::random_device source;
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for(int i=0;i<4;++i)out << std::setw(8) << source();
    return out.str();
}
struct State {
    std::string identity, run, pending, pending_results;
    uint64_t done=0,total=0;
    unsigned version=2;
    bool load(const fs::path &path) {
        if(!fs::exists(path))return false;
        std::ifstream in(path,std::ios::binary);
        unsigned schema=0;
        if(!(in >> schema >> done >> total >> std::quoted(run) >> std::quoted(pending) >> std::quoted(identity)) ||
           (schema!=1 && schema!=2) || done>total || (done==total && !pending.empty()) || run.empty())
            throw std::runtime_error("invalid Stage2 progress file; inspect: "+path.string());
        version=schema;pending_results.clear();
        if(schema==2 && !(in>>std::quoted(pending_results)))throw std::runtime_error("missing Stage2 receipt path");
        if(schema==2 && (identity.size()!=64 || identity.find_first_not_of("0123456789abcdef")!=identity.npos))
            throw std::runtime_error("invalid Stage2 task ID");
        in >> std::ws;
        if(!in.eof())throw std::runtime_error("trailing data in Stage2 progress file: "+path.string());
        return true;
    }
    void save(const fs::path &path) const {
        if(!path.parent_path().empty())fs::create_directories(path.parent_path());
        const fs::path temp(path.string()+".tmp."+std::to_string(GetCurrentProcessId()));
        std::ostringstream out;
        out << "2 " << done << ' ' << total << ' ' << std::quoted(run) << ' '
            << std::quoted(pending) << ' ' << std::quoted(identity) << ' ' << std::quoted(pending_results) << '\n';
        const auto bytes=out.str();
        HANDLE file=CreateFileW(temp.c_str(),GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot create Stage2 progress file");
        DWORD written=0;
        const bool ok=bytes.size()<=MAXDWORD && WriteFile(file,bytes.data(),static_cast<DWORD>(bytes.size()),&written,nullptr)
                      && written==bytes.size() && FlushFileBuffers(file);
        CloseHandle(file);
        if(!ok || !MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot publish Stage2 progress; previous state retained: "+path.string());
    }
};
struct Task {
    std::string aid,n,save_sha256;
    uint64_t skip=0,total=0,b2=0;
    bool auto_b2=false;
    std::string id()const {
        std::ostringstream out;
        out<<"stage2-curves-v2 "<<std::quoted(aid)<<' '<<std::quoted(n)<<' '<<save_sha256<<' '
           <<skip<<' '<<total<<' '<<b2<<' '<<(auto_b2 && !b2);
        return ecm_stage2::sha256_text(out.str());
    }
    fs::path path(const fs::path &directory,size_t bits)const {
        return directory/(".ecm_ckpt_stage2_"+std::to_string(bits)+'_'+id()+".dat");
    }
};
struct LegacyIdentity {
    std::string task,queue,queue_sha256,save,save_sha256,results;
    uint64_t skip=0,requested=0,b2=0,d=0;
    bool auto_b2=false;
    explicit LegacyIdentity(const std::string &text) {
        std::istringstream in(text);
        if(!(in>>std::quoted(task)>>std::quoted(queue)>>queue_sha256>>std::quoted(save)>>save_sha256
               >>std::quoted(results)>>skip>>requested>>b2>>d>>auto_b2))
            throw std::runtime_error("cannot parse legacy Stage2 task identity");
    }
};
// Task parsing/N reconstruction is supplied by the production queue parser.
// Ordering, queue path and unrelated settings never participate in this match.
template<class MatchTask> bool import_legacy(const fs::path &path,const Task &task,MatchTask match,State &out) {
    State old;if(!old.load(path))return false;
    if(old.version==2) {
        if(old.identity!=task.id())return false;
    } else {
        const LegacyIdentity identity(old.identity);
        if(old.total!=task.total || identity.save_sha256!=task.save_sha256 || identity.skip!=task.skip ||
           identity.b2!=task.b2 || (identity.auto_b2 && !identity.b2)!=(task.auto_b2 && !task.b2) ||
           !match(identity.task))return false;
        old.pending_results=old.pending.empty()?std::string{}:identity.results;
    }
    old.identity=task.id();old.version=2;out=std::move(old);return true;
}
inline bool result_written(const fs::path &path,const std::string &receipt) {
    if(receipt.empty() || !fs::exists(path))return false;
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot inspect Stage2 result receipts");
    const auto prefix="{\"queue_receipt\":\""+receipt+"\",";
    std::string line;
    while(std::getline(in,line)) {
        // An unterminated last line may be a torn append. Never acknowledge it.
        if(in.eof())break;
        if(!line.empty() && line.back()=='\r')line.pop_back();
        if(line.compare(0,prefix.size(),prefix)==0 && line.back()=='}')return true;
    }
    if(in.bad())throw std::runtime_error("cannot read Stage2 result receipts");
    return false;
}
inline bool finished_written(const fs::path &path,const std::string &run,const std::string &task) {
    if(!fs::exists(path))return false;
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot inspect Stage2 finished receipts");
    const std::string marker="# stage2_task_id="+run;
    std::string previous,line;
    while(std::getline(in,line)) {
        if(in.eof())break;
        if(!line.empty() && line.back()=='\r')line.pop_back();
        if(previous==marker && line==task)return true;
        previous=line;
    }
    if(in.bad())throw std::runtime_error("cannot read Stage2 finished receipts");
    return false;
}
} // namespace stage2_queue
