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
    std::string identity, run, pending;
    uint64_t done=0,total=0;
    bool load(const fs::path &path) {
        if(!fs::exists(path))return false;
        std::ifstream in(path,std::ios::binary);
        unsigned schema=0;
        if(!(in >> schema >> done >> total >> std::quoted(run) >> std::quoted(pending) >> std::quoted(identity)) ||
           schema!=1 || done>total || (done==total && !pending.empty()) || run.empty())
            throw std::runtime_error("invalid Stage2 progress file; inspect: "+path.string());
        in >> std::ws;
        if(!in.eof())throw std::runtime_error("trailing data in Stage2 progress file: "+path.string());
        return true;
    }
    void save(const fs::path &path) const {
        if(!path.parent_path().empty())fs::create_directories(path.parent_path());
        const fs::path temp(path.string()+".tmp."+std::to_string(GetCurrentProcessId()));
        std::ostringstream out;
        out << "1 " << done << ' ' << total << ' ' << std::quoted(run) << ' '
            << std::quoted(pending) << ' ' << std::quoted(identity) << '\n';
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
