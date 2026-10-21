#include "ecm_stage2_queue_state.h"
#include "ecm/platform/fingerprint.h"
#include "../results/ecm_result_json.h"
#include <windows.h>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
namespace stage2_queue {
bool State::load(const fs::path &path) {
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

void State::save(const fs::path &path) const {
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

std::string Task::id()const {
        std::ostringstream out;
        out<<"stage2-curves-v2 "<<std::quoted(aid)<<' '<<std::quoted(n)<<' '<<save_sha256<<' '
           <<skip<<' '<<total<<' '<<b2<<' '<<(auto_b2 && !b2);
        return ecm_stage2::sha256_text(out.str());
    }

fs::path Task::path(const fs::path &directory,size_t bits)const {
        return directory/(".ecm_ckpt_stage2_"+std::to_string(bits)+'_'+id()+".dat");
    }

std::string token() {
    std::random_device source;
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for(int i=0;i<4;++i)out << std::setw(8) << source();
    return out.str();
}

bool result_written(const fs::path &path,const std::string &receipt) {
    if(receipt.empty() || !fs::exists(path))return false;
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot inspect Stage2 result receipts");

    std::string line;
    while(std::getline(in,line)) {
        // An unterminated last line may be a torn append. Never acknowledge it.
        if(in.eof())break;
        if(!line.empty() && line.back()=='\r')line.pop_back();
        try {const auto row=ecm_results::json::parse(line);if(row.type=='o' && row.get("queue_receipt").text()==receipt)return true;}
        catch(const std::runtime_error &){/* A torn/invalid append is never a durable receipt. */}
    }
    if(in.bad())throw std::runtime_error("cannot read Stage2 result receipts");
    return false;
}

bool finished_written(const fs::path &path,const std::string &run,const std::string &task) {
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

LegacyIdentity::LegacyIdentity(const std::string &text) {
        std::istringstream in(text);
        if(!(in>>std::quoted(task)>>std::quoted(queue)>>queue_sha256>>std::quoted(save)>>save_sha256
               >>std::quoted(results)>>skip>>requested>>b2>>d>>auto_b2))
            throw std::runtime_error("cannot parse legacy Stage2 task identity");
    }
}
