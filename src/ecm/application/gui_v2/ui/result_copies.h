#pragma once
#include "io/files.h"
#include "io/json.h"
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include <windows.h>

namespace ecmgui_v2::ui {
// Fingerprint the source and raw JSON, independent of line position in the file.
inline std::string result_copy_key(const std::string& source,const std::string& raw) {
    return std::to_string(ecm_io::fingerprint(source+'\n'+raw));
}

class ResultCopies {
public:
    bool load(const std::filesystem::path& sidecar) {
        auto path=sidecar;
        if(!path.empty())path+=L".result-copies.json";
        if(path==path_)return false;
        path_=std::move(path);history_={};error.clear();
        try{history_=read();}catch(const std::exception& e){error=e.what();}
        return true;
    }

    bool contains(const std::string& key)const{return history_.keys.count(key)!=0;}
    const std::string& last_copy()const{return history_.time;}
    std::string error;

    bool remember(const std::vector<std::string>& keys) {
        SYSTEMTIME utc{};GetSystemTime(&utc);
        char timestamp[32];
        std::snprintf(timestamp,sizeof(timestamp),"%04u-%02u-%02u %02u:%02u:%02u UTC",
            unsigned(utc.wYear),unsigned(utc.wMonth),unsigned(utc.wDay),
            unsigned(utc.wHour),unsigned(utc.wMinute),unsigned(utc.wSecond));
        history_.time=timestamp;history_.keys.insert(keys.begin(),keys.end());
        if(path_.empty())return true;
        try{
            if(!path_.parent_path().empty())std::filesystem::create_directories(path_.parent_path());
            ecm_io::FileLock lock(path_.u8string(),ecm_io::FileLockDomain::Publication,200);
            auto on_disk=read(); // Keep receipts from other GUI instances sharing this sidecar.
            history_.keys.insert(on_disk.keys.begin(),on_disk.keys.end());
            using Value=ecm_results::json::Value;
            Value json;json.type='o';json.set("last_copy",Value::string(history_.time));
            Value entries;entries.type='a';
            for(const auto& key:history_.keys)entries.items.push_back(Value::string(key));
            json.set("copied",std::move(entries));
            ecm_io::write_atomic(path_,json.dump()+'\n');
            error.clear();return true;
        }catch(const std::exception& e){error=e.what();return false;}
    }
private:
    struct History {std::set<std::string> keys;std::string time;};
    History read() const {
        if(path_.empty()||!std::filesystem::exists(path_))return {};
        const auto json=ecm_results::json::parse(ecm_io::read_file(path_));
        const auto& time=json.get("last_copy");const auto& copied=json.get("copied");
        if(json.type!='o'||time.type!='s'||copied.type!='a')throw std::runtime_error("Invalid result copy history");
        History result;result.time=time.text();
        for(const auto& key:copied.items){
            const auto text=key.text();
            if(key.type!='s'||text.empty()||text.size()>20||text.find_first_not_of("0123456789")!=text.npos)
                throw std::runtime_error("Invalid result copy identity");
            result.keys.insert(text);
        }
        return result;
    }
    std::filesystem::path path_;
    History history_;
};
}
