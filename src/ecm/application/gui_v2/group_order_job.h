#pragma once
#include "../../storage/results/ecm_result_json.h"
#include <windows.h>
#include <string>
#include <vector>
namespace ecmgui_v2 {
namespace go_json=ecm_results::json;
struct OrderInput {std::string factor,sigma="6",gp;int profile=0,backend=0;};
bool valid_order_input(const OrderInput&,std::string& error);
struct OrderRecord {OrderInput input;std::string label,source;size_t line=0;};
std::vector<OrderRecord> read_order_records(const std::string& path,std::string& error);
std::vector<std::string> discover_order_files(const std::string& directory,const std::vector<std::string>& configured={});
class GroupOrderJob {
public:
    GroupOrderJob()=default;
    GroupOrderJob(const GroupOrderJob&)=delete;
    GroupOrderJob& operator=(const GroupOrderJob&)=delete;
    ~GroupOrderJob(){cancel();}
    bool start(const OrderInput&,const std::string& helper);
    bool poll();
    void cancel();
    bool busy()const{return process_!=nullptr;}
    DWORD pid()const{return process_?GetProcessId(process_):0;}
    std::string error,signature;
    go_json::Value result;
    OrderInput snapshot;
private:
    HANDLE process_=nullptr,job_=nullptr;
    std::wstring request_,output_;
    unsigned long long started_=0;
    void cleanup();
};
std::string order_signature(const OrderInput&);
}
