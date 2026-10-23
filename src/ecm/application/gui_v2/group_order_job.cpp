#include "group_order_job.h"
#include <filesystem>
#include <fstream>
#include <set>
#include <algorithm>
namespace ecmgui_v2 {
namespace fs=std::filesystem;
namespace {
bool digits(const std::string&s){return !s.empty()&&s.find_first_not_of("0123456789")==s.npos;}
std::wstring quote(const std::wstring&s){std::wstring r=L"\"";size_t slash=0;for(auto c:s){if(c==L'\\'){++slash;continue;}r.append(slash*(c==L'"'?2:1),L'\\');slash=0;if(c==L'"')r+=L'\\';r+=c;}r.append(slash*2,L'\\');return r+L'"';}
std::string read(const fs::path&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot read analysis file");return {std::istreambuf_iterator<char>(in),{}};}
}
std::string order_signature(const OrderInput&i){return i.factor+'\n'+i.sigma+'\n'+std::to_string(i.profile)+'\n'+i.gp+'\n'+std::to_string(i.backend);}
bool valid_order_input(const OrderInput&i,std::string&error){
    error.clear();if(!digits(i.factor)||i.factor.size()>1234){error="Factor must be a decimal prime (maximum 4096 bits)";return false;}
    if(!digits(i.sigma)||i.sigma.size()>20){error="Sigma must be a uint64 decimal integer";return false;}
    try{size_t n=0;const auto sigma=std::stoull(i.sigma,&n);if(n!=i.sigma.size()||(i.profile==0&&sigma<6))throw std::runtime_error("sigma");}
    catch(...){error="Sigma is outside the selected construction's range";return false;}
    if(i.profile!=0&&i.profile!=1&&i.profile!=3&&i.profile!=4&&i.profile!=5){error="Unsupported curve construction";return false;}
    if(i.backend<0||i.backend>2){error="Unknown backend";return false;}return true;
}
std::vector<OrderRecord> read_order_records(const std::string&path,std::string&error){
    std::vector<OrderRecord> rows;error.clear();size_t skipped=0,line=0;
    try{
        if(fs::file_size(fs::u8path(path))>32*1024*1024)throw std::runtime_error("Result file exceeds 32 MiB; select a smaller file");
        std::ifstream input(fs::u8path(path),std::ios::binary);if(!input)throw std::runtime_error("Cannot open result file");std::string raw;
        while(std::getline(input,raw)){
            ++line;if(line==1&&raw.compare(0,3,"\xef\xbb\xbf")==0)raw.erase(0,3);if(raw.empty())continue;
            try{
                auto v=go_json::parse(raw);if(v.get("status").text()!="F"||v.get("worktype").text()!="ECM")continue;
                const bool ed=v.has("Edwards"),m3=v.has("GMP-ECM-param3");
                if((ed&&m3)||(ed&&v.has("param")&&v.get("param").text()!="-1")||(m3&&v.has("param")&&v.get("param").text()!="3"))throw std::runtime_error("Ambiguous curve identity");
                const auto sigma=ed?v.get("Edwards").get("sigma").text():m3?v.get("GMP-ECM-param3").get("sigma").text():v.get("sigma").text();
                const auto param=v.has("param")?v.get("param").text():std::string("0");int profile=ed?1:m3?4:param=="0"?0:param=="1"?3:param=="2"?5:param=="3"?4:-1;
                const auto&factors=v.get("factors");if(factors.type!='a')throw std::runtime_error("Missing factors");
                for(const auto&factor:factors.items){OrderRecord record;record.input={factor.text(),sigma,"",profile,0};std::string invalid;
                    if(!valid_order_input(record.input,invalid)){++skipped;continue;}
                    record.source=path;record.line=line;record.label=(v.has("exponent")?"M"+v.get("exponent").text():"ECM")+" | "+record.input.factor+" | sigma="+sigma+" | #"+std::to_string(line);rows.push_back(std::move(record));
                    if(rows.size()>20000)throw std::length_error("Too many result factors");
                }
            }catch(const std::length_error&){throw;}catch(...){++skipped;}
        }
        if(input.bad())throw std::runtime_error("Cannot finish reading result file");
        if(skipped)error=std::to_string(skipped)+" malformed or unsupported entries skipped";
    }catch(const std::exception&e){error=e.what();rows.clear();}return rows;
}
std::vector<std::string> discover_order_files(const std::string&directory,const std::vector<std::string>&configured){
    std::set<std::string> seen;std::vector<std::string> paths;
    const auto add=[&](fs::path path){std::error_code e;if(!fs::is_regular_file(path,e))return;auto absolute=fs::weakly_canonical(path,e);if(e)return;auto name=absolute.u8string(),key=name;std::transform(key.begin(),key.end(),key.begin(),[](unsigned char c){return char(std::tolower(c));});if(seen.insert(key).second)paths.push_back(name);};
    for(const auto&path:configured)add(fs::u8path(path));std::error_code e;
    for(fs::directory_iterator it(fs::u8path(directory),e),end;!e&&it!=end;it.increment(e)){
        auto name=it->path().filename().u8string();std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(name.size()>8&&name.find("result")!=std::string::npos&&name.size()>=4&&name.substr(name.size()-4)==".txt")add(it->path());
    }return paths;
}
void GroupOrderJob::cleanup(){if(job_)CloseHandle(job_);if(process_)CloseHandle(process_);job_=process_=nullptr;std::error_code e;if(!request_.empty())fs::remove(request_,e);if(!output_.empty())fs::remove(output_,e);request_.clear();output_.clear();}
void GroupOrderJob::cancel(){if(process_){TerminateJobObject(job_,1);WaitForSingleObject(process_,1000);error="Analysis cancelled";}cleanup();}
bool GroupOrderJob::start(const OrderInput&i,const std::string&helper){
    if(busy())return false;error.clear();result={};if(!valid_order_input(i,error))return false;
    try{
        wchar_t directory[MAX_PATH],name[MAX_PATH];if(!GetTempPathW(MAX_PATH,directory)||!GetTempFileNameW(directory,L"ego",0,name))throw std::runtime_error("Cannot create analysis request");request_=name;
        if(!GetTempFileNameW(directory,L"ego",0,name))throw std::runtime_error("Cannot create analysis output");output_=name;
        go_json::Value request;request.type='o';request.set("factor",go_json::Value::string(i.factor));request.set("sigma",go_json::Value::string(i.sigma));request.set("gp",go_json::Value::string(i.gp));request.set("profile",go_json::Value::number(std::to_string(i.profile)));request.set("backend",go_json::Value::number(std::to_string(i.backend)));
        std::ofstream out(request_,std::ios::binary);out<<request.dump();out.close();if(!out)throw std::runtime_error("Cannot write analysis request");
        job_=CreateJobObjectW(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if(!job_||!SetInformationJobObject(job_,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))throw std::runtime_error("Cannot isolate analysis process");
        const auto exe=fs::u8path(helper).wstring();auto cmd=quote(exe)+L" "+quote(request_)+L" "+quote(output_);STARTUPINFOW si{sizeof(si)};PROCESS_INFORMATION pi{};
        if(!CreateProcessW(exe.c_str(),cmd.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&si,&pi))throw std::runtime_error("Analysis helper is missing or cannot start");
        process_=pi.hProcess;
        if(!AssignProcessToJobObject(job_,process_)){TerminateProcess(process_,1);CloseHandle(pi.hThread);throw std::runtime_error("Cannot contain analysis process");}
        if(ResumeThread(pi.hThread)==DWORD(-1)){CloseHandle(pi.hThread);throw std::runtime_error("Cannot resume analysis process");}CloseHandle(pi.hThread);
        snapshot=i;signature=order_signature(i);started_=GetTickCount64();return true;
    }catch(const std::exception&e){error=e.what();if(process_)TerminateProcess(process_,1);cleanup();return false;}
}
bool GroupOrderJob::poll(){
    if(!process_)return false;
    try{
        if(fs::file_size(output_)>2*1024*1024)throw std::runtime_error("Analysis output exceeds limit");
        if(GetTickCount64()-started_>45000)throw std::runtime_error("Analysis exceeded 45 second deadline");
        if(WaitForSingleObject(process_,0)==WAIT_TIMEOUT)return false;
        DWORD code=0;GetExitCodeProcess(process_,&code);auto value=go_json::parse(read(output_));
        if(value.get("protocol").text()!="1")throw std::runtime_error("Invalid analysis helper protocol");
        result=std::move(value);if(code||result.has("error"))error=result.get("error").text("Analysis helper failed");cleanup();return true;
    }catch(const std::exception&e){const std::string message=e.what();cancel();error=message;return true;}
}
}
