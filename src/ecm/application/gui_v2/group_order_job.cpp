#include "group_order_job.h"
#include "group_order_gp.h"
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
static std::string executable_in_path(const fs::path&name){
    const auto size=GetEnvironmentVariableW(L"PATH",nullptr,0);if(!size)return {};std::wstring paths(size,L'\0');const auto used=GetEnvironmentVariableW(L"PATH",paths.data(),size);if(!used||used>=size)return {};paths.resize(used);
    size_t begin=0;while(begin<=paths.size()){
        const auto end=paths.find(L';',begin);auto dir=paths.substr(begin,end==paths.npos?paths.npos:end-begin);
        if(dir.size()>=2&&dir.front()==L'"'&&dir.back()==L'"')dir=dir.substr(1,dir.size()-2);
        if(!dir.empty()){std::error_code e;auto file=fs::path(dir)/name;if(file.is_absolute()&&fs::is_regular_file(file,e))return file.lexically_normal().u8string();}
        if(end==paths.npos)break;begin=end+1;
    }return {};
}
std::string find_order_gp(){return executable_in_path(L"gp.exe");}
bool valid_order_input(const OrderInput&i,std::string&error){
    error.clear();if(!digits(i.factor)||i.factor.size()>1234){error="Factor must be a decimal prime (maximum 4096 bits)";return false;}
    if(!digits(i.sigma)||i.sigma.size()>20){error="Sigma must be a uint64 decimal integer";return false;}
    try{size_t n=0;const auto sigma=std::stoull(i.sigma,&n);if(n!=i.sigma.size()||!sigma||(i.profile==0&&sigma<6))throw std::runtime_error("sigma");}
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
void GroupOrderJob::stop(){if(process_){TerminateJobObject(job_,1);WaitForSingleObject(process_,1000);}cleanup();}
void GroupOrderJob::cancel(){if(process_)error="Analysis cancelled";stop();}
bool GroupOrderJob::start(const OrderInput&i,const std::string&helper){
    if(busy())return false;error.clear();result={};if(!valid_order_input(i,error))return false;
    snapshot=i;helper_=helper;fallback_.clear();signature=order_signature(i);started_=GetTickCount64();
    if(snapshot.backend!=2){
        if(snapshot.gp.empty())snapshot.gp=find_order_gp();
        else{auto name=fs::u8path(snapshot.gp);if(!name.has_parent_path()){if(!name.has_extension())name+=L".exe";const auto found=executable_in_path(name);if(!found.empty())snapshot.gp=found;}}
        if(!snapshot.gp.empty())try{launch(true);return true;}catch(const std::exception&){stop();return gp_failed("gp_start_failed");}
        return gp_failed("gp_not_found");
    }
    try{launch(false);return true;}catch(const std::exception&e){error=e.what();stop();return false;}
}
void GroupOrderJob::launch(bool gp){
    gp_phase_=gp;
    try{
        wchar_t directory[MAX_PATH],name[MAX_PATH];if(!GetTempPathW(MAX_PATH,directory)||!GetTempFileNameW(directory,L"ego",0,name))throw std::runtime_error("Cannot create analysis request");request_=name;
        if(!GetTempFileNameW(directory,L"ego",0,name))throw std::runtime_error("Cannot create analysis output");output_=name;
        go_json::Value request;request.type='o';request.set("factor",go_json::Value::string(snapshot.factor));request.set("sigma",go_json::Value::string(snapshot.sigma));request.set("gp",go_json::Value::string(""));request.set("profile",go_json::Value::number(std::to_string(snapshot.profile)));request.set("backend",go_json::Value::number("2"));
        std::ofstream out(request_,std::ios::binary);out<<(gp?order_gp::script(snapshot):request.dump());out.close();if(!out)throw std::runtime_error("Cannot write analysis request");
        job_=CreateJobObjectW(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if(!job_||!SetInformationJobObject(job_,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))throw std::runtime_error("Cannot isolate analysis process");
        const auto exe=fs::u8path(gp?snapshot.gp:helper_).wstring();auto cmd=quote(exe)+(gp?L" -q -f "+quote(request_):L" "+quote(request_)+L" "+quote(output_));STARTUPINFOW si{sizeof(si)};PROCESS_INFORMATION pi{};
        HANDLE capture=INVALID_HANDLE_VALUE,input=INVALID_HANDLE_VALUE;
        if(gp){SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};capture=CreateFileW(output_.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,nullptr);
            if(capture!=INVALID_HANDLE_VALUE)SetHandleInformation(capture,HANDLE_FLAG_INHERIT,HANDLE_FLAG_INHERIT);
            input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(capture==INVALID_HANDLE_VALUE||input==INVALID_HANDLE_VALUE){if(capture!=INVALID_HANDLE_VALUE)CloseHandle(capture);if(input!=INVALID_HANDLE_VALUE)CloseHandle(input);throw std::runtime_error("Cannot capture GP output");}
            si.dwFlags=STARTF_USESTDHANDLES;si.hStdInput=input;si.hStdOutput=si.hStdError=capture;}
        const bool launched=CreateProcessW(exe.c_str(),cmd.data(),nullptr,nullptr,gp?TRUE:FALSE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&si,&pi)!=FALSE;const auto code=GetLastError();
        if(capture!=INVALID_HANDLE_VALUE)CloseHandle(capture);if(input!=INVALID_HANDLE_VALUE)CloseHandle(input);
        if(!launched)throw std::runtime_error((gp?"Cannot start PARI/GP: ":"Local analysis helper missing or cannot start: ")+fs::u8path(gp?snapshot.gp:helper_).u8string()+" (Windows "+std::to_string(code)+")");
        process_=pi.hProcess;
        if(!AssignProcessToJobObject(job_,process_)){TerminateProcess(process_,1);CloseHandle(pi.hThread);throw std::runtime_error("Cannot contain analysis process");}
        if(ResumeThread(pi.hThread)==DWORD(-1)){CloseHandle(pi.hThread);throw std::runtime_error("Cannot resume analysis process");}CloseHandle(pi.hThread);
    }catch(...){stop();throw;}
}
bool GroupOrderJob::gp_failed(const std::string&reason){
    stop();fallback_=reason;
    if(snapshot.backend==1){result=order_gp::failure("gp",reason);return true;}
    try{launch(false);return true;}catch(const std::exception&e){error=reason+"; "+e.what();stop();return false;}
}
bool GroupOrderJob::poll(){
    if(!process_)return false;
    try{
        if(fs::file_size(output_)>2*1024*1024)throw std::runtime_error(gp_phase_?"gp_output_limit":"Analysis output exceeds limit");
        if(GetTickCount64()-started_>45000)throw std::runtime_error("Analysis exceeded 45 second deadline");
        const auto wait=WaitForSingleObject(process_,0);if(wait==WAIT_FAILED)throw std::runtime_error("Cannot wait for analysis process");
        if(wait==WAIT_TIMEOUT){if(gp_phase_&&GetTickCount64()-started_>=30000)throw std::runtime_error("gp_timeout");return false;}
        DWORD code=0;GetExitCodeProcess(process_,&code);
        if(gp_phase_){if(code)throw std::runtime_error("gp_exit_failed");result=order_gp::parse(read(output_),snapshot);cleanup();return true;}
        auto value=go_json::parse(read(output_));
        if(value.get("protocol").text()!="1")throw std::runtime_error("Invalid analysis helper protocol");
        result=std::move(value);if(!fallback_.empty())result.set("fallback",go_json::Value::string(fallback_));if(code||result.has("error"))error=result.get("error").text("Analysis helper failed");cleanup();return true;
    }catch(const std::invalid_argument&e){result=order_gp::failure(gp_phase_?"gp":"native",e.what(),"invalid_input");stop();return true;}
    catch(const std::exception&e){const std::string message=e.what();if(gp_phase_){gp_failed(message);return !busy();}stop();error=message;return true;}
}
}
