#include "order_analysis.h"
#include "pari_protocol.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ecm_diagnostics {
using namespace ecm_group_order;
namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t max_output=2*1024*1024;
double milliseconds(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct ProcessResult {std::string output, failure;};
struct Temporary {
    std::filesystem::path script;
    ~Temporary(){std::error_code error;if(!script.empty())std::filesystem::remove(script,error);}
};
#ifdef _WIN32
struct Handle {
    HANDLE value=INVALID_HANDLE_VALUE;
    ~Handle(){if(value && value!=INVALID_HANDLE_VALUE)CloseHandle(value);}
};
std::wstring process_quote(const std::wstring&s){
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:s){if(c==L'\\'){++slashes;continue;}out.append(slashes*(c==L'"'?2:1),L'\\');slashes=0;
        if(c==L'"')out+=L'\\';out+=c;}
    out.append(2*slashes,L'\\');return out+L'"';
}
#endif
std::filesystem::path host_path(const Settings& s){
    if(!s.pari_host.empty())return std::filesystem::u8path(s.pari_host);
#ifdef _WIN32
    wchar_t path[32768];const auto size=GetModuleFileNameW(nullptr,path,32768);
    if(!size||size==32768)throw std::runtime_error("Cannot locate diagnostic executable");
    return std::filesystem::path(path).parent_path()/L"ecm_diagnostics_host.exe";
#else
    return "/nonexistent/ecm_diagnostics_host"; // Win64 ABI is the only qualified DLL.
#endif
}
ProcessResult execute_process(const Settings&s,const std::string&body,bool pari=false){
    ProcessResult out;Temporary temporary;
#ifdef _WIN32
    wchar_t directory[MAX_PATH],script[MAX_PATH],log[MAX_PATH];
    if(!GetTempPathW(MAX_PATH,directory) || !GetTempFileNameW(directory,L"ecg",0,script)){
        out.failure="gp_temp_failed";return out;
    }
    temporary.script=script;
    if(!GetTempFileNameW(directory,L"eco",0,log)){out.failure="gp_temp_failed";return out;}
    SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};Handle output,input,process,thread,job;
    output.value=CreateFileW(log,GENERIC_WRITE|GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
        &sa,OPEN_EXISTING,FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
    if(output.value==INVALID_HANDLE_VALUE){DeleteFileW(log);out.failure="gp_capture_failed";return out;}
    input.value=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(input.value==INVALID_HANDLE_VALUE){out.failure="gp_capture_failed";return out;}
#else
    std::string pattern=(std::filesystem::temp_directory_path()/"ecm_gp_XXXXXX").string();
    std::vector<char> path(pattern.begin(),pattern.end());path.push_back(0);int scriptfd=mkstemp(path.data());
    if(scriptfd<0){out.failure="gp_temp_failed";return out;}close(scriptfd);temporary.script=path.data();
    std::vector<char> logpath(pattern.begin(),pattern.end());logpath.push_back(0);int output=mkstemp(logpath.data());
    if(output<0){out.failure="gp_capture_failed";return out;}unlink(logpath.data());
    struct Fd {int value;~Fd(){if(value>=0)close(value);}} captured{output};
#endif
    {std::ofstream file(temporary.script,std::ios::binary);file<<body;if(!file){out.failure="gp_temp_failed";return out;}}
    const auto start=Clock::now();
#ifdef _WIN32
    STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;
    startup.hStdOutput=startup.hStdError=output.value;startup.hStdInput=input.value;
    PROCESS_INFORMATION info{};
    auto command=process_quote(pari?host_path(s).wstring():std::filesystem::u8path(s.gp).wstring())+(pari?L" ":L" -q -f ");
    if(!pari&&s.gp_stack_bytes)command+=L"-s "+std::to_wstring(s.gp_stack_bytes)+L" ";
    command+=process_quote(temporary.script.wstring());
    if(!CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&startup,&info)){
        out.failure="gp_launch_failed";return out;
    }
    process.value=info.hProcess;thread.value=info.hThread;
    job.value=CreateJobObjectW(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(pari){limits.BasicLimitInformation.LimitFlags|=JOB_OBJECT_LIMIT_PROCESS_MEMORY;limits.ProcessMemoryLimit=512ull*1024*1024;}
    if(!job.value||!SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits))||!AssignProcessToJobObject(job.value,process.value)||ResumeThread(thread.value)==DWORD(-1)){
        TerminateProcess(process.value,2);WaitForSingleObject(process.value,1000);out.failure="gp_isolation_failed";return out;
    }
    for(;;){
        LARGE_INTEGER size{};
        if(!GetFileSizeEx(output.value,&size)){out.failure="gp_capture_failed";break;}
        if(size.QuadPart>max_output){out.failure="gp_output_limit";break;}
        DWORD wait=WaitForSingleObject(process.value,5);
        if(wait==WAIT_OBJECT_0)break;
        if(wait!=WAIT_TIMEOUT){out.failure="gp_wait_failed";break;}
        if(milliseconds(start)>=(pari?s.pari_timeout_ms:s.gp_timeout_ms)){out.failure="gp_timeout";break;}
    }
    if(!out.failure.empty()){TerminateJobObject(job.value,2);WaitForSingleObject(process.value,1000);}
    DWORD code=1;if(!GetExitCodeProcess(process.value,&code) || code){if(out.failure.empty())out.failure="gp_exit_failed";}
    LARGE_INTEGER zero{};SetFilePointerEx(output.value,zero,nullptr,FILE_BEGIN);
    char buffer[4096];DWORD read=0;
    while(out.output.size()<max_output && ReadFile(output.value,buffer,DWORD(std::min(sizeof(buffer),max_output-out.output.size())),&read,nullptr) && read)
        out.output.append(buffer,read);
    LARGE_INTEGER final_size{};if(GetFileSizeEx(output.value,&final_size) && final_size.QuadPart>max_output)out.failure="gp_output_limit";
#else
    pid_t child=fork();
    if(child<0){out.failure="gp_launch_failed";return out;}
    if(!child){
        int input=open("/dev/null",O_RDONLY);if(input>=0)dup2(input,STDIN_FILENO);
        dup2(output,STDOUT_FILENO);dup2(output,STDERR_FILENO);
        if(pari){auto host=host_path(s).string();execl(host.c_str(),host.c_str(),temporary.script.c_str(),static_cast<char*>(nullptr));_exit(127);}
        if(s.gp_stack_bytes){const auto size=std::to_string(s.gp_stack_bytes);execlp(s.gp.c_str(),s.gp.c_str(),"-q","-f","-s",size.c_str(),temporary.script.c_str(),static_cast<char*>(nullptr));}
        else execlp(s.gp.c_str(),s.gp.c_str(),"-q","-f",temporary.script.c_str(),static_cast<char*>(nullptr));_exit(127);
    }
    int status=0;
    for(;;){
        struct stat size{};if(fstat(output,&size) || size.st_size>off_t(max_output)){out.failure="gp_output_limit";break;}
        const pid_t done=waitpid(child,&status,WNOHANG);if(done==child)break;
        if(done<0 && errno!=EINTR){out.failure="gp_wait_failed";break;}
        if(milliseconds(start)>=(pari?s.pari_timeout_ms:s.gp_timeout_ms)){out.failure="gp_timeout";break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if(!out.failure.empty()){kill(child,SIGKILL);while(waitpid(child,&status,0)<0 && errno==EINTR){}}
    if((!WIFEXITED(status)||WEXITSTATUS(status))&&out.failure.empty())out.failure="gp_exit_failed";
    lseek(output,0,SEEK_SET);char buffer[4096];ssize_t size=0;
    while(out.output.size()<max_output && (size=read(output,buffer,std::min(sizeof(buffer),max_output-out.output.size())))>0)out.output.append(buffer,size_t(size));
    struct stat final_size{};if(!fstat(output,&final_size)&&final_size.st_size>off_t(max_output))out.failure="gp_output_limit";
#endif
    return out;
}
std::string model(const CurveRequest&r){
    if(r.family==Family::Edwards){
        const auto curve=construct_curve(r);std::ostringstream s;
        s<<"p="<<r.prime.str()<<";E=ellinit([Mod("<<curve.a.str()<<",p),Mod("<<curve.b.str()<<",p)]);P=";
        if(curve.point.infinity)s<<"[0];";else s<<"[Mod("<<curve.point.x.str()<<",p),Mod("<<curve.point.y.str()<<",p)];";
        return s.str();
    }
    if(r.family!=Family::Montgomery || r.parameter<0 || r.parameter>3)throw std::invalid_argument("unsupported GP curve identity");
    std::ostringstream s;s<<"p="<<r.prime.str()<<";s="<<r.sigma<<";";
    if(r.parameter==0)s<<"u=Mod(s^2-5,p);v=Mod(4*s,p);x=u^3;A=(3*u+v)*(v-u)^3/(4*x*v)-2;x=x/v^3;b=x*(x*(x+A)+1);y=1;";
    else if(r.parameter==1 || r.parameter==3)s<<"A=Mod(4*"<<(r.parameter==1?"s^2":"s")<<",p)/Mod(2^"<<(r.parameter==1?64:32)<<",p)-2;b=4*A+10;x=2;y=1;";
    else s<<"W=ellinit([0,Mod(36,p)]);t=ellmul(W,[-3,3],s);x3=(3*t[1]+t[2]+6)/(2*(t[2]-3));A=-(3*x3^4+6*x3^2-1)/(4*x3^3);y=4*A+10;b=1/y;x=2;";
    s<<"E=ellinit([0,b*A,0,b^2,0]);P=[b*x,b^2*y];";return s.str();
}
std::string group_script(const CurveRequest&r){
    return "ecm_go()={"+model(r)+
        "if(!isprime(p),print(\"ECM_MATH_ERROR|composite_modulus\");return());"
        "if(#E==0||!ellisoncurve(E,P),print(\"ECM_MATH_ERROR|invalid_curve\");return());"
        "my(n=ellcard(E),f=factor(n),o,fo);"
        "for(i=1,matsize(f)[1],if(!isprime(f[i,1]),error(\"unproven factor\")));"
        "o=ellorder(E,P,[n,f]);fo=factor(o);"
        "if(ellmul(E,P,o)!=[0],error(\"point multiple\"));"
        "for(i=1,matsize(fo)[1],if(!isprime(fo[i,1])||ellmul(E,P,o/fo[i,1])==[0],error(\"point minimality\")));"
        "print(\"GROUP|\",n);print(\"POINT|\",o);"
        "for(i=1,matsize(f)[1],print(\"G|\",f[i,1],\"|\",f[i,2]));"
        "for(i=1,matsize(fo)[1],print(\"P|\",fo[i,1],\"|\",fo[i,2]));print(\"ECM_OK\");};\n"
        "iferr(ecm_go(),err,print(\"ECM_GP_ERROR\"));quit();\n";
}
std::vector<std::string> split(const std::string&s){
    std::vector<std::string> out;size_t start=0;for(;;){size_t end=s.find('|',start);out.push_back(s.substr(start,end-start));if(end==s.npos)break;start=end+1;}return out;
}
void add_part(std::vector<PrimePower>&parts,const std::vector<std::string>&row,const Integer&bound){
    if(row.size()!=3 || row[1].size()>2500 || row[2].empty() || row[2].size()>5 || row[2].find_first_not_of("0123456789")!=row[2].npos)
        throw std::invalid_argument("invalid prime-power protocol");
    Integer p(row[1]);unsigned long e=std::stoul(row[2]);
    if(!e || e>8192 || mpz_cmp_ui(p.v,2)<0 || mpz_cmp(p.v,bound.v)>0 ||
       (mpz_sizeinbase(p.v,2)-1)*e>mpz_sizeinbase(bound.v,2) || classify_prime(p)==Primality::Composite)
        throw std::invalid_argument("invalid prime-power value");
    for(const auto&old:parts)if(mpz_cmp(old.prime.v,p.v)==0)throw std::invalid_argument("duplicate prime power");
    parts.push_back({std::move(p),uint32_t(e),Primality::Proven});
}
void product_check(const std::vector<PrimePower>&parts,const Integer&n){
    Integer product(1),power;
    for(const auto&p:parts){mpz_pow_ui(power.v,p.prime.v,p.exponent);mpz_mul(product.v,product.v,power.v);if(mpz_cmp(product.v,n.v)>0)throw std::invalid_argument("factor product exceeds value");}
    if(mpz_cmp(product.v,n.v))throw std::invalid_argument("factor product mismatch");
}
Analysis parse_group(const std::string&text,const Integer&p){
    Analysis out;std::istringstream input(text);std::string line;bool group=false,point=false,ok=false;
    std::vector<std::vector<std::string>> group_parts,point_parts;
    while(std::getline(input,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        auto row=split(line);
        if(row[0]=="ECM_OK"){if(ok)throw std::invalid_argument("duplicate completion marker");ok=true;}
        else if(row[0]=="GROUP" && row.size()==2){if(group)throw std::invalid_argument("duplicate group order");out.group_order=Integer(row[1]);group=true;}
        else if(row[0]=="POINT" && row.size()==2){if(point)throw std::invalid_argument("duplicate point order");out.point_order=Integer(row[1]);point=true;}
        else if(row[0]=="G")group_parts.push_back(row);
        else if(row[0]=="P")point_parts.push_back(row);
    }
    if(!ok || !group || !point)throw std::runtime_error("gp_protocol_failed");
    if(mpz_cmp_ui(out.group_order.v,1)<0 || mpz_cmp_ui(out.point_order.v,1)<0 || !mpz_divisible_p(out.group_order.v,out.point_order.v))
        throw std::invalid_argument("invalid group/point relationship");
    Integer delta,four_p;mpz_sub(delta.v,out.group_order.v,p.v);mpz_sub_ui(delta.v,delta.v,1);mpz_mul(delta.v,delta.v,delta.v);mpz_mul_ui(four_p.v,p.v,4);
    if(mpz_cmp(delta.v,four_p.v)>0)throw std::invalid_argument("group order violates Hasse bound");
    for(const auto&row:group_parts)add_part(out.multiple_factorization.factors,row,out.group_order);
    for(const auto&row:point_parts)add_part(out.point_factorization,row,out.point_order);
    product_check(out.multiple_factorization.factors,out.group_order);product_check(out.point_factorization,out.point_order);
    auto sort=[](std::vector<PrimePower>&parts){std::sort(parts.begin(),parts.end(),[](const PrimePower&a,const PrimePower&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});};
    sort(out.multiple_factorization.factors);sort(out.point_factorization);
    out.group_order_computed=true;out.multiple=out.group_order;out.point_minimality_checked=true;out.all_proven=true;
    out.modulus_primality=Primality::Proven;out.multiple_factorization.status=Status::Complete;out.multiple_factorization.all_proven=true;
    out.status=Status::Complete;return out;
}
std::string factor_script(const Integer&n){
    return "ecm_split()={my(n="+n.str()+",f=factor("+n.str()+"));"
        "for(i=1,matsize(f)[1],if(!isprime(f[i,1]),error(\"unproven factor\")));"
        "for(i=1,matsize(f)[1],print(\"F|\",f[i,1],\"|\",f[i,2]));print(\"ECM_OK\");};\n"
        "iferr(ecm_split(),err,print(\"ECM_GP_ERROR\"));quit();\n";
}
Factorization parse_factor(const std::string&text,const Integer&n){
    Factorization out;out.cofactor=n;std::istringstream input(text);std::string line;bool ok=false;
    while(std::getline(input,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();auto row=split(line);
        if(row[0]=="ECM_OK"){if(ok)throw std::invalid_argument("duplicate completion marker");ok=true;}
        else if(row[0]=="F")add_part(out.factors,row,n);
    }
    if(!ok || (out.factors.empty()&&mpz_cmp_ui(n.v,1)>0))throw std::runtime_error("gp_protocol_failed");
    product_check(out.factors,n);
    std::sort(out.factors.begin(),out.factors.end(),[](const PrimePower&a,const PrimePower&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});
    out.cofactor=Integer(1);out.status=Status::Complete;out.all_proven=true;return out;
}
} // namespace
Backend parse_backend(const std::string&s){if(s=="auto")return Backend::Auto;if(s=="gp")return Backend::Gp;if(s=="native"||s=="pari")return Backend::Native;if(s=="gmp")return Backend::Gmp;throw std::invalid_argument("--go-backend must be auto, pari/native, gp or gmp");}
const char *name(Backend b){return b==Backend::Auto?"auto":b==Backend::Gp?"gp":b==Backend::Gmp?"gmp":"pari";}
std::string quote(const std::string&s){
    std::string out="\"";const char*hex="0123456789abcdef";
    for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=char(c);}else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}else out+=char(c);}return out+'"';
}
namespace {
pari_protocol::V native_request(const Integer& n,const char* operation){auto v=pari_protocol::object();v.set("operation",pari_protocol::V::string(operation));v.set("integer",pari_protocol::V::string(n.str()));return v;}
pari_protocol::V last_checkpoint(const ProcessResult& process){
    pari_protocol::V value;std::istringstream lines(process.output);std::string line;
    while(std::getline(lines,line)){try{auto v=ecm_results::json::parse(line);if(v.get("protocol").text()=="2")value=std::move(v);}catch(const std::exception&) {}}
    if(value.type!='o')throw std::runtime_error("pari_protocol_failed");
    if(value.has("error"))throw std::runtime_error(value.get("error").text());return value;
}
std::string pari_failure(std::string s){if(s.rfind("gp_",0)==0)s.replace(0,3,"pari_");return s;}
}
Session::Session(Settings s):settings_(std::move(s)){}
GroupResult Session::group(const CurveRequest&r){
    GroupResult out;out.backend=name(settings_.backend==Backend::Auto?Backend::Native:settings_.backend);
    try{
        if(mpz_cmp_ui(r.prime.v,3)<=0||mpz_sizeinbase(r.prime.v,2)>settings_.native.max_bits||classify_prime(r.prime)==Primality::Composite){out.value.reason="modulus must be a prime >3 within the input bit limit";return out;}
        (void)construct_curve(r);
        if(settings_.backend==Backend::Auto||settings_.backend==Backend::Native){
            if(native_ms_>=settings_.native_total_ms){out.value.status=Status::ResourceLimit;out.pari_failure=out.value.reason="PARI session budget exhausted";}
            else{
                auto request=native_request(r.prime,"group");request.set("sigma",pari_protocol::V::string(std::to_string(r.sigma)));request.set("family",pari_protocol::V::string(r.family==Family::Edwards?"edwards":"montgomery"));request.set("parameter",pari_protocol::V::number(std::to_string(r.parameter)));
                auto settings=settings_;settings.pari_timeout_ms=std::min(settings.pari_timeout_ms,uint64_t(std::ceil(settings_.native_total_ms-native_ms_)));
                const auto start=Clock::now();auto process=execute_process(settings,request.dump(),true);native_ms_+=milliseconds(start);
                try{out.value=pari_protocol::group(last_checkpoint(process),r);}catch(const std::exception&e){out.value.status=Status::ResourceLimit;out.value.reason=e.what();}
                if(!process.failure.empty()&&out.value.status!=Status::InvalidInput){out.value.status=Status::ResourceLimit;out.value.all_proven=false;out.value.reason=pari_failure(process.failure)+"; "+out.value.reason;}
                out.pari_failure=out.value.reason;
                if(out.value.status==Status::Complete||out.value.status==Status::InvalidInput)return out;
            }
            if(settings_.backend==Backend::Native)return out;
        }
        if(settings_.backend!=Backend::Gmp){
            if(unavailable_.empty()){
                auto process=execute_process(settings_,group_script(r));out.gp_output=std::move(process.output);
                if(process.failure.empty()){
                    if(out.gp_output.find("ECM_MATH_ERROR|")!=std::string::npos){out.backend="gp";out.value=Analysis{};out.value.reason="gp_mathematical_error";return out;}
                    try{auto value=parse_group(out.gp_output,r.prime);out.value=std::move(value);out.backend="gp";return out;}
                    catch(const std::invalid_argument&e){out.backend="gp";out.value=Analysis{};out.value.reason=std::string("gp_invalid_result: ")+e.what();return out;}
                    catch(const std::runtime_error&e){process.failure=e.what();}
                }unavailable_=process.failure;
            }
            out.gp_failure=unavailable_;
            if(out.value.reason.empty())out.value.reason=unavailable_;else out.value.reason+="; "+unavailable_;
            if(out.value.status==Status::InvalidInput)out.value.status=Status::ResourceLimit;
            return out; // Never silently select the GMP algorithm.
        }
        if(native_ms_>=settings_.native_total_ms){out.value.status=Status::ResourceLimit;out.value.reason="GMP session budget exhausted";return out;}
        auto limits=settings_.native;limits.milliseconds=std::min(limits.milliseconds,uint64_t(std::ceil(settings_.native_total_ms-native_ms_)));
        const auto start=Clock::now();out.value=analyze(r,limits);native_ms_+=milliseconds(start);
    }catch(const std::exception&e){out.value.status=Status::InvalidInput;out.value.reason=e.what();}return out;
}
FactorResult Session::factor(const Integer&integer){
    FactorResult out;out.backend=name(settings_.backend==Backend::Auto?Backend::Native:settings_.backend);out.value.cofactor=integer;
    try{
        if(mpz_cmp_ui(integer.v,1)<0||mpz_sizeinbase(integer.v,2)>settings_.native.max_bits){out.value.reason="factorization requires a positive integer within the input bit limit";return out;}
        if(settings_.backend==Backend::Auto||settings_.backend==Backend::Native){
            if(native_ms_>=settings_.native_total_ms){out.value.status=Status::ResourceLimit;out.pari_failure=out.value.reason="PARI session budget exhausted";}
            else{
                auto settings=settings_;settings.pari_timeout_ms=std::min(settings.pari_timeout_ms,uint64_t(std::ceil(settings_.native_total_ms-native_ms_)));
                const auto start=Clock::now();auto process=execute_process(settings,native_request(integer,"factor").dump(),true);native_ms_+=milliseconds(start);
                try{out.value=pari_protocol::factor(last_checkpoint(process),integer);}catch(const std::exception&e){out.value.status=Status::ResourceLimit;out.value.reason=e.what();}
                if(!process.failure.empty()&&out.value.status!=Status::InvalidInput){out.value.status=Status::Partial;out.value.all_proven=false;out.value.reason=pari_failure(process.failure)+"; "+out.value.reason;}
                out.pari_failure=out.value.reason;
                if(out.value.status==Status::Complete||out.value.status==Status::InvalidInput)return out;
            }if(settings_.backend==Backend::Native)return out;
        }
        if(settings_.backend!=Backend::Gmp){
            if(unavailable_.empty()){
                auto process=execute_process(settings_,factor_script(integer));out.gp_output=std::move(process.output);
                if(process.failure.empty()){
                    try{auto value=parse_factor(out.gp_output,integer);out.value=std::move(value);out.backend="gp";return out;}
                    catch(const std::invalid_argument&e){out.backend="gp";out.value=Factorization{};out.value.cofactor=integer;out.value.reason=std::string("gp_invalid_result: ")+e.what();return out;}
                    catch(const std::runtime_error&e){process.failure=e.what();}
                }unavailable_=process.failure;
            }
            out.gp_failure=unavailable_;if(out.value.reason.empty())out.value.reason=unavailable_;else out.value.reason+="; "+unavailable_;
            if(out.value.status==Status::InvalidInput)out.value.status=Status::ResourceLimit;return out;
        }
        if(native_ms_>=settings_.native_total_ms){out.value.status=Status::ResourceLimit;out.value.reason="GMP session budget exhausted";return out;}
        auto limits=settings_.native;limits.milliseconds=std::min(limits.milliseconds,uint64_t(std::ceil(settings_.native_total_ms-native_ms_)));
        const auto start=Clock::now();out.value=factor_integer(integer,limits);native_ms_+=milliseconds(start);
    }catch(const std::exception&e){out.value.status=Status::InvalidInput;out.value.reason=e.what();}return out;
}
} // namespace ecm_diagnostics
