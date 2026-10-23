#include "../../../src/ecm/application/gui_v2/group_order_job.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <tlhelp32.h>
#include <iostream>
using namespace ecmgui_v2;
namespace fs=std::filesystem;
void write(const fs::path&p,const std::string&s){std::ofstream(p)<<s;}
std::string read(const fs::path&p){std::ifstream in(p,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
bool child_alive(DWORD parent){HANDLE all=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);assert(all!=INVALID_HANDLE_VALUE);PROCESSENTRY32W e{sizeof(e)};bool found=false;for(BOOL ok=Process32FirstW(all,&e);ok;ok=Process32NextW(all,&e))if(e.th32ParentProcessID==parent){HANDLE p=OpenProcess(SYNCHRONIZE,FALSE,e.th32ProcessID);if(p){found|=WaitForSingleObject(p,0)==WAIT_TIMEOUT;CloseHandle(p);}}CloseHandle(all);return found;}
void wait(GroupOrderJob&job){const auto deadline=GetTickCount64()+10000;while(job.busy()&&GetTickCount64()<deadline){job.poll();Sleep(5);}assert(!job.busy()&&job.error.empty());}
int main(int argc,char**argv){
    assert(argc==4);const auto root=fs::absolute(argv[3]);assert(!fs::exists(root));fs::create_directories(root);const std::string helper=fs::absolute(argv[1]).u8string(),fake=fs::absolute(argv[2]).u8string();
    const auto first=root/"results.json.txt",second=root/"results.json_2.txt";
    write(first,"{\"status\":\"F\",\"worktype\":\"ECM\",\"sigma\":18446744073709551615,\"factors\":[\"1009\",\"1013\"]}\n{broken\n{\"status\":\"NF\"}\n");
    write(second,"{\"status\":\"F\",\"worktype\":\"ECM\",\"GMP-ECM-param3\":{\"sigma\":26},\"factors\":[\"1009\"]}\n{\"status\":\"F\",\"worktype\":\"ECM\",\"Edwards\":{},\"factors\":[\"1009\"]}\n");
    const auto original_first=read(first),original_second=read(second);
    assert(discover_order_files(root.u8string(),{first.u8string(),first.u8string()}).size()==2);
    std::string error;auto rows=read_order_records(first.u8string(),error);assert(rows.size()==2&&!error.empty()&&rows[0].input.sigma=="18446744073709551615");rows=read_order_records(second.u8string(),error);assert(rows.size()==1&&rows[0].input.profile==4&&!error.empty());
    OrderInput input{"1009","26","",0,2};GroupOrderJob job;
    for(int profile:{0,1,3,4,5}){input.profile=profile;assert(job.start(input,helper));assert(!job.start(input,helper));wait(job);const auto&r=job.result;assert(r.get("status").text()=="complete"&&r.get("group_computed").scalar=="true"&&r.get("point_minimality").scalar=="true");
        unsigned long long product=1;for(const auto&factor:r.get("group_factors").items)for(int i=0;i<std::stoi(factor.get("exponent").text());++i)product*=std::stoull(factor.get("prime").text());assert(product==std::stoull(r.get("group_order").text())&&product%std::stoull(r.get("point_order").text())==0);
    }
    input.profile=0;input.backend=0;input.gp=(root/"missing GP.exe").u8string();assert(job.start(input,helper));wait(job);assert(job.result.get("backend").text()=="native"&&!job.result.get("fallback").text().empty());
    input.backend=1;assert(job.start(input,helper));wait(job);assert(job.result.get("status").text()=="resource_limit"&&job.result.get("group_computed").scalar=="false");
    input.backend=2;input.factor="15";assert(job.start(input,helper));wait(job);assert(job.result.get("status").text()=="invalid_input");
    input.factor="1009";input.sigma="18446744073709551616";assert(!job.start(input,helper)&&!job.error.empty());input.sigma="26";input.profile=2;assert(!job.start(input,helper));input.profile=0;
    const auto counter=root/"gp-counter.txt";SetEnvironmentVariableA("ECM_TEST_GP_COUNTER",counter.string().c_str());SetEnvironmentVariableA("ECM_TEST_GP_MODE","timeout");input.backend=0;input.gp=fake;
    assert(job.start(input,helper));const auto pid=job.pid();const auto deadline=GetTickCount64()+5000;while(!fs::exists(counter)&&GetTickCount64()<deadline)Sleep(10);assert(fs::exists(counter)&&child_alive(pid));
    const auto start=GetTickCount64();job.cancel();assert(!job.busy()&&GetTickCount64()-start<2000);Sleep(100);assert(!child_alive(pid));
    SetEnvironmentVariableA("ECM_TEST_GP_COUNTER",nullptr);SetEnvironmentVariableA("ECM_TEST_GP_MODE",nullptr);
    assert(read(first)==original_first&&read(second)==original_second);std::cout<<"PASS: five constructions, exact products, missing-GP fallback, GP-only failure, composite rejection, uint64 validation, read-only two-file import and cancellation of helper/GP tree\n";
}
