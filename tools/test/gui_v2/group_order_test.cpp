#include "../../../src/ecm/application/gui_v2/group_order_job.h"
#include "../../../src/ecm/application/gui_v2/group_order_gp.h"
#include "../../../src/ecm/application/gui_v2/ui/group_order_presentation.h"
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
    assert(argc==5);const auto root=fs::absolute(argv[3]);assert(!fs::exists(root));fs::create_directories(root);const std::string helper=fs::absolute(argv[1]).u8string(),fake=fs::absolute(argv[2]).u8string(),gp=fs::absolute(argv[4]).u8string(),missing=(root/"absent-helper.exe").u8string();
    const auto first=root/"results.json.txt",second=root/"results.json_2.txt";
    write(first,"{\"status\":\"F\",\"worktype\":\"ECM\",\"sigma\":18446744073709551615,\"factors\":[\"1009\",\"1013\"]}\n{broken\n{\"status\":\"NF\"}\n");
    write(second,"{\"status\":\"F\",\"worktype\":\"ECM\",\"GMP-ECM-param3\":{\"sigma\":26},\"factors\":[\"1009\"]}\n{\"status\":\"F\",\"worktype\":\"ECM\",\"Edwards\":{},\"factors\":[\"1009\"]}\n");
    const auto original_first=read(first),original_second=read(second);
    assert(discover_order_files(root.u8string(),{first.u8string(),first.u8string()}).size()==2);
    std::string error;auto rows=read_order_records(first.u8string(),error);assert(rows.size()==2&&!error.empty()&&rows[0].input.sigma=="18446744073709551615");rows=read_order_records(second.u8string(),error);assert(rows.size()==1&&rows[0].input.profile==4&&!error.empty());
    OrderInput input{"1009","26","",0,2};GroupOrderJob job;
    {using namespace order_gp;
     assert(multiply(std::string(500,'9'),std::string(500,'9'))==std::string(499,'9')+"8"+std::string(499,'0')+"1");
     assert(plus_one(std::string(500,'9'))=="1"+std::string(500,'0')&&difference("1"+std::string(500,'0'),"1")==std::string(500,'9'));
     const std::string valid="GROUP|996\nPOINT|249\nG|2|2\nG|3|1\nG|83|1\nP|3|1\nP|83|1\nECM_OK\n";
     auto formatted=parse(valid,input);assert(formatted.get("point_order").text()=="249");
     assert(ui::order_factorization(formatted,false)=="2^2 × 3 × 83"&&ui::order_factorization(formatted,true)=="3 × 83");
     assert(ui::order_succeeded(formatted,"")&&!ui::order_succeeded(formatted,"process failed"));
     formatted.set("status",go_json::Value::string("partial"));formatted.set("cofactor",go_json::Value::string("101"));
     assert(!ui::order_succeeded(formatted,"")&&ui::order_factorization(formatted,false)=="2^2 × 3 × 83 × [101]");
     auto identity=go_json::parse("{\"point_order\":\"1\",\"group_order\":\"1\",\"cofactor\":\"1\"}");
     assert(ui::order_factorization(identity,true)=="1"&&ui::order_factorization(identity,false)=="1");
     for(const auto bad:{"GROUP|992\nPOINT|249\nG|2|2\nG|3|1\nG|83|1\nP|3|1\nP|83|1\nECM_OK\n","GROUP|996\nPOINT|248\nG|2|2\nG|3|1\nG|83|1\nP|3|1\nP|83|1\nECM_OK\n","GROUP|996\nGROUP|996\nPOINT|1\nECM_OK\n"}){bool rejected=false;try{parse(bad,input);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
     auto large=input;large.factor="170141183460469231731687303715884105727";
     const auto checked=parse("GROUP|170141183460469231731687303715884105728\nPOINT|18446744073709551616\nG|2|127\nP|2|64\nECM_OK\n",large);
     assert(checked.get("point_order").text()=="18446744073709551616"); // Synthetic protocol arithmetic, not a curve oracle.
    }
    for(int profile:{0,1,3,4,5}){input.profile=profile;assert(job.start(input,helper));assert(!job.start(input,helper));wait(job);const auto&r=job.result;assert(r.get("status").text()=="complete"&&r.get("group_computed").scalar=="true"&&r.get("point_minimality").scalar=="true");
        unsigned long long product=1;for(const auto&factor:r.get("group_factors").items)for(int i=0;i<std::stoi(factor.get("exponent").text());++i)product*=std::stoull(factor.get("prime").text());assert(product==std::stoull(r.get("group_order").text())&&product%std::stoull(r.get("point_order").text())==0);
    }
    input.profile=0;input.backend=0;input.gp=(root/"missing GP.exe").u8string();assert(job.start(input,helper));wait(job);assert(job.result.get("backend").text()=="native"&&!job.result.get("fallback").text().empty());
    input.backend=1;assert(job.start(input,helper));wait(job);assert(job.result.get("status").text()=="resource_limit"&&job.result.get("group_computed").scalar=="false");
    input.gp=gp;
    for(const auto prime:{"101","103","1009","1013","10007","1000003","1000000007"})for(int profile:{0,1,3,4,5}){
        input.factor=prime;input.profile=profile;input.backend=2;assert(job.start(input,helper));wait(job);const auto expected=job.result;
        input.backend=1;assert(job.start(input,missing));wait(job);assert(job.result.get("backend").text()=="gp"&&job.result.get("status").text()==expected.get("status").text());
        if(expected.get("status").text()=="complete")for(const auto key:{"group_order","point_order","group_factors","point_factors"})assert(job.result.get(key).dump()==expected.get(key).dump());
    }
    input.factor="1000003";input.sigma="18446744073709551615";
    for(int profile:{0,1,3,4,5}){input.profile=profile;input.backend=2;assert(job.start(input,helper));wait(job);const auto expected=job.result;input.backend=0;assert(job.start(input,missing));wait(job);assert(job.result.get("status").text()==expected.get("status").text());if(expected.get("status").text()=="complete"){assert(job.result.get("group_order").text()==expected.get("group_order").text()&&job.result.get("point_order").text()==expected.get("point_order").text());}}
    input={"1009","26",gp,0,0};assert(job.start(input,missing));wait(job);assert(job.result.get("backend").text()=="gp"&&job.result.get("status").text()=="complete");
    {const auto path=root/"PATH GP";fs::create_directories(path);fs::copy_file(gp,path/"gp.exe");const auto size=GetEnvironmentVariableW(L"PATH",nullptr,0);std::wstring old(size,L'\0');if(size){GetEnvironmentVariableW(L"PATH",old.data(),size);old.resize(size-1);}
     SetEnvironmentVariableW(L"PATH",(L"\""+path.wstring()+L"\"").c_str());assert(fs::equivalent(fs::u8path(find_order_gp()),path/"gp.exe"));SetEnvironmentVariableW(L"PATH",root.wstring().c_str());assert(find_order_gp().empty());
     SetEnvironmentVariableW(L"PATH",fs::u8path(gp).parent_path().c_str());auto bare=input;bare.gp="gp.exe";assert(job.start(bare,missing));wait(job);assert(job.result.get("backend").text()=="gp"&&job.result.get("status").text()=="complete");
     SetEnvironmentVariableW(L"PATH",size?old.c_str():nullptr);}
    input.gp=fake;input.backend=0;SetEnvironmentVariableA("ECM_TEST_GP_MODE","protocol");assert(job.start(input,helper));wait(job);assert(job.result.get("backend").text()=="native"&&job.result.get("fallback").text()=="gp_protocol_failed");
    SetEnvironmentVariableA("ECM_TEST_GP_MODE","invalid");assert(job.start(input,missing));wait(job);assert(job.result.get("status").text()=="invalid_input"&&job.result.get("backend").text()=="gp");
    input.backend=2;assert(!job.start(input,missing)&&job.error.find(missing)!=std::string::npos);
    input.backend=2;input.factor="15";assert(job.start(input,helper));wait(job);assert(job.result.get("status").text()=="invalid_input");
    input.factor="1009";input.sigma="18446744073709551616";assert(!job.start(input,helper)&&!job.error.empty());input.sigma="26";input.profile=2;assert(!job.start(input,helper));input.profile=0;
    const auto counter=root/"gp-counter.txt";SetEnvironmentVariableA("ECM_TEST_GP_COUNTER",counter.string().c_str());SetEnvironmentVariableA("ECM_TEST_GP_MODE","timeout");input.backend=0;input.gp=fake;
    assert(job.start(input,missing));const auto pid=job.pid();const auto deadline=GetTickCount64()+5000;while(!fs::exists(counter)&&GetTickCount64()<deadline)Sleep(10);assert(fs::exists(counter));HANDLE running=OpenProcess(SYNCHRONIZE,FALSE,pid);assert(running&&WaitForSingleObject(running,0)==WAIT_TIMEOUT);
    const auto start=GetTickCount64();job.cancel();assert(!job.busy()&&GetTickCount64()-start<2000);assert(WaitForSingleObject(running,1000)==WAIT_OBJECT_0);CloseHandle(running);assert(!child_alive(pid));
    SetEnvironmentVariableA("ECM_TEST_GP_COUNTER",nullptr);SetEnvironmentVariableA("ECM_TEST_GP_MODE",nullptr);
    assert(read(first)==original_first&&read(second)==original_second);std::cout<<"PASS: 40 GP/native comparisons, GP without helper, PATH discovery, lazy native fallback, invalid GP math rejection, uint64, read-only import and cancellation\n";
}
