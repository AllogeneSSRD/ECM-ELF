#include "io/stage1_checkpoint.h"
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace fs=std::filesystem;
void put(const fs::path &p,const char *s){std::ofstream(p,std::ios::binary)<<s;}
std::string get(const fs::path &p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char **argv){
    ecm_stage1_checkpoint::Context checkpoint;

    if(argc!=2)return 2;
    const auto dir=fs::absolute(argv[1]);fs::create_directories(dir);
    const auto target=dir/"target",temp=dir/"target.new";
    put(target,"old");put(temp,"new");checkpoint.set_backup_limit(0);
    const auto lock=CreateFileW(target.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,0,nullptr);
    if(lock==INVALID_HANDLE_VALUE)return 3;
    std::thread release([lock]{Sleep(100);CloseHandle(lock);});
    const bool ok=checkpoint.commit(target.string(),temp.string());release.join();
    if(!ok || get(target)!="new" || fs::exists(temp)){std::cerr<<"transient replacement failed\n";return 1;}
    put(temp,"next");
    const auto permanent=CreateFileW(target.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,0,nullptr);
    if(permanent==INVALID_HANDLE_VALUE)return 4;
    const auto begin=GetTickCount64();
    const bool failed=checkpoint.commit(target.string(),temp.string());
    const auto elapsed=GetTickCount64()-begin;CloseHandle(permanent);
    if(failed || elapsed>3000 || get(target)!="new" || get(temp)!="next"){std::cerr<<"permanent lock did not preserve files\n";return 1;}
    std::cout<<"PASS transient lock recovery; permanent lock preserves target and pending write\n";
}
