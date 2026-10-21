#include "results.h"
#include "../../../src/ecm/storage/results/ecm_result_json.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs=std::filesystem;
using namespace ecmgui;
static int checks=0;
static void check(bool ok,const char *name) {
    if(!ok)throw std::runtime_error(name);
    ++checks;std::cout<<"[ok] "<<name<<'\n';
}
static void write(const fs::path &path,const std::string &text,bool append=false) {
    std::ofstream out(path,std::ios::binary|(append?std::ios::app:std::ios::trunc));
    out<<text;out.close();if(!out)throw std::runtime_error("test write failed");
}
static std::string read(const fs::path &path) {
    std::ifstream in(path,std::ios::binary);
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
static std::string row(bool factor=false,bool stage2=false) {
    return std::string("{\"status\":\"")+(factor?"F":"NF")+"\",\"exponent\":127,\"worktype\":\"ECM\",\"b1\":100,"+
        (factor?(stage2?"\"stage\":2,\"b2\":10000,":"\"stage\":1,")+std::string("\"factors\":[\"233\"],"):
                 stage2?"\"b2\":10000,":"\"b2\":100,")+
        "\"Edwards\":{},\"curves\":1,\"program\":{\"name\":\"ECM-ELF\",\"version\":\"0.44.9\"},"
        "\"timestamp\":\"2026-10-21 00:00:00\",\"checksum\":{\"version\":1,\"checksum\":\"1234ABCD\"}}";
}
int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("pass a fresh test directory");
        const auto root=fs::absolute(fs::u8path(argv[1]));
        if(fs::exists(root))throw std::runtime_error("test directory must be fresh");
        fs::create_directories(root);
        const auto config=root/"ecm.toml",first=root/"results.json.txt",second=root/"results.json_2.txt";
        write(config,"[gui]\nNumWorkers=2\n");
        std::vector<ecm_config::TomlLine> lines;
        check(ecm_config::read_toml(config.u8string(),lines),"read TOML");
        auto paths=result_paths(lines,config.u8string(),2);
        check(paths.size()==2 && paths[0]==first.u8string() && paths[1]==second.u8string(),"default worker paths and shared stage dedup");
        write(config,"[stage1]\nstage1_submission_file=\"s1.txt\"\n[workers.2.stage2]\nstage2_submission_file=\"custom.txt\"\n");
        ecm_config::read_toml(config.u8string(),lines);paths=result_paths(lines,config.u8string(),2);
        check(paths.size()==3 && fs::u8path(paths[0]).filename()=="s1.txt" && fs::u8path(paths[2]).filename()=="custom.txt","explicit paths are not worker-suffixed and overrides apply");
        ResultStore store;store.set_paths({first.u8string(),second.u8string(),first.u8string()});
        check(store.files().size()==2,"same source is read once");store.refresh();
        check(store.rows().empty() && !fs::exists(first),"missing source is read-only and empty");
        const auto nf=row(),f=row(true),s2=row(false,true),f2=row(true,true);
        write(first,"\xef\xbb\xbf"+nf+"\r\n"+f+"\r\n");write(second,s2+'\n'+f2+'\n');
        check(store.refresh() && store.rows().size()==4,"read completed F and NF from both sources");
        check(!store.refresh(),"unchanged files do not reload");
        const auto before_touch=fs::last_write_time(first);
        fs::last_write_time(first,before_touch+std::chrono::seconds(2));
        check(store.refresh() && store.rows().size()==4,"mtime-only change triggers automatic reload");
        auto replacement=nf;replacement.replace(replacement.find("127"),3,"131");
        write(first,"\xef\xbb\xbf"+replacement+"\r\n"+f+"\r\n");
        fs::last_write_time(first,before_touch+std::chrono::seconds(4));
        check(store.refresh() && store.rows()[0].target=="M131","same-size edit reloads when mtime changes");
        write(first,"\xef\xbb\xbf"+nf+"\r\n"+f+"\r\n");store.refresh(true);
        check(store.rows()[0].stage==1 && store.rows()[1].stage==1 && store.rows()[2].stage==2 && store.rows()[3].stage==2,"derive stage from published result fields");
        ResultFilter filter;filter.status="F";
        check(store.filtered(filter)==std::vector<std::size_t>({1,3}),"F filter");
        filter.status="NF";filter.stage=2;
        check(store.filtered(filter)==std::vector<std::size_t>{2},"NF and stage filters combine");
        filter=ResultFilter{};filter.source=1;filter.search="ecm-elf";
        check(store.filtered(filter)==std::vector<std::size_t>({2,3}),"source and case-insensitive text filters combine");
        check(store.copy_text({3,0})==f2+'\n'+nf+'\n',"copy preserves JSON and supplied order");
        const auto original=read(first);
        store.refresh(true);check(read(first)==original,"force refresh never writes source");
        write(first,"{\"status\":",true);store.refresh();
        check(store.rows().size()==4 && store.files()[0].pending_lines==1,"incomplete last row is withheld");
        write(first,nf.substr(10)+'\n',true); // Complete the prefix {"status":.
        store.refresh();check(store.rows().size()==5 && store.files()[0].pending_lines==0,"completed append is discovered");
        write(first,nf+'\n'+"{broken}\n"+f+'\n');store.refresh(true);
        check(store.files()[0].invalid_lines==1 && store.rows().size()==4,"malformed complete row is excluded while valid rows remain");
        check(!store.refresh(),"unchanged malformed source does not spin reload");
        write(first,"{}");store.refresh();
        check(store.files()[0].invalid_lines==1 && store.files()[0].pending_lines==0,
              "complete non-submission JSON without newline is invalid, not pending");
        write(first,f+'\n'+nf);store.refresh();
        check(store.rows()[0].status=="F" && store.rows()[1].status=="NF","replaced source keeps file publication order");
        check(store.rows().size()==4,"complete JSON without final newline is accepted");
        write(first,nf+'\n'+nf+'\n');store.refresh();
        check(store.rows().size()==4 && store.rows()[0].id!=store.rows()[1].id,"identical rows are displayed independently");
        auto generic=ecm_results::json::parse(nf);
        generic.members.erase(generic.members.begin()+1);
        generic.set("N",ecm_results::json::Value::string(std::string(2100,'9')));
        write(first,generic.dump()+'\n');store.refresh();
        check(store.rows()[0].target=="N (2100 digits)" && store.copy_text({0})==generic.dump()+'\n',"large generic N copies without numeric loss");
        const auto custom=root/fs::u8path("手动结果.txt");write(custom,f+'\n');store.add_path(custom.u8string());store.refresh();
        check(store.files().size()==3 && store.rows().size()==4,"manual source addition");
        fs::remove(second);store.refresh();check(store.rows().size()==2,"removed file does not leave stale records");
        std::cout<<"passed: "<<checks<<" failed: 0\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';std::cout<<"passed: "<<checks<<" failed: 1\n";return 1;}
}
