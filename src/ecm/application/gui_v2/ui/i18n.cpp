#include "i18n.h"
#include "../../gui/localization.h"
#include "../generated/localization_aliases.h"
#include <map>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cctype>
#include <cstring>
#include <iterator>
namespace ecmgui_v2::ui {
namespace {
using Catalog=std::map<std::string,std::string>;
Catalog english,chinese;std::string directory,selected_file;std::vector<std::string> files;
std::string readable_key(const std::string& key){
    const auto end=std::end(localization_aliases);
    const auto it=std::lower_bound(std::begin(localization_aliases),end,key,[](const auto& alias,const auto& value){return std::strcmp(alias.legacy,value.c_str())<0;});
    return it!=end&&key==it->legacy?it->readable:key;
}
bool interface_xml(const std::string& file){
    if(file.empty()||file=="."||file==".."||file.find_first_of("/\\:")!=file.npos)return false;
    auto lower=file;std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    return lower.size()>4&&lower.substr(lower.size()-4)==".xml"&&lower.rfind("options.",0)!=0;
}
// Compare printf argument signatures before a translator's text reaches ImGui.
std::vector<std::string> signature(const std::string& value){
    std::vector<std::string> result;
    for(size_t i=0;i<value.size();++i)if(value[i]=='%'){
        if(i+1<value.size()&&value[i+1]=='%'){++i;continue;}
        ++i;while(i<value.size()&&std::string("-+ #0123456789.*").find(value[i])!=std::string::npos){if(value[i]=='*')result.push_back("int");++i;}
        const auto start=i;while(i<value.size()&&std::string("hljztL").find(value[i])!=std::string::npos)++i;
        if(i>=value.size())return {"invalid"};
        if(std::string("diuoxXfFeEgGaAcspn").find(value[i])!=std::string::npos)
            result.push_back(value.substr(start,i-start+1));
    }
    return result;
}
bool read(const std::string& path,Catalog& result,std::string& error){
    std::vector<ecmgui::Localization::Entry> entries;
    if(!ecmgui::parse_localization_xml(path,entries,error))return false;
    for(const auto& e:entries){
        const auto key=e.panel+"."+(e.panel=="ui"?readable_key(e.id):e.id);
        if(!result.emplace(key,e.text).second){error="Duplicate localization key: "+key;return false;}
    }
    return true;
}
}
bool load_languages(const std::string& dir,std::string& error,const std::string& file){
    Catalog en,zh;error.clear();
    if(!interface_xml(file)){error="Invalid interface localization filename: "+file;return false;}
    if(!read(dir+"/english.xml",en,error))return false;
    if(!read(dir+"/options.english.xml",en,error))return false;
    std::string warning;
    if(file!="english.xml"){
        const auto path=dir+"/"+file;
        if(!std::filesystem::is_regular_file(std::filesystem::u8path(path)))warning="Missing translation: "+path+"; ";
        else {
            if(!read(path,zh,error))return false;
            if(std::none_of(zh.begin(),zh.end(),[](const auto& entry){return entry.first.rfind("ui.",0)==0;})){error="No UI translations in: "+path;return false;}
        }
        const auto options=dir+"/options."+file;
        if(std::filesystem::is_regular_file(std::filesystem::u8path(options))&&!read(options,zh,error))return false;
    }
    for(const auto& e:zh){const auto base=en.find(e.first);if(e.first.rfind("ui.option_",0)!=0&&base!=en.end()&&signature(e.second)!=signature(base->second)){
        error="Localization format mismatch: "+e.first;return false;
    }}
    english=std::move(en);chinese=std::move(zh);directory=dir;selected_file=file;error=warning;refresh_language_files();return true;
}
void refresh_language_files(){
    std::vector<std::string> found;std::error_code error;
    for(std::filesystem::directory_iterator it(std::filesystem::u8path(directory),error),end;!error&&it!=end;it.increment(error)){
        const auto name=it->path().filename().u8string();
        if(it->is_regular_file(error)&&interface_xml(name))found.push_back(name);
    }
    std::sort(found.begin(),found.end());files=std::move(found);
}
const std::vector<std::string>& language_files(){return files;}
const std::string& selected_language_file(){return selected_file;}
std::string language_label(const std::string& file){return std::filesystem::u8path(file).stem().u8string();}
const char* text(const char* key,bool zh){
    const std::string full="ui."+readable_key(key);
    if(zh){const auto it=chinese.find(full);if(it!=chinese.end()&&!it->second.empty())return it->second.c_str();}
    const auto it=english.find(full);return it==english.end()?key:it->second.c_str();
}
std::string message(const std::string& value,bool zh){
    if(!zh)return value;
    // Compatibility adapter for shared controller diagnostics; both texts are XML.
    const std::pair<const std::string,std::string>* best=nullptr;
    for(const auto& e:english)if(e.first.rfind("ui.message.",0)==0&&!e.second.empty()&&value.rfind(e.second,0)==0)
        if(!best||e.second.size()>best->second.size())best=&e;
    if(best){const auto it=chinese.find(best->first);if(it!=chinese.end()&&!it->second.empty())return it->second+value.substr(best->second.size());}
    return value;
}
const std::string& language_directory(){return directory;}
}
