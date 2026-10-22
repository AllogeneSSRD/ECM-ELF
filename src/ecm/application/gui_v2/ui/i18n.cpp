#include "i18n.h"
#include "../../gui/localization.h"
#include <map>
#include <vector>
#include <algorithm>
#include <filesystem>
namespace ecmgui_v2::ui {
namespace {
using Catalog=std::map<std::string,std::string>;
Catalog english,chinese;std::string directory;
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
        const auto key=e.panel+"."+e.id;
        if(!result.emplace(key,e.text).second){error="Duplicate localization key: "+key;return false;}
    }
    return true;
}
}
bool load_languages(const std::string& dir,std::string& error){
    Catalog en,zh;error.clear();
    if(!read(dir+"/english.xml",en,error))return false;
    if(!read(dir+"/options.english.xml",en,error))return false;
    std::string warning;
    for(const char* file:{"chineseSimplified.xml","options.chineseSimplified.xml"}){
        const auto path=dir+"/"+file;
        if(!std::filesystem::is_regular_file(std::filesystem::u8path(path))){warning+="Missing translation: "+path+"; ";continue;}
        if(!read(path,zh,error))return false;
    }
    for(const auto& e:zh){const auto base=en.find(e.first);if(e.first.rfind("ui.option_",0)!=0&&base!=en.end()&&signature(e.second)!=signature(base->second)){
        error="Localization format mismatch: "+e.first;return false;
    }}
    english=std::move(en);chinese=std::move(zh);directory=dir;error=warning;return true;
}
const char* text(const char* key,bool zh){
    const std::string full="ui."+std::string(key);
    if(zh){const auto it=chinese.find(full);if(it!=chinese.end()&&!it->second.empty())return it->second.c_str();}
    const auto it=english.find(full);return it==english.end()?key:it->second.c_str();
}
std::string message(const std::string& value,bool zh){
    if(!zh)return value;
    // Compatibility adapter for shared controller diagnostics; both texts are XML.
    const std::pair<const std::string,std::string>* best=nullptr;
    for(const auto& e:english)if(e.first.rfind("ui.message_",0)==0&&!e.second.empty()&&value.rfind(e.second,0)==0)
        if(!best||e.second.size()>best->second.size())best=&e;
    if(best){const auto it=chinese.find(best->first);if(it!=chinese.end()&&!it->second.empty())return it->second+value.substr(best->second.size());}
    return value;
}
const std::string& language_directory(){return directory;}
}
