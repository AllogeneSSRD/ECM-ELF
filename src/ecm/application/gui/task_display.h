#pragma once
#include <cctype>
#include <limits>
#include <string>
#include <vector>

namespace ecmgui {
struct TaskDisplay {
    bool valid=false,has_aid=false,has_factors=false;
    bool from_save=false;
    unsigned long long curves=0;
    unsigned long long mersenne_bits=0;
    std::string number;
};
// Presentation only: never return the raw assignment, AID, factor values or path.
inline TaskDisplay task_display(const std::string &line) {
    TaskDisplay out;
    const auto eq=line.find('=');if(eq==line.npos)return out;
    auto trim=[](std::string s) {
        const auto a=s.find_first_not_of(" \t\r\n"),b=s.find_last_not_of(" \t\r\n");
        return a==s.npos?std::string{}:s.substr(a,b-a+1);
    };
    auto keyword=trim(line.substr(0,eq));
    for(char &c:keyword)c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if(keyword=="ECMSTAGE2PART")keyword="ECMSTAGE2";
    if(keyword!="ECM" && keyword!="ECM2" && keyword!="ECMSTAGE1" && keyword!="ECMSTAGE2" && keyword!="ECMSTAGE2N")return out;
    std::vector<std::string> values;std::vector<bool> quoted;
    std::string value;bool in_quote=false,had_quote=false;
    for(size_t i=eq+1;i<=line.size();++i) {
        const char c=i<line.size()?line[i]:',';
        if(c=='"') {
            if(in_quote && i+1<line.size() && line[i+1]=='"'){value+='"';++i;continue;}
            in_quote=!in_quote;had_quote=true;
        } else if(c==',' && !in_quote) {
            values.push_back(trim(value));quoted.push_back(had_quote);value.clear();had_quote=false;
        } else value+=c;
    }
    if(in_quote || values.empty())return out;
    const auto count=[](const std::string& text) {
        unsigned long long value=0;
        if(text.empty())return value;
        for(unsigned char c:text) {
            if(c<'0'||c>'9'||value>(std::numeric_limits<unsigned long long>::max()-(c-'0'))/10)return 0ull;
            value=value*10+c-'0';
        }
        return value;
    };
    auto numeric=[](const std::string &s) {
        if(s.empty() || s.size()>64)return false;
        bool digit=false;
        for(unsigned char c:s) {
            digit|=std::isdigit(c)!=0;
            if(!std::isdigit(c) && c!='+' && c!='-' && c!='.' && c!='e' && c!='E')return false;
        }
        return digit;
    };
    size_t offset=0;
    const bool hex_aid=values[0].size()==32 && values[0].find_first_not_of("0123456789abcdefABCDEF")==values[0].npos;
    if(keyword=="ECMSTAGE2N") {
        if(values[0].empty()||values[0]=="N/A"||hex_aid||(values.size()>1&&!numeric(values[1])))offset=1;
        if(offset>=values.size()||values[offset].empty())return out;
        out.valid=out.from_save=true;out.number="N";
        out.has_aid=offset&&values[0]!="N/A"&&!values[0].empty();
        out.has_factors=values.size()>offset+1&&quoted.back()&&!values.back().empty();
        if(values.size()-out.has_factors>offset+3)out.curves=count(values[offset+3]);
        return out;
    }
    if((!numeric(values[0]) || hex_aid || quoted[0]) && values[0].compare(0,5,"FFT2=")) {out.has_aid=!values[0].empty() && values[0]!="N/A";++offset;}
    if(offset<values.size() && values[offset].compare(0,5,"FFT2=")==0)++offset;
    // A numeric AID in an ECMSTAGE2 row is distinguishable by the save position.
    if((keyword=="ECMSTAGE1" || keyword=="ECMSTAGE2") && offset==0 && values.size()>5 && numeric(values[4]) && !numeric(values[5])) {
        out.has_aid=true;offset=1;
    }
    if(values.size()<offset+5)return TaskDisplay{};
    for(size_t i=offset;i<offset+4;++i)if(!numeric(values[i]))return TaskDisplay{};
    if(values[offset+2].find_first_not_of("0123456789")!=values[offset+2].npos)return TaskDisplay{};
    if(values[offset]=="1" && values[offset+1]=="2" && values[offset+3]=="-1") {
        out.number="M"+values[offset+2];
        out.mersenne_bits=count(values[offset+2]);
    }
    else out.number=values[offset]+"*"+values[offset+1]+"^"+values[offset+2]+
        (values[offset+3].front()=='-' || values[offset+3].front()=='+'?"":"+")+values[offset+3];
    const bool saved=keyword=="ECMSTAGE2"||(keyword=="ECMSTAGE1"&&!numeric(values[offset+4]));
    const size_t curve_column=offset+(saved?7:keyword=="ECMSTAGE1"?5:6);
    out.has_factors=values.size()>offset+5 && quoted.back() && !values.back().empty();
    out.curves=values.size()-out.has_factors>curve_column?count(values[curve_column]):saved?0:100;
    out.valid=true;return out;
}
inline std::string task_caption(const std::string &line,const std::string &factor_label,
                               const std::string &aid_label,const std::string &unavailable) {
    const auto d=task_display(line);if(!d.valid)return unavailable;
    return d.number+(d.has_factors?" / "+factor_label:"")+(d.has_aid?"  |  "+aid_label:"");
}
} // namespace ecmgui
