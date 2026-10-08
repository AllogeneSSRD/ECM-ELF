#pragma once

// Shared lexical reader. CLI worker scope and GUI named sections are projections
// of the same lossless lines; preserving both keeps existing INI files valid.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ecm_config {
inline std::string trim(std::string s) {
    size_t first=0,last=s.size();
    while(first<last&&std::isspace(static_cast<unsigned char>(s[first])))++first;
    while(last>first&&std::isspace(static_cast<unsigned char>(s[last-1])))--last;
    return s.substr(first,last-first);
}
inline std::string lower(std::string s) {
    for(char &c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
inline int worker_header(const std::string &line,bool *bracket=nullptr) {
    if(bracket)*bracket=false;
    if(line.size()<3||line.front()!='['||line.back()!=']')return 0;
    if(bracket)*bracket=true;
    const auto body=trim(line.substr(1,line.size()-2));
    if(body.size()<6||lower(body.substr(0,6))!="worker")return 0;
    auto tail=trim(body.substr(6));
    if(tail.empty()||tail[0]!='#')return 0;
    tail=trim(tail.substr(1));
    if(tail.empty())return 0;
    int n=0;
    for(char c:tail){if(c<'0'||c>'9')return 0;n=n*10+c-'0';if(n>1000000)return 0;}
    return n;
}
struct IniLine {
    enum class Kind { Blank, Comment, Section, KeyValue, Other };
    Kind kind=Kind::Blank;
    std::string raw,section,key,value;
    int worker_scope=0;
};
inline bool read_ini(const std::string &path,std::vector<IniLine> &lines) {
    std::ifstream in(path,std::ios::binary);
    if(!in)return false;
    lines.clear();std::string raw,section;int scope=0;bool first=true;
    while(std::getline(in,raw)) {
        if(!raw.empty()&&raw.back()=='\r')raw.pop_back();
        if(first&&raw.compare(0,3,"\xef\xbb\xbf")==0)raw.erase(0,3);
        first=false;
        IniLine l;l.raw=raw;l.section=section;l.worker_scope=scope;
        const auto t=trim(raw);
        if(t.empty())l.kind=IniLine::Kind::Blank;
        else if(t[0]=='#'||t[0]==';')l.kind=IniLine::Kind::Comment;
        else if(t.size()>=3&&t.front()=='['&&t.back()==']') {
            l.kind=IniLine::Kind::Section;l.section=trim(t.substr(1,t.size()-2));
            if(l.section.empty()){l.kind=IniLine::Kind::Other;l.section=section;}
            else section=l.section;
            const int w=worker_header(t);if(w)scope=w;
            l.worker_scope=scope;
        } else {
            const auto eq=t.find('=');
            l.kind=eq==t.npos?IniLine::Kind::Other:IniLine::Kind::KeyValue;
            if(eq!=t.npos){l.key=trim(t.substr(0,eq));l.value=trim(t.substr(eq+1));}
        }
        lines.push_back(std::move(l));
    }
    return !in.bad();
}
using Entries=std::vector<std::pair<std::string,std::string>>;
inline void replace_entry(Entries &values,std::string key,const std::string &value) {
    values.erase(std::remove_if(values.begin(),values.end(),[&](const auto &p){return p.first==key;}),values.end());
    values.emplace_back(std::move(key),value);
}
inline Entries cli_entries(const std::vector<IniLine> &lines,int worker,bool insensitive=false) {
    Entries values;
    // Last occurrence wins within each layer; a worker layer always follows globals.
    for(int pass=0;pass<2;++pass)for(const auto &l:lines) {
        if(l.kind!=IniLine::Kind::KeyValue||l.key.empty())continue;
        if(pass==0?l.worker_scope!=0:(worker<=0||l.worker_scope!=worker))continue;
        replace_entry(values,insensitive?lower(l.key):l.key,l.value);
    }
    return values;
}
inline Entries section_entries(const std::vector<IniLine> &lines,const std::string &section) {
    Entries values;
    for(const auto &l:lines)if(l.kind==IniLine::Kind::KeyValue&&l.section==section)
        replace_entry(values,l.key,l.value);
    return values;
}

// Exact decimal/scientific uint64 input, shared by Stage2 INI and CLI. Decimal
// string scaling avoids a floating-point round trip and requires no GMP linkage.
inline uint64_t unsigned_integer(std::string s,const char *label) {
    s=trim(s);const auto bad=[&](){return std::runtime_error(std::string("invalid ")+label+": "+s);};
    if(s.empty()||s.size()>128||s[0]=='-')throw bad();
    if(s[0]=='+')s.erase(0,1);
    int exponent=0;const auto e=s.find_first_of("eE");
    if(e!=s.npos){const auto es=s.substr(e+1);size_t used=0;
        try{exponent=std::stoi(es,&used);}catch(...){throw bad();}
        if(used!=es.size()||exponent < -1000||exponent>1000)throw bad();s.resize(e);}
    std::string digits;bool dot=false;int fractional=0;
    for(char c:s){if(c=='.'&&!dot){dot=true;continue;}if(c<'0'||c>'9')throw bad();digits+=c;if(dot)++fractional;}
    if(digits.empty())throw bad();
    const auto nonzero=digits.find_first_not_of('0');if(nonzero==digits.npos)return 0;
    digits.erase(0,nonzero);const int power=exponent-fractional;
    if(power<0){const auto count=static_cast<size_t>(-power);
        if(count>=digits.size()||digits.find_first_not_of('0',digits.size()-count)!=digits.npos)throw bad();
        digits.resize(digits.size()-count);
    } else {if(digits.size()+static_cast<size_t>(power)>20)throw bad();digits.append(static_cast<size_t>(power),'0');}
    try{size_t used=0;const auto n=std::stoull(digits,&used);if(used!=digits.size())throw bad();return n;}
    catch(...){throw bad();}
}
inline double positive(const std::string &s,const char *key) {
    size_t used=0;const double n=std::stod(s,&used);
    if(used!=s.size()||!std::isfinite(n)||n<=0)throw std::runtime_error(std::string(key)+" must be finite and positive");
    return n;
}
inline bool boolean(const std::string &s,const char *key) {
    const auto v=lower(s);
    if(v=="1"||v=="true"||v=="yes"||v=="on")return true;
    if(v=="0"||v=="false"||v=="no"||v=="off")return false;
    throw std::runtime_error(std::string(key)+" must be true or false");
}
inline uint64_t bounded_integer(const std::string &s,const char *key,uint64_t low,uint64_t high) {
    const auto n=unsigned_integer(s,key);
    if(n<low||n>high)throw std::runtime_error(std::string(key)+" outside ["+std::to_string(low)+","+std::to_string(high)+"]");
    return n;
}
inline int log_level(const std::string &s) {
    const auto v=lower(s);const char *names[]={"quiet","curve","phases","batches","debug"};
    for(int i=0;i<5;++i)if(v==names[i]||v==std::to_string(i))return i;
    throw std::runtime_error("log level must be quiet|curve|phases|batches|debug or 0..4");
}
inline void legacy_assign(std::string &out,const std::string &v){out=v;}
inline void legacy_assign(int &out,const std::string &v){try{out=std::stoi(v);}catch(...) {}}
inline void legacy_assign(uint32_t &out,const std::string &v){try{const auto n=std::stoull(v);if(n<=UINT32_MAX)out=static_cast<uint32_t>(n);}catch(...) {}}
inline void legacy_assign(uint64_t &out,const std::string &v){try{out=std::stoull(v);}catch(...) {}}
inline void legacy_assign(double &out,const std::string &v){try{out=std::stod(v);}catch(...) {}}
inline void legacy_assign(bool &out,const std::string &v){try{out=boolean(v,"boolean");}catch(...) {}}
inline int gui_integer(const std::string &v,int fallback,int low,int high) {
    if(v.empty())return fallback;
    try{size_t used=0;const auto n=std::stoll(v,&used);if(used!=v.size())return fallback;
        return static_cast<int>(std::max<int64_t>(low,std::min<int64_t>(high,n)));}catch(...){return fallback;}
}
inline int64_t gui_long(const std::string &v,int64_t fallback,int64_t low,int64_t high) {
    try{return (std::max)(low,(std::min)(high,static_cast<int64_t>(std::stoll(v))));}catch(...){return fallback;}
}
inline float gui_font(const std::string &v,float low,float high) {
    if(v.empty()||v=="auto")return 0;
    try{const auto n=std::stof(v);return !std::isfinite(n)||n<low?0:(std::min)(n,high);}catch(...){return 0;}
}
inline void gui_rect(std::array<int,4> &out,const std::string &v) {
    std::array<int,4> parsed{};size_t pos=0;
    for(size_t i=0;i<parsed.size();++i) {
        const auto comma=v.find(',',pos);const auto token=v.substr(pos,comma==v.npos?v.npos:comma-pos);
        char *end=nullptr;const long n=std::strtol(token.c_str(),&end,10);
        if(token.empty()||!end||*end)return;parsed[i]=static_cast<int>(n);
        if(comma==v.npos){if(i+1==parsed.size())out=parsed;return;}pos=comma+1;
    }
}
template<class C> struct Binding {
    const char *key;
    void (*assign)(C &,const std::string &);
    bool deprecated;
    const char *fallback_for;
};
template<class C,size_t N> inline void apply(C &cfg,const Entries &entries,const Binding<C> (&bindings)[N]) {
    bool warned=false;
    for(const auto &kv:entries)for(const auto &b:bindings)if(kv.first==b.key) {
        if(b.fallback_for&&std::any_of(entries.begin(),entries.end(),[&](const auto &p){return p.first==b.fallback_for;}))break;
        b.assign(cfg,kv.second);
        if(b.deprecated&&!warned){std::fprintf(stderr,"[ecm] NOTE: legacy INI key '%s'; see ECM_INI_REFERENCE.md\n",b.key);warned=true;}
        break;
    }
}
inline std::string worker_file(const std::string &name,int worker) {
    if(worker==1)return name;
    const auto dot=name.find_last_of('.');
    return name.substr(0,dot)+"_"+std::to_string(worker)+(dot==name.npos?std::string():name.substr(dot));
}
} // namespace ecm_config
