#pragma once
#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ecm_stage2 { namespace tune {
struct Effort {int first,last,repeats;};
inline Effort effort(int level) {
    if(level<1 || level>10)throw std::runtime_error("tune-level must be 1..10");
    return {16,20+std::min(level-1,7),2*level*level+1};
}
// Converts the engine's flat JSON callback records to named TOML tables.
// Values remain JSON literals: the emitted subset (strings, numbers, booleans
// and numeric arrays) is also TOML. This is not a general JSON reader.
inline std::map<std::string,std::string> fields(const std::string &s) {
    size_t i=0;
    auto space=[&](){while(i<s.size() && std::isspace((unsigned char)s[i]))++i;};
    auto expect=[&](char c){space();if(i>=s.size() || s[i++]!=c)throw std::runtime_error("invalid tune callback");};
    auto string_end=[&](){
        expect('"');
        while(i<s.size()) {
            const char c=s[i++];
            if(c=='"')return;
            if((unsigned char)c<32)throw std::runtime_error("control character in tune string");
            if(c=='\\') {if(i>=s.size())break;++i;}
        }
        throw std::runtime_error("unterminated tune string");
    };
    std::map<std::string,std::string> out;
    expect('{');space();
    while(i<s.size() && s[i]!='}') {
        const size_t begin=i;string_end();
        const auto key=s.substr(begin+1,i-begin-2);
        if(key.empty() || key.find_first_not_of("abcdefghijklmnopqrstuvwxyz_0123456789")!=key.npos)
            throw std::runtime_error("invalid tune field name");
        expect(':');space();const size_t value_begin=i;
        if(i<s.size() && s[i]=='"')string_end();
        else if(i<s.size() && s[i]=='[') {
            ++i;
            while(i<s.size() && s[i]!=']') {
                if(std::string("0123456789.eE+-, \t").find(s[i])==std::string::npos)
                    throw std::runtime_error("tune arrays must be numeric");
                ++i;
            }
            expect(']');
        } else {
            while(i<s.size() && s[i]!=',' && s[i]!='}' && !std::isspace((unsigned char)s[i]))++i;
            const auto v=s.substr(value_begin,i-value_begin);
            if(v.empty() || (v!="true" && v!="false" && v.find_first_not_of("0123456789.eE+-")!=v.npos))
                throw std::runtime_error("unsupported tune value");
        }
        if(!out.emplace(key,s.substr(value_begin,i-value_begin)).second)
            throw std::runtime_error("duplicate tune field");
        space();if(i<s.size() && s[i]=='}')break;
        expect(',');space();
        if(i<s.size() && s[i]=='}')throw std::runtime_error("trailing tune comma");
    }
    expect('}');space();if(i!=s.size())throw std::runtime_error("data after tune callback");
    return out;
}
inline std::string table(const std::string &json) {
    const auto values=fields(json);
    auto required=[&](const char *key)->const std::string& {
        const auto it=values.find(key);if(it==values.end())throw std::runtime_error("missing tune field");return it->second;
    };
    const auto &type=required("type");std::string section;
    if(type=="\"device\"")section="device";
    else if(type=="\"complete\"")section="summary";
    else if(type=="\"sample\"") {
        const auto &length=required("length");
        if(length.empty() || length.find_first_not_of("0123456789")!=length.npos)
            throw std::runtime_error("invalid tune length");
        section="ntt.length_"+length;
    } else throw std::runtime_error("unsupported tune record");
    std::ostringstream out;out<<'\n'<<'['<<section<<"]\n";
    for(const auto &entry:values)if(entry.first!="type" && entry.first!="device_index")
        out<<entry.first<<" = "<<entry.second<<'\n';
    return out.str();
}
} }
