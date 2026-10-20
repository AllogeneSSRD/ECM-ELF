#pragma once
#include <string>
#include <cstdio>
namespace stage2_tune {
inline std::string quote(const char *s) {
    std::string out="\"";
    for(;*s;++s) {
        const unsigned char c=(unsigned char)*s;
        if(c=='"' || c=='\\') { out+='\\';out+=(char)c; }
        else if(c<32) { char b[7];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b; }
        else out+=(char)c;
    }
    return out+'"';
}
}
