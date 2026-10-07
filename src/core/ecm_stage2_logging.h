#pragma once
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cctype>

// One curve worker owns the engine. Verbosity never controls arithmetic checks,
// result publication, or stderr; quiet only suppresses progress/statistics.
namespace stage2_log {
enum Level { quiet=0, curve=1, phases=2, batches=3, debug=4 };
inline int level=batches;
inline bool enabled(int required) { return level>=required; }
inline int parse(std::string value) {
    for(char &c:value)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const char *names[]={"quiet","curve","phases","batches","debug"};
    for(int i=0;i<=debug;++i)
        if(value==names[i] || value==std::to_string(i))return i;
    throw std::runtime_error("log level must be quiet|curve|phases|batches|debug or 0..4");
}
inline void print(int required,const char *format,...) {
    if(!enabled(required))return;
    va_list args;va_start(args,format);std::vfprintf(stdout,format,args);va_end(args);
}
}
