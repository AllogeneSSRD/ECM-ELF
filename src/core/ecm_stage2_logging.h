#pragma once
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cctype>
#include <chrono>
#include <filesystem>

// One curve worker owns the engine. Verbosity never controls arithmetic checks,
// result publication, or stderr; quiet only suppresses progress/statistics.
namespace stage2_log {
enum Level { quiet=0, curve=1, phases=2, batches=3, debug=4 };
inline int level=batches;
inline FILE *debug_file=nullptr;
inline void open_debug(const std::filesystem::path &path) {
#ifdef _WIN32
    debug_file=_wfopen(path.c_str(),L"ab");
#else
    debug_file=std::fopen(path.c_str(),"ab");
#endif
    if(!debug_file)throw std::runtime_error("cannot open Stage2 debug log");
    std::setvbuf(debug_file,nullptr,_IONBF,0);
}
inline bool enabled(int required) { return level>=required || (required==debug && debug_file); }
inline int parse(std::string value) {
    for(char &c:value)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const char *names[]={"quiet","curve","phases","batches","debug"};
    for(int i=0;i<=debug;++i)
        if(value==names[i] || value==std::to_string(i))return i;
    throw std::runtime_error("log level must be quiet|curve|phases|batches|debug or 0..4");
}
inline void print(int required,const char *format,...) {
    if(!enabled(required))return;
    FILE *destination=required==debug && debug_file?debug_file:stdout;
    va_list args;va_start(args,format);const int n=std::vfprintf(destination,format,args);va_end(args);
    if(n<0)throw std::runtime_error("Stage2 log write failed");
    std::fflush(destination);
}
// Explicit milestones are separate from diagnostic statistics. The queue parent
// forwards these to the console while retaining batch statistics in the log.
inline void phase(const char *name) {
    static const auto start=std::chrono::steady_clock::now();
    static auto previous=start;
    const auto now=std::chrono::steady_clock::now();
    print(curve,"stage2_phase: %s previous=%.2f s elapsed=%.2f s\n",name,
          std::chrono::duration<double>(now-previous).count(),
          std::chrono::duration<double>(now-start).count());
    previous=now;
}
}
