#pragma once
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace ecm_workers {
inline std::string trim(const std::string &s) {
    const auto first=s.find_first_not_of(" \t\r\n");
    return first==s.npos?std::string{}:s.substr(first,s.find_last_not_of(" \t\r\n")-first+1);
}
// Shared list/range syntax. Callers decide whether out-of-range IDs are fatal.
inline bool parse_list(const std::string &spec,int maximum,std::vector<int> &workers,int &ignored) {
    workers.clear();ignored=0;
    if(maximum<1)return false;
    std::vector<unsigned char> seen(size_t(maximum)+1);
    const auto number=[](const std::string &text,long long &value) {
        const auto s=trim(text);if(s.empty())return false;
        char *end=nullptr;errno=0;value=std::strtoll(s.c_str(),&end,10);
        return errno!=ERANGE && end!=s.c_str() && !*end;
    };
    std::istringstream input(spec);std::string item;
    while(std::getline(input,item,',')) {
        item=trim(item);if(item.empty())continue;
        const auto dash=item.find('-');long long low=0,high=0;
        if(dash==item.npos){if(!number(item,low))return false;high=low;}
        else if(!number(item.substr(0,dash),low)||!number(item.substr(dash+1),high))return false;
        if(high<low)std::swap(low,high);
        // Bound iteration even for a very large user-supplied range.
        long double outside=0;
        if(low<1)outside+=static_cast<long double>((std::min)(high,0LL))-low+1;
        if(high>maximum)outside+=static_cast<long double>(high)-(std::max)(low,static_cast<long long>(maximum)+1)+1;
        ignored=int((std::min)(static_cast<long double>(INT_MAX),ignored+outside));
        low=(std::max)(1LL,low);high=(std::min)(static_cast<long long>(maximum),high);
        for(auto id=low;id<=high;++id)if(!seen[size_t(id)]) {
            seen[size_t(id)]=1;workers.push_back(int(id));
        }
    }
    return true;
}
}
