#pragma once
#include "io/files.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace ecm_opencl::stage2 {
// Closed TOML subset owned by this tuner. Reject unknown formats rather than
// replacing a user-selected CUDA profile (or a partially damaged file).
struct PointSample { unsigned count=0; double mean=0; };
using PointSamples=std::map<std::string,PointSample>;
inline PointSamples read_point_samples(const std::filesystem::path& path) {
    PointSamples rows;
    if(!std::filesystem::exists(path))return rows;
    std::istringstream input(ecm_io::read_file(path));
    std::string line;
    std::getline(input,line);
    if(!line.empty() && line.back()=='\r')line.pop_back();
    if(line.compare(0,3,"\xEF\xBB\xBF")==0)line.erase(0,3);
    if(line!="opencl_points_format = 1")
        throw std::runtime_error("not a native OpenCL point timing profile");
    while(std::getline(input,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        if(line.empty() || line.front()=='#')continue;
        std::istringstream in(line);
        std::string key; char eq,left,comma,right;PointSample s;
        if(!(in>>key>>eq>>left>>s.count>>comma>>s.mean>>right) || eq!='=' || left!='[' ||
            comma!=',' || right!=']' || !(in>>std::ws).eof() || key.compare(0,2,"g_") ||
            key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_.")!=std::string::npos ||
            !s.count || s.count>1000 || !(s.mean>0) || !std::isfinite(s.mean) || !rows.emplace(key,s).second)
            throw std::runtime_error("invalid OpenCL point timing row");
    }
    return rows;
}

struct PointTrial {
    std::filesystem::path path;
    std::string key,plan;
    unsigned repeats=1;
    bool collect=false;

    // Baseline first; interleave candidates across productive curves. A trial
    // budget smaller than the complete grid admits only fully comparable arms.
    void select(std::vector<std::string> candidates,unsigned budget) {
        if(!repeats || repeats>1000 || budget>1000)throw std::runtime_error("invalid OpenCL tune budget");
        candidates.resize(std::min(candidates.size(),size_t(budget/repeats)));
        if(candidates.size()<2)return;
        ecm_io::FileLock lock(path.u8string());
        auto rows=read_point_samples(path);
        unsigned smallest=repeats;
        double best=std::numeric_limits<double>::infinity();
        for(const auto& candidate:candidates){
            const auto s=rows[key+"."+candidate];
            if(s.count<smallest){smallest=s.count;plan=candidate;collect=true;}
            else if(!collect && s.count>=repeats && s.mean<best){best=s.mean;plan=candidate;}
        }
    }
    void record(double seconds)const {
        if(!collect || !(seconds>0) || !std::isfinite(seconds))return;
        ecm_io::FileLock lock(path.u8string());
        auto rows=read_point_samples(path);
        auto& s=rows[key+"."+plan];
        if(s.count>=repeats)return; // Concurrent workers can complete the same trial.
        s.mean+=(seconds-s.mean)/++s.count;
        std::ostringstream out;
        out<<"opencl_points_format = 1\n# device/host, point and arithmetic limbs, D, giants, carrier, budgets, plan = [curves, mean engine seconds]\n"<<std::setprecision(17);
        for(const auto& [name,row]:rows)out<<name<<" = ["<<row.count<<", "<<row.mean<<"]\n";
        ecm_io::write_atomic(path,out.str());
    }
};
}
