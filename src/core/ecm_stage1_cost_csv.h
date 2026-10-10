#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecm_stage2 { namespace stage1_cost {
struct Tier { unsigned bits=0,tpi=0; };
inline Tier tier(unsigned target_bits) {
    // PARAM0 dispatch grid; six carry bits are required by cgbn_stage1.cu.
    const unsigned grid[]={128,192,256,384,512,768,1024,1280,1536,1792,2048,
        2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192,
        9216,10240,11264,12288,13312,14336,15360,16384};
    if(target_bits<2 || target_bits>16378)throw std::runtime_error("Stage1 target exceeds supported PARAM0 dispatch");
    for(unsigned bits:grid)if(bits>=target_bits+6)return {bits,bits<=512?4u:bits<=2048?8u:bits<=8192?16u:32u};
    throw std::runtime_error("no Stage1 container fits target");
}
inline std::string trim(std::string s) {
    const auto a=s.find_first_not_of(" \t\r\n");
    return a==s.npos?std::string{}:s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
inline std::vector<std::string> row(const std::string &s) {
    std::vector<std::string> out;std::string cell;bool quoted=false,closed=false;
    for(size_t i=0;i<s.size();++i) {
        const char c=s[i];
        if(c=='"') {
            if(quoted && i+1<s.size() && s[i+1]=='"'){cell+='"';++i;}
            else if(quoted){quoted=false;closed=true;}
            else if(!closed && trim(cell).empty()){cell.clear();quoted=true;}
            else throw std::runtime_error("invalid CSV quote");
        } else if(c==',' && !quoted){out.push_back(trim(cell));cell.clear();closed=false;}
        else {if(closed && c!=' ' && c!='\t' && c!='\r')throw std::runtime_error("data after CSV quote");cell+=c;}
    }
    if(quoted)throw std::runtime_error("unterminated CSV quote");
    out.push_back(trim(cell));return out;
}
inline double number(const std::string &s) {
    size_t end=0;const auto v=std::stod(s,&end);
    if(end!=s.size() || !std::isfinite(v) || v<=0)throw std::runtime_error("CSV cost must be finite and positive");
    return v;
}
inline unsigned integer(const std::string &s) {
    const double v=number(s);
    if(v!=std::floor(v) || v>1048576)throw std::runtime_error("invalid CSV integer");return unsigned(v);
}
struct Anchor {
    unsigned container_bits=0,tpi=0,curves=0,tpb=128,target_bits=0;
    double b1=0,mhz=0,seconds=0;
    std::string exponent;
};
struct Prediction {
    double seconds=0,reference_mhz=0;
    Tier shape;
    unsigned reference_curves=0;
    std::string source;
    bool crosses_tpi=false;
};
class Table {
public:
    std::vector<Anchor> anchors;
    static Table load(const std::filesystem::path &path) {
        std::error_code error;const auto bytes=std::filesystem::file_size(path,error);
        if(error || bytes>1048576)throw std::runtime_error("Stage1 CSV missing or exceeds 1MiB");
        std::ifstream in(path,std::ios::binary);Table table;std::string line;
        std::map<std::string,size_t> header;bool first=true;
        while(std::getline(in,line)) {
            if(first && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first=false;
            line=trim(line);if(line.empty() || line[0]=='#')continue;
            const auto cells=row(line);
            if(header.empty()) {
                for(size_t i=0;i<cells.size();++i)if(cells[i].empty() || !header.emplace(cells[i],i).second)
                    throw std::runtime_error("duplicate/empty Stage1 CSV column");
                for(const char *key:{"b1","mhz","tpi","curves","seconds_per_curve"})
                    if(!header.count(key))throw std::runtime_error(std::string("Stage1 CSV missing column: ")+key);
                if(!header.count("container_bits") && !header.count("target_bits"))
                    throw std::runtime_error("Stage1 CSV requires container_bits or target_bits");
                continue;
            }
            if(cells.size()!=header.size())throw std::runtime_error("Stage1 CSV row width mismatch");
            auto get=[&](const char *key){auto i=header.find(key);return i==header.end()?std::string{}:cells[i->second];};
            Anchor a;a.b1=number(get("b1"));a.mhz=number(get("mhz"));
            a.tpi=integer(get("tpi"));a.curves=integer(get("curves"));
            if(!get("tpb").empty())a.tpb=integer(get("tpb"));
            if(!get("target_bits").empty())a.target_bits=integer(get("target_bits"));
            if(!get("container_bits").empty())a.container_bits=integer(get("container_bits"));
            else if(a.target_bits)a.container_bits=tier(a.target_bits).bits;
            if(a.container_bits<128 || a.container_bits>16384 ||
               tier(a.container_bits-6).bits!=a.container_bits ||
               a.tpi!=(a.container_bits<=512?4u:a.container_bits<=2048?8u:a.container_bits<=8192?16u:32u) ||
               a.tpb%a.tpi || a.b1<2 || a.b1!=std::floor(a.b1) ||
               (a.target_bits && tier(a.target_bits).bits!=a.container_bits))
                throw std::runtime_error("Stage1 CSV container/TPI inconsistency");
            a.exponent=get("exponent");
            if(!a.exponent.empty() && a.exponent!="lcm" && a.exponent!="choose12")
                throw std::runtime_error("Stage1 CSV exponent must be lcm, choose12 or empty");
            if(get("seconds_per_curve").empty())continue; // Explicit missing measurements.
            a.seconds=number(get("seconds_per_curve"));
            if(table.anchors.size()>=4096)throw std::runtime_error("too many Stage1 CSV anchors");
            for(const auto &old:table.anchors)if(old.container_bits==a.container_bits && old.tpi==a.tpi &&
                old.b1==a.b1 && old.mhz==a.mhz && old.curves==a.curves && old.exponent==a.exponent)
                throw std::runtime_error("duplicate Stage1 CSV measurement scope");
            table.anchors.push_back(std::move(a));
        }
        if(!in.eof() || table.anchors.empty())throw std::runtime_error("Stage1 CSV contains no usable measurements");
        return table;
    }
    Prediction predict(unsigned target_bits,double b1,double mhz=0,const std::string &exponent="lcm")const {
        if(!std::isfinite(b1) || b1<2 || !std::isfinite(mhz) || mhz<0)throw std::runtime_error("invalid Stage1 cost request");
        Prediction out;out.shape=tier(target_bits);
        std::vector<const Anchor*> usable,same;
        for(const auto &a:anchors)if(a.exponent.empty() || a.exponent==exponent) {
            usable.push_back(&a);if(a.tpi==out.shape.tpi)same.push_back(&a);
        }
        if(usable.empty())throw std::runtime_error("Stage1 CSV has no compatible exponent measurements");
        out.crosses_tpi=same.empty();if(!same.empty())usable=std::move(same);
        // A CSV describes one submitted batch geometry per TPI. Mixing curve
        // counts silently would turn amortized s/curve into incomparable costs.
        std::map<unsigned,std::pair<unsigned,unsigned>> geometry;
        for(auto a:usable) {
            const auto shape=std::make_pair(a->curves,a->tpb);
            auto inserted=geometry.emplace(a->tpi,shape);
            if(!inserted.second && inserted.first->second!=shape)
                throw std::runtime_error("Stage1 CSV mixes batch geometry within one TPI");
        }
        std::sort(usable.begin(),usable.end(),[](const Anchor *a,const Anchor *b){
            if(a->container_bits!=b->container_bits)return a->container_bits<b->container_bits;
            return a->seconds/a->b1*a->mhz < b->seconds/b->b1*b->mhz;
        });
        // Select the median normalized coefficient when multiple measurements
        // describe the same container rather than depending on input order.
        std::vector<const Anchor*> reduced;
        for(size_t i=0;i<usable.size();) {
            size_t j=i+1;while(j<usable.size() && usable[j]->container_bits==usable[i]->container_bits)++j;
            reduced.push_back(usable[i+(j-i)/2]);i=j;
        }
        usable=std::move(reduced);
        const Anchor *lo=usable.front(),*hi=usable.back();
        for(auto a:usable){if(a->container_bits<=out.shape.bits)lo=a;if(a->container_bits>=out.shape.bits){hi=a;break;}}
        if(out.shape.bits<usable.front()->container_bits)lo=hi=usable.front();
        if(out.shape.bits>usable.back()->container_bits)lo=hi=usable.back();
        const double frequency=mhz>0?mhz:lo->mhz;
        auto coefficient=[&](const Anchor *a){return a->seconds/a->b1*a->mhz/(double(a->container_bits)*a->container_bits);};
        double k=coefficient(lo);
        if(lo->container_bits!=hi->container_bits) {
            const double f=double(out.shape.bits-lo->container_bits)/(hi->container_bits-lo->container_bits);
            k+=(coefficient(hi)-k)*f;
        }
        out.seconds=k*b1/frequency*double(out.shape.bits)*out.shape.bits;
        out.reference_mhz=frequency;out.reference_curves=lo->curves;
        const bool inside=out.shape.bits>=usable.front()->container_bits && out.shape.bits<=usable.back()->container_bits;
        out.source=out.crosses_tpi || !inside?"extrapolated":lo->container_bits!=hi->container_bits?"interpolated":"scaled_measurement";
        if(!(out.seconds>0) || !std::isfinite(out.seconds))throw std::runtime_error("non-finite Stage1 cost prediction");
        return out;
    }
};
} }
