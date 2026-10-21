#include "results.h"
#include "../core/ecm_result_json.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

namespace ecmgui {
namespace {
namespace fs=std::filesystem;
namespace json=ecm_results::json;
std::string normalized_path(const std::string &path) {
    std::error_code error;
    auto absolute=fs::absolute(fs::u8path(path),error);
    if(error)throw std::runtime_error("cannot resolve Result path: "+path);
    auto resolved=fs::weakly_canonical(absolute,error);
    return (error?absolute.lexically_normal():resolved).u8string();
}
std::string lower(std::string text) {
    for(auto &c:text)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}
std::string path_key(const std::string &path) {
#ifdef _WIN32
    return lower(path);
#else
    return path;
#endif
}
bool unsigned_decimal(const std::string &text) {
    return !text.empty() && text.find_first_not_of("0123456789")==text.npos;
}
std::uint64_t number(const json::Value &value) {
    const auto text=value.text();
    if(!unsigned_decimal(text))throw std::runtime_error("missing or invalid numeric field");
    return std::stoull(text);
}
ResultRecord parse(const std::string &raw,std::size_t source,std::size_t line,const std::string &path) {
    const auto value=json::parse(raw);
    ResultRecord row;row.status=value.get("status").text();
    if(value.type!='o' || (row.status!="F" && row.status!="NF") || value.get("worktype").text()!="ECM")
        throw std::runtime_error("expected an ECM F/NF submission row");
    const auto b1=number(value.get("b1"));
    if(b1<2 || number(value.get("curves"))==0 || value.get("program").get("name").text().empty() ||
       value.get("program").get("version").text().empty() || value.get("timestamp").text().empty())
        throw std::runtime_error("incomplete submission row");
    const auto &checksum=value.get("checksum");
    const auto crc=checksum.get("checksum").text();
    if(checksum.get("version").text()!="1" || crc.size()!=8 || crc.find_first_not_of("0123456789abcdefABCDEF")!=crc.npos)
        throw std::runtime_error("missing submission checksum");
    if(value.has("exponent"))row.target="M"+value.get("exponent").text();
    else if(value.has("k") && value.has("b") && value.has("n") && value.has("c")) {
        const auto c=value.get("c").text();
        row.target=value.get("k").text()+"*"+value.get("b").text()+"^"+value.get("n").text()+(c.empty()||c[0]=='-'?"":"+")+c;
    } else if(value.has("N")) {
        const auto n=value.get("N").text();
        row.target=n.size()>64?"N ("+std::to_string(n.size())+" digits)":n;
    }
    if(row.target.empty())throw std::runtime_error("missing submission target");
    if(row.status=="NF") {
        const auto b2=number(value.get("b2"));
        if(b2<b1)throw std::runtime_error("invalid NF bounds");
        row.stage=b2==b1?1:2;
    } else {
        const auto stage=number(value.get("stage"));
        const auto &factors=value.get("factors");
        if((stage!=1 && stage!=2) || factors.type!='a' || factors.items.empty())
            throw std::runtime_error("incomplete factor submission");
        for(const auto &factor:factors.items)if(!unsigned_decimal(factor.text()))
            throw std::runtime_error("invalid factor field");
        row.stage=static_cast<int>(stage);row.factors=json::joined(factors);
    }
    row.b1=value.get("b1").text();row.b2=value.get("b2").text();row.curves=value.get("curves").text();
    row.model=value.has("Edwards")?"Edwards":value.has("GMP-ECM-param3")?"Montgomery param3":"Montgomery";
    if(!value.has("Edwards") && value.has("param"))row.model+=" param"+value.get("param").text();
    row.timestamp=value.get("timestamp").text();row.raw=raw;row.source=source;row.line=line;
    row.id=path+'\n'+std::to_string(line)+'\n'+raw;
    return row;
}
}
std::vector<std::string> result_paths(const std::vector<ecm_config::TomlLine> &lines,
                                     const std::string &config_path,int workers) {
    std::vector<std::string> paths;std::set<std::string> seen;
    const auto base=fs::u8path(normalized_path(config_path)).parent_path();
    for(int worker=1;worker<=workers;++worker) {
        const auto entries=ecm_config::cli_entries(lines,worker);
        const auto first=ecm_config::read_stage1(entries);
        const auto second=ecm_config::read_stage2(entries);
        for(const auto &name:{first.has_submission?first.submission:ecm_config::worker_file(first.submission,worker),
                             second.has_submission?second.submission:ecm_config::worker_file(second.submission,worker)}) {
            if(name.empty())throw std::runtime_error("empty submission path");
            const auto path=normalized_path((base/fs::u8path(name)).u8string());
            if(seen.insert(path_key(path)).second)paths.push_back(path);
        }
    }
    return paths;
}
bool ResultFilter::matches(const ResultRecord &row) const {
    return (status.empty() || status==row.status) && (!stage || stage==row.stage) &&
        (source<0 || static_cast<std::size_t>(source)==row.source) &&
        (search.empty() || lower(row.raw).find(lower(search))!=std::string::npos);
}
void ResultStore::set_paths(const std::vector<std::string> &paths) {
    files_.clear();cache_.clear();rows_.clear();
    for(const auto &path:paths)add_path(path);
}
void ResultStore::add_path(const std::string &path) {
    const auto normalized=normalized_path(path),key=path_key(normalized);
    if(std::none_of(files_.begin(),files_.end(),[&](const auto &file){return path_key(file.path)==key;})) {
        files_.push_back({normalized});cache_.emplace_back();
    }
}
bool ResultStore::refresh(bool force) {
    bool changed=false;
    for(std::size_t index=0;index<files_.size();++index) {
        auto &file=files_[index];auto &cache=cache_[index];std::error_code error;
        const bool exists=fs::exists(fs::u8path(file.path),error);
        std::uintmax_t size=0;fs::file_time_type modified{};
        if(!error && exists) {
            size=fs::file_size(fs::u8path(file.path),error);
            if(!error)modified=fs::last_write_time(fs::u8path(file.path),error);
        }
        if(!force && cache.loaded && !error && exists==cache.exists &&
           (!exists || (size==cache.size && modified==cache.modified)))continue;
        changed=true;cache.loaded=true;cache.exists=exists;cache.size=size;cache.modified=modified;
        cache.rows.clear();file.exists=exists;file.error.clear();file.invalid_lines=0;file.pending_lines=0;
        if(error){file.error=error.message();cache.loaded=false;continue;}
        if(!exists)continue;
        std::ifstream in(fs::u8path(file.path),std::ios::binary);
        if(!in){file.error="cannot read Result file";cache.loaded=false;continue;}
        std::string raw;std::size_t line=0;
        while(std::getline(in,raw)) {
            ++line;
            if(!raw.empty() && raw.back()=='\r')raw.pop_back();
            if(line==1 && raw.compare(0,3,"\xef\xbb\xbf")==0)raw.erase(0,3);
            if(raw.find_first_not_of(" \t")==raw.npos)continue;
            if(in.eof()) {
                try{(void)json::parse(raw);}
                catch(const std::exception &){++file.pending_lines;continue;}
            }
            try{cache.rows.push_back(parse(raw,index,line,file.path));}
            catch(const std::exception &e) {
                ++file.invalid_lines;
                if(file.error.empty())file.error="line "+std::to_string(line)+": "+e.what();
            }
        }
        if(in.bad()){file.error="cannot finish reading Result file";cache.loaded=false;cache.rows.clear();}
    }
    if(changed) {
        rows_.clear();
        for(const auto &cache:cache_)rows_.insert(rows_.end(),cache.rows.begin(),cache.rows.end());
    }
    return changed;
}
std::vector<std::size_t> ResultStore::filtered(const ResultFilter &filter) const {
    std::vector<std::size_t> result;
    for(std::size_t i=0;i<rows_.size();++i)if(filter.matches(rows_[i]))result.push_back(i);
    return result;
}
std::string ResultStore::copy_text(const std::vector<std::size_t> &indices) const {
    std::string text;
    for(const auto index:indices)text+=rows_.at(index).raw+'\n';
    return text;
}
} // namespace ecmgui
