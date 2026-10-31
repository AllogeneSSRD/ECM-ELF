#pragma once
#include <cstdint>
#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "d_catalog.h"

namespace ecm_stage2 { namespace tune { namespace portable {
// Cost-search hints cross curve workers, never arithmetic or allocations.
// A new parent starts empty; hints are not checkpoint or tune-file data.
struct RuntimePauseHints {
    std::vector<std::string> keys;
    static bool valid_key(const std::string &key) {
        return key.size()==64 && key.find_first_not_of("0123456789abcdef")==key.npos;
    }
    bool contains(const std::string &key)const {
        for(const auto &old:keys)if(old==key)return true;return false;
    }
    void add(const std::string &key) {
        if(valid_key(key) && !contains(key) && keys.size()<2)keys.push_back(key);
    }
    bool assign(const std::string &text) {
        keys.clear();if(text.empty())return true;if(text.size()>129)return false;
        size_t begin=0;
        for(;;) {
            const auto end=text.find(',',begin);const auto key=text.substr(begin,end==text.npos?end:end-begin);
            if(!valid_key(key) || keys.size()==2){keys.clear();return false;}
            add(key);if(end==text.npos)break;begin=end+1;
        }
        return true;
    }
    std::string text()const {
        std::string out;for(const auto &key:keys){if(!out.empty())out+=',';out+=key;}return out;
    }
};
struct RuntimeMemorySnapshot {
    uint64_t free=0,total=0;
    bool valid()const{return total && free<=total;}
    bool operator==(const RuntimeMemorySnapshot &other)const{return free==other.free && total==other.total;}
};
// Failed *memory* admissions are useful only at the exact live free/total
// snapshot. One bit per catalogue D avoids an unbounded command-line journal.
// The enclosing route scope binds N/bounds/device/version/full environment.
struct RuntimeMemoryRejections {
    static constexpr size_t digits=(d_catalog::count+3)/4;
    RuntimeMemorySnapshot memory;
    std::map<uint64_t,std::string> masks;
    static size_t index(uint64_t d) {
        size_t a=0,b=d_catalog::count;
        while(a<b){const auto m=(a+b)/2;if(d_catalog::seeds[m].d<d)a=m+1;else b=m;}
        return a<d_catalog::count && d_catalog::seeds[a].d==d?a:d_catalog::count;
    }
    static bool carrier_valid(uint64_t c){return !c || (c>=2 && c<=16384);}
    static unsigned nibble(char c){return c<='9'?unsigned(c-'0'):unsigned(c-'a'+10);}
    bool refresh(const RuntimeMemorySnapshot &now) {
        if(!now.valid() || !(memory==now)){masks.clear();memory=now;}
        return now.valid();
    }
    bool contains(uint64_t d,uint64_t carrier,const RuntimeMemorySnapshot &now) {
        if(!refresh(now))return false;
        const auto i=index(d);const auto at=masks.find(carrier);
        return i<d_catalog::count && at!=masks.end() && (nibble(at->second[i/4])&(1u<<(i%4)));
    }
    bool put(uint64_t d,uint64_t carrier,const RuntimeMemorySnapshot &before,uint64_t reported_free,
             uint64_t required,const RuntimeMemorySnapshot &after) {
        refresh(after);const auto i=index(d);
        if(!before.valid() || !(before==after) || reported_free!=before.free || required<=reported_free ||
            i==d_catalog::count || !carrier_valid(carrier) || (!masks.count(carrier) && masks.size()==2))return false;
        auto &mask=masks[carrier];if(mask.empty())mask.assign(digits,'0');
        mask[i/4]="0123456789abcdef"[nibble(mask[i/4])|(1u<<(i%4))];return true;
    }
    std::string text()const {
        if(masks.empty() || !memory.valid())return {};
        std::string out=std::to_string(memory.free)+','+std::to_string(memory.total);
        for(const auto &row:masks)out+=';'+std::to_string(row.first)+','+row.second;return out;
    }
    bool assign(const std::string &text) {
        *this={};if(text.empty())return true;
        auto fail=[&](){*this={};return false;};size_t pos=0;
        auto number=[&](uint64_t &v,char delimiter) {
            v=0;const auto begin=pos;
            while(pos<text.size() && text[pos]>='0' && text[pos]<='9') {
                const auto digit=unsigned(text[pos++]-'0');
                if(v>(std::numeric_limits<uint64_t>::max()-digit)/10)return false;v=v*10+digit;
            }
            return pos>begin && pos<text.size() && text[pos++]==delimiter;
        };
        if(text.size()>2*digits+128 || !number(memory.free,',') || !number(memory.total,';') || !memory.valid())return fail();
        while(pos<text.size()) {
            uint64_t carrier=0;if(!number(carrier,',') || !carrier_valid(carrier) || masks.count(carrier) || masks.size()==2)return fail();
            const auto mask=text.substr(pos,digits);pos+=digits;
            if(mask.size()!=digits || mask.find_first_not_of("0123456789abcdef")!=mask.npos ||
                (d_catalog::count%4 && nibble(mask.back())>=(1u<<(d_catalog::count%4))))return fail();
            masks.emplace(carrier,mask);if(pos==text.size())return true;
            if(pos>text.size() || text[pos++]!=';' || pos==text.size())return fail();
        }
        return fail();
    }
};
struct RuntimeRouteHint {
    uint64_t d=0,carrier=0,free=0,total=0,fold=0,frontier=0,chunk=0,chain_min=0,ladder=0;
    std::array<uint64_t,9> values()const{return {d,carrier,free,total,fold,frontier,chunk,chain_min,ladder};}
    bool valid()const {
        return d>=6 && !(d%2) && (carrier==0 || (carrier>=2 && carrier<=16384)) &&
            free && RuntimeMemorySnapshot{free,total}.valid() && fold<=1 && frontier<=1 &&
            (!frontier || fold) && chunk && ladder<=1;
    }
};
// Passed only between children of the same live parent. The owning frontend
// seals the encoded payload and binds scope to N/bounds/device/environment.
// A hint is usable only after checking the current exact free/total snapshot.
struct RuntimeRouteHints {
    static constexpr size_t capacity=32;
    std::string scope_key;std::vector<RuntimeRouteHint> entries;
    RuntimeMemoryRejections rejected;
    void scope(const std::string &key) {
        if(key!=scope_key){entries.clear();rejected={};scope_key=key;}
    }
    bool find(uint64_t d,uint64_t carrier,const RuntimeMemorySnapshot &memory,RuntimeRouteHint &out) {
        if(!RuntimePauseHints::valid_key(scope_key)){entries.clear();return false;}
        for(auto i=entries.begin();i!=entries.end();++i)if(i->d==d && i->carrier==carrier) {
            if(!memory.valid() || i->free!=memory.free || i->total!=memory.total){entries.erase(i);return false;}
            out=*i;return true;
        }
        return false;
    }
    bool put(RuntimeRouteHint entry,bool admitted,const RuntimeMemorySnapshot &after) {
        for(auto i=entries.begin();i!=entries.end();++i)if(i->d==entry.d && i->carrier==entry.carrier){entries.erase(i);break;}
        if(!RuntimePauseHints::valid_key(scope_key) || !admitted || !entry.valid() || !after.valid() || entry.free!=after.free || entry.total!=after.total)return false;
        if(entries.size()>=capacity)entries.erase(entries.begin());entries.push_back(entry);return true;
    }
    std::string text()const {
        if(!RuntimePauseHints::valid_key(scope_key))return {};
        std::string out=scope_key+'|';
        for(const auto &entry:entries) {
            if(out.back()!='|')out+=';';bool first=true;
            for(auto v:entry.values()){if(!first)out+=',';first=false;out+=std::to_string(v);}
        }
        const auto failures=rejected.text();if(!failures.empty())out+='!'+failures;return out;
    }
    bool assign(std::string text) {
        scope_key.clear();entries.clear();rejected={};if(text.empty())return true;
        auto fail=[&](){scope_key.clear();entries.clear();rejected={};return false;};
        if(text.size()>8192)return fail();
        const auto marker=text.find('!');
        if(marker!=text.npos){if(!rejected.assign(text.substr(marker+1)))return fail();text.resize(marker);}
        if(text.size()>8192 || text.size()<65 || text[64]!='|' || !RuntimePauseHints::valid_key(text.substr(0,64)))return fail();
        scope_key=text.substr(0,64);if(text.size()==65)return true;
        size_t start=65;
        while(start<text.size()) {
            const auto end=text.find(';',start);const auto row=text.substr(start,end==text.npos?end:end-start);
            std::array<uint64_t,9> values{};size_t p=0;
            for(size_t field=0;field<values.size();++field) {
                if(p==row.size() || row[p]<'0' || row[p]>'9')return fail();
                while(p<row.size() && row[p]>='0' && row[p]<='9') {
                    const auto digit=unsigned(row[p++]-'0');
                    if(values[field]>(std::numeric_limits<uint64_t>::max()-digit)/10)return fail();
                    values[field]=values[field]*10+digit;
                }
                if(field+1<values.size()){if(p==row.size() || row[p++]!=',')return fail();}
            }
            if(p!=row.size() || entries.size()>=capacity)return fail();
            RuntimeRouteHint entry{values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8]};
            if(!entry.valid())return fail();
            for(const auto &old:entries)if(old.d==entry.d && old.carrier==entry.carrier)return fail();
            entries.push_back(entry);if(end==text.npos)break;start=end+1;if(start==text.size())return fail();
        }
        return true;
    }
};
inline bool runtime_plan_may_reuse(const RuntimeMemorySnapshot &before,uint64_t reported_free,
                                  const RuntimeMemorySnapshot &after) {
    return before.valid() && after.valid() && before==after && before.free==reported_free;
}
// Worker-local successful-plan memo. The caller supplies the full request and
// current environment context. Changed free memory invalidates that entry;
// failed/incomplete admissions are never inserted.
template<class Key,class Value> class RuntimePlanMemo {
    struct Entry {RuntimeMemorySnapshot memory;Value value;};
    std::string context;std::map<Key,Entry> entries;
public:
    static constexpr size_t capacity=32;
    void scope(const std::string &current) {
        if(current!=context){entries.clear();context=current;}
    }
    const Value *find(const Key &key,const RuntimeMemorySnapshot &memory) {
        auto i=entries.find(key);if(i==entries.end())return nullptr;
        if(!memory.valid() || !(memory==i->second.memory)){entries.erase(i);return nullptr;}
        return &i->second.value;
    }
    bool put(const Key &key,Value value,const RuntimeMemorySnapshot &before,uint64_t reported_free,
             const RuntimeMemorySnapshot &after,bool admitted) {
        if(!admitted || !runtime_plan_may_reuse(before,reported_free,after)){entries.erase(key);return false;}
        if(!entries.count(key) && entries.size()>=capacity)entries.erase(entries.begin());
        entries[key]={before,std::move(value)};return true;
    }
    size_t size()const{return entries.size();}
};
} } }
