#pragma once
#include "ecm_cuda_stage2.h"
#include "ecm_stage2_geometry.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Compact, versioned runtime profile. No Python or JSON parser is required by
// the production binary; the exporter retains the original JSON/audit hashes.
namespace ecm_stage2 { namespace cost {
inline Word integer(const std::string &s) {
    if(s.empty() || s.find_first_not_of("0123456789")!=s.npos)throw std::runtime_error("invalid profile integer");
    size_t used=0;const auto n=std::stoull(s,&used);if(used!=s.size())throw std::runtime_error("invalid profile integer");return n;
}
inline double number(const std::string &s) {
    size_t used=0;const double n=std::stod(s,&used);
    if(used!=s.size() || !std::isfinite(n) || n<0)throw std::runtime_error("profile numbers must be finite and nonnegative");return n;
}
inline bool hex(const std::string &s,size_t count) {return s.size()==count && s.find_first_not_of("0123456789abcdef")==s.npos;}
struct Scope {
    Word bits=0,b1=0,arena_mb=0,resident=0,b2min=0,b2max=0,pmin=0,pmax=0,gmin=0,gmax=0;
    double cold=0;std::vector<Word> d;std::array<double,17> rate{};std::array<Word,3> covered{};
};
struct Stage1 {Word bits=0,b1=0,batch=0,torsion=0;double seconds=0;};
struct Profile {
    std::string binary,uuid,source,audit;
    Word major=0,minor=0,runtime=0,driver=0,fixed=0,outer=0,accounting=0,naming=0,features=0,chain_min=32768;
    std::vector<Stage1> stage1;std::vector<Scope> scopes;
    static Profile load(const std::filesystem::path &path) {
        std::error_code error;const auto size=std::filesystem::file_size(path,error);
        if(error || size>1048576)throw std::runtime_error("cost profile is missing or exceeds 1MiB");
        std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open cost profile");
        Profile p;std::string line;bool magic=false,identity=false,ended=false;
        std::set<std::string> keys;
        while(std::getline(in,line)) {
            if(!magic && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);
            std::istringstream row(line);row.imbue(std::locale::classic());std::vector<std::string> t;std::string token;
            while(row>>token)t.push_back(token);
            if(t.empty() || t[0][0]=='#')continue;
            if(ended)throw std::runtime_error("data after cost profile END");
            if(!magic) {if(t!=std::vector<std::string>{"ECM_STAGE2_COST_PROFILE","2"})throw std::runtime_error("unsupported cost profile format (requires exact-tree v2)");magic=true;continue;}
            if(t[0]=="identity") {
                if(identity || t.size()!=15)throw std::runtime_error("invalid/duplicate profile identity");
                p.binary=t[1];p.uuid=t[2];p.major=integer(t[3]);p.minor=integer(t[4]);p.runtime=integer(t[5]);p.driver=integer(t[6]);
                p.fixed=integer(t[7]);p.outer=integer(t[8]);p.accounting=integer(t[9]);p.naming=integer(t[10]);p.features=integer(t[11]);p.chain_min=integer(t[12]);p.source=t[13];p.audit=t[14];
                if(!hex(p.binary,64)||!hex(p.uuid,32)||!hex(p.source,64)||!hex(p.audit,64)||p.accounting!=2||p.fixed!=3||p.outer!=0||p.naming>1||p.features!=7||p.chain_min>100000000)
                    throw std::runtime_error("unsupported/corrupt cost profile identity");identity=true;
            } else if(t[0]=="stage1") {
                if(!identity || t.size()!=6 || p.stage1.size()>=1024)throw std::runtime_error("invalid Stage1 profile row");
                Stage1 s{integer(t[1]),integer(t[2]),integer(t[3]),integer(t[4]),number(t[5])};
                if(s.bits<2||s.bits>8192||s.b1<2||!s.batch||s.batch>1048576||s.torsion!=1||s.seconds<=0)throw std::runtime_error("invalid Stage1 scope");
                const auto key="s1:"+std::to_string(s.bits)+":"+std::to_string(s.b1)+":"+std::to_string(s.batch);if(!keys.insert(key).second)throw std::runtime_error("duplicate Stage1 scope");p.stage1.push_back(s);
            } else if(t[0]=="scope") {
                if(!identity || t.size()<34 || p.scopes.size()>=512)throw std::runtime_error("invalid Stage2 profile row");
                Scope s;s.bits=integer(t[1]);s.b1=integer(t[2]);s.arena_mb=integer(t[3]);s.resident=integer(t[4]);
                s.b2min=integer(t[5]);s.b2max=integer(t[6]);s.pmin=integer(t[7]);s.pmax=integer(t[8]);s.gmin=integer(t[9]);s.gmax=integer(t[10]);
                s.cold=number(t[11]);const Word count=integer(t[12]);
                if(!count||count>256||t.size()!=33+count||s.bits<2||s.bits>8192||s.b1<2||!s.arena_mb||s.arena_mb>1048576||s.resident>1||
                   s.b2min<=s.b1||s.b2min>s.b2max||s.b2max>((Word)INT64_MAX-8192)||!s.pmin||s.pmin>s.pmax||s.pmax>(1ull<<26)||!s.gmin||s.gmin>s.gmax||
                   (s.gmin==1&&(s.gmax!=1||s.resident)))
                    throw std::runtime_error("invalid/unsupported Stage2 scope");
                for(size_t i=0;i<count;++i){const Word d=integer(t[13+i]);if(d<6||d%2||d>200000000)throw std::runtime_error("invalid profile D");s.d.push_back(d);}
                if(std::set<Word>(s.d.begin(),s.d.end()).size()!=s.d.size())throw std::runtime_error("duplicate profile D");
                double sum=0;for(size_t i=0;i<17;++i){s.rate[i]=number(t[13+count+i]);sum+=s.rate[i];}
                for(size_t i=0;i<3;++i){s.covered[i]=integer(t[30+count+i]);if(s.covered[i]>1)throw std::runtime_error("invalid giant coverage");}
                if(!std::isfinite(sum)||sum<=0)throw std::runtime_error("empty profile rates");
                std::string key="s2";for(size_t i=1;i<=10;++i)key+=':'+std::to_string(integer(t[i]));
                auto ordered=s.d;std::sort(ordered.begin(),ordered.end());for(Word d:ordered)key+=':'+std::to_string(d);
                if(!keys.insert(key).second)throw std::runtime_error("duplicate Stage2 scope");p.scopes.push_back(s);
            } else if(t[0]=="END") {
                if(t.size()!=3||integer(t[1])!=p.stage1.size()||integer(t[2])!=p.scopes.size()||p.scopes.empty())throw std::runtime_error("incomplete cost profile");ended=true;
            } else throw std::runtime_error("unknown cost profile row");
        }
        if(!in.eof()||!magic||!identity||!ended)throw std::runtime_error("truncated cost profile");return p;
    }
};
inline Word phi(Word d) {Word value=d;for(Word p=2;p<=d/p;++p)if(d%p==0){while(d%p==0)d/=p;value-=value/p;}if(d>1)value-=value/d;return value;}
struct Work {
    struct Tree {double work=0;Word pairs=0,groups=0,copies=0,coeffs=0;};
    int bits;std::map<Word,double> units;std::map<Word,Tree> trees;
    double unit(Word p) {auto it=units.find(p);if(it!=units.end())return it->second;uint64_t n=0;
        if(!ecm_cuda_stage2_shape_query(p,bits,&n,nullptr))throw std::runtime_error("NTT packing refused profile candidate");
        return units[p]=(double)n*std::log2((double)n);}
    Tree stats(Word p) {auto it=trees.find(p);if(it!=trees.end())return it->second;Tree result;
        for(Word h=1;h<p;h*=2){const Word q=p/(2*h),r=p%(2*h),pairs=q+(r>h);
            result.work+=(double)pairs*unit(h+1);result.pairs+=pairs;result.groups+=(q>0)+(r>h);
            if(r&&r<=h){++result.copies;result.coeffs+=r+1;}}
        return trees[p]=result;}
    double tree(Word p){return stats(p).work;}
    double inverse(Word p) {double result=0;for(Word m=1;m<p;){m=std::min(2*m,p);result+=2*unit(m);}return result;}
};
struct Request {Word b1=0,b2min=0,b2max=0,d=0,arena_mb=0,owner_mb=640,batch=1;int bits=0;double t1=0,adjust=1;};
struct Choice {
    Word b2=0,d=0,p=0,i=0,g=0,arena_mb=0,owner_mb=0,candidates=0;
    Geometry geometry;double t1=0,t2=0,engine=0,cold=0,k=0,score=0;
    bool resident=false,limited=false,t1_explicit=false;Word stage1_batch=1,free_bytes=0;std::array<double,10> phases{};
    Word chain_min=32768,gt_pairs=0,gt_groups=0,gt_copy_words=0;double local_inverse=0,root_reduction=0;
};
inline Choice choose(const Profile &profile,const Request &r,const EcmStage2DeviceInfo &device) {
    if(r.bits<2||r.bits>8192||r.b1<2||r.adjust<=0||!std::isfinite(r.adjust)||r.t1<0||!std::isfinite(r.t1))throw std::runtime_error("invalid Auto B2 request");
    if(profile.uuid!=device.uuid_hex||profile.major!=(Word)device.major||profile.minor!=(Word)device.minor||profile.runtime!=(Word)device.runtime||
       profile.driver!=(Word)device.driver||profile.fixed!=(Word)device.fixed_mode||profile.outer!=(Word)device.outer_unroll_u)
        throw std::runtime_error("cost profile device/runtime/backend mismatch");
    double t1=r.t1;
    if(!t1)for(const auto &s:profile.stage1)if(s.bits==(Word)r.bits&&s.b1==r.b1&&s.batch==r.batch)t1=s.seconds;
    if(t1<=0)throw std::runtime_error("no matching Stage1 amortization; provide --stage1-seconds-per-curve");
    std::vector<const Scope*> scopes;
    for(const auto &s:profile.scopes)if(s.bits==(Word)r.bits&&s.b1==r.b1&&(!r.arena_mb||s.arena_mb==r.arena_mb)&&
        (!r.d||std::find(s.d.begin(),s.d.end(),r.d)!=s.d.end()))scopes.push_back(&s);
    if(scopes.empty())throw std::runtime_error("no measured bit-width/B1/arena/D scope");
    Word lo=(Word)INT64_MAX-8192,hi=0;
    for(const auto *s:scopes){lo=std::min(lo,s->b2min);hi=std::max(hi,s->b2max);}
    if(r.b2min){if(r.b2min<lo||r.b2min>hi)throw std::runtime_error("Auto B2 lower bound outside measured scope");lo=r.b2min;}
    if(r.b2max){if(r.b2max>hi||r.b2max<lo)throw std::runtime_error("Auto B2 upper bound outside measured scope");hi=r.b2max;}
    if(lo>hi)throw std::runtime_error("empty Auto B2 search interval");
    std::set<Word> grid;
    for(const auto *s:scopes){const Word low=std::max(lo,s->b2min),high=std::min(hi,s->b2max);if(low>high)continue;
        grid.insert(low);grid.insert(high);
        for(int j=0;j<33;++j){const double x=std::exp(std::log((double)low)+(std::log((double)high)-std::log((double)low))*j/32.0);
            const Word value=x<=(double)low?low:x>=(double)high?high:(Word)std::floor(x+0.5);
            grid.insert(std::max(low,std::min(high,value)));}}
    const auto initial=grid;
    for(const auto *s:scopes)for(Word d:s->d) {
        if(r.d&&r.d!=d)continue;const Word p=phi(d)/2;
        auto edge=[&](Word points){if(points<2||points-2>hi/d)return;const Word b=d*(points-2);
            for(int delta=-1;delta<=1;++delta){const Word v=delta<0?(b?b-1:0):b+(Word)delta;if(v>=lo&&v<=hi)grid.insert(v);}};
        edge(profile.chain_min);
        for(Word b:initial){const Word g=(b/d+2)/p;for(Word k=g>0?g-1:0;k<=g+1;++k)if(k<=hi/d/p+2)edge(k*p);}
    }
    Work work{r.bits};Choice best;Word candidates=0;
    const Word reserve=768ull<<20;
    const Word available=device.free_bytes>reserve?device.free_bytes-reserve:0;
    for(const auto *s:scopes)for(Word d:s->d) {
        if(r.d&&r.d!=d)continue;const Word p=phi(d)/2,w=((Word)r.bits+63)/64;
        if(p<s->pmin||p>s->pmax||s->arena_mb>available/(1ull<<20))continue;
        Geometry geom;
        const auto query=[](Word p,int bits,Word *n,Word *out){uint64_t a=0,b=0;const bool ok=ecm_cuda_stage2_shape_query(p,bits,&a,&b);*n=a;*out=b;return ok;};
        if(!geometry(p,r.bits,query,geom)||geom.arena_estimate_bytes>s->arena_mb*(1ull<<20)||
           (s->resident&&geom.fold_owner_bytes>r.owner_mb*(1ull<<20)))continue;
        for(Word b:grid) {
            if(b<s->b2min||b>s->b2max)continue;
            const Word i=b/d+2,g=i/p+(i%p!=0);if(g<s->gmin||g>s->gmax)continue;
            const Word k=std::max(p,(256ull<<20)/(16*w)),chunk=p*((k+p-1)/p),full=i/chunk,tail=i%chunk;
            Word chain=chunk>=profile.chain_min?full*chunk:0,ladder=chunk>=profile.chain_min?0:full*chunk,chains=chunk>=profile.chain_min?full:0;
            Word ladder_launches=chunk>=profile.chain_min?0:full*((chunk+8191)/8192);
            if(tail){if(tail>=profile.chain_min){chain+=tail;++chains;}else{ladder+=tail;ladder_launches+=(tail+8191)/8192;}}
            if((chain&&!s->covered[0])||(chains&&!s->covered[1])||(ladder&&!s->covered[2]))continue;
            const double per=6+22*std::log2((double)b)/64,tw=work.tree(p);
            Choice c;c.b2=b;c.d=d;c.p=p;c.i=i;c.g=g;c.geometry=geom;c.arena_mb=s->arena_mb;c.owner_mb=s->resident?r.owner_mb:0;c.resident=s->resident!=0;
            c.stage1_batch=r.batch;c.t1_explicit=r.t1>0;c.free_bytes=device.free_bytes;
            const auto ft=work.stats(p),gt=work.stats(i%p);c.gt_pairs=(i/p)*ft.pairs+gt.pairs;
            c.gt_groups=(i/p)*ft.groups+gt.groups;c.gt_copy_words=((i/p)*ft.coeffs+gt.coeffs)*w;
            c.chain_min=profile.chain_min;c.local_inverse=g==1?work.inverse(p):0;
            c.root_reduction=(g==1&&i==p)?work.unit(1)+work.unit(p+1):0;
            c.phases={s->rate[0]*p*std::max(1.0,std::log2((double)d)-2),s->rate[1]*p,s->rate[2]*tw,
                s->rate[3]*((i/p)*tw+gt.work)+s->rate[13]*c.gt_groups+s->rate[14]*c.gt_copy_words,
                s->rate[4]*(g-1)*work.unit(p+1),s->rate[5]*tw+s->rate[12]*c.local_inverse+s->rate[15]*c.root_reduction,
                s->rate[6]*(g>1?work.inverse(p+1):0),s->rate[7]*p,s->rate[8]*g,s->rate[9]*chain*per+s->rate[10]*chains+s->rate[11]*ladder*per+s->rate[16]*ladder_launches};
            for(double x:c.phases)c.engine+=x;
            c.cold=s->cold;c.t1=t1;c.t2=r.adjust*(c.engine+c.cold);
            c.k=0.11343+0.88657*std::pow(std::log10((double)b/r.b1)/2,1.96617-0.06781*std::log10((double)r.b1));
            c.score=c.k/(c.t1+c.t2);c.limited=b==lo||b==hi||b==s->b2min||b==s->b2max;
            if(!std::isfinite(c.engine)||!std::isfinite(c.t2)||!std::isfinite(c.t1+c.t2)||!std::isfinite(c.score)||c.t2<=0||c.score<=0)continue;++candidates;
            if(!best.d||c.score>best.score||(c.score==best.score&&(c.resident?c.geometry.fold_owner_bytes:0)<(best.resident?best.geometry.fold_owner_bytes:0)))best=c;
        }
    }
    if(!best.d)throw std::runtime_error("no feasible calibrated Auto B2 candidate at current memory state");best.candidates=candidates;return best;
}
inline std::string json(const Choice &c,const std::string &hash) {
    std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)
        <<"{\"type\":\"stage2_auto_plan\",\"schema\":2,\"feature_profile\":7,\"chain_min\":"<<c.chain_min<<",\"B2\":"<<c.b2<<",\"D\":"<<c.d<<",\"P\":"<<c.p<<",\"I\":"<<c.i<<",\"G\":"<<c.g
        <<",\"T1\":"<<c.t1<<",\"T2\":"<<c.t2<<",\"engine_seconds\":"<<c.engine<<",\"cold_seconds\":"<<c.cold<<",\"K\":"<<c.k<<",\"score\":"<<c.score
        <<",\"arena_mb\":"<<c.arena_mb<<",\"owner_runtime_mb\":"<<c.owner_mb<<",\"owner_resident\":"<<(c.resident?"true":"false")
        <<",\"fold_length\":"<<c.geometry.fold_length<<",\"arena_estimate_bytes\":"<<c.geometry.arena_estimate_bytes<<",\"owner_bytes\":"<<(c.resident?c.geometry.fold_owner_bytes:0)
        <<",\"candidate_count\":"<<c.candidates<<",\"range_limited\":"<<(c.limited?"true":"false")<<",\"process_peak_guaranteed\":false,\"free_bytes\":"<<c.free_bytes
        <<",\"g_tree_pairs\":"<<c.gt_pairs<<",\"g_tree_groups\":"<<c.gt_groups<<",\"g_tree_copy_words\":"<<c.gt_copy_words
        <<",\"local_inverse_work\":"<<c.local_inverse<<",\"root_reduction_work\":"<<c.root_reduction
        <<",\"T1_source\":\""<<(c.t1_explicit?"explicit_seconds":"measured_process_batch")<<"\",\"stage1_batch\":"<<c.stage1_batch<<",\"profile_sha256\":\""<<hash<<"\",\"phases\":{";
    const char *keys[]={"baby","affine","ftree","gtrees","fold","descent","inv","accum","glue","giant"};
    for(size_t i=0;i<10;++i){if(i)out<<',';out<<'"'<<keys[i]<<"\":"<<c.phases[i];}out<<"}}";return out.str();
}
}} // namespace ecm_stage2::cost
