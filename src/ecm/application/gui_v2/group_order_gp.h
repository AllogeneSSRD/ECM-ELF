#pragma once
#include "group_order_job.h"
#include "group_order_result.h"
#include <map>
#include <sstream>
#include <stdexcept>
#include <algorithm>
namespace ecmgui_v2::order_gp {
inline go_json::Value flag(bool b){go_json::Value v;v.scalar=b?"true":"false";return v;}
inline go_json::Value failure(const std::string&backend,const std::string&reason,const char*status="resource_limit",const char*primality="unknown"){
    go_json::Value r;r.type='o';r.set("protocol",go_json::Value::number("1"));
    for(auto kv:{std::pair<const char*,std::string>{"backend",backend},{"status",status},{"reason",reason},{"fallback",""},{"modulus_primality",primality},{"group_order","0"},{"point_order","0"},{"cofactor","1"}})r.set(kv.first,go_json::Value::string(kv.second));
    for(auto key:{"all_proven","group_computed","point_minimality"})r.set(key,flag(false));
    go_json::Value empty;empty.type='a';r.set("group_factors",empty);r.set("point_factors",empty);return r;
}
// Same checked models as core/group_order and tools/ecm_dataset/edwards_order.gp.
// Keep the Edwards B coefficient and direct base-point map: no square-root choice.
inline std::string script(const OrderInput&i){
    return std::string(R"GP(ecm_model(p,s,k)={
my(E,P,W,Q,a,b,x,y,t,x3,alpha,beta,h,d,u,v);
if(k==1,
 W=ellinit([-8,-32],p);if(#W==0,error("auxiliary curve"));
 Q=ellmul(W,[Mod(12,p),Mod(40,p)],s);if(Q==[0],error("auxiliary infinity"));
 x=Q[1];t=Q[2];alpha=(x-9)/(t+x+16);beta=2*alpha*(4*alpha+1)/(8*alpha^2-1);h=2*beta-1;
 d=(2*h^2-1)/h^4;if(d==0||d==1,error("singular Edwards curve"));
 y=h*(t^2+50*t-2*x^3+27*x^2-104)/((t+3*x-2)*(t+x+16));x=h*(4*beta-3)/(6*beta-5);
 if(x^2+y^2!=1+d*x^2*y^2,error("Edwards base point"));a=2*(1+d)/(1-d);b=4/(1-d);
 E=ellinit([0,b*a,0,b^2,0]);if(y==1,P=[0],if(y==-1,P=[Mod(0,p),Mod(0,p)],u=(1+y)/(1-y);P=[b*u,b^2*u/x]));
 ,
 x=2;y=1;
 if(k==0,u=Mod(s^2-5,p);v=Mod(4*s,p);a=(3*u+v)*(v-u)^3/(4*u^3*v)-2;x=(u/v)^3;b=x*(x*(x+a)+1),
 if(k==3||k==4,a=Mod(4*if(k==3,s^2,s),p)/Mod(2^if(k==3,64,32),p)-2;b=4*a+10,
 W=ellinit([0,Mod(36,p)]);Q=ellmul(W,[-3,3],s);if(Q==[0],error("auxiliary infinity"));
 x3=(3*Q[1]+Q[2]+6)/(2*(Q[2]-3));a=-(3*x3^4+6*x3^2-1)/(4*x3^3);y=4*a+10;b=1/y));
 if(b==0||a^2==4,error("singular Montgomery curve"));E=ellinit([0,b*a,0,b^2,0]);P=[b*x,b^2*y]);
if(#E==0||!ellisoncurve(E,P),error("invalid curve"));[E,P];
};
)GP")+R"GP(ecm_prime(p,s,k)={my(m,E,P,n,f,o,fo);
if(p<=3,print("ECM_MATH_ERROR|unsupported_characteristic");return());
iferr(m=ecm_model(p,s,k),err,print("ECM_MATH_ERROR|invalid_curve");return());E=m[1];P=m[2];
n=ellcard(E);f=factor(n);for(i=1,matsize(f)[1],if(!isprime(f[i,1]),error("unproven factor")));
o=ellorder(E,P,[n,f]);fo=factor(o);if(n%o||ellmul(E,P,o)!=[0],error("point order"));
for(i=1,matsize(fo)[1],if(!isprime(fo[i,1])||ellmul(E,P,o/fo[i,1])==[0],error("point minimality")));
print("GROUP|",n);print("POINT|",o);
for(i=1,matsize(f)[1],print("G|",f[i,1],"|",f[i,2]));
for(i=1,matsize(fo)[1],print("P|",fo[i,1],"|",fo[i,2]));print("ECM_OK");
};
)GP"+"ecm_go()={my(n="+i.factor+",s="+i.sigma+",k="+std::to_string(i.profile)+R"GP(,f,p);
if(n<=1||n>=2^4096||s==0,print("ECM_MATH_ERROR|invalid_input");return());
if(isprime(n),ecm_prime(n,s,k);return());
f=factor(n);for(j=1,matsize(f)[1],if(!isprime(f[j,1]),error("unproven input factor")));
for(j=1,matsize(f)[1],p=f[j,1];print("COMPONENT|",p,"|",f[j,2]);ecm_prime(p,s,k);print("END_COMPONENT"));
print("ECM_BATCH_OK");};
iferr(ecm_go(),err,print("ECM_GP_ERROR"));quit();
)GP";
}
// Bounded decimal checks avoid adding GMP to the GUI. GP proves primality and
// point minimality; the host independently checks products, divisibility/Hasse.
inline std::string natural(std::string s){if(s.empty()||s.size()>1236||s.find_first_not_of("0123456789")!=s.npos)throw std::invalid_argument("Invalid GP integer");const auto at=s.find_first_not_of('0');return at==s.npos?"0":s.substr(at);}
inline int compare(const std::string&a,const std::string&b){return a.size()!=b.size()?(a.size()<b.size()?-1:1):a.compare(b);}
inline std::string multiply(const std::string&a,const std::string&b){
    std::vector<unsigned> d(a.size()+b.size());for(size_t i=0;i<a.size();++i)for(size_t j=0;j<b.size();++j)d[i+j]+=(a[a.size()-1-i]-'0')*(b[b.size()-1-j]-'0');
    for(size_t i=0;i+1<d.size();++i){d[i+1]+=d[i]/10;d[i]%=10;}while(d.size()>1&&!d.back())d.pop_back();std::string s;for(auto it=d.rbegin();it!=d.rend();++it)s+=char('0'+*it);return s;
}
inline std::string plus_one(std::string s){for(auto it=s.rbegin();it!=s.rend();++it)if(*it!='9'){++*it;return s;}else *it='0';return '1'+s;}
inline std::string difference(std::string a,std::string b){if(compare(a,b)<0)a.swap(b);int borrow=0;for(size_t i=0;i<a.size();++i){int d=a[a.size()-1-i]-'0'-borrow-(i<b.size()?b[b.size()-1-i]-'0':0);borrow=d<0;a[a.size()-1-i]=char('0'+(d<0?d+10:d));}return natural(a);}
inline go_json::Value parse(const std::string&raw,const OrderInput&i);
inline go_json::Value parse_components(const std::string&raw,const OrderInput&i){
    using V=go_json::Value;V components,factors;components.type=factors.type='a';
    std::istringstream in(raw);std::string line,prime,exponent,body,product="1",previous;
    bool active=false,done=false;
    const auto bound=natural(i.factor);
    while(std::getline(in,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line.rfind("COMPONENT|",0)==0){
            if(active||done||components.items.size()>=4096)throw std::invalid_argument("Invalid GP component boundary");
            const auto split=line.find('|',10);if(split==line.npos)throw std::invalid_argument("Invalid GP component");
            prime=natural(line.substr(10,split-10));exponent=natural(line.substr(split+1));
            if(compare(prime,"2")<0||compare(prime,bound)>=0||(!previous.empty()&&compare(prime,previous)<=0)||exponent.size()>4)throw std::invalid_argument("Invalid GP input factor");
            const auto e=std::stoul(exponent);if(!e||e>4096)throw std::invalid_argument("Invalid GP input exponent");
            for(unsigned n=0;n<e;++n){product=multiply(product,prime);if(compare(product,bound)>0)throw std::invalid_argument("GP input factors exceed modulus");}
            active=true;body.clear();previous=prime;
        }else if(line=="END_COMPONENT"){
            if(!active)throw std::invalid_argument("Unexpected GP component end");
            auto input=i;input.factor=prime;auto component=parse(body,input);
            component.set("prime",V::string(prime));component.set("exponent",V::number(exponent));components.items.push_back(std::move(component));
            V f;f.type='o';f.set("prime",V::string(prime));f.set("exponent",V::number(exponent));f.set("primality",V::string("proven"));factors.items.push_back(std::move(f));active=false;
        }else if(line=="ECM_BATCH_OK"){
            if(active||done)throw std::invalid_argument("Invalid GP batch completion");done=true;
        }else if(active){
            if(line=="ECM_GP_ERROR"||line=="ECM_BATCH_OK")throw std::runtime_error("gp_protocol_failed");
            body+=line+'\n';
        }else if(line=="ECM_GP_ERROR")throw std::runtime_error("gp_protocol_failed");
        else if(line.rfind("GROUP|",0)==0||line.rfind("POINT|",0)==0||line=="ECM_OK")throw std::invalid_argument("GP result outside component");
    }
    if(!done||active||components.items.empty())throw std::runtime_error("gp_protocol_failed");
    if(product!=bound)throw std::invalid_argument("GP input factor product mismatch");
    return order_components(std::move(components),std::move(factors),"1","gp",true);
}
inline go_json::Value parse(const std::string&raw,const OrderInput&i){
    if(raw.find("COMPONENT|")!=raw.npos)return parse_components(raw,i);
    std::istringstream in(raw);std::string line,group,point;bool ok=false;std::map<std::string,unsigned> g,p;
    while(std::getline(in,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line.rfind("ECM_MATH_ERROR|",0)==0)return failure("gp",line.substr(15),"invalid_input",line.substr(15)=="composite_modulus"?"composite":line.substr(15)=="invalid_curve"?"proven":"unknown");
        if(line=="ECM_OK"){if(ok)throw std::invalid_argument("Duplicate GP completion");ok=true;}
        else if(line.rfind("GROUP|",0)==0){if(!group.empty())throw std::invalid_argument("Duplicate GP group");group=natural(line.substr(6));}
        else if(line.rfind("POINT|",0)==0){if(!point.empty())throw std::invalid_argument("Duplicate GP point");point=natural(line.substr(6));}
        else if(line.rfind("G|",0)==0||line.rfind("P|",0)==0){
            const auto split=line.find('|',2);if(split==line.npos)throw std::invalid_argument("Invalid GP factor");const auto prime=natural(line.substr(2,split-2)),exponent=natural(line.substr(split+1));
            if(prime=="0"||prime=="1"||exponent.size()>4)throw std::invalid_argument("Invalid GP prime power");const auto e=std::stoul(exponent);auto&parts=line[0]=='G'?g:p;
            if(!e||e>8192||!parts.emplace(prime,unsigned(e)).second)throw std::invalid_argument("Invalid GP prime power");
        }
    }
    if(!ok||group.empty()||point.empty())throw std::runtime_error("gp_protocol_failed");
    if(group=="0"||point=="0")throw std::invalid_argument("Invalid GP orders");
    const auto delta=difference(group,plus_one(natural(i.factor)));
    if(compare(multiply(delta,delta),multiply(natural(i.factor),"4"))>0)throw std::invalid_argument("GP group violates Hasse bound");
    const auto factors=[](const std::map<std::string,unsigned>&parts,const std::string&bound){
        std::string product="1";go_json::Value list;list.type='a';std::vector<std::pair<std::string,unsigned>> sorted(parts.begin(),parts.end());std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){return compare(a.first,b.first)<0;});
        for(const auto&f:sorted){for(unsigned n=0;n<f.second;++n){product=multiply(product,f.first);if(compare(product,bound)>0)throw std::invalid_argument("GP factor product exceeds order");}
            go_json::Value row;row.type='o';row.set("prime",go_json::Value::string(f.first));row.set("exponent",go_json::Value::number(std::to_string(f.second)));row.set("primality",go_json::Value::string("proven"));list.items.push_back(row);}
        if(product!=bound)throw std::invalid_argument("GP factor product mismatch");return list;
    };
    for(const auto&f:p){auto it=g.find(f.first);if(it==g.end()||f.second>it->second)throw std::invalid_argument("GP point does not divide group");}
    auto r=failure("gp","","complete","proven");r.set("group_order",go_json::Value::string(group));r.set("point_order",go_json::Value::string(point));r.set("group_factors",factors(g,group));r.set("point_factors",factors(p,point));
    for(auto key:{"all_proven","group_computed","point_minimality"})r.set(key,flag(true));return r;
}
}
