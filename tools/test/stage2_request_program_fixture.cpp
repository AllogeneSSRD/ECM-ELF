// CPU-only independent dense topology oracle. No CUDA/runtime initialization.
#include "../../src/core/ecm_stage2_requests.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ecm_stage2;
using Requests=std::vector<MultiplyRequest>;
static unsigned long long checks=0;
static std::string context;
static void check(bool value) {++checks;if(!value)throw std::runtime_error("request fixture mismatch "+context);}
static std::vector<Word> degrees(Word p) {
    Word pad=1;while(pad<p)pad*=2;
    std::vector<Word> d(2*pad);for(Word i=0;i<p;++i)d[pad+i]=1;
    for(Word i=pad;--i;)d[i]=d[2*i]+d[2*i+1];return d;
}
static void dense_tree(Word p,unsigned phase,Requests &out) {
    const auto d=degrees(p);
    for(Word base=d.size()/4;base;base/=2) {
        std::map<std::pair<Word,Word>,Word> groups;
        for(Word i=base;i<2*base;++i)if(d[2*i] && d[2*i+1])
            ++groups[{std::min(d[2*i],d[2*i+1])+1,std::max(d[2*i],d[2*i+1])+1}];
        for(const auto &g:groups)out.push_back({phase,g.first.first,g.first.second,g.second,0,g.first.first+g.first.second-1,
            phase==RequestGtrees?RequestTreeRaw:RequestHost});
    }
}
static Requests dense_program(Word p,Word I) {
    Requests out;dense_tree(p,RequestFtree,out);
    Word inverse=1;
    while(inverse<=p) {
        const Word n=std::min(2*inverse,p+1);
        out.push_back({RequestInverse,n,inverse,1,0,n});out.push_back({RequestInverse,inverse,n,1,0,n});inverse=n;
    }
    Word H=0;
    for(Word begin=0;begin<I;begin+=p) {
        const Word g=std::min(p,I-begin);dense_tree(g,RequestGtrees,out);
        if(!H)H=g+1;
        else {
            const Word product=g+H;
            out.push_back({RequestFold,g+1,H,1,0,product,RequestFoldOwner});
            if(product>p) {
                const Word q=product-p;
                out.push_back({RequestFold,q,q,1,0,q,RequestFoldOwner});out.push_back({RequestFold,q,p+1,1,0,p,RequestFoldOwner});H=p;
            } else H=product;
        }
    }
    out.push_back({RequestDescent,p,p,1,0,p,RequestFoldOwner});
    const auto d=degrees(p);const Word pad=d.size()/2;
    for(Word base=1;base<pad;base*=2) {
        std::map<std::pair<Word,Word>,Word> groups;
        for(Word i=base;i<2*base;++i) {
            const Word a=d[2*i],b=d[2*i+1];if(a && b){++groups[{a,b}];++groups[{b,a}];}
        }
        for(const auto &g:groups) {const Word a=g.first.first,b=g.first.second;
            out.push_back({RequestDescent,a+b,b+1,g.second,b,a,RequestFrontierOwner});}
    }
    return out;
}
static bool equal(const MultiplyRequest &a,const MultiplyRequest &b) {
    return a.phase==b.phase && a.ma==b.ma && a.mb==b.mb && a.pairs==b.pairs && a.first==b.first && a.count==b.count && a.input==b.input;
}
int main() {
    try {
        for(Word p=1;p<=257;++p)for(Word I:{p+1,2*p,2*p+1,4*p+std::max(Word(1),p/3)}) {
            context="p="+std::to_string(p)+" I="+std::to_string(I);
            RequestProgram program;check(request_program(p,I,program) && program.supported);
            Requests actual;for(const auto &b:program.blocks)for(Word i=0;i<b.repeat;++i)actual.insert(actual.end(),b.requests.begin(),b.requests.end());
            const auto expected=dense_program(p,I);check(actual.size()==expected.size());
            for(size_t i=0;i<actual.size();++i)check(equal(actual[i],expected[i]));
        }
        for(Word p:{Word(65535),Word(65536),Word(65537),Word(126720),Word(138240),Word(262145)}) {
            context="large p="+std::to_string(p);
            RequestProgram program;check(request_program(p,3*p+7,program));
            Requests actual;for(const auto &b:program.blocks)for(Word i=0;i<b.repeat;++i)actual.insert(actual.end(),b.requests.begin(),b.requests.end());
            const auto expected=dense_program(p,3*p+7);check(actual.size()==expected.size());
            for(size_t i=0;i<actual.size();++i)check(equal(actual[i],expected[i]));
        }
        RequestSignature one;one.append_word(17);one.append_word(23);
        context="signature";
        for(Word repeat=0;repeat<1000;++repeat) {
            auto compressed=one;compressed.repeat(repeat);RequestSignature expanded;
            for(Word i=0;i<repeat;++i)expanded.then(one);
            check(compressed.multiplier==expanded.multiplier && compressed.addend==expanded.addend);
        }
        RequestProgram invalid;check(!request_program(0,9,invalid));check(!request_program(~Word(0),9,invalid));
        context="boundary";
        for(Word I:{Word(1),Word(48)})check(request_program(48,I,invalid) && !invalid.supported);
        // Huge B2 topology stays compressed. Overflowed aggregate counters fail closed.
        RequestProgram huge;check(request_program(48,Word(1)<<60,huge) && huge.blocks.size()<=6);
        auto query=[](Word m,int,Word *n,Word *out){*n=1;while(*n<4*m)*n*=2;*out=2*m-1;return true;};
        auto describe=[](Word n,Word &table,Word &base){table=8*n;base=16*n;return true;};
        for(unsigned buffers:{2u,3u})for(bool physical:{false,true}) {
            context="payload buffers="+std::to_string(buffers)+" physical="+std::to_string(physical);
            RequestPlan plan;check(request_plan(48,200,129,query,describe,1024,buffers,physical,0,true,plan) && plan.valid);
            Word pairs[RequestPhaseCount]{},groups[RequestPhaseCount]{};RequestSignature signatures[RequestPhaseCount],all;
            for(const auto &r:dense_program(48,200)) {
                Word n=0,slots=0;query(std::max(r.ma,r.mb),129,&n,&slots);
                Word c=1;for(Word test=r.pairs;test;test/=2)if(8*test*((physical?buffers:3)*n+slots+(physical?2:0))<=1024){c=test;break;}
                ++groups[r.phase];pairs[r.phase]+=r.pairs;signatures[r.phase].append(r,n,slots,c);all.append(r,n,slots,c);
            }
            for(unsigned i=0;i<RequestPhaseCount;++i)check(plan.phases[i].groups==groups[i] && plan.phases[i].pairs==pairs[i] &&
                plan.signatures[i].multiplier==signatures[i].multiplier && plan.signatures[i].addend==signatures[i].addend);
            check(plan.signature.multiplier==all.multiplier && plan.signature.addend==all.addend);
        }
        context="overflow";RequestPlan overflow;check(!request_plan(1,~Word(0),129,query,describe,1024,2,false,0,true,overflow));
        std::cout<<"{\"checks\":"<<checks<<",\"bad\":0}\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<" at check "<<checks<<'\n';return 1;}
}
