#include "../../src/ecm/stage2/cuda/api.h"
#include "../../src/ecm/stage2/relocation.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

static void require(bool ok) {if(!ok)throw std::runtime_error("relocation geometry invariant failed");}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--geometry") {
            using namespace ecm_stage2;
            Word checks=0;
            for(Word d:{6ull,10ull,30ull,42ull,210ull,2310ull,30030ull,180180ull,570570ull})
                for(Word b2:{100ull,10000ull,1000000ull,27015978990ull,273142759890ull}) {
                    const auto a=giant_range(b2,d,false),b=giant_range(b2,d,true);
                    require(a.first==1 && a.points==b2/d+2 && b.last==a.last);
                    require(b.first+b.points-1==b.last && b.points<=a.points);checks+=2;
                }
            for(Word p:{1ull,24ull,48ull,240ull,2880ull,17280ull})
                for(Word cap:{p,2*p,4*p,7*p})for(Word minimum:{8192ull,32768ull})
                    for(Word tail:{1ull,17ull,2881ull,8191ull,8192ull,p}) {
                        const Word total=2*cap+tail;
                        Word offset=0,chunks=0;
                        while(offset<total) {
                            const auto n=giant_chunk_take(total-offset,cap,p,minimum,true);
                            require(n && n<=cap && n<=total-offset);
                            require(offset+n==total || n%p==0);
                            offset+=n;++chunks;checks+=2;
                        }
                        require(chunks==total/cap+(total%cap!=0));++checks;
                    }
            require(giant_chunk_take(72001,69120,17280,8192,true)==51840);
            require(giant_chunk_take(72001,69120,17280,8192,false)==69120);
            std::cout<<"relocation_geometry: checks="<<checks+2<<" bad=0\n";return 0;
        }
        if(argc!=10)throw std::runtime_error("Expected Nhex Xhex a24hex B1 B2 D device relocation carrier");
        const auto number=[](const char *s){return std::stoull(s);};
        require(number(argv[8])<=1);
        ecm_cuda_stage2_set_log_level(4);
        require(ecm_cuda_stage2_set_relocation(number(argv[8])!=0)==0);
        EcmStage2Metrics m;std::string result;
        const int code=ecm_cuda_stage2_run(argv[1],argv[2],1,number(argv[4]),number(argv[5]),number(argv[6]),
            (int)number(argv[7]),[](const char *s,void *ctx){*static_cast<std::string*>(ctx)=s;},&result,
            (unsigned)number(argv[9]),&m,argv[3]);
        if(code)return code;
        require(!result.empty() && !m.bad && m.checked && m.phases.complete);
        require(std::abs(m.phases.total()-m.total_seconds)<=1e-8*std::max(1.,m.total_seconds));
        std::cout<<"relocation_probe_result: {"<<result<<",\"relocation\":"<<int(m.relocation)
            <<",\"replayed\":"<<int(m.relocation_replayed)<<",\"first\":"<<m.giant_first
            <<",\"points\":"<<m.giant_points<<",\"probe_seconds\":"<<m.relocation_probe_seconds
            <<",\"replay_seconds\":"<<m.relocation_replay_seconds<<",\"main_seconds\":"<<m.main_seconds<<"}\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
