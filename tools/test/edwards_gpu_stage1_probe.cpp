#include "../../src/ecm/stage1/edwards/cuda/api.h"
#include "../../src/ecm/stage1/edwards/cuda/naf.h"
#include "ecm/curve/edwards.h"
#include "../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.h"
#include "ecm/io/save.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace ecm_edwards;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void identity(mpz_t *s,const mpz_t d,const mpz_t x,const mpz_t y,const mpz_t n) {
    mpz_set(s[0],d);mpz_set(s[1],x);mpz_set(s[2],y);mpz_set_ui(s[3],1);
    mpz_mul(s[4],x,y);mpz_mod(s[4],s[4],n);mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
}
static void run(unsigned bits,unsigned b1,unsigned count,uint64_t sigma,int device,
                const std::filesystem::path &out,unsigned other=0,unsigned window=0,int algorithm=4,bool boundary_input=false) {
    Integer n,s,d,x,y,a,f,old_d,old_x,old_y,cpu_x,cpu_z,qx,qz,lhs,rhs,g,inv;
    mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,bits);mpz_sub_ui(n.v,n.v,1);
    if(boundary_input){
        // ECM arithmetic is defined over odd composite moduli as well. Select
        // a near-limit odd N with valid curve setup; primality is irrelevant.
        // Never select/skip an input based on a GPU or scalar-multiplication result.
        static Integer boundary;static unsigned boundary_bits=0,boundary_count=0;
        static uint64_t boundary_sigma=0;
        if(boundary_bits!=bits || boundary_count!=count || boundary_sigma!=sigma){
            bool ready=false;
            for(unsigned offset=0;offset<4096 && !ready;++offset){
                mpz_sub_ui(boundary.v,n.v,2*offset);ready=true;
                for(unsigned i=0;i<count && ready;++i)
                    ready=construct(d.v,x.v,y.v,a.v,f.v,sigma+i,boundary.v)==Setup::Ready;
            }
            require(ready && mpz_sizeinbase(boundary.v,2)==bits,"no valid near-boundary modulus");
            boundary_bits=bits;boundary_count=count;boundary_sigma=sigma;
        }
        mpz_set(n.v,boundary.v);
    }
    if(other){mpz_set_ui(g.v,1);mpz_mul_2exp(g.v,g.v,other);mpz_sub_ui(g.v,g.v,1);mpz_mul(n.v,n.v,g.v);}
    mpz_set_ui(s.v,1);for(unsigned k=2;k<=b1;++k)mpz_lcm_ui(s.v,s.v,k);mpz_mul_ui(s.v,s.v,48);
    Array state(size_t(count)*9),resumed(size_t(count)*9),adaptive(size_t(count)*9);std::vector<uint64_t> sigmas;
    for(unsigned i=0;i<count;++i) {
        const auto sg=sigma+i;
        require(construct(d.v,x.v,y.v,a.v,f.v,sg,n.v)==Setup::Ready,"unexpected curve setup classification");
        edwards_atkin_morain(old_d.v,old_x.v,old_y.v,sg,n.v);
        require(!mpz_cmp(d.v,old_d.v) && !mpz_cmp(x.v,old_x.v) && !mpz_cmp(y.v,old_y.v),"checked constructor differs from CPU reference");
        identity(state.data()+size_t(i)*9,d.v,x.v,y.v,n.v);sigmas.push_back(sg);
    }
    for(size_t i=0;i<size_t(count)*9;++i){mpz_set(resumed[i],state[i]);mpz_set(adaptive[i],state[i]);}
    uint32_t done=0,split=0;std::string error;
    EcmEdwardsNafMetrics metrics;
    auto execute=[&](mpz_t *data,uint32_t &pos,uint32_t chunk,EcmEdwardsProgress cb){
        const auto before=pos;
        const auto code=window?ecm_cuda_edwards_stage1_naf(n.v,s.v,data,count,pos,chunk,device,cb,nullptr,error,window,algorithm,&metrics)
                     :ecm_cuda_edwards_stage1(n.v,s.v,data,count,pos,chunk,device,cb,nullptr,error);
        if(window && (code==0 || code==2)) {
            if(before==metrics.columns)
                require(!metrics.container_bits && !metrics.tpi && !metrics.dictionary_bytes,"completed NAF state selected device resources");
            else {
                uint32_t expected_bits=0,expected_tpi=0;
                require(ecm_cuda_edwards_tier(mpz_sizeinbase(n.v,2),expected_bits,expected_tpi),"missing compiled NAF tier");
                require(metrics.container_bits==expected_bits && metrics.tpi==expected_tpi,"executed NAF tier differs from query");
            }
        }
        return code;
    };
    require(execute(state.data(),done,0,nullptr)==0,error.c_str());
    require(execute(resumed.data(),split,17,[](uint32_t,void *){return false;})==2,"slice stop failed");
    // A new invocation uploads the complete ordinary-domain state, just as a
    // restarted worker does. Chunk boundaries deliberately differ.
    require(execute(resumed.data(),split,113,nullptr)==0,error.c_str());
    uint32_t paused=0;
    require(execute(adaptive.data(),paused,0,[](uint32_t,void *){return false;})==2,"adaptive slice stop failed");
    require(execute(adaptive.data(),paused,0,nullptr)==0,error.c_str());
    uint32_t container=0,tpi=0;
    require(ecm_cuda_edwards_tier(mpz_sizeinbase(n.v,2),container,tpi),"missing compiled Edwards tier");
    for(size_t i=0;i<size_t(count)*9;++i)require(!mpz_cmp(state[i],resumed[i]),"resumed result differs from uninterrupted run");
    for(size_t i=0;i<size_t(count)*9;++i)require(!mpz_cmp(state[i],adaptive[i]),"adaptive resume differs from uninterrupted run");
    std::string lines;
    for(unsigned i=0;i<count;++i) {
        auto *p=state.data()+size_t(i)*9;
        require(point_valid(n.v,p[0],p+5),"GPU point invariant failed");
        mpz_add(qx.v,p[7],p[6]);mpz_mod(qx.v,qx.v,n.v);mpz_sub(qz.v,p[7],p[6]);mpz_mod(qz.v,qz.v,n.v);
        require(edwards_stage1_curve(f.v,cpu_x.v,cpu_z.v,n.v,sigmas[i],s.v)>=0,"CPU reference failed");
        mpz_mul(lhs.v,qx.v,cpu_z.v);mpz_mul(rhs.v,cpu_x.v,qz.v);mpz_sub(lhs.v,lhs.v,rhs.v);mpz_mod(lhs.v,lhs.v,n.v);
        require(!mpz_sgn(lhs.v),"GPU final point differs from CPU");
        mpz_gcd(g.v,qz.v,n.v);mpz_gcd(f.v,cpu_z.v,n.v);require(!mpz_cmp(g.v,f.v),"GPU factor GCD differs from CPU");
        if(mpz_cmp_ui(g.v,1)==0){mpz_invert(inv.v,qz.v,n.v);mpz_mul(qx.v,qx.v,inv.v);mpz_mod(qx.v,qx.v,n.v);lines+=save_line(n.v,qx.v,sigmas[i],b1,false);}
        else if(mpz_cmp(g.v,n.v)<0)lines+=save_line(n.v,g.v,sigmas[i],b1,true);
    }
    const auto name=std::string(boundary_input?"n":"m")+std::to_string(bits)+(other?"x"+std::to_string(other):"")+"_b"+std::to_string(b1)+
        (window?"_w"+std::to_string(window)+"_a"+std::to_string(algorithm):"");
    std::ofstream file(out/(name+".save"),std::ios::binary);file<<lines;file.close();require(bool(file),"save fixture write failed");
    std::printf("PASS %s device=%d curves=%u columns=%u split_resume=17/113 adaptive_restart=1 cpu_match=1 tier=%u tpi=%u\n",name.c_str(),device,count,done,container,tpi);
    std::fflush(stdout);
}
int main(int argc,char **argv) {
    try {
        if(argc!=3 && argc!=4)throw std::runtime_error("usage: edwards_gpu_stage1_probe DEVICE OUTPUT_DIRECTORY [SINGLE_COMPILED_TIER]");
        const int device=std::atoi(argv[1]);std::filesystem::path out(argv[2]);std::filesystem::create_directories(out);
        // Short-scalar oracle: use the existing CPU implementation at w=2,
        // avoiding a fixed 1024-point CPU dictionary for 32-column tests.
        edwards_set_naf_w(2);
        if(argc==4 && std::string(argv[3])=="--compiled-tiers") {
            uint32_t bits=0,w=0,t=0;unsigned count=0;
            while(ecm_cuda_edwards_tier(bits,w,t)) {
                require(w>bits && ++count<=32,"compiled tier query did not advance");
                run(w-1,20,1,17,device,out,0,0,4,true);
                for(int algorithm:{4,5})run(w-1,20,1,17,device,out,0,6,algorithm,true);
                bits=w;
            }
            require(count!=0,"no compiled Edwards tiers");return 0;
        }
        if(argc==4 && std::string(argv[3])=="--wait-slices") {
            for(unsigned bits:{347,677,1279,9689})for(unsigned window:{0u,6u})
                run(bits,200,3,17,device,out,0,window);
            return 0;
        }
        if(argc==4){
            const unsigned expected=unsigned(std::stoul(argv[3]));uint32_t w=0,t=0;
            require(ecm_cuda_edwards_tier(1,w,t) && w==expected,"restricted minimum/fallback failed");
            require(ecm_cuda_edwards_tier(expected-1,w,t) && w==expected,"restricted upper boundary failed");
            require(!ecm_cuda_edwards_tier(expected,w,t) && !w && !t,"restricted ceiling failed");
            run(expected-1,20,33,17,device,out,0,0,4,true);
            for(int algorithm:{4,5})run(expected-1,20,33,17,device,out,0,6,algorithm,true);
            return 0;
        }
        const unsigned tiers[]={128,192,256,384,512,768,1024,1280,1536,1792,2048,
            2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192,
            9216,10240,11264,12288,13312,14336,15360,16384,17408};
        for(size_t i=0;i<std::size(tiers);++i){
            uint32_t w=0,t=0;
            require(ecm_cuda_edwards_tier(tiers[i]-1,w,t) && w==tiers[i],"smallest container dispatch failed");
            const unsigned expected_tpi=w<=512?4:w<=2048?8:w<=8192?16:32;
            require(t==expected_tpi,"TPI dispatch failed");
            const bool next=ecm_cuda_edwards_tier(tiers[i],w,t);
            require(i+1<std::size(tiers)?next && w==tiers[i+1]:!next && !w && !t,"one-spare-bit boundary failed");
            // Synthetic odd targets exercise every container's upper bit boundary.
            run(tiers[i]-1,20,1,17,device,out,0,0,4,true);
            for(int algorithm:{4,5})run(tiers[i]-1,20,1,17,device,out,0,6,algorithm,true);
        }
        run(127,20,33,17,device,out);
        run(127,20,4,17,device,out,89);
        run(347,1000,4,20260922,device,out);
        run(677,1000,4,20260922,device,out);
        run(991,1000,4,20260922,device,out);
        run(2203,1000,4,20260922,device,out);
        run(4003,1000,4,20260922,device,out);
        for(unsigned w: {2u,3u,5u,8u,12u})for(int algorithm: {4,5})run(127,1000,33,17,device,out,0,w,algorithm);
        for(int algorithm: {4,5}) {
            run(127,20,4,17,device,out,89,6,algorithm);
            run(347,1000,4,20260922,device,out,0,6,algorithm);
            run(677,1000,4,20260922,device,out,0,6,algorithm);
            run(991,1000,4,20260922,device,out,0,6,algorithm);
            run(1279,20,1,17,device,out,0,6,algorithm);
            run(2203,1000,4,20260922,device,out,0,6,algorithm);
            run(4003,1000,4,20260922,device,out,0,6,algorithm);
            run(4423,20,1,17,device,out,0,6,algorithm);
            run(9689,20,1,17,device,out,0,6,algorithm);
        }
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
