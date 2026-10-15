#include "ecm_cuda_edwards.h"
#include "ecm_edwards_cpu.h"
#include "ecm_edwards_io.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace ecm_edwards;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void identity(mpz_t *s,const mpz_t d,const mpz_t x,const mpz_t y,const mpz_t n) {
    mpz_set(s[0],d);mpz_set(s[1],x);mpz_set(s[2],y);mpz_set_ui(s[3],1);
    mpz_mul(s[4],x,y);mpz_mod(s[4],s[4],n);mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
}
static void run(unsigned bits,unsigned b1,unsigned count,uint64_t sigma,int device,
                const std::filesystem::path &out,unsigned other=0) {
    Integer n,s,d,x,y,a,f,old_d,old_x,old_y,cpu_x,cpu_z,qx,qz,lhs,rhs,g,inv;
    mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,bits);mpz_sub_ui(n.v,n.v,1);
    if(other){mpz_set_ui(g.v,1);mpz_mul_2exp(g.v,g.v,other);mpz_sub_ui(g.v,g.v,1);mpz_mul(n.v,n.v,g.v);}
    mpz_set_ui(s.v,1);for(unsigned k=2;k<=b1;++k)mpz_lcm_ui(s.v,s.v,k);mpz_mul_ui(s.v,s.v,48);
    Array state(size_t(count)*9),resumed(size_t(count)*9);std::vector<uint64_t> sigmas;
    for(unsigned i=0;i<count;++i) {
        const auto sg=sigma+i;
        require(construct(d.v,x.v,y.v,a.v,f.v,sg,n.v)==Setup::Ready,"unexpected curve setup classification");
        edwards_atkin_morain(old_d.v,old_x.v,old_y.v,sg,n.v);
        require(!mpz_cmp(d.v,old_d.v) && !mpz_cmp(x.v,old_x.v) && !mpz_cmp(y.v,old_y.v),"checked constructor differs from CPU reference");
        identity(state.data()+size_t(i)*9,d.v,x.v,y.v,n.v);sigmas.push_back(sg);
    }
    for(size_t i=0;i<size_t(count)*9;++i)mpz_set(resumed[i],state[i]);
    uint32_t done=0,split=0;std::string error;
    require(ecm_cuda_edwards_stage1(n.v,s.v,state.data(),count,done,256,device,nullptr,nullptr,error)==0,error.c_str());
    require(ecm_cuda_edwards_stage1(n.v,s.v,resumed.data(),count,split,17,device,
                [](uint32_t,void *){return false;},nullptr,error)==2,"slice stop failed");
    // A new invocation uploads the complete ordinary-domain state, just as a
    // restarted worker does. Chunk boundaries deliberately differ.
    require(ecm_cuda_edwards_stage1(n.v,s.v,resumed.data(),count,split,113,device,nullptr,nullptr,error)==0,error.c_str());
    for(size_t i=0;i<size_t(count)*9;++i)require(!mpz_cmp(state[i],resumed[i]),"resumed result differs from uninterrupted run");
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
    const auto name="m"+std::to_string(bits)+(other?"x"+std::to_string(other):"")+"_b"+std::to_string(b1);
    std::ofstream file(out/(name+".save"),std::ios::binary);file<<lines;file.close();require(bool(file),"save fixture write failed");
    std::printf("PASS %s device=%d curves=%u scalar_bits=%u split_resume=17/113 cpu_match=1\n",name.c_str(),device,count,done);
}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: edwards_gpu_stage1_probe DEVICE OUTPUT_DIRECTORY");
        const int device=std::atoi(argv[1]);std::filesystem::path out(argv[2]);std::filesystem::create_directories(out);
        run(127,20,33,17,device,out);
        run(127,20,4,17,device,out,89);
        run(347,1000,4,20260922,device,out);
        run(677,1000,4,20260922,device,out);
        run(991,1000,4,20260922,device,out);
        run(2203,1000,4,20260922,device,out);
        run(4003,1000,4,20260922,device,out);
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
