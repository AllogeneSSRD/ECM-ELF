#include "checkpoint.h"
#include "host_math.h"
#include "../../../../storage/save/ecm_edwards_io.h"
#include "ecm/core/stage1/scalar.h"
#include "../../../../storage/checkpoint/ecm_stage1_binary.h"
#include "ecm/platform/logging.h"
#include <cstring>
#include <stdexcept>
namespace ed_gpu_driver {
static void binary_value(std::string &out,mpz_srcptr value) {
    if(mpz_sgn(value)<0)throw std::runtime_error("negative Edwards checkpoint value");
    const auto size=mpz_sgn(value)?(mpz_sizeinbase(value,2)+7)/8:0;
    if(size>UINT32_MAX)throw std::runtime_error("oversized Edwards checkpoint value");
    ecm_stage1_binary::u32(out,uint32_t(size));const auto at=out.size();out.resize(at+size);
    if(size)mpz_export(out.data()+at,nullptr,-1,1,0,0,value);
}
static void binary_value(ecm_stage1_binary::Reader &in,mpz_ptr value,size_t maximum) {
    const auto size=in.u32();if(size>maximum)throw std::runtime_error("oversized Edwards checkpoint value");
    auto bytes=in.take(size);
    if(size && !bytes.back())throw std::runtime_error("noncanonical Edwards checkpoint value");
    mpz_import(value,size,-1,1,0,0,bytes.data());
}
static uint64_t scalar_fingerprint(mpz_srcptr scalar) {
    std::string bytes;binary_value(bytes,scalar);return hash(bytes);
}
std::string checkpoint_body(Batch &b,const mpz_t n,const mpz_t scalar,uint64_t b1) {
    using namespace ecm_stage1_binary;
    std::string out("ECMEDB3",8);u32(out,3);u32(out,b.window);u32(out,uint32_t(b.doubling));
    u32(out,b.count);u32(out,b.done);u32(out,0);u64(out,b1);
    u64(out,mpz_sizeinbase(scalar,2));u64(out,scalar_fingerprint(scalar));binary_value(out,n);
    for(uint32_t i=0;i<b.count;++i) {
        u64(out,b.sigmas[i]);u32(out,uint32_t(b.hit[i]));binary_value(out,b.factors[i]);
        for(unsigned j=0;j<9;++j)binary_value(out,b.state[size_t(i)*9+j]);
    }
    u64(out,hash(out));return out;
}
bool save_checkpoint(const std::string &path,Batch &b,const mpz_t n,const mpz_t scalar,uint64_t b1) {
    const auto body=checkpoint_body(b,n,scalar,b1);
    return ecm_stage1_write_blob(path,body);
}
bool restore_checkpoint(const std::string &path,Batch &b,const mpz_t n,const mpz_t scalar,
                    uint64_t b1,bool fixed,bool window_fixed,bool doubling_fixed,int mode_fixed) {
    for(const auto &candidate:ecm_stage1_checkpoint_candidates(path)) {
        std::string text;if(!ecm_stage1_read_blob(candidate,text))continue;
        Batch temp(b.count);unsigned window=0;int doubling=0;bool valid=true;
        if(text.size()>=8 && !std::memcmp(text.data(),"ECMEDB3",8)) {
            using namespace ecm_stage1_binary;
            try {
                if(text.size()<68)continue;
                const auto body=std::string_view(text).substr(0,text.size()-8);
                Reader footer(std::string_view(text).substr(text.size()-8));
                if(footer.u64()!=hash(std::string(body)))continue;
                Reader data(body);data.take(8);if(data.u32()!=3)continue;
                window=data.u32();doubling=int(data.u32());
                if(data.u32()!=b.count)continue;temp.done=data.u32();
                if(data.u32()!=0 || data.u64()!=b1 || data.u64()!=mpz_sizeinbase(scalar,2) ||
                   data.u64()!=scalar_fingerprint(scalar))continue;
                Integer saved_n;const auto maximum=(mpz_sizeinbase(n,2)+7)/8;
                binary_value(data,saved_n.v,maximum);if(mpz_cmp(saved_n.v,n))continue;
                if(b.count>data.left()/52)continue;
                for(uint32_t i=0;i<b.count;++i) {
                    temp.sigmas[i]=data.u64();temp.hit[i]=int(data.u32());
                    binary_value(data,temp.factors[i],maximum);
                    for(unsigned j=0;j<9;++j)binary_value(data,temp.state[size_t(i)*9+j],maximum);
                }
                if(data.left())continue;
            }catch(const std::runtime_error &){continue;}
        } else {
        const auto end=text.rfind("CHECKSUM ");if(end==std::string::npos)continue;
        const auto body=text.substr(0,end);std::istringstream check(text.substr(end+9));uint64_t sum;std::string extra;
        if(!(check>>sum) || (check>>extra) || sum!=hash(body))continue;
        std::istringstream data(body);std::string tag,nh,sh,fh;uint64_t bound;uint32_t count,done;
        if(!(data>>tag))continue;
        if(tag=="ECM_GPU_EDWARDS_NAF_V2") {
            if(!(data>>window>>doubling) || window<2 || window>12 || (doubling!=4 && doubling!=5))continue;
        } else if(tag!="ECM_GPU_EDWARDS_BINARY_V1")continue;
        if(!(data>>nh>>sh>>bound>>count>>done) || nh!=hex(n) || sh!=hex(scalar) || bound!=b1 || count!=b.count)continue;
        temp.done=done;
        for(uint32_t i=0;i<count && valid;++i) {
            if(!(data>>temp.sigmas[i]>>temp.hit[i]>>fh) ||
               (fixed && temp.sigmas[i]!=b.sigmas[i]) || (temp.hit[i]!=0 && temp.hit[i]!=1) ||
               mpz_set_str(temp.factors[i],fh.c_str(),16)){valid=false;break;}
            for(unsigned j=0;j<9;++j){if(!(data>>fh) || mpz_set_str(temp.state[size_t(i)*9+j],fh.c_str(),16)){valid=false;break;}}
        }
        if(!valid || (data>>extra))continue;
        }
        if((window && (window<2 || window>12 || (doubling!=4 && doubling!=5))) ||
           (!window && doubling!=0))continue;
        const auto total=window?EcmEdwardsNaf(scalar,window).bits:mpz_sizeinbase(scalar,2);
        if(temp.done>total)continue;
        for(uint32_t i=0;i<b.count;++i)
            if((fixed && temp.sigmas[i]!=b.sigmas[i]) || (temp.hit[i]!=0 && temp.hit[i]!=1))valid=false;
        if(!valid)continue;
        temp.window=window;temp.doubling=doubling;
        Batch reference(b.count);reference.sigmas=temp.sigmas;std::string error;
        if(!initialize(reference,n,error))continue;
        for(uint32_t i=0;i<b.count && valid;++i) {
            if(reference.hit[i]!=temp.hit[i] || mpz_cmp(reference.factors[i],temp.factors[i])){valid=false;break;}
            for(unsigned j=0;j<5;++j)if(mpz_cmp(reference.state[size_t(i)*9+j],temp.state[size_t(i)*9+j]))valid=false;
            if(!point_valid(n,temp.state[size_t(i)*9],temp.state.data()+size_t(i)*9+5))valid=false;
        }
        if(!valid)continue;
        if((window_fixed && b.window!=window) || (doubling_fixed && b.doubling!=doubling) ||
           (mode_fixed>=0 && bool(mode_fixed)!=bool(window)))
            throw std::runtime_error("GPU Edwards checkpoint algorithm differs from explicit NAF/doubling selection");
        b.window=window;b.doubling=doubling;
        b.done=temp.done;b.sigmas=temp.sigmas;b.hit=temp.hit;
        for(uint32_t i=0;i<b.count;++i)mpz_set(b.factors[i],temp.factors[i]);
        for(size_t i=0;i<size_t(b.count)*9;++i)mpz_set(b.state[i],temp.state[i]);
        if(candidate!=path)ecm_stage1_mark_bad_primary(path);
        ecm_ts_fprintf(stdout,"GPU Edwards resume: columns=%u window=%u doubling=%d path=%s\n",b.done,b.window,b.doubling,candidate.c_str());return true;
    }
    return false;
}
}
