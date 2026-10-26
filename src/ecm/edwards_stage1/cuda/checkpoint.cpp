#include "../../../io/stage1_checkpoint.h"
#include "checkpoint.h"
#include "host_math.h"
#include "ecm/io/save.h"
#include "ecm/stage1/scalar.h"
#include "../../../io/stage1_binary.h"
#include "ecm/platform/logging.h"
#include "../../../io/files.h"
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <utility>
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
    std::string bytes;binary_value(bytes,scalar);return ecm_io::fingerprint(bytes);
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
    u64(out,ecm_io::fingerprint(out));return out;
}
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &path,Batch &b,const mpz_t n,const mpz_t scalar,uint64_t b1) {
    const auto body=checkpoint_body(b,n,scalar,b1);
    return checkpoint.write_blob(path,body);
}
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &text,Batch &b,
        mpz_srcptr n,mpz_srcptr scalar,uint64_t b1,bool fixed,bool window_fixed,bool doubling_fixed,int mode_fixed) {
    using namespace ecm_stage1_checkpoint;
        Batch temp(b.count);unsigned window=0;int doubling=0;bool valid=true;
        if(text.size()>=8 && !std::memcmp(text.data(),"ECMEDB3",8)) {
            using namespace ecm_stage1_binary;
            try {
                if(text.size()<68)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
                const auto body=std::string_view(text).substr(0,text.size()-8);
                Reader footer(std::string_view(text).substr(text.size()-8));
                if(footer.u64()!=ecm_io::fingerprint(std::string(body)))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
                Reader data(body);data.take(8);if(data.u32()!=3)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
                window=data.u32();doubling=int(data.u32());
                if(data.u32()!=b.count)return {RestoreState::Incompatible,"GPU Edwards checkpoint curve count differs"};temp.done=data.u32();
                if(data.u32()!=0)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint flags"};
                if(data.u64()!=b1 || data.u64()!=mpz_sizeinbase(scalar,2) || data.u64()!=scalar_fingerprint(scalar))
                    return {RestoreState::Incompatible,"GPU Edwards checkpoint scalar or B1 differs"};
                Integer saved_n;const auto maximum=(mpz_sizeinbase(n,2)+7)/8;
                binary_value(data,saved_n.v,maximum);if(mpz_cmp(saved_n.v,n))return {RestoreState::Incompatible,"GPU Edwards checkpoint target differs"};
                if(b.count>data.left()/52)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
                for(uint32_t i=0;i<b.count;++i) {
                    temp.sigmas[i]=data.u64();temp.hit[i]=int(data.u32());
                    binary_value(data,temp.factors[i],maximum);
                    for(unsigned j=0;j<9;++j)binary_value(data,temp.state[size_t(i)*9+j],maximum);
                }
                if(data.left())return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
            }catch(const std::runtime_error &){return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};}
        } else {
        const auto end=text.rfind("CHECKSUM ");if(end==std::string::npos)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        const auto body=text.substr(0,end);std::istringstream check(text.substr(end+9));uint64_t sum;std::string extra;
        if(!(check>>sum) || (check>>extra) || sum!=ecm_io::fingerprint(body))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        std::istringstream data(body);std::string tag,nh,sh,fh;uint64_t bound;uint32_t count,done;
        if(!(data>>tag))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        if(tag=="ECM_GPU_EDWARDS_NAF_V2") {
            if(!(data>>window>>doubling) || window<2 || window>12 || (doubling!=4 && doubling!=5))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        } else if(tag!="ECM_GPU_EDWARDS_BINARY_V1")return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        if(!(data>>nh>>sh>>bound>>count>>done))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        if(nh!=hex(n) || sh!=hex(scalar) || bound!=b1 || count!=b.count)
            return {RestoreState::Incompatible,"GPU Edwards checkpoint target/scalar/B1/curve count differs"};
        temp.done=done;
        for(uint32_t i=0;i<count && valid;++i) {
            if(!(data>>temp.sigmas[i]>>temp.hit[i]>>fh) ||
               (temp.hit[i]!=0 && temp.hit[i]!=1) ||
               mpz_set_str(temp.factors[i],fh.c_str(),16)){valid=false;break;}
            for(unsigned j=0;j<9;++j){if(!(data>>fh) || mpz_set_str(temp.state[size_t(i)*9+j],fh.c_str(),16)){valid=false;break;}}
        }
        if(!valid || (data>>extra))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        }
        if((window && (window<2 || window>12 || (doubling!=4 && doubling!=5))) ||
           (!window && doubling!=0))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        const auto total=window?EcmEdwardsNaf(scalar,window).bits:mpz_sizeinbase(scalar,2);
        if(temp.done>total)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        for(uint32_t i=0;i<b.count;++i)
            if((temp.hit[i]!=0 && temp.hit[i]!=1))valid=false;
        if(!valid)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        temp.window=window;temp.doubling=doubling;
        Batch reference(b.count);reference.sigmas=temp.sigmas;std::string error;
        if(!initialize(reference,n,error))return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        for(uint32_t i=0;i<b.count && valid;++i) {
            if(reference.hit[i]!=temp.hit[i] || mpz_cmp(reference.factors[i],temp.factors[i])){valid=false;break;}
            for(unsigned j=0;j<5;++j)if(mpz_cmp(reference.state[size_t(i)*9+j],temp.state[size_t(i)*9+j]))valid=false;
            if(!point_valid(n,temp.state[size_t(i)*9],temp.state.data()+size_t(i)*9+5))valid=false;
        }
        if(!valid)return {RestoreState::Corrupt,"invalid GPU Edwards checkpoint payload"};
        if(fixed)for(uint32_t i=0;i<b.count;++i)
            if(temp.sigmas[i]!=b.sigmas[i])return {RestoreState::Incompatible,"GPU Edwards checkpoint sigma differs from requested sigma"};
        if((window_fixed && b.window!=window) || (doubling_fixed && b.doubling!=doubling) ||
           (mode_fixed>=0 && bool(mode_fixed)!=bool(window)))
            return {RestoreState::Incompatible,"GPU Edwards checkpoint algorithm differs from explicit NAF/doubling selection"};
        b.window=window;b.doubling=doubling;
        b.done=temp.done;b.sigmas=temp.sigmas;b.hit=temp.hit;
        for(uint32_t i=0;i<b.count;++i)mpz_set(b.factors[i],temp.factors[i]);
        for(size_t i=0;i<size_t(b.count)*9;++i)mpz_set(b.state[i],temp.state[i]);
    return {b.done==total?
        RestoreState::AlreadyComputed:RestoreState::Resumed,{}};
}

ecm_stage1_checkpoint::RestoreResult restore_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &path,Batch &b,
        mpz_srcptr n,mpz_srcptr scalar,uint64_t b1,bool fixed,bool window_fixed,bool doubling_fixed,int mode_fixed) {
    using namespace ecm_stage1_checkpoint;
    RestoreResult result;
    for(const auto &candidate:checkpoint.candidates(path)) {
        std::string text;const auto read=checkpoint.read_checkpoint_blob(candidate,text);
        if(read==BlobRead::Missing)continue;
        if(read==BlobRead::IoFailure){result={RestoreState::IoFailure,"cannot read GPU Edwards checkpoint"};continue;}
        auto decoded=decode_checkpoint(text,b,n,scalar,b1,fixed,window_fixed,doubling_fixed,mode_fixed);
        if(decoded.state==RestoreState::Incompatible)return decoded;
        if(!decoded.restored()){result=std::move(decoded);continue;}
        if(candidate!=path)checkpoint.mark_bad_primary(path);
        ecm_ts_fprintf(stdout,"GPU Edwards resume: columns=%u window=%u doubling=%d path=%s\n",b.done,b.window,b.doubling,candidate.c_str());
        return decoded;
    }
    return result;
}
} // namespace ed_gpu_driver
