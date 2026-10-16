// Included by the shared Stage1 driver after its save-name helpers.
#if ECM_HAS_CUDA_EDWARDS
#include "../cuda/ecm_cuda_edwards.h"
#include "../cuda/ecm_cuda_edwards_naf.h"
#include "ecm_edwards_naf.h"
#include "ecm_edwards_io.h"
#include <iterator>
#include <stdexcept>

namespace ed_gpu_driver {
using namespace ecm_edwards;
struct Batch {
    uint32_t count,done=0;
    unsigned window=6;
    int doubling=4;
    Array state,factors;
    std::vector<uint64_t> sigmas;
    std::vector<int> hit;
    explicit Batch(uint32_t c):count(c),state(size_t(c)*9),factors(c),sigmas(c),hit(c,0){}
};
static bool initialize(Batch &b,const mpz_t n,std::string &error) {
    Integer d,px,py,a24,f;
    for(uint32_t i=0;i<b.count;++i) {
        const auto rc=construct(d.v,px.v,py.v,a24.v,f.v,b.sigmas[i],n);
        auto *s=b.state.data()+size_t(i)*9;
        if(rc==Setup::Degenerate){error="degenerate Atkin-Morain sigma="+std::to_string(b.sigmas[i]);return false;}
        if(rc==Setup::Factor) {
            b.hit[i]=1;mpz_set(b.factors[i],f.v);
            // A harmless identity lane keeps batch indexing stable.
            mpz_set_ui(s[2],1);mpz_set_ui(s[3],1);
        } else {
            mpz_set(s[0],d.v);mpz_set(s[1],px.v);mpz_set(s[2],py.v);
            mpz_set_ui(s[3],1);mpz_mul(s[4],px.v,py.v);mpz_mod(s[4],s[4],n);
        }
        mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
    }
    return true;
}
static std::string checkpoint_body(Batch &b,const mpz_t n,const mpz_t scalar,uint64_t b1) {
    std::ostringstream out;
    if(b.window)out<<"ECM_GPU_EDWARDS_NAF_V2\n"<<b.window<<' '<<b.doubling<<'\n';
    else out<<"ECM_GPU_EDWARDS_BINARY_V1\n";
    out<<hex(n)<<'\n'<<hex(scalar)<<'\n'
       <<b1<<' '<<b.count<<' '<<b.done<<'\n';
    for(uint32_t i=0;i<b.count;++i){out<<b.sigmas[i]<<' '<<b.hit[i]<<' '<<hex(b.factors[i]);
        for(unsigned j=0;j<9;++j)out<<' '<<hex(b.state[size_t(i)*9+j]);out<<'\n';}
    return out.str();
}
static bool write_checkpoint(const std::string &path,Batch &b,const mpz_t n,const mpz_t scalar,uint64_t b1) {
    const auto body=checkpoint_body(b,n,scalar,b1),temp=ecm_stage1_checkpoint_temp_path(path);
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);
    out<<body<<"CHECKSUM "<<hash(body)<<'\n';out.close();
    return bool(out) && ecm_stage1_commit_checkpoint(path,temp);
}
static bool restore(const std::string &path,Batch &b,const mpz_t n,const mpz_t scalar,
                    uint64_t b1,bool fixed,bool window_fixed,bool doubling_fixed,int mode_fixed) {
    for(const auto &candidate:ecm_stage1_checkpoint_candidates(path)) {
        std::ifstream in(candidate,std::ios::binary);
        if(!in)continue;
        const std::string text((std::istreambuf_iterator<char>(in)),{});
        const auto end=text.rfind("CHECKSUM ");if(end==std::string::npos)continue;
        const auto body=text.substr(0,end);std::istringstream check(text.substr(end+9));uint64_t sum;std::string extra;
        if(!(check>>sum) || (check>>extra) || sum!=hash(body))continue;
        std::istringstream data(body);std::string tag,nh,sh,fh;uint64_t bound;uint32_t count,done;
        unsigned window=0;int doubling=0;
        if(!(data>>tag))continue;
        if(tag=="ECM_GPU_EDWARDS_NAF_V2") {
            if(!(data>>window>>doubling) || window<2 || window>12 || (doubling!=4 && doubling!=5))continue;
        } else if(tag!="ECM_GPU_EDWARDS_BINARY_V1")continue;
        if(!(data>>nh>>sh>>bound>>count>>done) || nh!=hex(n) || sh!=hex(scalar) || bound!=b1 || count!=b.count)continue;
        const auto total=window?EcmEdwardsNaf(scalar,window).bits:mpz_sizeinbase(scalar,2);
        if(done>total)continue;
        Batch temp(count);temp.done=done;temp.window=window;temp.doubling=doubling;bool valid=true;
        for(uint32_t i=0;i<count && valid;++i) {
            if(!(data>>temp.sigmas[i]>>temp.hit[i]>>fh) ||
               (fixed && temp.sigmas[i]!=b.sigmas[i]) || (temp.hit[i]!=0 && temp.hit[i]!=1) ||
               mpz_set_str(temp.factors[i],fh.c_str(),16)){valid=false;break;}
            for(unsigned j=0;j<9;++j){if(!(data>>fh) || mpz_set_str(temp.state[size_t(i)*9+j],fh.c_str(),16)){valid=false;break;}}
        }
        if(!valid || (data>>extra))continue;
        Batch reference(count);reference.sigmas=temp.sigmas;std::string error;
        if(!initialize(reference,n,error))continue;
        for(uint32_t i=0;i<count && valid;++i) {
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
        for(uint32_t i=0;i<count;++i)mpz_set(b.factors[i],temp.factors[i]);
        for(size_t i=0;i<size_t(count)*9;++i)mpz_set(b.state[i],temp.state[i]);
        if(candidate!=path)ecm_stage1_mark_bad_primary(path);
        ecm_ts_fprintf(stdout,"GPU Edwards resume: columns=%u window=%u doubling=%d path=%s\n",b.done,b.window,b.doubling,candidate.c_str());return true;
    }
    return false;
}
struct Progress {
    Batch &b;mpz_srcptr n,scalar;const Stage1RunOptions &opt;std::string path;
    uint64_t b1;long long last;bool failed=false;
};
static bool progress(uint32_t done,void *context) {
    auto &p=*static_cast<Progress *>(context);p.b.done=done;
    const bool stop=ecm_stage1_stop_requested();const auto now=stage1_now_ms();
    if(!stop && (!p.opt.ckpt_ms || now-p.last<p.opt.ckpt_ms))return true;
    if(!p.path.empty() && !write_checkpoint(p.path,p.b,p.n,p.scalar,p.b1)) {
        p.failed=true;ecm_ts_fprintf(stderr,"ERROR: GPU Edwards checkpoint write failed: %s\n",p.path.c_str());return false;
    }
    p.last=now;
    if(stop)ecm_ts_fprintf(stdout,"GPU Edwards stopped at %s %u\n",p.b.window?"NAF column":"binary bit",done);
    return !stop;
}
static bool publish(const std::string &path,const std::string &lines,bool append) {
    const auto temp=ecm_stage1_checkpoint_temp_path(path);
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);if(!out)return false;
    if(append && access(path.c_str(),0)==0){std::ifstream old(path,std::ios::binary);if(!old)return false;if(old.peek()!=std::char_traits<char>::eof())out<<old.rdbuf();if(old.bad())return false;}
    out<<lines;out.close();return bool(out) && ecm_stage1_commit_checkpoint(path,temp);
}
}

static int run_edwards_gpu_stage1(const mpz_t n,double b1,double b2,uint32_t curves,
        const std::string &savefilename,bool saveappend,const Stage1RunOptions &opt,Stage1RunResult *out) {
    using namespace ed_gpu_driver;
    *out=Stage1RunResult{};out->curves=curves;
    if(!curves)return ECM_ERROR;
    try {
        if(opt.naf_w!=0 && (opt.naf_w<2 || opt.naf_w>12))throw std::runtime_error("GPU Edwards NAF window must be 0 (auto) or 2..12");
        Integer scalar;if(!compute_batch_s(scalar.v,b1,48))throw std::runtime_error("cannot build Edwards exponent");
        if(!opt.tmp_dir.empty() && !ensure_dir(opt.tmp_dir))throw std::runtime_error("cannot create Edwards save directory");
        Batch batch(curves);
        batch.window=opt.naf_w?unsigned(opt.naf_w):6;
        int mode_fixed=-1;
        if(const char *mode=std::getenv("ECM_EDWARDS_STAGE1_ALGO")) {
            const std::string value=mode;
            if(value!="binary" && value!="naf")throw std::runtime_error("ECM_EDWARDS_STAGE1_ALGO must be binary or naf");
            mode_fixed=value=="naf"?1:0;
            if(!mode_fixed) {
                if(opt.naf_w)throw std::runtime_error("binary Edwards mode cannot use --naf-w");
                batch.window=0;batch.doubling=0;
            }
        }
        const char *dbl_env=std::getenv("ECM_EDWARDS_DBL_ALGO");
        if(dbl_env) {
            if(!batch.window)throw std::runtime_error("binary Edwards mode cannot select a NAF doubling algorithm");
            const std::string value=dbl_env;
            if(value!="4" && value!="5")throw std::runtime_error("ECM_EDWARDS_DBL_ALGO must be 4 or 5");
            batch.doubling=value=="5"?5:4;
        }
        for(uint32_t i=0;i<curves;++i)batch.sigmas[i]=opt.sigma_fixed?opt.fixed_sigma64+i:random_sigma_u64();
        const auto stem="ed_"+mont_format_save_name(opt.save_name_pattern,n,b1);
        const auto path=opt.tmp_dir.empty()?std::string{}:opt.tmp_dir+"/"+stem+".gpu.ckpt";
        if(path.empty() || !restore(path,batch,n,scalar.v,uint64_t(b1),opt.sigma_fixed,opt.naf_w!=0,dbl_env!=nullptr,mode_fixed)) {
            std::string error;if(!initialize(batch,n,error))throw std::runtime_error(error);
        }
        ecm_ts_fprintf(stdout,"GPU Edwards Stage1: device=%d curves=%u N_bits=%zu scalar_bits=%zu window=%u doubling=%d\n",
                       opt.device_index,curves,mpz_sizeinbase(n,2),mpz_sizeinbase(scalar.v,2),batch.window,batch.doubling);
        uint32_t container=0,tpi=0;
        if(!ecm_cuda_edwards_tier(mpz_sizeinbase(n,2),container,tpi))
            throw std::runtime_error("no compiled GPU Edwards container with one spare bit for target");
        ecm_ts_fprintf(stdout,"GPU Edwards geometry: container_bits=%u tpi=%u tpb=128 blocks=%u\n",
                       container,tpi,(curves+128/tpi-1)/(128/tpi));
        Progress p{batch,n,scalar.v,opt,path,uint64_t(b1),stage1_now_ms()};std::string error;
        EcmEdwardsNafMetrics metrics;
        const auto arithmetic_start=std::chrono::steady_clock::now();
        const int rc=batch.window?
            ecm_cuda_edwards_stage1_naf(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error,batch.window,batch.doubling,&metrics):
            ecm_cuda_edwards_stage1(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error);
        const double arithmetic_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-arithmetic_start).count();
        ecm_ts_fprintf(stdout,"GPU Edwards arithmetic: algorithm=%s api_seconds=%.9f\n",batch.window?"naf":"binary",arithmetic_seconds);
        if(batch.window)ecm_ts_fprintf(stdout,"GPU Edwards NAF: columns=%u additions=%u container_bits=%u tpi=%u dictionary_bytes=%llu dictionary_gpu_ms=%.6f loop_gpu_ms=%.6f launches=%u\n",
            metrics.columns,metrics.additions,metrics.container_bits,metrics.tpi,
            static_cast<unsigned long long>(metrics.dictionary_bytes),metrics.dictionary_gpu_ms,metrics.loop_gpu_ms,metrics.launches);
        if(rc<0)throw std::runtime_error(error);
        if(rc==2){out->publish_failed=p.failed;return ECM_ERROR;}
        Integer x,z,g,inv;std::string lines;bool found=false;
        for(uint32_t i=0;i<curves;++i) {
            auto *s=batch.state.data()+size_t(i)*9;
            if(!batch.hit[i]) {
                if(!point_valid(n,s[0],s+5))throw std::runtime_error("invalid GPU Edwards final point");
                mpz_add(x.v,s[7],s[6]);mpz_mod(x.v,x.v,n);
                mpz_sub(z.v,s[7],s[6]);mpz_mod(z.v,z.v,n);mpz_gcd(g.v,z.v,n);
                if(mpz_cmp_ui(g.v,1)>0 && mpz_cmp(g.v,n)<0){batch.hit[i]=1;mpz_set(batch.factors[i],g.v);}
                else if(mpz_cmp(g.v,n)==0) {
                    ecm_ts_fprintf(stdout,"Edwards identity: sigma=%llu gcd(Qz,N)=N, no Stage2 point\n",
                                   static_cast<unsigned long long>(batch.sigmas[i]));
                    mpz_set_ui(batch.factors[i],0);continue;
                }
                else {mpz_invert(inv.v,z.v,n);mpz_mul(x.v,x.v,inv.v);mpz_mod(x.v,x.v,n);}
            }
            if(batch.hit[i]){mpz_set(x.v,batch.factors[i]);found=true;}
            else mpz_set(batch.factors[i],x.v);
            lines+=save_line(n,x.v,batch.sigmas[i],uint64_t(b1),batch.hit[i]!=0);
        }
        const auto save=savefilename.empty()?(opt.tmp_dir.empty()?std::string{}:opt.tmp_dir+"/"+stem):savefilename;
        if(!save.empty() && !publish(save,lines,saveappend)) {
            out->publish_failed=true;throw std::runtime_error("cannot publish Edwards save "+save);
        }
        // Only remove the resume state after a durable handoff (or an explicitly
        // fileless run); an I/O failure must keep the queue task unfinished.
        if(!path.empty())ecm_stage1_remove_checkpoints(path);
        auto *factors=static_cast<mpz_t *>(malloc(sizeof(mpz_t)*curves));
        auto *hits=static_cast<int *>(malloc(sizeof(int)*curves));
        auto *sigmas=static_cast<uint64_t *>(malloc(sizeof(uint64_t)*curves));
        if(!factors || !hits || !sigmas){free(factors);free(hits);free(sigmas);throw std::bad_alloc();}
        for(uint32_t i=0;i<curves;++i){mpz_init_set(factors[i],batch.factors[i]);hits[i]=batch.hit[i]?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;sigmas[i]=batch.sigmas[i];}
        out->factors=factors;out->array_found=hits;out->sigmas=sigmas;
        out->firstsigma64=sigmas[0];out->firstsigma=uint32_t(sigmas[0]);
        out->ret=found?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
        if(!save.empty())ecm_ts_fprintf(stdout,"Edwards Stage1 save: %s (%u curves)\n",save.c_str(),curves);
        (void)b2;return out->ret;
    } catch(const std::exception &e) {
        out->prepare_failed=!out->publish_failed;
        ecm_ts_fprintf(stderr,"ERROR: GPU Edwards Stage1: %s\n",e.what());return ECM_ERROR;
    }
}
#endif
