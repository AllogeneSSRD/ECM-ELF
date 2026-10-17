// Included by the shared Stage1 driver after its save-name helpers.
#if ECM_HAS_CUDA_EDWARDS
#include "../cuda/ecm_cuda_edwards.h"
#include "../cuda/ecm_cuda_edwards_naf.h"
#include "ecm_edwards_naf.h"
#include "ecm_edwards_io.h"
#include "ecm_stage1_binary.h"
#include <iterator>
#include <stdexcept>

#include "ecm_edwards_gpu_checkpoint.h"

namespace ed_gpu_driver {
struct Progress {
    Batch &b;mpz_srcptr n,scalar;const Stage1RunOptions &opt;std::string path;
    uint64_t b1,total;long long last;const EcmEdwardsNafMetrics *metrics;bool failed=false;
};
static bool progress(uint32_t done,void *context) {
    auto &p=*static_cast<Progress *>(context);p.b.done=done;
    const auto total=p.b.window?p.metrics->columns:p.total;
    stage1_progress_set(total?double(done)*p.b.count/total:0,p.b.count);
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
    return ecm_edwards::publish_save(path,lines,append);
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
        for(uint32_t i=0;i<curves;++i)batch.sigmas[i]=ecm_stage1_batch_sigma(i,opt.sigma_fixed?opt.fixed_sigma64+i:random_sigma_u64());
        const auto stem=mont_format_save_name(opt.save_name_pattern,n,b1,&opt);
        const auto path=opt.tmp_dir.empty()?std::string{}:ecm_stage1_curve_record(0);
        bool restored=!path.empty() && restore(path,batch,n,scalar.v,uint64_t(b1),opt.sigma_fixed,opt.naf_w!=0,dbl_env!=nullptr,mode_fixed);
        std::string existing;
        if(!restored && !path.empty() && !ecm_stage1_read_blob(path,existing)) {
            // Pre-container GPU Edwards used ed_<old-pattern>.save.gpu.ckpt.
            for(const auto &pattern:{std::string("m{n}_{b1}.save"),opt.save_name_pattern}) {
                auto old=mont_format_save_name(pattern,n,b1);
                if(old.rfind("m0_",0)==0)old.erase(0,3);
                const auto legacy=(std::filesystem::path(opt.tmp_dir)/("ed_"+old+".gpu.ckpt")).string();
                Batch saved(curves);saved.sigmas=batch.sigmas;saved.window=batch.window;saved.doubling=batch.doubling;
                if(restore(legacy,saved,n,scalar.v,uint64_t(b1),opt.sigma_fixed,opt.naf_w!=0,dbl_env!=nullptr,mode_fixed)) {
                    if(stage1_results_complete(opt,n,uint64_t(b1),true,0,saved.sigmas))continue;
                    batch.done=saved.done;batch.window=saved.window;batch.doubling=saved.doubling;
                    batch.sigmas=saved.sigmas;batch.hit=saved.hit;
                    for(uint32_t i=0;i<curves;++i)mpz_set(batch.factors[i],saved.factors[i]);
                    for(size_t i=0;i<size_t(curves)*9;++i)mpz_set(batch.state[i],saved.state[i]);
                    if(!ecm_stage1_import_gpu_payload(checkpoint_body(batch,n,scalar.v,uint64_t(b1)),batch.sigmas))
                        throw std::runtime_error("cannot import legacy Edwards checkpoint");
                    ecm_ts_fprintf(stdout,"Stage1 legacy Edwards checkpoint imported: path=%s original_retained=1\n",legacy.c_str());
                    restored=true;break;
                }
            }
        }
        if(!restored) {
            if(!path.empty() && ecm_stage1_read_blob(path,existing))throw std::runtime_error("GPU Edwards batch payload is invalid or unsupported");
            std::string error;if(!initialize(batch,n,error))throw std::runtime_error(error);
        }
        ecm_ts_fprintf(stdout,"GPU Edwards Stage1: device=%d curves=%u N_bits=%zu scalar_bits=%zu window=%u doubling=%d\n",
                       opt.device_index,curves,mpz_sizeinbase(n,2),mpz_sizeinbase(scalar.v,2),batch.window,batch.doubling);
        uint32_t container=0,tpi=0;
        if(!ecm_cuda_edwards_tier(mpz_sizeinbase(n,2),container,tpi))
            throw std::runtime_error("no compiled GPU Edwards container with one spare bit for target");
        ecm_ts_fprintf(stdout,"GPU Edwards geometry: container_bits=%u tpi=%u tpb=128 blocks=%u\n",
                       container,tpi,(curves+128/tpi-1)/(128/tpi));
        auto *bar=stage1_bar_create();stage1_progress_set(0,curves);
        EcmEdwardsNafMetrics metrics;
        Progress p{batch,n,scalar.v,opt,path,uint64_t(b1),mpz_sizeinbase(scalar.v,2),stage1_now_ms(),&metrics};std::string error;
        const auto arithmetic_start=std::chrono::steady_clock::now();
        const int rc=batch.window?
            ecm_cuda_edwards_stage1_naf(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error,batch.window,batch.doubling,&metrics):
            ecm_cuda_edwards_stage1(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error);
        const double arithmetic_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-arithmetic_start).count();
        if(rc==0)stage1_progress_set(curves,curves);
        stage1_bar_destroy(bar);
        ecm_ts_fprintf(stdout,"GPU Edwards arithmetic: algorithm=%s api_seconds=%.9f\n",batch.window?"naf":"binary",arithmetic_seconds);
        if(batch.window)ecm_ts_fprintf(stdout,"GPU Edwards NAF: columns=%u additions=%u container_bits=%u tpi=%u dictionary_bytes=%llu dictionary_gpu_ms=%.6f loop_gpu_ms=%.6f launches=%u\n",
            metrics.columns,metrics.additions,metrics.container_bits,metrics.tpi,
            static_cast<unsigned long long>(metrics.dictionary_bytes),metrics.dictionary_gpu_ms,metrics.loop_gpu_ms,metrics.launches);
        if(rc<0)throw std::runtime_error(error);
        if(rc==2){out->publish_failed=p.failed;return ECM_ERROR;}
        if(!path.empty() && !write_checkpoint(path,batch,n,scalar.v,uint64_t(b1))) {
            out->publish_failed=true;throw std::runtime_error("cannot retain completed Edwards batch");
        }
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
        const auto save=savefilename.empty()?(opt.tmp_dir.empty()?std::string{}:opt.tmp_dir+"/")+stem:savefilename;
        if(!publish(save,lines,saveappend)) {
            out->publish_failed=true;throw std::runtime_error("cannot publish Edwards save "+save);
        }
        // Only remove the resume state after publishing the completed batch.
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
