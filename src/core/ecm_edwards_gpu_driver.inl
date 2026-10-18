// Included by the shared Stage1 driver after its save-name helpers.
#if ECM_HAS_CUDA_EDWARDS
#include "../cuda/ecm_cuda_edwards.h"
#include "../cuda/ecm_cuda_edwards_naf.h"
#include "ecm_edwards_naf.h"
#include "ecm_edwards_io.h"
#include "ecm_stage1_binary.h"
#include "ecm_stage1_gpu_progress.h"
#include <iterator>
#include <stdexcept>

#include "ecm_edwards_gpu_checkpoint.h"

namespace ed_gpu_driver {
struct Progress {
    using Clock=std::chrono::steady_clock;
    Batch &b;mpz_srcptr n,scalar;const Stage1RunOptions &opt;std::string path;
    uint64_t b1,total;long long last;const EcmEdwardsNafMetrics *metrics;bool failed=false;
    Clock::time_point initialization_start,loop_start,sample_start,last_print;
    uint64_t sampled=0,delta=0;
    bool printed=false,line_open=false;
    ecm_stage1_gpu_progress::SpeedWindow speed;
    Progress(Batch &batch,mpz_srcptr modulus,mpz_srcptr exponent,const Stage1RunOptions &options,
             const std::string &checkpoint,uint64_t bound,uint64_t bits,
             const EcmEdwardsNafMetrics *measurements,Clock::time_point start)
        :b(batch),n(modulus),scalar(exponent),opt(options),path(checkpoint),b1(bound),total(bits),
         last(stage1_now_ms()),metrics(measurements),initialization_start(start){}
    void finish() {
        if(line_open){std::fputc('\n',stdout);std::fflush(stdout);line_open=false;}
    }
    void display(Clock::time_point now) {
        const auto text=ecm_stage1_gpu_progress::line(total?100.0*b.done/total:100.0,b.done,delta,
            b.window?"columns":"bits",speed.per_curve(total,b.count),
            std::chrono::duration<double>(now-loop_start).count(),speed.remaining(total,b.done));
        if(stdout_is_tty_local()) {
            std::fprintf(stdout,"\r%s%s%s",ecm_log_progress_color_code(),text.c_str(),ecm_log_progress_color_reset());
            std::fflush(stdout);line_open=true;
        } else ecm_ts_fprintf(stdout,"%s\n",text.c_str());
        printed=true;last_print=now;
    }
};
static void prepared(void *context) {
    auto &p=*static_cast<Progress *>(context);
    const double dictionary=p.metrics->dictionary_wall_ms/1000.0;
    const double preparation=std::chrono::duration<double>(Progress::Clock::now()-p.initialization_start).count();
    ecm_ts_fprintf(stdout,"GPU Edwards initialization: %.3f s\n",(std::max)(0.0,preparation-dictionary));
    ecm_ts_fprintf(stdout,"GPU Edwards dictionary: %.3f s (gpu=%.3f ms)\n",dictionary,p.metrics->dictionary_gpu_ms);
    p.total=p.b.window?p.metrics->columns:p.total;
    // Restored work is the baseline, never a speed sample. Preparation and its
    // console I/O finish before the live loop clock starts.
    p.sampled=p.b.done;p.loop_start=p.sample_start=Progress::Clock::now();
    if(p.b.done==p.total)p.display(p.loop_start);
}
static bool progress(uint32_t done,void *context) {
    auto &p=*static_cast<Progress *>(context);p.b.done=done;
    const auto sample_time=Progress::Clock::now();p.delta=done-p.sampled;
    p.speed.add(p.delta,std::chrono::duration<double,std::milli>(sample_time-p.sample_start).count());
    p.sampled=done;
    const bool stop=ecm_stage1_stop_requested();const auto now=stage1_now_ms();
    if(!p.printed || stop || done==p.total || sample_time-p.last_print>=std::chrono::milliseconds(200))
        p.display(sample_time);
    if(stop || (p.opt.ckpt_ms && now-p.last>=p.opt.ckpt_ms)) {
        if(!p.path.empty() && !write_checkpoint(p.path,p.b,p.n,p.scalar,p.b1)) {
            p.finish();p.failed=true;
            ecm_ts_fprintf(stderr,"ERROR: GPU Edwards checkpoint write failed: %s\n",p.path.c_str());return false;
        }
        p.last=now;
    }
    // The rate samples include each slice and its state readback, excluding
    // checkpoint/console callback time. Elapsed is the complete loop wall time.
    p.sample_start=Progress::Clock::now();
    if(stop){p.finish();ecm_ts_fprintf(stdout,"GPU Edwards stopped at %s %u\n",p.b.window?"NAF column":"binary bit",done);}
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
        const auto initialization_start=Progress::Clock::now();
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
        EcmEdwardsNafMetrics metrics;
        Progress p{batch,n,scalar.v,opt,path,uint64_t(b1),mpz_sizeinbase(scalar.v,2),&metrics,initialization_start};std::string error;
        const auto arithmetic_start=std::chrono::steady_clock::now();
        const int rc=batch.window?
            ecm_cuda_edwards_stage1_naf(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error,batch.window,batch.doubling,&metrics,prepared):
            ecm_cuda_edwards_stage1(n,scalar.v,batch.state.data(),curves,batch.done,256,
                opt.device_index,progress,&p,error,prepared);
        const double arithmetic_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-arithmetic_start).count();
        p.finish();
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
            if(batch.hit[i]){mpz_set(x.v,batch.factors[i]);found=true;
                if(!stage1_publish_factor(opt,n,uint64_t(b1),batch.sigmas[i],batch.factors[i])){out->publish_failed=true;throw std::runtime_error("cannot publish Edwards factor");}}
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
