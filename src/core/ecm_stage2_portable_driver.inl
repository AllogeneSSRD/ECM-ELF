// Included inside s2prod after the save/worker/planning helpers.
namespace pt=ecm_stage2::tune::portable;
namespace et=ecm_stage2::tune;
struct PortableQuiet {
    int previous=stage2_log::level;
    PortableQuiet(){ecm_cuda_stage2_set_log_level(stage2_log::quiet);stage2_log::level=stage2_log::quiet;}
    ~PortableQuiet(){ecm_cuda_stage2_set_log_level(previous);stage2_log::level=previous;}
};
std::string portable_identity(const fs::path &evidence) {
    const auto hash=ecm_stage2::sha256_file(executable());
    append(evidence/"identity.jsonl","{\"binary_sha256\":"+json_string(hash)+",\"binary\":"+json_string(executable().string())+"}");
    const auto manifest=executable().parent_path()/"build_manifest.json";
    if(fs::is_regular_file(manifest))fs::copy_file(manifest,evidence/"build_manifest.json",fs::copy_options::none);
    return hash;
}
pt::Profile portable_read(const fs::path &path) {
    if(!fs::is_regular_file(path))return {};
    try {return pt::Profile::load(path);}
    catch(const pt::UnsupportedFormat &) {
        std::cout<<"tune_profile_reset: incompatible development format; calibration will regenerate "<<path.string()<<std::endl;
        return {};
    }
}
et::Fields portable_condition(const Options &o) {
    EcmStage2DeviceInfo info;if(ecm_cuda_stage2_device_info(o.device,&info))throw std::runtime_error("cannot query tuning device");
    et::Fields f={{"uuid_hex",json_string(info.uuid_hex)},{"sm_major",std::to_string(info.major)},
        {"sm_minor",std::to_string(info.minor)},{"cuda_runtime",std::to_string(info.runtime)},
        {"cuda_driver",std::to_string(info.driver)},{"gl_fixed_mode",std::to_string(info.fixed_mode)},
        {"outer_unroll_u",std::to_string(info.outer_unroll_u)},{"add_sub_mask",std::to_string(NTT_GL_ADD_SUB_MASK)},
        {"batch_mb",std::to_string(o.batch)},{"arena_mb",std::to_string(o.arena)},{"fold_mb",std::to_string(o.owner_mb)},
        {"env_condition_tag",json_string(o.condition_tag)}};
    for(const auto &e:tune_environment())f["env_"+e.first]=e.second;
    return f;
}
void configure_short_ntt_batches(const Options &o) {
    // Matching ignores only the dimensions explicitly allowed by tune-ignore;
    // memory admission always uses the current request and live device budget.
    _putenv_s("NTT_S4_SHORT_BATCH_TUNE",o.short_batch_tune?"1":"");
    _putenv_s("NTT_S4_SHORT_BATCH_HASH",o.short_batch_tune?"0":"");
    std::vector<EcmStage2NttBatchSample> samples;uint64_t hash=0;
    if(o.short_batch_tune && !o.tune_profile.empty() && fs::is_regular_file(o.tune_profile)) {
        const auto profile=portable_read(o.tune_profile);const auto current=portable_condition(o);
        const auto ignore=pt::Ignore::parse(o.tune_ignore);ecm_stage2::ShortNttBatchPolicy policy;
        for(const auto &s:profile.ntt)if(pt::ntt_matches(profile.conditions.at(et::uint(s,"condition")),current,ignore))
            policy.observe(et::uint(s,"length"),et::uint(s,"batch"),et::real(s,"mean_seconds"));
        hash=14695981039346656037ull;
        for(const auto &length:policy.seconds)for(const auto &batch:length.second) {
            samples.push_back({length.first,batch.first,batch.second});
            const auto key=std::to_string(length.first)+":"+std::to_string(batch.first)+":"+pt::numeric(batch.second)+";";
            for(unsigned char c:key)hash=(hash^c)*1099511628211ull;
        }
        if(samples.empty())hash=0;
    }
    if(ecm_cuda_stage2_set_short_ntt_batches(samples.data(),samples.size()))
        throw std::runtime_error("short NTT batch policy is unsupported or invalid");
    _putenv_s("NTT_S4_SHORT_BATCH_HASH",o.short_batch_tune?std::to_string(hash).c_str():"");
    if(o.short_batch_tune)stage2_log::print(stage2_log::phases,
        "ntt_batch_policy: mode=tune max_log2=19 samples=%llu hash=%llu fallback=%s\n",
        (unsigned long long)samples.size(),(unsigned long long)hash,samples.empty()?"no_matching_short_ntt":"legacy_when_unmeasured");
}
void portable_prediction_fields(et::Fields &sample,const pt::WidthEstimate &prediction,const char *basis) {
    sample["predicted_seconds"]=pt::numeric(prediction.seconds);
    sample["measured_model_ratio"]=pt::numeric(et::real(sample,"mean_seconds")/prediction.seconds);
    sample["prediction_model"]=pt::quoted(prediction.model);sample["prediction_basis"]=pt::quoted(basis);
}
void portable_publish(const fs::path &path,pt::Profile &profile) {
    if(!path.parent_path().empty())fs::create_directories(path.parent_path());
    const fs::path partial(path.string()+".partial."+std::to_string(GetCurrentProcessId()));
    std::ofstream out(partial,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot write portable tune summary");
    out<<profile.serialize();out.flush();if(!out)throw std::runtime_error("portable tune write failed");
    out.close();if(!out)throw std::runtime_error("portable tune close failed");
    pt::Profile::load(partial);
    if(!MoveFileExW(partial.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish portable tune summary; partial retained");
}
Handle portable_lock(const fs::path &path) {
    if(!path.parent_path().empty())fs::create_directories(path.parent_path());
    Handle guard;const fs::path lock(path.string()+".tune.lock");
    guard.value=CreateFileW(lock.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(guard.value==INVALID_HANDLE_VALUE)throw std::runtime_error("another process owns the tune update lock");
    return guard;
}
bool portable_query(ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots) {
    // Pure packing geometry has no device/free-memory dependencies.
    struct Result {bool ok;ecm_stage2::Word n=0,slots=0;};
    static std::map<std::pair<ecm_stage2::Word,int>,Result> cache;
    const auto key=std::make_pair(m,bits);auto at=cache.find(key);
    if(at==cache.end()) {Result r;r.ok=ecm_cuda_stage2_shape_query(m,bits,&r.n,&r.slots);at=cache.emplace(key,r).first;}
    if(n)*n=at->second.n;if(slots)*slots=at->second.slots;return at->second.ok;
}
pt::Shape portable_shape(const Options &o,const Record &r,uint64_t b2,uint64_t d,unsigned carrier) {
    Big n,plus;mpz_set_str(n.z,r.n.c_str(),16);mpz_add_ui(plus.z,n.z,1);
    pt::Shape s;s.target_bits=unsigned(mpz_sizeinbase(n.z,2));s.arithmetic_bits=carrier?carrier:s.target_bits;
    s.carrier=carrier;s.b1=r.b1;s.b2=b2;s.d=d;s.mersenne=carrier || mpz_popcount(plus.z)==1;
    const auto packing=et::ntt_packing_policy(tune_environment());
    s.batch_bytes=et::uint(packing,"batch_bytes");s.buffers=unsigned(et::uint(packing,"buffers"));
    s.physical=et::uint(packing,"physical_chunks")!=0;s.chunk_max=et::uint(packing,"chunk_max");
    s.short_batch_tune=o.short_batch_tune;
    if(const char *hash=std::getenv("NTT_S4_SHORT_BATCH_HASH"))s.short_batch_hash=ecm_stage2::cost::integer(hash);
    const auto p=ecm_stage2::cost::phi(d)/2;
    const uint64_t w=(s.arithmetic_bits+63)/64;
    s.fold=ecm_stage2::owner_bytes(p,w,3)<=o.owner_mb*1048576;
    const auto env=tune_environment();const auto front=et::environment_uint(env,"scaled_frontier_max_mb",o.owner_mb);
    s.frontier=s.fold && ecm_stage2::owner_bytes(p,w,3)+24*p<=std::min(o.owner_mb,front)*1048576;
    const auto leaf_mb=et::environment_uint(env,"device_gleaf_max_mb",512);
    const auto leaf_bytes=leaf_mb<=~uint64_t(0)/1048576?leaf_mb*1048576:0;
    s.chain_min=ecm_stage2::giant_chain_minimum(w,et::environment_uint(env,"giant_chain_min",32768),leaf_bytes);
    s.force_ladder=et::environment_uint(env,"giant_ladder",0)!=0;
    const auto kb=et::environment_uint(env,"giant_point_budget_kb",262144);
    const auto k=kb*1024/(w*16);
    const bool floor=et::environment_uint(env,"giant_chunk_floor",0)!=0;
    s.giant_chunk=std::max(uint64_t(1),k/p+uint64_t(!floor && k%p))*p;return s;
}
bool portable_admit(const Options &o,const Record &r,pt::Shape &s,std::string &plan) {
    const int code=ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,s.b2,s.d,o.device,
        [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&plan,s.carrier);
    if(code==3)return false;
    if(code)
        throw std::runtime_error("portable candidate planning failed");
    const auto memory=curve_memory_record(plan);
    if(plan_scalar(memory,"valid")!="true" || plan_scalar(memory,"finished")!="true" ||
       plan_scalar(memory,"initial_free_snapshot_fits")!="true")return false;
    if(memory.find("\"fold_enabled\":")!=memory.npos) {
        s.fold=plan_scalar(memory,"fold_enabled")=="true";s.frontier=plan_scalar(memory,"frontier_enabled")=="true";
    } else {s.fold=s.frontier=true;}
    const auto giant=giant_memory_record(plan);if(plan_scalar(giant,"valid")!="true")return false;
    s.giant_chunk=u64(plan_scalar(giant,"chunk_points"),"giant chunk");
    s.chain_min=u64(plan_scalar(giant,"chain_min"),"giant chain minimum");s.force_ladder=plan_scalar(giant,"force_ladder")=="true";
    return true;
}
uint64_t portable_legacy_d(const Options &o,const Record &r,uint64_t b2,unsigned carrier,bool allow_no_plan);
RuntimeTuneObservation prepare_runtime_tune(const Options &o,const Record &r) {
    RuntimeTuneObservation result;
    const auto profile=portable_read(o.tune_profile);result.condition=portable_condition(o);
    pt::Estimator engine(profile,result.condition,pt::Ignore::parse(o.tune_ignore),portable_query);
    pt::WidthPredictor predictor(engine);
    // Selection normally fixes D. A protected legacy fallback may leave it 0.
    const auto d=o.d?o.d:portable_legacy_d(o,r,o.b2,o.carrier_exponent,false);
    auto shape=portable_shape(o,r,o.b2,d,o.carrier_exponent);pt::Work work;
    if(!engine.cached_work(shape,work) || work.p<512 || !work.exact_topology)return result;
    std::string plan;if(!portable_admit(o,r,shape,plan) || engine.runtime_statistics(shape,true).count>=o.runtime_samples)return result;
    pt::WidthEstimate prediction;if(!predictor.predict(shape,prediction))return result;
    result.shape=shape;
    auto &s=result.sample;
    s["target_bits"]=std::to_string(shape.target_bits);s["arithmetic_bits"]=std::to_string(shape.arithmetic_bits);
    s["carrier_exponent"]=std::to_string(shape.carrier);s["modulus_kind"]=pt::quoted(shape.mersenne?"mersenne":"generic");
    s["b1"]=std::to_string(r.b1);s["b2"]=std::to_string(o.b2);s["source"]="\"measured\"";
    s["benchmark_kind"]="\"runtime_save\"";s["repeats"]="1";s["warmup_runs"]="0";s["stddev_seconds"]="0";
    s["condition"]="0";s["execution_path"]=pt::quoted(shape.fold?(shape.frontier?"resident":"host_frontier"):"host_fold_frontier");
    s["giant_chunk_points"]=std::to_string(shape.giant_chunk);s["giant_chain_min"]=std::to_string(shape.chain_min);
    s["giant_force_ladder"]=shape.force_ladder?"1":"0";
    s["predicted_seconds"]=pt::numeric(prediction.seconds);s["prediction_model"]=pt::quoted(prediction.model);
    s["prediction_basis"]="\"before_measurement\"";
    s["program_version"]=pt::quoted(ecm_program::version);
    s["runtime_sample_id"]=pt::quoted(o.receipt.empty()?stage2_queue::token():o.receipt);
    s["runtime_probe"]=o.runtime_probe?"1":"0";
    return result;
}
void complete_runtime_tune(RuntimeTuneObservation &observation,const EcmStage2Metrics &m) {
    const auto &s=observation.shape;
    if(!m.clean || m.bad || !m.checked || !m.selftest_cases || !m.phases.complete ||
       !(m.total_seconds>0) || m.d!=s.d || m.p!=ecm_stage2::cost::phi(s.d)/2 ||
       m.giant_points!=s.b2/s.d+2 || m.fold_resident!=s.fold || m.frontier_resident!=s.frontier)
        throw std::runtime_error("curve is incomplete, unchecked, or changed its planned execution path");
    auto metrics=et::fields(tune_metrics_json(m));metrics.erase("total_seconds");
    for(const auto &field:metrics)observation.sample[field.first]=field.second;
    auto &sample=observation.sample;
    sample["mean_seconds"]=pt::numeric(m.total_seconds);sample["min_seconds"]=sample["max_seconds"]=sample["mean_seconds"];
    sample["runtime_completed_ns"]=std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    pt::stamp_provenance(sample,et::uint(sample,"runtime_completed_ns"));
    const auto predicted=et::real(sample,"predicted_seconds");
    sample["measured_model_ratio"]=pt::numeric(m.total_seconds/predicted);
    sample["validation_count"]="1";sample["validation_max_relative_error"]=pt::numeric(std::abs(predicted-m.total_seconds)/m.total_seconds);
    pt::validate(sample);
}
std::string portable_fields_json(const et::Fields &fields) {
    std::string out="{";for(const auto &f:fields){if(out.size()>1)out+=',';out+=json_string(f.first)+':'+f.second;}return out+'}';
}
std::string runtime_tune_json(const RuntimeTuneObservation &observation) {
    if(observation.sample.empty())return {};
    return ",\"tune_observation\":{\"condition\":"+portable_fields_json(observation.condition)+
        ",\"sample\":"+portable_fields_json(observation.sample)+"}";
}
void publish_runtime_tune(const Options &o,const RuntimeTuneObservation &observation) {
    // Reload under the shared writer lock; never overwrite concurrent tune work.
    auto guard=portable_lock(o.tune_profile);auto profile=portable_read(o.tune_profile);
    auto sample=observation.sample;sample["condition"]=std::to_string(profile.condition(observation.condition));
    if(!profile.update(std::move(sample),o.runtime_samples))return;
    portable_publish(o.tune_profile,profile);
    stage2_log::print(stage2_log::phases,"stage2_tune_runtime: arithmetic_bits=%u kind=%s D=%llu B2=%llu ratio=%.3f profile=%s\n",
        observation.shape.arithmetic_bits,observation.shape.mersenne?"mersenne":"generic",
        (unsigned long long)observation.shape.d,(unsigned long long)observation.shape.b2,
        et::real(observation.sample,"measured_model_ratio"),o.tune_profile.c_str());
}
std::vector<unsigned> portable_carriers(const Options &o,const Record &r) {
    if(o.has_carrier)return {o.carrier_exponent};
    std::vector<unsigned> out{0};
    if(o.auto_carrier && o.inferred_carrier) {
        ecm_stage2::ModulusContext m;std::string error;
        if(m.configure(r.n.c_str(),16,o.inferred_carrier,error)) {if(m.lifted())out.push_back(o.inferred_carrier);}
        else throw std::runtime_error("validated worktodo carrier failed divisibility: "+error);
    }
    return out;
}
uint64_t portable_legacy_d(const Options &o,const Record &r,uint64_t b2,unsigned carrier,bool allow_no_plan=false) {
    std::string plan;
    const int code=ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,b2,0,o.device,
        [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&plan,carrier);
    if(code==3 && allow_no_plan)return 0;
    if(code)
        throw std::runtime_error("default D planning failed");
    return u64(plan_scalar(plan,"D"),"default D");
}
uint64_t portable_default_d(const Options &o,const Record &r,uint64_t b2,unsigned carrier,bool allow_no_plan=false) {
    const auto legacy=portable_legacy_d(o,r,b2,carrier,allow_no_plan);if(!legacy)return 0;
    auto baseline=portable_shape(o,r,b2,legacy,carrier);pt::Work before;
    auto ntt=[](uint64_t n,uint64_t batch){return double(n)*batch*std::log2(double(n));};
    if(!baseline.fold || !baseline.frontier || !pt::work(baseline,portable_query,ntt,before) ||
        before.largest_ntt!=(1ull<<27) || (!baseline.mersenne && before.points/before.p<16))return legacy;
    // At the portable cost boundary, compare established larger D inside the
    // same NTT plateau. Lower lengths retain the established default model.
    // These normalized weights describe the observed production split, not
    // absolute seconds or a claim that all devices have identical rates.
    auto ratio=[](double a,double b){return b>0?a/b:1.;};
    double best=.95;uint64_t winner=legacy;
    for(const auto &entry:ecm_stage2::d_catalog::entries())if(entry.default_plateau && entry.d>legacy) {
        const auto d=entry.d;
        auto shape=portable_shape(o,r,b2,d,carrier);pt::Work after;
        if(!shape.fold || !shape.frontier || !pt::work(shape,portable_query,ntt,after) ||
            !after.exact_topology || after.largest_ntt!=before.largest_ntt)continue;
        auto poly=[&](unsigned phase){return .7*ratio(after.ntt[phase],before.ntt[phase])+
            .3*ratio(after.reduce[phase],before.reduce[phase]);};
        const double score=.58*poly(ecm_stage2::RequestGtrees)+.22*poly(ecm_stage2::RequestFold)+
            .13*ratio(after.giant,before.giant)+.07*std::max(poly(ecm_stage2::RequestFtree),ratio(after.baby,before.baby));
        if(score>=best)continue;
        std::string plan;if(!portable_admit(o,r,shape,plan) || !shape.fold || !shape.frontier)continue;
        best=score;winner=d;
    }
    return winner;
}
std::string portable_runtime_search(const Options &o,const Record &r,const pt::Shape &reference,
        pt::Estimator &engine,pt::WidthPredictor &predictor,std::vector<pt::RuntimeCandidate> &candidates) {
    if(!o.tune_runtime || (o.has_d && o.d))return {};
    std::vector<pt::DSearchNode> existing;bool partial=false;
    for(const auto &c:candidates)if(c.shape.carrier==reference.carrier) {
        existing.push_back({c.shape.d,ecm_stage2::d_catalog::degree(c.shape.d),{},1,
            c.statistics.seconds,c.statistics.stddev,c.statistics.count});
        partial|=c.shape.d!=reference.d && c.statistics.count && c.statistics.count<2;
    }
    const auto initial_budget=pt::runtime_search_budget(existing,reference.d,o.runtime_samples);
    if((pt::DSearch(existing,reference.d).explored()>=initial_budget.candidates && !partial) ||
       !initial_budget.remaining || (o.runtime_remaining && o.runtime_remaining<4))return {};
    std::vector<pt::DSearchNode> nodes;std::vector<pt::Shape> shapes;std::vector<pt::WidthEstimate> estimates;
    std::set<uint64_t> ds;for(const auto &entry:ecm_stage2::d_catalog::entries())ds.insert(entry.d);
    for(const auto &c:candidates)if(c.shape.carrier==reference.carrier)ds.insert(c.shape.d);
    for(auto d:ds) {
        auto s=portable_shape(o,r,o.b2,d,reference.carrier);pt::Work w;pt::WidthEstimate e;
        if(!engine.cached_work(s,w) || w.p<512 || !w.exact_topology || !predictor.predict(s,e))continue;
        // Reuse the admitted shape of existing candidates; predicted chunk and
        // residency are not a substitute for their exact runtime identity.
        for(const auto &c:candidates)if(c.shape.d==d && c.shape.carrier==reference.carrier){s=c.shape;break;}
        if(!engine.cached_work(s,w) || !predictor.predict(s,e))continue;
        const auto stats=engine.runtime_statistics(s);
        nodes.push_back({d,w.p,pt::interval(s,w).key,e.rank,stats.seconds,stats.stddev,stats.count});
        shapes.push_back(s);estimates.push_back(e);
    }
    auto budget=pt::runtime_search_budget(nodes,reference.d,o.runtime_samples);
    const char *reason="admission_attempt_limit";uint64_t proposed=0;double scale=0;
    for(unsigned attempt=0;attempt<16;++attempt) {
        budget=pt::runtime_search_budget(nodes,reference.d,o.runtime_samples);
        const pt::DSearch search(nodes,reference.d,2,budget.candidates,pt::RuntimeSearchBudget::regions);
        const auto step=search.next();reason=step.reason;scale=step.scale;if(!step.available)break;
        const auto i=step.index;auto s=shapes[i];std::string plan;
        if(estimates[i].seconds>1.5*budget.reference) {
            nodes[i].rejected=true;reason="model_uncompetitive";continue;
        }
        if(!budget.permits(estimates[i].seconds*(1+estimates[i].relative_error)+2*estimates[i].stddev,
                           nodes[i].samples,o.runtime_remaining)) {
            nodes[i].rejected=true;reason="exploration_budget";continue;
        }
        if(!portable_admit(o,r,s,plan)){nodes[i].rejected=true;reason="memory_admission";continue;}
        pt::Work w;pt::WidthEstimate e;
        if(!engine.cached_work(s,w) || !w.exact_topology || !predictor.predict(s,e)) {
            nodes[i].rejected=true;reason="admitted_model_unavailable";continue;
        }
        const auto stats=engine.runtime_statistics(s);
        // A physical route may differ from the cheap ranking shape. Reconcile
        // its existing observations before exposing another trial.
        if(stats.count>=2) {
            nodes[i]={s.d,w.p,pt::interval(s,w).key,e.rank,stats.seconds,stats.stddev,stats.count};
            bool present=false;for(auto &c:candidates)if(c.shape.d==s.d && c.shape.carrier==s.carrier) {
                c.shape=s;c.statistics=stats;present=true;break;
            }
            if(!present)candidates.push_back({s,stats,2,false});continue;
        }
        if(!budget.permits(e.seconds*(1+e.relative_error)+2*e.stddev,stats.count,o.runtime_remaining)) {
            nodes[i].rejected=true;reason="admitted_exploration_budget";continue;
        }
        bool present=false;for(auto &c:candidates)if(c.shape.d==s.d && c.shape.carrier==s.carrier) {
            c.shape=s;c.statistics=stats;c.probe_allowed=true;present=true;break;
        }
        if(!present)candidates.push_back({s,stats,3,true});proposed=s.d;break;
    }
    std::ostringstream out;out<<std::setprecision(17)<<"{\"carrier_exponent\":"<<reference.carrier
        <<",\"candidate_limit\":"<<budget.candidates<<",\"explored\":"<<pt::DSearch(nodes,reference.d).explored()
        <<",\"extra_seconds\":"<<budget.spent<<",\"remaining_extra_seconds\":"<<budget.remaining
        <<",\"proposed_D\":"<<proposed<<",\"reason\":"<<json_string(reason)<<",\"scale\":"<<scale<<'}';
    return out.str();
}
std::string portable_runtime_select(Options &o,const Record &r,pt::Estimator &engine,pt::WidthPredictor &predictor) {
    if(o.relocation || o.tune_child || o.tune_profile.empty())return {};

    const auto carriers=portable_carriers(o,r);
    std::vector<pt::RuntimeCandidate> candidates;
    std::vector<pt::Shape> references;std::vector<std::string> searches;bool references_ready=true;
    uint64_t ordinary_default=0;
    for(auto carrier:carriers) {
        const auto center=o.has_d && o.d?o.d:portable_default_d(o,r,o.b2,carrier,true);if(!center)continue;
        if(!carrier || !ordinary_default)ordinary_default=center;
        const auto original=portable_shape(o,r,o.b2,center,carrier);
        std::map<uint64_t,unsigned> ds{{center,carrier?0u:1u}};
        const et::Fields *historical=nullptr;
        for(const auto &sample:engine.profile.samples)if(pt::measured(sample) &&
            pt::matches(engine.profile.conditions.at(et::uint(sample,"condition")),engine.condition,engine.ignore)) {
            const auto shape=engine.shape(sample);
            if(shape.target_bits!=original.target_bits || shape.arithmetic_bits!=original.arithmetic_bits ||
               shape.carrier!=carrier || shape.b2!=o.b2 || (o.has_d && o.d && shape.d!=o.d))continue;
            if(!historical || et::real(sample,"mean_seconds")<et::real(*historical,"mean_seconds"))historical=&sample;
            if(pt::runtime_observation(sample)) {
                if(!ds.count(shape.d))ds[shape.d]=2;
            }
        }
        if(historical && !ds.count(et::uint(*historical,"d")))ds[et::uint(*historical,"d")]=2;
        for(const auto &entry:ds) {
            auto shape=portable_shape(o,r,o.b2,entry.first,carrier);pt::Work w;std::string plan;
            if(!engine.cached_work(shape,w) || w.p<512 || !w.exact_topology || !portable_admit(o,r,shape,plan))continue;
            const auto stats=engine.runtime_statistics(shape);
            candidates.push_back({shape,stats,entry.second,entry.first==center || (o.has_d && o.d)});
            if(entry.first==center) {references.push_back(shape);references_ready&=stats.count>=2;}
        }
    }
    if(references_ready)for(const auto &reference:references) {
        const auto search=portable_runtime_search(o,r,reference,engine,predictor,candidates);
        if(!search.empty())searches.push_back(search);
    }
    const auto choice=pt::runtime_choice(candidates,o.tune_runtime);if(!choice.available)return {};
    const auto &chosen=candidates[choice.index];const auto &shape=chosen.shape;
    o.d=shape.d;o.carrier_exponent=shape.carrier;o.runtime_probe=choice.probe;
    const char *reason=!choice.probe?"runtime_measured_hysteresis":chosen.priority==0?
        "runtime_probe_carrier_first":chosen.priority==1?"runtime_probe_reference":"runtime_probe_candidate";
    std::ostringstream out;out<<std::setprecision(17)<<"{\"type\":\"tune_selection\",\"selected\":true,\"policy\":\"model\""
        <<",\"reason\":"<<json_string(reason)<<",\"default_D\":"<<ordinary_default
        <<",\"B1\":"<<r.b1<<",\"B2\":"<<o.b2<<",\"D\":"<<shape.d
        <<",\"P\":"<<ecm_stage2::d_catalog::degree(shape.d)<<",\"target_bits\":"<<shape.target_bits
        <<",\"arithmetic_bits\":"<<shape.arithmetic_bits<<",\"carrier_exponent\":"<<shape.carrier
        <<",\"source\":"<<json_string(choice.probe?"runtime_trial":"measured")
        <<",\"model\":\"runtime_shape_mean_v2\",\"runtime_probe\":"<<(choice.probe?"true":"false")
        <<",\"runtime_samples\":"<<chosen.statistics.count<<",\"runtime_sample_limit\":"<<o.runtime_samples;
    if(chosen.statistics.count)out<<",\"mean_seconds\":"<<chosen.statistics.seconds<<",\"stddev_seconds\":"<<chosen.statistics.stddev;
    out<<",\"runtime_candidates\":[";
    for(size_t i=0;i<candidates.size();++i) {
        if(i)out<<',';const auto &c=candidates[i];
        out<<"{\"D\":"<<c.shape.d<<",\"carrier_exponent\":"<<c.shape.carrier
            <<",\"priority\":"<<c.priority<<",\"samples\":"<<c.statistics.count;
        if(c.statistics.count)out<<",\"mean_seconds\":"<<c.statistics.seconds<<",\"stddev_seconds\":"<<c.statistics.stddev;
        out<<'}';
    }
    out<<"],\"runtime_search\":[";for(size_t i=0;i<searches.size();++i){if(i)out<<',';out<<searches[i];}out<<']';
    return out.str()+'}';
}
std::string portable_width_select(Options &o,const Record &r) {
    const auto profile=portable_read(o.tune_profile);const auto condition=portable_condition(o);
    pt::Estimator engine(profile,condition,pt::Ignore::parse(o.tune_ignore),portable_query);
    pt::WidthPredictor predictor(engine);
    const auto runtime=portable_runtime_select(o,r,engine,predictor);if(!runtime.empty())return runtime;
    const auto carriers=portable_carriers(o,r);const unsigned initial=carriers.front();
    const auto default_d=o.has_d && o.d?o.d:portable_default_d(o,r,o.b2,initial);
    auto baseline=portable_shape(o,r,o.b2,default_d,initial);std::string baseline_plan;
    pt::WidthEstimate reference;
    if(!portable_admit(o,r,baseline,baseline_plan) || !predictor.predict(baseline,reference))
        return "{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"default_prediction_or_memory_unavailable\",\"default_D\":"+std::to_string(default_d)+'}';
    struct Candidate {pt::Shape shape;pt::WidthEstimate estimate,family;uint64_t center;};
    std::vector<Candidate> candidates;std::string alternatives;
    auto describe=[&](const pt::Shape &shape,const pt::WidthEstimate &e) {
        std::ostringstream out;out<<std::setprecision(17)<<"{\"target_bits\":"<<shape.target_bits
            <<",\"arithmetic_bits\":"<<shape.arithmetic_bits<<",\"carrier_exponent\":"<<shape.carrier
            <<",\"D\":"<<shape.d<<",\"P\":"<<ecm_stage2::d_catalog::degree(shape.d)
            <<",\"estimated_seconds\":"<<e.seconds<<",\"upper_seconds\":"<<e.rank
            <<",\"lower_seconds\":"<<pt::WidthPredictor::lower(e)<<",\"model\":"<<json_string(e.model)
            <<",\"source\":"<<json_string(e.exact?"measured":e.anchors?(e.extrapolated?"extrapolated":"interpolated"):"builtin_model")
            <<",\"arithmetic_low\":"<<e.arithmetic_low<<",\"arithmetic_high\":"<<e.arithmetic_high
            <<",\"anchors\":"<<e.anchors<<",\"region_covered\":"<<(e.region_covered?"true":"false")
            <<",\"relative_error_allowance\":"<<e.relative_error<<",\"uncertainty_source\":"<<json_string(e.uncertainty)<<'}';
        return out.str();
    };
    for(auto carrier:carriers) {
        const auto center=o.has_d && o.d?o.d:portable_default_d(o,r,o.b2,carrier,true);if(!center)continue;
        auto shape=portable_shape(o,r,o.b2,center,carrier);pt::WidthEstimate estimate;
        std::string center_plan;
        if(!portable_admit(o,r,shape,center_plan) || !predictor.predict(shape,estimate))continue;
        const auto family_reference=estimate;
        if(!alternatives.empty())alternatives+=',';alternatives+=describe(shape,estimate);
        std::vector<uint64_t> ds{center};
        if(!(o.has_d && o.d))for(const auto &entry:ecm_stage2::d_catalog::entries())ds.push_back(entry.d);
        std::sort(ds.begin(),ds.end());ds.erase(std::unique(ds.begin(),ds.end()),ds.end());
        for(auto d:ds) {
            shape=portable_shape(o,r,o.b2,d,carrier);
            if(!predictor.predict(shape,estimate))continue;
            const bool paired=estimate.exact && reference.exact && pt::LocalSearch::improves(
                {estimate.seconds,estimate.stddev},{reference.seconds,reference.stddev});
            const bool family_pair=estimate.exact && family_reference.exact && pt::LocalSearch::improves(
                {estimate.seconds,estimate.stddev},{family_reference.seconds,family_reference.stddev});
            if(d!=center && !paired && !family_pair && !pt::WidthPredictor::improves(estimate,family_reference))continue;
            if(paired || pt::WidthPredictor::improves(estimate,reference))candidates.push_back({shape,estimate,family_reference,center});
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate &a,const Candidate &b){return a.estimate.rank<b.estimate.rank;});
    auto chosen=baseline;auto prediction=reference;uint64_t rejected=0;
    for(auto &candidate:candidates) {
        std::string plan;auto shape=candidate.shape;
        if(!portable_admit(o,r,shape,plan) || !predictor.predict(shape,candidate.estimate)){++rejected;continue;}
        const auto &estimate=candidate.estimate;
        const bool paired=estimate.exact && reference.exact && pt::LocalSearch::improves(
            {estimate.seconds,estimate.stddev},{reference.seconds,reference.stddev});
        const bool family_pair=estimate.exact && candidate.family.exact && pt::LocalSearch::improves(
            {estimate.seconds,estimate.stddev},{candidate.family.seconds,candidate.family.stddev});
        if(shape.d!=candidate.center && !paired && !family_pair && !pt::WidthPredictor::improves(estimate,candidate.family)){++rejected;continue;}
        if(!paired && !pt::WidthPredictor::improves(estimate,reference)){++rejected;continue;}
        chosen=shape;prediction=estimate;break;
    }
    const bool selected=chosen.d!=baseline.d || chosen.carrier!=baseline.carrier;
    o.d=chosen.d;o.carrier_exponent=chosen.carrier;
    auto details=describe(chosen,prediction);details.erase(details.begin());
    return "{\"type\":\"tune_selection\",\"selected\":"+std::string(selected?"true":"false")+
        ",\"reason\":"+json_string(selected?"arithmetic_width_cost_win":"default_protected_by_prediction_uncertainty")+
        ",\"policy\":\"model\",\"default_D\":"+std::to_string(default_d)+
        ",\"B1\":"+std::to_string(r.b1)+",\"B2\":"+std::to_string(o.b2)+
        ",\"memory_rejected\":"+std::to_string(rejected)+",\"alternatives\":["+alternatives+"],"+details;
}
std::string portable_select(Options &o,const Record &r,bool automatic) {
    if(!automatic && o.prediction_policy=="model")return portable_width_select(o,r);
    if(o.tune_profile.empty() && o.prediction_policy!="model")return {};
    if(!fs::is_regular_file(o.tune_profile) && o.prediction_policy!="model") {
        if(automatic)throw std::runtime_error("Auto B2 needs a tune summary; enable short calibration or supply a profile");
        return "{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"missing_profile\"}";
    }
    const auto profile=portable_read(o.tune_profile);const auto condition=portable_condition(o);
    pt::Estimator estimator(profile,condition,pt::Ignore::parse(o.tune_ignore),portable_query);
    pt::WidthPredictor width_predictor(estimator);
    double t1=o.stage1_seconds;std::string t1_source="provided_per_curve";
    Big n;mpz_set_str(n.z,r.n.c_str(),16);const unsigned bits=unsigned(mpz_sizeinbase(n.z,2));
    if(automatic && !(t1>0) && !o.stage1_csv.empty()) {
        const auto prediction=ecm_stage2::stage1_cost::Table::load(o.stage1_csv).predict(bits,double(r.b1),double(o.stage1_mhz),o.stage1_exponent);
        t1=prediction.seconds;t1_source="csv_"+prediction.source+(prediction.crosses_tpi?"_cross_tpi":"");
    }
    if(automatic && !(t1>0) && !o.stage1_profile.empty()) {
        const auto old=et::Stage1Profile::load(o.stage1_profile);
        auto ign=pt::Ignore::parse(o.tune_ignore);ign.backend=ign.environment=true;
        if(!pt::matches(old.device,condition,ign))throw std::runtime_error("Stage1 cost conditions rejected by tune-ignore policy");
        Big plus;mpz_add_ui(plus.z,n.z,1);const auto kind=json_string(mpz_popcount(plus.z)==1?"mersenne":"generic");
        const et::Fields *anchor=nullptr;double distance=std::numeric_limits<double>::infinity();
        const auto requested=ecm_stage2::stage1_cost::tier(bits);
        for(const auto &s:old.samples)if(et::uint(s,"batch")==o.stage1_batch && et::required(s,"modulus_kind")==kind &&
            et::required(s,"exponent")==json_string(o.stage1_exponent)) {
            const auto container=ecm_stage2::stage1_cost::tier(unsigned(et::uint(s,"target_bits")));
            const double d=std::abs(std::log2(double(requested.bits)/container.bits))+(container.tpi!=requested.tpi?2:0);
            if(d<distance){distance=d;anchor=&s;}
        }
        if(!anchor)throw std::runtime_error("no Stage1 cost anchor for batch/type/exponent");
        const auto container=ecm_stage2::stage1_cost::tier(unsigned(et::uint(*anchor,"target_bits")));
        t1=et::real(*anchor,"mean_seconds")*double(r.b1)/et::uint(*anchor,"b1")*
            double(requested.bits)*requested.bits/(double(container.bits)*container.bits);
        t1_source="legacy_profile_scaled_estimate";
    }
    if(automatic && (!(t1>0) || !std::isfinite(t1)))throw std::runtime_error("Auto B2 needs Stage1 seconds/curve, cost CSV, or a Stage1 profile");
    const unsigned factor=o.factor_bits?unsigned(o.factor_bits):ecm_stage2::probability::recommended_factor_bits(double(r.b1),bits);
    std::vector<uint64_t> ds=pt::grid(10).ds;
    if(o.has_d && o.d)ds={o.d};
    std::vector<uint64_t> b2s;
    if(automatic) {
        const auto low=std::max(r.b1+1,o.auto_min),high=o.auto_max?o.auto_max:pt::default_b2_max;
        if(low>high || high>uint64_t(INT64_MAX)-8192)throw std::runtime_error("Auto B2 has an invalid search interval");
        for(unsigned i=0;i<=48;++i)b2s.push_back(i==48?high:std::max(low,uint64_t(std::exp(std::log(double(low))+
            (std::log(double(high))-std::log(double(low)))*i/48))));
        for(const auto &s:profile.samples){const auto b=et::uint(s,"b2");if(b>=low && b<=high)b2s.push_back(b);}
        std::sort(b2s.begin(),b2s.end());b2s.erase(std::unique(b2s.begin(),b2s.end()),b2s.end());
    } else b2s={o.b2};
    struct Candidate {pt::Shape shape;pt::Estimate estimate;double score=0,probability=0;};
    std::vector<Candidate> candidates;
    const auto carriers=portable_carriers(o,r);
    std::map<std::pair<uint64_t,unsigned>,const et::Fields*> baselines;
    std::map<std::pair<uint64_t,unsigned>,uint64_t> default_ds;
    if(automatic && !(o.has_d && o.d)) {
        for(auto b:b2s)for(auto carrier:carriers) {
            const auto d=portable_default_d(o,r,b,carrier,true);default_ds[{b,carrier}]=d;if(d)ds.push_back(d);
        }
        std::sort(ds.begin(),ds.end());ds.erase(std::unique(ds.begin(),ds.end()),ds.end());
    }
    if(!automatic && !(o.has_d && o.d))for(auto b:b2s)for(auto carrier:carriers) {
        const auto d=portable_default_d(o,r,b,carrier);default_ds[{b,carrier}]=d;
        auto shape=portable_shape(o,r,b,d,carrier);std::string plan;
        baselines[{b,carrier}]=portable_admit(o,r,shape,plan)?estimator.observed(shape):nullptr;
    }
    auto protected_default=[&](const char *reason) {
        std::string out="{\"type\":\"tune_selection\",\"selected\":false,\"reason\":"+json_string(reason);
        if(!automatic && !(o.has_d && o.d))out+=",\"default_D\":"+std::to_string(default_ds.at({o.b2,carriers.front()}));
        return out+'}';
    };
    if(!automatic && !(o.has_d && o.d)) {
        bool covered=false;for(auto b:b2s)covered|=baselines.at({b,carriers.front()})!=nullptr;
        if(!covered)return protected_default("default_planner_protected_missing_paired_baseline");
    }
    std::map<std::pair<uint64_t,unsigned>,std::set<uint64_t>> paired_ds;
    if(!automatic && !(o.has_d && o.d))for(auto b:b2s)for(auto carrier:carriers) {
        const auto base=baselines.at({b,carriers.front()});if(!base)continue;
        for(const auto &s:profile.samples)if(pt::measured(s) && et::uint(s,"target_bits")==bits &&
            et::uint(s,"arithmetic_bits")==unsigned(carrier?carrier:bits) &&
            et::uint(s,"carrier_exponent")==carrier && et::uint(s,"b2")==b &&
            ecm_stage2::d_catalog::contains(et::uint(s,"d")) &&
            pt::LocalSearch::improves({et::real(s,"mean_seconds"),et::real(s,"stddev_seconds")},
                {et::real(*base,"mean_seconds"),et::real(*base,"stddev_seconds")}))
            paired_ds[{b,carrier}].insert(et::uint(s,"d"));
    }
    for(auto b:b2s)for(auto d:ds)for(auto carrier:carriers) {
        if(automatic && !(o.has_d && o.d) && !default_ds.at({b,carrier}))continue;
        const bool paired=!automatic && !(o.has_d && o.d);
        if(paired && !paired_ds[{b,carrier}].count(d))continue;
        auto shape=portable_shape(o,r,b,d,carrier);pt::Estimate e;
        if(paired) {
            // Only paired measurements can replace the default planner. A
            // sparse extrapolation is useful for scheduling tune, not a win.
            std::string trial_plan;if(!portable_admit(o,r,shape,trial_plan))continue;
            const auto base=baselines.at({b,carriers.front()}),trial=estimator.observed(shape);
            if(!base || !trial || !pt::LocalSearch::improves(
                {et::real(*trial,"mean_seconds"),et::real(*trial,"stddev_seconds")},
                {et::real(*base,"mean_seconds"),et::real(*base,"stddev_seconds")}))continue;
            e.seconds=et::real(*trial,"mean_seconds");e.stddev=et::real(*trial,"stddev_seconds");
            e.rank=e.seconds+2*e.stddev;e.exact=true;e.model="paired_measured_d_v1";
            e.anchor=trial;e.anchors=1;
            e.validated=pt::optional_uint(*trial,"validation_count")>0 &&
                pt::matches(profile.conditions.at(et::uint(*trial,"condition")),condition,pt::Ignore{});
            e.relative_error=e.validated?pt::optional_real(*trial,"validation_max_relative_error"):.25;
        } else {
            pt::WidthEstimate reusable;
            if(o.prediction_policy=="model") {if(!width_predictor.predict(shape,reusable))continue;e=reusable;}
            else if(!estimator.predict(shape,e))continue;
            if(automatic && !(o.has_d && o.d) && d!=default_ds.at({b,carrier}) &&
                !e.exact && (o.prediction_policy=="model"?!reusable.region_covered:std::string(e.model)!="portable_nonnegative_v1"))continue;
        }
        const double probability=automatic?ecm_stage2::probability::success(double(r.b1),double(b),factor):0;
        const double score=automatic?probability/(t1+o.ratio_adjust*e.rank):-e.rank;
        candidates.push_back({shape,e,score,probability});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate &a,const Candidate &b){return a.score>b.score;});
    size_t rejected=0;
    for(auto &candidate:candidates) {
        auto shape=candidate.shape;std::string plan;if(!portable_admit(o,r,shape,plan)){++rejected;continue;}
        pt::Estimate e=candidate.estimate;
        if(automatic || (o.has_d && o.d)) {
            pt::WidthEstimate reusable;
            if(o.prediction_policy=="model") {if(!width_predictor.predict(shape,reusable)){++rejected;continue;}e=reusable;}
            else if(!estimator.predict(shape,e)){++rejected;continue;}
        }
        // Do not publish a winner whose route changed after ranking. Re-rank
        // changed paths on the next pass rather than transferring resident data.
        if(shape.fold!=candidate.shape.fold || shape.frontier!=candidate.shape.frontier){++rejected;continue;}
        o.b2=shape.b2;o.d=shape.d;o.carrier_exponent=shape.carrier;
        std::ostringstream out;out<<std::setprecision(17)<<"{\"type\":"<<json_string(automatic?"auto_b2":"tune_selection")
            <<",\"selected\":true,\"model\":"<<json_string(e.model)<<",\"B1\":"<<r.b1<<",\"B2\":"<<shape.b2
            <<",\"D\":"<<shape.d<<",\"P\":"<<ecm_stage2::cost::phi(shape.d)/2<<",\"carrier_exponent\":"<<shape.carrier
            <<",\"estimated_seconds\":"<<e.seconds<<",\"rank_seconds\":"<<e.rank<<",\"source\":"<<json_string(e.exact?"measured":"estimated")
            <<",\"independently_validated\":"<<(e.validated?"true":"false")<<",\"relative_error_allowance\":"<<e.relative_error
            <<",\"fold_resident\":"<<(shape.fold?1:0)<<",\"frontier_resident\":"<<(shape.frontier?1:0)
            <<",\"memory_rejected\":"<<rejected<<",\"required_free_bytes\":"<<plan_scalar(curve_memory_record(plan),"required_free_bytes");
        if(automatic)out<<",\"T1\":"<<t1<<",\"T1_source\":"<<json_string(t1_source)<<",\"target_factor_bits\":"<<factor
            <<",\"success_probability\":"<<candidate.probability<<",\"benefit_per_second\":"<<candidate.score;
        out<<'}';return out.str();
    }
    if(automatic)throw std::runtime_error("Auto B2 has no costed candidate admitted by the current memory model");
    return protected_default("default_planner_protected_no_measured_win");
}
std::string select_tuned(Options &o,const Record &r){
    auto result=portable_select(o,r,false);
    if(!(o.has_d && o.d) && (result.empty() || result.find("\"selected\":false")!=result.npos)) {
        const auto planned=result.find("\"default_D\":")==result.npos?std::string{}:plan_scalar(result,"default_D");
        o.d=planned.empty()?portable_default_d(o,r,o.b2,o.carrier_exponent):u64(planned,"default D");
        if(result.empty())result="{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"no_suitable_tune_profile\"}";
        result.pop_back();
        if(planned.empty())result+=",\"default_D\":"+std::to_string(o.d);
        result+=",\"default_model\":\"ntt_ceiling_plateau_v1\"}";
    }
    return result;
}
std::string select_auto_tuned(Options &o,const Record &r){return portable_select(o,r,true);}
std::string select_runtime_tuned(Options &o,const Record &r) {
    if(o.prediction_policy!="model")return {};
    const auto profile=portable_read(o.tune_profile);
    pt::Estimator engine(profile,portable_condition(o),pt::Ignore::parse(o.tune_ignore),portable_query);
    pt::WidthPredictor predictor(engine);return portable_runtime_select(o,r,engine,predictor);
}

struct PortableInput {fs::path save;Record record;std::string kind;};
void merge_portable_tune(const Options &o,const fs::path &destination) {
    if(o.tune_merge.size()>64 || !o.tune_save.empty() || !o.tune_worktodo.empty() || o.tune_carrier || !o.tune_ds.empty() ||
       !o.tune_b2s.empty() || !o.tune_exponents.empty())throw std::runtime_error("merge accepts at most 64 profiles and no curve grid");
    auto guard=portable_lock(destination);auto combined=portable_read(destination);
    for(const auto &path:o.tune_merge) {
        const auto source=pt::Profile::load(absolute_from(fs::current_path(),path));
        for(auto s:source.samples){s["condition"]=std::to_string(combined.condition(source.conditions.at(et::uint(s,"condition"))));combined.update(std::move(s),o.runtime_samples);}
        for(auto s:source.ntt){s["condition"]=std::to_string(combined.condition(source.conditions.at(et::uint(s,"condition"))));combined.update_ntt(std::move(s));}
    }
    for(const auto &path:o.tune_ntt_profiles) {
        const auto source=pt::Profile::load(absolute_from(fs::current_path(),path));
        for(auto s:source.ntt){s["condition"]=std::to_string(combined.condition(source.conditions.at(et::uint(s,"condition"))));combined.update_ntt(std::move(s));}
    }
    portable_publish(destination,combined);
    std::cout<<"ecm_tune_merge_complete: summaries="<<combined.samples.size()<<" ntt="<<combined.ntt.size()
        <<" profile="<<destination.string()<<std::endl;
}
PortableInput portable_input(unsigned bits,const fs::path &evidence,bool mersenne=false) {
    if(bits<5 || bits>16384)throw std::runtime_error("portable benchmark width must be 5..16384");
    Big n;mpz_set_ui(n.z,1);
    if(mersenne){
        mpz_mul_2exp(n.z,n.z,bits);mpz_sub_ui(n.z,n.z,1);
        // Arithmetic-band fixture only; never used to rewrite production factors.
        // Drop factors inherited from proper exponent divisors. For power-of-two
        // bands this removes earlier Fermat factors which would always be hits.
        Big inherited,common;
        auto strip_exponent=[&](unsigned exponent) {
            mpz_set_ui(inherited.z,1);mpz_mul_2exp(inherited.z,inherited.z,exponent);mpz_sub_ui(inherited.z,inherited.z,1);
            for(;;) {mpz_gcd(common.z,n.z,inherited.z);if(mpz_cmp_ui(common.z,1)<=0)break;mpz_divexact(n.z,n.z,common.z);}
        };
        unsigned rest=bits;
        for(unsigned prime=2;prime<=rest/prime;++prime)if(rest%prime==0) {
            while(rest%prime==0)rest/=prime;strip_exponent(bits/prime);
        }
        if(rest>1)strip_exponent(bits/rest);
        for(unsigned divisor=3;divisor<=10000;divisor+=2)
            while(mpz_divisible_ui_p(n.z,divisor))mpz_divexact_ui(n.z,n.z,divisor);
    }
    else {mpz_mul_2exp(n.z,n.z,bits-1);mpz_add_ui(n.z,n.z,bits<16?1:12345);mpz_nextprime(n.z,n.z);}
    if((!mersenne && mpz_sizeinbase(n.z,2)!=bits) || mpz_cmp_ui(n.z,3)<=0)
        throw std::runtime_error("benchmark construction has no usable modulus at requested width");
    const auto hex=number(n.z);
    std::string x;
    try {x=et::benchmark_point(hex);}
    catch(const std::runtime_error &error) {throw TaskInputError(error.what());}
    const auto save=evidence/((mersenne?"m":"p")+std::to_string(bits)+".save");
    append(save,"METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x"+hex+"; X=0x"+x+"; Z=1;");
    return {save,records(save,0,1).at(0),mersenne?"mersenne_small_factor_stripped_band":"probable_prime_gmp_nextprime"};
}
bool portable_measure(Options o,const Settings &settings,const PortableInput &input,uint64_t b2,uint64_t d,unsigned carrier,
    unsigned repeats,pt::Budget &budget,const fs::path &evidence,pt::Profile &profile,ecm_stage2::Word condition,
    bool skip_verified_factor=false,unsigned *completed_repeats=nullptr,et::Fields *completed_sample=nullptr) {
    if(completed_repeats)*completed_repeats=0;
    if(completed_sample)completed_sample->clear();
    if(budget.expired() || stop_requests)return false;
    // Full-engine calibration remains independent of a mutable NTT tune file.
    // Store that legacy policy in its actual condition, never label it as tuned.
    o.short_batch_tune=false;configure_short_ntt_batches(o);
    condition=profile.condition(portable_condition(o));
    auto shape=portable_shape(o,input.record,b2,d,carrier);std::string plan;
    if(!portable_admit(o,input.record,shape,plan))return false;
    pt::Estimator before_model(profile,profile.conditions.at(condition),pt::Ignore::parse(o.tune_ignore),portable_query);
    pt::WidthPredictor before_predictor(before_model);pt::WidthEstimate prediction;
    const bool predicted=before_predictor.predict(shape,prediction);
    const std::string stem="measure_"+std::to_string(profile.revision)+"_"+std::to_string(GetTickCount64());
    append(evidence/(stem+".plan.jsonl"),plan);
    std::vector<et::Fields> trials;
    for(unsigned repeat=o.tune_warmup?0:1;repeat<=repeats;++repeat) {
        // Optional warmup consumes the same budget; never start after expiry.
        if(budget.expired() || stop_requests) {if(trials.empty())return false;break;}
        Options worker=o;worker.tune_child=true;worker.auto_b2=false;worker.factor_only=false;worker.factorize_hits=false;
        worker.carrier_exponent=carrier;worker.has_carrier=true;worker.log_level=stage2_log::quiet;
        worker.debug_log=false;worker.debug_file.clear();worker.tune_profile.clear();
        const auto base=stem+"_"+std::to_string(repeat);
        const auto result=evidence/(base+".jsonl"),log=evidence/(base+".log");
        const int code=child_run(worker,input.save,input.record,b2,d,o.device,result,log,o.batch,settings);
        if(code==3 && skip_verified_factor) {
            std::ifstream receipt(result);std::string row,extra;
            if(!std::getline(receipt,row) || std::getline(receipt,extra))throw std::runtime_error("invalid factor calibration receipt");
            const auto f=et::fields(row);
            if(et::required(f,"status")!="\"verified_factor_not_timing_sample\"" || !et::uint(f,"clean") ||
                et::uint(f,"bad") || !et::uint(f,"selftest_cases") || !et::uint(f,"checked"))
                throw std::runtime_error("unverified factor calibration receipt");
            std::cout<<"stage2_short_calibration_skip: reason=verified_factor evidence="<<result.string()<<std::endl;
            return false;
        }
        if(code==3)
            throw std::runtime_error("ECM tune cannot time a factor-producing curve; inspect "+result.string());
        if(code)
            throw std::runtime_error("portable calibration failed arithmetic/curve checks; inspect "+log.string());
        std::ifstream in(result);std::string row,extra;if(!std::getline(in,row) || std::getline(in,extra))throw std::runtime_error("invalid calibration receipt");
        auto f=et::fields(row);if(et::uint(f,"d")!=d || et::uint(f,"giant_points")!=b2/d+2 || et::uint(f,"bad") ||
            !et::uint(f,"clean") || !et::uint(f,"selftest_cases") || !et::uint(f,"checked"))throw std::runtime_error("invalid completed calibration shape");
        if(bool(et::uint(f,"fold_resident"))!=shape.fold || bool(et::uint(f,"frontier_resident"))!=shape.frontier)
            throw std::runtime_error("calibration execution path differs from planned path; evidence retained");
        if(repeat)trials.push_back(std::move(f));
        std::cout<<"ecm_tune_progress: N="<<shape.target_bits<<" D="<<d<<" B2="<<b2<<" repeat="<<repeat<<'/'<<repeats
            <<" arithmetic_bits="<<shape.arithmetic_bits<<" kind="<<(shape.mersenne?"mersenne":"generic")
            <<" budget_left="<<budget.remaining()<<" s"<<std::endl;
    }
    if(trials.empty())return false;
    auto sample=trials.front();std::vector<double> seconds;
    for(const auto &f:trials)seconds.push_back(et::real(f,"total_seconds"));
    sample.erase("total_seconds");sample["mean_seconds"]=pt::numeric(pt::average(seconds));sample["stddev_seconds"]=pt::numeric(pt::deviation(seconds));
    for(auto &field:sample)if(field.first.size()>8 && field.first.compare(field.first.size()-8,8,"_seconds")==0 && field.first!="mean_seconds" && field.first!="stddev_seconds") {
        std::vector<double> values;for(const auto &f:trials)values.push_back(et::real(f,field.first.c_str()));field.second=pt::numeric(pt::average(values));
    }
    for(const char *key:{"selftest_cases","checked"}) {auto minimum=et::uint(sample,key);for(const auto &f:trials)minimum=std::min(minimum,et::uint(f,key));sample[key]=std::to_string(minimum);}
    sample["condition"]=std::to_string(condition);sample["target_bits"]=std::to_string(shape.target_bits);
    sample["arithmetic_bits"]=std::to_string(shape.arithmetic_bits);sample["carrier_exponent"]=std::to_string(carrier);
    sample["modulus_kind"]=json_string(shape.mersenne?"mersenne":"generic");sample["benchmark_kind"]=json_string(input.kind);
    sample["b1"]=std::to_string(input.record.b1);sample["b2"]=std::to_string(b2);sample["repeats"]=std::to_string(trials.size());
    sample["warmup_runs"]=o.tune_warmup?"1":"0";
    pt::stamp_provenance(sample);
    sample["measurement_ids"]=pt::encoded_ids({stage2_queue::token()});
    sample["source"]="\"measured\"";sample["execution_path"]=json_string(shape.fold?(shape.frontier?"resident":"host_frontier"):"host_fold_frontier");
    sample["giant_chunk_points"]=std::to_string(shape.giant_chunk);sample["giant_chain_min"]=std::to_string(shape.chain_min);
    sample["giant_force_ladder"]=shape.force_ladder?"1":"0";
    // The prediction was formed before admitting this new measurement. Store
    // its error as independent holdout evidence, never the training residual.
    if(predicted) {
        portable_prediction_fields(sample,prediction,"before_measurement");
        sample["validation_count"]="1";
        sample["validation_max_relative_error"]=pt::numeric(std::abs(prediction.seconds-pt::average(seconds))/pt::average(seconds));
    }
    if(completed_repeats)*completed_repeats=unsigned(trials.size());
    if(completed_sample)*completed_sample=sample;
    return profile.update(std::move(sample));
}
void portable_ntt(Options o,pt::Profile &profile,ecm_stage2::Word condition,pt::Budget &budget,const fs::path &evidence,
    const std::vector<std::pair<unsigned,uint64_t>> &grid,unsigned repeats) {
    struct Sink {pt::Profile &profile;ecm_stage2::Word condition;unsigned repeats;fs::path evidence;bool skipped=false;};
    Sink sink{profile,condition,repeats,evidence};
    for(auto item:grid) {
        if(budget.expired() || stop_requests)break;
        bool exists=false;for(const auto &s:profile.ntt)if(et::uint(s,"condition")==condition && et::uint(s,"length")==1ull<<item.first &&
            et::uint(s,"batch")==item.second && et::uint(s,"repeats")>=repeats){exists=true;break;}
        if(exists)continue;
        sink.skipped=false;
        const auto callback=[](const char *json,void *ctx) {
            auto &sink=*static_cast<Sink*>(ctx);append(sink.evidence/"ntt.jsonl",json);
            auto f=et::fields(json);if(et::required(f,"type")!="\"sample\"")return;
            if(et::required(f,"status")=="\"skipped_memory\""){sink.skipped=true;return;}
            if(et::required(f,"status")!="\"measured\"")return;
            f["repeats"]=std::to_string(sink.repeats);et::validate_ntt_measurement(f);
            const auto times=et::array(et::required(f,"seconds"));
            f["mean_seconds"]=pt::numeric(pt::average(times));f["stddev_seconds"]=pt::numeric(pt::deviation(times));
            f["conv_iter_per_s"]=pt::numeric(et::uint(f,"batch")/pt::average(times));f["batch_iter_per_s"]=pt::numeric(1/pt::average(times));
            pt::stamp_provenance(f);
            f["measurement_ids"]=pt::encoded_ids({stage2_queue::token()});f.erase("median_seconds");
            for(auto i=f.begin();i!=f.end();)if(!i->second.empty() && i->second.front()=='[')i=f.erase(i);else ++i;
            f.erase("type");f.erase("device_index");
            f["condition"]=std::to_string(sink.condition);f["source"]="\"measured\"";sink.profile.update_ntt(std::move(f));
        };
        const uint64_t slices=item.second;
        if(ecm_cuda_stage2_tune_ntt_batches(o.device,item.first,item.first,repeats,
            o.tune_memory_mb*1048576,&slices,1,callback,&sink,o.tune_warmup) && !sink.skipped)
            throw std::runtime_error("NTT calibration failed or shape did not fit; inspect evidence");
    }
}
void run_portable_ntt(const Options &o,const fs::path &destination) {
    PortableQuiet quiet;
    auto guard=portable_lock(destination);auto profile=portable_read(destination);
    const auto condition=profile.condition(portable_condition(o));pt::Budget budget(double(o.tune_budget));
    const auto evidence=fs::current_path()/"data"/"experiments"/
        ("portable_ntt_tune_"+std::to_string(GetCurrentProcessId())+"_"+std::to_string(GetTickCount64()));fs::create_directories(evidence);
    const auto binary=portable_identity(evidence);
    std::vector<std::pair<unsigned,uint64_t>> grid;
    for(unsigned n=o.tune_first;n<=unsigned(o.tune_last);++n)for(auto b:o.tune_slices)grid.push_back({n,b});
    portable_ntt(o,profile,condition,budget,evidence,grid,unsigned(o.tune_repeats));
    profile.metadata["component_unit"]="\"field_convolution\"";profile.metadata["effort_level"]=std::to_string(o.tune_level);
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("NTT calibration binary changed; evidence retained");
    portable_publish(destination,profile);
    std::cout<<"tune_complete: ntt_summaries="<<profile.ntt.size()<<" elapsed="<<budget.elapsed()<<" s profile="<<destination.string()
        <<" evidence="<<evidence.string()<<std::endl;
}
std::vector<Record> portable_queue_range(const fs::path &worktodo,int worker,const fs::path &save_dir) {
    std::vector<Record> range;std::ifstream in(worktodo);std::string line;int section=1;
    while(std::getline(in,line)) {
        bool header=false;const int next=ecm_worktodo_parse_worker_header(line,&header);
        if(header){section=next;continue;}
        if(section!=worker || trim(line).empty() || trim(line)[0]=='#')continue;
        try {
            const auto fields=queue_fields(line);Big expected;std::string error;
            if(!ecm_compute_stage2_n(fields.task,expected.z,error))continue;
            const auto records_in_task=records(absolute_from(save_dir,fields.task.save_name),fields.skip,1);
            if(!records_in_task.empty() && records_in_task.front().n==number(expected.z))range.push_back(records_in_task.front());
        } catch(const std::exception &) {
            // Invalid future tasks are handled by the normal queue consumer.
            // They provide no calibration coverage and are never rewritten here.
        }
    }
    return range;
}
void ensure_portable_calibration(Options &o,const Record &r,const Settings &settings,const std::vector<Record> &range={}) {
    auto profile=portable_read(o.tune_profile);const auto current=portable_condition(o);const auto ignore=pt::Ignore::parse(o.tune_ignore);
    Big n,plus;mpz_set_str(n.z,r.n.c_str(),16);mpz_add_ui(plus.z,n.z,1);const unsigned bits=unsigned(mpz_sizeinbase(n.z,2));
    const bool ordinary_kind=mpz_popcount(plus.z)==1;
    const unsigned needed_carrier=o.has_carrier?o.carrier_exponent:(o.auto_carrier && !ordinary_kind?o.inferred_carrier:0);
    unsigned low=16385,high=0;double ordinary_distance=100,carrier_distance=100;
    bool components=false,ordinary=false,carrier_covered=!needed_carrier;
    const auto path=portable_shape(o,r,std::min(r.b1,uint64_t(20))+42000,210,0);
    for(const auto &s:profile.samples)if(pt::measured(s) && pt::matches(profile.conditions.at(et::uint(s,"condition")),current,ignore)) {
        if(!ignore.memory && (bool(et::uint(s,"fold_resident"))!=path.fold || bool(et::uint(s,"frontier_resident"))!=path.frontier))continue;
        if((pt::text(s,"modulus_kind")=="mersenne")==ordinary_kind) {
            const auto width=unsigned(et::uint(s,"arithmetic_bits"));
            low=std::min(low,width);high=std::max(high,width);ordinary=true;
            ordinary_distance=std::min(ordinary_distance,std::abs(std::log2(double(bits)/width)));
        }
        if(needed_carrier && pt::text(s,"modulus_kind")=="mersenne")
            carrier_distance=std::min(carrier_distance,std::abs(std::log2(double(needed_carrier)/et::uint(s,"arithmetic_bits"))));
    }
    for(const auto &s:profile.ntt)if(pt::matches(profile.conditions.at(et::uint(s,"condition")),current,ignore))components=true;
    bool width_covered=(low<=bits && bits<=high) || (o.prediction_policy=="model" && ordinary_distance<=std::log2(1.25));
    if(o.prediction_policy=="model")carrier_covered|=carrier_distance<=std::log2(1.25);
    else for(const auto &sample:profile.samples)if(pt::measured(sample) && et::uint(sample,"carrier_exponent")==needed_carrier &&
            pt::matches(profile.conditions.at(et::uint(sample,"condition")),current,ignore))carrier_covered=true;
    for(const auto &future:range) {
        Big target,next;mpz_set_str(target.z,future.n.c_str(),16);mpz_add_ui(next.z,target.z,1);
        const auto width=mpz_sizeinbase(target.z,2);
        if((mpz_popcount(next.z)==1)==ordinary_kind && low<=width && width<=high)width_covered=true;
    }
    if(width_covered && components && ordinary && carrier_covered)return;
    if(!o.short_calibration)return;
    auto guard=portable_lock(o.tune_profile);profile=portable_read(o.tune_profile);
    const auto condition=profile.condition(current);const auto evidence=fs::current_path()/"data"/"experiments"/
        ("stage2_short_calibration_"+std::to_string(GetCurrentProcessId())+"_"+std::to_string(GetTickCount64()));fs::create_directories(evidence);
    const auto binary=portable_identity(evidence);
    // Device initialization above lies outside the measurement budget.
    pt::Budget budget(double(o.short_budget));std::cout<<"stage2_short_calibration: N="<<bits<<" budget="<<o.short_budget<<" s reason="
        <<(low>high?"no_suitable_profile":!components?"missing_ntt":"outside_calibrated_widths")<<std::endl;
    if(!width_covered || !ordinary || !carrier_covered) {
        // Reuse the validated affine Stage1 point. A fixed sigma=26 curve may
        // be singular or infinity on a composite target whose saved curve is
        // valid; calibration must not make that otherwise valid task fail.
        // Probe small bounds on the same valid curve point. Do not use a
        // production B1 here: giant_points scales with absolute B2, so D=210
        // with B2=production_B1+42000 would defeat the short budget.
        const auto probe_b1=std::min(r.b1,uint64_t(20));
        append(evidence/"target_origin.jsonl","{\"saved_b1\":"+std::to_string(r.b1)+
            ",\"probe_b1\":"+std::to_string(probe_b1)+",\"point_reused\":true}");
        const auto save=evidence/"target.save";
        append(save,"METHOD=ECM; PARAM=0; SIGMA="+std::to_string(r.sigma)+"; B1="+std::to_string(probe_b1)+
            "; N=0x"+r.n+"; X=0x"+r.x+"; Z=1;");
        PortableInput input{save,records(save,0,1).at(0),"validated_saved_point_cost_probe"};
        ecm_stage2::Word calibration_b2=0;
        if(!ecm_stage2::add(probe_b1,42000,calibration_b2))throw std::runtime_error("short calibration B2 overflows");
        if(!ordinary || !width_covered)portable_measure(o,settings,input,calibration_b2,210,0,2,budget,evidence,profile,condition,true);
        if(!carrier_covered && !budget.expired())portable_measure(o,settings,input,calibration_b2,210,needed_carrier,2,budget,evidence,profile,condition,true);
    }
    portable_ntt(o,profile,condition,budget,evidence,{{16,1},{18,1},{20,1}},2);
    profile.metadata["short_calibration_budget_seconds"]=std::to_string(o.short_budget);
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("short calibration binary changed; evidence retained");
    portable_publish(o.tune_profile,profile);
    std::cout<<"stage2_short_calibration_done: samples="<<profile.samples.size()<<" ntt="<<profile.ntt.size()
        <<" elapsed="<<budget.elapsed()<<" s profile="<<o.tune_profile<<std::endl;
}

struct PortableTuneTarget {
    unsigned bits=0;
    PortableInput input;
    std::vector<uint64_t> bounds;
    unsigned carrier=0;
    bool grid_mersenne=false;
};
PortableInput portable_freeze_input(const PortableInput &input,const fs::path &snapshot,const fs::path &destination) {
    std::error_code error;
    if((fs::equivalent(destination,input.save,error) && !error) ||
       upper(destination.lexically_normal().string())==upper(input.save.lexically_normal().string()))
        throw std::runtime_error("tune-file must differ from every task save");
    std::ifstream in(input.save,std::ios::binary);std::string line;
    in.seekg(std::streamoff(input.record.offset));
    if(!std::getline(in,line) || fingerprint(line)!=input.record.hash)
        throw TaskInputError("tune save changed before snapshot");
    append(snapshot,line);
    auto record=records(snapshot,0,1).at(0);
    if(record.n!=input.record.n || record.x!=input.record.x || record.b1!=input.record.b1 || record.sigma!=input.record.sigma)
        throw TaskInputError("tune save snapshot differs from selected record");
    return {snapshot,record,input.kind};
}
unsigned portable_task_carrier(const EcmStage2Task &task,const Record &record,bool enabled) {
    if(!enabled || task.k!="1" || task.b!="2" || task.c!="-1" ||
       task.n<2 || task.n>ecm_stage2::max_input_bits)return 0;
    ecm_stage2::ModulusContext modulus;std::string error;
    if(!modulus.configure(record.n.c_str(),16,unsigned(task.n),error))
        throw TaskInputError("worktodo carrier: "+error);
    // Full M_p already uses native Mersenne reduction in ordinary mode.
    if(modulus.target_bits()==task.n && !modulus.lifted())return 0;
    return unsigned(task.n);
}
std::vector<PortableTuneTarget> portable_tune_tasks(const Options &o,const Settings &settings,
    const fs::path &worktodo,const fs::path &save_dir,const std::vector<uint64_t> &fallback_bounds,
    const fs::path &evidence,const fs::path &destination,bool require_tasks) {
    std::ifstream in(worktodo);std::vector<PortableTuneTarget> targets;
    if(!in) {
        if(require_tasks || fs::exists(worktodo))throw std::runtime_error("cannot read tune worktodo: "+worktodo.string());
        return targets;
    }
    std::string line;int section=1;size_t line_number=0,active=0;
    std::set<std::string> seen;
    while(std::getline(in,line)) {
        ++line_number;
        if(line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);
        bool header=false;const auto next=ecm_worktodo_parse_worker_header(line,&header);
        if(next){section=next;continue;}
        line=trim(line);
        if(header || section!=o.worker || line.empty() || line.front()=='#' || line.front()==';')continue;
        if(line.compare(0,10,"ECMSTAGE2="))continue;
        ++active;
        try {
            const auto task=queue_fields(line);Big expected;std::string error;
            if(!ecm_compute_stage2_n(task.task,expected.z,error))throw TaskInputError("worktodo N: "+error);
            const auto save=absolute_from(save_dir,task.task.save_name);
            auto records_in_task=records(save,task.skip,1);
            if(records_in_task.empty())throw TaskInputError("no available save record after skip");
            const auto &r=records_in_task.front();
            if(r.n!=number(expected.z))throw TaskInputError("worktodo N differs from save N");
            if(r.x=="0")throw TaskInputError("saved X=0 gives no usable Stage1 point");
            PortableTuneTarget target;
            target.bits=unsigned(mpz_sizeinbase(expected.z,2));
            if(target.bits<2 || target.bits>ecm_stage2::max_input_bits)throw TaskInputError("target width is outside 2..16384 bits");
            target.input={save,r,"validated_worktodo_save"};
            target.carrier=portable_task_carrier(task.task,r,o.auto_carrier);
            const auto b2=task.b2?task.b2:settings.b2;
            target.bounds=!o.tune_b2s.empty()?o.tune_b2s:b2?std::vector<uint64_t>{b2}:fallback_bounds;
            for(auto b:target.bounds)if(b>uint64_t(INT64_MAX)-8192)
                throw TaskInputError("tune B2 must fit signed indices");
            target.bounds.erase(std::remove_if(target.bounds.begin(),target.bounds.end(),
                [&](uint64_t b){return b<=r.b1;}),target.bounds.end());
            if(target.bounds.empty())throw TaskInputError("no tune B2 exceeds saved B1");
            std::string key=r.n+":"+std::to_string(r.b1)+":"+std::to_string(target.carrier);
            for(auto b:target.bounds)key+=':'+std::to_string(b);
            if(!seen.insert(key).second)continue;
            target.input=portable_freeze_input(target.input,evidence/("task_"+std::to_string(targets.size())+".save"),destination);
            append(evidence/"target_origin.jsonl","{\"worktodo\":"+json_string(worktodo.string())+
                ",\"line\":"+std::to_string(line_number)+",\"task\":"+json_string(line)+",\"save\":"+json_string(save.string())+
                ",\"snapshot\":"+json_string(target.input.save.string())+
                ",\"record\":"+std::to_string(r.index)+",\"record_hash\":"+std::to_string(r.hash)+
                ",\"target_bits\":"+std::to_string(target.bits)+",\"carrier_exponent\":"+std::to_string(target.carrier)+'}');
            targets.push_back(std::move(target));
        } catch(const TaskInputError &error) {
            std::cout<<"ecm_tune_input_skip: line="<<line_number<<" reason="<<error.what()<<std::endl;
        }
    }
    if(in.bad())throw std::runtime_error("tune worktodo read failed");
    if(targets.empty() && (require_tasks || active))throw std::runtime_error("tune worktodo has no valid available save tasks");
    return targets;
}
unsigned portable_tune_save_carrier(const Options &o,const Record &record,const fs::path &worktodo) {
    // This optional lookup never needs to open unrelated future save files.
    std::ifstream in(worktodo);if(!in)return 0;
    std::string line;int section=1;
    while(std::getline(in,line)) {
        if(line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);
        bool header=false;const auto next=ecm_worktodo_parse_worker_header(line,&header);
        if(next){section=next;continue;}
        line=trim(line);
        if(header || section!=o.worker || line.compare(0,10,"ECMSTAGE2="))continue;
        try {
            const auto task=queue_fields(line);
            if(task.task.k!="1" || task.task.b!="2" || task.task.c!="-1" ||
               task.task.n<2 || task.task.n>ecm_stage2::max_input_bits)continue;
            Big expected;std::string error;
            if(ecm_compute_stage2_n(task.task,expected.z,error) && number(expected.z)==record.n)
                return portable_task_carrier(task.task,record,o.auto_carrier);
        } catch(const std::runtime_error &) {
            // A future malformed task is not authority for this valid save.
        }
    }
    if(in.bad())throw std::runtime_error("tune carrier worktodo read failed");
    return 0;
}
void run_portable_tune(Options o,const Settings &settings,const fs::path &destination,
    const fs::path &default_worktodo,const fs::path &save_dir) {
    PortableQuiet quiet;
    o.short_batch_tune=false;configure_short_ntt_batches(o);
    if(!o.tune_save.empty() && !o.tune_exponents.empty())throw std::runtime_error("choose tune-save or tune-exponents");
    if(!o.tune_worktodo.empty() && (!o.tune_save.empty() || !o.tune_exponents.empty()))
        throw std::runtime_error("choose tune-worktodo, tune-save or tune-exponents");
    if(o.tune_carrier && o.tune_save.empty())throw std::runtime_error("carrier tuning requires a validated --tune-save");
    if(o.has_tune_tail_samples && o.tune_tail_samples)throw std::runtime_error("portable tune uses the default grid; legacy tune-tail-samples accepts only 0");
    auto guard=portable_lock(destination);auto profile=portable_read(destination);const auto current=portable_condition(o);
    profile.metadata["error_limit"]=pt::numeric(o.tune_error);
    const auto condition=profile.condition(current);const unsigned level=o.tune_level?o.tune_level:1;
    auto grid=pt::grid(level,o.auto_max?o.auto_max:pt::default_b2_max);
    if(!o.tune_ds.empty())grid.ds=o.tune_ds;if(!o.tune_b2s.empty())grid.b2s=o.tune_b2s;
    if(!o.tune_exponents.empty()){grid.widths.clear();for(auto bits:o.tune_exponents)grid.widths.push_back(unsigned(bits));}
    if(o.has_tune_repeats)grid.repeats=o.tune_repeats;
    const auto evidence=fs::current_path()/"data"/"experiments"/
        ("portable_ecm_tune_"+std::to_string(GetCurrentProcessId())+"_"+std::to_string(GetTickCount64()));fs::create_directories(evidence);
    const auto binary=portable_identity(evidence);
    for(const auto &path:o.tune_ntt_profiles) {
        const auto old=pt::Profile::load(absolute_from(fs::current_path(),path));
        for(auto s:old.ntt){s["condition"]=std::to_string(profile.condition(old.conditions.at(et::uint(s,"condition"))));profile.update_ntt(std::move(s));}
    }
    pt::Budget budget(double(o.tune_budget));
    std::vector<PortableTuneTarget> targets;
    const auto task_path=o.tune_worktodo.empty()?default_worktodo:absolute_from(fs::current_path(),o.tune_worktodo);
    if(!o.tune_save.empty()) {
        const auto path=absolute_from(fs::current_path(),o.tune_save);auto rec=records(path,0,1);if(rec.empty())throw std::runtime_error("empty tune save");
        Big n,plus;mpz_set_str(n.z,rec[0].n.c_str(),16);mpz_add_ui(plus.z,n.z,1);
        const auto bits=unsigned(mpz_sizeinbase(n.z,2));
        unsigned carrier=o.tune_carrier;
        if(!carrier && o.auto_carrier) {
            // A cofactor save alone does not reveal p. Obtain it only from a
            // validated matching task. Full M_p already has native reduction.
            carrier=portable_tune_save_carrier(o,rec[0],task_path);
            if(carrier==bits && mpz_popcount(plus.z)==1)carrier=0;
        }
        if(carrier) {ecm_stage2::ModulusContext modulus;std::string error;
            if(!modulus.configure(rec[0].n.c_str(),16,carrier,error))throw std::runtime_error(error);}
        auto input=portable_freeze_input({path,rec[0],"validated_save"},evidence/"target.save",destination);
        append(evidence/"target_origin.jsonl","{\"save\":"+json_string(path.string())+
            ",\"snapshot\":"+json_string(input.save.string())+",\"record\":"+std::to_string(rec[0].index)+
            ",\"record_hash\":"+std::to_string(rec[0].hash)+",\"carrier_exponent\":"+std::to_string(carrier)+'}');
        targets.push_back({bits,std::move(input),grid.b2s,carrier});
    } else if(o.tune_exponents.empty()) {
        targets=portable_tune_tasks(o,settings,task_path,save_dir,grid.b2s,evidence,destination,!o.tune_worktodo.empty());
    }
    if(targets.empty())for(auto bits:grid.widths) {
        if(o.tune_arithmetic!="mersenne")targets.push_back({bits,{},grid.b2s,0,false});
        if(o.tune_arithmetic!="generic")targets.push_back({bits,{},grid.b2s,0,true});
    }
    for(const auto &target:targets)if(!target.input.save.empty()) {
        std::error_code error;
        if((fs::equivalent(destination,target.input.save,error) && !error) ||
           upper(destination.lexically_normal().string())==upper(target.input.save.lexically_normal().string()))
            throw std::runtime_error("tune-file must differ from every task save");
    }
    std::sort(grid.ds.begin(),grid.ds.end());grid.ds.erase(std::unique(grid.ds.begin(),grid.ds.end()),grid.ds.end());
    size_t full_grid=0;for(const auto &target:targets)full_grid+=target.bounds.size()*grid.ds.size()*(target.carrier?2:1);
    struct Case {size_t target;uint64_t b2,d;unsigned carrier;double priority;};std::vector<Case> cases;
    size_t measured=0,skipped=0;
    std::cout<<"ecm_tune_inputs: targets="<<targets.size()<<" source="<<
        (!o.tune_save.empty()?"save":targets.front().input.save.empty()?"width_grid":"worktodo")
        <<" auto_carrier="<<o.auto_carrier<<" warmup="<<o.tune_warmup<<std::endl;
    // Complete ordinary and carrier default references before spending the
    // remaining budget on neighborhoods. Explicit D lists remain exact.
    size_t paired_references=0;
    std::map<std::pair<size_t,pt::Shape::Key>,et::Fields> reference_samples;
    for(size_t target_index=0;target_index<targets.size();++target_index) {
        if(budget.expired() || stop_requests)break;
        auto &target=targets[target_index];if(!target.carrier)continue;
        for(auto b:target.bounds)for(auto carrier:{0u,target.carrier}) {
            if(budget.expired() || stop_requests)break;
            const auto d=portable_default_d(o,target.input.record,b,carrier);
            if(!std::binary_search(grid.ds.begin(),grid.ds.end(),d))continue;
            auto shape=portable_shape(o,target.input.record,b,d,carrier);std::string plan;
            if(!portable_admit(o,target.input.record,shape,plan))continue;
            const auto p=ecm_stage2::cost::phi(d)/2,points=b/d+2;
            if(points<=p || (o.tune_max_batches && (points+p-1)/p>o.tune_max_batches))continue;
            pt::Estimator estimator(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
            const auto prior=estimator.observed(shape);
            if(prior && et::uint(*prior,"repeats")>=grid.repeats){++skipped;++paired_references;continue;}
            unsigned completed=0;
            et::Fields formal_sample;
            const bool changed=portable_measure(o,settings,target.input,b,d,carrier,grid.repeats,
                budget,evidence,profile,condition,false,&completed,&formal_sample);
            if(completed)++measured;
            if(changed)portable_publish(destination,profile);
            if(completed>=grid.repeats) {
                ++paired_references;
                pt::Estimator updated(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
                reference_samples[{target_index,updated.shape(formal_sample).key()}]=std::move(formal_sample);
            }
        }
    }
    // NTT measurements are component evidence, never a substitute for paired
    // full-engine D comparisons. No tiny-P full-engine seed is required.
    portable_ntt(o,profile,condition,budget,evidence,{{16,1},{18,1},{20,1},{22,1}},grid.repeats);
    for(size_t target_index=0;target_index<targets.size();++target_index) {
        if(budget.expired() || stop_requests)break;
        auto &target=targets[target_index];
        if(target.input.save.empty()) {
            try {target.input=portable_input(target.bits,evidence,target.grid_mersenne);}
            catch(const TaskInputError &error) {
                if(!target.grid_mersenne)throw;
                std::cout<<"ecm_tune_input_skip: arithmetic_bits="<<target.bits<<" kind=mersenne reason="<<error.what()<<std::endl;
                continue;
            }
            if(target.grid_mersenne)target.carrier=target.bits;
            Big n;mpz_set_str(n.z,target.input.record.n.c_str(),16);target.bits=unsigned(mpz_sizeinbase(n.z,2));
        }
        const auto bits=target.bits;
        const auto &input=target.input;const auto &r=input.record;
        for(auto b:target.bounds)for(unsigned mode=0;mode<(target.carrier && !target.grid_mersenne?2u:1u);++mode) {
            if(b<=r.b1 || budget.expired() || stop_requests)continue;
            const unsigned carrier=target.grid_mersenne || mode?target.carrier:0;
            const auto baseline=portable_default_d(o,r,b,carrier);
            struct Outcome {enum State {Tested,Rejected,Deferred} state=Deferred;pt::LocalSearch::Timing time;};
            std::map<uint64_t,Outcome> outcomes;
            std::map<uint64_t,std::pair<bool,pt::Shape>> admissions;
            auto admit=[&](uint64_t d,pt::Shape &shape) {
                auto at=admissions.find(d);
                if(at==admissions.end()) {
                    shape=portable_shape(o,r,b,d,carrier);std::string plan;
                    const bool ok=portable_admit(o,r,shape,plan);
                    if(!plan.empty())append(evidence/"candidate_plans.jsonl",plan);
                    at=admissions.emplace(d,std::make_pair(ok,shape)).first;
                }
                shape=at->second.second;return at->second.first;
            };
            auto measure=[&](uint64_t d) {
                if(outcomes.count(d))return outcomes.at(d);
                Outcome result;
                if(budget.expired() || stop_requests)return outcomes[d]=result;
                auto shape=portable_shape(o,r,b,d,carrier);pt::Work topology;
                const auto p=ecm_stage2::cost::phi(d)/2,points=b/d+2;
                if(points<=p || (o.tune_max_batches && (points+p-1)/p>o.tune_max_batches) ||
                   !pt::work(shape,portable_query,[](uint64_t n,uint64_t batch){return double(n)*batch;},topology) ||
                   !admit(d,shape)) {result.state=Outcome::Rejected;++skipped;return outcomes[d]=result;}
                pt::Estimator model(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
                const auto prior=model.observed(shape);
                const auto reference=reference_samples.find({target_index,shape.key()});
                const bool reuse_current=reference!=reference_samples.end();
                const bool reuse=reuse_current || (prior && et::uint(*prior,"repeats")>=grid.repeats);
                unsigned completed=0;
                et::Fields formal_sample;
                if(!reuse) {
                    // A known local duration is a budget bound. A remote tiny-P
                    // prediction cannot reject a production-sized measurement.
                    if(prior && et::real(*prior,"mean_seconds")*(grid.repeats+(o.tune_warmup?1:0))>budget.remaining())
                        return outcomes[d]=result;
                    const bool changed=portable_measure(o,settings,input,b,d,carrier,grid.repeats,
                        budget,evidence,profile,condition,target.grid_mersenne,&completed,&formal_sample);
                    if(completed)++measured;
                    if(changed)portable_publish(destination,profile);
                } else ++skipped;
                pt::Estimator updated(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
                const auto observation=updated.observed(shape);
                if(!observation || (!reuse && completed<grid.repeats))return outcomes[d]=result;
                result.state=Outcome::Tested;
                // Rank this completed trial population; persistent rows combine
                // independent batches with their true repeat counts.
                const auto &timing=reuse_current?reference->second:reuse?*observation:formal_sample;
                result.time={et::real(timing,"mean_seconds"),et::real(timing,"stddev_seconds")};
                cases.push_back({target_index,b,d,carrier,0});
                std::cout<<"ecm_tune_progress: N="<<bits<<" B2="<<b<<" D="<<d
                    <<" mean="<<result.time.seconds<<" s reused="<<reuse<<std::endl;
                return outcomes[d]=result;
            };
            // Measure the protected reference independently of model seeds.
            // Explicit user lists remain exact and may intentionally omit it.
            const bool has_baseline=std::binary_search(grid.ds.begin(),grid.ds.end(),baseline);
            std::cout<<"ecm_tune_window: N="<<bits<<" B2="<<b<<" carrier="<<carrier
                <<" baseline_D="<<baseline<<" search=ntt_intervals basins_limit="<<grid.basins
                <<" catalog_candidates="<<ecm_stage2::d_catalog::count<<std::endl;
            if(has_baseline)measure(baseline);
            struct Node {uint64_t d,p;double rank;};
            struct Basin {pt::NttInterval key;std::vector<Node> nodes;double rank;};
            std::map<pt::NttInterval,std::vector<Node>> groups;
            auto reference=portable_shape(o,r,b,baseline,carrier);pt::Work reference_work;
            if(admissions.count(baseline) && admissions.at(baseline).first)reference=admissions.at(baseline).second;
            pt::Estimator guide(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
            pt::WidthPredictor reusable_guide(guide);
            const bool reference_ok=guide.cached_work(reference,reference_work);
            pt::WidthEstimate reference_estimate;
            const bool reference_predicted=reusable_guide.predict(reference,reference_estimate);
            bool ranking_complete=true;
            for(auto d:grid.ds) {
                if(budget.expired() || stop_requests){ranking_complete=false;break;}
                const auto p=ecm_stage2::cost::phi(d)/2,points=b/d+2;
                if(points<=p || (o.tune_max_batches && (points+p-1)/p>o.tune_max_batches))continue;
                auto shape=portable_shape(o,r,b,d,carrier);pt::Work w;
                if(!guide.cached_work(shape,w) || !w.exact_topology)continue;
                double score=reference_ok?pt::relative_work(w,reference_work):double(w.largest_ntt);
                pt::WidthEstimate prediction;
                if(reference_predicted && reusable_guide.predict(shape,prediction))score=prediction.rank/reference_estimate.rank;
                if(score>0 && std::isfinite(score))groups[pt::interval(shape,w)].push_back({d,p,score});
            }
            std::vector<Basin> basins;
            for(auto &group:groups) {
                auto &nodes=group.second;
                std::sort(nodes.begin(),nodes.end(),[](const Node &a,const Node &b){return a.p!=b.p?a.p<b.p:a.d>b.d;});
                double rank=std::numeric_limits<double>::infinity();for(const auto &node:nodes)rank=std::min(rank,node.rank);
                basins.push_back({group.first,std::move(nodes),rank});
            }
            std::sort(basins.begin(),basins.end(),[](const Basin &a,const Basin &b){return a.rank!=b.rank?a.rank<b.rank:a.key<b.key;});
            // Bounded multi-region restarts may cross an uphill transform step.
            // Sampling scheduling is separate from paired/model production guards.
            size_t requested=0;
            while(requested<basins.size() && requested<grid.basins &&
                  (requested==0 || basins[requested].rank<=1.5*basins.front().rank))++requested;
            size_t explored=0;bool complete=ranking_complete && requested!=0;
            for(size_t region=0;region<requested;++region) {
                if(budget.expired() || stop_requests){complete=false;break;}
                auto &basin=basins[region];const auto &nodes=basin.nodes;
                std::vector<size_t> seeds;for(size_t i=0;i<nodes.size();++i)seeds.push_back(i);
                std::sort(seeds.begin(),seeds.end(),[&](size_t a,size_t b){return nodes[a].rank!=nodes[b].rank?
                    nodes[a].rank<nodes[b].rank:nodes[a].d<nodes[b].d;});
                size_t center=nodes.size();
                // Existing paired winners get priority, with actual route admission.
                const auto base=outcomes.find(baseline);
                if(base!=outcomes.end() && base->second.state==Outcome::Tested) {
                    pt::LocalSearch::Timing best=base->second.time;
                    pt::Estimator observed(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
                    for(size_t i=0;i<nodes.size();++i) {
                        if(budget.expired() || stop_requests){complete=false;break;}
                        auto shape=portable_shape(o,r,b,nodes[i].d,carrier);
                        const auto sample=observed.observed(shape);
                        if(!sample || !pt::LocalSearch::improves({et::real(*sample,"mean_seconds"),et::real(*sample,"stddev_seconds")},best))continue;
                        if(!admit(nodes[i].d,shape))continue;
                        const auto actual=observed.observed(shape);
                        if(actual && pt::LocalSearch::improves({et::real(*actual,"mean_seconds"),et::real(*actual,"stddev_seconds")},best)) {
                            best={et::real(*actual,"mean_seconds"),et::real(*actual,"stddev_seconds")};center=i;
                        }
                    }
                }
                // A rejected model seed does not reject every other D in the region.
                if(center==nodes.size())for(auto i:seeds) {
                    if(budget.expired() || stop_requests){complete=false;break;}
                    pt::Shape shape;
                    if(admit(nodes[i].d,shape)){center=i;break;}
                    Outcome rejected;rejected.state=Outcome::Rejected;outcomes[nodes[i].d]=rejected;
                }
                if(center==nodes.size()) {if(budget.expired() || stop_requests)break;++explored;continue;}
                std::cout<<"ecm_tune_window: N="<<bits<<" B2="<<b<<" carrier="<<carrier
                    <<" region="<<(region+1)<<'/'<<requested<<" center_D="<<nodes[center].d
                    <<" fold_ntt="<<basin.key.key[0]<<" tree_ntt="<<basin.key.key[2]
                    <<" predicted_ratio="<<basin.rank<<" restart="<<(region!=0)<<std::endl;
                if(!o.tune_ds.empty()) {
                    // Explicit research lists retain their established local
                    // traversal; automatic catalogues use the shared search.
                    pt::LocalSearch search(center,nodes.size());
                    for(;;) {
                        const auto i=search.next();if(i==nodes.size())break;
                        if(budget.expired() || stop_requests){search.deferred.insert(i);outcomes[nodes[i].d]=Outcome{};break;}
                        const auto result=measure(nodes[i].d);
                        if(result.state==Outcome::Tested)search.tested[i]=result.time;
                        else if(result.state==Outcome::Rejected)search.rejected.insert(i);
                        else search.deferred.insert(i);
                    }
                    complete=search.complete() && complete;
                } else {
                    std::vector<pt::DSearchNode> pool;
                    for(const auto &n:nodes)pool.push_back({n.d,n.p,basin.key.key,n.rank});
                    for(size_t i=0;i<pool.size();++i)if(outcomes.count(pool[i].d)) {
                        const auto &prior=outcomes.at(pool[i].d);
                        if(prior.state==Outcome::Tested) {
                            pool[i].seconds=prior.time.seconds;pool[i].stddev=prior.time.stddev;pool[i].samples=grid.repeats;
                        } else pool[i].rejected=true;
                    }
                    for(;;) {
                        const auto step=pt::DSearch(pool,nodes[center].d,grid.repeats,std::max(4u,grid.basins*2),1).next();
                        if(!step.available)break;const auto i=step.index;
                        if(budget.expired() || stop_requests){complete=false;outcomes[pool[i].d]=Outcome{};break;}
                        const auto result=measure(pool[i].d);
                        if(result.state==Outcome::Tested) {
                            pool[i].seconds=result.time.seconds;pool[i].stddev=result.time.stddev;pool[i].samples=grid.repeats;
                        } else {pool[i].rejected=true;if(result.state==Outcome::Deferred)complete=false;}
                    }
                }
                ++explored;
            }
            size_t tested=0,rejected=0,deferred=0;uint64_t best_d=baseline;
            pt::LocalSearch::Timing best_time;
            if(outcomes.count(baseline) && outcomes.at(baseline).state==Outcome::Tested)best_time=outcomes.at(baseline).time;
            for(const auto &item:outcomes) {
                if(item.second.state==Outcome::Tested) {
                    ++tested;
                    if(!best_time.seconds || pt::LocalSearch::improves(item.second.time,best_time))
                        {best_time=item.second.time;best_d=item.first;}
                } else if(item.second.state==Outcome::Rejected)++rejected;
                else ++deferred;
            }
            complete=complete && tested!=0 && explored==requested && !deferred && !(budget.expired() || stop_requests);
            std::cout<<"ecm_tune_window_done: N="<<bits<<" B2="<<b<<" carrier="<<carrier
                <<" best_D="<<best_d<<" tested="<<tested<<" rejected="<<rejected
                <<" deferred="<<deferred<<" regions="<<explored<<'/'<<requested<<" complete="<<complete
                <<" reason="<<(stop_requests?"interrupted":budget.expired()?"budget_expired":deferred?"budget_deferred":
                              !ranking_complete?"ranking_incomplete":!requested?"no_supported_interval":!tested?"no_admissible_interval":
                              complete?"local_regions_complete":"region_deferred")<<std::endl;
        }
    }
    // Persist only final candidate summaries. Estimates cannot replace measured
    // measured values; their source and validation state remain explicit.
    pt::Estimator estimator(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
    std::ostringstream evidence_key;
    for(const auto &s:profile.samples)if(pt::measured(s))evidence_key<<pt::scope(s)<<et::required(s,"mean_seconds")<<';';
    for(const auto &s:profile.ntt)evidence_key<<et::required(s,"condition")<<':'<<et::required(s,"length")<<':'
        <<et::required(s,"batch")<<':'<<et::required(s,"mean_seconds")<<';';
    const auto model_evidence=std::to_string(fingerprint(evidence_key.str()));
    std::vector<et::Fields> estimates;
    for(const auto &c:cases) {
        const auto &r=targets.at(c.target).input.record;
        auto shape=portable_shape(o,r,c.b2,c.d,c.carrier);pt::Estimate e;
        if(!estimator.predict(shape,e))continue;
        et::Fields s={{"condition",std::to_string(condition)},{"target_bits",std::to_string(shape.target_bits)},
            {"arithmetic_bits",std::to_string(shape.arithmetic_bits)},{"carrier_exponent",std::to_string(shape.carrier)},
            {"modulus_kind",json_string(shape.mersenne?"mersenne":"generic")},{"b1",std::to_string(r.b1)},{"b2",std::to_string(c.b2)},
            {"d",std::to_string(c.d)},{"p",std::to_string(ecm_stage2::cost::phi(c.d)/2)},{"fold_resident",shape.fold?"1":"0"},
            {"frontier_resident",shape.frontier?"1":"0"},{"source","\"model\""},{"repeats","0"},
            {"execution_path",json_string(shape.fold?(shape.frontier?"resident":"host_frontier"):"host_fold_frontier")},
            {"mean_seconds",pt::numeric(e.seconds)},{"stddev_seconds",pt::numeric(e.stddev)},
            {"model_anchor_count",std::to_string(e.anchors)},{"model_evidence",json_string(model_evidence)},
            {"model_relative_error_allowance",pt::numeric(e.relative_error)}};
        pt::stamp_provenance(s);estimates.push_back(std::move(s));
    }
    for(auto &s:estimates)profile.update(std::move(s));
    profile.metadata["effort_level"]=std::to_string(level);profile.metadata["budget_seconds"]=std::to_string(o.tune_budget);
    profile.metadata["error_limit"]=pt::numeric(o.tune_error);profile.metadata["grid_candidates"]=std::to_string(full_grid);
    profile.metadata["selected_candidates"]=std::to_string(cases.size());
    profile.metadata["selection"]="\"ntt_interval_descent_v1\"";
    profile.metadata["catalogue_candidates"]=std::to_string(ecm_stage2::d_catalog::count);
    profile.metadata["input_targets"]=std::to_string(targets.size());
    profile.metadata["paired_references"]=std::to_string(paired_references);
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("ECM calibration binary changed; evidence retained");
    portable_publish(destination,profile);
    std::cout<<"ecm_tune_complete: measured="<<measured<<" skipped="<<skipped<<" grid="<<full_grid<<" selected="<<cases.size()<<" summaries="<<profile.samples.size()
        <<" budget_elapsed="<<budget.elapsed()<<" s profile="<<destination.string()<<" evidence="<<evidence.string()<<std::endl;
}
