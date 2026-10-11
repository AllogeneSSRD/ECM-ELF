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
        const auto profile=pt::Profile::load(o.tune_profile);const auto current=portable_condition(o);
        const auto ignore=pt::Ignore::parse(o.tune_ignore);ecm_stage2::ShortNttBatchPolicy policy;
        for(const auto &s:profile.ntt)if(pt::ntt_matches(profile.conditions.at(et::uint(s,"condition")),current,ignore))
            policy.observe(et::uint(s,"length"),et::uint(s,"batch"),et::real(s,"median_seconds"));
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
void portable_publish(const fs::path &path,const pt::Profile &profile) {
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
    s.chain_min=et::environment_uint(env,"giant_chain_min",32768);s.force_ladder=et::environment_uint(env,"giant_ladder",0)!=0;
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
std::vector<unsigned> portable_carriers(const Options &o,const Record &r) {
    if(o.has_carrier)return {o.carrier_exponent};
    std::vector<unsigned> out{0};
    if(o.inferred_carrier) {
        ecm_stage2::ModulusContext m;std::string error;
        if(m.configure(r.n.c_str(),16,o.inferred_carrier,error))out.push_back(o.inferred_carrier);
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
std::string portable_select(Options &o,const Record &r,bool automatic) {
    if(o.tune_profile.empty())return {};
    if(!fs::is_regular_file(o.tune_profile)) {
        if(automatic)throw std::runtime_error("Auto B2 needs a tune summary; enable short calibration or supply a profile");
        return "{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"missing_profile\"}";
    }
    const auto profile=pt::Profile::load(o.tune_profile);const auto condition=portable_condition(o);
    pt::Estimator estimator(profile,condition,pt::Ignore::parse(o.tune_ignore),portable_query);
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
        t1=et::real(*anchor,"median_seconds")*double(r.b1)/et::uint(*anchor,"b1")*
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
            pt::LocalSearch::improves({et::real(s,"median_seconds"),et::real(s,"mad_seconds")},
                {et::real(*base,"median_seconds"),et::real(*base,"mad_seconds")}))
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
                {et::real(*trial,"median_seconds"),et::real(*trial,"mad_seconds")},
                {et::real(*base,"median_seconds"),et::real(*base,"mad_seconds")}))continue;
            e.seconds=et::real(*trial,"median_seconds");e.mad=et::real(*trial,"mad_seconds");
            e.rank=e.seconds+2*e.mad;e.exact=true;e.model="paired_measured_d_v1";
            e.anchor=trial;e.anchors=1;
            e.validated=pt::optional_uint(*trial,"validation_count")>0 &&
                pt::matches(profile.conditions.at(et::uint(*trial,"condition")),condition,pt::Ignore{});
            e.relative_error=e.validated?pt::optional_real(*trial,"validation_max_relative_error"):.25;
        } else {
            if(!estimator.predict(shape,e))continue;
            if(automatic && !(o.has_d && o.d) && d!=default_ds.at({b,carrier}) &&
                !e.exact && std::string(e.model)!="portable_nonnegative_v1")continue;
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
        if((automatic || (o.has_d && o.d)) && !estimator.predict(shape,e)){++rejected;continue;}
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
        result="{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"default_planner_protected_no_measured_win\",\"default_D\":"+
            std::to_string(o.d)+",\"default_model\":\"ntt_ceiling_plateau_v1\"}";
    }
    return result;
}
std::string select_auto_tuned(Options &o,const Record &r){return portable_select(o,r,true);}

struct PortableInput {fs::path save;Record record;std::string kind;};
void merge_portable_tune(const Options &o,const fs::path &destination) {
    if(o.tune_merge.size()>64 || !o.tune_save.empty() || o.tune_carrier || !o.tune_ds.empty() ||
       !o.tune_b2s.empty() || !o.tune_exponents.empty())throw std::runtime_error("merge accepts at most 64 profiles and no curve grid");
    auto guard=portable_lock(destination);auto combined=portable_read(destination);
    for(const auto &path:o.tune_merge) {
        const auto source=pt::Profile::load(absolute_from(fs::current_path(),path));
        for(auto s:source.samples){s["condition"]=std::to_string(combined.condition(source.conditions.at(et::uint(s,"condition"))));combined.update(std::move(s));}
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
    if(mersenne){mpz_mul_2exp(n.z,n.z,bits);mpz_sub_ui(n.z,n.z,1);}
    else {mpz_mul_2exp(n.z,n.z,bits-1);mpz_add_ui(n.z,n.z,bits<16?1:12345);mpz_nextprime(n.z,n.z);}
    if(mpz_sizeinbase(n.z,2)!=bits)throw std::runtime_error("benchmark prime construction exceeded requested width");
    const auto hex=number(n.z),x=et::benchmark_point(hex);
    const auto save=evidence/((mersenne?"m":"p")+std::to_string(bits)+".save");
    append(save,"METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x"+hex+"; X=0x"+x+"; Z=1;");
    return {save,records(save,0,1).at(0),mersenne?"mersenne_benchmark":"probable_prime_gmp_nextprime"};
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
    const std::string stem="measure_"+std::to_string(profile.revision)+"_"+std::to_string(GetTickCount64());
    append(evidence/(stem+".plan.jsonl"),plan);
    std::vector<et::Fields> trials;
    for(unsigned repeat=0;repeat<=repeats;++repeat) {
        // Warmup is one measurement; no new formal repeat starts after expiry.
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
            <<" budget_left="<<budget.remaining()<<" s"<<std::endl;
    }
    if(trials.empty())return false;
    auto sample=trials.front();std::vector<double> seconds;
    for(const auto &f:trials)seconds.push_back(et::real(f,"total_seconds"));
    sample.erase("total_seconds");sample["median_seconds"]=pt::numeric(et::median(seconds));sample["mad_seconds"]=pt::numeric(et::mad(seconds));
    for(auto &field:sample)if(field.first.size()>8 && field.first.compare(field.first.size()-8,8,"_seconds")==0 && field.first!="median_seconds" && field.first!="mad_seconds") {
        std::vector<double> values;for(const auto &f:trials)values.push_back(et::real(f,field.first.c_str()));field.second=pt::numeric(et::median(values));
    }
    for(const char *key:{"selftest_cases","checked"}) {auto minimum=et::uint(sample,key);for(const auto &f:trials)minimum=std::min(minimum,et::uint(f,key));sample[key]=std::to_string(minimum);}
    sample["condition"]=std::to_string(condition);sample["target_bits"]=std::to_string(shape.target_bits);
    sample["arithmetic_bits"]=std::to_string(shape.arithmetic_bits);sample["carrier_exponent"]=std::to_string(carrier);
    sample["modulus_kind"]=json_string(shape.mersenne?"mersenne":"generic");sample["benchmark_kind"]=json_string(input.kind);
    sample["b1"]=std::to_string(input.record.b1);sample["b2"]=std::to_string(b2);sample["repeats"]=std::to_string(trials.size());
    sample["source"]="\"measured\"";sample["execution_path"]=json_string(shape.fold?(shape.frontier?"resident":"host_frontier"):"host_fold_frontier");
    sample["giant_chunk_points"]=std::to_string(shape.giant_chunk);sample["giant_chain_min"]=std::to_string(shape.chain_min);
    sample["giant_force_ladder"]=shape.force_ladder?"1":"0";
    // The prediction was formed before admitting this new measurement. Store
    // its error as independent holdout evidence, never the training residual.
    pt::Estimator model(profile,profile.conditions.at(condition),pt::Ignore::parse(o.tune_ignore),portable_query);pt::Estimate prediction;
    if(model.predict(shape,prediction)) {
        sample["validation_count"]="1";
        sample["validation_max_relative_error"]=pt::numeric(std::abs(prediction.seconds-et::median(seconds))/et::median(seconds));
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
            f["mad_seconds"]=pt::numeric(et::mad(et::array(et::required(f,"seconds"))));
            for(auto i=f.begin();i!=f.end();)if(!i->second.empty() && i->second.front()=='[')i=f.erase(i);else ++i;
            f.erase("type");f.erase("device_index");
            f["condition"]=std::to_string(sink.condition);f["source"]="\"measured\"";sink.profile.update_ntt(std::move(f));
        };
        const uint64_t slices=item.second;
        if(ecm_cuda_stage2_tune_ntt_batches(o.device,item.first,item.first,repeats,
            o.tune_memory_mb*1048576,&slices,1,callback,&sink) && !sink.skipped)
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
    const unsigned needed_carrier=o.has_carrier?o.carrier_exponent:o.inferred_carrier;
    unsigned low=16385,high=0;bool components=false,ordinary=false,carrier_covered=!needed_carrier;
    const auto path=portable_shape(o,r,std::min(r.b1,uint64_t(20))+42000,210,0);
    for(const auto &s:profile.samples)if(pt::measured(s) && pt::matches(profile.conditions.at(et::uint(s,"condition")),current,ignore)) {
        if(!ignore.memory && (bool(et::uint(s,"fold_resident"))!=path.fold || bool(et::uint(s,"frontier_resident"))!=path.frontier))continue;
        if(!et::uint(s,"carrier_exponent") && (pt::text(s,"modulus_kind")=="mersenne")==ordinary_kind) {
            low=std::min(low,unsigned(et::uint(s,"target_bits")));high=std::max(high,unsigned(et::uint(s,"target_bits")));ordinary=true;
        }
        if(needed_carrier && et::uint(s,"carrier_exponent")==needed_carrier)carrier_covered=true;
    }
    for(const auto &s:profile.ntt)if(pt::matches(profile.conditions.at(et::uint(s,"condition")),current,ignore))components=true;
    bool width_covered=low<=bits && bits<=high;
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

void run_portable_tune(Options o,const Settings &settings,const fs::path &destination) {
    PortableQuiet quiet;
    o.short_batch_tune=false;configure_short_ntt_batches(o);
    if(!o.tune_save.empty() && !o.tune_exponents.empty())throw std::runtime_error("choose tune-save or tune-exponents");
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
    std::map<unsigned,PortableInput> inputs;
    if(!o.tune_save.empty()) {
        const auto path=absolute_from(fs::current_path(),o.tune_save);auto rec=records(path,0,1);if(rec.empty())throw std::runtime_error("empty tune save");
        Big n;mpz_set_str(n.z,rec[0].n.c_str(),16);grid.widths={unsigned(mpz_sizeinbase(n.z,2))};inputs.emplace(grid.widths[0],PortableInput{path,rec[0],"validated_save"});
    }
    std::sort(grid.ds.begin(),grid.ds.end());grid.ds.erase(std::unique(grid.ds.begin(),grid.ds.end()),grid.ds.end());
    const size_t full_grid=grid.widths.size()*grid.b2s.size()*grid.ds.size()*(o.tune_carrier?2:1);
    struct Case {unsigned bits;uint64_t b2,d;unsigned carrier;double priority;};std::vector<Case> cases;
    size_t measured=0,skipped=0;
    // NTT measurements are component evidence, never a substitute for paired
    // full-engine D comparisons. No tiny-P full-engine seed is required.
    portable_ntt(o,profile,condition,budget,evidence,{{16,1},{18,1},{20,1},{22,1}},grid.repeats);
    for(auto bits:grid.widths) {
        if(budget.expired() || stop_requests)break;
        if(!inputs.count(bits))inputs.emplace(bits,portable_input(bits,evidence));
        const auto &input=inputs.at(bits);const auto &r=input.record;
        for(auto b:grid.b2s)for(unsigned mode=0;mode<(o.tune_carrier?2u:1u);++mode) {
            if(b<=r.b1 || budget.expired() || stop_requests)continue;
            const unsigned carrier=mode?o.tune_carrier:0;
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
                const bool reuse=prior && et::uint(*prior,"repeats")>=grid.repeats;
                unsigned completed=0;
                et::Fields formal_sample;
                if(!reuse) {
                    // A known local duration is a budget bound. A remote tiny-P
                    // prediction cannot reject a production-sized measurement.
                    if(prior && et::real(*prior,"median_seconds")*(grid.repeats+1)>budget.remaining())
                        return outcomes[d]=result;
                    const bool changed=portable_measure(o,settings,input,b,d,carrier,grid.repeats,
                        budget,evidence,profile,condition,false,&completed,&formal_sample);
                    if(completed)++measured;
                    if(changed)portable_publish(destination,profile);
                } else ++skipped;
                pt::Estimator updated(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
                const auto observation=updated.observed(shape);
                if(!observation || (!reuse && completed<grid.repeats))return outcomes[d]=result;
                result.state=Outcome::Tested;
                // Fastest-on-disk may still be a partial earlier sample. The
                // current neighborhood uses the completed requested repeats,
                // without replacing that older fastest summary with a slower one.
                const auto &timing=reuse?*observation:formal_sample;
                result.time={et::real(timing,"median_seconds"),et::real(timing,"mad_seconds")};
                cases.push_back({bits,b,d,carrier,0});
                std::cout<<"ecm_tune_progress: N="<<bits<<" B2="<<b<<" D="<<d
                    <<" median="<<result.time.seconds<<" s reused="<<reuse<<std::endl;
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
            const bool reference_ok=guide.cached_work(reference,reference_work);
            pt::Estimate reference_estimate;
            const bool reference_predicted=guide.predict(reference,reference_estimate);
            bool ranking_complete=true;
            for(auto d:grid.ds) {
                if(budget.expired() || stop_requests){ranking_complete=false;break;}
                const auto p=ecm_stage2::cost::phi(d)/2,points=b/d+2;
                if(points<=p || (o.tune_max_batches && (points+p-1)/p>o.tune_max_batches))continue;
                auto shape=portable_shape(o,r,b,d,carrier);pt::Work w;
                if(!guide.cached_work(shape,w) || !w.exact_topology)continue;
                double score=reference_ok?pt::relative_work(w,reference_work):double(w.largest_ntt);
                pt::Estimate prediction;
                if(reference_predicted && guide.predict(shape,prediction))score=prediction.rank/reference_estimate.rank;
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
            // They schedule calibration; production still requires paired wins.
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
                        if(!sample || !pt::LocalSearch::improves({et::real(*sample,"median_seconds"),et::real(*sample,"mad_seconds")},best))continue;
                        if(!admit(nodes[i].d,shape))continue;
                        const auto actual=observed.observed(shape);
                        if(actual && pt::LocalSearch::improves({et::real(*actual,"median_seconds"),et::real(*actual,"mad_seconds")},best)) {
                            best={et::real(*actual,"median_seconds"),et::real(*actual,"mad_seconds")};center=i;
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
                pt::LocalSearch search(center,nodes.size());
                std::cout<<"ecm_tune_window: N="<<bits<<" B2="<<b<<" carrier="<<carrier
                    <<" region="<<(region+1)<<'/'<<requested<<" center_D="<<nodes[center].d
                    <<" fold_ntt="<<basin.key.key[0]<<" tree_ntt="<<basin.key.key[2]
                    <<" predicted_ratio="<<basin.rank<<" restart="<<(region!=0)<<std::endl;
                for(;;) {
                    const auto i=search.next();if(i==nodes.size())break;
                    if(budget.expired() || stop_requests){search.deferred.insert(i);outcomes[nodes[i].d]=Outcome{};break;}
                    const auto result=measure(nodes[i].d);
                    if(result.state==Outcome::Tested)search.tested[i]=result.time;
                    else if(result.state==Outcome::Rejected)search.rejected.insert(i);
                    else search.deferred.insert(i);
                }
                complete=search.complete() && complete;++explored;
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
    // fastest values; their source and validation state remain explicit.
    pt::Estimator estimator(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);
    std::ostringstream evidence_key;
    for(const auto &s:profile.samples)if(pt::measured(s))evidence_key<<pt::scope(s)<<et::required(s,"median_seconds")<<';';
    for(const auto &s:profile.ntt)evidence_key<<et::required(s,"condition")<<':'<<et::required(s,"length")<<':'
        <<et::required(s,"batch")<<':'<<et::required(s,"median_seconds")<<';';
    const auto model_evidence=std::to_string(fingerprint(evidence_key.str()));
    std::vector<et::Fields> estimates;
    for(const auto &c:cases) {
        Record r;
        if(inputs.count(c.bits))r=inputs.at(c.bits).record;
        else {Big n;mpz_set_ui(n.z,1);mpz_mul_2exp(n.z,n.z,c.bits-1);mpz_add_ui(n.z,n.z,12345);r.n=number(n.z);r.b1=20;r.sigma=26;}
        auto shape=portable_shape(o,r,c.b2,c.d,c.carrier);pt::Estimate e;
        if(!estimator.predict(shape,e))continue;
        et::Fields s={{"condition",std::to_string(condition)},{"target_bits",std::to_string(shape.target_bits)},
            {"arithmetic_bits",std::to_string(shape.arithmetic_bits)},{"carrier_exponent",std::to_string(shape.carrier)},
            {"modulus_kind",json_string(shape.mersenne?"mersenne":"generic")},{"b1",std::to_string(r.b1)},{"b2",std::to_string(c.b2)},
            {"d",std::to_string(c.d)},{"p",std::to_string(ecm_stage2::cost::phi(c.d)/2)},{"fold_resident",shape.fold?"1":"0"},
            {"frontier_resident",shape.frontier?"1":"0"},{"source","\"model\""},{"repeats","0"},
            {"execution_path",json_string(shape.fold?(shape.frontier?"resident":"host_frontier"):"host_fold_frontier")},
            {"median_seconds",pt::numeric(e.seconds)},{"mad_seconds",pt::numeric(e.mad)},
            {"model_anchor_count",std::to_string(e.anchors)},{"model_evidence",json_string(model_evidence)},
            {"model_relative_error_allowance",pt::numeric(e.relative_error)}};
        estimates.push_back(std::move(s));
    }
    for(auto &s:estimates)profile.update(std::move(s));
    profile.metadata["effort_level"]=std::to_string(level);profile.metadata["budget_seconds"]=std::to_string(o.tune_budget);
    profile.metadata["error_limit"]=pt::numeric(o.tune_error);profile.metadata["grid_candidates"]=std::to_string(full_grid);
    profile.metadata["selected_candidates"]=std::to_string(cases.size());
    profile.metadata["selection"]="\"ntt_interval_descent_v1\"";
    profile.metadata["catalogue_candidates"]=std::to_string(ecm_stage2::d_catalog::count);
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("ECM calibration binary changed; evidence retained");
    portable_publish(destination,profile);
    std::cout<<"ecm_tune_complete: measured="<<measured<<" skipped="<<skipped<<" grid="<<full_grid<<" selected="<<cases.size()<<" summaries="<<profile.samples.size()
        <<" budget_elapsed="<<budget.elapsed()<<" s profile="<<destination.string()<<" evidence="<<evidence.string()<<std::endl;
}
