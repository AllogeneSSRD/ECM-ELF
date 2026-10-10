// Included inside s2prod after the save/worker/planning helpers.
namespace pt=ecm_stage2::tune::portable;
namespace et=ecm_stage2::tune;
std::string portable_identity(const fs::path &evidence) {
    const auto hash=ecm_stage2::sha256_file(executable());
    append(evidence/"identity.jsonl","{\"binary_sha256\":"+json_string(hash)+",\"binary\":"+json_string(executable().string())+"}");
    const auto manifest=executable().parent_path()/"build_manifest.json";
    if(fs::is_regular_file(manifest))fs::copy_file(manifest,evidence/"build_manifest.json",fs::copy_options::none);
    return hash;
}
pt::Profile portable_read(const fs::path &path) {
    return fs::is_regular_file(path)?pt::Profile::load(path):pt::Profile{};
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
    return ecm_cuda_stage2_shape_query(m,bits,n,slots);
}
pt::Shape portable_shape(const Options &o,const Record &r,uint64_t b2,uint64_t d,unsigned carrier) {
    Big n,plus;mpz_set_str(n.z,r.n.c_str(),16);mpz_add_ui(plus.z,n.z,1);
    pt::Shape s;s.target_bits=unsigned(mpz_sizeinbase(n.z,2));s.arithmetic_bits=carrier?carrier:s.target_bits;
    s.carrier=carrier;s.b1=r.b1;s.b2=b2;s.d=d;s.mersenne=carrier || mpz_popcount(plus.z)==1;
    const auto packing=et::ntt_packing_policy(tune_environment());
    s.batch_bytes=et::uint(packing,"batch_bytes");s.buffers=unsigned(et::uint(packing,"buffers"));
    s.physical=et::uint(packing,"physical_chunks")!=0;s.chunk_max=et::uint(packing,"chunk_max");
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
    if(ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,s.b2,s.d,o.device,
        [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&plan,s.carrier))
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
    std::vector<uint64_t> ds=pt::grid(10).ds;if(o.has_d && o.d)ds={o.d};
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
    for(auto b:b2s)for(auto d:ds)for(auto carrier:carriers) {
        auto shape=portable_shape(o,r,b,d,carrier);pt::Estimate e;if(!estimator.predict(shape,e))continue;
        const double probability=automatic?ecm_stage2::probability::success(double(r.b1),double(b),factor):0;
        const double score=automatic?probability/(t1+o.ratio_adjust*e.rank):-e.rank;
        candidates.push_back({shape,e,score,probability});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate &a,const Candidate &b){return a.score>b.score;});
    size_t rejected=0;
    for(auto &candidate:candidates) {
        auto shape=candidate.shape;std::string plan;if(!portable_admit(o,r,shape,plan)){++rejected;continue;}
        pt::Estimate e;if(!estimator.predict(shape,e)){++rejected;continue;}
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
    return "{\"type\":\"tune_selection\",\"selected\":false,\"reason\":\"no_costed_candidate_fits_current_memory\"}";
}
std::string select_tuned(Options &o,const Record &r){return portable_select(o,r,false);}
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
    unsigned repeats,pt::Budget &budget,const fs::path &evidence,pt::Profile &profile,ecm_stage2::Word condition) {
    if(budget.expired() || stop_requests)return false;
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
        if(child_run(worker,input.save,input.record,b2,d,o.device,result,log,o.batch,settings))
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
    const auto path=portable_shape(o,r,r.b1+42000,210,0);
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
        // Prepare a fresh, valid B1=20 point on the already validated target.
        // This also permits a known cofactor to exercise its legal carrier.
        const auto save=evidence/"target.save";
        const auto x=et::benchmark_point(r.n);
        append(save,"METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x"+r.n+"; X=0x"+x+"; Z=1;");
        PortableInput input{save,records(save,0,1).at(0),"validated_target_benchmark"};
        if(!ordinary || !width_covered)portable_measure(o,settings,input,42000,210,0,2,budget,evidence,profile,condition);
        if(!carrier_covered && !budget.expired())portable_measure(o,settings,input,42000,210,needed_carrier,2,budget,evidence,profile,condition);
    }
    portable_ntt(o,profile,condition,budget,evidence,{{16,1},{18,1},{20,1}},2);
    profile.metadata["short_calibration_budget_seconds"]=std::to_string(o.short_budget);
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("short calibration binary changed; evidence retained");
    portable_publish(o.tune_profile,profile);
    std::cout<<"stage2_short_calibration_done: samples="<<profile.samples.size()<<" ntt="<<profile.ntt.size()
        <<" elapsed="<<budget.elapsed()<<" s profile="<<o.tune_profile<<std::endl;
}

void run_portable_tune(Options o,const Settings &settings,const fs::path &destination) {
    if(!o.tune_save.empty() && !o.tune_exponents.empty())throw std::runtime_error("choose tune-save or tune-exponents");
    if(o.tune_carrier && o.tune_save.empty())throw std::runtime_error("carrier tuning requires a validated --tune-save");
    if(o.has_tune_tail_samples && o.tune_tail_samples)throw std::runtime_error("portable tune uses the default grid; legacy tune-tail-samples accepts only 0");
    auto guard=portable_lock(destination);auto profile=portable_read(destination);const auto current=portable_condition(o);
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
    struct Case {unsigned bits;uint64_t b2,d;unsigned carrier;double priority;};std::vector<Case> cases;
    for(auto bits:grid.widths)for(auto b:grid.b2s)for(auto d:grid.ds)for(unsigned mode=0;mode<(o.tune_carrier?2u:1u);++mode) {
        const unsigned carrier=mode?o.tune_carrier:0;double priority=10;
        for(const auto &s:profile.samples)if(et::uint(s,"condition")==condition && et::uint(s,"target_bits")==bits &&
            et::uint(s,"b2")==b && et::uint(s,"d")==d && et::uint(s,"carrier_exponent")==carrier && pt::measured(s)) {
            priority=pt::optional_uint(s,"validation_count")?pt::optional_real(s,"validation_max_relative_error")/o.tune_error:1;
            if(et::uint(s,"repeats")<grid.repeats)priority+=2;
        }
        // First cover widths and coarse D/NTT transitions; later tune runs spend
        // budget on independent validation/error regions, not completed cells.
        priority+=double(bits==grid.widths.front())*.1;
        cases.push_back({bits,b,d,carrier,priority});
    }
    std::stable_sort(cases.begin(),cases.end(),[](const Case &a,const Case &b){return a.priority>b.priority;});
    size_t measured=0,skipped=0;
    // Seed a cheap full-engine anchor per width before attempting long cells.
    for(auto bits:grid.widths) {
        if(budget.expired() || stop_requests)break;
        bool exists=false;for(const auto &s:profile.samples)if(pt::measured(s) && et::uint(s,"condition")==condition && et::uint(s,"target_bits")==bits)exists=true;
        if(exists)continue;
        if(!inputs.count(bits))inputs.emplace(bits,portable_input(bits,evidence));
        const auto &r=inputs.at(bits).record;
        const uint64_t seed_d=r.b1<42000?210:30030;
        const auto seed_b2=std::max(r.b1+1,2*seed_d*(ecm_stage2::cost::phi(seed_d)/2));
        if(portable_measure(o,settings,inputs.at(bits),seed_b2,seed_d,0,grid.repeats,budget,evidence,profile,condition)){++measured;portable_publish(destination,profile);}
    }
    portable_ntt(o,profile,condition,budget,evidence,{{16,1},{18,1},{20,1},{22,1}},grid.repeats);
    for(const auto &c:cases) {
        if(budget.expired() || stop_requests)break;
        if(c.priority<1){++skipped;continue;}
        if(!inputs.count(c.bits))inputs.emplace(c.bits,portable_input(c.bits,evidence));
        const auto &input=inputs.at(c.bits);if(c.b2<=input.record.b1){++skipped;continue;}
        const auto p=ecm_stage2::cost::phi(c.d)/2,g=(c.b2/c.d+2+p-1)/p;
        if(o.tune_max_batches && g>o.tune_max_batches){++skipped;continue;}
        auto shape=portable_shape(o,input.record,c.b2,c.d,c.carrier);
        pt::Estimator estimator(profile,current,pt::Ignore::parse(o.tune_ignore),portable_query);pt::Estimate prediction;
        if(estimator.predict(shape,prediction) && prediction.seconds*(grid.repeats+1)>budget.remaining()){++skipped;continue;}
        if(portable_measure(o,settings,input,c.b2,c.d,c.carrier,grid.repeats,budget,evidence,profile,condition)){++measured;portable_publish(destination,profile);}
        else ++skipped;
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
    profile.metadata["error_limit"]=pt::numeric(o.tune_error);profile.metadata["grid_candidates"]=std::to_string(cases.size());
    if(binary!=ecm_stage2::sha256_file(executable()))throw std::runtime_error("ECM calibration binary changed; evidence retained");
    portable_publish(destination,profile);
    std::cout<<"ecm_tune_complete: measured="<<measured<<" skipped="<<skipped<<" grid="<<cases.size()<<" summaries="<<profile.samples.size()
        <<" budget_elapsed="<<budget.elapsed()<<" s profile="<<destination.string()<<" evidence="<<evidence.string()<<std::endl;
}
