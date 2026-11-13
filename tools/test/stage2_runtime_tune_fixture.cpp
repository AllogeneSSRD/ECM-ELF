#include "../../src/ecm/stage2/planning/runtime_tune.h"
#include <iostream>
namespace p=ecm_stage2::tune::portable;
using ecm_stage2::tune::Fields;
static void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
static Fields sample(unsigned id,unsigned d,double seconds) {
    return {{"condition","0"},{"target_bits","4916"},{"arithmetic_bits","4916"},
        {"carrier_exponent","0"},{"modulus_kind","\"generic\""},{"b1","260000000"},
        {"b2","1300000000000"},{"d",std::to_string(d)},
        {"p",std::to_string(ecm_stage2::cost::phi(d)/2)},
        {"fold_resident","1"},{"frontier_resident","1"},{"execution_path","\"resident\""},
        {"source","\"measured\""},{"benchmark_kind","\"runtime_save\""},{"repeats","1"},
        {"clean","1"},{"bad","0"},{"hits","0"},{"selftest_cases","1"},{"checked","96"},
        {"mean_seconds",p::numeric(seconds)},{"min_seconds",p::numeric(seconds)},
        {"max_seconds",p::numeric(seconds)},{"stddev_seconds","0"},
        {"giant_chunk_points","253440"},{"giant_chain_min","32768"},{"giant_force_ladder","0"},
        {"program_version",p::quoted(ecm_program::version)},{"updated_at",p::quoted(p::utc_updated_at(1700000000000000000ull+id))},
        {"required_free_bytes",std::to_string(1000000000ull+id)}};
}
int main(int argc,char **argv) {
    try {
        p::Profile profile;profile.condition({{"uuid_hex","\"fixture-device\""},{"sm_major","8"},{"sm_minor","9"},
            {"cuda_runtime","13030"},{"cuda_driver","13030"},{"gl_fixed_mode","3"},
            {"outer_unroll_u","0"},{"add_sub_mask","1"}});
        require(profile.update(sample(1,690690,52.8733957)),"first sample missing");
        require(profile.update(sample(2,690690,53.1)),"slower independent observation was discarded");
        require(profile.samples.size()==1,"identical parameters did not merge");
        require(!profile.update(sample(2,690690,53.1),10,false),"imported shorter snapshot counted twice");
        for(unsigned i=3;i<=10;++i)require(profile.update(sample(i,690690,52.9+i*.01)),"cap reached too early");
        require(!profile.update(sample(11,690690,1)),"cap did not retain first ten independent samples");
        require(profile.update(sample(21,1381380,45.1830623)),"candidate first observation missing");
        require(profile.update(sample(22,1381380,45.3)),"candidate second observation missing");
        auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
            *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+23)/24))*n*=2;return true;};
        p::Estimator model(profile,profile.conditions.at(0),p::Ignore::parse(""),query);
        auto old=model.shape(profile.samples[0]),fast=model.shape(profile.samples[1]);
        auto protected_point=old;protected_point.giant_chunk=4*ecm_stage2::d_catalog::degree(old.d);
        p::Work default_work,new_work;ecm_stage2::GiantWork legacy_work;ecm_stage2::GiantPointWork point_work;
        require(model.cached_work(protected_point,default_work) &&
            ecm_stage2::giant_work(protected_point.b2/protected_point.d+2,protected_point.d,protected_point.giant_chunk,
                protected_point.chain_min,false,legacy_work),"default point work unavailable");
        const double point_words=(protected_point.arithmetic_bits+63)/64.;
        require(default_work.giant==point_words*point_words*(6.*legacy_work.chain_points+12.*legacy_work.ladder_steps),
            "default D point-cost prior changed without matrix qualification");
        protected_point.chain_block=32;
        require(model.cached_work(protected_point,new_work) &&
            ecm_stage2::giant_point_work(protected_point.b2/protected_point.d+2,protected_point.d,protected_point.giant_chunk,
                protected_point.chain_min,false,32,ecm_stage2::d_catalog::degree(old.d),false,point_work) &&
            new_work.giant==point_words*point_words*point_work.modmuls(),"new policy omitted seed cost");
        auto before=model.runtime_statistics(old),after=model.runtime_statistics(fast);
        require(before.count==10 && after.count==2 && before.stddev>0 && after.stddev>0,"actual dispersion not accumulated");
        p::WidthPredictor predictor(model);p::WidthEstimate estimate;
        p::Estimator uncached(profile,profile.conditions.at(0),p::Ignore::parse(""),query,false);
        p::WidthPredictor uncached_predictor(uncached);
        // Replay the real ranking pattern: different D share NTT shapes, while
        // each predicted/admitted shape is requested repeatedly. Memoization
        // must return the original doubles and retain B1 lookup semantics.
        for(const auto &entry:ecm_stage2::d_catalog::entries())if(entry.p>=512) {
            auto shape=old;shape.d=entry.d;
            p::Work a,b;require(model.cached_work(shape,a)==uncached.cached_work(shape,b),"cached work validity changed");
            require(a.ntt==b.ntt && a.reduce==b.reduce && a.calls==b.calls,"cached NTT model changed work");
            p::WidthEstimate left,right;
            require(predictor.predict(shape,left)==uncached_predictor.predict(shape,right),"cached prediction validity changed");
            require(left.seconds==right.seconds && left.rank==right.rank && left.stddev==right.stddev &&
                left.relative_error==right.relative_error && std::string(left.model)==right.model,"cached prediction changed ranking");
            require(predictor.predict(shape,left)==uncached_predictor.predict(shape,right) && left.rank==right.rank,"repeated prediction changed ranking");
        }
        require(model.ntt_cost_hits>model.ntt_cost_evaluations && model.ntt_cost_evaluations<uncached.ntt_cost_evaluations,
            "D ranking still resolves every repeated NTT cost");
        require(predictor.prediction_hits>0 && predictor.prediction_evaluations<uncached_predictor.prediction_evaluations,
            "D ranking still repeats width prediction");
        auto changed_b1=old;changed_b1.b1*=2;
        require(model.runtime_statistics(changed_b1).count==before.count && model.runtime_statistics(changed_b1,true).count==0,
            "statistics cache confused complete B1 identity with runtime lookup");
        require(predictor.predict(fast,estimate) && estimate.exact && estimate.runtime_samples==2 &&
            std::string(estimate.model)=="runtime_shape_mean_v2","runtime samples still treated as uncertain extrapolation");
        std::vector<p::RuntimeCandidate> routes{{old,before,1},{fast,after,2}};
        auto chosen=p::runtime_choice(routes);
        require(chosen.available && !chosen.probe && chosen.index==1,"4916-bit faster observed D rejected");
        routes[1].statistics.preferred_order=100;routes[1].statistics.seconds=50;routes[1].statistics.stddev=3;
        chosen=p::runtime_choice(routes);
        require(chosen.index==1 && !chosen.probe,"overlapping bands reset established winner to default");
        routes[1].statistics.seconds=65;routes[1].statistics.stddev=.01;
        require(p::runtime_choice(routes).index==0,"real measured regression cannot switch away from incumbent");
        p::RuntimeCandidate carry;carry.shape=fast;carry.shape.carrier=5351;carry.shape.arithmetic_bits=5351;carry.priority=0;
        routes={routes[0],carry,routes[1]};
        require(p::runtime_choice(routes).index==1 && p::runtime_choice(routes).probe,"legal carrier was not first trial");
        require(p::runtime_choice(routes,false).index!=1 && !p::runtime_choice(routes,false).probe,
            "read-only mode scheduled an unmeasured trial");
        routes[1].statistics.count=2;routes[1].statistics.seconds=37;routes[1].statistics.stddev=.1;
        require(p::runtime_choice(routes).index==1 && !p::runtime_choice(routes).probe,"measured carrier not adopted");
        require(p::runtime_choice(routes,false).index==1,"disabling writes discarded an established measurement");
        auto isolated=fast;++isolated.giant_chunk;require(model.runtime_statistics(isolated).count==0,"different giant shape mixed");
        isolated=fast;++isolated.b1;require(model.runtime_statistics(isolated).count==2,"B1 blocked timing reuse");
        auto historical=profile;historical.samples[1]["program_version"]="\"0.0.1\"";
        p::Estimator reused(historical,historical.conditions.at(0),p::Ignore::parse(""),query);
        require(reused.runtime_statistics(fast).count==2,"older program version blocked statistics reuse");
        auto foreign_condition=model.condition;foreign_condition["uuid_hex"]="\"another-device\"";
        p::Estimator foreign_model(profile,foreign_condition,p::Ignore::parse(""),query);
        require(foreign_model.runtime_statistics(fast).count==0,"different device treated as exact");
        p::Profile bounded;bounded.condition(profile.conditions.at(0));
        auto legacy=sample(1,690690,52.8);legacy.erase("required_free_bytes");legacy["measurement_ids"]="\"8:legacy-1\"";
        require(bounded.update(legacy,2) && !bounded.update(legacy,2,false) && bounded.update(sample(30,690690,53),2) &&
            !bounded.update(sample(31,690690,50),2),"legacy extension or configured cap failed");
        auto formal=sample(90,690690,50);formal["benchmark_kind"]="\"formal_fixture\"";
        require(bounded.update(formal),"formal sample mixed with runtime");formal["mean_seconds"]="60";
        require(bounded.update(formal),"independent formal summary was discarded");
        const auto path=std::filesystem::path(argv[1])/"roundtrip.toml";
        {std::ofstream out(path);out<<profile.serialize();}
        const auto loaded=p::Profile::load(path);require(loaded.samples.size()==2,"bounded samples lost on TOML roundtrip");
        require(loaded.serialize()==profile.serialize(),"runtime serialization is not stable");
        require(p::optional_uint(loaded.samples[0],"required_free_bytes")==1000000010ull &&
            bounded.serialize().find("measurement_ids")==std::string::npos,"memory maximum or legacy column removal failed");
        require(ecm_stage2::tune::uint(profile.samples[0],"repeats")==10 && ecm_stage2::tune::uint(profile.samples[1],"repeats")==2,"merged repeat count wrong");
        require(profile.serialize().find(".table.")==std::string::npos && profile.serialize().find("selftest_cases")==std::string::npos,"redundant serialized fields remain");
        long double sum=52.8733957L+53.1L;std::vector<long double> values{52.8733957L,53.1L};
        for(unsigned i=3;i<=10;++i){values.push_back(52.9L+i*.01L);sum+=values.back();}
        const long double mean=sum/values.size();long double variance=0;
        for(auto v:values)variance+=(v-mean)*(v-mean);
        require(std::abs(before.seconds-double(mean))<1e-12 && std::abs(before.stddev-std::sqrt(double(variance/9)))<1e-12,
            "aggregate mean/variance differs from independent two-pass reference");
        p::Profile partial;partial.condition(profile.conditions.at(0));
        partial.update(sample(1,690690,52.8733957));partial.update(sample(2,690690,53.1));
        require(partial.update(profile.samples[0],10,false) && !partial.update(profile.samples[0],10,false),
            "larger snapshot should replace once");
        p::Profile left,right;left.condition(profile.conditions.at(0));right.condition(profile.conditions.at(0));
        left.update(sample(1,690690,10));left.update(sample(2,690690,11));
        right.update(sample(2,690690,11));right.update(sample(3,690690,12));
        require(left.update(right.samples.front(),10,false) &&
            p::optional_uint(left.samples.front(),"repeats")==2 && left.samples.front()==right.samples.front(),
            "overlapping snapshots must select the newer summary without combining");
        auto next_b1=sample(40,1381380,46);next_b1["b1"]="22000000";
        require(profile.update(next_b1),"different B1 parameter row was discarded");
        require(profile.samples.size()==3,"different complete parameters merged into one row");
        p::Estimator pooled(profile,profile.conditions.at(0),p::Ignore::parse(""),query);
        require(pooled.runtime_statistics(fast).count==3 && pooled.runtime_statistics(fast,true).count==2,"B1 storage/reuse populations mixed");
        auto other_b1=sample(41,690690,54);other_b1["b1"]="22000000";
        require(profile.update(other_b1),"full row cap discarded a different complete parameter row");
        auto current=profile.conditions.at(0),different=current;different["program_version"]="\"0.0.1\"";
        require(p::matches(current,different,p::Ignore::parse("")),"provenance version became a measurement condition");
        require(profile.condition(different)==0,"version-only difference created a new condition");
        p::Profile field;field.condition(current);
        auto ntt=[](double time,unsigned tile) {
            Fields f={{"coop_outer","false"},{"warp_tail","true"},{"compact_scratch","true"},{"condition","0"},{"length","8"},{"log2_length","3"},{"batch","1"},{"tile",std::to_string(tile)},
                {"bad","0"},{"source","\"measured\""},{"status","\"measured\""},{"unit","\"field_convolution\""},
                {"reference_kind","\"gmp_3x3_distinct_constant_v1\""},{"verified_words_per_sample","8"},{"repeats","1"},
                {"mean_seconds",p::numeric(time)},{"stddev_seconds","0"},{"conv_iter_per_s",p::numeric(1/time)},
                {"batch_iter_per_s",p::numeric(1/time)}};p::stamp_provenance(f,1700000000000000000ull+tile);return f;
        };
        require(field.update_ntt(ntt(1,3)) && field.update_ntt(ntt(3,3)),"NTT repeat accumulation failed");
        require(field.ntt.size()==1 && ecm_stage2::tune::uint(field.ntt.front(),"repeats")==2 &&
            std::abs(ecm_stage2::tune::real(field.ntt.front(),"mean_seconds")-2)<1e-12,"NTT mean is not weighted by repeats");
        require(field.update_ntt(ntt(2,4)) && field.ntt.size()==2,"different NTT parameters were discarded");
        {std::ofstream out(path);out<<field.serialize();}
        require(p::Profile::load(path).serialize()==field.serialize(),"NTT matrix roundtrip changed statistics");
        auto foreign=field.serialize();const auto at=foreign.find(ecm_program::version);foreign.replace(at,std::string(ecm_program::version).size(),"0.0.1");
        {std::ofstream out(path);out<<foreign;}
        const auto older=p::Profile::load(path);require(older.ntt.size()==2,"older version NTT rows lost");
        require(p::text(older.ntt.front(),"program_version")=="0.0.1","source version was rewritten to current version");
        require(!older.conditions.at(0).count("program_version"),"version still stored in condition");
        p::Estimator table(field,current,p::Ignore::parse(""),query),plain_table(field,current,p::Ignore::parse(""),query,false);
        for(unsigned repeat=0;repeat<3;++repeat)for(unsigned power=3;power<=26;++power)for(unsigned batch:{1u,2u,3u,8u})
            require(table.ntt_cost(1ull<<power,batch)==plain_table.ntt_cost(1ull<<power,batch),"measured/interpolated NTT cache changed cost");
        require(table.ntt_cost_evaluations==24*4 && table.ntt_cost_hits==2*24*4,"NTT cost reuse not keyed by length/batch");
        // A layout is part of arithmetic topology, not a new timing population
        // for the same shape. Historical rows retain their legacy identity.
        auto legacy_narrow=sample(201,1141140,70);legacy_narrow["arithmetic_bits"]="5987";
        legacy_narrow["carrier_exponent"]="5987";legacy_narrow["modulus_kind"]="\"mersenne\"";
        auto blocked=legacy_narrow;blocked["groot_balanced"]="1";blocked["mean_seconds"]="55";
        auto legacy_wide=legacy_narrow;legacy_wide["arithmetic_bits"]="9000";
        require(!p::sample_groot_layout(legacy_narrow) && p::sample_groot_layout(legacy_wide),"historical layout was reinterpreted");
        p::Profile layouts;layouts.condition(current);layouts.update(legacy_narrow);layouts.update(blocked);
        p::Estimator lm(layouts,current,p::Ignore::parse(""),query);
        const auto dense=lm.shape(layouts.samples[0]),tiled=lm.shape(layouts.samples[1]);
        require(layouts.samples.size()==2 && dense.key()!=tiled.key() && lm.runtime_statistics(dense).seconds==70 &&
            lm.runtime_statistics(tiled).seconds==55,"different layouts pooled their timings");
        require(p::runtime_group(layouts.samples[0])!=p::runtime_group(layouts.samples[1]),"layout absent from runtime identity");
        {std::ofstream out(path);out<<layouts.serialize();}
        require(p::Profile::load(path).serialize()==layouts.serialize(),"layout identity lost in matrix roundtrip");
        auto compatible=layouts.serialize();
        const auto current_format="format = "+std::to_string(p::format);
        compatible.replace(compatible.find(current_format),current_format.size(),"format = 8");
        {std::ofstream out(path);out<<compatible;}
        require(p::Profile::load(path).serialize()==layouts.serialize(),"format-8 migration changed measurement identity");
        auto forced=sample(202,1141140,1);forced["groot_balanced"]="2";
        bool invalid=false;try{layouts.update(forced);}catch(const std::exception&){invalid=true;}
        require(invalid,"invalid layout accepted");
        p::Profile points;points.condition(current);
        auto original=sample(211,1141140,70),balanced=sample(212,1141140,60),shorter=sample(213,1141140,50);
        balanced["giant_tail_balance"]="1";shorter["giant_chain_block"]="32";
        points.update(original);points.update(balanced);points.update(shorter);
        require(points.samples.size()==3,"point policies pooled their measurements");
        p::Estimator pm(points,current,p::Ignore::parse("environment"),query);
        const auto legacy_point=pm.shape(points.samples[0]),balanced_point=pm.shape(points.samples[1]),short_point=pm.shape(points.samples[2]);
        auto noop=legacy_point;noop.d=207480;noop.b2=uint64_t(130213-2)*noop.d;
        noop.giant_chunk=145152;noop.chain_min=32768;noop.force_ladder=false;noop.chain_block=64;noop.tail_balance=false;
        auto permitted=noop;permitted.tail_balance=true;permitted.canonicalize_point_policy();
        require(!permitted.tail_balance && permitted.key()==noop.key(),"no-op borrowing split the complete shape identity");
        require(p::runtime_point_route_cacheable(permitted.tail_balance,64),"no-op borrowing disabled route hints");
        permitted.d=180180;permitted.b2=uint64_t(149940-2)*permitted.d;permitted.giant_chunk=138240;permitted.tail_balance=true;
        permitted.canonicalize_point_policy();
        require(permitted.tail_balance && !p::runtime_point_route_cacheable(permitted.tail_balance,64),"active borrowing reused an incomplete route hint");
        require(!p::runtime_point_route_cacheable(false,32),"short chains reused a legacy route hint");
        permitted.force_ladder=true;permitted.canonicalize_point_policy();
        require(!permitted.tail_balance,"forced ladder kept a fictitious balanced schedule");
        require(pm.runtime_statistics(legacy_point).seconds==70 && pm.runtime_statistics(balanced_point).seconds==60 &&
            pm.runtime_statistics(short_point).seconds==50,"point policy absent from exact lookup");
        require(!pm.eligible(points.samples[1],legacy_point) && !pm.eligible(points.samples[2],legacy_point),"ignore environment crossed point policy");
        auto request0=current,request1=current;
        request0["env_giant_tail_balance"]="0";request1["env_giant_tail_balance"]="1";
        require(p::matches(request0,request1,p::Ignore::parse("")) && p::matches(request0,current,p::Ignore::parse("")),
            "redundant request environment split the actual point-policy identity");
        p::Profile old_requests;old_requests.condition(request0);old_requests.update(original);old_requests.update(balanced);
        p::Estimator request_model(old_requests,request1,p::Ignore::parse(""),query);
        require(request_model.runtime_statistics(legacy_point).seconds==70 &&
            request_model.runtime_statistics(balanced_point).seconds==60 &&
            !request_model.eligible(old_requests.samples[1],legacy_point),
            "old request metadata blocked no-op reuse or pooled a different actual schedule");
        request1["env_giant_chain_min"]="16384";
        require(!p::matches(request0,request1,p::Ignore::parse("")),"unrelated environment constraints were relaxed");
        {std::ofstream out(path);out<<points.serialize();}
        require(p::Profile::load(path).serialize()==points.serialize(),"point policy roundtrip changed");
        std::vector<p::RuntimeCandidate> chains{{legacy_point,{2,0,70,0},1,true},
            {balanced_point,{2,0,60,0},2,true}};
        for(auto &c:chains)c.shape.giant_chunk=2*ecm_stage2::d_catalog::degree(c.shape.d);
        auto trials=p::runtime_chain_trials(chains);
        require(trials.size()==1 && trials[0].chain_block==32 && trials[0].tail_balance,"chain search expanded all D/layouts");
        chains.push_back({trials[0],{1,0,50,0},2,true});
        require(p::runtime_chain_trials(chains).empty(),"partial chain pair was reproposed");
        require(p::runtime_choice(chains).index==2 && p::runtime_choice(chains).probe,"partial chain pair not completed");
        chains.back().statistics.count=2;
        require(p::runtime_choice(chains,false).index==2 && !p::runtime_choice(chains,false).probe,"faster repeated chain not selected");
        chains.back().statistics.seconds=100;
        require(p::runtime_choice(chains,false).index==1,"slower chain displaced protected incumbent");
        auto single=chains;for(auto &c:single)c.statistics.count=1;
        require(!p::runtime_choice(single,false).available,"default minimum accepted one observation");
        require(p::runtime_choice(single,false,1).index==1,"minimum one did not adopt fastest observation");
        require(p::runtime_choice(single,true,3).probe,"higher minimum did not request more evidence");
        single.resize(1);
        require(p::runtime_chain_trials(single,1).size()==1 && p::runtime_chain_trials(single,2).empty(),"chain reference minimum ignored");
        p::RuntimeSearchBudget one_budget;one_budget.reference=10;one_budget.minimum=1;
        require(one_budget.permits(11,0,2) && !one_budget.permits(11,0,1),"queue budget ignored configured repeats");
        p::Profile capped;capped.condition(current);
        require(capped.update(sample(221,1141140,12),1),"one observation cap rejected first evidence");
        require(!capped.update(sample(222,1141140,1),1),"one observation cap discarded its boundary");
        p::Estimator one_model(capped,current,p::Ignore::parse(""),query,true,1);
        p::WidthPredictor one_predictor(one_model);p::WidthEstimate one_estimate;
        require(one_predictor.predict(one_model.shape(capped.samples[0]),one_estimate) && one_estimate.runtime_samples==1,
            "exact runtime prediction ignored minimum one");
        require(std::string(one_estimate.uncertainty)=="single_runtime_observation","one sample claimed measured dispersion");
        p::Profile formal_profile;formal_profile.condition(current);auto formal_one=sample(223,1141140,12);
        formal_one["benchmark_kind"]="\"save\"";formal_profile.update(formal_one);
        p::Estimator formal_min1(formal_profile,current,p::Ignore::parse(""),query,true,1);
        p::Estimator formal_min2(formal_profile,current,p::Ignore::parse(""),query,true,2);
        const auto formal_shape=formal_min1.shape(formal_profile.samples[0]);
        require(formal_min1.observed(formal_shape) && !formal_min2.observed(formal_shape),"formal exact lookup bypassed minimum");
        p::Estimate formal_estimate;require(formal_min2.predict(formal_shape,formal_estimate) && !formal_estimate.exact,
            "one formal sample remained exact after raising minimum");
        std::vector<p::DSearchNode> one_nodes{{1141140,103680,{},1,12,0,1},{1381380,124416,{},2,0,0,0}};
        require(p::DSearch(one_nodes,1141140,1).next().index==1 && p::DSearch(one_nodes,1141140,2).next().index==0,
            "D search ignored configured minimum");
        auto cliff=[](ecm_stage2::Word m,int,ecm_stage2::Word *n,ecm_stage2::Word *slots){
            *slots=2*m-1;*n=8;while(*n<m*1024)*n*=2;return true;};
        require(ecm_stage2::select_groot_layout(94,1250000,103680,5987,cliff),"NTT cliff was not selected");
        auto flat=[](ecm_stage2::Word,int,ecm_stage2::Word *n,ecm_stage2::Word *slots){*n=1ull<<26;*slots=1;return true;};
        require(!ecm_stage2::select_groot_layout(70,1250000,103680,4423,flat),"layout enabled without an NTT cliff");
        require(!ecm_stage2::select_groot_layout(94,300000,103680,5987,cliff),"short curve enabled extra layout setup");
        {ecm_stage2::GrootLayoutScope scope(0);require(!ecm_stage2::select_groot_layout(94,1250000,103680,5987,cliff),"layout scope ignored");}
        std::vector<p::RuntimeCandidate> families(2);families[0].shape=dense;families[0].shape.carrier=0;
        families[1].shape=tiled;families[0].statistics={};families[1].statistics={};
        families[0].statistics.count=families[1].statistics.count=2;
        families[0].statistics.seconds=100;families[1].statistics.seconds=75;
        families[1].statistics.seconds=80;
        require(p::runtime_allocation(families).ordinary_reduced,"exact 20 percent boundary lost to cancellation");
        families[1].statistics.seconds=75;
        require(p::runtime_allocation(families).ordinary_reduced,"clear carrier advantage did not reduce exploration");
        families[1].statistics.count=1;require(!p::runtime_allocation(families).ordinary_reduced,"single observation throttled exploration");
        families[1].statistics.count=2;families[1].statistics.stddev=4;
        require(!p::runtime_allocation(families).ordinary_reduced,"uncertain advantage throttled exploration");
        families[1].statistics.stddev=0;families[1].statistics.seconds=85;
        require(!p::runtime_allocation(families).ordinary_reduced,"shrinking advantage did not restore exploration");
        families[0].statistics.seconds=70;
        require(p::runtime_choice(families,false).index==0,"throttling discarded a faster ordinary incumbent");
        families[0].statistics.seconds=100;families[1].statistics.seconds=75;
        auto extra=families[0];extra.shape.groot_balanced=!extra.shape.groot_balanced;
        extra.statistics.seconds=110;families.push_back(extra);
        const auto allocation=p::runtime_allocation(families);
        const auto shared=p::runtime_family_budget(families,families[0].shape,10,allocation);
        require(shared.candidates==1 && shared.spent==20 && shared.remaining==100,
            "alternate layout received a duplicate candidate/time budget");
        families[0].statistics.count=0;families[0].priority=2;
        families[1].statistics.count=0;families[1].priority=2;families.pop_back();
        require(p::runtime_choice(families).index==1,"equal-priority layout probe did not prefer carrier");
        families[0].statistics.count=1;
        require(p::runtime_choice(families).index==0,"partial ordinary pair was abandoned");
        std::cout<<"{\"layout_identity\":true,\"ntt_cliff_selection\":true,\"carrier_allocation\":true,\"complete\":true,\"retains_slower_samples\":true,\"cap\":10,\"independent_samples\":12,"
            "\"observed_D_win\":true,\"fastest_mean\":true,\"carrier_first\":true,\"deduplication\":true,"
            "\"cross_version_reuse\":true,\"matrix_roundtrip\":true,\"mean_variance_reference\":true,"
            "\"snapshot_replacement\":true,\"memory_requirement_roundtrip\":true,\"ntt_parameter_rows\":true,"
            "\"model_memoization\":true,\"ntt_cost_evaluations\":"<<model.ntt_cost_evaluations<<",\"ntt_cost_cache_hits\":"<<model.ntt_cost_hits<<",\"CUDA_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
