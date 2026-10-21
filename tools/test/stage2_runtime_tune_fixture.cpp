#include "../../src/ecm/service/planning/ecm_stage2_runtime_tune.h"
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
        {"runtime_sample_id",p::quoted("sample-"+std::to_string(id))}};
}
int main(int argc,char **argv) {
    try {
        p::Profile profile;profile.condition({{"uuid_hex","\"fixture-device\""},{"sm_major","8"},{"sm_minor","9"},
            {"cuda_runtime","13030"},{"cuda_driver","13030"},{"gl_fixed_mode","3"},
            {"outer_unroll_u","0"},{"add_sub_mask","1"}});
        require(profile.update(sample(1,690690,52.8733957)),"first sample missing");
        require(profile.update(sample(2,690690,53.1)),"slower independent observation was discarded");
        require(profile.samples.size()==1,"identical parameters did not merge");
        require(!profile.update(sample(2,690690,53.1)),"replayed sample counted twice");
        for(unsigned i=3;i<=10;++i)require(profile.update(sample(i,690690,52.9+i*.01)),"cap reached too early");
        require(!profile.update(sample(11,690690,1)),"cap did not retain first ten independent samples");
        require(profile.update(sample(21,1381380,45.1830623)),"candidate first observation missing");
        require(profile.update(sample(22,1381380,45.3)),"candidate second observation missing");
        auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
            *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+23)/24))*n*=2;return true;};
        p::Estimator model(profile,profile.conditions.at(0),p::Ignore::parse(""),query);
        auto old=model.shape(profile.samples[0]),fast=model.shape(profile.samples[1]);
        auto before=model.runtime_statistics(old),after=model.runtime_statistics(fast);
        require(before.count==10 && after.count==2 && before.stddev>0 && after.stddev>0,"actual dispersion not accumulated");
        p::WidthPredictor predictor(model);p::WidthEstimate estimate;
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
        model.condition["uuid_hex"]="\"another-device\"";
        require(model.runtime_statistics(fast).count==0,"different device treated as exact");
        p::Profile bounded;bounded.condition(profile.conditions.at(0));
        auto legacy=sample(1,690690,52.8);legacy.erase("runtime_sample_id");
        require(bounded.update(legacy,2) && !bounded.update(legacy,2) && bounded.update(sample(30,690690,53),2) &&
            !bounded.update(sample(31,690690,50),2),"legacy extension or configured cap failed");
        auto formal=sample(90,690690,50);formal["benchmark_kind"]="\"formal_fixture\"";
        require(bounded.update(formal),"formal sample mixed with runtime");formal["mean_seconds"]="60";formal["runtime_sample_id"]="\"formal-second\"";
        require(bounded.update(formal),"independent formal summary was discarded");
        const auto path=std::filesystem::path(argv[1])/"roundtrip.toml";
        {std::ofstream out(path);out<<profile.serialize();}
        const auto loaded=p::Profile::load(path);require(loaded.samples.size()==2,"bounded samples lost on TOML roundtrip");
        require(loaded.serialize()==profile.serialize(),"runtime serialization is not stable");
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
        require(partial.update(profile.samples[0]) && !partial.update(profile.samples[0]),"snapshot superset/subset deduplication failed");
        p::Profile left,right;left.condition(profile.conditions.at(0));right.condition(profile.conditions.at(0));
        left.update(sample(1,690690,10));left.update(sample(2,690690,11));
        right.update(sample(2,690690,11));right.update(sample(3,690690,12));
        const auto frozen=left.serialize();bool overlap=false;
        try{left.update(right.samples.front());}catch(const std::runtime_error&){overlap=true;}
        require(overlap && frozen==left.serialize(),"partial overlap was silently double-counted or mutated the profile");
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
        auto ntt=[](const char *id,double time,unsigned tile) {
            Fields f={{"coop_outer","false"},{"warp_tail","true"},{"compact_scratch","true"},{"condition","0"},{"length","8"},{"log2_length","3"},{"batch","1"},{"tile",std::to_string(tile)},
                {"bad","0"},{"source","\"measured\""},{"status","\"measured\""},{"unit","\"field_convolution\""},
                {"reference_kind","\"gmp_3x3_distinct_constant_v1\""},{"verified_words_per_sample","8"},{"repeats","1"},
                {"mean_seconds",p::numeric(time)},{"stddev_seconds","0"},{"conv_iter_per_s",p::numeric(1/time)},
                {"batch_iter_per_s",p::numeric(1/time)},{"measurement_ids",p::encoded_ids({id})}};p::stamp_provenance(f,1700000000000000000ull+tile);return f;
        };
        require(field.update_ntt(ntt("ntt-a",1,3)) && field.update_ntt(ntt("ntt-b",3,3)),"NTT repeat accumulation failed");
        require(field.ntt.size()==1 && ecm_stage2::tune::uint(field.ntt.front(),"repeats")==2 &&
            std::abs(ecm_stage2::tune::real(field.ntt.front(),"mean_seconds")-2)<1e-12,"NTT mean is not weighted by repeats");
        require(field.update_ntt(ntt("ntt-c",2,4)) && field.ntt.size()==2,"different NTT parameters were discarded");
        {std::ofstream out(path);out<<field.serialize();}
        require(p::Profile::load(path).serialize()==field.serialize(),"NTT matrix roundtrip changed statistics");
        auto foreign=field.serialize();const auto at=foreign.find(ecm_program::version);foreign.replace(at,std::string(ecm_program::version).size(),"0.0.1");
        {std::ofstream out(path);out<<foreign;}
        const auto older=p::Profile::load(path);require(older.ntt.size()==2,"older version NTT rows lost");
        require(p::text(older.ntt.front(),"program_version")=="0.0.1","source version was rewritten to current version");
        require(!older.conditions.at(0).count("program_version"),"version still stored in condition");
        std::cout<<"{\"complete\":true,\"retains_slower_samples\":true,\"cap\":10,\"independent_samples\":12,"
            "\"observed_D_win\":true,\"fastest_mean\":true,\"carrier_first\":true,\"deduplication\":true,"
            "\"cross_version_reuse\":true,\"matrix_roundtrip\":true,\"mean_variance_reference\":true,"
            "\"snapshot_deduplication\":true,\"partial_overlap_rejected\":true,\"ntt_parameter_rows\":true,\"CUDA_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
