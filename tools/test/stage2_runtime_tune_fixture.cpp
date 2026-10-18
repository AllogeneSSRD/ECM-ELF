#include "../../src/core/ecm_stage2_runtime_tune.h"
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
        {"median_seconds",p::numeric(seconds)},{"min_seconds",p::numeric(seconds)},
        {"max_seconds",p::numeric(seconds)},{"mad_seconds","0"},
        {"giant_chunk_points","253440"},{"giant_chain_min","32768"},{"giant_force_ladder","0"},
        {"measurement_binary_sha256","\"fixture-build\""},
        {"runtime_sample_id",p::quoted("sample-"+std::to_string(id))}};
}
int main(int argc,char **argv) {
    try {
        p::Profile profile;profile.condition({{"uuid_hex","\"fixture-device\""},{"sm_major","8"},{"sm_minor","9"},
            {"cuda_runtime","13030"},{"cuda_driver","13030"},{"gl_fixed_mode","3"},
            {"outer_unroll_u","0"},{"add_sub_mask","1"}});
        require(profile.update(sample(1,690690,52.8733957)),"first sample missing");
        require(profile.update(sample(2,690690,53.1)),"slower independent observation was discarded");
        require(profile.samples.size()==2,"independent sample count is not two");
        require(!profile.update(sample(2,690690,53.1)),"replayed sample counted twice");
        for(unsigned i=3;i<=10;++i)require(profile.update(sample(i,690690,52.9+i*.01)),"cap reached too early");
        require(!profile.update(sample(11,690690,1)),"cap did not retain first ten independent samples");
        require(profile.update(sample(21,1381380,45.1830623)),"candidate first observation missing");
        require(profile.update(sample(22,1381380,45.3)),"candidate second observation missing");
        auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
            *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+23)/24))*n*=2;return true;};
        p::Estimator model(profile,profile.conditions.at(0),p::Ignore::parse(""),query);model.runtime_binary="fixture-build";
        auto old=model.shape(profile.samples[0]),fast=model.shape(profile.samples[10]);
        auto before=model.runtime_statistics(old),after=model.runtime_statistics(fast);
        require(before.count==10 && after.count==2 && before.mad>0 && after.mad>0,"actual dispersion not accumulated");
        p::WidthPredictor predictor(model);p::WidthEstimate estimate;
        require(predictor.predict(fast,estimate) && estimate.exact && estimate.runtime_samples==2 &&
            std::string(estimate.model)=="runtime_shape_statistics_v1","runtime samples still treated as uncertain extrapolation");
        std::vector<p::RuntimeCandidate> routes{{old,before,1},{fast,after,2}};
        auto chosen=p::runtime_choice(routes);
        require(chosen.available && !chosen.probe && chosen.index==1,"4916-bit faster observed D rejected");
        routes[1].statistics.preferred_order=100;routes[1].statistics.seconds=50;routes[1].statistics.mad=3;
        chosen=p::runtime_choice(routes);
        require(chosen.index==1 && !chosen.probe,"overlapping bands reset established winner to default");
        routes[1].statistics.seconds=65;routes[1].statistics.mad=.01;
        require(p::runtime_choice(routes).index==0,"real measured regression cannot switch away from incumbent");
        p::RuntimeCandidate carry;carry.shape=fast;carry.shape.carrier=5351;carry.shape.arithmetic_bits=5351;carry.priority=0;
        routes={routes[0],carry,routes[1]};
        require(p::runtime_choice(routes).index==1 && p::runtime_choice(routes).probe,"legal carrier was not first trial");
        require(p::runtime_choice(routes,false).index!=1 && !p::runtime_choice(routes,false).probe,
            "read-only mode scheduled an unmeasured trial");
        routes[1].statistics.count=2;routes[1].statistics.seconds=37;routes[1].statistics.mad=.1;
        require(p::runtime_choice(routes).index==1 && !p::runtime_choice(routes).probe,"measured carrier not adopted");
        require(p::runtime_choice(routes,false).index==1,"disabling writes discarded an established measurement");
        auto isolated=fast;++isolated.giant_chunk;require(model.runtime_statistics(isolated).count==0,"different giant shape mixed");
        isolated=fast;++isolated.b1;require(model.runtime_statistics(isolated).count==2,"B1 blocked timing reuse");
        model.runtime_binary="different-build";require(model.runtime_statistics(fast).count==0,"different binary treated as exact");
        model.runtime_binary="fixture-build";model.condition["uuid_hex"]="\"another-device\"";
        require(model.runtime_statistics(fast).count==0,"different device treated as exact");
        p::Profile bounded;bounded.condition(profile.conditions.at(0));
        auto legacy=sample(1,690690,52.8);legacy.erase("runtime_sample_id");
        require(bounded.update(legacy,2) && !bounded.update(legacy,2) && bounded.update(sample(30,690690,53),2) &&
            !bounded.update(sample(31,690690,50),2),"legacy extension or configured cap failed");
        auto formal=sample(90,690690,50);formal["benchmark_kind"]="\"formal_fixture\"";
        require(bounded.update(formal),"formal sample mixed with runtime");formal["median_seconds"]="60";
        require(!bounded.update(formal),"formal fastest-summary policy changed");
        const auto path=std::filesystem::path(argv[1])/"roundtrip.toml";
        {std::ofstream out(path);out<<profile.serialize();}
        const auto loaded=p::Profile::load(path);require(loaded.samples.size()==12,"bounded samples lost on TOML roundtrip");
        require(loaded.serialize()==profile.serialize(),"runtime serialization is not stable");
        p::Profile old_profile;old_profile.condition(profile.conditions.at(0));old_profile.samples.push_back(legacy);
        legacy["b1"]="22000000";old_profile.samples.push_back(legacy);
        {std::ofstream out(path);out<<old_profile.serialize();}
        require(p::Profile::load(path).samples.size()==2,"distinct legacy B1 scopes became duplicate identities");
        std::cout<<"{\"complete\":true,\"retains_slower_samples\":true,\"cap\":10,\"independent_samples\":12,"
            "\"observed_D_win\":true,\"hysteresis\":true,\"carrier_first\":true,\"deduplication\":true,"
            "\"condition_and_build_isolation\":true,\"legacy_and_roundtrip\":true,\"CUDA_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
