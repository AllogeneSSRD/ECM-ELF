#include "../../src/ecm/stage2/planning/runtime_tune.h"
#include "ecm/platform/program_version.h"
#include <fstream>
#include <iostream>
namespace p=ecm_stage2::tune::portable;
using ecm_stage2::tune::Fields;
static void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
static Fields sample(unsigned id,unsigned mode,double seconds,double checks) {
    return {{"condition","0"},{"target_bits","9216"},{"arithmetic_bits","9216"},{"carrier_exponent","0"},
        {"modulus_kind","\"generic\""},{"b1","50000"},{"b2","26000000"},{"d","6006"},{"p","720"},
        {"fold_resident","1"},{"frontier_resident","1"},{"execution_path","\"resident\""},
        {"giant_chunk_points","5760"},{"giant_chain_min","32768"},{"giant_force_ladder","0"},
        {"baby_coop",std::to_string(mode)},{"giant_coop","0"},{"groot_balanced","0"},
        {"source","\"measured\""},{"benchmark_kind","\"runtime_save\""},{"repeats","1"},
        {"mean_seconds",p::numeric(seconds)},{"stddev_seconds","0"},{"min_seconds",p::numeric(seconds)},
        {"max_seconds",p::numeric(seconds)},{"point_check_seconds",p::numeric(checks)},
        {"selftest_cases","1"},{"checked","96"},{"clean","1"},{"bad","0"},{"hits","0"},
        {"runtime_sample_id",p::quoted("backend-"+std::to_string(id))},
        {"program_version",p::quoted(ecm_program::version)},{"updated_at",p::quoted(p::utc_updated_at(1700000000000000000ull+id))}};
}
static void giant_tests() {
    p::Profile profile;profile.condition({{"uuid_hex","\"giant-device\""},{"sm_major","8"},{"sm_minor","9"},
        {"cuda_driver","13030"},{"cuda_runtime","13030"},{"gl_fixed_mode","3"},{"outer_unroll_u","0"},{"add_sub_mask","1"}});
    unsigned id=100;
    for(const auto &mode:std::vector<std::array<unsigned,3>>{{0,0,10},{32,0,8},{32,32,6},{32,16,7}})
        for(unsigned i=0;i<2;++i) {
            auto row=sample(id++,mode[0],mode[2]+i*.1,.1);
            row["b2"]="51879828";row["giant_chunk_points"]="8640";row["giant_chain_min"]="8192";
            row["giant_coop"]=std::to_string(mode[1]);require(profile.update(row),"giant backend sample");
        }
    p::Estimator engine(profile,profile.conditions.at(0),p::Ignore::parse(""),{});
    auto scalar=engine.shape(profile.samples[1]),coop=engine.shape(profile.samples[2]);
    const auto choice=p::qualified_giant_backend(engine,scalar,2);
    if(!choice.available || choice.mode!=32)throw std::runtime_error("giant choice: available="+std::to_string(choice.available)+
        " mode="+std::to_string(choice.mode)+" baby="+std::to_string(scalar.baby_coop)+" giant="+std::to_string(scalar.giant_coop)+
        " B2="+std::to_string(scalar.b2)+" chunk="+std::to_string(scalar.giant_chunk)+" bucket="+std::to_string(p::giant_backend_bucket(scalar)));
    require(p::qualified_point_backends(engine,coop,true,true,2),"coordinate proof rejected without exhaustive grid");
    auto nearby=scalar;nearby.d=5460;nearby.b1=1000000;
    require(p::qualified_giant_backend(engine,nearby,2).mode==32,"paired giant region not reused across D/B1");
    nearby.giant_chunk=4096;require(!p::qualified_giant_backend(engine,nearby,2).available,"chain boundary crossed");
    nearby=scalar;nearby.frontier=false;require(!p::qualified_giant_backend(engine,nearby,2).available,"residency boundary crossed");
    nearby=scalar;nearby.chain_block=32;require(!p::qualified_giant_backend(engine,nearby,2).available,"chain length identity crossed");
    p::Profile unpaired=profile;
    unpaired.samples.erase(std::remove_if(unpaired.samples.begin(),unpaired.samples.end(),[](const auto &row){return !ecm_stage2::tune::uint(row,"giant_coop");}),unpaired.samples.end());
    p::Estimator alone(unpaired,unpaired.conditions.at(0),p::Ignore::parse(""),{});
    require(!p::qualified_point_backends(alone,coop,true,true,2),"forced cooperative data treated as automatic proof");
    std::vector<p::RuntimeCandidate> candidates{{scalar,engine.runtime_statistics(scalar),1,true}};
    auto trials=p::runtime_giant_trials(candidates,2);require(trials.size()==1 && trials[0].giant_coop==32,"giant search not bounded to one alternative");
    auto partial=engine.runtime_statistics(coop);partial.count=1;candidates.push_back({coop,partial,1,true});
    require(p::runtime_giant_trials(candidates,2).empty(),"second giant trial started before first completed");
    candidates.back().statistics.count=2;trials=p::runtime_giant_trials(candidates,2);
    require(trials.size()==1 && trials[0].giant_coop==16,"TPI16 alternative not considered");
    auto fixed=scalar;fixed.giant_coop=16;candidates.push_back({fixed,engine.runtime_statistics(fixed),1,true});
    require(p::runtime_giant_trials(candidates,2).empty(),"completed giant alternatives repeated");
    require(p::runtime_choice(candidates,false,2).index==1,"giant measured winner not selected");
    for(auto &row:profile.samples)if(ecm_stage2::tune::uint(row,"giant_coop")==32)row["point_check_seconds"]="5";
    p::Estimator startup(profile,profile.conditions.at(0),p::Ignore::parse(""),{});
    require(p::qualified_giant_backend(startup,scalar,2).mode==16,"giant startup checks omitted from comparison");
}
static void seed_policy_tests() {
    const auto ignore=p::Ignore::parse("gpu,driver,cuda,backend,environment");
    Fields old{{"env_giant_seed_coop","0"}},current{{"env_giant_seed_coop","32"}};
    require(!p::matches(old,current,ignore),"ignored environment mixed seed arithmetic costs");
    require(p::ntt_matches(old,current,ignore),"seed arithmetic split unchanged NTT tables");
    Fields legacy;require(p::matches(legacy,old,ignore),"legacy scalar condition not retained");
    require(!p::matches(legacy,current,ignore),"legacy scalar costs relabelled cooperative");
    // Strict matching still accepts an omitted scalar policy in old profiles.
    Fields strict{{"uuid_hex","\"seed-device\""},{"sm_major","8"},{"sm_minor","9"},
        {"cuda_driver","13030"},{"cuda_runtime","13030"},{"gl_fixed_mode","3"},
        {"outer_unroll_u","0"},{"add_sub_mask","1"}};
    auto explicit_scalar=strict;explicit_scalar["env_giant_seed_coop"]="0";
    require(p::matches(strict,explicit_scalar,p::Ignore::parse("")),"strict legacy scalar condition lost");
    require(ecm_stage2::seed_backend("scalar")==0 && ecm_stage2::seed_backend("coop32")==32,"seed option parsing");
    bool rejected=false;try{ecm_stage2::seed_backend("auto");}catch(...){rejected=true;}
    require(rejected,"unqualified automatic seed policy accepted");
}
int main(int argc,char **argv) {
    try {
        if(argc==4 && std::string(argv[1])=="--make-save") {
            const unsigned bits=unsigned(std::stoul(argv[2]));require(bits>=8193 && bits<=16384,"invalid fixture width");
            mpz_t n;mpz_init_set_ui(n,1);mpz_mul_2exp(n,n,bits-1);mpz_add_ui(n,n,12345);mpz_nextprime(n,n);
            std::vector<char> buffer(bits/4+4);mpz_get_str(buffer.data(),16,n);const std::string hex=buffer.data();mpz_clear(n);
            const auto x=ecm_stage2::tune::benchmark_point(hex);
            std::ofstream out(argv[3]);out<<"METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x"<<hex<<"; X=0x"<<x<<"; Z=1;\n";
            require(bool(out),"save write failed");std::cout<<"generated probable-prime Stage1 fixture "<<bits<<" bits\n";return 0;
        }
        require(argc==2,"need output directory");
        giant_tests();seed_policy_tests();
        p::Profile profile;
        profile.condition({{"uuid_hex","\"device\""},{"sm_major","8"},{"sm_minor","9"},{"cuda_driver","13030"},
            {"cuda_runtime","13030"},{"gl_fixed_mode","3"},{"outer_unroll_u","0"},{"add_sub_mask","1"}});
        for(unsigned i=0;i<2;++i){require(profile.update(sample(i,0,10+i*.1,.1)),"scalar sample");
            require(profile.update(sample(10+i,32,7+i*.1,.2)),"cooperative sample");}
        require(profile.samples.size()==2,"backends merged into one row");
        auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
            *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+23)/24))*n*=2;return true;};
        p::Estimator engine(profile,profile.conditions.at(0),p::Ignore::parse("environment"),query);
        auto scalar=engine.shape(profile.samples[0]),coop=engine.shape(profile.samples[1]);
        require(scalar.key()!=coop.key(),"backend absent from shape key");
        const auto a=engine.runtime_statistics(scalar),b=engine.runtime_statistics(coop);
        require(a.count==2 && b.count==2 && a.seconds>b.seconds && b.point_check_seconds==.2,"statistics crossed backend");
        require(!engine.eligible(profile.samples[0],coop),"ignored environment crossed arithmetic backend");
        auto cliff=coop;cliff.target_bits=cliff.arithmetic_bits=10241;
        require(!engine.eligible(profile.samples[1],cliff),"CGBN capacity boundary reused as measured");
        const auto selected=p::qualified_baby_backend(engine,scalar,2);
        require(selected.available && selected.mode==32,"paired backend not selected");
        p::Profile unpaired=profile;
        unpaired.samples.erase(unpaired.samples.begin());
        p::Estimator alone(unpaired,unpaired.conditions.at(0),p::Ignore::parse("environment"),query);
        require(!p::qualified_baby_backend(alone,scalar,2).available,"forced backend alone treated as paired proof");
        auto nearby=scalar;nearby.d=6930;nearby.b1=1000000;
        require(p::qualified_baby_backend(engine,nearby,2).mode==32,"paired result not reused across D/B1");
        nearby.d=2310;require(!p::qualified_baby_backend(engine,nearby,2).available,"point-batch boundary crossed");
        std::vector<p::RuntimeCandidate> all{{scalar,a,1,true}};
        auto trials=p::runtime_baby_trials(all,2);require(trials.size()==1,"backend trial grid expanded");
        require(trials[0].baby_coop==32 && trials[0].d==scalar.d && trials[0].chain_block==scalar.chain_block,"trial changed D/chain");
        all.push_back({coop,b,1,true});require(p::runtime_baby_trials(all,2).empty(),"qualified backend retested");
        require(p::runtime_choice(all,false,2).index==1,"measured winner not selected");
        all[1].statistics.seconds=9.99;all[1].statistics.point_check_seconds=.5;
        require(p::runtime_choice(all,false,2).index==0,"startup checks excluded from backend comparison");
        auto legacy=sample(30,32,1,0);legacy.erase("baby_coop");legacy.erase("giant_coop");
        auto condition=profile.conditions.at(0);condition["env_baby_coop"]="32";
        legacy["condition"]=std::to_string(profile.condition(condition));require(profile.update(legacy),"legacy data lost");
        require(ecm_stage2::tune::uint(profile.samples.back(),"baby_coop")==255,"unobserved legacy request relabelled as actual kernel");
        const auto file=std::filesystem::path(argv[1])/"profile.toml";std::ofstream(file)<<profile.serialize();
        auto restored=p::Profile::load(file);require(restored.samples.size()==3,"backend profile roundtrip");
        require(ecm_stage2::baby_backend("auto")==-1 && ecm_stage2::baby_backend("coop32")==32 &&
            ecm_stage2::giant_backend("coop16")==16 && ecm_stage2::giant_backend("auto")==-1,"backend option parsing");
        bool rejected=false;try{ecm_stage2::baby_backend("coop16");}catch(...){rejected=true;}require(rejected,"invalid baby backend accepted");
        require(!ecm_stage2::cooperative_point_width(8192) && ecm_stage2::cooperative_point_width(8193),"arithmetic width boundary");
        require(ecm_stage2::point_batch_bucket(720,0)==512 && ecm_stage2::point_batch_bucket(1ull<<28,1ull<<30)==(1ull<<24),"ladder batch cap normalization");
        for(const auto &invalid:std::vector<std::pair<std::string,std::string>>{{"point_check_seconds","-1"},{"point_check_seconds","nan"},{"arithmetic_bits","8192"}}) {
            auto row=sample(90,32,1,.1);row[invalid.first]=invalid.second;
            if(invalid.first=="arithmetic_bits")row["target_bits"]="8192";
            bool failed=false;try{p::validate(row);}catch(...){failed=true;}require(failed,"invalid backend observation accepted");
        }
        std::cout<<"{\"complete\":true,\"backend_identity\":true,\"paired_choice\":true,\"bounded_trials\":true,"
            "\"capacity_boundary\":true,\"startup_cost\":true,\"legacy_retained_unknown\":true,\"roundtrip\":true,\"CUDA_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
