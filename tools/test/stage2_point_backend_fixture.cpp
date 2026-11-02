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
static void s4_policy_tests(const std::filesystem::path &out) {
    Fields old{{"uuid_hex","\"S4-device\""},{"sm_major","8"},{"sm_minor","9"},{"cuda_driver","13030"},
        {"cuda_runtime","13030"},{"gl_fixed_mode","3"},{"outer_unroll_u","0"},{"add_sub_mask","1"}};
    auto coop_condition=old;coop_condition["env_s4_coop_reduce"]="32";
    require(p::matches(old,coop_condition,p::Ignore::parse("")),"S4 request still splits actual row identity");
    require(p::ntt_matches(old,coop_condition,p::Ignore::parse("")),"S4 invalidated standalone NTT");
    for(const char *key:{"env_s4_oracle_pin_kb","env_s4_oracle_pin_pool","env_carry_residual_mode"}) {
        auto check_condition=old;check_condition[key]="1";
        require(!p::matches(old,check_condition,p::Ignore::parse("")),"Strict ECM environment lost checker identity");
        require(p::ntt_matches(old,check_condition,p::Ignore::parse("")),"Non-transform checking invalidated standalone NTT");
    }
    p::Profile profile;profile.condition(old);const auto id=profile.condition(coop_condition);
    for(unsigned mode=0;mode<3;++mode)for(unsigned repeat=0;repeat<2;++repeat) {
        auto row=sample(700+mode*2+repeat,0,10.-2*mode,0);
        row["condition"]=std::to_string(mode?id:0);row["reduce_coop"]=mode?"32":"0";
        row["reduce_threads"]=mode?"128":"0";row["reduce_kernel"]=std::to_string(mode);
        require(profile.update(row),"S4 sample missing");
    }
    const auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){*slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+19)/20))*n*=2;return true;};
    p::Estimator engine(profile,old,p::Ignore::parse(""),query);
    auto scalar=engine.shape(profile.samples[0]),warp=engine.shape(profile.samples[1]),merged=engine.shape(profile.samples[2]);
    require(!engine.eligible(profile.samples[0],warp) && !engine.eligible(profile.samples[1],merged),"S4 variants merged");
    const auto choice=p::qualified_reduce_backend(engine,scalar,2);
    require(choice.available && choice.mode==32 && choice.kernel==2 && choice.threads==128,"S4 fastest repeated variant not selected");
    require(p::qualified_point_backends(engine,merged,false,false,2,false,true),"S4 checked edge rejected");
    require(!p::qualified_reduce_backend(engine,scalar,3).available,"S4 repeat threshold ignored");
    auto nearby=scalar;nearby.d=6930;require(p::qualified_reduce_backend(engine,nearby,2).available,"same NTT interval reuse rejected");
    nearby.d=30030;require(!p::qualified_reduce_backend(engine,nearby,2).available,"S4 crossed NTT cliff without pair");
    auto x=warp;x.target_bits=x.arithmetic_bits=10176;
    require(engine.eligible(profile.samples[1],x) && engine.s4_coop_tiles(x)==5,"within-tile sharing rejected");
    x.target_bits=x.arithmetic_bits=10240;require(!engine.eligible(profile.samples[1],x) && engine.s4_coop_tiles(x)==6,"high-word tile boundary crossed");
    x=warp;x.reduce_threads=96;require(!engine.eligible(profile.samples[1],x),"S4 thread identity crossed");
    profile.samples[2]["point_check_seconds"]="5";
    p::Estimator checks(profile,old,p::Ignore::parse(""),query);require(p::qualified_reduce_backend(checks,scalar,2).kernel==1,"S4 ignored startup checks");
    profile.samples[2]["point_check_seconds"]="0";
    p::Profile unpaired=profile;unpaired.samples.erase(unpaired.samples.begin());p::Estimator alone(unpaired,old,p::Ignore::parse(""),query);
    require(!p::qualified_reduce_backend(alone,warp,2).available,"unpaired S4 treated as qualified");
    std::vector<p::RuntimeCandidate> all{{scalar,{2,0,10,0,0},1,true},{warp,{2,0,8,0,0},1,true}};
    auto trials=p::runtime_reduce_trials(all,2);require(trials.size()==1 && trials[0].reduce_kernel==2,"merged S4 trial not bounded");
    all.push_back({merged,{2,0,6,0,0},1,true});require(p::runtime_reduce_trials(all,2).empty(),"completed S4 repeated");
    auto next=scalar;next.d=30030;all.push_back({next,{2,0,12,0,0},2,true});trials=p::runtime_reduce_trials(all,2);
    require(trials.size()==1 && trials[0].d==30030 && trials[0].reduce_kernel==1,"completed old D starved new interval");
    std::ofstream(out/"s4.toml")<<profile.serialize();auto loaded=p::Profile::load(out/"s4.toml");
    p::Estimator restored(loaded,old,p::Ignore::parse(""),query);
    require(loaded.samples.size()==3 && p::qualified_reduce_backend(restored,scalar,2).kernel==2,"S4 variants lost through serialized profile");
    auto legacy=sample(790,0,3,0);legacy["condition"]=std::to_string(id);require(profile.update(legacy),"legacy S4 lost");
    require(p::optional_uint(profile.samples.back(),"reduce_coop")==255,"unobserved legacy S4 relabelled");
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
    require(p::matches(old,current,ignore) && p::ntt_matches(old,current,ignore),"legacy requests split actual row identities");
    require(ecm_stage2::seed_backend("auto")==-1 && ecm_stage2::seed_backend("scalar")==0 &&
            ecm_stage2::seed_backend("coop32")==32,"seed option parsing");
    bool rejected=false;try{ecm_stage2::seed_backend("coop16");}catch(...){rejected=true;}
    require(rejected,"unsupported seed option accepted");
    p::Profile profile;profile.condition({{"uuid_hex","\"seed-device\""},{"sm_major","8"},{"sm_minor","9"},
        {"cuda_driver","13030"},{"cuda_runtime","13030"},{"gl_fixed_mode","3"},{"outer_unroll_u","0"},{"add_sub_mask","1"}});
    unsigned id=200;
    for(const auto &mode:std::vector<std::array<unsigned,4>>{{0,0,0,10},{32,0,0,8},{32,32,0,6},{32,32,32,4}})
        for(unsigned i=0;i<2;++i) {
            auto row=sample(id++,mode[0],mode[3]+i*.1,.1);row["giant_coop"]=std::to_string(mode[1]);
            row["seed_coop"]=std::to_string(mode[2]);row["b2"]="51879828";row["giant_chunk_points"]="8640";row["giant_chain_min"]="8192";
            require(profile.update(row),"seed coordinate sample");
        }
    p::Estimator engine(profile,profile.conditions.at(0),ignore,{});
    auto scalar=engine.shape(profile.samples[2]),coop=engine.shape(profile.samples[3]);
    require(scalar.key()!=coop.key() && !engine.eligible(profile.samples[2],coop),"ignored environment mixed actual seeds");
    require(p::qualified_seed_backend(engine,scalar,2).mode==32,"paired seed winner not selected");
    require(p::qualified_point_backends(engine,coop,true,true,2,true),"three-axis coordinate path requires Cartesian grid");
    auto nearby=scalar;nearby.d=5460;nearby.b1=1000000;
    require(p::qualified_seed_backend(engine,nearby,2).mode==32,"seed region not reused across D/B1");
    nearby.giant_chunk=4096;require(!p::qualified_seed_backend(engine,nearby,2).available,"seed reused across chain occupancy boundary");
    p::Profile unpaired=profile;unpaired.samples.erase(unpaired.samples.begin());
    p::Estimator alone(unpaired,unpaired.conditions.at(0),ignore,{});
    require(!p::qualified_point_backends(alone,coop,true,true,2,true),"unproved scalar prefix accepted as automatic path");
    std::vector<p::RuntimeCandidate> candidates{{scalar,engine.runtime_statistics(scalar),1,true}};
    auto trials=p::runtime_seed_trials(candidates,2);require(trials.size()==1 && trials[0].seed_coop==32 &&
        trials[0].baby_coop==32 && trials[0].giant_coop==32 && trials[0].d==scalar.d,"seed trial expanded geometry/backends");
    candidates.push_back({coop,engine.runtime_statistics(coop),1,true});
    require(p::runtime_seed_trials(candidates,2).empty(),"completed seed trial repeated");
    require(p::runtime_choice(candidates,false,2).index==1,"fastest checked seed not selected");
    candidates.back().statistics.point_check_seconds=5;
    require(p::runtime_choice(candidates,false,2).index==0,"seed startup check cost omitted");
    auto chain=coop;chain.chain_block=32;
    for(unsigned i=0;i<2;++i) {
        auto row=sample(id++,32,3+i*.1,.1);row["giant_coop"]=row["seed_coop"]="32";
        row["b2"]="51879828";row["giant_chunk_points"]="8640";row["giant_chain_min"]="8192";row["giant_chain_block"]="32";
        require(profile.update(row),"measured chain coordinate sample");
        p::Estimator next(profile,profile.conditions.at(0),ignore,{});
        require(p::qualified_point_backends(next,chain,true,true,2,true)==(i==1),"chain winner lost or premature chain proof");
    }
    auto legacy=sample(id++,0,1,0);legacy.erase("seed_coop");auto condition=profile.conditions.at(0);
    condition["env_giant_seed_coop"]="32";legacy["condition"]=std::to_string(profile.condition(condition));
    require(profile.update(legacy),"legacy seed observation lost");
    require(ecm_stage2::tune::uint(profile.samples.back(),"seed_coop")==255,"legacy request guessed as observed seed arithmetic");
}
int main(int argc,char **argv) {
    try {
        if(argc==4 && (std::string(argv[1])=="--make-save" || std::string(argv[1])=="--make-save-from-modulus")) {
            mpz_t n;mpz_init(n);
            if(std::string(argv[1])=="--make-save") {
                const unsigned requested=unsigned(std::stoul(argv[2]));require(requested>=8193 && requested<=16384,"invalid fixture width");
                mpz_set_ui(n,1);mpz_mul_2exp(n,n,requested-1);mpz_add_ui(n,n,12345);mpz_nextprime(n,n);
            } else {
                std::ifstream input(argv[2]);std::string modulus;input>>modulus;
                require(bool(input) && mpz_set_str(n,modulus.c_str(),16)==0,"invalid fixture modulus");
            }
            const unsigned bits=unsigned(mpz_sizeinbase(n,2));require(bits>=8193 && bits<=16384 && mpz_odd_p(n),"invalid fixture modulus width/parity");
            std::vector<char> buffer(bits/4+4);mpz_get_str(buffer.data(),16,n);const std::string hex=buffer.data();mpz_clear(n);
            const auto x=ecm_stage2::tune::benchmark_point(hex);
            std::ofstream out(argv[3]);out<<"METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x"<<hex<<"; X=0x"<<x<<"; Z=1;\n";
            require(bool(out),"save write failed");std::cout<<"generated Montgomery Stage1 fixture "<<bits<<" bits\n";return 0;
        }
        require(argc==2,"need output directory");
        s4_policy_tests(argv[1]);
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
