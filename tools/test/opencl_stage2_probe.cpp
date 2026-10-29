#include "ecm/stage2/opencl/tree.h"
#include "ecm/platform/program_version.h"
#include "opencl_stage2_oracle.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace cl2=ecm_opencl::stage2;
namespace ntt=ecm_opencl::ntt;
namespace ref=stage2_oracle;
using Word=ntt::Word;
namespace {
struct Counts {Word words=0,trees=0,folds=0,curves=0,guards=0;};
std::string quote(const std::string&s){std::string out="\"";for(char c:s){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else out+=c;}return out+'"';}
Word random(Word&s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
template<class F>void rejects(const char *message,F f,Counts&c){try{f();}catch(const std::exception&e){if(std::string(e.what()).find(message)==std::string::npos)throw;++c.guards;return;}throw std::runtime_error(std::string("missing rejection: ")+message);}
void same(cl2::Arithmetic &a,const cl2::Polynomial&p,const ref::Poly&expected,Counts&c,const char *where){const auto actual=a.download(p),want=ref::flatten(expected,a.words());
    if(actual.size()!=want.size())throw std::runtime_error(std::string(where)+" size mismatch");for(std::size_t i=0;i<actual.size();++i)if(actual[i]!=want[i])throw std::runtime_error(std::string(where)+" GMP mismatch word="+std::to_string(i));c.words+=actual.size();}
std::vector<Word> modulus(unsigned bits,bool mersenne){std::vector<Word> n((bits+63)/64,~Word(0));if(bits%64)n.back()=(Word(1)<<(bits%64))-1;if(!mersenne)n[0]-=32;return n;}
ref::Poly random_roots(const ref::Big&n,std::size_t count,Word&seed){ref::Poly p(count);std::vector<Word> v((mpz_sizeinbase(n.v,2)+63)/64);
    for(auto &x:p){for(auto&w:v)w=random(seed);mpz_import(x.v,v.size(),-1,8,0,0,v.data());mpz_mod(x.v,x.v,n.v);}if(count>2){p[0]=ref::Big(0);p[1]=p[2];}return p;}
cl2::ArithmeticOptions options(const std::vector<Word>&n,bool mersenne,bool group=true,bool reuse=true){cl2::ArithmeticOptions o;o.product.modulus=n;o.product.mersenne=mersenne;o.group_batches=group;o.reuse_workspace=reuse;return o;}
void check_tree(cl2::Arithmetic&a,const cl2::ProductTree&t,const ref::Poly&roots,const ref::Big&n,Counts&c){
    std::vector<ref::Poly> oracle(2*t.padded);for(std::size_t i=0;i<t.padded;++i)oracle[t.padded+i]=i<roots.size()?ref::roots({roots[i]},n):ref::Poly{ref::Big(1)};
    for(std::size_t i=t.padded;i-->1;)oracle[i]=ref::multiply(oracle[2*i],oracle[2*i+1],n);
    for(std::size_t i=1;i<oracle.size();++i)same(a,t.nodes[i],oracle[i],c,"tree node");++c.trees;
}
void algebra(ntt::Runtime&r,unsigned bits,std::size_t count,bool mersenne,bool grouped,Counts&c){
    const auto limbs=modulus(bits,mersenne);const auto n=ref::import(limbs);Word seed=0xb1d21836a09fe734ULL+bits+count;
    auto babies=random_roots(n,count,seed),giants=random_roots(n,2*count+3,seed);cl2::Arithmetic a(r,options(limbs,mersenne,grouped));
    auto roots=a.upload(ref::flatten(babies,a.words()),babies.size());cl2::Resultant resultant(a,roots);check_tree(a,resultant.tree(),babies,n,c);
    auto f=ref::roots(babies,n),reverse=f;std::reverse(reverse.begin(),reverse.end());same(a,resultant.inverse(),ref::inverse(reverse,count+1,n),c,"inverse");
    ref::Poly h{ref::Big(1)};auto gs=a.upload(ref::flatten(giants,a.words()),giants.size());
    for(std::size_t at=0;at<giants.size();at+=count){const auto end=std::min(at+count,giants.size());ref::Poly block(giants.begin()+at,giants.begin()+end);
        resultant.append(a.window(gs,at,end-at));h=ref::remainder(ref::multiply(h,ref::roots(block,n),n),f,n);same(a,resultant.fold(),h,c,"fold");++c.folds;}
    const auto values=resultant.values();same(a,values,ref::direct_values(babies,giants,n),c,"scaled leaves");
    ref::Poly horner;for(const auto &b:babies)horner.push_back(ref::evaluate(h,b,n));same(a,values,horner,c,"Horner leaves");
    rejects("window",[&]{a.window(roots,count,1);},c);rejects("noncanonical",[&]{a.upload(ref::flatten({n},a.words()),1);},c);
    rejects("a(0)=1",[&]{a.inverse_series(a.constant(2),4);},c);
    rejects("exceeds baby",[&]{resultant.append(gs);},c);
    auto root_only=cl2::product_tree(a,roots,false);same(a,root_only.nodes[1],f,c,"root-only tree");
    for(std::size_t i=2;i<root_only.nodes.size();++i)if(root_only.nodes[i].buffer.words())throw std::runtime_error("G tree retained a child");++c.guards;
    if(count>1)rejects("retained children",[&]{cl2::evaluate_scaled(a,root_only,resultant.fold(),resultant.inverse());},c);
    auto scalar=a.constant(2);same(a,a.multiply(scalar,scalar),{ref::Big(4)},c,"same-source square");
    a.finish();std::cout<<"opencl_tree_verified: bits="<<bits<<" P="<<count<<" mersenne="<<mersenne<<" grouped="<<grouped<<" bad=0\n";
}
ref::Curve hit_fixture(){
    // Deterministic CPU-only search for a real Stage2 hit, excluding setup,
    // Stage1 and normalization hits. No GPU timing contains this search.
    for(unsigned p=101;p<4000;++p)if(ref::prime(p))for(unsigned sigma:{6u,11u,26u}){
        auto n=ref::Big(p);mpz_mul_ui(n.v,n.v,1000003);
        try{auto c=ref::curve(n,sigma,20,300,30);auto factor=ref::factor(ref::direct_values(c.babies,c.giants,n),n);
            if(mpz_cmp_ui(factor.v,1)>0 && mpz_cmp(factor.v,n.v)<0){
                for(unsigned q=21;q<=300;++q)if(ref::prime(q)){
                    const auto z=ref::ladder(c.stage1,ref::Big(q),c.a24,n).z;
                    if(mpz_divisible_p(z.v,factor.v))return c;
                }
            }
        }catch(const std::runtime_error&){} }
    throw std::runtime_error("no Stage2-hit fixture found");
}
void curve_case(ntt::Runtime&r,const ref::Curve&curve,const ref::Big&target,bool mersenne,Counts&counts){
    const auto nw=(mpz_sizeinbase(curve.modulus.v,2)+63)/64;auto limbs=ref::flatten({curve.modulus},nw);cl2::Arithmetic a(r,options(limbs,mersenne));
    auto babies=a.upload(ref::flatten(curve.babies,nw),curve.babies.size()),giants=a.upload(ref::flatten(curve.giants,nw),curve.giants.size());
    const auto begin=ref::now();cl2::Resultant result(a,babies);for(std::size_t at=0;at<curve.giants.size();at+=curve.babies.size())result.append(a.window(giants,at,std::min(curve.babies.size(),curve.giants.size()-at)));
    auto output=a.download(result.values());const double seconds=ref::now()-begin;
    const auto expected=ref::direct_values(curve.babies,curve.giants,curve.modulus);if(output!=ref::flatten(expected,nw))throw std::runtime_error("full Stage2 leaf mismatch");
    ref::Poly leaves(expected.size());for(std::size_t i=0;i<leaves.size();++i)mpz_import(leaves[i].v,nw,-1,8,0,0,output.data()+i*nw);
    const auto actual=ref::factor(leaves,target),want=ref::factor(expected,target);if(mpz_cmp(actual.v,want.v))throw std::runtime_error("full Stage2 GCD mismatch");
    // Every prime in (B1,B2] is represented by a giant +/- a coprime baby.
    for(unsigned p=curve.b1+1;p<=curve.b2;++p)if(ref::prime(p)){bool covered=false;for(unsigned i=1;i<=curve.b2/curve.d+2;++i)for(unsigned j=1;j<=curve.d/2;++j)
        if(std::gcd(j,curve.d)==1 && (i*curve.d+j==p || i*curve.d-j==p))covered=true;if(!covered)throw std::runtime_error("fixture prime coverage gap");}
    ++counts.curves;counts.words+=output.size();std::cout<<"opencl_curve_verified: target="<<ref::text(target)<<" arithmetic_bits="<<mpz_sizeinbase(curve.modulus.v,2)<<" sigma="<<curve.sigma
        <<" point_bits="<<mpz_sizeinbase(curve.point_modulus.v,2)<<" B1="<<curve.b1<<" B2="<<curve.b2<<" D="<<curve.d<<" P="<<curve.babies.size()<<" giant="<<curve.giants.size()<<" mersenne="<<mersenne<<" gcd="<<ref::text(actual)<<" cold_poly_wall="<<seconds<<" cpu_points=1 cpu_gcd=1 bad=0\n";
}
void quick_bench(ntt::Runtime&r,std::ostringstream&json){
    struct Setting{bool batch,reuse,borrow;const char*name;};const std::array<Setting,4> settings={{{false,true,true,"serial"},{true,false,true,"batched_transient"},{true,true,false,"batched_copy"},{true,true,true,"batched_reuse"}}};
    json<<'[';bool comma=false;
    for(auto spec:std::array<std::pair<unsigned,std::size_t>,2>{{{512,32},{2203,32}}}) {
        const auto limbs=modulus(spec.first,false);const auto n=ref::import(limbs);Word seed=0x7c1529beULL;auto babies=random_roots(n,spec.second,seed),giants=random_roots(n,2*spec.second+3,seed);
        const auto expected=ref::flatten(ref::direct_values(babies,giants,n),limbs.size());
        for(const auto &setting:settings){auto config=options(limbs,false,setting.batch,setting.reuse);config.borrow_contiguous=setting.borrow;
            cl2::Arithmetic a(r,config);auto b=a.upload(ref::flatten(babies,limbs.size()),babies.size()),g=a.upload(ref::flatten(giants,limbs.size()),giants.size());
            auto run=[&]{const auto t0=ref::now();cl2::Resultant result(a,b);const auto t1=ref::now();
                for(std::size_t at=0;at<giants.size();at+=babies.size())result.append(a.window(g,at,std::min(babies.size(),giants.size()-at)));
                const auto t2=ref::now();auto out=a.download(result.values());const auto t3=ref::now();
                if(out!=expected)throw std::runtime_error("benchmark GMP mismatch");return std::array<double,3>{t1-t0,t2-t1,t3-t2};};
            // Explicit benchmark priming, distinct from the user's tune warmup policy.
            const auto cold=ref::now();run();const auto cold_seconds=ref::now()-cold;const auto before=a.stats();std::array<double,3> samples{};
            std::array<double,3> phases{};
            for(auto &seconds:samples){const auto start=ref::now();const auto phase=run();seconds=ref::now()-start;for(std::size_t i=0;i<phases.size();++i)phases[i]+=phase[i]/3;}
            std::sort(samples.begin(),samples.end());const auto after=a.stats();
            if(comma)json<<',';comma=true;json<<"{\"bits\":"<<spec.first<<",\"P\":"<<spec.second<<",\"giants\":"<<giants.size()<<",\"setting\":"<<quote(setting.name)
                <<",\"median_seconds\":"<<samples[1]<<",\"min_seconds\":"<<samples[0]<<",\"max_seconds\":"<<samples[2]<<",\"repeats\":3,\"cold_seconds\":"<<cold_seconds
                <<",\"multiply_calls_per_run\":"<<(after.multiply_calls-before.multiply_calls)/3<<",\"pairs_per_run\":"<<(after.multiply_pairs-before.multiply_pairs)/3
                <<",\"h2d_bytes_per_run\":"<<(after.upload_bytes-before.upload_bytes)/3<<",\"d2h_bytes_per_run\":"<<(after.download_bytes-before.download_bytes)/3
                <<",\"device_copies_per_run\":"<<(after.device_copies-before.device_copies)/3<<",\"avoided_copy_bytes_per_run\":"<<(after.avoided_copy_bytes-before.avoided_copy_bytes)/3
                <<",\"phase_mean_seconds\":{\"F_tree_inverse\":"<<phases[0]<<",\"G_trees_fold\":"<<phases[1]<<",\"descent_readback\":"<<phases[2]<<"}}";
            std::cout<<"opencl_stage2_bench: bits="<<spec.first<<" P="<<spec.second<<" setting="<<setting.name<<" median="<<samples[1]<<" repeats=3 checked=1\n";
        }
    }json<<']';
}
}
int main(int argc,char**argv){std::cout<<std::unitbuf;std::cerr<<std::unitbuf;
    try{std::size_t device=1;bool verify=false,bench=false,smoke=false;std::string output,root=ECM_OPENCL_NTT_RESOURCE_ROOT;
        for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--verify"){verify=true;continue;}if(key=="--bench"){bench=true;continue;}if(key=="--smoke"){smoke=true;continue;}
            if(key=="--help"){std::cout<<"opencl_stage2_probe --verify [--bench] [--smoke] --device N --output FILE [--resource-root DIR]\n";return 0;}
            if(i+1>=argc)throw std::invalid_argument("missing argument");const std::string value=argv[++i];if(key=="--device")device=std::stoull(value);else if(key=="--output")output=value;else if(key=="--resource-root")root=value;else throw std::invalid_argument("unknown option "+key);}
        if(!verify && !bench)throw std::invalid_argument("use --verify or --bench");if(!output.empty() && std::filesystem::exists(output))throw std::invalid_argument("output already exists; use fresh evidence");
        ntt::Options ro;ro.device=device;ro.resource_root=root;ro.budget_bytes=256ULL<<20;ntt::Runtime runtime(ro);Counts counts;const auto start=ref::now();
        if(verify){const std::array<std::pair<unsigned,std::size_t>,8> grid={{{31,1},{63,3},{127,5},{512,17},{2203,33},{4423,7},{8191,3},{16384,3}}};
            for(const auto &spec:grid){if(smoke && spec.first>127)continue;for(bool mersenne:{false,true})algebra(runtime,spec.first,spec.second,mersenne,true,counts);}
            algebra(runtime,127,5,false,false,counts);auto hit=hit_fixture();curve_case(runtime,hit,hit.modulus,false,counts);
            ref::Big prime;mpz_set_ui(prime.v,1);mpz_mul_2exp(prime.v,prime.v,127);mpz_sub_ui(prime.v,prime.v,1);auto curve=ref::curve(prime,26,20,300,30);curve_case(runtime,curve,prime,false,counts);curve_case(runtime,curve,prime,true,counts);
            // A unit-normalized cofactor pair. A separate negative fixture keeps
            // the small-factor/nonunit boundary visible; it is not a GPU failure.
            ref::Big carrier;mpz_set_ui(carrier.v,1);mpz_mul_2exp(carrier.v,carrier.v,41);mpz_sub_ui(carrier.v,carrier.v,1);ref::Big target;
            if(!mpz_divisible_ui_p(carrier.v,13367))throw std::runtime_error("cofactor fixture divisibility");mpz_divexact_ui(target.v,carrier.v,13367);
            ref::Big small_carrier;mpz_set_ui(small_carrier.v,1);mpz_mul_2exp(small_carrier.v,small_carrier.v,91);mpz_sub_ui(small_carrier.v,small_carrier.v,1);
            rejects("fixture ",[&]{ref::curve(small_carrier,26,20,300,30);},counts);
            bool paired=false;for(unsigned sigma=6;sigma<60 && !paired;++sigma){try{auto lifted=ref::curve(carrier,sigma,20,300,30),ordinary=ref::curve(target,sigma,20,300,30);
                const auto lv=ref::direct_values(lifted.babies,lifted.giants,carrier),ov=ref::direct_values(ordinary.babies,ordinary.giants,target);for(std::size_t i=0;i<lv.size();++i){ref::Big v;mpz_mod(v.v,lv[i].v,target.v);if(mpz_cmp(v.v,ov[i].v))throw std::runtime_error("cofactor root projection mismatch");}
                curve_case(runtime,ordinary,target,false,counts);curve_case(runtime,lifted,target,true,counts);paired=true;
            }catch(const std::runtime_error&e){if(std::string(e.what()).find("fixture ")==0)continue;throw;}}if(!paired)throw std::runtime_error("no unit-normalized cofactor fixture");
            // With an extraneous small factor in M, normalize/compute points
            // modulo the target and lift their representatives into M. Polynomial
            // additions/multiplications commute with projection M -> target.
            ref::Big large_target;mpz_divexact_ui(large_target.v,small_carrier.v,127);bool lifted_pair=false;
            for(unsigned sigma=6;sigma<60 && !lifted_pair;++sigma){try{
                auto ordinary=ref::curve(large_target,sigma,20,300,30),lifted=ordinary;lifted.modulus=small_carrier;
                curve_case(runtime,ordinary,large_target,false,counts);curve_case(runtime,lifted,large_target,true,counts);lifted_pair=true;
            }catch(const std::runtime_error&e){if(std::string(e.what()).find("fixture ")==0)continue;throw;}}
            if(!lifted_pair)throw std::runtime_error("no target-normalized lifted fixture");
            auto tiny_options=ro;tiny_options.budget_bytes=4096;ntt::Runtime tiny(tiny_options);
            rejects("budget",[&]{cl2::Arithmetic a(tiny,options(modulus(512,false),false));Word seed=1;auto n=ref::import(modulus(512,false));auto rs=random_roots(n,32,seed);auto roots=a.upload(ref::flatten(rs,a.words()),rs.size());cl2::Resultant result(a,roots);},counts);
            if(tiny.memory().live_bytes)throw std::runtime_error("failed request payload leak");++counts.guards;
        }
        std::ostringstream benchmarks;if(bench)quick_bench(runtime,benchmarks);else benchmarks<<"[]";runtime.finish();if(runtime.memory().live_bytes)throw std::runtime_error("Stage2 payload leak");
        std::ostringstream json;json<<"{\"status\":\"passed\",\"version\":"<<quote(ecm_program::version)<<",\"device\":"<<quote(runtime.device().name)<<",\"driver\":"<<quote(runtime.device().driver)
            <<",\"checked_words\":"<<counts.words<<",\"trees\":"<<counts.trees<<",\"folds\":"<<counts.folds<<",\"curves\":"<<counts.curves<<",\"guards\":"<<counts.guards
            <<",\"payload_peak_bytes\":"<<runtime.memory().peak_bytes<<",\"verification_and_bench_wall\":"<<ref::now()-start<<",\"cpu_points\":true,\"cpu_gcd\":true,\"production_queue\":false,\"benchmarks\":"<<benchmarks.str()<<",\"bad\":0}";
        if(!output.empty()){std::ofstream file(output,std::ios::binary);if(!file)throw std::runtime_error("cannot write report");file<<json.str()<<'\n';file.close();if(!file)throw std::runtime_error("report write failed");}std::cout<<"opencl_stage2_summary: "<<json.str()<<'\n';return 0;
    }catch(const std::exception&e){std::cerr<<"opencl_stage2_failed: "<<e.what()<<'\n';return 2;}}
