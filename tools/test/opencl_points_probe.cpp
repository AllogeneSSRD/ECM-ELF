#include "ecm/stage2/opencl/points.h"
#include "ecm/stage2/opencl/tree.h"
#include "opencl_stage2_oracle.h"
#include "cuda_points_reference.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace ref=stage2_oracle;
namespace cl2=ecm_opencl::stage2;
namespace ntt=ecm_opencl::ntt;
using Word=ntt::Word;
namespace {
Word checked=0,guards=0,curves=0,qualification_rejections=0;
std::size_t ladder_group=0;
std::string quote(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';if(c=='\n')o+="\\n";else if(c=='\r')o+="\\r";else o+=c;}return o+'"';}
void same(const std::vector<Word>&a,const std::vector<Word>&b,const char*where){if(a!=b){std::size_t i=0;while(i<std::min(a.size(),b.size())&&a[i]==b[i])++i;throw std::runtime_error(std::string(where)+" mismatch word="+std::to_string(i)+(i<a.size()&&i<b.size()?" actual="+std::to_string(a[i])+" expected="+std::to_string(b[i]):" size"));}checked+=a.size();}
template<class F>void rejects(F f){try{f();}catch(const std::invalid_argument&){++guards;return;}throw std::runtime_error("missing invalid argument rejection");}
ref::Poly split(const std::vector<Word>&v,std::size_t nw){ref::Poly out(v.size()/nw);for(std::size_t i=0;i<out.size();++i)mpz_import(out[i].v,nw,-1,8,0,0,v.data()+i*nw);return out;}
std::vector<Word> flat(const ref::Big&x,std::size_t nw){return ref::flatten({x},nw);}
cl2::Polynomial roots_view(cl2::Polynomial p){p.size=p.batch;p.batch=1;p.stride=p.size*(p.buffer.words()/p.size);return p;}
ref::Poly cpu_normalize(const ref::Poly&xs,const ref::Poly&zs,const ref::Big&n,std::size_t segment){ref::Poly out(xs.size());
    for(std::size_t lo=0;lo<xs.size();lo+=segment){const auto hi=std::min(lo+segment,xs.size());ref::Poly prefix(hi-lo);ref::Big acc(1);
        for(std::size_t i=lo;i<hi;++i){acc=ref::mul(acc,zs[i],n);prefix[i-lo]=acc;}auto g=ref::gcd(acc,n);
        if(mpz_cmp_ui(g.v,1)){for(std::size_t i=lo;i<hi;++i)if(ref::unit(zs[i],n))out[i]=ref::mul(xs[i],ref::invert(zs[i],n),n);continue;}
        auto inv=ref::invert(acc,n);for(std::size_t i=hi;i-->lo;){auto iz=ref::mul(inv,i==lo?ref::Big(1):prefix[i-lo-1],n);out[i]=ref::mul(xs[i],iz,n);inv=ref::mul(inv,zs[i],n);}}
    return out;
}
ref::Poly cpu_products(const ref::Poly&v,const ref::Big&n,std::size_t block){ref::Poly out;for(std::size_t lo=0;lo<v.size();lo+=block){ref::Big a(1);for(std::size_t j=lo;j<std::min(lo+block,v.size());++j)a=ref::mul(a,v[j],n);out.push_back(a);}return out;}
void check_points(ntt::Runtime&r,const ref::Big&n,const std::vector<Word>&ks){const auto nw=(mpz_sizeinbase(n.v,2)+63)/64;cl2::Points p(r,flat(n,nw),ladder_group);
    // Separate SOS, binary GCD and inverse-normalization seams before ladder.
    // Includes a multi-limb inverse and maximal canonical residue.
    ref::Poly edge{ref::Big(0),ref::Big(1),ref::Big(3),ref::sub(n,ref::Big(1),n)};auto edge_view=p.upload(ref::flatten(edge,nw));
    same(p.download(p.products(edge_view,1)),ref::flatten(edge,nw),"single-value product identity");
    ref::Poly edge_g;for(const auto&z:edge)edge_g.push_back(ref::gcd(z,n));same(p.download(p.gcd(edge_view)),ref::flatten(edge_g,nw),"direct edge gcd");
    ref::Big over;mpz_add_ui(over.v,n.v,1);mpz_fdiv_r_2exp(over.v,over.v,mp_bitcnt_t(64*nw));ref::Poly wide{n,over};
    auto wide_words=ref::flatten(wide,nw);auto wide_buffer=r.allocate(wide_words.size());r.upload(wide_buffer,wide_words.data(),wide_words.size());
    same(p.download(p.gcd({wide_buffer,1,wide.size(),0,nw})),ref::flatten({n,ref::gcd(over,n)},nw),"noncanonical full-width gcd");
    auto edge_x=p.upload(ref::flatten(ref::Poly(edge.size(),ref::Big(1)),nw));auto edge_affine=p.normalize({edge_x,edge_view},1);
    same(p.download(edge_affine.gcd),ref::flatten(edge_g,nw),"edge inverse gcd");same(p.download(edge_affine.x),ref::flatten(cpu_normalize(ref::Poly(edge.size(),ref::Big(1)),edge,n,1),nw),"edge inverse value");
    ref::Point base{ref::Big(17),ref::Big(3)};ref::Big a24(7);auto projective=p.ladder(flat(base.x,nw),flat(base.z,nw),flat(a24,nw),ks);
    const auto xs=split(p.download(projective.x),nw),zs=split(p.download(projective.z),nw);ref::Poly expected,gs;
    for(std::size_t i=0;i<ks.size();++i){auto q=ref::ladder(base,ref::Big(ks[i]),a24,n);
        if(mpz_cmp(ref::mul(q.x,zs[i],n).v,ref::mul(q.z,xs[i],n).v))throw std::runtime_error("projective ladder mismatch");checked+=2*nw;
        auto g=ref::gcd(zs[i],n);if(mpz_cmp(g.v,ref::gcd(q.z,n).v))throw std::runtime_error("ladder Z gcd mismatch");
        if(!mpz_sgn(xs[i].v)&&!mpz_sgn(zs[i].v)&&(mpz_sgn(q.x.v)||mpz_sgn(q.z.v)))throw std::runtime_error("unexpected zero projective point");
        gs.push_back(g);expected.push_back(mpz_cmp_ui(g.v,1)?ref::Big(0):ref::mul(xs[i],ref::invert(zs[i],n),n));}
    for(std::size_t segment:{1u,3u,32u}){auto a=p.normalize(projective,segment);same(p.download(a.gcd),ref::flatten(gs,nw),"affine gcd");same(p.download(a.x),ref::flatten(expected,nw),"affine");}
    ref::Poly raw{ref::Big(0),ref::Big(1),ref::sub(n,ref::Big(1),n),ref::Big(3),ref::Big(17)};auto v=p.upload(ref::flatten(raw,nw));
    ref::Poly randoms;for(unsigned i=0;i<24;++i){ref::Big value;mpz_setbit(value.v,(i*997+13)%mpz_sizeinbase(n.v,2));mpz_mul_ui(value.v,value.v,7919+i);mpz_add_ui(value.v,value.v,i*1234567);mpz_mod(value.v,value.v,n.v);randoms.push_back(value);}
    ref::Poly random_gcds;for(const auto&value:randoms)random_gcds.push_back(ref::gcd(value,n));same(p.download(p.gcd(p.upload(ref::flatten(randoms,nw)))),ref::flatten(random_gcds,nw),"random gcd");
    // Read-only strided scalar windows exercise the same input shape as leaves.
    auto strided=v;strided.batch=3;strided.stride=2*nw;ref::Poly sparse{raw[0],raw[2],raw[4]};
    for(auto pair:{std::make_pair(v,raw),std::make_pair(strided,sparse)}){
        ref::Poly want;for(const auto&x:pair.second)want.push_back(ref::gcd(x,n));same(p.download(p.gcd(pair.first)),ref::flatten(want,nw),"gcd");
        for(std::size_t block:{1u,2u,64u})same(p.download(p.products(pair.first,block)),ref::flatten(cpu_products(pair.second,n,block),nw),"products");}
    rejects([&]{p.normalize(projective,0);});rejects([&]{p.products(v,0);});rejects([&]{p.upload(flat(n,nw));});
    std::cout<<"opencl_points_verified: bits="<<mpz_sizeinbase(n.v,2)<<" scalars="<<ks.size()<<" bad=0\n";
}
void recovery(ntt::Runtime&r){ref::Big n(257*263);cl2::Points p(r,flat(n,1));ref::Poly xs(5,ref::Big(1)),zs{ref::Big(257),ref::Big(263),ref::Big(0),ref::Big(2),ref::Big(1)};
    auto a=p.normalize({p.upload(ref::flatten(xs,1)),p.upload(ref::flatten(zs,1))},5);ref::Poly g{ref::Big(257),ref::Big(263),n,ref::Big(1),ref::Big(1)};
    same(p.download(a.gcd),ref::flatten(g,1),"complementary factors recovery");same(p.download(a.x),ref::flatten(cpu_normalize(xs,zs,n,5),1),"recovery affine");
    rejects([&]{cl2::Points invalid(r,{4});});rejects([&]{cl2::Points invalid(r,{});});
}
void full_curve(ntt::Runtime&r,const ref::Curve&c,const ref::Big&carrier){const auto nw=(mpz_sizeinbase(c.modulus.v,2)+63)/64;
    if((mpz_sizeinbase(carrier.v,2)+63)/64!=nw)throw std::runtime_error("fixture carrier limb mismatch");
    cl2::Points p(r,flat(c.modulus,nw));std::vector<Word> bs,gs;for(unsigned j=1;j<=c.d/2;++j)if(std::gcd(j,c.d)==1)bs.push_back(j);for(unsigned i=1;i<=c.b2/c.d+2;++i)gs.push_back(Word(i)*c.d);
    auto b=p.normalize(p.ladder(flat(c.stage1.x,nw),flat(c.stage1.z,nw),flat(c.a24,nw),bs));
    auto g=p.normalize(p.ladder(flat(c.stage1.x,nw),flat(c.stage1.z,nw),flat(c.a24,nw),gs));
    same(p.download(b.x),ref::flatten(c.babies,nw),"real baby points");same(p.download(g.x),ref::flatten(c.giants,nw),"real giant points");
    auto require_units=[&](const cl2::Polynomial&v){auto all=split(p.download(v),nw);for(const auto&x:all)if(mpz_cmp_ui(x.v,1))throw std::runtime_error("curve normalization unexpected hit");};require_units(b.gcd);require_units(g.gcd);
    cl2::ArithmeticOptions o;o.product.modulus=flat(carrier,nw);ref::Big next;mpz_add_ui(next.v,carrier.v,1);o.product.mersenne=mpz_popcount(next.v)==1;
    cl2::Arithmetic a(r,o);auto baby=roots_view(b.x),giant=roots_view(g.x);cl2::Resultant resultant(a,baby);
    for(std::size_t at=0;at<giant.size;at+=baby.size)resultant.append(a.window(giant,at,std::min(baby.size,giant.size-at)));
    auto leaves=resultant.values();const auto expected=ref::direct_values(c.babies,c.giants,carrier);same(a.download(leaves),ref::flatten(expected,nw),"device points to resultant");
    // gcd(a mod carrier, target) remains exact when target divides carrier;
    // no target-canonical assumption is made by the binary GCD operator.
    ref::Poly expected_gcd;for(const auto&x:expected)expected_gcd.push_back(ref::gcd(x,c.modulus));same(p.download(p.gcd(leaves)),ref::flatten(expected_gcd,nw),"carrier leaf gcd");
    if(mpz_cmp(carrier.v,c.modulus.v)==0){auto block=p.products(leaves,64);same(p.download(p.gcd(block)),ref::flatten({ref::factor(expected,c.modulus)},nw),"resultant product gcd");}
    ++curves;std::cout<<"opencl_point_curve: bits="<<mpz_sizeinbase(c.modulus.v,2)<<" carrier_bits="<<mpz_sizeinbase(carrier.v,2)<<" factor="<<ref::text(ref::factor(expected,c.modulus))<<" bad=0\n";
}
struct Cuda {
    CudaPointRun run=nullptr;
#ifdef _WIN32
    HMODULE module=nullptr;
    explicit Cuda(const std::string&path){if(path.empty())return;module=LoadLibraryW(std::filesystem::absolute(path).wstring().c_str());if(!module)throw std::runtime_error("cannot load CUDA reference DLL");run=reinterpret_cast<CudaPointRun>(GetProcAddress(module,"ecm_cuda_point_run"));if(!run)throw std::runtime_error("CUDA adapter export missing");}
    ~Cuda(){if(module)FreeLibrary(module);}
#else
    explicit Cuda(const std::string&path){if(!path.empty())throw std::runtime_error("CUDA comparison DLL currently Windows only");}
#endif
};
void bench(ntt::Runtime&r,Cuda&cuda,int cuda_device,std::ostringstream&json){bool comma=false;json<<'[';
    for(unsigned bits:{512u,2203u,4423u}){ref::Big n;mpz_setbit(n.v,bits-1);mpz_add_ui(n.v,n.v,1337);mpz_nextprime(n.v,n.v);const auto nw=(bits+63)/64;
        const auto t0=ref::now();cl2::Points p(r,flat(n,nw));const auto cold=ref::now()-t0;ref::Point base{ref::Big(17),ref::Big(3)};ref::Big a24(7);
        std::vector<Word>ks(1024);for(std::size_t i=0;i<ks.size();++i)ks[i]=((i+1)*7919)%1048573+1;
        ref::Poly cx,cz;double cpu_ladder=0,cpu_norm=0,cpu_prod=0;ref::Poly norm,products;
        for(int repeat=0;repeat<3;++repeat){ref::Poly x,z;auto start=ref::now();for(auto k:ks){auto q=ref::ladder(base,ref::Big(k),a24,n);x.push_back(q.x);z.push_back(q.z);}const auto lt=ref::now()-start;
            start=ref::now();auto affine=cpu_normalize(x,z,n,32);const auto nt=ref::now()-start;start=ref::now();auto prod=cpu_products(affine,n,64);const auto pt=ref::now()-start;
            cpu_ladder+=lt/3;cpu_norm+=nt/3;cpu_prod+=pt/3;if(!repeat){cx=x;cz=z;norm=affine;products=prod;}}
        // Explicit once-only benchmark priming. This is not production tune.
        auto projective=p.ladder(flat(base.x,nw),flat(base.z,nw),flat(a24,nw),ks);auto affine=p.normalize(projective,32);auto prod=p.products(affine.x,64);p.gcd(prod);
        for(int repeat=0;repeat<3;++repeat){if(comma)json<<',';comma=true;json<<"{\"bits\":"<<bits<<",\"count\":"<<ks.size()<<",\"repeat\":"<<repeat<<",\"cold_seconds\":"<<cold;
            auto start=ref::now();projective=p.ladder(flat(base.x,nw),flat(base.z,nw),flat(a24,nw),ks);double wall=ref::now()-start;auto lt=p.timing();json<<",\"cl_ladder_wall\":"<<wall<<",\"cl_ladder_kernel\":"<<lt.kernel_seconds<<",\"point_upload_bytes\":"<<lt.upload_bytes;
            start=ref::now();auto readx=p.download(projective.x),readz=p.download(projective.z);json<<",\"point_download_seconds\":"<<ref::now()-start;
            // CPU and GPU use equivalent points with different projective scale.
            const auto gx=split(readx,nw),gz=split(readz,nw);for(std::size_t i=0;i<ks.size();++i)if(mpz_cmp(ref::mul(gx[i],cz[i],n).v,ref::mul(gz[i],cx[i],n).v))throw std::runtime_error("bench projective mismatch");
            start=ref::now();affine=p.normalize(projective,32);wall=ref::now()-start;json<<",\"cl_normalize_wall\":"<<wall<<",\"cl_normalize_kernel\":"<<p.timing().kernel_seconds;
            same(p.download(affine.x),ref::flatten(norm,nw),"bench affine");start=ref::now();prod=p.products(affine.x,64);wall=ref::now()-start;json<<",\"cl_products_wall\":"<<wall<<",\"cl_products_kernel\":"<<p.timing().kernel_seconds;
            start=ref::now();const auto read_products=p.download(prod);json<<",\"cl_products_download\":"<<ref::now()-start;same(read_products,ref::flatten(products,nw),"bench products");
            json<<",\"cpu_ladder_mean\":"<<cpu_ladder<<",\"cpu_normalize_mean\":"<<cpu_norm<<",\"cpu_products_mean\":"<<cpu_prod;
            if(cuda.run){ref::Big rr;mpz_setbit(rr.v,nw*64);rr=ref::sub(rr,ref::Big(0),n);Word inverse=1;auto nl=flat(n,nw);for(int i=0;i<6;++i)inverse*=2-nl[0]*inverse;
                auto mx=flat(ref::mul(base.x,rr,n),nw),mz=flat(ref::mul(base.z,rr,n),nw),ma=flat(ref::mul(a24,rr,n),nw),mo=flat(rr,nw);ref::Poly mv;for(const auto&v:norm)mv.push_back(ref::mul(v,rr,n));auto vals=ref::flatten(mv,nw);
                CudaPointInput input{cuda_device,int(nw),int(ks.size()),64,0-inverse,nl.data(),mx.data(),mz.data(),ma.data(),mo.data(),ks.data(),vals.data()};std::vector<Word>ox(ks.size()*nw),oz(ox.size());char error[2048]{};double kernel=0,total=0;
                if(!repeat&&cuda.run(&input,0,ox.data(),oz.data(),&kernel,&total,error,sizeof(error)))throw std::runtime_error(error);
                if(cuda.run(&input,0,ox.data(),oz.data(),&kernel,&total,error,sizeof(error)))throw std::runtime_error(error);same(ox,readx,"CUDA/OpenCL X");same(oz,readz,"CUDA/OpenCL Z");json<<",\"cuda_ladder_kernel\":"<<kernel<<",\"cuda_ladder_roundtrip\":"<<total;
                if(!repeat&&cuda.run(&input,1,ox.data(),oz.data(),&kernel,&total,error,sizeof(error)))throw std::runtime_error(error);
                if(cuda.run(&input,1,ox.data(),oz.data(),&kernel,&total,error,sizeof(error)))throw std::runtime_error(error);ox.resize(products.size()*nw);ref::Poly mp;for(const auto&v:products)mp.push_back(ref::mul(v,rr,n));same(ox,ref::flatten(mp,nw),"CUDA products");json<<",\"cuda_products_kernel\":"<<kernel<<",\"cuda_products_roundtrip\":"<<total;
            }json<<'}';}
        for(std::size_t count:{1u,16u,256u,1024u}){ref::Poly values;for(std::size_t i=0;i<count;++i)values.push_back(norm[i%norm.size()]);auto input=p.upload(ref::flatten(values,nw));ref::Poly want;for(auto&v:values)want.push_back(ref::gcd(v,n));
            for(int repeat=0;repeat<3;++repeat){auto start=ref::now();ref::Poly cpu;for(auto&v:values)cpu.push_back(ref::gcd(v,n));const auto ct=ref::now()-start;
                start=ref::now();auto out=p.gcd(input);const auto wall=ref::now()-start,kt=p.timing().kernel_seconds;start=ref::now();auto downloaded=p.download(out);const auto read=ref::now()-start;same(downloaded,ref::flatten(want,nw),"bench gcd");
                json<<",{\"bits\":"<<bits<<",\"gcd_count\":"<<count<<",\"repeat\":"<<repeat<<",\"cpu_gcd_seconds\":"<<ct<<",\"cl_gcd_wall\":"<<wall<<",\"cl_gcd_kernel\":"<<kt<<",\"cl_gcd_download\":"<<read<<'}';}}
        std::cout<<"opencl_points_bench: bits="<<bits<<" count="<<ks.size()<<" repeats=3 bad=0\n";
    }json<<']';
}
}
int main(int argc,char**argv){std::string output,resource,cuda_path;std::size_t device=1;int cuda_device=1;unsigned only_bits=0;bool benchmark=false;
    std::cout<<std::unitbuf;std::cerr<<std::unitbuf;
    try{for(int i=1;i<argc;++i){const std::string a=argv[i];auto value=[&]{if(++i>=argc)throw std::runtime_error("missing option value");return std::string(argv[i]);};
        if(a=="--device")device=std::stoull(value());else if(a=="--output")output=value();else if(a=="--resource-root")resource=value();else if(a=="--cuda-dll")cuda_path=value();else if(a=="--cuda-device")cuda_device=std::stoi(value());else if(a=="--ladder-group")ladder_group=std::stoull(value());else if(a=="--only-bits")only_bits=unsigned(std::stoul(value()));else if(a=="--bench")benchmark=true;else if(a!="--verify")throw std::runtime_error("unknown option "+a);}
        if(output.empty()||std::filesystem::exists(output))throw std::runtime_error("output must be a new file");ntt::Options o;o.device=device;o.resource_root=resource;ntt::Runtime runtime(o);Cuda cuda(cuda_path);std::ostringstream rows;
        if(benchmark)bench(runtime,cuda,cuda_device,rows);else{
            const std::vector<unsigned> grid=only_bits?std::vector<unsigned>{only_bits}:std::vector<unsigned>{31,64,65,127,512,2203,4423,8191,16384};
            for(unsigned bits:grid){if(bits<5||bits>16384)throw std::runtime_error("only-bits outside fixture range 5..16384");std::cout<<"opencl_points_begin: bits="<<bits<<'\n';ref::Big n;mpz_setbit(n.v,bits);mpz_sub_ui(n.v,n.v,1);
                const auto&vendor=runtime.device().vendor;const bool amd=vendor.find("Advanced Micro Devices")!=std::string::npos||vendor=="AMD";
                if(amd&&bits>512){rejects([&]{cl2::Points unsupported(runtime,flat(n,(bits+63)/64));});++qualification_rejections;std::cout<<"opencl_points_unqualified: bits="<<bits<<" rejected=1\n";continue;}
                check_points(runtime,n,{0,1,2,3,17,255,65537,~Word(0)});}
            recovery(runtime);const auto&vendor=runtime.device().vendor;if(vendor.find("Advanced Micro Devices")!=std::string::npos||vendor=="AMD"){
                std::vector<Word>ordinary(9);ordinary[0]=13;ordinary.back()=1;rejects([&]{cl2::Points unsupported(runtime,ordinary);});}
            if(!only_bits){auto hit=ref::curve(ref::Big(257000771),6,20,300,30);full_curve(runtime,hit,hit.modulus);ref::Big m;mpz_setbit(m.v,127);mpz_sub_ui(m.v,m.v,1);auto miss=ref::curve(m,26,20,300,30);full_curve(runtime,miss,m);
            mpz_set_ui(m.v,0);mpz_setbit(m.v,91);mpz_sub_ui(m.v,m.v,1);ref::Big target;mpz_divexact_ui(target.v,m.v,127);auto cofactor=ref::curve(target,6,20,300,30);full_curve(runtime,cofactor,target);full_curve(runtime,cofactor,m);}rows<<"[]";}
        if(runtime.memory().live_bytes)throw std::runtime_error("point resources still live");std::ofstream f(output);f<<"{\"status\":\"passed\",\"device\":"<<quote(runtime.device().name)<<",\"driver\":"<<quote(runtime.device().driver)<<",\"cuda_comparison\":"<<(cuda.run?"true":"false")<<",\"checked_words\":"<<checked<<",\"guards\":"<<guards<<",\"curves\":"<<curves<<",\"qualification_rejections\":"<<qualification_rejections<<",\"peak_payload_bytes\":"<<runtime.memory().peak_bytes<<",\"bad\":0,\"production_queue\":false,\"rows\":"<<rows.str()<<"}\n";if(!f)throw std::runtime_error("cannot write evidence");std::cout<<"opencl_points_passed: words="<<checked<<" guards="<<guards<<" curves="<<curves<<" unqualified="<<qualification_rejections<<" bad=0\n";return 0;
    }catch(const std::exception&e){std::cerr<<"opencl_points_failed: "<<e.what()<<'\n';return 1;}
}
