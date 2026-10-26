#include "pari_native.h"
#include <windows.h>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace ecm_diagnostics {
namespace g=ecm_group_order;
namespace {
using Word=int64_t;
using Gen=Word*;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
// The pinned upstream Win64 ABI deliberately uses 64-bit words (including its
// C `long` parameters). MSVC long is 32-bit: do not declare these as `long`.
size_t length(Gen value){if(!value)throw std::runtime_error("Null PARI value");return uint64_t(value[0])&((uint64_t(1)<<56)-1);}
unsigned type(Gen value){if(!value)throw std::runtime_error("Null PARI value");return unsigned(uint64_t(value[0])>>57);}
Gen item(Gen value,size_t index){if(index==0||index>=length(value))throw std::runtime_error("PARI vector index");return reinterpret_cast<Gen>(value[index]);}
struct Deadline:std::runtime_error{Deadline():std::runtime_error("native five-second budget exhausted") {}};
}
struct NativeOrder::Library {
    HMODULE module=nullptr;bool initialized=false;std::string error;
    void (*init)(size_t,uint64_t)=nullptr;
    void (*close)()=nullptr;
    Gen (*version)()=nullptr;
    Gen (*integer)(const char*)=nullptr;
    char* (*string)(Gen)=nullptr;
    void (*release)(void*)=nullptr;
    Gen (*vec2)(Gen,Gen)=nullptr;
    Gen (*curve)(Gen,Gen,Word)=nullptr;
    Gen (*card)(Gen,Gen)=nullptr;
    Gen (*order)(Gen,Gen,Gen)=nullptr;
    Gen (*factor)(Gen,Word)=nullptr;
    Word (*prime)(Gen)=nullptr;
    Gen (*data)(const char*,Word)=nullptr;
    template<class T>void bind(T& function,const char*name){function=reinterpret_cast<T>(GetProcAddress(module,name));if(!function)throw std::runtime_error(std::string("Missing PARI export: ")+name);}
    explicit Library(const std::filesystem::path&directory){
        try{
            const auto dll=directory/L"libpari.dll",tables=directory/L"pari-data"/L"seadata"/L"sea0";
            if(!std::filesystem::is_regular_file(dll)||!std::filesystem::is_regular_file(tables))throw std::runtime_error("Local PARI library or SEA data missing");
            module=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if(!module)throw std::runtime_error("Cannot load local PARI library (Windows "+std::to_string(GetLastError())+")");
            bind(init,"pari_init");bind(close,"pari_close");bind(version,"pari_version");bind(integer,"strtoi");bind(string,"GENtostr");bind(release,"pari_free");
            bind(vec2,"mkvec2");bind(curve,"ellinit");bind(card,"ellcard");bind(order,"ellorder");bind(factor,"factorint");bind(prime,"isprime");bind(data,"sd_datadir");
            init(128*1024*1024,500000);initialized=true;
            // Version and word-layout check before traversing any math result.
            if(text(version())!="[2, 17, 3]")throw std::runtime_error("Unsupported local PARI ABI (requires Win64 2.17.3)");
            const auto path=(directory/L"pari-data").u8string();data(path.c_str(),0);
        }catch(const std::exception&e){error=e.what();}
    }
    ~Library(){if(initialized)close();if(module)FreeLibrary(module);}
    std::string text(Gen value){auto*raw=string(value);if(!raw)throw std::runtime_error("Cannot format PARI result");std::string out(raw);release(raw);return out;}
    Gen number(const g::Integer& value){return integer(value.str().c_str());}
    g::Integer number(Gen value){if(type(value)!=1)throw std::runtime_error("Expected PARI integer");return g::Integer(text(value));}
    static void check(Clock::time_point start){if(elapsed(start)>=5000)throw Deadline();}
    g::Factorization parts(Gen matrix,const g::Integer& input,Clock::time_point start,const NativeOrder::FactorCheckpoint&checkpoint){
        if(type(matrix)!=19||length(matrix)!=3)throw std::runtime_error("Invalid PARI factor matrix");
        auto primes=item(matrix,1),exponents=item(matrix,2);
        if(type(primes)!=18||type(exponents)!=18||length(primes)!=length(exponents)||length(primes)>4097)throw std::runtime_error("Invalid PARI factor columns");
        g::Factorization out;out.cofactor=input;out.status=g::Status::Partial;
        for(size_t i=1;i<length(primes);++i){
            check(start);auto p=number(item(primes,i)),e=number(item(exponents,i));
            if(mpz_cmp_ui(p.v,2)<0||!mpz_fits_ulong_p(e.v)||mpz_cmp_ui(e.v,1)<0||mpz_cmp_ui(e.v,4096)>0)throw std::runtime_error("Invalid PARI factor power");
            const auto exponent=uint32_t(mpz_get_ui(e.v));
            if(!out.factors.empty()&&mpz_cmp(out.factors.back().prime.v,p.v)>=0)throw std::runtime_error("Unordered PARI factors");
            // factorint alone only promises probable primes. Call the actual
            // proof routine before marking a factor Proven or consuming it.
            if(!prime(item(primes,i)))throw std::runtime_error("Composite PARI factor");check(start);
            g::Integer divisor;mpz_pow_ui(divisor.v,p.v,exponent);
            if(!mpz_divisible_p(out.cofactor.v,divisor.v))throw std::runtime_error("PARI factor product mismatch");
            mpz_divexact(out.cofactor.v,out.cofactor.v,divisor.v);out.factors.push_back({std::move(p),exponent,g::Primality::Proven});
            if(checkpoint)checkpoint(out);
        }
        if(mpz_cmp_ui(out.cofactor.v,1))throw std::runtime_error("Incomplete PARI factor product");
        out.status=g::Status::Complete;out.all_proven=true;return out;
    }
};
NativeOrder::NativeOrder(const std::filesystem::path& directory):library_(std::make_unique<Library>(directory)){}
NativeOrder::~NativeOrder()=default;
bool NativeOrder::available()const{return library_->error.empty();}
const std::string& NativeOrder::failure()const{return library_->error;}
g::Factorization NativeOrder::factor(const g::Integer&input,const FactorCheckpoint& checkpoint){
    g::Factorization out;out.cofactor=input;out.status=g::Status::Partial;const auto start=Clock::now();auto&lib=*library_;
    try{
        if(!available())throw std::runtime_error(failure());
        if(mpz_cmp_ui(input.v,1)<0||mpz_sizeinbase(input.v,2)>4096)throw std::invalid_argument("Input must be a positive integer of at most 4096 bits");
        if(checkpoint)checkpoint(out);
        // Expose cheap factors before entering an indivisible PARI operation.
        // The helper's hard deadline can then preserve useful prime components.
        g::Limits cheap;cheap.milliseconds=50;cheap.rho_iterations=8192;
        out=g::factor_integer(input,cheap);out.status=g::Status::Partial;out.all_proven=false;
        for(auto&part:out.factors){Library::check(start);if(!lib.prime(lib.number(part.prime)))throw std::runtime_error("Composite seed factor");part.primality=g::Primality::Proven;}
        out.reason="native integer factorization in progress";if(checkpoint)checkpoint(out);
        if(mpz_cmp_ui(out.cofactor.v,1)>0){
            const auto known=out.factors;const auto remaining=out.cofactor;
            auto matrix=lib.factor(lib.number(remaining),0);
            const auto merge=[&](const g::Factorization&tail){out=tail;out.factors.insert(out.factors.end(),known.begin(),known.end());std::sort(out.factors.begin(),out.factors.end(),[](const auto&a,const auto&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});if(checkpoint)checkpoint(out);};
            const auto tail=lib.parts(matrix,remaining,start,merge);merge(tail);
        }
        out.status=g::Status::Complete;out.all_proven=true;out.reason.clear();
    }catch(const std::invalid_argument&e){out.status=g::Status::InvalidInput;out.reason=e.what();}
    catch(const std::exception&e){out.reason=e.what();}
    out.statistics.total_ms=out.statistics.factor_ms=elapsed(start);return out;
}
g::Analysis NativeOrder::group(const g::CurveRequest& request,const GroupCheckpoint& checkpoint){
    g::Analysis out;out.status=g::Status::ResourceLimit;out.reason="native primality proof in progress";out.modulus_primality=g::classify_prime(request.prime);
    const auto start=Clock::now();auto&lib=*library_;
    const auto publish=[&](){out.statistics.total_ms=elapsed(start);if(checkpoint)checkpoint(out);};
    try{
        if(!available())throw std::runtime_error(failure());
        if(mpz_cmp_ui(request.prime.v,3)<=0||out.modulus_primality==g::Primality::Composite)throw std::invalid_argument("modulus must be a prime >3");
        if(mpz_sizeinbase(request.prime.v,2)>192)throw std::runtime_error("Local SEA supports moduli through 192 bits");
        publish();auto p=lib.number(request.prime);if(!lib.prime(p))throw std::invalid_argument("composite modulus");Library::check(start);
        out.modulus_primality=g::Primality::Proven;
        const auto construction=Clock::now();const auto model=g::construct_curve(request);
        auto E=lib.curve(lib.vec2(lib.number(model.a),lib.number(model.b)),p,64);
        if(length(E)<5)throw std::invalid_argument("singular curve");out.statistics.construction_ms=elapsed(construction);
        out.reason="native SEA count in progress";publish();const auto counting=Clock::now();
        auto n=lib.card(E,nullptr);out.group_order=lib.number(n);Library::check(start);out.statistics.count_ms=elapsed(counting);
        g::Integer delta,four_p;mpz_sub(delta.v,out.group_order.v,request.prime.v);mpz_sub_ui(delta.v,delta.v,1);mpz_mul(delta.v,delta.v,delta.v);mpz_mul_ui(four_p.v,request.prime.v,4);
        if(mpz_cmp_ui(out.group_order.v,1)<0||mpz_cmp(delta.v,four_p.v)>0)throw std::runtime_error("Native SEA group violates Hasse bound");
        out.group_order_computed=true;out.multiple=out.group_order;out.multiple_factorization.cofactor=out.group_order;out.multiple_factorization.status=g::Status::Partial;
        out.reason="native group factorization in progress";publish();const auto factoring=Clock::now();
        auto f=lib.factor(n,0);out.multiple_factorization=lib.parts(f,out.group_order,start,[&](const auto&value){out.multiple_factorization=value;publish();});
        out.statistics.factor_ms=elapsed(factoring);out.reason="native point order in progress";publish();const auto pointing=Clock::now();
        auto P=model.point.infinity?lib.integer("0"):lib.vec2(lib.number(model.point.x),lib.number(model.point.y));
        // PARI represents the point at infinity by a one-element vector [0].
        if(model.point.infinity){out.point_order=g::Integer(1);}
        else out.point_order=lib.number(lib.order(E,P,lib.vec2(n,f)));
        Library::check(start);
        if(mpz_cmp_ui(out.point_order.v,1)<0||!mpz_divisible_p(out.group_order.v,out.point_order.v)||!g::multiply(model,model.point,out.point_order).infinity)throw std::runtime_error("Native point order validation failed");
        for(const auto&factor:out.multiple_factorization.factors){
            Library::check(start);g::Integer work=out.point_order;uint32_t exponent=0;
            while(mpz_divisible_p(work.v,factor.prime.v)){mpz_divexact(work.v,work.v,factor.prime.v);++exponent;}
            if(exponent){
                g::Integer reduced;mpz_divexact(reduced.v,out.point_order.v,factor.prime.v);
                if(g::multiply(model,model.point,reduced).infinity)throw std::runtime_error("Native point minimality validation failed");
                out.point_factorization.push_back({factor.prime,exponent,g::Primality::Proven});
            }
        }
        Library::check(start);out.statistics.point_ms=elapsed(pointing);out.point_minimality_checked=true;out.all_proven=true;out.status=g::Status::Complete;out.reason.clear();
    }catch(const std::invalid_argument&e){out.status=g::Status::InvalidInput;out.reason=e.what();}
    catch(const std::exception&e){out.reason=e.what();}
    out.statistics.total_ms=elapsed(start);return out;
}
}
