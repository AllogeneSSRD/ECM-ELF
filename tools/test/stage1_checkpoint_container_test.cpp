#include "io/stage1_checkpoint.h"
#include "ecm/montgomery_stage1/cuda/checkpoint.h"
#include "../../src/ecm/edwards_stage1/cuda/checkpoint.h"
#include "../../src/ecm/edwards_stage1/cuda/host_math.h"
#include "ecm/io/save.h"
#include "io/stage1_binary.h"
#include "io/files.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace fs=std::filesystem;
static void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static std::string read(const fs::path &p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
static void write(const fs::path &p,const std::string &s){std::ofstream f(p,std::ios::binary);f.write(s.data(),s.size());check(bool(f),"fixture write");}
static std::string v1(const std::string &id,const std::vector<uint64_t> &sigmas,const std::vector<std::string> &records) {
    std::ostringstream out;out<<"ECM_STAGE1_BATCH_V1\n"<<id<<'\n'<<records.size()<<'\n';
    for(size_t i=0;i<records.size();++i)out<<sigmas[i]<<' '<<records[i].size()<<'\n'<<records[i]<<'\n';
    const auto body=out.str();return body+"CHECKSUM "+std::to_string(ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,body.data(),body.size()))+'\n';
}
static void container(ecm_stage1_checkpoint::Context& checkpoint, const fs::path &dir) {
    check(ecm_stage1_binary::crc32c("123456789")==0xe3069283u,"CRC32C reference vector");
    fs::create_directories(dir);const auto p=(dir/".m0_s1_m347_1000.ckpt").string();
    checkpoint.set_backup_limit(2);checkpoint.set_path(p);
    checkpoint.begin("target|mont0|1000",2,true,26);
    check(checkpoint.sigma(0,26)==26 && checkpoint.sigma(1,27)==27,"sigma initialization");
    const std::string a("a\0\nCHECKSUM ",12),b("b\0second",8);
    check(checkpoint.write_blob(ecm_stage1_curve_record(0),a),"first record");
    check(checkpoint.write_blob(ecm_stage1_curve_record(1),b),"second record");
    check(ecm_stage1_binary::is_batch(read(p)),"writer retained obsolete text container");
    checkpoint.close(false);
    auto damaged=read(p);damaged[damaged.size()-6]^=1;write(p,damaged);
    checkpoint.set_path(p);checkpoint.begin("target|mont0|1000",2,true,26);
    std::string restored;check(checkpoint.read_blob(ecm_stage1_curve_record(0),restored) && restored==a,"CRC fallback");
    check(checkpoint.write_blob(ecm_stage1_curve_record(1),b),"post-recovery write");checkpoint.close(false);
    const auto intact=read(p);
    for(unsigned kind=0;kind<4;++kind) {
        auto malformed=intact;
        if(kind==0)malformed.resize(malformed.size()-3);
        else {
            ecm_stage1_binary::Reader layout(std::string_view(malformed).substr(16));const auto identity_size=layout.u64();
            const auto at=kind==1?size_t(16):kind==2?size_t(24+identity_size+8):size_t(24+identity_size+16);
            if(kind<3)for(unsigned i=0;i<8;++i)malformed[at+i]=char(255);
            else malformed[at]^=1; // Bad record CRC with a valid whole-file CRC.
            malformed.resize(malformed.size()-4);ecm_stage1_binary::u32(malformed,ecm_stage1_binary::crc32c(malformed));
        }
        const auto bad=(dir/("malformed_"+std::to_string(kind)+".ckpt")).string();write(bad,malformed);
        bool rejected=false;checkpoint.set_path(bad);
        try{checkpoint.begin("target|mont0|1000",2,true,26);}catch(const std::runtime_error &){rejected=true;}
        checkpoint.close(false);check(rejected && read(bad)==malformed,"malformed checkpoint accepted or changed");
    }
    for(const auto &trial:{std::string("different|mont0|1000"),std::string("target|mont0|1000")}) {
        bool rejected=false;checkpoint.set_path(p);
        try{checkpoint.begin(trial,2,true,trial[0]=='d'?26:30);}catch(const std::runtime_error &){rejected=true;}
        checkpoint.close(false);check(rejected && read(p)==intact,"identity/sigma rejection");
    }
    // Retain the original V1 bytes as an explicit compatibility fixture.
    const auto old=v1("target|mont0|1000",{26,27},{a,b});write(dir/"legacy_v1.bin",old);write(p,old);
    checkpoint.set_path(p);checkpoint.begin("target|mont0|1000",2,true,26);
    check(checkpoint.read_blob(ecm_stage1_curve_record(1),restored) && restored==b,"V1 read");
    check(checkpoint.write_blob(ecm_stage1_curve_record(1),b),"V1 migration");
    check(ecm_stage1_binary::is_batch(read(p)),"V1 rewrite");checkpoint.close(true);
    check(!fs::exists(p),"completion cleanup");
}
static void edwards(ecm_stage1_checkpoint::Context& checkpoint, const fs::path &dir,const fs::path &audit,bool naf=true) {
    using namespace ed_gpu_driver;fs::create_directories(dir);
    const auto p=(dir/".e_s1_test.ckpt").string();
    ecm_edwards::Integer n,scalar;
    std::string identity,payload;uint32_t count=4;uint64_t b1=1000;
    if(!audit.empty()) {
        std::istringstream source(read(audit));std::string line;std::getline(source,line);check(line=="ECM_STAGE1_BATCH_V1","audit V1");
        std::getline(source,identity);source>>count;uint64_t sigma,size;source>>sigma>>size;check(source.get()=='\n',"audit boundary");
        payload.resize(size);source.read(payload.data(),size);check(bool(source),"audit payload");
        std::istringstream data(payload);std::string tag,nh,sh;unsigned window,doubling;
        data>>tag>>window>>doubling>>nh>>sh>>b1;check(tag=="ECM_GPU_EDWARDS_NAF_V2" && bool(data),"audit codec");
        check(!mpz_set_str(n.v,nh.c_str(),16) && !mpz_set_str(scalar.v,sh.c_str(),16),"audit GMP values");
    } else {
        mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,127);mpz_sub_ui(n.v,n.v,1);
        mpz_set_ui(scalar.v,12345);mpz_setbit(scalar.v,200);identity="small|ed";
        Batch initial(count);for(unsigned i=0;i<count;++i)initial.sigmas[i]=26+i;
        std::string error;check(initialize(initial,n.v,error),"Edwards setup");
        std::ostringstream old;
        if(naf)old<<"ECM_GPU_EDWARDS_NAF_V2\n6 4\n";
        else old<<"ECM_GPU_EDWARDS_BINARY_V1\n";
        old<<hex(n.v)<<'\n'<<hex(scalar.v)<<"\n1000 "<<count<<" 0\n";
        for(unsigned i=0;i<count;++i){old<<initial.sigmas[i]<<' '<<initial.hit[i]<<' '<<hex(initial.factors[i]);for(unsigned j=0;j<9;++j)old<<' '<<hex(initial.state[size_t(i)*9+j]);old<<'\n';}
        payload=old.str();payload+="CHECKSUM "+std::to_string(ecm_io::fingerprint(payload))+'\n';
    }
    std::vector<uint64_t> sigmas(count,0);std::vector<std::string> records(count);records[0]=payload;
    write(p,v1(identity,sigmas,records));checkpoint.set_path(p);checkpoint.begin(identity,count,false,0);
    Batch before(count);check(restore_checkpoint(checkpoint, ecm_stage1_curve_record(0),before,n.v,scalar.v,b1,false,false,false,-1).restored(),"legacy Edwards restore");
    const auto bytes=checkpoint_body(before,n.v,scalar.v,b1);
    check(bytes.size()<payload.size() || audit.empty(),"Edwards compactness");
    check(checkpoint.write_blob(ecm_stage1_curve_record(0),bytes),"Edwards binary write");checkpoint.close(false);
    checkpoint.set_path(p);checkpoint.begin(identity,count,false,0);
    Batch after(count);check(restore_checkpoint(checkpoint, ecm_stage1_curve_record(0),after,n.v,scalar.v,b1,false,false,false,-1).restored(),"binary Edwards restore");
    check(before.done==after.done && before.sigmas==after.sigmas && before.hit==after.hit,"Edwards metadata changed");
    for(size_t i=0;i<count*9;++i)check(mpz_cmp(before.state[i],after.state[i])==0,"Edwards coordinates changed");
    for(size_t i=0;i<count;++i)check(mpz_cmp(before.factors[i],after.factors[i])==0,"Edwards factor changed");
    mpz_add_ui(scalar.v,scalar.v,2);Batch wrong(count);
    check(!restore_checkpoint(checkpoint, ecm_stage1_curve_record(0),wrong,n.v,scalar.v,b1,false,false,false,-1).restored(),"scalar fingerprint did not reject");
    std::cout<<"edwards_payload old_bytes="<<payload.size()<<" new_bytes="<<bytes.size()<<" curves="<<count<<" all_coordinates_equal=1\n";
    checkpoint.close(false);
}
static void deferred_restore(ecm_stage1_checkpoint::Context& checkpoint, const fs::path &dir) {
    using namespace ed_gpu_driver;using namespace ecm_stage1_checkpoint;
    fs::create_directories(dir);const auto p=(dir/"pending.ckpt").string();
    Integer n,scalar;mpz_ui_pow_ui(n.v,2,127);mpz_sub_ui(n.v,n.v,1);mpz_set_ui(scalar.v,12345);
    checkpoint.set_path(p);checkpoint.begin("r5|ed",2,true,26);
    Batch initial(2);for(uint32_t i=0;i<2;++i)initial.sigmas[i]=checkpoint.sigma(i,26+i);
    std::string error;check(initialize(initial,n.v,error),"R5 curve setup");
    check(save_checkpoint(checkpoint, ecm_stage1_curve_record(0),initial,n.v,scalar.v,1000),"R5 valid save");
    const auto valid=checkpoint_body(initial,n.v,scalar.v,1000);
    for(unsigned j=5;j<9;++j)mpz_set_ui(initial.state[j],0);
    check(save_checkpoint(checkpoint, ecm_stage1_curve_record(0),initial,n.v,scalar.v,1000),"R5 bad math fixture");
    checkpoint.close(false);const auto damaged=read(p);
    checkpoint.set_path(p);checkpoint.begin("r5|ed",2,false,0,true);
    uint64_t sigma=999;check(!checkpoint.saved_sigma(0,sigma) && sigma==999,"pending seeds exposed");
    bool rejected=false;try{checkpoint.read_blob(ecm_stage1_curve_record(0),error);}catch(const Error &){rejected=true;}
    check(rejected,"pending payload exposed");unsigned attempts=0;
    checkpoint.validate_pending([&](const std::vector<std::string> &records,std::vector<uint64_t> &sigmas) {
        ++attempts;Batch candidate(2);
        if(!decode_checkpoint(records[0],candidate,n.v,scalar.v,1000,false,false,false,-1).restored())return false;
        sigmas=candidate.sigmas;return true;
    });
    check(attempts==2 && !checkpoint.pending(),"valid mathematical backup not selected");
    check(checkpoint.saved_sigma(0,sigma) && sigma==26,"accepted sigma lost");
    std::string restored;check(checkpoint.read_blob(ecm_stage1_curve_record(0),restored) && restored==valid,"backup payload changed");
    check(read(p)==damaged,"selection overwrote damaged evidence");
    const auto blocker=ecm_stage1_checkpoint_temp_path(p);fs::create_directory(blocker);
    check(!checkpoint.write_blob(ecm_stage1_curve_record(0),"replacement"),"blocked write succeeded");
    check(checkpoint.read_blob(ecm_stage1_curve_record(0),restored) && restored==valid,"failed write changed accepted state");
    check(read(p)==damaged,"failed write changed physical state");fs::remove(blocker);
    check(checkpoint.write_blob(ecm_stage1_curve_record(0),valid),"retry after blocked write");checkpoint.close(false);
    // An archive containing only bad mathematical state must fail, without RNG or overwrite.
    const auto bad=(dir/"bad_only.ckpt").string();write(bad,damaged);
    checkpoint.set_path(bad);checkpoint.begin("r5|ed",2,false,0,true);
    rejected=false;try{checkpoint.validate_pending([](const auto &,auto &){return false;});}
    catch(const Error &e){rejected=e.state==RestoreState::Corrupt;}
    check(rejected && read(bad)==damaged,"invalid checkpoint silently restarted");checkpoint.close(false);
    const auto seed_only=(dir/"seed_only.ckpt").string();checkpoint.set_path(seed_only);
    checkpoint.begin("r5|ed",2,false,0);const auto import_blocker=ecm_stage1_checkpoint_temp_path(seed_only);
    fs::create_directory(import_blocker);rejected=false;
    try{checkpoint.import_payload(valid,std::vector<uint64_t>{26,27});}
    catch(const Error &e){rejected=e.state==RestoreState::IoFailure;}
    check(rejected && !checkpoint.saved_sigma(0,sigma),"failed import committed seeds");
    check(!checkpoint.read_blob(ecm_stage1_curve_record(0),restored),"failed import committed payload");
    fs::remove(import_blocker);check(checkpoint.import_payload(valid,std::vector<uint64_t>{26,27}),"import retry");
    checkpoint.close(false);
    std::cout<<"deferred_restore=PASS math_backup=1 seed_visibility=1 write_rollback=1 import_rollback=1 invalid_only=1 gpu_runs=0\n";
}
static void construction_publication() {
    using namespace ed_gpu_driver;
    Integer n,d,x,y,a24,f;mpz_set_ui(n.v,77);uint64_t factor_sigma=0,degenerate_sigma=0;
    for(uint64_t sigma=1;sigma<200 && (!factor_sigma || !degenerate_sigma);++sigma) {
        const auto setup=construct(d.v,x.v,y.v,a24.v,f.v,sigma,n.v);
        if(setup==Setup::Factor && !factor_sigma)factor_sigma=sigma;
        if(setup==Setup::Degenerate && !degenerate_sigma)degenerate_sigma=sigma;
    }
    check(factor_sigma && degenerate_sigma,"missing Edwards construction fixtures");
    Batch batch(2);batch.sigmas={factor_sigma,degenerate_sigma};unsigned calls=0;std::string error;
    check(!initialize(batch,n.v,error,[&](uint32_t lane,mpz_srcptr value) {
        check(lane==0 && mpz_cmp_ui(value,1)>0 && mpz_cmp(value,n.v)<0 && mpz_divisible_p(n.v,value),"bad construction factor");
        ++calls;return true;
    }),"degenerate later curve accepted");
    check(calls==1 && batch.hit[0] && error.find("degenerate")!=std::string::npos,"earlier Edwards factor not published before failure");
    Batch blocked(1);blocked.sigmas[0]=factor_sigma;
    check(!initialize(blocked,n.v,error,[](uint32_t,mpz_srcptr){return false;}),"Edwards publication failure ignored");
    std::cout<<"edwards_construction_publication=PASS factor_sigma="<<factor_sigma<<" degenerate_sigma="<<degenerate_sigma<<"\n";
}
static void legacy(ecm_stage1_checkpoint::Context& checkpoint, const fs::path &dir,const fs::path &source,const std::string &nh) {
    fs::create_directories(dir);const auto bytes=read(source);check(bytes.size()>80,"legacy source length");
    ecm_stage1_binary::Reader h(bytes);check(h.u32()==0x45555047 && h.u32()==5,"legacy V5");
    h.u64();const auto bits=h.u64();h.u32();const auto curves=h.u32();const auto sigma=h.u64();h.u32();h.u32();const auto param=h.u32();
    ecm_edwards::Integer n,s;check(!mpz_set_str(n.v,nh.c_str(),16),"legacy N");mpz_setbit(s.v,bits-1);
    const auto copy=dir/source.filename();write(copy,bytes);
    const auto p=(dir/".m0_s1_legacy.ckpt").string();checkpoint.set_path(p);checkpoint.begin(nh+"|mont",curves,false,0);
    check(!ecm_stage1_import_legacy_gpu(checkpoint, n.v,s.v,curves,param,true,sigma+1),"legacy sigma mismatch accepted");
    check(!ecm_stage1_import_legacy_gpu(checkpoint, n.v,s.v,curves,param==0?3:0,false,0),"legacy parametrization mismatch accepted");
    auto corrupt=bytes;corrupt[72]^=1;corrupt.resize(corrupt.size()-8);
    ecm_stage1_binary::u64(corrupt,ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,corrupt.data(),corrupt.size()));
    write(copy,corrupt);
    check(!ecm_stage1_import_legacy_gpu(checkpoint, n.v,s.v,curves,param,false,0),"full lane modulus mismatch accepted");
    write(copy,bytes);
    check(ecm_stage1_import_legacy_gpu(checkpoint, n.v,s.v,curves,param,false,0),"legacy V5 import");
    std::string restored;check(checkpoint.read_blob(ecm_stage1_curve_record(0),restored) && restored==bytes,"legacy bytes changed");
    check(checkpoint.sigma(0,1)==sigma,"legacy sigma changed");
    check(read(copy)==bytes,"legacy source overwritten");checkpoint.close(false);
    std::cout<<"legacy_payload bytes="<<bytes.size()<<" curves="<<curves<<" original_equal=1\n";
}
int main(int argc,char **argv) {
    ecm_stage1_checkpoint::Context checkpoint;

    try {
        check(argc>=2,"output required");const fs::path out=argv[1];
        check(!fs::exists(out),"use a fresh test directory");fs::create_directories(out);
        container(checkpoint, out/"container");edwards(checkpoint, out/"ed_small",{});edwards(checkpoint, out/"ed_binary",{},false);deferred_restore(checkpoint, out/"deferred");construction_publication();
        if(argc==5){edwards(checkpoint, out/"ed_audit",argv[2]);legacy(checkpoint, out/"mont_audit",argv[3],argv[4]);}
        std::cout<<"checkpoint_container_checks=PASS\n";return 0;
    }catch(const std::exception &e){checkpoint.close(false);std::cerr<<e.what()<<'\n';return 1;}
}
