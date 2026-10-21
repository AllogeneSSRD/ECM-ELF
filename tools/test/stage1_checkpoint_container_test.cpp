#include "../../src/ecm/storage/checkpoint/ecm_stage1_legacy_gpu.h"
#include "../../src/ecm/backend/cuda/stage1/edwards/ecm_edwards_gpu_checkpoint.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

namespace fs=std::filesystem;
static void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static std::string read(const fs::path &p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
static void write(const fs::path &p,const std::string &s){std::ofstream f(p,std::ios::binary);f.write(s.data(),s.size());check(bool(f),"fixture write");}
static std::string v1(const std::string &id,const std::vector<uint64_t> &sigmas,const std::vector<std::string> &records) {
    std::ostringstream out;out<<"ECM_STAGE1_BATCH_V1\n"<<id<<'\n'<<records.size()<<'\n';
    for(size_t i=0;i<records.size();++i)out<<sigmas[i]<<' '<<records[i].size()<<'\n'<<records[i]<<'\n';
    const auto body=out.str();return body+"CHECKSUM "+std::to_string(ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,body.data(),body.size()))+'\n';
}
static void container(const fs::path &dir) {
    check(ecm_stage1_binary::crc32c("123456789")==0xe3069283u,"CRC32C reference vector");
    fs::create_directories(dir);const auto p=(dir/".m0_s1_m347_1000.ckpt").string();
    ecm_stage1_set_backup_limit(2);ecm_stage1_set_checkpoint_path(p);
    ecm_stage1_begin_batch("target|mont0|1000",2,true,26);
    check(ecm_stage1_batch_sigma(0,26)==26 && ecm_stage1_batch_sigma(1,27)==27,"sigma initialization");
    const std::string a("a\0\nCHECKSUM ",12),b("b\0second",8);
    check(ecm_stage1_write_blob(ecm_stage1_curve_record(0),a),"first record");
    check(ecm_stage1_write_blob(ecm_stage1_curve_record(1),b),"second record");
    check(ecm_stage1_binary::is_batch(read(p)),"writer retained obsolete text container");
    ecm_stage1_end_batch(false);
    auto damaged=read(p);damaged[damaged.size()-6]^=1;write(p,damaged);
    ecm_stage1_set_checkpoint_path(p);ecm_stage1_begin_batch("target|mont0|1000",2,true,26);
    std::string restored;check(ecm_stage1_read_blob(ecm_stage1_curve_record(0),restored) && restored==a,"CRC fallback");
    check(ecm_stage1_write_blob(ecm_stage1_curve_record(1),b),"post-recovery write");ecm_stage1_end_batch(false);
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
        bool rejected=false;ecm_stage1_set_checkpoint_path(bad);
        try{ecm_stage1_begin_batch("target|mont0|1000",2,true,26);}catch(const std::runtime_error &){rejected=true;}
        ecm_stage1_end_batch(false);check(rejected && read(bad)==malformed,"malformed checkpoint accepted or changed");
    }
    for(const auto &trial:{std::string("different|mont0|1000"),std::string("target|mont0|1000")}) {
        bool rejected=false;ecm_stage1_set_checkpoint_path(p);
        try{ecm_stage1_begin_batch(trial,2,true,trial[0]=='d'?26:30);}catch(const std::runtime_error &){rejected=true;}
        ecm_stage1_end_batch(false);check(rejected && read(p)==intact,"identity/sigma rejection");
    }
    // Retain the original V1 bytes as an explicit compatibility fixture.
    const auto old=v1("target|mont0|1000",{26,27},{a,b});write(dir/"legacy_v1.bin",old);write(p,old);
    ecm_stage1_set_checkpoint_path(p);ecm_stage1_begin_batch("target|mont0|1000",2,true,26);
    check(ecm_stage1_read_blob(ecm_stage1_curve_record(1),restored) && restored==b,"V1 read");
    check(ecm_stage1_write_blob(ecm_stage1_curve_record(1),b),"V1 migration");
    check(ecm_stage1_binary::is_batch(read(p)),"V1 rewrite");ecm_stage1_end_batch(true);
    check(!fs::exists(p),"completion cleanup");
}
static void edwards(const fs::path &dir,const fs::path &audit,bool naf=true) {
    using namespace ed_gpu_driver;fs::create_directories(dir);
    const auto p=(dir/".e_s1_test.ckpt").string();
    ecm_edwards::Integer n,scalar;
    std::string identity,payload;uint32_t count=4;
    if(!audit.empty()) {
        std::istringstream source(read(audit));std::string line;std::getline(source,line);check(line=="ECM_STAGE1_BATCH_V1","audit V1");
        std::getline(source,identity);source>>count;uint64_t sigma,size;source>>sigma>>size;check(source.get()=='\n',"audit boundary");
        payload.resize(size);source.read(payload.data(),size);check(bool(source),"audit payload");
        std::istringstream data(payload);std::string tag,nh,sh;unsigned window,doubling;
        data>>tag>>window>>doubling>>nh>>sh;check(tag=="ECM_GPU_EDWARDS_NAF_V2","audit codec");
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
        payload=old.str();payload+="CHECKSUM "+std::to_string(hash(payload))+'\n';
    }
    const uint64_t b1=audit.empty()?1000:260000000;
    std::vector<uint64_t> sigmas(count,0);std::vector<std::string> records(count);records[0]=payload;
    write(p,v1(identity,sigmas,records));ecm_stage1_set_checkpoint_path(p);ecm_stage1_begin_batch(identity,count,false,0);
    Batch before(count);check(restore(ecm_stage1_curve_record(0),before,n.v,scalar.v,b1,false,false,false,-1),"legacy Edwards restore");
    const auto bytes=checkpoint_body(before,n.v,scalar.v,b1);
    check(bytes.size()<payload.size() || audit.empty(),"Edwards compactness");
    check(ecm_stage1_write_blob(ecm_stage1_curve_record(0),bytes),"Edwards binary write");ecm_stage1_end_batch(false);
    ecm_stage1_set_checkpoint_path(p);ecm_stage1_begin_batch(identity,count,false,0);
    Batch after(count);check(restore(ecm_stage1_curve_record(0),after,n.v,scalar.v,b1,false,false,false,-1),"binary Edwards restore");
    check(before.done==after.done && before.sigmas==after.sigmas && before.hit==after.hit,"Edwards metadata changed");
    for(size_t i=0;i<count*9;++i)check(mpz_cmp(before.state[i],after.state[i])==0,"Edwards coordinates changed");
    for(size_t i=0;i<count;++i)check(mpz_cmp(before.factors[i],after.factors[i])==0,"Edwards factor changed");
    mpz_add_ui(scalar.v,scalar.v,2);Batch wrong(count);
    check(!restore(ecm_stage1_curve_record(0),wrong,n.v,scalar.v,b1,false,false,false,-1),"scalar fingerprint did not reject");
    std::cout<<"edwards_payload old_bytes="<<payload.size()<<" new_bytes="<<bytes.size()<<" curves="<<count<<" all_coordinates_equal=1\n";
    ecm_stage1_end_batch(false);
}
static void legacy(const fs::path &dir,const fs::path &source,const std::string &nh) {
    fs::create_directories(dir);const auto bytes=read(source);check(bytes.size()>80,"legacy source length");
    ecm_stage1_binary::Reader h(bytes);check(h.u32()==0x45555047 && h.u32()==5,"legacy V5");
    h.u64();const auto bits=h.u64();h.u32();const auto curves=h.u32();const auto sigma=h.u64();h.u32();h.u32();const auto param=h.u32();
    ecm_edwards::Integer n,s;check(!mpz_set_str(n.v,nh.c_str(),16),"legacy N");mpz_setbit(s.v,bits-1);
    const auto copy=dir/source.filename();write(copy,bytes);
    const auto p=(dir/".m0_s1_legacy.ckpt").string();ecm_stage1_set_checkpoint_path(p);ecm_stage1_begin_batch(nh+"|mont",curves,false,0);
    check(!ecm_stage1_import_legacy_gpu(n.v,s.v,curves,param,true,sigma+1),"legacy sigma mismatch accepted");
    check(!ecm_stage1_import_legacy_gpu(n.v,s.v,curves,param==0?3:0,false,0),"legacy parametrization mismatch accepted");
    auto corrupt=bytes;corrupt[72]^=1;corrupt.resize(corrupt.size()-8);
    ecm_stage1_binary::u64(corrupt,ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,corrupt.data(),corrupt.size()));
    write(copy,corrupt);
    check(!ecm_stage1_import_legacy_gpu(n.v,s.v,curves,param,false,0),"full lane modulus mismatch accepted");
    write(copy,bytes);
    check(ecm_stage1_import_legacy_gpu(n.v,s.v,curves,param,false,0),"legacy V5 import");
    std::string restored;check(ecm_stage1_read_blob(ecm_stage1_curve_record(0),restored) && restored==bytes,"legacy bytes changed");
    check(ecm_stage1_batch_sigma(0,1)==sigma,"legacy sigma changed");
    check(read(copy)==bytes,"legacy source overwritten");ecm_stage1_end_batch(false);
    std::cout<<"legacy_payload bytes="<<bytes.size()<<" curves="<<curves<<" original_equal=1\n";
}
int main(int argc,char **argv) {
    try {
        check(argc>=2,"output required");const fs::path out=argv[1];
        check(!fs::exists(out),"use a fresh test directory");fs::create_directories(out);
        container(out/"container");edwards(out/"ed_small",{});edwards(out/"ed_binary",{},false);
        if(argc==5){edwards(out/"ed_audit",argv[2]);legacy(out/"mont_audit",argv[3],argv[4]);}
        std::cout<<"checkpoint_container_checks=PASS\n";return 0;
    }catch(const std::exception &e){ecm_stage1_end_batch(false);std::cerr<<e.what()<<'\n';return 1;}
}
