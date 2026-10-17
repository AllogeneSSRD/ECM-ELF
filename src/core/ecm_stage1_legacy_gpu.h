#pragma once
#include "ecm_stage1_binary.h"
#include "ecm_stage1_checkpoint_runtime.h"
#include <gmp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <functional>

// V5 CUDA ladder snapshots contain N in every lane. Validate it, geometry,
// progress and the original checksum before importing into the new container.
inline bool ecm_stage1_import_legacy_gpu(mpz_srcptr n,mpz_srcptr scalar,uint32_t curves,
                                        uint32_t param,bool fixed,uint64_t requested,
                                        const std::function<bool(uint64_t,uint32_t)> &completed={}) {
    if(ecm_stage1_checkpoint_path().empty())return false;
    std::string existing;
    if(ecm_stage1_read_blob(ecm_stage1_curve_record(0),existing))return false;
    std::vector<char> text(mpz_sizeinbase(n,16)+2);mpz_get_str(text.data(),16,n);
    const std::string nh=text.data();
    auto stem=".ecm_ckpt_"+std::to_string(mpz_sizeinbase(n,2))+"_"+nh.substr(0,8);
    if(nh.size()>8)stem+="_"+nh.substr(nh.size()-8);
    const auto legacy=(std::filesystem::path(ecm_stage1_checkpoint_path()).parent_path()/(stem+".dat")).string();
    for(const auto &candidate:ecm_stage1_checkpoint_candidates(legacy)) {
        std::ifstream file(candidate,std::ios::binary);if(!file)continue;
        std::string header(72,'\0');file.read(header.data(),72);if(!file)continue;
        ecm_stage1_binary::Reader h(header);
        if(h.u32()!=0x45555047u || h.u32()!=5)continue;
        const auto done=h.u64(),total=h.u64();h.u32();const auto count=h.u32();
        const auto sigma=h.u64();const auto bits=h.u32(),tpi=h.u32(),kind=h.u32();h.u32();
        const auto size=h.u64();h.u64();const uint64_t words=param==3?5:7;
        if(count!=curves || kind!=param || total!=mpz_sizeinbase(scalar,2) || done>total ||
           bits<mpz_sizeinbase(n,2)+6 || bits>16384 || bits%32 ||
           (tpi!=4 && tpi!=8 && tpi!=16 && tpi!=32) ||
           size!=words*count*(uint64_t(bits)/8) || !count || sigma>UINT64_MAX-(count-1) ||
           (param!=0 && (sigma>0xffffffffull || count>0x100000000ull-sigma)) ||
           (fixed && sigma!=requested))continue;
        if(completed && completed(sigma,count))continue;
        file.seekg(0,std::ios::end);const auto length=file.tellg();
        if(length!=std::streamoff(72+size+8))continue;
        file.seekg(72);std::string bytes=header;bytes.resize(size_t(72+size+8));
        file.read(bytes.data()+72,std::streamsize(size+8));if(!file)continue;
        ecm_stage1_binary::Reader footer(std::string_view(bytes).substr(bytes.size()-8));
        if(footer.u64()!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,bytes.data(),bytes.size()-8))continue;
        mpz_t lane_n;mpz_init(lane_n);bool same=true;
        for(uint32_t i=0;i<count && same;++i) {
            mpz_import(lane_n,bits/32,-1,4,-1,0,bytes.data()+72+size_t(i)*words*(bits/8));
            same=mpz_cmp(lane_n,n)==0;
        }
        mpz_clear(lane_n);if(!same)continue;
        if(ecm_stage1_import_gpu_payload(bytes,sigma)) {
            std::cout<<"Stage1 legacy GPU checkpoint imported: path="<<candidate
                     <<" progress="<<done<<'/'<<total<<" original_retained=1\n";
            return true;
        }
    }
    return false;
}
