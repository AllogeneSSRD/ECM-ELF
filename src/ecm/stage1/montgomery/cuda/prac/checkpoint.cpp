#include "../checkpoint.h"
#include "../../prac/ecm_prac_plan.h"

#include <cstring>
#include <sstream>

namespace ecm_cuda_mont_host {
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &path, ResidentCheckpoint h,
                                      const std::vector<uint32_t> &data) {
    h.checksum = ecm_prac_hash(data.data(), data.size() * sizeof(uint32_t));
    std::ostringstream out(std::ios::binary|std::ios::out);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(data.data()), h.bytes);
    return out && checkpoint.write_blob(path,out.str());
}
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &saved,mpz_srcptr n,
    ResidentCheckpoint &h,std::vector<uint32_t> &data,bool prac) {
    using namespace ecm_stage1_checkpoint;
    if(saved.size()<sizeof(h))return {RestoreState::Corrupt,"short resident checkpoint"};
    ResidentCheckpoint old{};memcpy(&old,saved.data(),sizeof(old));
    if(old.magic!=h.magic || old.version!=h.version || old.b1!=h.b1 || old.scalar!=h.scalar ||
       old.modulus!=h.modulus || old.identity!=h.identity || old.total!=h.total ||
       old.algorithm!=h.algorithm || old.domain!=1 || old.bits!=h.bits || old.tpi!=h.tpi ||
       old.curves!=h.curves || old.torsion!=h.torsion)
        return {RestoreState::Incompatible,"resident checkpoint request/layout mismatch"};
    if(old.bytes!=h.bytes || saved.size()-sizeof(old)!=old.bytes || old.next>h.total ||
       (!prac && old.next<1) || old.sigma>UINT64_MAX-(h.curves-1ull) ||
       ecm_prac_hash(saved.data()+sizeof(old),size_t(old.bytes))!=old.checksum)
        return {RestoreState::Corrupt,"invalid resident payload/cursor"};
    std::vector<uint32_t> decoded(size_t(old.bytes)/4);
    memcpy(decoded.data(),saved.data()+sizeof(old),size_t(old.bytes));
    mpz_t value;mpz_init(value);bool valid=true;const auto limbs=h.bits/32;
    for(uint32_t i=0;i<h.curves && valid;++i)for(uint32_t j=0;j<7;++j) {
        mpz_import(value,limbs,-1,4,0,0,decoded.data()+(size_t(i)*7+j)*limbs);
        if(j==0?mpz_cmp(value,n)!=0:mpz_cmp(value,n)>=0){valid=false;break;}
    }
    mpz_clear(value);
    if(!valid)return {RestoreState::Corrupt,"resident modulus/residue mismatch"};
    h=old;data.swap(decoded);
    return {old.next==old.total?RestoreState::AlreadyComputed:RestoreState::Resumed,{}};
}
}
