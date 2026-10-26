#include "checkpoint.h"
#include "construction.h"
#include "../../../cuda/runtime/cuda_ecm_shim.h"
#include "../../../../io/stage1_checkpoint.h"
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "../../../../io/stage1_binary.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#if ECM_CUDA_ENABLE_PRAC
#include "prac/ecm_prac_plan.h"
#include <sstream>
#endif
namespace ecm_cuda_mont_host {
/**
 * Get checkpoint filename for given N (more readable format)
 * Uses bit length and first/last few hex chars as identifier
 */
std::string checkpoint_record(ecm_stage1_checkpoint::Context& checkpoint) {
  return checkpoint.path().empty()?std::string{}:ecm_stage1_curve_record(0);
}

/**
 * Save checkpoint to file
 */
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const char *filename,
                    const checkpoint_header_t *header,
                    const uint32_t *data,
                    size_t data_size) {
  uint64_t sum=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,header,sizeof(*header));
  sum=ecm_stage1_checksum(sum,data,data_size);
  std::string bytes(reinterpret_cast<const char *>(header),sizeof(*header));
  bytes.append(reinterpret_cast<const char *>(data),data_size);
  bytes.append(reinterpret_cast<const char *>(&sum),sizeof(sum));
  if(!checkpoint.write_blob(filename,bytes))return false;
  outputf(OUTPUT_VERBOSE, "Checkpoint saved: s_partial=%lu/%lu (%.1f%%)\n",
          header->s_partial, header->s_num_bits,
          100.0 * header->s_partial / header->s_num_bits);
  return true;
}

ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &bytes,mpz_srcptr n,
    uint64_t scalar_bits,uint32_t curves,uint32_t param,checkpoint_header_t &header,
    std::vector<uint32_t> &data) {
  using namespace ecm_stage1_checkpoint;
  if(bytes.size()<sizeof(header)+sizeof(uint64_t))return {RestoreState::Corrupt,"short Montgomery checkpoint"};
  if(param!=0 && param!=2 && param!=3)return {RestoreState::Incompatible,"unsupported Montgomery parametrization"};
  const auto layout=ladder_construction(param);
  checkpoint_header_t h;memcpy(&h,bytes.data(),sizeof(h));
  if(h.magic!=CHECKPOINT_MAGIC || h.version!=CHECKPOINT_VERSION ||
     !h.data_size || h.data_size!=bytes.size()-sizeof(h)-sizeof(uint64_t))
      return {RestoreState::Corrupt,"invalid Montgomery header or length"};
  uint64_t sum;memcpy(&sum,bytes.data()+bytes.size()-sizeof(sum),sizeof(sum));
  if(sum!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,bytes.data(),bytes.size()-sizeof(sum)))
      return {RestoreState::Corrupt,"Montgomery checksum mismatch"};
  if(h.curves!=curves || h.s_num_bits!=scalar_bits || h.gpu_param!=param)
      return {RestoreState::Incompatible,"Montgomery checkpoint request mismatch"};
  if(h.BITS<mpz_sizeinbase(n,2)+6 || h.BITS>16384 || h.BITS%32 ||
     (h.TPI!=4 && h.TPI!=8 && h.TPI!=16 && h.TPI!=32) ||
     h.data_size!=uint64_t(layout.words)*curves*(h.BITS/8) ||
     !h.s_partial || h.s_partial>scalar_bits || h.batches_complete<0 ||
     h.sigma>UINT64_MAX-(curves-1ull) ||
     (param!=0 && (h.sigma>UINT32_MAX || curves>0x100000000ull-h.sigma)))
      return {RestoreState::Corrupt,"invalid Montgomery geometry, cursor or sigma"};
  std::vector<uint32_t> decoded(h.data_size/sizeof(uint32_t));
  memcpy(decoded.data(),bytes.data()+sizeof(h),h.data_size);
  const uint32_t words=layout.words,limbs=h.BITS/32;
  mpz_t value;mpz_init(value);bool valid=true;
  for(uint32_t i=0;i<curves && valid;++i)for(uint32_t j=0;j<words;++j) {
      mpz_import(value,limbs,-1,sizeof(uint32_t),0,0,decoded.data()+(size_t(i)*words+j)*limbs);
      if(j==0?mpz_cmp(value,n)!=0:mpz_cmp(value,n)>=0){valid=false;break;}
  }
  mpz_clear(value);
  if(!valid)return {RestoreState::Corrupt,"Montgomery modulus/residue mismatch"};
  header=h;data.swap(decoded);
  return {h.s_partial==scalar_bits?RestoreState::AlreadyComputed:RestoreState::Resumed,{}};
}

#if ECM_CUDA_ENABLE_PRAC
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
#endif
}

// V5 CUDA ladder snapshots contain N in every lane. Validate it, geometry,
// progress and the original checksum before importing into the new container.
bool ecm_stage1_import_legacy_gpu(ecm_stage1_checkpoint::Context &checkpoint, mpz_srcptr n,
                                  mpz_srcptr scalar, uint32_t curves, uint32_t param, bool fixed,
                                  uint64_t requested,
                                  const std::function<bool(uint64_t, uint32_t)> &completed) {
    if (checkpoint.path().empty() || checkpoint.pending())
        return false;
    std::string existing;
    if (checkpoint.read_blob(ecm_stage1_curve_record(0), existing))
        return false;
    std::vector<char> text(mpz_sizeinbase(n, 16) + 2);
    mpz_get_str(text.data(), 16, n);
    const std::string nh = text.data();
    auto stem = ".ecm_ckpt_" + std::to_string(mpz_sizeinbase(n, 2)) + "_" + nh.substr(0, 8);
    if (nh.size() > 8)
        stem += "_" + nh.substr(nh.size() - 8);
    const auto legacy =
        (std::filesystem::path(checkpoint.path()).parent_path() / (stem + ".dat")).string();
    for (const auto &candidate : checkpoint.candidates(legacy)) {
        std::ifstream file(candidate, std::ios::binary);
        if (!file)
            continue;
        std::string header(72, '\0');
        file.read(header.data(), 72);
        if (!file)
            continue;
        ecm_stage1_binary::Reader h(header);
        if (h.u32() != 0x45555047u || h.u32() != 5)
            continue;
        const auto done = h.u64(), total = h.u64();
        h.u32();
        const auto count = h.u32();
        const auto sigma = h.u64();
        const auto bits = h.u32(), tpi = h.u32(), kind = h.u32();
        h.u32();
        const auto size = h.u64();
        h.u64();
        const uint64_t words = param == 3 ? 5 : 7;
        if (count != curves || kind != param || total != mpz_sizeinbase(scalar, 2) ||
            done > total || bits < mpz_sizeinbase(n, 2) + 6 || bits > 16384 || bits % 32 ||
            (tpi != 4 && tpi != 8 && tpi != 16 && tpi != 32) ||
            size != words * count * (uint64_t(bits) / 8) || !count ||
            sigma > UINT64_MAX - (count - 1) ||
            (param != 0 && (sigma > 0xffffffffull || count > 0x100000000ull - sigma)) ||
            (fixed && sigma != requested))
            continue;
        if (completed && completed(sigma, count))
            continue;
        file.seekg(0, std::ios::end);
        const auto length = file.tellg();
        if (length != std::streamoff(72 + size + 8))
            continue;
        file.seekg(72);
        std::string bytes = header;
        bytes.resize(size_t(72 + size + 8));
        file.read(bytes.data() + 72, std::streamsize(size + 8));
        if (!file)
            continue;
        ecm_stage1_binary::Reader footer(std::string_view(bytes).substr(bytes.size() - 8));
        if (footer.u64() !=
            ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED, bytes.data(), bytes.size() - 8))
            continue;
        mpz_t lane_n;
        mpz_init(lane_n);
        bool same = true;
        for (uint32_t i = 0; i < count && same; ++i) {
            mpz_import(lane_n, bits / 32, -1, 4, -1, 0,
                       bytes.data() + 72 + size_t(i) * words * (bits / 8));
            same = mpz_cmp(lane_n, n) == 0;
        }
        mpz_clear(lane_n);
        if (!same)
            continue;
        if (checkpoint.import_payload(bytes, sigma)) {
            std::cout << "Stage1 legacy GPU checkpoint imported: path=" << candidate
                      << " progress=" << done << '/' << total << " original_retained=1\n";
            return true;
        }
    }
    return false;
}
