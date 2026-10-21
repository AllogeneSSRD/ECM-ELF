#include "checkpoint.h"
#include "../../runtime/cuda_ecm_shim.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#if ECM_CUDA_ENABLE_PRAC
#include "prac/ecm_prac_plan.h"
#include <sstream>
#endif
namespace ecm_cuda_mont_host {
/**
 * Get checkpoint filename for given N (more readable format)
 * Uses bit length and first/last few hex chars as identifier
 */
char* get_checkpoint_filename(const mpz_t N) {
  static char filename[1024];
  const auto path=ecm_stage1_checkpoint_path().empty()?std::string(".mont_stage1.ckpt"):ecm_stage1_curve_record(0);
  snprintf(filename,sizeof(filename),"%s",path.c_str());(void)N;return filename;
}

/**
 * Save checkpoint to file
 */
int save_checkpoint(const char *filename,
                    const checkpoint_header_t *header,
                    const uint32_t *data,
                    size_t data_size) {
  uint64_t sum=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,header,sizeof(*header));
  sum=ecm_stage1_checksum(sum,data,data_size);
  std::string bytes(reinterpret_cast<const char *>(header),sizeof(*header));
  bytes.append(reinterpret_cast<const char *>(data),data_size);
  bytes.append(reinterpret_cast<const char *>(&sum),sizeof(sum));
  if(!ecm_stage1_write_blob(filename,bytes))return ECM_ERROR;
  outputf(OUTPUT_VERBOSE, "Checkpoint saved: s_partial=%lu/%lu (%.1f%%)\n",
          header->s_partial, header->s_num_bits,
          100.0 * header->s_partial / header->s_num_bits);
  return ECM_NO_FACTOR_FOUND;
}

/**
 * Load checkpoint from file if it exists
 * Returns 0 if successful, -1 if file doesn't exist or is invalid
 */
int restore_checkpoint(const char *filename,
                    checkpoint_header_t *header,
                    uint32_t **data_ptr,
                    size_t *data_size_ptr) {
  std::string bytes;if(!ecm_stage1_read_blob(filename,bytes))return -1;
  checkpoint_header_t temp_header;
  if(bytes.size()<sizeof(temp_header)+sizeof(uint64_t))return -1;
  memcpy(&temp_header,bytes.data(),sizeof(temp_header));
  if(temp_header.magic!=CHECKPOINT_MAGIC || temp_header.version!=CHECKPOINT_VERSION || !temp_header.data_size ||
     temp_header.data_size>bytes.size()-sizeof(temp_header)-sizeof(uint64_t) ||
     bytes.size()!=sizeof(temp_header)+temp_header.data_size+sizeof(uint64_t))return -1;
  uint64_t sum;memcpy(&sum,bytes.data()+bytes.size()-sizeof(sum),sizeof(sum));
  if(sum!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,bytes.data(),bytes.size()-sizeof(sum)))return -1;
  auto *data=static_cast<uint32_t *>(malloc(temp_header.data_size));if(!data)return -1;
  memcpy(data,bytes.data()+sizeof(temp_header),temp_header.data_size);
  *header = temp_header;
  *data_ptr = data;
  *data_size_ptr = temp_header.data_size;

  time_t now = time(NULL);
  time_t age = now - temp_header.timestamp;
  outputf(OUTPUT_NORMAL, "Checkpoint loaded: s_partial=%lu/%lu (%.1f%%), age=%ld seconds\n",
          temp_header.s_partial, temp_header.s_num_bits,
          100.0 * temp_header.s_partial / temp_header.s_num_bits, age);

  return ECM_NO_FACTOR_FOUND;
}

#if ECM_CUDA_ENABLE_PRAC
bool save_checkpoint(const std::string &path, ResidentCheckpoint h,
                                      const std::vector<uint32_t> &data) {
    h.checksum = ecm_prac_hash(data.data(), data.size() * sizeof(uint32_t));
    std::ostringstream out(std::ios::binary|std::ios::out);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(data.data()), h.bytes);
    return out && ecm_stage1_write_blob(path,out.str());
}
bool restore_checkpoint(const std::string &path,ResidentCheckpoint &h,
                                 std::vector<uint32_t> &data,uint64_t *sigma,bool prac) {
    const auto total=h.total,bytes=h.bytes;
    const auto bits=h.bits,tpi=h.tpi,curves=h.curves,torsion=h.torsion;
    bool resumed=false;
    std::string saved;ecm_stage1_read_blob(path,saved);
    std::istringstream in(saved,std::ios::binary|std::ios::in); ResidentCheckpoint old{};
    if (in.read(reinterpret_cast<char *>(&old), sizeof(old))) {
        bool compatible = old.magic == h.magic && old.version == h.version && old.b1 == h.b1 &&
            old.scalar == h.scalar && old.modulus == h.modulus && old.identity == h.identity &&
            old.total == total && old.bytes == bytes && old.algorithm == h.algorithm && old.domain == 1 &&
            old.bits == bits && old.tpi == tpi && old.curves == curves && old.torsion == torsion &&
            old.next <= total && (prac || old.next >= 1) && old.sigma <= UINT64_MAX - curves;
        if (compatible && in.read(reinterpret_cast<char *>(data.data()), bytes) &&
            in.peek() == std::char_traits<char>::eof() && ecm_prac_hash(data.data(), bytes) == old.checksum) {
            h = old; *sigma = h.sigma; resumed = true;
            outputf(OUTPUT_ALWAYS, "GPU: resident checkpoint resumed (next=%llu/%llu, sigma=%llu)\n",
                (unsigned long long)h.next, (unsigned long long)total, (unsigned long long)*sigma);
        } else throw std::runtime_error("resident checkpoint parameters/payload mismatch");
    }
    if(!saved.empty() && !resumed)throw std::runtime_error("invalid resident checkpoint payload");
    return resumed;
}
#endif
}
