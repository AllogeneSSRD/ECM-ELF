#pragma once
#include <cstdint>
#include <cstddef>
#include <ctime>
#include <string>
#include <vector>
#include <gmp.h>
#include "io/stage1_checkpoint.h"
#define CHECKPOINT_MAGIC 0x45555047
#define CHECKPOINT_VERSION 5
namespace ecm_cuda_mont_host {
typedef struct {
  uint32_t magic;            // Magic number for validation
  uint32_t version;          // Checkpoint format version (5)
  uint64_t s_partial;        // Current bit progress
  uint64_t s_num_bits;       // Total bits to process
  int32_t batches_complete;  // Number of completed batches
  uint32_t curves;           // Number of curves
  uint64_t sigma;            // Starting sigma (full 64 bits since v4)
  uint32_t BITS;             // Kernel bit size
  uint32_t TPI;              // Threads per instance (needed for kernel selection)
  uint32_t gpu_param;        // 0 = Suyama, 2 = batch-2, 3 = batch
  uint32_t reserved;         // padding, keeps data_size 8-byte aligned
  size_t data_size;          // Size of GPU data
  time_t timestamp;          // When checkpoint was created
} checkpoint_header_t;
struct ResidentCheckpoint {
    uint64_t magic, b1, scalar, modulus, identity, next, total, sigma, bytes, checksum;
    uint32_t version, algorithm, domain, bits, tpi, curves, torsion, reserved;
};

std::string checkpoint_record(ecm_stage1_checkpoint::Context& checkpoint);
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const char *,const checkpoint_header_t *,const uint32_t *,size_t);
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &,mpz_srcptr,
    uint64_t scalar_bits,uint32_t curves,uint32_t param,checkpoint_header_t &,std::vector<uint32_t> &);
#if ECM_CUDA_ENABLE_PRAC
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &,ResidentCheckpoint,const std::vector<uint32_t> &);
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &,mpz_srcptr,
    ResidentCheckpoint &,std::vector<uint32_t> &,bool prac);
#endif
}
