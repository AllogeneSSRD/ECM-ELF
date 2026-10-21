#pragma once
#include <cstdint>
#include <cstddef>
#include <ctime>
#include <string>
#include <vector>
#include <gmp.h>
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
  uint32_t gpu_param;        // 3 = batch parametrization, 0 = Suyama param0
  uint32_t reserved;         // padding, keeps data_size 8-byte aligned
  size_t data_size;          // Size of GPU data
  time_t timestamp;          // When checkpoint was created
} checkpoint_header_t;
struct ResidentCheckpoint {
    uint64_t magic, b1, scalar, modulus, identity, next, total, sigma, bytes, checksum;
    uint32_t version, algorithm, domain, bits, tpi, curves, torsion, reserved;
};

char *get_checkpoint_filename(mpz_srcptr);
int save_checkpoint(const char *,const checkpoint_header_t *,const uint32_t *,size_t);
int restore_checkpoint(const char *,checkpoint_header_t *,uint32_t **,size_t *);
#if ECM_CUDA_ENABLE_PRAC
bool save_checkpoint(const std::string &,ResidentCheckpoint,const std::vector<uint32_t> &);
bool restore_checkpoint(const std::string &,ResidentCheckpoint &,std::vector<uint32_t> &,uint64_t *,bool);
#endif
}
