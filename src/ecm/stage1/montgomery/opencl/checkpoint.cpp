#include "ecm/stage1/montgomery/opencl/checkpoint.h"
#include "../../../../io/stage1_checkpoint.h"

#include "ecm/platform/logging.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <filesystem>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "../../../../io/files.h"

std::string opencl_ecm_checkpoint_filename(ecm_stage1_checkpoint::Context& checkpoint, const mpz_t N) {
    (void)N;
    return checkpoint.path().empty() ? ecm_io::resolve_data_path(".mont_stage1.ckpt") : ecm_stage1_curve_record(0);
}

int opencl_ecm_checkpoint_save(ecm_stage1_checkpoint::Context& checkpoint, const char *filename, const opencl_ecm_checkpoint_header_t *header,
                               const uint32_t *data, size_t data_size) {
    uint64_t sum=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,header,sizeof(*header));
    sum=ecm_stage1_checksum(sum,data,data_size);
    std::string bytes(reinterpret_cast<const char *>(header),sizeof(*header));
    bytes.append(reinterpret_cast<const char *>(data),data_size);
    bytes.append(reinterpret_cast<const char *>(&sum),sizeof(sum));
    if(!checkpoint.write_blob(filename,bytes))return ECM_ERROR;
    ecm_ts_fprintf(stderr, "Checkpoint saved: %s s_partial=%llu/%llu (%.1f%%)\n", filename,
                   (unsigned long long)header->s_partial,
                   (unsigned long long)header->s_num_bits,
                   header->s_num_bits ? (100.0 * header->s_partial / header->s_num_bits) : 0.0);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_load(ecm_stage1_checkpoint::Context& checkpoint, const char *filename, opencl_ecm_checkpoint_header_t *header,
                               uint32_t **data_ptr, size_t *data_size_ptr) {
    std::string bytes;if(!checkpoint.read_blob(filename,bytes))return -1;
    opencl_ecm_checkpoint_header_t temp_header{};
    if(bytes.size()<sizeof(temp_header)+sizeof(uint64_t))return -1;
    memcpy(&temp_header,bytes.data(),sizeof(temp_header));
    if(temp_header.magic!=OPENCL_ECM_CHECKPOINT_MAGIC || temp_header.version!=OPENCL_ECM_CHECKPOINT_VERSION || !temp_header.data_size ||
       temp_header.data_size>bytes.size()-sizeof(temp_header)-sizeof(uint64_t) ||
       bytes.size()!=sizeof(temp_header)+temp_header.data_size+sizeof(uint64_t))return -1;
    uint64_t sum;memcpy(&sum,bytes.data()+bytes.size()-sizeof(sum),sizeof(sum));
    if(sum!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,bytes.data(),bytes.size()-sizeof(sum)))return -1;
    auto *data=static_cast<uint32_t *>(malloc(static_cast<size_t>(temp_header.data_size)));if(!data)return -1;
    memcpy(data,bytes.data()+sizeof(temp_header),static_cast<size_t>(temp_header.data_size));
    *header = temp_header;
    *data_ptr = data;
    *data_size_ptr = (size_t)temp_header.data_size;
    const time_t now = time(nullptr);
    const time_t age = now - (time_t)temp_header.timestamp;
    ecm_ts_fprintf(stderr,
                   "Checkpoint loaded: s_partial=%llu/%llu (%.1f%%), age=%lld seconds\n",
                   (unsigned long long)temp_header.s_partial,
                   (unsigned long long)temp_header.s_num_bits,
                   temp_header.s_num_bits ? (100.0 * temp_header.s_partial / temp_header.s_num_bits)
                                          : 0.0,
                   (long long)age);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_remove(ecm_stage1_checkpoint::Context& checkpoint, const char *filename) {
    checkpoint.remove(filename);
    return 0;
}
