#include "ecm_checkpoint.h"
#include "ecm_stage1_checkpoint_runtime.h"

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

namespace {

std::string g_work_dir;

bool is_absolute_path(const char *path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
#ifdef _WIN32
    if ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) {
        return path[1] == ':' && (path[2] == '\\' || path[2] == '/');
    }
#endif
    return path[0] == '/';
}

void trim_trailing_slashes(std::string &s) {
    while (!s.empty() && (s.back() == '/' || s.back() == '\\')) {
        s.pop_back();
    }
}

} // namespace

void opencl_ecm_set_work_dir(const char *dir) {
    if (dir == nullptr || dir[0] == '\0') {
        g_work_dir.clear();
        return;
    }
    g_work_dir = dir;
    trim_trailing_slashes(g_work_dir);
}

const char *opencl_ecm_get_work_dir() {
    return g_work_dir.c_str();
}

std::string opencl_ecm_resolve_data_path_buf(const char *path) {
    if (path == nullptr || path[0] == '\0') {
        return {};
    }
    if (is_absolute_path(path)) {
        return path;
    }
    if (g_work_dir.empty()) {
        return path;
    }
    return g_work_dir + "/" + path;
}

bool opencl_ecm_ensure_parent_dir(const char *filepath) {
    if (filepath == nullptr || filepath[0] == '\0') {
        return false;
    }
    std::string parent;
    const char *slash = std::strrchr(filepath, '/');
#ifdef _WIN32
    const char *bslash = std::strrchr(filepath, '\\');
    if (bslash != nullptr && (slash == nullptr || bslash > slash)) {
        slash = bslash;
    }
#endif
    if (slash == nullptr || slash == filepath) {
        return true;
    }
    parent.assign(filepath, slash);
    if (parent.empty()) {
        return true;
    }

    std::string built;
    size_t i = 0;
    if (parent.size() >= 2 && parent[1] == ':') {
        built = parent.substr(0, 2);
        i = 2;
    } else if (!parent.empty() && parent[0] == '/') {
        built = "/";
        i = 1;
    }

    while (i < parent.size()) {
        while (i < parent.size() && (parent[i] == '/' || parent[i] == '\\')) {
            ++i;
        }
        if (i >= parent.size()) {
            break;
        }
        size_t j = i;
        while (j < parent.size() && parent[j] != '/' && parent[j] != '\\') {
            ++j;
        }
        if (!built.empty() && built.back() != '/') {
            built.push_back('/');
        }
        built.append(parent, i, j - i);
#ifdef _WIN32
        _mkdir(built.c_str());
#else
        mkdir(built.c_str(), 0755);
#endif
        i = j;
    }
    return true;
}

const char *opencl_ecm_checkpoint_filename(const mpz_t N) {
    static std::string name;
    name=ecm_stage1_checkpoint_path().empty()?opencl_ecm_resolve_data_path_buf(".mont_stage1.ckpt"):ecm_stage1_curve_record(0);
    (void)N;return name.c_str();
}

int opencl_ecm_checkpoint_save(const char *filename, const opencl_ecm_checkpoint_header_t *header,
                               const uint32_t *data, size_t data_size) {
    uint64_t sum=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,header,sizeof(*header));
    sum=ecm_stage1_checksum(sum,data,data_size);
    std::string bytes(reinterpret_cast<const char *>(header),sizeof(*header));
    bytes.append(reinterpret_cast<const char *>(data),data_size);
    bytes.append(reinterpret_cast<const char *>(&sum),sizeof(sum));
    if(!ecm_stage1_write_blob(filename,bytes))return ECM_ERROR;
    ecm_ts_fprintf(stderr, "Checkpoint saved: %s s_partial=%llu/%llu (%.1f%%)\n", filename,
                   (unsigned long long)header->s_partial,
                   (unsigned long long)header->s_num_bits,
                   header->s_num_bits ? (100.0 * header->s_partial / header->s_num_bits) : 0.0);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_load(const char *filename, opencl_ecm_checkpoint_header_t *header,
                               uint32_t **data_ptr, size_t *data_size_ptr) {
    std::string bytes;if(!ecm_stage1_read_blob(filename,bytes))return -1;
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

int opencl_ecm_checkpoint_remove(const char *filename) {
    ecm_stage1_remove_checkpoints(filename);
    return 0;
}
