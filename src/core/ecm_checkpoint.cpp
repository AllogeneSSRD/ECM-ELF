#include "ecm_checkpoint.h"
#include "ecm_stage1_checkpoint_runtime.h"

#include "opencl_ecm_log.h"

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
    static char filename[1024];
    const size_t nbits = mpz_sizeinbase(N, 2);
    char *N_str = mpz_get_str(nullptr, 16, N);
    if (!N_str) {
        const std::string resolved =
            opencl_ecm_resolve_data_path_buf(".ecm_ckpt_alloc_fail.dat");
        std::snprintf(filename, sizeof(filename), "%s", resolved.c_str());
        return filename;
    }
    const size_t len = strlen(N_str);
    char first_hex[16] = {0};
    char last_hex[16] = {0};
    // Use memcpy to avoid MSVC Debug CRT strncpy /GS interaction with
    // GMP-allocated N_str (GMP uses its own allocator; CRT secure-strncpy
    // may probe the buffer length through CRT heap metadata and crash).
    // This fix is not needed for Release CRT (/MD) builds.
    strncpy(first_hex, N_str, (len >= 8u) ? 8u : len);
    // std::memcpy(first_hex, N_str, (len >= 8u) ? 8u : len);
    if (len > 8u) {
        strncpy(last_hex, N_str + len - 8u, 8u);
        // std::memcpy(last_hex, N_str + len - 8u, 8u);
    }

    char basename[512];
    if (len > 8u && last_hex[0] != '\0') {
        std::snprintf(basename, sizeof(basename), ".ecm_ckpt_%zu_%s_%s.dat", nbits, first_hex,
                      last_hex);
    } else {
        std::snprintf(basename, sizeof(basename), ".ecm_ckpt_%zu_%s.dat", nbits, first_hex);
    }
    free(N_str);

    const std::string resolved = opencl_ecm_resolve_data_path_buf(basename);
    std::snprintf(filename, sizeof(filename), "%s", resolved.c_str());
    return filename;
}

int opencl_ecm_checkpoint_save(const char *filename, const opencl_ecm_checkpoint_header_t *header,
                               const uint32_t *data, size_t data_size) {
    if (!opencl_ecm_ensure_parent_dir(filename)) {
        ecm_ts_fprintf(stderr, "Warning: Could not create directory for checkpoint '%s'\n",
                       filename);
        return ECM_ERROR;
    }
    const std::string temp = ecm_stage1_checkpoint_temp_path(filename);
    FILE *f = fopen(temp.c_str(), "wb");
    if (!f) {
        ecm_ts_fprintf(stderr, "Warning: Could not open checkpoint file '%s' for writing (%s)\n",
                       filename, std::strerror(errno));
        return ECM_ERROR;
    }
    uint64_t sum = ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED, header, sizeof(*header));
    sum = ecm_stage1_checksum(sum, data, data_size);
    const bool written = fwrite(header, sizeof(*header), 1, f) == 1 &&
                         fwrite(data, 1, data_size, f) == data_size &&
                         fwrite(&sum, sizeof(sum), 1, f) == 1;
    const bool closed = fclose(f) == 0;
    if (!written || !closed || !ecm_stage1_commit_checkpoint(filename, temp)) {
        ecm_ts_fprintf(stderr, "Error writing or rotating checkpoint '%s'\n", filename);
        remove(temp.c_str());
        return ECM_ERROR;
    }
    ecm_ts_fprintf(stderr, "Checkpoint saved: %s s_partial=%llu/%llu (%.1f%%)\n", filename,
                   (unsigned long long)header->s_partial,
                   (unsigned long long)header->s_num_bits,
                   header->s_num_bits ? (100.0 * header->s_partial / header->s_num_bits) : 0.0);
    return ECM_NO_FACTOR_FOUND;
}

int opencl_ecm_checkpoint_load(const char *filename, opencl_ecm_checkpoint_header_t *header,
                               uint32_t **data_ptr, size_t *data_size_ptr) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        return -1;
    }
    opencl_ecm_checkpoint_header_t temp_header{};
    if (fread(&temp_header, sizeof(opencl_ecm_checkpoint_header_t), 1, f) != 1) {
        ecm_ts_fprintf(stderr, "Warning: Could not read checkpoint header\n");
        fclose(f);
        return -1;
    }
    if (temp_header.magic != OPENCL_ECM_CHECKPOINT_MAGIC) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint file has invalid magic number\n");
        fclose(f);
        return -1;
    }
    if (temp_header.version != OPENCL_ECM_CHECKPOINT_VERSION) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint version mismatch (expected %d, got %u)\n",
                       OPENCL_ECM_CHECKPOINT_VERSION, temp_header.version);
        fclose(f);
        return -1;
    }
    if (temp_header.data_size == 0u) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint data_size is zero\n");
        fclose(f);
        return -1;
    }
    std::error_code size_error;
    const auto file_bytes = std::filesystem::file_size(filename, size_error);
    const uint64_t payload_end = sizeof(temp_header) + temp_header.data_size;
    if (size_error || payload_end < temp_header.data_size ||
        (file_bytes != payload_end && file_bytes != payload_end + sizeof(uint64_t))) {
        ecm_ts_fprintf(stderr, "Warning: Checkpoint length mismatch\n");
        fclose(f);
        return -1;
    }
    uint32_t *data = (uint32_t *)malloc((size_t)temp_header.data_size);
    if (!data) {
        ecm_ts_fprintf(stderr, "Error: Could not allocate memory for checkpoint data\n");
        fclose(f);
        return -1;
    }
    if (fread(data, 1, (size_t)temp_header.data_size, f) != (size_t)temp_header.data_size) {
        ecm_ts_fprintf(stderr, "Warning: Could not read checkpoint data completely\n");
        free(data);
        fclose(f);
        return -1;
    }
    if (file_bytes == payload_end + sizeof(uint64_t)) {
        uint64_t stored = 0;
        const uint64_t actual = ecm_stage1_checksum(
            ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED, &temp_header,
                                sizeof(temp_header)), data, (size_t)temp_header.data_size);
        if (fread(&stored, sizeof(stored), 1, f) != 1 || stored != actual) {
            ecm_ts_fprintf(stderr, "Warning: Checkpoint checksum mismatch\n");
            free(data);
            fclose(f);
            return -1;
        }
    }
    fclose(f);
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
