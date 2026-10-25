#pragma once

#include <cstddef>
#include <memory>
#include "ecm/platform/stage1_stop.h"
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

namespace ecm_stage1_checkpoint {
enum class RestoreState { Missing, Resumed, AlreadyComputed, Incompatible, Corrupt, IoFailure };

struct RestoreResult {
    RestoreState state = RestoreState::Missing;
    std::string error;

    bool restored() const noexcept {
        return state == RestoreState::Resumed || state == RestoreState::AlreadyComputed;
    }
};

class Error : public std::runtime_error {
public:
    RestoreState state;

    Error(RestoreState kind, const std::string& message)
        : std::runtime_error(message), state(kind) {
    }
};
enum class BlobRead { Missing, Read, IoFailure };
} // namespace ecm_stage1_checkpoint

constexpr unsigned ECM_STAGE1_MAX_BACKUPS = 16;
constexpr uint64_t ECM_STAGE1_CHECKSUM_SEED = 1469598103934665603ull;
uint64_t ecm_stage1_checksum(uint64_t sum, const void* data, size_t size);
std::string ecm_stage1_backup_path(const std::string&, unsigned generation);
std::string ecm_stage1_checkpoint_temp_path(const std::string&);
bool ecm_stage1_is_curve_record(const std::string&);
std::string ecm_stage1_curve_record(uint32_t);
// Offline validation may inspect decoded seeds, never reenter this Context's I/O.
using EcmStage1PayloadValidator =
    std::function<bool(const std::vector<std::string>&, std::vector<uint64_t>&)>;

namespace ecm_stage1_checkpoint {
class Context {
    struct State;
    std::unique_ptr<State> state_;

public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void set_backup_limit(unsigned count);
    unsigned backup_limit();
    std::vector<std::string> candidates(const std::string& path);
    void mark_bad_primary(const std::string& path);
    bool commit(const std::string& path, const std::string& temp);
    void remove(const std::string& path);
    void set_path(const std::string& path);
    const std::string& path();
    void begin(const std::string& identity, uint32_t curves, bool fixed, uint64_t first_sigma,
               bool defer_payload_validation = false);
    bool pending();
    void validate_pending(const EcmStage1PayloadValidator& validate);
    bool saved_sigma(uint32_t curve, uint64_t& sigma);
    uint64_t sigma(uint32_t curve, uint64_t proposed);
    BlobRead read_checkpoint_blob(const std::string& path, std::string& bytes);
    bool read_blob(const std::string& path, std::string& bytes);
    bool write_blob(const std::string& path, const std::string& bytes);
    bool import_payload(const std::string& bytes, uint64_t first_sigma);
    bool import_payload(const std::string& bytes, const std::vector<uint64_t>& sigmas);
    void close(bool completed);
};
} // namespace ecm_stage1_checkpoint
