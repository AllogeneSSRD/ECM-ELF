#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Native OpenCL field-convolution foundation. No CUDA or parser dependency.
// A Runtime is submitted by one host thread; all its plans use one in-order
// queue. Upload/download are blocking; transform submissions are asynchronous.
namespace ecm_opencl::ntt {
using Word = std::uint64_t;
inline constexpr Word modulus = 0xffffffff00000001ULL;

struct DeviceInfo {
    std::size_t index = 0;
    std::string platform, name, vendor, version, driver, profile;
    Word global_bytes = 0, max_allocation_bytes = 0, local_bytes = 0;
    std::size_t max_work_group = 0;
    unsigned address_bits = 0, compute_units = 0;
    bool little_endian = false, compiler = false, available = false;
    bool supported = false;
    std::string reason;
};
std::vector<DeviceInfo> devices();

struct Options {
    std::size_t device = 0;
    Word budget_bytes = 512ULL << 20; // Owned payload, not device free memory.
    bool profiling = true;
    bool cache_programs = true; // Context-local exact source/options key; no disk cache.
    std::string resource_root;       // Directory containing goldilocks.clh/transform.cl.
};
struct PlanOptions {
    unsigned log2_length = 10;
    unsigned tile_bits = 8;
    std::size_t work_group = 128;
    bool split_multiply = false;
};
struct MemoryInfo { Word live_bytes = 0, peak_bytes = 0, budget_bytes = 0; };
struct ProgramInfo { Word builds=0,hits=0;double build_seconds=0; };
struct RuntimeState;
struct BufferState;
struct CompletionState;
struct PlanState;
struct NativeAccess; // Private bridge used by native device operators.

class Buffer {
    std::shared_ptr<BufferState> state_;
    explicit Buffer(std::shared_ptr<BufferState>);
    friend class Runtime;
    friend class Plan;
    friend struct NativeAccess;
public:
    Buffer() = default;
    std::size_t words() const;
};

class Completion {
    std::shared_ptr<CompletionState> state_;
    explicit Completion(std::shared_ptr<CompletionState>);
    friend class Plan;
public:
    Completion() = default;
    void wait() const;  // Propagates query/wait and negative execution errors.
    double seconds() const; // First kernel start -> last kernel end; waits first.
};

class Runtime {
    std::shared_ptr<RuntimeState> state_;
    friend class Plan;
    friend struct NativeAccess;
public:
    explicit Runtime(const Options &);
    const DeviceInfo &device() const;
    MemoryInfo memory() const;
    ProgramInfo programs() const;
    Buffer allocate(std::size_t words);
    void upload(const Buffer &, const Word *, std::size_t words, std::size_t offset = 0);
    void download(const Buffer &, Word *, std::size_t words, std::size_t offset = 0);
    void finish();
    std::string source_identity() const; // Diagnostic FNV identity; build manifest uses SHA256.
};

class Plan {
    std::shared_ptr<PlanState> state_;
public:
    Plan(Runtime &, const PlanOptions &);
    std::size_t length() const;
    unsigned tile_bits() const;
    Word table_bytes() const;
    std::string build_log() const;
    // Input elements must be canonical in [0,modulus). Each batch occupies
    // stride words, starting at offset. Zero stride means
    // length(). Offsets and strides are element counts, never device pointers.
    Completion forward(const Buffer &, std::size_t batch = 1,
                       std::size_t stride = 0, std::size_t offset = 0);
    Completion inverse(const Buffer &, std::size_t batch = 1,
                       std::size_t stride = 0, std::size_t offset = 0);
    // Cyclic field convolution. A becomes the result; B becomes its spectrum.
    // A and B must be different owners. Integer pack/carry/S4 are a later layer.
    Completion convolve(const Buffer &a, const Buffer &b, std::size_t batch = 1,
                        std::size_t stride = 0, std::size_t offset = 0);
};
} // namespace ecm_opencl::ntt
