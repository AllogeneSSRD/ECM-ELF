#include "thread_kernels.h"
#include "../thread_workflow.h"
#include <utility>

namespace ecm_cuda_edwards_thread {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T, auto Release> struct Handle {
    T value = nullptr;
    Handle() = default;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& rhs) noexcept : value(std::exchange(rhs.value, nullptr)) {}
    ~Handle() { if (value) Release(value); }
};

using Buffer = Handle<void*, cudaFree>;
using Event = Handle<cudaEvent_t, cudaEventDestroy>;
using Stream = Handle<cudaStream_t, cudaStreamDestroy>;

class Device {
    const unsigned limbs_;
    const unsigned block_;
    Stream stream_;
    Event start_, stop_;
    KernelPair kernels_{};
    cudaFuncAttributes attributes_{};
    bool cios_ = false;
    int index_;

    template<class Launch>
    double time(Launch launch) {
        check(cudaEventRecord(start_.value, stream_.value), "record Edwards CUDA start");
        launch();
        check(cudaGetLastError(), "launch Edwards CUDA kernel");
        check(cudaEventRecord(stop_.value, stream_.value), "record Edwards CUDA end");
        check(cudaEventSynchronize(stop_.value), "wait for Edwards CUDA kernel");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start_.value, stop_.value), "time Edwards CUDA kernel");
        return ms;
    }

public:
    static constexpr const char* name = "cuda-thread";
    static constexpr const char* label = "CUDA thread";
    static constexpr const char* checkpoint_format = "CUDA_ED_THREAD_V1";

    Device(const ecm_stage1::ExecutionOptions& opt, unsigned limbs)
        : limbs_(limbs), block_(opt.execution.cuda_thread_block_size),
          cios_(limbs > 16), index_(opt.execution.device_index) {
        check(cudaSetDevice(index_), "select Edwards CUDA device");
        if (block_ != 32 && block_ != 64 && block_ != 128 && block_ != 256)
            throw std::invalid_argument("CUDA Edwards thread block size must be 32, 64, 128 or 256");
        check(cudaStreamCreateWithFlags(&stream_.value, cudaStreamNonBlocking), "create Edwards CUDA stream");
        check(cudaEventCreate(&start_.value), "create Edwards CUDA start event");
        check(cudaEventCreateWithFlags(&stop_.value, cudaEventBlockingSync), "create Edwards CUDA stop event");
    }

    uint64_t max_buffer() const { return SIZE_MAX; }
    uint64_t memory_budget() const {
        size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "query Edwards CUDA memory");
        return ecm_edwards_memory::usable_bytes(free, 64 * ecm_edwards_memory::mib);
    }

    void prepare(bool two_xy) {
        if (limbs_ <= 8)
            kernels_ = small_kernels(limbs_, two_xy);
        else if (limbs_ <= 16)
            kernels_ = medium_kernels(limbs_, two_xy);
        else
            kernels_ = wide_kernels(limbs_, two_xy);
        if (!kernels_.slice)
            throw std::invalid_argument("unsupported CUDA Edwards thread container");
        check(kernels_.attributes(&attributes_), "query Edwards CUDA kernel attributes");
        if (block_ > static_cast<unsigned>(attributes_.maxThreadsPerBlock))
            throw std::invalid_argument("CUDA Edwards thread block exceeds the kernel's device limit");
    }

    std::string details() const {
        return "tpb=" + std::to_string(block_) + " field_mul=" + (cios_ ? "cios" : "fips_ptx") +
               " field_sqr=" + (cios_ ? "cios" : "fips_ptx_square") + " device=" + std::to_string(index_) +
               " registers=" + std::to_string(attributes_.numRegs) +
               " local_bytes=" + std::to_string(attributes_.localSizeBytes);
    }

    Buffer buffer(size_t bytes, bool) const {
        Buffer out;
        check(cudaMalloc(&out.value, bytes), "allocate Edwards CUDA buffer");
        return out;
    }
    void upload(const Buffer& to, const void* from, size_t bytes) const {
        // Pageable H2D copies can return while DMA is still pending. Order
        // them on the same non-blocking stream as the first consuming kernel.
        check(cudaMemcpyAsync(to.value, from, bytes, cudaMemcpyHostToDevice, stream_.value),
              "upload Edwards CUDA buffer");
    }
    void download(const Buffer& from, void* to, size_t bytes) const {
        check(cudaMemcpyAsync(to, from.value, bytes, cudaMemcpyDeviceToHost, stream_.value),
              "download Edwards CUDA buffer");
        check(cudaEventRecord(stop_.value, stream_.value), "record Edwards CUDA download");
        check(cudaEventSynchronize(stop_.value), "wait for Edwards CUDA download");
    }
    double dictionary(const Buffer& params, const Buffer& seeds, const Buffer& table, const Buffer& scratch,
                      uint32_t count, uint32_t np0, uint32_t first, uint32_t stop, uint32_t weights) {
        return time([&] {
            kernels_.dictionary(words(params), words(seeds), words(table), words(scratch), count, np0,
                                first, stop, weights, block_, stream_.value);
        });
    }
    double slice(const Buffer& params, const Buffer& points, const Buffer& table, const Buffer& nonzero,
                 const Buffer& codes, uint32_t count, uint32_t np0, uint32_t bits, uint32_t first,
                 uint32_t stop, uint32_t weights) {
        return time([&] {
            kernels_.slice(words(params), words(points), words(table), words(nonzero), words(codes), count,
                           np0, bits, first, stop, weights, block_, stream_.value);
        });
    }

private:
    static uint32_t* words(const Buffer& buffer) { return static_cast<uint32_t*>(buffer.value); }
};
} // namespace
} // namespace ecm_cuda_edwards_thread

namespace ecm_cuda_stage1 {
ecm_stage1::ComputationResult run_edwards_thread_stage1(const ecm_stage1::Stage1Request& request,
        ecm_stage1::ExecutionOptions& opt, ecm_stage1_checkpoint::Context& checkpoint) {
    return ecm_edwards_thread::run_stage1<ecm_cuda_edwards_thread::Device>(request, opt, checkpoint);
}
}
