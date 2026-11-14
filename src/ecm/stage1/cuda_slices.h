#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "ecm/platform/stop.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "../cuda/arithmetic/cgbn/types.cuh"
#include "../platform/generated/configuration_defaults.h"

namespace ecm_cuda_stage1 {

template<class Cursor> struct SliceWindow {
    Cursor end;
    unsigned launches = 0;
};

inline unsigned rolling_slice_depth(bool enabled, unsigned configured) {
    if (!enabled)
        return 0;
    if (configured)
        return configured == 8 ? 0 : configured;
    const char *value = std::getenv("ECM_CUDA_STAGE1_SLICE_DEPTH");
    if (!value)
        return ecm_config::defaults::stage1_cuda_slice_depth;
    if (std::strcmp(value, "8") == 0)
        return 0;
    if (std::strcmp(value, "2") == 0)
        return 2;
    if (std::strcmp(value, "3") == 0)
        return 3;
    throw std::invalid_argument("ECM_CUDA_STAGE1_SLICE_DEPTH must be 2, 3 or 8");
}

inline unsigned slice_wait_flags(bool low_cpu) {
    return low_cpu ? cudaEventBlockingSync : cudaEventDefault;
}

inline void check_slice_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw RunError(Failure::Device, cudaGetErrorString(result));
}

// The same device report keeps the first arithmetic error. The host reads a
// stream-ordered copy after each slice, even if the next slice is already running.
template<class Cursor> class RollingSlices {
    struct Slot {
        Cursor end{};
        cudaEvent_t begin = nullptr, kernel_end = nullptr, ready = nullptr;
    };
    std::array<Slot, 3> slots_{};
    cgbn_error_report_t *device_ = nullptr, *host_ = nullptr;
    Cursor submitted_;
    unsigned capacity_, oldest_ = 0, pending_ = 0;

    explicit RollingSlices(Cursor first, unsigned depth) : submitted_(first), capacity_(depth) {}

public:
    struct Completion {
        Cursor end;
        float milliseconds;
        cudaEvent_t kernel_end;
        cgbn_error_report_t *report;
    };

    static std::unique_ptr<RollingSlices> create(Cursor first, unsigned depth, bool low_cpu,
                                                 const cgbn_error_report_t *initial) {
        if (!depth)
            return {};
        if (depth != 2 && depth != 3)
            throw std::invalid_argument("CUDA Stage1 rolling depth must be 2 or 3");
        std::unique_ptr<RollingSlices> queue(new RollingSlices(first, depth));
        check_slice_cuda(cudaMalloc(&queue->device_, sizeof(*initial)));
        check_slice_cuda(cudaMallocHost(&queue->host_, depth * sizeof(*initial)));
        check_slice_cuda(cudaMemcpy(queue->device_, initial, sizeof(*initial), cudaMemcpyHostToDevice));
        for (unsigned i = 0; i < depth; ++i) {
            auto &slot = queue->slots_[i];
            check_slice_cuda(cudaEventCreate(&slot.begin));
            check_slice_cuda(cudaEventCreate(&slot.kernel_end));
            check_slice_cuda(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming | slice_wait_flags(low_cpu)));
        }
        return queue;
    }

    ~RollingSlices() {
        // A launch or a host callback may throw with work still in flight.
        if (pending_)
            cudaStreamSynchronize(nullptr);
        for (auto &slot : slots_) {
            if (slot.begin) cudaEventDestroy(slot.begin);
            if (slot.kernel_end) cudaEventDestroy(slot.kernel_end);
            if (slot.ready) cudaEventDestroy(slot.ready);
        }
        if (device_) cudaFree(device_);
        if (host_) cudaFreeHost(host_);
    }

    bool empty() const { return pending_ == 0; }

    RollingSlices(const RollingSlices &) = delete;
    RollingSlices &operator=(const RollingSlices &) = delete;

    template<class Launch>
    Completion next(Cursor total, Cursor length, bool barrier, Launch launch) {
        while (pending_ < capacity_ && submitted_ < total) {
            // At a checkpoint or stop, finish only the existing queue. An
            // empty queue may still issue one slice to establish a safe cursor.
            if (pending_ && (barrier || ecm_stage1_stop_requested()))
                break;
            const unsigned index = (oldest_ + pending_) % capacity_;
            auto &slot = slots_[index];
            const Cursor count = (std::min)(length, total - submitted_);
            ++pending_;
            check_slice_cuda(cudaEventRecord(slot.begin));
            launch(device_, submitted_, count);
            check_slice_cuda(cudaEventRecord(slot.kernel_end));
            check_slice_cuda(cudaMemcpyAsync(host_ + index, device_, sizeof(*device_), cudaMemcpyDeviceToHost));
            check_slice_cuda(cudaEventRecord(slot.ready));
            submitted_ += count;
            slot.end = submitted_;
        }
        const unsigned index = oldest_;
        const auto &slot = slots_[index];
        check_slice_cuda(cudaEventSynchronize(slot.ready));
        float milliseconds = 0;
        check_slice_cuda(cudaEventElapsedTime(&milliseconds, slot.begin, slot.kernel_end));
        oldest_ = (oldest_ + 1) % capacity_;
        --pending_;
        return {slot.end, milliseconds, slot.kernel_end, host_ + index};
    }
};

// Keep successive updates to the resident point buffer in the same stream.
// Each window has one host completion boundary for errors, progress and saves.
template<class Cursor, class Launch>
SliceWindow<Cursor> submit_slice_window(Cursor first, Cursor total, Cursor length,
                                        bool lookahead, Launch launch) {
    constexpr unsigned max_launches = 8;
    SliceWindow<Cursor> window{first};
    const unsigned limit = lookahead ? max_launches : 1;
    do {
        const Cursor count = (std::min)(length, total - window.end);
        launch(window.end, count);
        window.end += count;
        ++window.launches;
    } while (window.end < total && window.launches < limit && !ecm_stage1_stop_requested());
    return window;
}

}
