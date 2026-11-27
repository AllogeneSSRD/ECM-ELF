#pragma once
#include "pm1_probe.h"
#include "../../../cuda/ntt/runtime.h"
#include <condition_variable>
#include <future>
#include <mutex>

namespace ecm_mersenne_ntt {
// A common GPU timestamp makes overlapping streams one batch measurement.
// A lane that fails during setup must release peers already at the barrier.
class Pm1TimingGroup {
public:
    explicit Pm1TimingGroup(unsigned lanes) : elapsed(lanes) { CK(cudaEventCreate(&start)); }
    ~Pm1TimingGroup() { cudaEventDestroy(start); }

    void cancel(std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error) error = failure;
        ready.notify_all();
    }

    std::vector<double> times;
    unsigned steps = 8;
    size_t peak_bytes = 0;

    void measure(unsigned lane, cudaStream_t stream, const Pm1Advance& step, unsigned group_bits,
                 unsigned repeats, unsigned target_ms, size_t initial_free,
                 const std::vector<uint8_t>& exponent_bits) {
        cudaEvent_t end;
        CK(cudaEventCreate(&end));
        struct EndEvent { cudaEvent_t value; ~EndEvent() { cudaEventDestroy(value); } } cleanup{end};
        size_t cursor = size_t(lane)*exponent_bits.size()/elapsed.size();
        const auto iterate = [&](unsigned count) {
            while (count) {
                const unsigned width = std::min(count, group_bits);
                unsigned pattern = 0;
                for (unsigned i = 0; i < width; ++i) {
                    pattern = 2 * pattern + exponent_bits[cursor];
                    if (++cursor == exponent_bits.size()) cursor = 0;
                }
                step(pattern, width);
                count -= width;
            }
        };
        iterate(8);
        CK(cudaStreamSynchronize(stream));
        for (unsigned group=0;group<=repeats;++group) {
            rendezvous([&] {
                if (!group) {
                    size_t free,total;
                    CK(cudaMemGetInfo(&free,&total));
                    peak_bytes = initial_free > free ? initial_free-free : 0;
                }
                CK(cudaEventRecord(start,stream));
            });
            CK(cudaStreamWaitEvent(stream,start,0));
            iterate(steps);
            CK(cudaEventRecord(end,stream));
            CK(cudaEventSynchronize(end));
            CK(cudaEventElapsedTime(&elapsed[lane],start,end));
            rendezvous([&] {
                const auto ms = *std::max_element(elapsed.begin(),elapsed.end());
                if (!(ms>0)) throw std::runtime_error("invalid P-1 device timing");
                if (group) times.push_back(ms);
                else steps = unsigned(std::clamp(std::ceil(steps*double(target_ms)/ms),1.,double(UINT32_MAX)));
            });
        }
    }
private:
    std::vector<float> elapsed;
    cudaEvent_t start{};
    std::mutex mutex;
    std::condition_variable ready;
    unsigned waiting=0, generation=0;
    std::exception_ptr error;

    template<class F> void rendezvous(F last) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto phase=generation;
        if (!error && ++waiting==elapsed.size()) {
            last();
            waiting=0;
            ++generation;
            ready.notify_all();
        } else ready.wait(lock,[&]{return error || phase!=generation;});
        if (error) std::rethrow_exception(error);
    }
};
}
