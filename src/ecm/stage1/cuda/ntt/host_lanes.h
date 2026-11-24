#pragma once

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

namespace ecm_mersenne_ntt {

// Large GMP inversions are independent across lanes. Keep CUDA calls and
// factor classification on the caller thread.
template<class Work>
void for_host_lanes(unsigned bits, size_t count, const Work& work) {
    const auto available = std::max(1u, std::thread::hardware_concurrency());
    const auto workers = bits >= (1u << 20) ? std::min<size_t>({count,available,4}) : 1;
    if (!workers) return;
    const auto run = [&](size_t worker) {
        for (size_t lane=worker; lane<count; lane+=workers) work(lane);
    };
    std::vector<std::future<void>> pending;
    for (size_t worker=1; worker<workers; ++worker)
        pending.push_back(std::async(std::launch::async,run,worker));
    run(0);
    for (auto& job:pending) job.get();
}

} // namespace ecm_mersenne_ntt
