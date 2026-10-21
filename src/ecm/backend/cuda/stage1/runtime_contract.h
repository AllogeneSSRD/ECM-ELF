#pragma once
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace ecm_cuda_stage1 {
enum class Termination { Completed, Stopped, SampleLimit, Failed };
enum class Failure { None, Preparation, Execution, Persistence, Publication, Device };
struct RunResult {
    int code=-1;
    Termination termination=Termination::Failed;
    Failure failure=Failure::Preparation;
    uint64_t cursor=0,total=0,durable_cursor=0;
    std::string error;
};
class RunError : public std::runtime_error {
public:
    Failure failure;
    RunError(Failure cause,const std::string &message):std::runtime_error(message),failure(cause){}
};
inline size_t checked_size(std::initializer_list<size_t> factors) {
    size_t result=1;
    for(auto value:factors) {
        if(value && result>std::numeric_limits<size_t>::max()/value)
            throw std::length_error("CUDA Stage1 allocation size overflow");
        result*=value;
    }
    return result;
}
}
