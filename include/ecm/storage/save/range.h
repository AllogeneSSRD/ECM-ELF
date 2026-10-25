#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace ecm_save {
// METHOD records, not physical lines. Records are captured under the save lock;
// recovery/deduplication may refer to an already published contiguous range.
struct Range {
    std::string path;
    uint64_t skip=0;
    std::vector<std::string> records;
    bool valid=false;
};
}
