#pragma once
#include "ecm/storage/save/range.h"
#include <string>
#include <gmp.h>
namespace ecm_save {
class Capture {
    Capture *previous_=nullptr;
public:
    Range range;
    Capture();
    ~Capture();
    Capture(const Capture &)=delete;
    Capture &operator=(const Capture &)=delete;
};
// Missing/empty is zero; unreadable/corrupt files throw. Old comments are retained.
uint64_t record_count(const std::string &path);
std::string verified_snapshot(const Range &range,uint64_t *record_count=nullptr);
Range inspect_range(const std::string &path,uint64_t skip,uint32_t count);
bool publish(const std::string &path,const std::string &lines,bool append);
// Serialization identity/checksums shared by new batches and legacy P95 replay.
// param=-1 denotes Edwards. Legacy Edwards may complete lanes without a point;
// allow_missing_points accepts only a subset of the supplied batch identities.
void validate_batch(const Range &,mpz_srcptr modulus,uint64_t b1,int param,
                    const std::vector<uint64_t> &sigmas={},bool allow_missing_points=false);
void validate_p95_batch(const Range &,mpz_srcptr modulus);
}
