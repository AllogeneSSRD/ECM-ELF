#pragma once
#include "arithmetic.h"

namespace ecm_opencl::stage2 {
struct ProjectivePoints { Polynomial x,z; }; // Ordinary canonical residues.
struct AffinePoints {
    Polynomial x,gcd; // x=0 for nonunits; gcd(Z,N) identifies each such point.
};
struct PointTiming { double kernel_seconds=0; Word upload_bytes=0,download_bytes=0; };
struct PointState;
// Experimental, synchronous device operators. One host thread and one Runtime.
// The odd target modulus is independent of a possible polynomial carrier: first
// normalize on the target, then explicitly lift residues to a divisible carrier.
class Points {
    std::unique_ptr<PointState> state_;
public:
    Points(ntt::Runtime &,const std::vector<Word> &modulus);
    ~Points();
    Points(const Points&)=delete;
    Points& operator=(const Points&)=delete;
    std::size_t words()const;
    const PointTiming &timing()const; // Last operation only; excludes construction.
    ProjectivePoints ladder(const std::vector<Word>&x,const std::vector<Word>&z,
        const std::vector<Word>&a24,const std::vector<Word>&scalars);
    // One inversion per segment in the normal case; a nonunit segment retries
    // its points independently so one bad Z never hides the other point factors.
    AffinePoints normalize(const ProjectivePoints &,std::size_t segment=32);
    // values must be canonical on this target. A carrier residue must first be
    // projected to the target; direct gcd below accepts any full-width value.
    Polynomial products(const Polynomial &values,std::size_t per_block=64);
    Polynomial gcd(const Polynomial &values); // gcd(value,N), including 0 -> N.
    std::vector<Word> download(const Polynomial &);
    Polynomial upload(const std::vector<Word>&); // Canonical ordinary residues.
};
} // namespace ecm_opencl::stage2
