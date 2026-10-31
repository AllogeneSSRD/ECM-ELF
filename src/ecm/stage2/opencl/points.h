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
    Points(ntt::Runtime &,const std::vector<Word> &modulus,std::size_t ladder_group=0);
    ~Points();
    Points(const Points&)=delete;
    Points& operator=(const Points&)=delete;
    std::size_t words()const;
    const PointTiming &timing()const; // Last operation only; excludes construction.
    ProjectivePoints ladder(const std::vector<Word>&x,const std::vector<Word>&z,
        const std::vector<Word>&a24,const std::vector<Word>&scalars);
    // Consecutive [first+i]Q, seeded independently per bounded segment. On a
    // composite target the caller must recover nonunit denominators through
    // independent ladders before interpreting them as factors/affine roots.
    ProjectivePoints consecutive(const std::vector<Word>&x,const std::vector<Word>&z,
        const std::vector<Word>&a24,Word first,std::size_t count,std::size_t segment);
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
