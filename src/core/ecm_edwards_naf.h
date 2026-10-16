#pragma once
#include <gmp.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Standard width-w NAF. Two bit planes keep storage near twice the binary
// scalar size, independent of w. A nonzero at bit i stores its (w-1)-bit code
// at i in codes: low w-2 bits are (abs(digit)-1)/2, high bit is the sign.
// Nonzero positions are >=w apart, so codes never overlap. Positions and
// checkpoint progress count MSB-first columns, including an optional carry.
struct EcmEdwardsNaf {
    unsigned window;
    uint32_t bits=0, additions=0;
    std::vector<uint32_t> nonzero,codes;

    static uint32_t read(const std::vector<uint32_t> &words,uint64_t bit,unsigned width) {
        const size_t i=size_t(bit/32);const unsigned offset=unsigned(bit%32);
        uint64_t value=i<words.size()?words[i]:0;
        if(offset && i+1<words.size())value|=uint64_t(words[i+1])<<32;
        return uint32_t(value>>offset)&((1u<<width)-1);
    }
    explicit EcmEdwardsNaf(const mpz_t scalar,unsigned w):window(w) {
        if(w<2 || w>12 || mpz_sgn(scalar)<=0 || mpz_sizeinbase(scalar,2)>=UINT32_MAX)
            throw std::runtime_error("invalid GPU Edwards NAF scalar/window (2..12)");
        const uint64_t length=mpz_sizeinbase(scalar,2);
        std::vector<uint32_t> words(size_t((length+31)/32),0);size_t used=0;
        mpz_export(words.data(),&used,-1,4,0,0,scalar);
        // Extra word permits a top carry and a code crossing the last limb.
        nonzero.resize(size_t((length+32)/32)+1,0);codes.resize(nonzero.size(),0);
        uint32_t carry=0;
        for(uint64_t i=0;i<length || carry;) {
            const auto low=read(words,i,1)+carry;
            if(!(low&1u)){carry=low>>1;++i;continue;}
            const auto value=read(words,i,w)+carry;
            const bool negative=value>=(1u<<(w-1));
            const auto magnitude=negative?(1u<<w)-value:value;
            const auto code=((magnitude-1)/2)|(uint32_t(negative)<<(w-2));
            nonzero[size_t(i/32)]|=1u<<(i%32);
            const uint64_t packed=uint64_t(code)<<(i%32);
            codes[size_t(i/32)]|=uint32_t(packed);
            codes[size_t(i/32)+1]|=uint32_t(packed>>32);
            bits=uint32_t(i+1);++additions;carry=negative;i+=w;
        }
    }
    int digit(uint32_t bit) const {
        if(!read(nonzero,bit,1))return 0;
        const auto code=read(codes,bit,window-1);
        const int value=int(2*(code&((1u<<(window-2))-1))+1);
        return (code>>(window-2))?-value:value;
    }
};
