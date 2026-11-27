// Exercise rollback and subblock exponent sums with a small independent ring.
#include "../../src/ecm/stage1/cuda/ntt/pm1_replay.h"
#include <array>
#include <cstdio>

int main() {
    using namespace ecm_mersenne_ntt;
    using namespace pm1_detail;
    constexpr uint64_t modulus = (1ull << 31) - 1;
    const std::vector<uint32_t> exponent{0, ~0u, 0x7201ABEF, ~0u, 0, ~0u, ~0u, 0x7fff};
    std::array<uint64_t,SlotCount> registers{};
    Pm1Progress saved{1,{3},0};
    registers[Current] = registers[PowerBase] = 3;
    unsigned calls = 0, verified_count = 0;
    const auto step = [&](bool bit) {
        auto& x = registers[Current];
        x = x*x % modulus;
        if (bit) x = 3*x % modulus;
        // Once in the first block and again much later, require recovery from
        // the most recent verified position. Corrupt no verification callback.
        if (++calls == 2 || calls == 101) x ^= 4;
    };
    const auto copy = [&](Slot dst, Slot src){ registers[dst] = registers[src]; };
    const auto mul = [&](Slot dst, Slot a, Slot b){ registers[dst] = registers[a]*registers[b] % modulus; };
    const auto read = [&](Slot src){ return std::vector<uint32_t>{uint32_t(registers[src])}; };
    const auto restore = [&](const auto& x){ registers[Current] = x[0]; };
    ecm_edwards::Integer e, m, base, prefix, want;
    mpz_import(e.v,exponent.size(),-1,4,0,0,exponent.data());
    mpz_set_ui(m.v,modulus);
    mpz_set_ui(base.v,3);
    const size_t bits = mpz_sizeinbase(e.v,2);
    const auto verified = [&](const Pm1Progress& value) {
        mpz_fdiv_q_2exp(prefix.v,e.v,bits-value.k);
        mpz_powm(want.v,base.v,prefix.v,m.v);
        if (mpz_cmp_ui(want.v,value.residue[0]))
            throw std::runtime_error("P-1 published an incorrect endpoint");
        ++verified_count;
        return true;
    };
    replay(exponent,saved,verified,step,copy,mul,read,restore);
    if (saved.k != bits || saved.errors != 2 || verified_count < 2) return 1;
    std::printf("P-1 Gerbicz: %u verified endpoints, %u recovered faults\n",verified_count,saved.errors);
}
