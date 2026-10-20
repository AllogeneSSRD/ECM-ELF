#pragma once
#include <cstdint>
struct EcmPracPrime {uint32_t p,d,repetitions,work;};
static_assert(sizeof(EcmPracPrime)==16,"PRAC descriptor ABI");
