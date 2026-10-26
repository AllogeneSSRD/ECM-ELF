#pragma once
#include "../types.h"
namespace s2prod {
struct PackZeroDefaults {bool has_pack_zero,pack_zero;std::string cost_profile;};
void configure_pack_zero(Options &o,const PackZeroDefaults *s=nullptr);
void configure_cuda_wait(int device);
struct HostWaitDefaults {bool present,value;};
void configure_host_wait(Options &o,const HostWaitDefaults *defaults=nullptr);
}
