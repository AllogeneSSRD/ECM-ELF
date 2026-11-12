#pragma once
#include "../types.h"
namespace s2prod {
struct PackZeroDefaults {bool has_pack_zero,pack_zero;std::string cost_profile;};
void configure_pack_zero(Options &o,const PackZeroDefaults *s=nullptr);
void configure_weighted_ntt(const Options &o);
void configure_cuda_wait(int device);
struct HostWaitDefaults {bool present,value;};
void configure_host_wait(Options &o,const HostWaitDefaults *defaults=nullptr);
void configure_phase_experiment(Options &o);
void configure_groot_experiment(Options &o);
void configure_giant_policy(Options &o);
struct PointBackendDefaults {bool baby_present,giant_present;std::string baby,giant;bool seed_present=false;std::string seed="scalar";};
void configure_point_backends(Options &o,const PointBackendDefaults *defaults=nullptr);
void apply_point_backends(const Options &o);
void configure_s4_backend(Options &o,bool present,const std::string &value);
}
