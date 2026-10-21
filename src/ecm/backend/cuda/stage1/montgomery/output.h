#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>
namespace ecm_cuda_mont_host {
bool stdout_is_tty();
void print_progress(double,uint64_t,uint64_t,double,double,double,bool,const char *unit="bits");
bool should_emit_progress_line(int,double,double);
FILE *open_dump_csv(const char *);
void dump_curve_state_csv(FILE *,const char *,int,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,const uint32_t *,uint32_t,size_t);
struct PracWindow;
void write_prac_window_csv(const PracWindow &,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,const std::vector<uint32_t> &);
}
