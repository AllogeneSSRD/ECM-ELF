#pragma once
#include <stdexcept>
#include <string>
namespace ecm_stage2 {
constexpr unsigned reduce_kernel_current=1,reduce_default_threads=128;
inline bool cooperative_reduce_width(unsigned bits,bool mersenne) {return !mersenne && bits>8192 && bits<=16384;}
inline bool valid_reduce_variant(unsigned threads,unsigned kernel) {return (threads==64 || threads==96 || threads==128) && (kernel==1 || kernel==2);}
inline int reduce_backend(const std::string &value) {
    if(value=="auto")return -1;if(value=="scalar")return 0;if(value=="coop32")return 32;
    throw std::runtime_error("reduce backend must be auto, scalar or coop32");
}
}
