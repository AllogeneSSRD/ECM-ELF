#pragma once
#include "ecm/service/stage1/task.h"
#include <string>
#include <vector>
namespace ecm_stage1_app {
std::string get_gp_executable(const std::string &explicit_path="");
std::string resolve_gp_path(const std::string &);
bool gp_executable_exists(const std::string &);
bool build_primes_up_to_B1(double,std::vector<uint32_t> &);
bool print_group_order_factor(const ecm_stage1::TaskResult &,uint32_t,int,const std::string &,const std::vector<uint32_t> &);
}
