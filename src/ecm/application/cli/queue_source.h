#pragma once
#include <string>
namespace ecm_stage1 {struct TaskDependencies;}
namespace ecm_stage1_app {
int run_queue_manager(const std::string &,int,const ecm_stage1::TaskDependencies &);
}
