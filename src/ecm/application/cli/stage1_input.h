#pragma once
#include "ecm/service/stage1/task.h"
#include <filesystem>
#include <string>
#include <vector>
struct EcmQueueConfig;
namespace ecm_stage1_app {
std::string normalize_path_local(std::filesystem::path);
std::string resolve_rel_local(const std::string &,const std::string &);
std::string get_exe_dir_local();
bool stage1_save_construction(const std::string &,ecm_stage1::Stage1Options &,std::string &);
bool stage1_selection_error(const ecm_stage1::Selection &,std::string &);
bool stage1_task_error(const ecm_stage1::Stage1Options &,uint32_t,std::string &);
std::vector<unsigned> parse_affinity_spec(const std::string &);
void stage1_result_settings(ecm_stage1::Stage1Options &,const EcmQueueConfig &,const std::string &,int);
}
