#pragma once
#include <filesystem>
#include <string>
namespace s2prod {
struct ResultIdentity {std::string result_user,result_computer,submission;};
ResultIdentity read_result_identity(const std::filesystem::path &identity_ini,int worker,bool explicit_config);
}
