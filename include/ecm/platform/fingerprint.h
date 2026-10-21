#pragma once
#include <filesystem>
#include <iosfwd>
#include <string>
namespace ecm_stage2 {
std::string sha256_stream(std::istream &);
std::string sha256_file(const std::filesystem::path &);
std::string sha256_text(const std::string &);
}
