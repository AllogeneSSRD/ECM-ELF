#pragma once
#include "types.h"
namespace s2prod {
std::string trim(std::string s);
std::string upper(std::string s);
std::string number(const mpz_t z,int base=16);
uint64_t u64(std::string s,const char *label);
std::string json_string(const std::string &s);
uint64_t fingerprint(const std::string &s);
Record parse_record(std::string line);
std::vector<Record> records(const fs::path &path,uint64_t skip,uint64_t count);
fs::path absolute_from(const fs::path &base,const std::string &p);
QueueSelection queue_fields(std::string line);
void append(const fs::path &path,const std::string &line);
}
