#pragma once
#include "geometry.h"
#include "types.h"
namespace s2prod {
std::string trim(std::string s);
std::string upper(std::string s);
std::string number(const mpz_t z,int base=16);
uint64_t u64(std::string s,const char *label);
std::string json_string(const std::string &s);
uint64_t fingerprint(const std::string &s);
Record parse_record(std::string line,unsigned input_max_bits=ecm_stage2::max_input_bits,
                    EdwardsPreparation edwards=EdwardsPreparation::Construct,
                    const std::string& projection_target = {});
std::string saved_record_line(std::istream& input, uint64_t offset, uint64_t index, uint64_t key);
std::vector<Record> records(const fs::path &path,uint64_t skip,uint64_t count,
                            unsigned input_max_bits=ecm_stage2::max_input_bits,
                            EdwardsPreparation edwards=EdwardsPreparation::Construct,
                            const std::string& projection_target = {});
fs::path absolute_from(const fs::path &base,const std::string &p);
QueueSelection queue_fields(std::string line,unsigned input_max_bits=ecm_stage2::max_input_bits);
std::string queue_modulus(const EcmStage2Task &task,const Record *first_record);
void append(const fs::path &path,const std::string &line);
}
