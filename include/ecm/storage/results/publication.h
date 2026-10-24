#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <gmp.h>

namespace ecm_results {
struct Submission {
    uint64_t exponent=0,sigma=0,b1=0,b2=0,curves=1;
    uint64_t d=0,poly_size=0,stage2_fft_length=0; // Actual completed Stage2 geometry; zero means unavailable.
    int stage=0;bool edwards=true;
    int param=0;
    bool stage1_only=false,kbnc=false;
    std::string k="1";uint32_t b=2;int32_t c=-1;
    std::vector<std::string> factors,known;
    std::string user,computer,aid,when,publication_task;
    uint64_t publication_order=0; // Position in the selected Stage2 task; not part of the public JSON.
    std::string save_name; // Private Stage2 receipt namespace, derived from the save filename.
};
struct Stage1BatchRecord { Submission submission; std::string internal; };
std::string quoted(const std::string &);
std::string decimal(mpz_srcptr);
std::string utc_timestamp();
uint64_t target_exponent(const Submission &,mpz_srcptr);
std::string submission_json(const Submission &,mpz_srcptr);
std::string internal_json(Submission,mpz_srcptr,uint64_t,const std::string &,const std::string &extra_fields={});
void check_stage1_outputs(const std::filesystem::path &,const std::filesystem::path &);
void check_stage2_outputs(const std::filesystem::path &,const std::filesystem::path &,bool resumed=false);
void publish_stage1_batch(const std::filesystem::path &,const std::filesystem::path &,
                         std::vector<Stage1BatchRecord>,mpz_srcptr,const std::string &);
bool has_result_ledger(const std::filesystem::path &);
bool contains_stage1_curves(const std::filesystem::path &,mpz_srcptr,uint64_t,bool,unsigned,const std::vector<uint64_t> &);
void publish_submission(const std::filesystem::path &,const Submission &,mpz_srcptr,bool render_now=false);
void seal_submissions(const std::filesystem::path &,int stage_filter=0,const std::string &task={},bool whole_task=false);
void publish_internal_batch(const std::filesystem::path &,const std::vector<std::pair<std::string,std::string>> &,bool render_now=true);
void publish_internal(const std::filesystem::path &,const std::string &,const std::string &,bool render_now=true);
void refresh_internal(const std::filesystem::path &);
void refresh_submission_if_owned(const std::filesystem::path &);
} // namespace ecm_results
