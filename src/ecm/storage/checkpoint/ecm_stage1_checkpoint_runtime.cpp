#include "ecm_stage1_checkpoint_runtime.h"
#include "../../platform/ecm_file_publication.h"
#include "ecm_stage1_binary.h"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <utility>
#include <memory>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {
std::atomic<bool> stop_requested{false};
std::atomic<unsigned> backup_limit{3};
std::mutex bad_primary_mutex;
std::unordered_set<std::string> bad_primaries;
std::string batch_path, batch_identity;
std::vector<std::string> batch_records;
std::vector<uint64_t> batch_sigmas;
std::mutex batch_mutex;
struct DecodedBatch {
    std::vector<std::string> records;
    std::vector<uint64_t> sigmas;
    bool binary=false;
};
std::unique_ptr<DecodedBatch> pending_batch;
std::string pending_path;
std::vector<std::string> pending_older;
bool batch_fixed=false;
uint64_t batch_requested_sigma=0;
bool payload_validation_pending=false;
constexpr char record_prefix[] = "ecm-batch-record:";
#ifdef _WIN32
std::atomic<unsigned> console_stop_count{0};
HANDLE stop_event = nullptr; // borrowed inherited handle; GUI owns its lifetime
HANDLE batch_lock = INVALID_HANDLE_VALUE;
BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    if (console_stop_count.fetch_add(1, std::memory_order_relaxed) > 0) return FALSE;
    stop_requested.store(true, std::memory_order_relaxed);
    return TRUE;
}
#else
int batch_lock=-1;
void signal_control(int) { stop_requested.store(true, std::memory_order_relaxed); }
#endif

bool replace_file(const std::string &source, const std::string &target) {
    return ecm_replace_file(source,target);
}
} // namespace

void ecm_stage1_install_stop_handler() {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_control, TRUE);
#else
    std::signal(SIGINT, signal_control);
    std::signal(SIGTERM, signal_control);
#endif
}

bool ecm_stage1_set_stop_handle(uintptr_t value) {
#ifdef _WIN32
    HANDLE candidate = reinterpret_cast<HANDLE>(value);
    DWORD flags = 0;
    if (!candidate || !GetHandleInformation(candidate, &flags)) return false;
    stop_event = candidate;
    return true;
#else
    (void)value;
    return false;
#endif
}

bool ecm_stage1_stop_requested() {
    if (stop_requested.load(std::memory_order_relaxed)) return true;
#ifdef _WIN32
    if (stop_event && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        stop_requested.store(true, std::memory_order_relaxed);
        return true;
    }
#endif
    return false;
}

void ecm_stage1_set_backup_limit(unsigned count) {
    backup_limit.store(count <= ECM_STAGE1_MAX_BACKUPS ? count : ECM_STAGE1_MAX_BACKUPS,
                       std::memory_order_relaxed);
}
unsigned ecm_stage1_backup_limit() { return backup_limit.load(std::memory_order_relaxed); }

std::string ecm_stage1_backup_path(const std::string &path, unsigned generation) {
    const auto slash = path.find_last_of("/\\");
    const auto dot = path.find_last_of('.');
    const auto at = dot != std::string::npos && (slash == std::string::npos || dot > slash)
                        ? dot : path.size();
    return path.substr(0, at) + ".bak" + (generation == 1 ? "" : std::to_string(generation)) +
           path.substr(at);
}

std::vector<std::string> ecm_stage1_checkpoint_candidates(const std::string &path) {
    std::vector<std::string> result{path};
    if (ecm_stage1_is_curve_record(path)) return result;
    for (unsigned i = 1; i <= ecm_stage1_backup_limit(); ++i)
        result.push_back(ecm_stage1_backup_path(path, i));
    return result;
}

std::string ecm_stage1_checkpoint_temp_path(const std::string &path) { return path + ".new"; }

void ecm_stage1_mark_bad_primary(const std::string &path) {
    std::lock_guard<std::mutex> lock(bad_primary_mutex);
    bad_primaries.insert(path);
}

bool ecm_stage1_commit_checkpoint(const std::string &path, const std::string &temp) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const unsigned limit = ecm_stage1_backup_limit();
    bool bad_primary = false;
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primary = bad_primaries.count(path) != 0;
    }
    if (limit && !bad_primary) {
        for (unsigned i = limit; i > 1; --i) {
            const auto from = ecm_stage1_backup_path(path, i - 1);
            if (fs::exists(from, ec) && !replace_file(from, ecm_stage1_backup_path(path, i)))
                return false;
            if (ec) return false;
        }
        if (fs::exists(path, ec)) {
            fs::copy_file(path, ecm_stage1_backup_path(path, 1),
                          fs::copy_options::overwrite_existing, ec);
            if (ec) return false;
        } else if (ec) return false;
    }
    if (!replace_file(temp, path)) return false;
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primaries.erase(path);
    }
    // A reduced limit takes effect only after the new primary is safely installed.
    for (unsigned i = limit + 1; i <= ECM_STAGE1_MAX_BACKUPS; ++i)
        fs::remove(ecm_stage1_backup_path(path, i), ec);
    return true;
}

void ecm_stage1_remove_checkpoints(const std::string &path) {
    // The completed text save AND both result streams must publish first.
    if (ecm_stage1_is_curve_record(path) || (!batch_path.empty() && path == batch_path)) return;
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primaries.erase(path);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(ecm_stage1_checkpoint_temp_path(path), ec);
    for (unsigned i = 1; i <= ECM_STAGE1_MAX_BACKUPS; ++i)
        std::filesystem::remove(ecm_stage1_backup_path(path, i), ec);
}

uint64_t ecm_stage1_checksum(uint64_t sum, const void *data, size_t size) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        sum ^= p[i];
        sum *= 1099511628211ull;
    }
    return sum;
}

namespace {
bool flush_batch() {
    using namespace ecm_stage1_binary;
    std::string body(batch_magic,8);
    u32(body,2);u32(body,uint32_t(batch_records.size()));u64(body,batch_identity.size());
    body+=batch_identity;
    for(size_t i=0;i<batch_records.size();++i) {
        u64(body,batch_sigmas[i]);u64(body,batch_records[i].size());
        u32(body,crc32c(batch_records[i]));u32(body,0);body+=batch_records[i];
    }
    u32(body,crc32c(body));
    const auto temp=ecm_stage1_checkpoint_temp_path(batch_path);
    std::ofstream file(temp,std::ios::binary|std::ios::trunc);
    file.write(body.data(),static_cast<std::streamsize>(body.size()));
    file.close();
    return bool(file) && ecm_stage1_commit_checkpoint(batch_path,temp);
}
size_t record_index(const std::string &path) {
    const auto text=path.substr(sizeof(record_prefix)-1);size_t used=0;
    const auto index=std::stoull(text,&used);
    if(used!=text.size() || index>=batch_records.size())throw std::runtime_error("invalid batch checkpoint curve index");
    return static_cast<size_t>(index);
}
}
void ecm_stage1_set_checkpoint_path(const std::string &path) { batch_path=path; }
const std::string &ecm_stage1_checkpoint_path() { return batch_path; }
bool ecm_stage1_is_curve_record(const std::string &path) { return path.rfind(record_prefix,0)==0; }
std::string ecm_stage1_curve_record(uint32_t curve) { return std::string(record_prefix)+std::to_string(curve); }
namespace {
enum class Decode { Valid, Corrupt, Incompatible };
Decode decode_batch(const std::string &bytes,const std::string &identity,uint32_t curves,DecodedBatch &decoded) {
    if(ecm_stage1_binary::is_batch(bytes)) {
        using namespace ecm_stage1_binary;
        std::vector<std::string> records;std::vector<uint64_t> sigmas;
        std::string saved_identity;uint32_t count=0;
    try {
        if(bytes.size()<28)return Decode::Corrupt;
        Reader footer(std::string_view(bytes).substr(bytes.size()-4));
        if(footer.u32()!=crc32c(std::string_view(bytes).substr(0,bytes.size()-4)))return Decode::Corrupt;
        Reader in(std::string_view(bytes).substr(0,bytes.size()-4));in.take(8);
        if(in.u32()!=2)return Decode::Corrupt;
        count=in.u32();saved_identity=std::string(in.take(in.u64()));
        if(count>in.left()/24)return Decode::Corrupt;
        records.resize(count);sigmas.resize(count);
        for(uint32_t i=0;i<count;++i) {
            sigmas[i]=in.u64();const auto length=in.u64();const auto sum=in.u32();
            if(in.u32()!=0)throw std::runtime_error("unsupported checkpoint record flags");
            auto payload=in.take(length);
            if(crc32c(payload)!=sum)throw std::runtime_error("checkpoint record CRC32C mismatch");
            records[i]=std::string(payload);
        }
        if(in.left())return Decode::Corrupt;
    }catch(const std::runtime_error &){return Decode::Corrupt;}

        if(saved_identity!=identity || count!=curves)return Decode::Incompatible;
        decoded.records=std::move(records);decoded.sigmas=std::move(sigmas);decoded.binary=true;
        return Decode::Valid;
    }
        const auto at=bytes.rfind("CHECKSUM ");if(at==bytes.npos)return Decode::Corrupt;
        const auto body=bytes.substr(0,at);std::istringstream footer(bytes.substr(at+9));uint64_t sum;std::string extra;
        if(!(footer>>sum) || (footer>>extra) || sum!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,body.data(),body.size()))return Decode::Corrupt;
        std::istringstream in(body);std::string tag,saved_identity;uint64_t count;
        if(!std::getline(in,tag) || tag!="ECM_STAGE1_BATCH_V1" || !std::getline(in,saved_identity) || !(in>>count))return Decode::Corrupt;
        if(saved_identity!=identity || count!=curves)return Decode::Incompatible;
        std::vector<std::string> records(curves);std::vector<uint64_t> sigmas(curves);bool valid=true;
        for(uint32_t i=0;i<curves;++i) {
            uint64_t size;
            if(!(in>>sigmas[i]>>size) || size>body.size() || in.get()!='\n'){valid=false;break;}
            records[i].resize(static_cast<size_t>(size));in.read(records[i].data(),static_cast<std::streamsize>(size));
            if(!in || in.get()!='\n'){valid=false;break;}
        }
        if(!valid || in.peek()!=std::char_traits<char>::eof())return Decode::Corrupt;

    decoded.records=std::move(records);decoded.sigmas=std::move(sigmas);decoded.binary=false;
    return Decode::Valid;
}
bool sigma_matches(const std::vector<uint64_t> &sigmas) {
    if(sigmas.size()!=batch_records.size())return false;
    for(size_t i=0;i<sigmas.size();++i)
        if(batch_fixed && sigmas[i] && sigmas[i]!=batch_requested_sigma+i)return false;
    return true;
}
void accept_batch(DecodedBatch &decoded,const std::string &path) {
    batch_records.swap(decoded.records);batch_sigmas.swap(decoded.sigmas);
    if(path!=batch_path)ecm_stage1_mark_bad_primary(batch_path);
    std::cout<<"Stage1 batch resume: path="<<path<<" curves="<<batch_records.size()
             <<(decoded.binary?" format=binary-v2":"")<<'\n';
}
}
void ecm_stage1_begin_batch(const std::string &identity,uint32_t curves,bool fixed,uint64_t first_sigma,bool defer_payload_validation) {
    std::lock_guard<std::mutex> lock(batch_mutex);
    pending_batch.reset();pending_path.clear();pending_older.clear();payload_validation_pending=false;
    batch_fixed=fixed;batch_requested_sigma=first_sigma;
    if(fixed && curves && first_sigma>UINT64_MAX-(curves-1ull))
        throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,"batch sigma range overflow");
    batch_identity=identity;batch_records.assign(curves,{});batch_sigmas.assign(curves,0);
    if(batch_path.empty())return;
#ifdef _WIN32
    const auto lock_path=std::filesystem::path(batch_path+".lock");
    batch_lock=CreateFileW(lock_path.c_str(),GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
    if(batch_lock==INVALID_HANDLE_VALUE)throw std::runtime_error("another worker owns this batch checkpoint; use a separate checkpoint directory");
#else
    batch_lock=::open((batch_path+".lock").c_str(),O_CREAT|O_WRONLY,0600);
    if(batch_lock<0)throw std::runtime_error("cannot open batch checkpoint lock");
    if(::flock(batch_lock,LOCK_EX|LOCK_NB)){::close(batch_lock);batch_lock=-1;throw std::runtime_error("another worker owns this batch checkpoint; use a separate checkpoint directory");}
#endif
    using namespace ecm_stage1_checkpoint;
    bool existing=false,io_failure=false;
    const auto candidates=ecm_stage1_checkpoint_candidates(batch_path);
    for(size_t index=0;index<candidates.size();++index) {
        const auto &candidate=candidates[index];std::string bytes;
        const auto read=ecm_stage1_read_checkpoint_blob(candidate,bytes);
        if(read==BlobRead::Missing)continue;
        existing=true;
        if(read==BlobRead::IoFailure){io_failure=true;continue;}
        auto decoded=std::make_unique<DecodedBatch>();
        const auto result=decode_batch(bytes,identity,curves,*decoded);
        if(result==Decode::Corrupt)continue;
        if(result==Decode::Incompatible)
            throw Error(RestoreState::Incompatible,"batch checkpoint belongs to different target/parameters/curve count/CPU algorithm; select another checkpoint directory");
        if(!sigma_matches(decoded->sigmas))throw Error(RestoreState::Incompatible,"batch checkpoint sigma differs from requested sigma");
        if(defer_payload_validation) {
            pending_batch=std::move(decoded);pending_path=candidate;payload_validation_pending=true;
            pending_older.assign(candidates.begin()+index+1,candidates.end());
            std::cout<<"Stage1 batch candidate: path="<<candidate<<" curves="<<curves<<"; payload validation pending\n";
        } else accept_batch(*decoded,candidate);
        return;
    }
    if(existing)throw Error(io_failure?RestoreState::IoFailure:RestoreState::Corrupt,
        "no valid batch checkpoint generation; preserve damaged evidence and select another checkpoint directory");
    std::cout<<"Stage1 batch checkpoint: path="<<batch_path<<" curves="<<curves<<'\n';
}

bool ecm_stage1_checkpoint_pending() {
    std::lock_guard<std::mutex> lock(batch_mutex);return payload_validation_pending;
}
void ecm_stage1_validate_pending_checkpoint(const EcmStage1PayloadValidator &validate) {
    using namespace ecm_stage1_checkpoint;
    std::lock_guard<std::mutex> lock(batch_mutex);
    if(!payload_validation_pending)return;
    if(!pending_batch)throw Error(RestoreState::Corrupt,"checkpoint selection already failed");
    if(!validate)throw Error(RestoreState::Incompatible,"missing native checkpoint validator");
    // At most one outer payload generation is retained while candidates are tried.
    auto current=std::move(pending_batch);auto path=std::move(pending_path);
    auto older=std::move(pending_older);bool io_failure=false;
    for(size_t index=0;;++index) {
        if(current) {
            auto sigmas=current->sigmas;
            if(validate(current->records,sigmas)) {
                if(!sigma_matches(sigmas))throw Error(RestoreState::Incompatible,"batch checkpoint payload sigma differs from requested sigma");
                current->sigmas.swap(sigmas);accept_batch(*current,path);payload_validation_pending=false;return;
            }
            std::cout<<"Stage1 checkpoint payload rejected: path="<<path<<"; trying older generation\n";
            current.reset();
        }
        if(index>=older.size())break;
        path=older[index];std::string bytes;
        const auto read=ecm_stage1_read_checkpoint_blob(path,bytes);
        if(read==BlobRead::IoFailure){io_failure=true;continue;}
        if(read==BlobRead::Missing)continue;
        auto next=std::make_unique<DecodedBatch>();
        const auto parsed=decode_batch(bytes,batch_identity,uint32_t(batch_records.size()),*next);
        // Older unrelated generations cannot become another task's resume state.
        if(parsed!=Decode::Valid || !sigma_matches(next->sigmas))continue;
        current=std::move(next);
    }
    throw Error(io_failure?RestoreState::IoFailure:RestoreState::Corrupt,"no valid mathematical batch checkpoint generation");
}

bool ecm_stage1_saved_sigma(uint32_t curve,uint64_t &sigma) {
    std::lock_guard<std::mutex> lock(batch_mutex);
    if(payload_validation_pending || curve>=batch_sigmas.size() || !batch_sigmas[curve])return false;
    sigma=batch_sigmas[curve];return true;
}
uint64_t ecm_stage1_batch_sigma(uint32_t curve,uint64_t proposed) {
    std::lock_guard<std::mutex> lock(batch_mutex);
    if(payload_validation_pending || batch_path.empty() || curve>=batch_sigmas.size())return proposed;
    if(batch_sigmas[curve])return batch_sigmas[curve];
    batch_sigmas[curve]=proposed;
    // Every lane is seeded before arithmetic. Persist once at the end of that
    // pass; CPU codecs may additionally persist seed/state records along it.
    if(curve+1==batch_sigmas.size()) {
        try {
            if(!flush_batch())throw ecm_stage1_checkpoint::Error(
                ecm_stage1_checkpoint::RestoreState::IoFailure,"cannot persist batch checkpoint sigma");
        }catch(...){batch_sigmas[curve]=0;throw;}
    }
    return proposed;
}
ecm_stage1_checkpoint::BlobRead ecm_stage1_read_checkpoint_blob(const std::string &path,std::string &bytes) {
    using namespace ecm_stage1_checkpoint;
    if(ecm_stage1_is_curve_record(path)) {
        std::lock_guard<std::mutex> lock(batch_mutex);
        if(payload_validation_pending)throw Error(RestoreState::Incompatible,"native payload read before validation");
        bytes=batch_records.at(record_index(path));return bytes.empty()?BlobRead::Missing:BlobRead::Read;
    }
    std::ifstream in(path,std::ios::binary);
    if(!in) {std::error_code error;const bool exists=std::filesystem::exists(path,error);
        bytes.clear();return exists || error?BlobRead::IoFailure:BlobRead::Missing;}
    bytes.assign(std::istreambuf_iterator<char>(in),{});
    if(in.bad()){bytes.clear();return BlobRead::IoFailure;}
    return BlobRead::Read;
}
bool ecm_stage1_read_blob(const std::string &path,std::string &bytes) {
    return ecm_stage1_read_checkpoint_blob(path,bytes)==ecm_stage1_checkpoint::BlobRead::Read;
}

bool ecm_stage1_write_blob(const std::string &path,const std::string &bytes) {
    if(ecm_stage1_is_curve_record(path)) {
        std::lock_guard<std::mutex> lock(batch_mutex);
        if(payload_validation_pending)throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,"native checkpoint write before validation");
        std::string staged(bytes);auto &record=batch_records.at(record_index(path));record.swap(staged);
        try {if(flush_batch())return true;}
        catch(...){record.swap(staged);throw;}
        record.swap(staged);return false;
    }
    const auto temp=ecm_stage1_checkpoint_temp_path(path);
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));out.close();
    return bool(out) && ecm_stage1_commit_checkpoint(path,temp);
}
bool ecm_stage1_import_gpu_payload(const std::string &bytes,uint64_t first_sigma) {
    if(first_sigma>UINT64_MAX-(batch_sigmas.empty()?0:batch_sigmas.size()-1))throw std::runtime_error("legacy checkpoint sigma range overflow");
    std::vector<uint64_t> sigmas(batch_sigmas.size());
    for(size_t i=0;i<sigmas.size();++i)sigmas[i]=first_sigma+i;
    return ecm_stage1_import_gpu_payload(bytes,sigmas);
}
bool ecm_stage1_import_gpu_payload(const std::string &bytes,const std::vector<uint64_t> &sigmas) {
    std::lock_guard<std::mutex> lock(batch_mutex);
    if(payload_validation_pending || batch_path.empty() || batch_records.empty() || bytes.empty() || sigmas.size()!=batch_sigmas.size())return false;
    for(const auto &record:batch_records)if(!record.empty())return false;
    auto staged_sigmas=sigmas;std::string staged(bytes);
    batch_sigmas.swap(staged_sigmas);batch_records[0].swap(staged);
    try {
        if(!flush_batch())throw ecm_stage1_checkpoint::Error(
            ecm_stage1_checkpoint::RestoreState::IoFailure,"cannot import legacy GPU checkpoint");
    }catch(...){batch_sigmas.swap(staged_sigmas);batch_records[0].swap(staged);throw;}
    return true;
}
void ecm_stage1_end_batch(bool completed) {
    pending_batch.reset();pending_path.clear();pending_older.clear();payload_validation_pending=false;
    const auto path=std::move(batch_path);batch_path.clear();batch_identity.clear();batch_records.clear();batch_sigmas.clear();
    if(completed && !path.empty())ecm_stage1_remove_checkpoints(path);
#ifdef _WIN32
    if(batch_lock!=INVALID_HANDLE_VALUE){CloseHandle(batch_lock);batch_lock=INVALID_HANDLE_VALUE;}
#else
    if(batch_lock>=0){::flock(batch_lock,LOCK_UN);::close(batch_lock);batch_lock=-1;}
#endif
}
