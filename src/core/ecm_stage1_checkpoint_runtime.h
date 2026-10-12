#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// One process-wide stop request, shared by CUDA, OpenCL and CPU Stage1.
// Signal handlers only set the flag; checkpoint I/O happens at a safe boundary.
void ecm_stage1_install_stop_handler();
bool ecm_stage1_stop_requested();
bool ecm_stage1_set_stop_handle(uintptr_t value);

constexpr unsigned ECM_STAGE1_MAX_BACKUPS = 16;
void ecm_stage1_set_backup_limit(unsigned count);
unsigned ecm_stage1_backup_limit();

std::string ecm_stage1_backup_path(const std::string &path, unsigned generation);
std::vector<std::string> ecm_stage1_checkpoint_candidates(const std::string &path);
std::string ecm_stage1_checkpoint_temp_path(const std::string &path);
// A valid backup was selected; do not rotate the damaged primary over it.
void ecm_stage1_mark_bad_primary(const std::string &path);
bool ecm_stage1_commit_checkpoint(const std::string &path, const std::string &temp);
void ecm_stage1_remove_checkpoints(const std::string &path);

// A checksum trailer covers the complete binary header and payload. Older files
// without the trailer remain readable; every newly written file has one.
uint64_t ecm_stage1_checksum(uint64_t sum, const void *data, size_t size);
constexpr uint64_t ECM_STAGE1_CHECKSUM_SEED = 1469598103934665603ull;
