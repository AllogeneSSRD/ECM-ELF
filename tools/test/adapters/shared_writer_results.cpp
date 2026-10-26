#include "shared_writer_pause.h"
bool scheduled_replace_file(const std::filesystem::path& from, const std::filesystem::path& to) {
    shared_writer_before_publish(to);
    return ecm_replace_file(from, to);
}
#define ecm_replace_file scheduled_replace_file
#include "../../../src/io/results.cpp"
#undef ecm_replace_file
