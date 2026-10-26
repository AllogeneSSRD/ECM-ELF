#include "shared_writer_pause.h"
namespace ecm_io {
void scheduled_result_write(const std::filesystem::path& path, const std::string& text,
                            const std::string& suffix = ".publish.tmp") {
    shared_writer_before_publish(path);
    write_atomic(path, text, suffix);
}
}
#define write_atomic scheduled_result_write
#include "../../../src/io/results.cpp"
#undef write_atomic
