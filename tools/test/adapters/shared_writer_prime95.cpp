#include "shared_writer_pause.h"
namespace ecm_io {
void scheduled_write_atomic(const std::filesystem::path& path, const std::string& text,
                            const std::string& suffix) {
    shared_writer_before_publish(path);
    write_atomic(path, text, suffix);
    if (GetEnvironmentVariableA("ECM_TEST_AFTER_REPLACE", nullptr, 0))
        throw PublicationUncertain("injected acknowledgement failure after real replacement");
}
} // namespace ecm_io
#define write_atomic scheduled_write_atomic
#include "../../../src/io/prime95.cpp"
#undef write_atomic
