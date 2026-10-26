#include "shared_writer_pause.h"
BOOL WINAPI scheduled_MoveFileExA(LPCSTR from, LPCSTR to, DWORD flags) {
    shared_writer_before_publish(std::filesystem::u8path(to));
    return MoveFileExA(from, to, flags);
}
#define MoveFileExA scheduled_MoveFileExA
#include "../../../src/ecm/application/gui/worktodo_gen.cpp"
#undef MoveFileExA
