#pragma once
#include "../../service/stage2/runtime_types.h"
namespace s2prod {
extern HANDLE inherited_stop_event;
fs::path executable();
void poll_inherited_stop();
BOOL WINAPI console_control(DWORD event);
int child_run(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings);
}
