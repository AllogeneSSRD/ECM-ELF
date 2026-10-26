#pragma once
#include <cstdint>
// Process-wide stop state; handlers only request a stop, never perform I/O.
void ecm_stage1_install_stop_handler();
bool ecm_stage1_stop_requested();
bool ecm_stage1_set_stop_handle(uintptr_t value);
