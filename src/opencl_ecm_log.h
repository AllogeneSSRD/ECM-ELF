#pragma once

#include <cstdarg>
#include <cstdio>

// Install timestamp prefixing for std::cout/std::cerr.
void ecm_install_timestamped_iostreams();
bool ecm_log_timestamp_enabled();

// Mirror every timestamped line (both the C++ streams and ecm_ts_* output) to
// an extra FILE* (e.g. screen.log). Pass nullptr to disable. Used by the queue
// manager so a single run writes to both the console and the log file.
void ecm_log_set_mirror(FILE *mirror);

// Progress-bar colour. `name` is one of none|red|green|yellow|blue|magenta|cyan|
// white|grey (unknown → none). The returned code is an ANSI escape sequence.
void ecm_log_set_progress_color(const char *name);
const char *ecm_log_progress_color_code();   // e.g. "\033[36m" ("" when none)
const char *ecm_log_progress_color_reset();  // "\033[0m" ("" when none)

// Enable ANSI escape handling on the Windows console (no-op elsewhere).
void ecm_enable_console_ansi();

// Timestamped wrappers for C stdio output.
int ecm_ts_vfprintf(FILE *stream, const char *fmt, va_list ap);
int ecm_ts_fprintf(FILE *stream, const char *fmt, ...);
