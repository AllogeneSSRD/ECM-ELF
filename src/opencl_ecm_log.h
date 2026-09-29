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

// How often a PROGRESS line may reach the mirror file (ini key progress_log_seconds):
//   > 0 : at most one per N seconds   (default 60)
//   = 0 : no progress line in the file at all
//   < 0 : every progress line (the pre-D4 behaviour)
// The console/pipe is never rate-limited -- the GUI tails the worker's stdout and needs
// the ~200 ms cadence -- and a line reporting 100.0% always reaches the file, so a
// finished task is always visible in the log. See docs/DEV_ECM_GUI.md 7.2.
void ecm_log_set_progress_log_seconds(double seconds);
double ecm_log_progress_log_seconds();

// Enable ANSI escape handling on the Windows console (no-op elsewhere).
void ecm_enable_console_ansi();

// Timestamped wrappers for C stdio output.
int ecm_ts_vfprintf(FILE *stream, const char *fmt, va_list ap);
int ecm_ts_fprintf(FILE *stream, const char *fmt, ...);
