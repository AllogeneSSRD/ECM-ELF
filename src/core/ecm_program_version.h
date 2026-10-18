#pragma once

// Public software identity. Source/build fingerprints belong in diagnostics.
namespace ecm_program {
inline constexpr char name[] = "ECM-ELF";
inline constexpr char version[] = "0.38.4";
#if defined(_WIN32)
inline constexpr char host_os[] = "Windows";
#elif defined(__ANDROID__)
inline constexpr char host_os[] = "Android";
#elif defined(__APPLE__)
inline constexpr char host_os[] = "macOS";
#elif defined(__linux__)
inline constexpr char host_os[] = "Linux";
#else
inline constexpr char host_os[] = "unknown";
#endif
#if defined(_M_X64) || defined(__x86_64__)
inline constexpr char host_arch[] = "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
inline constexpr char host_arch[] = "aarch64";
#elif defined(_M_IX86) || defined(__i386__)
inline constexpr char host_arch[] = "x86";
#else
inline constexpr char host_arch[] = "unknown";
#endif
}
