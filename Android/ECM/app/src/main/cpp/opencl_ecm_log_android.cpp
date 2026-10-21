#include "ecm/platform/logging.h"

#include "ecm_log_android.h"
#include "jni_utf8.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string* g_capture = nullptr;
thread_local std::string g_line_pending;

JNIEnv* g_listener_env = nullptr;
jobject g_listener = nullptr;
jmethodID g_listener_on_line = nullptr;

void emit_listener_text(const char* text, size_t len) {
    if (g_listener == nullptr || g_listener_env == nullptr || g_listener_on_line == nullptr ||
        text == nullptr || len == 0) {
        return;
    }
    g_line_pending.append(text, len);
    for (;;) {
        const size_t nl = g_line_pending.find('\n');
        if (nl == std::string::npos) {
            break;
        }
        const std::string line = g_line_pending.substr(0, nl + 1);
        g_line_pending.erase(0, nl + 1);
        const jstring jline = new_jstring_utf8(g_listener_env, line);
        if (jline != nullptr) {
            g_listener_env->CallVoidMethod(g_listener, g_listener_on_line, jline);
            if (g_listener_env->ExceptionCheck()) {
                g_listener_env->ExceptionClear();
            }
            g_listener_env->DeleteLocalRef(jline);
        }
    }
}

void flush_listener_pending() {
    if (g_line_pending.empty() || g_listener == nullptr || g_listener_env == nullptr ||
        g_listener_on_line == nullptr) {
        g_line_pending.clear();
        return;
    }
    const jstring jline = new_jstring_utf8(g_listener_env, g_line_pending);
    if (jline != nullptr) {
        g_listener_env->CallVoidMethod(g_listener, g_listener_on_line, jline);
        if (g_listener_env->ExceptionCheck()) {
            g_listener_env->ExceptionClear();
        }
        g_listener_env->DeleteLocalRef(jline);
    }
    g_line_pending.clear();
}

} // namespace

void android_ecm_log_begin(std::string* out) {
    g_capture = out;
    g_line_pending.clear();
}

void android_ecm_log_end() {
    flush_listener_pending();
    g_capture = nullptr;
}

void android_ecm_log_set_listener(JNIEnv* env, jobject listener) {
    android_ecm_log_clear_listener(env);
    if (env == nullptr || listener == nullptr) {
        return;
    }
    g_listener_env = env;
    g_listener = env->NewGlobalRef(listener);
    const jclass cls = env->GetObjectClass(listener);
    if (cls == nullptr) {
        android_ecm_log_clear_listener(env);
        return;
    }
    g_listener_on_line =
        env->GetMethodID(cls, "onLine", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cls);
    if (g_listener_on_line == nullptr) {
        android_ecm_log_clear_listener(env);
    }
}

void android_ecm_log_clear_listener(JNIEnv* env) {
    (void)env;
    if (g_listener != nullptr && g_listener_env != nullptr) {
        g_listener_env->DeleteGlobalRef(g_listener);
    }
    g_listener = nullptr;
    g_listener_env = nullptr;
    g_listener_on_line = nullptr;
    g_line_pending.clear();
}

bool android_ecm_log_listener_active() {
    return g_listener != nullptr;
}

bool ecm_log_timestamp_enabled() {
    return false;
}

void ecm_install_timestamped_iostreams() {}

int ecm_ts_vfprintf(FILE* stream, const char* fmt, va_list ap) {
    char buf[8192];
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    if (n <= 0) {
        return n;
    }
    const int clipped = (n < static_cast<int>(sizeof(buf))) ? n : static_cast<int>(sizeof(buf) - 1);
    if (g_capture != nullptr) {
        g_capture->append(buf, static_cast<size_t>(clipped));
    }
    if (g_listener != nullptr) {
        emit_listener_text(buf, static_cast<size_t>(clipped));
    }
    if (g_capture == nullptr && g_listener == nullptr) {
        return vfprintf(stream, fmt, ap);
    }
    return clipped;
}

int ecm_ts_fprintf(FILE* stream, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int n = ecm_ts_vfprintf(stream, fmt, ap);
    va_end(ap);
    return n;
}
