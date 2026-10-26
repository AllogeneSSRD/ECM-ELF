#include "ecm/stage1/progress.h"
#include "cpu_support.h"
#include "ecm/backend/cpu/arithmetic/ifma/simd_mont_ifma.h"
#include "ecm/stage1/preparation.h"
#include "ecm/platform/logging.h"
#include "../../../include/indicators/indicators.hpp"
#include <chrono>
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif
namespace ecm_cpu_stage1 {
using ecm_stage1::stage1_now_ms;
using ecm_stage1::stdout_is_tty_local;
std::mutex g_stage1_out_mutex;
std::atomic<uint32_t> g_stage1_done{0};

int to_ifma_field_mode(ecm_stage1::SimdReduction reduction) {
    switch (reduction) {
    case ecm_stage1::SimdReduction::Auto:
        return IFMA_FIELD_AUTO;
    case ecm_stage1::SimdReduction::Montgomery:
        return IFMA_FIELD_MONT;
    case ecm_stage1::SimdReduction::MersenneFold:
        return IFMA_FIELD_MERS;
    }
    return IFMA_FIELD_AUTO;
}

static indicators::ProgressBar* g_stage1_bar = nullptr;

/* ---------------------------------------------------------------------------
 * 速率显示 (对齐 src/cuda/stage1/montgomery.cu 的 print_progress)
 *
 * CUDA 那套的要点有两个, 这里都照做:
 *  1) 平均的是"速度样本"而不是时间: 每个采样点算 speed = delta_work/delta_t,
 *     放进环形窗口, avg = sum/count; 然后用 avg 反推 per_curve 与 remaining。
 *     这样并行批数/频率变化时显示不会乱跳 (直接平均时间就会跳)。
 *  2) TTY 与非 TTY 两种输出: 终端里原地 \r 更新; 重定向到日志时按衰减节奏打
 *     整行 (前 3 次、之后 10/100/1000/10000 的倍数), 否则日志会被刷爆。
 * work 单位是"曲线"(可为小数, 批内按 bit 比例折算), 与总曲线数同一量纲。
 * ------------------------------------------------------------------------- */
#define ED_SPEED_WINDOW 12

struct EdSpeedMeter {
    double ring[ED_SPEED_WINDOW];
    int count, idx;
    double sum;
    double t_last, w_last, t0;
    bool started; /* 第一个采样点只用来定基准, 不产生速度样本 */

    EdSpeedMeter() : count(0), idx(0), sum(0.0), t_last(0.0), w_last(0.0), t0(0.0), started(false) {
    }

    void reset() {
        count = 0;
        idx = 0;
        sum = 0.0;
        t_last = 0.0;
        w_last = 0.0;
        t0 = 0.0;
        started = false;
    }

    void sample(double now, double work) {
        if (!started) {
            started = true;
            t0 = now;
            t_last = now;
            w_last = work;
            return;
        }
        const double dt = now - t_last, dw = work - w_last;
        t_last = now;
        w_last = work;
        if (dt < 0.05 || dw <= 1e-12)
            return;                   /* 极小 dt (收尾跳变) 会造出虚高速度样本 */
        const double speed = dw / dt; /* 曲线/秒 */
        if (count < ED_SPEED_WINDOW) {
            ring[count++] = speed;
            sum += speed;
        } else {
            sum -= ring[idx];
            ring[idx] = speed;
            sum += speed;
            idx = (idx + 1) % ED_SPEED_WINDOW;
        }
    }

    double avg_speed() const {
        return count > 0 ? sum / (double)count : 0.0;
    }
};

static EdSpeedMeter g_speed;

/* 重定向到日志时的打印节奏 (与 CUDA 的 emit_progress_line 一致) */
static bool emit_progress_line(unsigned n) {
    return (n < 3u) || (n < 30u && n % 10u == 0u) || (n < 500u && n % 100u == 0u) ||
           (n < 5000u && n % 1000u == 0u) || (n % 10000u == 0u);
}

/* ASCII 进度条 (日志模式下用, 与 CUDA print_progress 的条形一致) */
static const char* progress_bar_ascii(double pct) {
    static char bar[41];
    const int width = 40;
    int filled = (int)(width * (pct / 100.0));
    if (filled < 0)
        filled = 0;
    if (filled > width)
        filled = width;
    for (int i = 0; i < width; i++)
        bar[i] = (i < filled) ? '=' : ' ';
    bar[width] = '\0';
    if (filled > 0 && filled < width)
        bar[filled - 1] = '>';
    return bar;
}

void stage1_progress_set(double done, uint32_t total) {
    if (total == 0)
        return;
    /* 进度/速率估计必须单调: 多线程下各线程上报的 (已完成曲线 + 本曲线比例) 会互相穿插
       (A 报 3.5, B 接着报 5.2, A 下一次又报 3.6), 不单调会产生"大增量/小时间"的虚高
       速度样本, 把显示速率放大十几倍。取运行最大值即可。 */
    static double s_max_done = 0.0;
    if (done <= 0.0) {
        g_speed.reset();
        s_max_done = 0.0;
    }
    if (done < s_max_done)
        done = s_max_done;
    else
        s_max_done = done;
    const double now = (double)stage1_now_ms() / 1000.0;
    g_speed.sample(now, done);

    double pct = 100.0 * done / (double)total;
    if (pct > 100.0)
        pct = 100.0;
    const double avg = g_speed.avg_speed();
    const double per_curve_s = avg > 0.0 ? 1.0 / avg : 0.0; /* 由均值反推 */
    const double elapsed_s = (g_speed.t0 > 0.0) ? now - g_speed.t0 : 0.0;
    double remaining_s = 0.0;
    if (avg > 0.0 && done < (double)total)
        remaining_s = ((double)total - done) / avg;

    if (!stdout_is_tty_local()) {
        /* 日志模式: 不打 \r, 按衰减节奏打整行 (带时间戳, 会被 mirror 到 log 文件)。
           但"只看批号"是错的: 批号会从 checkpoint 恢复, 恢复后的运行只在 n % 10000 == 0
           时才打 —— 每批约 1 s 的话就是 ~1.7 小时一行, GUI 会一直显示"等待第一行进度"
           (2026-09-29 实测)。这里同样按 **时间** 兜底: 距上一行 ≥200 ms 就打一行,
           与交互式分支每批更新的节奏一致。 */
        static unsigned lines = 0;
        static std::atomic<uint32_t> last_emitted(0xFFFFFFFFu);
        static double last_line_s = -1.0e9;
        const uint32_t done_i = (uint32_t)done;
        const bool by_time = (now - last_line_s) >= 0.2;
        if (done >= (double)total || by_time || emit_progress_line(++lines)) {
            last_emitted.store(done_i, std::memory_order_relaxed);
            last_line_s = now;
            ecm_ts_fprintf(stdout,
                           "stage1: [%s] %.1f%%  %.1f/%u (~%.2f s/curve)  "
                           "elapsed %.1fs  ETA %.1fs\n",
                           progress_bar_ascii(pct), pct, done, total, per_curve_s, elapsed_s,
                           remaining_s);
        }
        return;
    }

    if (!g_stage1_bar)
        return;
    try {
        g_stage1_bar->set_progress((float)pct);
        char postfix[200];
        snprintf(postfix, sizeof(postfix),
                 "%.1f%%  %.1f/%u (~%.2f s/curve)  elapsed %.1fs  ETA %.1fs", pct, done, total,
                 per_curve_s, elapsed_s, remaining_s);
        g_stage1_bar->set_option(indicators::option::PostfixText{postfix});
        if (done >= (double)total)
            g_stage1_bar->mark_as_completed();
    } catch (...) {
        /* 进度条永远不能影响 stage-1 的正确性 */
    }
}

/* ---------------------------------------------------------------------------
 * 进度条对象: Edwards 与 Montgomery 各跑一个 stage-1 任务, 所以做成 create/destroy
 * 一对, 由 run_*_stage1 自己持有。g_stage1_bar 只是给 worker 回调看的"当前条",
 * destroy 里必须清空 —— 否则调用方返回后它还指着已析构的局部对象。
 *
 * 只对 TTY 建条: 重定向到日志时 stage1_progress_set() 走 ASCII 整行输出, 不需要
 * 终端控制序列 (建条本身也会往 stdout 写东西, 日志里会多出垃圾)。
 * ------------------------------------------------------------------------- */
static indicators::ProgressBar* stage1_bar_create() {
    if (!stdout_is_tty_local())
        return nullptr;
    auto* bar = new indicators::ProgressBar{
        indicators::option::BarWidth{40},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PrefixText{"stage1: "},
        indicators::option::PostfixText{""},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::ForegroundColor{indicators::Color::cyan},
        indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};
    g_stage1_bar = bar;
    return bar;
}

static void stage1_bar_destroy(indicators::ProgressBar* bar) {
    g_stage1_bar = nullptr;
    if (!bar)
        return;
    try {
        bar->mark_as_completed();
    } catch (...) {
    }
    delete bar;
}

/* 批内进度: bits 粒度, 把"已完成曲线 + 本批已跑比例 × 批大小"折算成总进度。
   同时承担 checkpoint: 每 ckpt_ms(或收到 SIGINT) 就按 lane 把当前点落盘, 存档格式
   与标量路径**完全相同**, 所以后续用标量或 SIMD 续跑都能读。 */
#if defined(_MSC_VER)
// AVX512-F + DQ + IFMA 且 OS 已启用 zmm/opmask 状态 (XCR0) 才允许进 simd_edwards。
// 必须在基线 TU 里探测: simd_*.cpp 是 /arch:AVX512 编的, 在它内部执行任何代码都可能
// 让编译器在探测前就发出 AVX512 指令。
int driver_simd_isa_ok(void) {
    int r[4] = {0, 0, 0, 0};
    __cpuid(r, 0);
    if (r[0] < 7)
        return 0;
    __cpuidex(r, 1, 0);
    if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1))
        return 0; /* OSXSAVE, AVX */
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6ULL) != 0xE6ULL)
        return 0; /* opmask+ZMM_Hi256+Hi16_ZMM */
    __cpuidex(r, 7, 0);
    return (((r[1] >> 16) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 21) & 1)) ? 1 : 0;
}
#else
int driver_simd_isa_ok(void) {
    return 0;
} /* 非 MSVC 由 CMake 侧保证 */
#endif

/* ---------------------------------------------------------------------------
 * 亲核性 (Affinity): "1,3,5,7" -> {1,3,5,7}
 * 空 / "none" / "auto" -> 空列表 = 不绑定, 交给系统调度。
 * 第 t 个 worker 绑到 list[t % count]，因此列表长度通常取 = 线程数或物理核数。
 * ------------------------------------------------------------------------- */

#if defined(_WIN32)
void apply_thread_affinity(const std::vector<unsigned>& cpus, uint32_t t) {
    if (cpus.empty())
        return;
    const unsigned cpu = cpus[t % cpus.size()];
    if (cpu >= 64)
        return; /* 单处理器组只有 64 个逻辑 CPU 可用位图表达 */
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
}
#else
void apply_thread_affinity(const std::vector<unsigned>&, uint32_t) {
}
#endif

Stage1Progress::Stage1Progress() : bar_(stage1_bar_create()) {
}

Stage1Progress::~Stage1Progress() {
    finish();
}

void Stage1Progress::finish() noexcept {
    stage1_bar_destroy(bar_);
    bar_ = nullptr;
}
} // namespace ecm_cpu_stage1
