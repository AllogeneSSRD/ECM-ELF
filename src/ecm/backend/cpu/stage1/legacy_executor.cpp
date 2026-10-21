#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>     /* _mkdir for the local save directory */
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>   /* __rdtsc() for the Prime95-style sigma entropy mix */
#endif

#include <gmp.h>

#include "legacy_executor.h"
#include "../../../service/stage1/preparation.h"
#include "../../../storage/save/ecm_save.h"
#include "../../../storage/checkpoint/ecm_checkpoint.h"
#include "../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../../storage/checkpoint/ecm_mont_ckpt.h"
#include "edwards/ecm_edwards_cpu.h"
#include "montgomery/ecm_mont_cpu.h"
#include "montgomery/simd_mont_curve.h"
#include "../arithmetic/ifma/simd_mont_ifma.h"
#include "../../../storage/save/ecm_edwards_save.h"
#include "../../../storage/save/ecm_edwards_io.h"
#include "ecm/platform/logging.h"
#include <iomanip>
#include <regex>
#include "edwards/simd_edwards.h"   /* SoA 批量 Edwards 层 (AVX512-IFMA TU, 见 §13.8) */
#include "../../../../../include/indicators/indicators.hpp"
#include <intrin.h>
#include <windows.h>

namespace ecm_cpu_stage1 {
using ecm_stage1::Stage1Options;
using ecm_stage1::ExecutionOptions;
using ecm_stage1::ExecutionResult;
using ecm_stage1::stage1_publish_factor;
using ecm_stage1::compute_batch_s;
using ecm_stage1::random_sigma_u64;
using ecm_stage1::mont_format_save_name;
using ecm_stage1::fixed_sigma_batch_exceeds_prime95_range;
using ecm_stage1::ensure_dir;
using ecm_stage1::stdout_is_tty_local;
using ecm_stage1::stage1_now_ms;

static int to_ifma_field_mode(ecm_stage1::SimdReduction reduction) {
    switch (reduction) {
    case ecm_stage1::SimdReduction::Auto: return IFMA_FIELD_AUTO;
    case ecm_stage1::SimdReduction::Montgomery: return IFMA_FIELD_MONT;
    case ecm_stage1::SimdReduction::MersenneFold: return IFMA_FIELD_MERS;
    }
    return IFMA_FIELD_AUTO;
}

static bool edwards_capture_save(std::string &line,const mpz_t n,uint64_t sigma,uint64_t b1,
                                  const mpz_t qx,const mpz_t qz,const mpz_t factor,bool hit) {
    ecm_edwards::Integer x,g,inv;
    if(hit) {
        if(mpz_cmp_ui(factor,1)<=0 || mpz_cmp(factor,n)>=0 || !mpz_divisible_p(n,factor))return false;
        mpz_set(x.v,factor);
    } else {
        mpz_gcd(g.v,qz,n);
        if(mpz_cmp(g.v,n)==0) {
            ecm_ts_fprintf(stdout,"Edwards identity: sigma=%llu gcd(Qz,N)=N, no Stage2 point\n",
                           static_cast<unsigned long long>(sigma));
            line.clear();return true;
        }
        if(mpz_cmp_ui(g.v,1)!=0 || !mpz_invert(inv.v,qz,n))return false;
        mpz_mul(x.v,qx,inv.v);mpz_mod(x.v,x.v,n);
    }
    line=ecm_edwards::save_line(n,x.v,sigma,b1,hit);return true;
}

static std::mutex g_stage1_out_mutex;



// Local Stage1 text-save publication and batch checkpoint records.

// 创建目录 (已存在视为成功).


// 自我 checkpoint 上下文 + 进度回调 (写 STAGE1 存档)
struct EdwardsCheckpointCtx {
    const ExecutionOptions *opt;
    uint64_t sigma;
    uint64_t B1, B2;
    std::string save_path;
    long long interval_ms;
    long long last_ms;
    mpz_srcptr N;
    mpz_srcptr s;
    bool write_failed = false;
};

static int edwards_checkpoint_progress(void *ctx, const edwards_checkpoint_t *cur) {
    EdwardsCheckpointCtx *c = (EdwardsCheckpointCtx *)ctx;
    const long long now = stage1_now_ms();
    if (c->save_path.empty()) return ecm_stage1_stop_requested()?0:1;
    const bool due = c->interval_ms>0 && (now - c->last_ms >= c->interval_ms);
    if (!due && !ecm_stage1_stop_requested()) return 1;   // 继续

    mpz_t d, Px, Py;
    mpz_inits(d, Px, Py, NULL);
    edwards_atkin_morain(d, Px, Py, c->sigma, c->N);

    ecm_save_common cm;
    cm.k = c->opt->identity.handoff_k;
    cm.b = c->opt->identity.handoff_b;
    cm.n = c->opt->identity.handoff_n;
    cm.c = c->opt->identity.handoff_c;
    cm.curve = 1;
    cm.B1 = c->B1;
    cm.B2 = c->B2;
    cm.sigma = c->sigma;
    const uint32_t expbuf = (uint32_t)mpz_sizeinbase(c->s, 2);
    const uint32_t dict_size = (uint32_t)1u << (edwards_get_naf_w() - 2);
    const bool written = ecm_edwards_write_stage1(c->save_path, cm, 2, expbuf, cur->bitnum, dict_size,
                                 Px, Py, cur->Rx, cur->Ry, cur->Rz);
    c->last_ms = now;
    mpz_clears(d, Px, Py, NULL);
    if (!written) {
        c->write_failed = true;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 checkpoint %s\n",
                       c->save_path.c_str());
        return 0;
    }

    if (ecm_stage1_stop_requested()) {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stdout, "checkpoint written (stop): %s @ bitnum=%u\n",
                       c->save_path.c_str(), cur->bitnum);
        return 0;   // 中止
    }
    return 1;       // 继续
}

// ---------------------------------------------------------------------------
// 单条曲线的 stage-1 执行 (顺序/并行共用)
// ---------------------------------------------------------------------------
//
// 线程安全约定: N 与 s 只读共享 (GMP 允许同一 mpz_t 的并发只读访问);
// 输出 (factor_out / Qx,Qz / checkpoint 文件) 均按曲线索引独立, 无共享写.
// 返回: 1=因子, 0=无因子, -1=错误, 2=被 SIGINT 中止(已写 checkpoint).
/* 已完成曲线数 (文件作用域, 让"曲线内"的进度回调也能读到) */
static std::atomic<uint32_t> g_stage1_done(0);

/* 定义在后面 (需要 g_stage1_bar / stage1_progress_set), 这里先声明给调用点用 */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur);

static bool edwards_completed_read(uint32_t curve,uint64_t sigma,mpz_t factor,int &rc,std::string &line) {
    std::string bytes;if(!ecm_stage1_read_blob(ecm_stage1_curve_record(curve),bytes))return false;
    std::istringstream in(bytes);std::string tag,value,extra;uint64_t saved_sigma;
    if(!(in>>tag) || tag!="ECM_EDWARDS_DONE_V1" || !(in>>saved_sigma>>rc>>value>>std::quoted(line)) ||
       (in>>extra) || saved_sigma!=sigma || (rc!=0 && rc!=1) || mpz_set_str(factor,value.c_str(),16))return false;
    return true;
}
static bool edwards_completed_write(uint32_t curve,uint64_t sigma,const mpz_t factor,int rc,const std::string &line) {
    std::ostringstream out;out<<"ECM_EDWARDS_DONE_V1\n"<<sigma<<' '<<rc<<' '<<ecm_edwards::hex(factor)<<' '<<std::quoted(line)<<'\n';
    return ecm_stage1_write_blob(ecm_stage1_curve_record(curve),out.str());
}

static int edwards_run_one_curve(mpz_srcptr N, mpz_srcptr s,
                                 const ExecutionOptions &opt, double B1, double B2,
                                 uint32_t idx, uint64_t sigma, uint32_t curves,
                                 mpz_t factor_out, const std::string &ckpt_path,std::string &save_line) {
    mpz_t Qx, Qz;
    mpz_inits(Qx, Qz, NULL);

    int rc;
    if(!ckpt_path.empty() && edwards_completed_read(idx,sigma,factor_out,rc,save_line)) {
        mpz_clears(Qx,Qz,NULL);return rc;
    }
    {
        EdwardsCheckpointCtx ck;
        ck.opt = &opt;
        ck.sigma = sigma;
        ck.B1 = (uint64_t)B1;
        ck.B2 = (uint64_t)B2;
        ck.save_path = ckpt_path;
        ck.interval_ms = (long long)opt.persistence.ckpt_ms;
        ck.last_ms = stage1_now_ms();
        ck.N = N;
        ck.s = s;

        // 检测并恢复 STAGE1 checkpoint
        edwards_checkpoint_t resume;
        edwards_checkpoint_init(&resume);
        {
            ecm_save_common rcm;
            uint64_t rsp; uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy, rex, rey, rez;
            mpz_inits(rdx, rdy, rex, rey, rez, NULL);
            if (ecm_edwards_read_stage1(ckpt_path, rcm, &rsp, &rebs, &rbn, &rds,
                                        rdx, rdy, rex, rey, rez)) {
                if (rcm.sigma == sigma && rcm.B1 == (uint64_t)B1 && rbn > 0) {
                    resume.bitnum = rbn;
                    mpz_set(resume.Rx, rex);
                    mpz_set(resume.Ry, rey);
                    mpz_set(resume.Rz, rez);
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stdout, "curve %u: resume STAGE1 checkpoint @ bitnum=%u\n",
                                   idx + 1, rbn);
                }
            }
            mpz_clears(rdx, rdy, rex, rey, rez, NULL);
        }

        const uint32_t chunk_bits = static_cast<uint32_t>(std::clamp<size_t>(mpz_sizeinbase(s,2)/512,64,4096));
        rc = edwards_stage1_curve_progress(factor_out, Qx, Qz, N, sigma, s, chunk_bits,
                                           resume.bitnum > 0 ? &resume : nullptr,
                                           edwards_checkpoint_progress_bar, &ck);
        edwards_checkpoint_clear(&resume);

        if (rc == 2) {
            mpz_clears(Qx, Qz, NULL);
            return ck.write_failed ? ECM_ERROR : 2;
        }
    }

    // Capture the canonical text-save record and retain the completed result.
    if (rc == 0 || rc == 1) {
        if(!edwards_capture_save(save_line,N,sigma,uint64_t(B1),Qx,Qz,factor_out,rc==1))rc=ECM_ERROR;
        if(rc>=0 && !ckpt_path.empty() && !edwards_completed_write(idx,sigma,factor_out,rc,save_line))rc=ECM_ERROR;
    }
    mpz_clears(Qx, Qz, NULL);
    return rc;
}

// ---------------------------------------------------------------------------
// 多曲线并行 (每线程一条曲线, 动态取号)
// ---------------------------------------------------------------------------

struct EdwardsWorkerCtx {
    mpz_srcptr N;
    mpz_srcptr s;
    const ExecutionOptions *opt;
    double B1, B2;
    uint32_t curves;
    const uint64_t *sigmas;
    const std::string *ckpt_paths;
    mpz_t *factors;
    int *array_found;
    std::string *save_lines;
    std::atomic<uint32_t> *next;
    std::atomic<int> *aborted;
    /* SIMD 批量 (8 曲线/批) 相关 */
    int simd_ok = 0;
    std::atomic<uint32_t> *done_ctr = nullptr;
};

// ---------------------------------------------------------------------------
// SIMD 批量 stage-1 (8 曲线/批, AVX512-IFMA)
//
// 每批构造独立字典，逐 lane 生成统一文本记录，整个任务发布后清理检查点。
// SIMD 进度回调写入各 lane 检查点；同批位号一致时可恢复。
// 构造或字典逆元异常时逐曲线检查因子，不能继续使用失效的仿射字典。
// ---------------------------------------------------------------------------

/* ---------------------------------------------------------------------------
 * stage-1 进度条 (照 opencl_ecm_stage1.cpp 的 CUDA host 写法)
 *
 * 批量模式下一批 8 条曲线是不可分割的工作单元, 所以进度以"曲线"为粒度、每批跳一格;
 * 更新点都在 g_stage1_out_mutex 保护下, 避免多条 worker 线程同时动光标。
 * ------------------------------------------------------------------------- */

static indicators::ProgressBar *g_stage1_bar = nullptr;

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
    bool started;                    /* 第一个采样点只用来定基准, 不产生速度样本 */
    EdSpeedMeter() : count(0), idx(0), sum(0.0), t_last(0.0), w_last(0.0), t0(0.0),
                     started(false) {}
    void reset() { count = 0; idx = 0; sum = 0.0; t_last = 0.0; w_last = 0.0; t0 = 0.0;
                   started = false; }
    void sample(double now, double work) {
        if (!started) { started = true; t0 = now; t_last = now; w_last = work; return; }
        const double dt = now - t_last, dw = work - w_last;
        t_last = now;
        w_last = work;
        if (dt < 0.05 || dw <= 1e-12) return;       /* 极小 dt (收尾跳变) 会造出虚高速度样本 */
        const double speed = dw / dt;               /* 曲线/秒 */
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
    double avg_speed() const { return count > 0 ? sum / (double)count : 0.0; }
};

static EdSpeedMeter g_speed;



/* 重定向到日志时的打印节奏 (与 CUDA 的 emit_progress_line 一致) */
static bool emit_progress_line(unsigned n) {
    return (n < 3u) || (n < 30u && n % 10u == 0u) || (n < 500u && n % 100u == 0u) ||
           (n < 5000u && n % 1000u == 0u) || (n % 10000u == 0u);
}

/* ASCII 进度条 (日志模式下用, 与 CUDA print_progress 的条形一致) */
static const char *progress_bar_ascii(double pct) {
    static char bar[41];
    const int width = 40;
    int filled = (int)(width * (pct / 100.0));
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    for (int i = 0; i < width; i++) bar[i] = (i < filled) ? '=' : ' ';
    bar[width] = '\0';
    if (filled > 0 && filled < width) bar[filled - 1] = '>';
    return bar;
}

static void stage1_progress_set(double done, uint32_t total) {
    if (total == 0) return;
    /* 进度/速率估计必须单调: 多线程下各线程上报的 (已完成曲线 + 本曲线比例) 会互相穿插
       (A 报 3.5, B 接着报 5.2, A 下一次又报 3.6), 不单调会产生"大增量/小时间"的虚高
       速度样本, 把显示速率放大十几倍。取运行最大值即可。 */
    static double s_max_done = 0.0;
    if (done <= 0.0) { g_speed.reset(); s_max_done = 0.0; }
    if (done < s_max_done) done = s_max_done; else s_max_done = done;
    const double now = (double)stage1_now_ms() / 1000.0;
    g_speed.sample(now, done);

    double pct = 100.0 * done / (double)total;
    if (pct > 100.0) pct = 100.0;
    const double avg = g_speed.avg_speed();
    const double per_curve_s = avg > 0.0 ? 1.0 / avg : 0.0;              /* 由均值反推 */
    const double elapsed_s = (g_speed.t0 > 0.0) ? now - g_speed.t0 : 0.0;
    double remaining_s = 0.0;
    if (avg > 0.0 && done < (double)total) remaining_s = ((double)total - done) / avg;

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
                           progress_bar_ascii(pct), pct, done, total, per_curve_s,
                           elapsed_s, remaining_s);
        }
        return;
    }

    if (!g_stage1_bar) return;
    try {
        g_stage1_bar->set_progress((float)pct);
        char postfix[200];
        snprintf(postfix, sizeof(postfix),
                 "%.1f%%  %.1f/%u (~%.2f s/curve)  elapsed %.1fs  ETA %.1fs",
                 pct, done, total, per_curve_s, elapsed_s, remaining_s);
        g_stage1_bar->set_option(indicators::option::PostfixText{postfix});
        if (done >= (double)total) g_stage1_bar->mark_as_completed();
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
static indicators::ProgressBar *stage1_bar_create() {
    if (!stdout_is_tty_local()) return nullptr;
    auto *bar = new indicators::ProgressBar{
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

static void stage1_bar_destroy(indicators::ProgressBar *bar) {
    g_stage1_bar = nullptr;
    if (!bar) return;
    try { bar->mark_as_completed(); } catch (...) {}
    delete bar;
}

/* 批内进度: bits 粒度, 把"已完成曲线 + 本批已跑比例 × 批大小"折算成总进度。
   同时承担 checkpoint: 每 ckpt_ms(或收到 SIGINT) 就按 lane 把当前点落盘, 存档格式
   与标量路径**完全相同**, 所以后续用标量或 SIMD 续跑都能读。 */
struct EdSimdProg {
    std::atomic<uint32_t> *done_ctr;
    uint32_t curves, batch_first, batch_count;
    size_t   s_bits;
    long long t0_ms;
    const ExecutionOptions *opt;
    double   B1, B2;
    const uint64_t *sigmas;
    const ifma_ctx_t *mc;
    long long last_ms;
    long long interval_ms;
    bool write_failed = false;
};

static int edwards_simd_progress(void *p, size_t bits_done, size_t bits_total,
                                const uint64_t *Rx, const uint64_t *Ry, const uint64_t *Rz) {
    EdSimdProg *pr = (EdSimdProg *)p;
    const ifma_ctx_t *mc = pr->mc;

    /* --- checkpoint (与标量路径同一套写入器/路径/字段) --- */
    const long long now = stage1_now_ms();
    const bool due = !pr->opt->persistence.tmp_dir.empty() && ((pr->interval_ms>0 && now-pr->last_ms>=pr->interval_ms) || ecm_stage1_stop_requested());
    if (due && bits_done > 0) {
        if (mc) {
            mpz_t d, Px, Py, rx, ry, rz;
            mpz_inits(d, Px, Py, rx, ry, rz, NULL);
            size_t written = 0;
            for (uint32_t k = 0; k < pr->batch_count; k++) {
                const uint32_t i = pr->batch_first + k;
                const std::string path = ecm_stage1_curve_record(i);
                ecm_save_common cm;
                cm.k = pr->opt->identity.handoff_k;
                cm.b = pr->opt->identity.handoff_b;
                cm.n = pr->opt->identity.handoff_n;
                cm.c = pr->opt->identity.handoff_c;
                cm.curve = 1;
                cm.B1 = (uint64_t)pr->B1;
                cm.B2 = (uint64_t)pr->B2;
                cm.sigma = pr->sigmas[i];
                edwards_atkin_morain(d, Px, Py, pr->sigmas[i], mc->N);
                ifma_to_mpz_lane(rx, Rx, k, mc);
                ifma_to_mpz_lane(ry, Ry, k, mc);
                ifma_to_mpz_lane(rz, Rz, k, mc);
                const uint32_t expbuf = (uint32_t)pr->s_bits;
                const uint32_t dict_size = (uint32_t)1u << (edwards_get_naf_w() - 2);
                if (ecm_edwards_write_stage1(path, cm, 2, expbuf, (uint32_t)bits_done,
                                             dict_size, Px, Py, rx, ry, rz)) {
                    written++;
                } else {
                    pr->write_failed = true;
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 checkpoint %s\n",
                                   path.c_str());
                }
            }
            pr->last_ms = now;
            mpz_clears(d, Px, Py, rx, ry, rz, NULL);
            if (written && (ecm_stage1_stop_requested() || getenv("ED_SOA_DEBUG"))) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                ecm_ts_fprintf(stdout, "checkpoint written (simd batch %u): %zu lane(s) @ bitnum=%zu\n",
                               pr->batch_first, written, bits_done);
            }
        }
    }

    /* --- 进度条 --- */
    {
        const uint32_t base = pr->done_ctr ? pr->done_ctr->load(std::memory_order_relaxed) : 0;
        const double frac = (bits_total > 0) ? (double)bits_done / (double)bits_total : 0.0;
        const double done_f = (double)base + frac * (double)pr->batch_count;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done_f, pr->curves);
    }

    return (ecm_stage1_stop_requested() || pr->write_failed) ? 1 : 0;
}

/* 标量路径的"曲线内"进度: 包一层 checkpoint 回调, 按动态分块位数更新。
   没有它的话 B1 很大时一条曲线要跑几分钟, 进度条看着就是卡死的。
   (定义在这里而不是函数旁边, 因为要用到上面的 g_stage1_bar/stage1_progress_set。) */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur) {
    if (ctx) {
        EdwardsCheckpointCtx *ck = (EdwardsCheckpointCtx *)ctx;
        if (ck->opt && ck->s && ck->opt->stage1_curves > 0) {
            const size_t s_bits = (size_t)mpz_sizeinbase(ck->s, 2);
            if (s_bits > 0) {
                const uint32_t base = g_stage1_done.load(std::memory_order_relaxed);
                double frac = (double)cur->bitnum / (double)s_bits;
                if (frac > 1.0) frac = 1.0;
                const double done_f = (double)base + frac;
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                stage1_progress_set(done_f, ck->opt->stage1_curves);
            }
        }
    }
    return edwards_checkpoint_progress(ctx, cur);
}

static int edwards_run_batch_simd_impl(const ExecutionOptions &opt, mpz_srcptr N, mpz_srcptr s,
                                       double B1, double B2, uint32_t first, uint32_t count,
                                       const uint64_t *sigmas, mpz_t *factors, int *array_found,
                                       std::atomic<uint32_t> *done_ctr,std::string *save_lines) {
    if(!opt.persistence.tmp_dir.empty()) {
        bool completed=true;std::vector<int> statuses(count);
        for(uint32_t k=0;k<count;++k)if(!edwards_completed_read(first+k,sigmas[first+k],factors[first+k],statuses[k],save_lines[first+k])){completed=false;break;}
        if(completed) {
            for(uint32_t k=0;k<count;++k)array_found[first+k]=statuses[k]?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
            if(done_ctr)done_ctr->fetch_add(count,std::memory_order_relaxed);
            return 0;
        }
    }
    // Exceptional setup lanes need scalar factor/error classification before
    // building an affine SIMD dictionary (which requires invertible inputs).
    bool exceptional=false;
    for(uint32_t k=0;k<count;++k) {
        ecm_edwards::Integer d,x,y,a24,f;
        if(ecm_edwards::construct(d.v,x.v,y.v,a24.v,f.v,sigmas[first+k],N)!=ecm_edwards::Setup::Ready)exceptional=true;
    }
    auto run_exceptional=[&]() {
        for(uint32_t k=0;k<count;++k) {
            const uint32_t i=first+k;
            const auto path=opt.persistence.tmp_dir.empty()?std::string():ecm_stage1_curve_record(i);
            const int rc=edwards_run_one_curve(N,s,opt,B1,B2,i,sigmas[i],opt.stage1_curves,factors[i],path,save_lines[i]);
            if(rc==2 || rc<0)return rc;
            array_found[i]=rc==1?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
            if(rc==1 && !stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i]))return ECM_ERROR;
            if(done_ctr)done_ctr->fetch_add(1,std::memory_order_relaxed);
        }
        return 0;
    };
    if(exceptional)return run_exceptional();
    ed_soa_ctx_t ctx;
    /* 字典窗口跟随配置 (naf_w / ini), 不再硬编码 w=8:
       这样 SIMD 与标量用同一套 digit/字典, 存档逐字节可比, 也便于混合与续跑。
       代价是字典内存 = 3*2^(w-2)*8n 字节 (w=12/n=155 约 29 MB/批), 见启动信息。 */
    int w = edwards_get_naf_w();
    if (w < 3 || w > 12) w = 8;
    {
        const int rc = ed_soa_init_ex(&ctx, N, w, to_ifma_field_mode(opt.execution.selection.simd_reduction));
        if (rc != 0) {
            /* 保持与原来完全相同的控制流 (先尝试建 ctx, 失败就返回 -1), 只把
               "SIMD stage-1 internal error" 这句含糊的日志换成能看出原因的话:
               reduction=mersenne-fold 要求 N = 2^k-1 (k>=64), 这是最常见的误用。 */
            bool want_mers = (opt.execution.selection.simd_reduction == ecm_stage1::SimdReduction::MersenneFold);
            if (want_mers) {
                mpz_t t;
                mpz_init(t);
                mpz_add_ui(t, N, 1);
                const bool is_mersenne = (mpz_popcount(t) == 1 && mpz_scan1(t, 0) >= 64);
                mpz_clear(t);
                want_mers = !is_mersenne;
            }
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            if (want_mers)
                ecm_ts_fprintf(stderr,
                               "ERROR: --edwards-mersenne on was requested, but this N is not "
                               "2^k-1 with k >= 64 (N+1 is not a power of two), so the Mersenne "
                               "fold domain does not apply. Use --edwards-mersenne auto "
                               "(default) or off.\n");
            else
                ecm_ts_fprintf(stderr,
                               "ERROR: edwards SIMD context init failed (rc=%d) for a %zu-bit N "
                               "with dict w=%d.\n", rc, mpz_sizeinbase(N, 2), w);
            return -1;
        }
    }

    /* 第一次成功建 ctx 时把实际选中的域打印出来 (auto 会按 N 的形状决定) */
    {
        static std::atomic<int> field_logged(0);
        if (field_logged.fetch_add(1, std::memory_order_relaxed) == 0) {
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "field           : %s\n", ed_soa_field_name(&ctx));
        }
    }

    /* 不足 8 条时把最后一个 sigma 补满 (被补的 lane 结果丢弃) */
    uint64_t sg[8];
    for (uint32_t k = 0; k < 8; k++) sg[k] = sigmas[first + (k < count ? k : count - 1)];

    const int setup_rc=ed_soa_set_curves(&ctx,sg,8);
    if(setup_rc!=0){ed_soa_clear(&ctx);return setup_rc==-4?run_exceptional():-1;}

    EdSimdProg pr;
    pr.done_ctr = done_ctr;
    pr.curves = (done_ctr && opt.stage1_curves) ? opt.stage1_curves : count;
    pr.batch_first = first;
    pr.batch_count = count;
    pr.s_bits = mpz_sizeinbase(s, 2);
    pr.t0_ms = opt.stage1_t0_ms;
    pr.opt = &opt;
    pr.B1 = B1;
    pr.B2 = B2;
    pr.sigmas = sigmas;
    pr.last_ms = stage1_now_ms();
    pr.interval_ms = (long long)opt.persistence.ckpt_ms;
    pr.mc = &ctx.mc;
    /* ---- resume: 一批 8 条 lane 必须共用恢复点, 所以仅当**所有** lane 都有有效检查点
       且 bitnum 完全相同时才续跑 (这正是"整批一起被中止、一起落盘"的常态); 否则从头跑。 ---- */
    {
        std::vector<mpz_t> rxs(count), rys(count), rzs(count);
        for (uint32_t k = 0; k < count; k++) mpz_inits(rxs[k], rys[k], rzs[k], NULL);
        uint32_t common_bitnum = 0;
        int all_ok = (count > 0);
        for (uint32_t k = 0; k < count && all_ok; k++) {
            const uint32_t i = first + k;
            const std::string path = ecm_stage1_curve_record(i);
            ecm_save_common rcm;
            uint64_t rsp; uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy;
            mpz_inits(rdx, rdy, NULL);
            const bool ok = ecm_edwards_read_stage1(path, rcm, &rsp, &rebs, &rbn, &rds,
                                                    rdx, rdy, rxs[k], rys[k], rzs[k]);
            mpz_clears(rdx, rdy, NULL);
            if (!ok || rbn == 0 || rcm.sigma != sigmas[i] || rcm.B1 != (uint64_t)B1) { all_ok = 0; break; }
            if (k == 0) common_bitnum = rbn;
            else if (rbn != common_bitnum) { all_ok = 0; break; }
        }
        if (all_ok && common_bitnum > 0) {
            const int rr = ed_soa_set_resume(&ctx, common_bitnum, (int)count,
                                             rxs.data(), rys.data(), rzs.data());
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "simd batch %u: resume from .ckpt @ bitnum=%u (%u lane(s))%s\n",
                           first, common_bitnum, count,
                           rr == 0 ? "" : " -- unusable, restarting from scratch");
            if (rr != 0) ctx.resume_digit = 0;
        }
        for (uint32_t k = 0; k < count; k++) mpz_clears(rxs[k], rys[k], rzs[k], NULL);
    }
    ed_soa_set_progress(&ctx, edwards_simd_progress, &pr);

    mpz_t Qx[8], Qz[8], fac[8];
    for (int k = 0; k < 8; k++) mpz_inits(Qx[k], Qz[k], fac[k], NULL);
    const int aborted = ed_soa_stage1(&ctx, s, 8, Qx, Qz, fac);
    if (aborted) {                                  /* 中途停: 不写任何存档 */
        for (int k = 0; k < 8; k++) mpz_clears(Qx[k], Qz[k], fac[k], NULL);
        ed_soa_clear(&ctx);
        return pr.write_failed ? ECM_ERROR : 2;
    }

    int rc = 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = first + k;
        if (mpz_cmp_ui(fac[k], 1) > 0) {
            mpz_set(factors[i], fac[k]);
            array_found[i] = ECM_FACTOR_FOUND_STEP1;
            if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i]))rc=ECM_ERROR;
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                           i, (unsigned long long)sigmas[i]);
        }
        if(!edwards_capture_save(save_lines[i],N,sigmas[i],uint64_t(B1),Qx[k],Qz[k],fac[k],array_found[i]==ECM_FACTOR_FOUND_STEP1))rc=ECM_ERROR;
        if(rc==0 && !opt.persistence.tmp_dir.empty() && !edwards_completed_write(i,sigmas[i],factors[i],array_found[i]==ECM_FACTOR_FOUND_STEP1?1:0,save_lines[i]))rc=ECM_ERROR;
    }
    for (int k = 0; k < 8; k++) mpz_clears(Qx[k], Qz[k], fac[k], NULL);
    ed_soa_clear(&ctx);

    if (done_ctr) {
        const uint32_t done = done_ctr->fetch_add(count, std::memory_order_relaxed) + count;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done, opt.stage1_curves);
    }
    (void)s;
    return rc;
}

static void edwards_worker(EdwardsWorkerCtx *ctx) {
    const int use_simd = (ctx->opt->execution.selection.cpu_implementation != ecm_stage1::CpuImplementation::GmpScalar) && ctx->simd_ok;
    for (;;) {
        if (ecm_stage1_stop_requested()) { ctx->aborted->store(1, std::memory_order_relaxed); break; }
        uint32_t i, n;
        if (use_simd) {                                   /* 静态分批: 一批 8 条 */
            i = ctx->next->fetch_add(8, std::memory_order_relaxed);
            if (i >= ctx->curves) break;
            n = ctx->curves - i; if (n > 8) n = 8;
        } else {
            i = ctx->next->fetch_add(1, std::memory_order_relaxed);
            if (i >= ctx->curves) break;
            n = 1;
        }
        if (use_simd && n >= 2) {
            /* 批内至少 2 条才值得批量 (1 条走标量, 见 §13.6 的 hybrid 规则) */
            const int rc = edwards_run_batch_simd_impl(*ctx->opt, ctx->N, ctx->s, ctx->B1, ctx->B2,
                                                       i, n, ctx->sigmas, ctx->factors,
                                                       ctx->array_found, ctx->done_ctr,ctx->save_lines);
            if(rc==2){ctx->aborted->store(1,std::memory_order_relaxed);break;}
            if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  batch " << i << " -> SIMD stage-1 internal error" << std::endl;
                ctx->aborted->store(1, std::memory_order_relaxed);
                break;
            }
            continue;
        }
        if (use_simd && n == 1) { /* fallthrough to scalar for the tail curve */ }
        {
            const uint64_t sigma = ctx->sigmas[i];
            const int rc = edwards_run_one_curve(ctx->N, ctx->s, *ctx->opt, ctx->B1, ctx->B2,
                                                 i, sigma, ctx->curves,
                                                 ctx->factors[i], ctx->ckpt_paths[i],ctx->save_lines[i]);
            if (rc == 1) {
                ctx->array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if(!stage1_publish_factor(*ctx->opt,ctx->N,uint64_t(ctx->B1),sigma,ctx->factors[i])){ctx->aborted->store(1);break;}
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                if (ctx->opt->execution.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigma
                              << " -> factor found" << std::endl;
                } else {
                    ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                                   i, (unsigned long long)sigma);
                }
            } else if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  curve " << i << " sigma=" << sigma
                          << " -> Edwards stage-1 internal error" << std::endl;
                ctx->aborted->store(1,std::memory_order_relaxed);break;
            } else if (rc == 2) {
                ctx->aborted->store(1, std::memory_order_relaxed);
                break;
            }
            if (ctx->done_ctr) {
                const uint32_t done = ctx->done_ctr->fetch_add(1, std::memory_order_relaxed) + 1;
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                stage1_progress_set(done, ctx->opt->stage1_curves);
            }
        }
    }
}

// 默认线程数: min(curves, 硬件并发数).
static uint32_t edwards_default_threads(uint32_t curves) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    uint32_t t = (uint32_t)hw;
    if (t > curves) t = curves;
    return t;
}

// Run one Edwards (Atkin-Morain) stage-1 batch on the CPU. One curve per sigma.
//
#if defined(_MSC_VER)
// AVX512-F + DQ + IFMA 且 OS 已启用 zmm/opmask 状态 (XCR0) 才允许进 simd_edwards。
// 必须在基线 TU 里探测: simd_*.cpp 是 /arch:AVX512 编的, 在它内部执行任何代码都可能
// 让编译器在探测前就发出 AVX512 指令。
static int driver_simd_isa_ok(void) {
    int r[4] = {0,0,0,0};
    __cpuid(r, 0);
    if (r[0] < 7) return 0;
    __cpuidex(r, 1, 0);
    if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1)) return 0;   /* OSXSAVE, AVX */
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6ULL) != 0xE6ULL) return 0;                  /* opmask+ZMM_Hi256+Hi16_ZMM */
    __cpuidex(r, 7, 0);
    return (((r[1] >> 16) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 21) & 1)) ? 1 : 0;
}
#else
static int driver_simd_isa_ok(void) { return 0; }   /* 非 MSVC 由 CMake 侧保证 */
#endif

/* ---------------------------------------------------------------------------
 * 亲核性 (Affinity): "1,3,5,7" -> {1,3,5,7}
 * 空 / "none" / "auto" -> 空列表 = 不绑定, 交给系统调度。
 * 第 t 个 worker 绑到 list[t % count]，因此列表长度通常取 = 线程数或物理核数。
 * ------------------------------------------------------------------------- */


#if defined(_WIN32)
static void apply_thread_affinity(const std::vector<unsigned> &cpus, uint32_t t) {
    if (cpus.empty()) return;
    const unsigned cpu = cpus[t % cpus.size()];
    if (cpu >= 64) return;      /* 单处理器组只有 64 个逻辑 CPU 可用位图表达 */
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
}
#else
static void apply_thread_affinity(const std::vector<unsigned> &, uint32_t) {}
#endif

// curves == 1 or opt.execution.stage1_threads == 1 -> 顺序执行 (与历史行为一致);
// 否则按曲线并行 (每线程独立曲线, 动态取号), 吞吐随核数线性提升.
int run_edwards_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                              const std::string &savefilename, bool saveappend,
                              const std::string &n_expr,
                              ExecutionOptions &opt, ExecutionResult *out) {
    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->publish_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->firstsigma64 = 0;
    out->factors = nullptr;
    out->array_found = nullptr;
    out->sigmas = nullptr;

    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    // NAF 窗口 (字典大小 2^(w-2)); 0 = 保持 ecm_edwards_cpu 的默认值.
    if (opt.execution.naf_w >= 2) {
        edwards_set_naf_w(opt.execution.naf_w);
    } else if (opt.execution.naf_w != 0) {
        std::cerr << "naf_w must be 0 (default) or >= 2" << std::endl;
        return ECM_ERROR;
    }

    // s = 48 * lcm(1..B1). (GPU batch uses lcm(1..B1); Edwards adds the
    // lcm(12,16)=48 torsion factor, matching Prime95 ecm_calc_exp.)
    mpz_t s;
    mpz_init(s);
    if (!compute_batch_s(s, B1)) {
        std::cerr << "Failed to compute Edwards stage-1 exponent" << std::endl;
        mpz_clear(s);
        return ECM_ERROR;
    }
    mpz_mul_ui(s, s, 48);

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    std::vector<std::string> save_lines(curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    const bool use_ckpt = !opt.persistence.tmp_dir.empty();
    if (use_ckpt) {
        if (!ensure_dir(opt.persistence.tmp_dir)) {
            std::cerr << "ERROR: cannot create tmp_dir: " << opt.persistence.tmp_dir << std::endl;
            mpz_clear(s);
            for (uint32_t j = 0; j < curves; j++) mpz_clear(factors[j]);
            free(factors);
            free(array_found);
            return ECM_ERROR;
        }
    }

    // sigma 预生成: random_sigma_u64 使用全局 rand()/srand(), 非线程安全,
    // 因此全部在主线程生成后再分发给工作线程.
    //
    std::vector<uint64_t> sigmas(curves);
    for (uint32_t i = 0; i < curves; i++) {
        // Fixed sigma → batch start + i (matches -sigma/-gpucurves semantics);
        // otherwise a fresh Prime95-style random sigma per curve -- 但若该曲线已有合格的
        // .ckpt 就**采用存档里的 sigma**（存档覆盖参数）。否则每轮都随机出新的 sigma,
        // 与存档永久不匹配, 自动续跑永远不会发生（这正是"不会自动从 saves/ 续跑"的根因）。
        if (opt.identity.sigma_fixed) {
            sigmas[i] = ecm_stage1_batch_sigma(i,opt.identity.fixed_sigma64 + i);
        } else {
            sigmas[i] = ecm_stage1_batch_sigma(i,random_sigma_u64());
            if (!opt.persistence.tmp_dir.empty()) {
                const std::string cp = ecm_stage1_curve_record(i);
                ecm_save_common rcm;
                uint64_t rsp; uint32_t rebs, rbn, rds;
                mpz_t dx, dy, rx, ry, rz;
                mpz_inits(dx, dy, rx, ry, rz, NULL);
                if (ecm_edwards_read_stage1(cp, rcm, &rsp, &rebs, &rbn, &rds,
                                           dx, dy, rx, ry, rz) &&
                    rbn > 0 && rcm.sigma != 0 && rcm.B1 == (uint64_t)B1) {
                    sigmas[i] = (uint64_t)rcm.sigma;
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stdout,
                                   "  curve %u: found .ckpt -> adopting sigma=%llu (bitnum=%u)\n",
                                   i + 1, (unsigned long long)rcm.sigma, rbn);
                }
                mpz_clears(dx, dy, rx, ry, rz, NULL);
            }
        }
    }
    const uint64_t firstsigma = sigmas[0];

    // Curve codecs address records inside the single process-owned batch file.
    std::vector<std::string> ckpt_paths(curves);
    if (use_ckpt) {
        for (uint32_t i = 0; i < curves; i++) {
            ckpt_paths[i] = ecm_stage1_curve_record(i);
        }
    }

    uint32_t nthreads = opt.execution.stage1_threads;
    if (nthreads == 0) nthreads = edwards_default_threads(curves);
    if (nthreads > curves) nthreads = curves;
    if (nthreads < 1) nthreads = 1;

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " Edwards curves, CPU";
    if (nthreads > 1) std::cout << ", " << nthreads << " threads";
    std::cout << ")";
    if (use_ckpt) std::cout << " [saves -> " << opt.persistence.tmp_dir << "]";
    std::cout << std::endl;

    auto *bar = stage1_bar_create();
    opt.stage1_t0_ms = stage1_now_ms();
    opt.stage1_curves = curves;
    g_stage1_done.store(0);stage1_progress_set(0,curves);
    bool aborted = false;
    if (nthreads <= 1 && curves == 1) {
        for (uint32_t i = 0; i < curves; i++) {
            const int rc = edwards_run_one_curve(N, s, opt, B1, B2, i, sigmas[i], curves,
                                                 factors[i], ckpt_paths[i],save_lines[i]);
            if (rc == 1) {
                array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i])){aborted=true;break;}
                if (opt.execution.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigmas[i]
                              << " -> factor found" << std::endl;
                }
            } else if (rc < 0) {
                std::cerr << "  curve " << i << " sigma=" << sigmas[i]
                          << " -> Edwards stage-1 internal error" << std::endl;
                aborted=true;break;
            } else if (rc == 2) {
                aborted = true;
                break;
            }
        }
    } else {
        EdwardsWorkerCtx wctx;
        wctx.N = N;
        wctx.s = s;
        wctx.opt = &opt;
        /* ---- backend 选择 + 起始信息 (见 §13.6 第 7 条: simd 不允许静默降级) ---- */
        g_stage1_done.store(0, std::memory_order_relaxed);
        std::atomic<uint32_t> &done_ctr = g_stage1_done;
        {
            const int isa = driver_simd_isa_ok();
            if (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd && !isa) {
                ecm_ts_fprintf(stderr,
                               "ERROR: --impl simd requested but this CPU lacks "
                               "AVX512-F/DQ/IFMA. Refusing to fall back silently; use "
                               "--impl auto or scalar.\n");
                return -1;
            }
            wctx.simd_ok = (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd) ? 1
                         : (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::Auto) ? (isa && curves >= 8) : 0;
            wctx.done_ctr = &done_ctr;
            char sbuf[160];
            snprintf(sbuf, sizeof(sbuf),
                     "simd (AVX512-IFMA: 8 curves per batch, 1 thread/batch, dict w=%d)",
                     edwards_get_naf_w() >= 3 ? edwards_get_naf_w() : 8);
            ecm_ts_fprintf(stdout, "CPU implementation : %s\n",
                           wctx.simd_ok ? sbuf : "gmp (scalar mpn, 1 curve/thread)");
            ecm_ts_fprintf(stdout, "N / s           : %zu bit N, %zu bit s\n",
                           mpz_sizeinbase(N, 2), mpz_sizeinbase(s, 2));
            if (wctx.simd_ok) {
                /* 批量模式下真正的并行度是"批数"而不是线程数: 8 条曲线 = 1 批 = 1 个线程在干活,
                   其余线程空闲。这里如实显示, 免得用户以为 8 线程都忙。 */
                const uint32_t batches = (curves + 7u) / 8u;
                const uint32_t busy = batches < nthreads ? batches : nthreads;
                ecm_ts_fprintf(stdout,
                               "work split      : %u batch(es) of 8 curves -> %u thread(s) busy "
                               "(%u requested)%s\n",
                               batches, busy, nthreads,
                               busy < nthreads ? "  [raise --gpucurves for more parallelism]" : "");
            }
            if (!opt.execution.affinity_cpus.empty()) {
                std::string a;
                for (size_t i = 0; i < opt.execution.affinity_cpus.size(); i++) {
                    if (i) a += ",";
                    a += std::to_string(opt.execution.affinity_cpus[i]);
                }
                ecm_ts_fprintf(stdout, "affinity        : %s (worker t -> cpu[%s][t %% %zu])\n",
                               a.c_str(), a.c_str(), opt.execution.affinity_cpus.size());
            } else {
                ecm_ts_fprintf(stdout, "affinity        : none (OS scheduler)\n");
            }
            fflush(stdout);
        }
        /* 进度条: 照 CUDA stage-1 host 的样式, prefix 换成 stage1 */
        opt.stage1_t0_ms = stage1_now_ms();
        opt.stage1_curves = curves;
        stage1_progress_set(0, curves);
        wctx.B1 = B1;
        wctx.B2 = B2;
        wctx.curves = curves;
        wctx.sigmas = sigmas.data();
        wctx.ckpt_paths = ckpt_paths.data();
        wctx.factors = factors;
        wctx.array_found = array_found;
        wctx.save_lines = save_lines.data();
        std::atomic<uint32_t> next(0);
        std::atomic<int> aborted_flag(0);
        wctx.next = &next;
        wctx.aborted = &aborted_flag;

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.emplace_back([&wctx, t]() {
                apply_thread_affinity(wctx.opt->execution.affinity_cpus, t);
                edwards_worker(&wctx);
            });
        for (auto &th : pool) th.join();
        aborted = (aborted_flag.load(std::memory_order_relaxed) != 0);
    }

    if (!aborted) stage1_progress_set(curves,curves);
    stage1_bar_destroy(bar);
    if (aborted) {
        ecm_ts_fprintf(stdout, "stage-1 incomplete; checkpoints retained\n");
        mpz_clear(s);
        for (uint32_t j = 0; j < curves; j++) mpz_clear(factors[j]);
        free(factors);
        free(array_found);
        out->ret = ECM_ERROR;
        return ECM_ERROR;
    }

    const auto stem=mont_format_save_name(opt.persistence.save_name_pattern,N,B1,&opt);
    const auto save=savefilename.empty()?(opt.persistence.tmp_dir.empty()?std::string{}:opt.persistence.tmp_dir+"/")+stem:savefilename;
    std::string lines;for(const auto &line:save_lines)lines+=line;
    if(!ecm_edwards::publish_save(save,lines,saveappend)) {
        out->publish_failed=true;out->ret=ECM_ERROR;
        ecm_ts_fprintf(stderr,"ERROR: cannot publish Edwards save %s\n",save.c_str());
        mpz_clear(s);for(uint32_t i=0;i<curves;++i)mpz_clear(factors[i]);free(factors);free(array_found);return ECM_ERROR;
    }
    for(const auto &path:ckpt_paths)if(!path.empty())ecm_stage1_remove_checkpoints(path);
    ecm_ts_fprintf(stdout,"Edwards Stage1 save: %s (%u attempted curves)\n",save.c_str(),curves);
    (void)n_expr;

    mpz_clear(s);

    out->ret = ECM_NO_FACTOR_FOUND;
    out->firstsigma = (uint32_t)(firstsigma & 0xFFFFFFFFu);
    out->firstsigma64 = firstsigma;
    out->factors = factors;
    out->array_found = array_found;
    /* Edwards draws a random sigma per curve, so publish the exact per-curve values
       instead of letting consumers assume a sequence. */
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = sigmas[i];
    }
    return out->ret;
}

// Run one stage-1 batch. N and the caller's n_expr are borrowed (not cleared);
// factors / array_found are allocated here and returned via `out` (caller frees).
/* ---------------------------------------------------------------------------
 * Shared automatic Stage1 save names (ecm.toml: save_name_pattern).
 *
 *   {curve}: e / m0 / m2 / m3; {target}: m<original exponent> / bit<actual bits>;
 *   {n}  : original assignment exponent when available, else actual N bits;
 *   {b1} : compact bound -- 100000 -> "1e5", 110000000 -> "110e6", 12345 -> "12345".
 *
 * ecm_extract_b1_from_save_name() documents the same pattern on the reading side.
 * ------------------------------------------------------------------------- */


/* Standalone codec fallback stem; the driver routes records into its batch. */
static std::string mont_save_stem(const std::string &pattern, const mpz_t N, double B1) {
    std::string f = mont_format_save_name(pattern, N, B1);
    const size_t dot = f.rfind('.');
    if (dot != std::string::npos) f.erase(dot);
    return f;
}

// ---------------------------------------------------------------------------
// Suyama-sigma Montgomery stage 1 (Prime95 sigma_type = 1 / gmp-ecm -param 0).
//
// Same result contract as run_edwards_stage1: fills out->factors / out->array_found
// and returns an ECM_* code, so the CLI caller needs no changes.
//
//   * s = torsion * lcm(1..B1)   (torsion 1 matches gmp-ecm, 12 matches Prime95;
//     see docs/architecture/STAGE1.md 4.1)
//   * 8 curves per AVX512-IFMA batch when available, else the scalar MPN path
//   * ONE shared save file per task (same N and B1, one self-contained line per
//     curve -- the reference reader parses METHOD/B1/N from every line, so the
//     lines must stay self-contained; docs 9.2)
// ---------------------------------------------------------------------------
/* Default worker count for the Montgomery stage-1 path.
 *
 * The unit of parallel work differs per backend: the SIMD path runs 8 curves per
 * batch, so its parallelism is capped by the number of batches (curves/8) and the
 * scalar path by the number of curves.  Extra threads beyond that would only
 * contend, so clamp -- 8 curves under the IFMA SIMD implementation is one thread of real
 * work no matter how many cores the machine has. */
static uint32_t mont_default_threads(uint32_t curves, bool use_simd) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    uint32_t tasks = use_simd ? ((curves + IFMA_LANES - 1) / IFMA_LANES) : curves;
    if (tasks == 0) tasks = 1;
    uint32_t t = (uint32_t)hw;
    if (t > tasks) t = tasks;
    return t;
}

/* ---------------------------------------------------------------------------
 * Montgomery stage-1 progress + checkpoint glue
 * ------------------------------------------------------------------------- */

/* Checkpoint / progress granularity: aim at ~512 callbacks per curve (a progress
   update every ~0.2% of the ladder) with a 4096-bit floor, rounded up to a power
   of two so the offsets in the checkpoint files stay readable.  It is derived
   from the exponent only, so a resumed run with the same command line computes
   the identical offsets and can match what the interrupted run wrote. */
static size_t mont_ckpt_chunk_bits(size_t nbits)
{
    size_t p = 4096;
    const size_t want = nbits / 512;
    while (p < want && p < ((size_t)1 << 30)) p <<= 1;
    return p;
}

/* Progress + pause decision for one SIMD batch.  Called from inside the ladder
   every `chunk` bits, with the batch holding the whole worker thread. */
struct MontProgCtx {
    std::atomic<uint64_t> *lane_bits;    /* [curves]: exponent bits done per curve */
    const uint32_t        *lane_curve;   /* [lanes] : curve index of each lane */
    uint32_t               lanes;
    uint32_t               total_curves;
    size_t                 nbits;
    std::atomic<long long> *last_ckpt_ms; /* shared by all workers: one timer */
    long long              interval_ms;   /* < 0 = no periodic autosave */
};

static int mont_progress_cb(void *p, size_t bitnum)
{
    MontProgCtx *pc = (MontProgCtx *)p;
    for (uint32_t k = 0; k < pc->lanes; k++)
        pc->lane_bits[pc->lane_curve[k]].store((uint64_t)bitnum, std::memory_order_relaxed);

    /* Work unit = one curve; the fractional part of a lane is bitnum/nbits.  A
       running sum over the per-curve counters is what keeps the number monotone
       when several workers report at different offsets, and it needs no lock. */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < pc->total_curves; i++)
        sum += pc->lane_bits[i].load(std::memory_order_relaxed);
    const double done_f = pc->nbits ? (double)sum / (double)pc->nbits : 0.0;
    {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done_f, pc->total_curves);
    }

    if (ecm_stage1_stop_requested()) return 1;                       /* Ctrl+C: stop at this chunk */
    if (pc->interval_ms > 0) {
        const long long now = stage1_now_ms();
        if (now - pc->last_ckpt_ms->load(std::memory_order_relaxed) >= pc->interval_ms) {
            pc->last_ckpt_ms->store(now, std::memory_order_relaxed);
            return 1;                                  /* autosave interval elapsed */
        }
    }
    return 0;
}

/* Scalar ladder wrapper: same policy, the offset arrives inside the state. */
static int mont_progress_cb_scalar(void *p, const mont_ladder_state_t *st)
{
    return mont_progress_cb(p, st->bitnum);
}

int run_mont_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           ExecutionOptions &opt, ExecutionResult *out) {
    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->firstsigma64 = 0;
    out->factors = nullptr;
    out->array_found = nullptr;
    out->sigmas = nullptr;
    (void)saveappend;
    (void)B2;

    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    mpz_t s;
    mpz_init(s);
    const int torsion = opt.execution.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12 : 1;
    std::string s_status;
    const size_t s_bits = mont_build_s(s, (uint64_t)B1, (uint64_t)torsion, &s_status);
    if (s_bits == 0) {
        /* only reachable for B1 > 5e9 or an allocation failure; say so instead of
           running a ladder over the torsion-only exponent */
        ecm_ts_fprintf(stderr,
                       "ERROR: cannot build the stage-1 exponent for B1=%.0f "
                       "(bound must be <= 5e9 and the prime sieve must fit in memory)\n",
                       B1);
        mpz_clear(s);
        return ECM_ERROR;
    }

    const bool isa = driver_simd_isa_ok();
    if (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd && !isa) {
        ecm_ts_fprintf(stderr,
                       "ERROR: --impl simd requested but this CPU lacks "
                       "AVX512-F/DQ/IFMA. Use auto or scalar.\n");
        mpz_clear(s);
        return ECM_ERROR;
    }
    const bool use_simd = (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd) ? true
                        : (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::GmpScalar) ? false
                        : (isa && curves >= 2);
    ecm_ts_fprintf(stdout, "curve model     : montgomery (Suyama sigma, %s, torsion=%d)\n",
                   use_simd ? "AVX512-IFMA 8-lane batch" : "scalar mpn", torsion);
    ecm_ts_fprintf(stdout, "stage1 exponent : s_bits=%zu (lcm(1..%.0f) x %d) [%s]\n",
                   s_bits, B1, torsion, s_status.c_str());

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    /* The exponent bit array depends only on (B1, torsion) -- not on N, not on sigma
       -- so it is built once here and read by every task below.  It is built before
       the sigmas because the checkpoint pre-pass needs s_bits to validate a resume. */
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    const size_t ckpt_chunk = mont_ckpt_chunk_bits(nbits);

    /* ---- mid-stage-1 checkpoints (docs/architecture/STAGE1.md) ----------
       One independently resumable record per curve in a shared batch file. The
       interval comes from ckpt_seconds / --ckpt; 0 = no autosave (Ctrl+C still
       checkpoints, which is what turns an interrupted run into a resumed one). */
    const bool use_ckpt = !opt.persistence.tmp_dir.empty();
    const std::string ckpt_stem = use_ckpt
        ? (opt.persistence.tmp_dir + "/" + mont_save_stem(opt.persistence.save_name_pattern, N, B1))
        : std::string();
    const long long ckpt_interval_ms =
        (opt.persistence.ckpt_ms > 0 && opt.persistence.ckpt_ms != ULONG_MAX) ? (long long)opt.persistence.ckpt_ms : -1;

    /* per-curve results: the normalised x for misses, the factor for hits */
    std::vector<mpz_t> xs(curves);
    int *hit = (int *)calloc(curves, sizeof(int));
    for (uint32_t i = 0; i < curves; i++) mpz_init(xs[i]);

    /* curve state recovered from checkpoints */
    std::vector<size_t> bit_off(curves, 0);      /* exponent bits already consumed */
    std::vector<char>   ckpt_done(curves, 0);    /* result restored: curve is finished */
    uint32_t n_done_ck = 0, n_inflight_ck = 0, n_seeded = 0;

    /* Sigmas: fixed base (from -sigma, then sigma+i) or random.
     *
     * Interoperability constraint (why this is not a full 64-bit random): Prime95's
     * ECMSTAGE2 path -- the one that runs stage 2 on a gmp-ecm/PARAM=0 save file --
     * parses SIGMA with  mpz_get_str() -> atoll()  (ecm.cpp:7375).  A sigma >= 2^63
     * overflows atoll, so Prime95 would rebuild a DIFFERENT curve from a mangled
     * sigma and quietly waste the stage-2 run (the bitmap/x-coordinate it loads came
     * from our curve).  Prime95 itself only ever generates sigmas below 2^53
     * (ecm.cpp:7416: (rand()&0x1F)<<48 + (rand()&0xFFFF)<<32 + rdtsc bits), so we use
     * the same generator the Edwards path already uses.  gmp-ecm's own reader is
     * mpz-based and has no such limit, but a sigma that works with every consumer is
     * strictly better -- and 2^53 still leaves ~9e15 curves to choose from. */
    std::vector<uint64_t> sigmas(curves);
    std::atomic<int> checkpoint_failed(0);
    for (uint32_t i = 0; i < curves; i++) {
        const uint64_t fixed_sigma = opt.identity.fixed_sigma64 + i;
        bool adopted = false;
        if (use_ckpt) {
            /* A checkpoint owns its curve's sigma.  Adopting it is what makes the
               identical command line continue the identical curve set; without this
               a resumed run would draw fresh random sigmas and the checkpoints
               would never match.  (The Edwards path decides the same way.) */
            mont_ckpt_t ck;
            mont_ckpt_init(&ck);
            std::string why;
            if (mont_ckpt_read(mont_ckpt_path(ckpt_stem, i), N, B1, torsion, nbits, &ck, &why) &&
                ck.curve == i &&
                (!opt.identity.sigma_fixed || ck.sigma == fixed_sigma)) {
                sigmas[i] = ck.sigma;
                adopted = true;
                if (ck.status == MONT_CKPT_DONE) {
                    ckpt_done[i] = 1;
                    n_done_ck++;
                    if (ck.hit) {
                        mpz_set(factors[i], ck.factor);
                        array_found[i] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[i], ck.factor);
                        hit[i] = 1;
                    } else {
                        mpz_set(xs[i], ck.xout);
                    }
                } else if (ck.bitnum > 0) {
                    bit_off[i] = ck.bitnum;
                    n_inflight_ck++;
                } else {
                    n_seeded++;
                }
            }
            mont_ckpt_clear(&ck);
        }
        if (!adopted) {
            sigmas[i] = ecm_stage1_batch_sigma(i,opt.identity.sigma_fixed ? fixed_sigma : random_sigma_u64());
            if (use_ckpt) {
                /* Seed record: pins the sigma before any ladder work happens, so a
                   run that is killed while this curve is still queued resumes the
                   SAME curve later instead of substituting a new random one. */
                mont_ckpt_t ck;
                mont_ckpt_init(&ck);
                ck.status = MONT_CKPT_INFLIGHT;
                ck.bitnum = 0;
                mont_ckpt_ident(&ck, i, sigmas[i], B1, torsion, nbits,
                                use_simd ? "ifma" : "mpn");
                if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, i), N, &ck) != 0)
                    checkpoint_failed.store(1);
                mont_ckpt_clear(&ck);
                n_seeded++;
            }
        }
    }
    if (opt.identity.sigma_fixed && fixed_sigma_batch_exceeds_prime95_range(opt.identity.fixed_sigma64, curves)) {
        ecm_ts_fprintf(stderr,
            "WARNING: sigma >= 2^63; Prime95's ECMSTAGE2 reader (atoll) cannot read it.\n"
            "         Use gmp-ecm for stage 2 with these sigmas, or pass a smaller -sigma.\n");
    }
    out->firstsigma64 = sigmas[0];
    out->firstsigma = (uint32_t)(sigmas[0] & 0xFFFFFFFFu);
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = sigmas[i];
    }

    /* Work split.  A "task" is one SIMD batch (<= 8 curves, one thread) or, on the
       scalar backend, one curve.  Tasks are handed out by an atomic counter, so a
       worker that finishes a batch picks up the next one -- with uniform batch cost
       that is equivalent to a static split, but it degrades gracefully.

       Curves are grouped by "bits already consumed" (descending) before chunking,
       because the lanes of a batch share ONE exponent prefix: every lane has to
       start at the same offset.  On a fresh run every offset is 0, so this is
       exactly the old sequential split; after a resume it costs at most one partly
       empty batch per batch that the interrupted run was in the middle of. */
    std::vector<uint32_t> pending;
    for (uint32_t i = 0; i < curves; i++) if (!ckpt_done[i]) pending.push_back(i);
    std::stable_sort(pending.begin(), pending.end(),
                     [&](uint32_t a, uint32_t b) { return bit_off[a] > bit_off[b]; });

    struct MontTask { uint32_t first, count; size_t start_bit; };
    std::vector<MontTask> task_list;
    {
        const size_t per_task = use_simd ? (size_t)IFMA_LANES : (size_t)1;
        for (size_t p = 0; p < pending.size(); ) {
            size_t q = p;
            while (q < pending.size() && (q - p) < per_task &&
                   bit_off[pending[q]] == bit_off[pending[p]]) q++;
            MontTask tk = { (uint32_t)p, (uint32_t)(q - p), bit_off[pending[p]] };
            task_list.push_back(tk);
            p = q;
        }
    }
    const uint32_t tasks = (uint32_t)task_list.size();
    uint32_t nthreads = opt.execution.stage1_threads ? opt.execution.stage1_threads
                                         : mont_default_threads(curves, use_simd);
    if (nthreads > tasks) nthreads = tasks;
    if (nthreads < 1) nthreads = 1;
    ecm_ts_fprintf(stdout, "stage1 threads  : %u worker(s) x %u task(s) of %s\n",
                   nthreads, tasks, use_simd ? "8 curves" : "1 curve");
    if (use_ckpt) {
        char iv[48];
        if (ckpt_interval_ms > 0) snprintf(iv, sizeof(iv), "%.3g s", ckpt_interval_ms / 1000.0);
        else                      snprintf(iv, sizeof(iv), "off (Ctrl+C still saves)");
        ecm_ts_fprintf(stdout,
                       "checkpoint      : %s_c*.ckpt  [%u done, %u mid-ladder, %zu to run]"
                       "  autosave %s\n",
                       ckpt_stem.c_str(), n_done_ck, n_inflight_ck, pending.size(), iv);
    }
    if (!opt.execution.affinity_cpus.empty()) {
        std::string a;
        for (size_t i = 0; i < opt.execution.affinity_cpus.size(); i++) {
            if (i) a += ",";
            a += std::to_string(opt.execution.affinity_cpus[i]);
        }
        ecm_ts_fprintf(stdout, "affinity        : %s (worker t -> cpu[%s][t %% %zu])\n",
                       a.c_str(), a.c_str(), opt.execution.affinity_cpus.size());
    }
    fflush(stdout);

    std::atomic<uint32_t> next_task(0);
    std::atomic<uint32_t> found_atomic(0);
    std::atomic<int> init_failed(0);
    std::atomic<int> paused_flag(0);
    std::atomic<long long> last_ckpt_ms(stage1_now_ms());
    /* per-curve progress in exponent bits: restored curves start at their own
       offset (done ones at nbits), so the bar is right from the first sample */
    std::vector<std::atomic<uint64_t>> lane_bits(curves);
    for (uint32_t i = 0; i < curves; i++)
        lane_bits[i].store(ckpt_done[i] ? (uint64_t)nbits : (uint64_t)bit_off[i],
                           std::memory_order_relaxed);

    auto worker = [&]() {
        if (use_simd) {
            mont_soa_ctx_t ctx;                     /* per-thread scratch pool */
            if (mont_soa_init(&ctx, N, to_ifma_field_mode(opt.execution.selection.simd_reduction)) != 0) {
                init_failed.store(1);
                return;
            }
            /* Report the field layer the context actually resolved to: with
               field = auto the choice depends on N (Mersenne fold for 2^k-1), so
               printing the request alone would be misleading. */
            {
                static std::atomic<int> shown(0);
                if (shown.fetch_add(1) == 0)
                    ecm_ts_fprintf(stdout, "field layer     : %s\n", ifma_field_name(&ctx.mc));
            }
            const size_t lw = 8 * ctx.n;
            std::vector<uint64_t> state(mont_soa_state_words(&ctx));
            std::vector<mpz_t> bx(IFMA_LANES), bg(IFMA_LANES);
            for (unsigned k = 0; k < IFMA_LANES; k++) { mpz_inits(bx[k], bg[k], NULL); }
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks) break;
                const MontTask &tk = task_list[t];
                uint32_t cur[IFMA_LANES];
                uint64_t sg[IFMA_LANES];
                /* a short batch repeats its last sigma: unread lanes only waste
                   SIMD slots, the used lanes stay exactly as requested */
                for (uint32_t k = 0; k < IFMA_LANES; k++) {
                    const uint32_t c = pending[tk.first + (k < tk.count ? k : tk.count - 1)];
                    cur[k] = c;
                    sg[k] = sigmas[c];
                }

                /* Resume: every lane of the task shares the offset, so each lane's
                   p0/p1 pair comes from its own checkpoint file.  If any lane's
                   state cannot be read (deleted, truncated, hand-edited) the whole
                   task restarts from bit 0 -- the lanes cannot be at different
                   offsets, the ladder walks one shared bit sequence. */
                size_t start = tk.start_bit;
                if (start > 0) {
                    for (uint32_t k = 0; k < tk.count; k++) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        std::string why;
                        const int ok = mont_ckpt_read(mont_ckpt_path(ckpt_stem, cur[k]), N, B1,
                                                      torsion, nbits, &ck, &why);
                        if (ok && ck.bitnum == start) {
                            ifma_from_mpz_lane(state.data() + 0 * lw, k, ck.X0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 1 * lw, k, ck.Z0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 2 * lw, k, ck.X1, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 3 * lw, k, ck.Z1, &ctx.mc);
                        } else {
                            start = 0;
                        }
                        mont_ckpt_clear(&ck);
                        if (!start) break;
                    }
                }

                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = cur;
                pc.lanes = tk.count;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* The ladder stops at a checkpoint boundary for two different
                   reasons -- the autosave interval elapsed, or SIGINT arrived --
                   and they must NOT be confused: an autosave pause persists the
                   state and carries on with the same batch, while SIGINT ends the
                   run after persisting.  (Getting this wrong turns a 1 s autosave
                   interval into "the run stops after 1 s".) */
                int rc = MONT_SOA_DONE;
                for (;;) {
                    size_t bitnum = start;
                    rc = mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, start,
                                                 state.data(), &bitnum,
                                                 bx.data(), bg.data(),
                                                 mont_progress_cb, &pc, ckpt_chunk);
                    if (rc != MONT_SOA_PAUSED) break;
                    if (use_ckpt) {
                        for (uint32_t k = 0; k < tk.count; k++) {
                            const uint32_t c = cur[k];
                            mont_ckpt_t ck;
                            mont_ckpt_init(&ck);
                            ck.status = MONT_CKPT_INFLIGHT;
                            ck.bitnum = bitnum;
                            mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                            ifma_to_mpz_lane(ck.X0, state.data() + 0 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z0, state.data() + 1 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.X1, state.data() + 2 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z1, state.data() + 3 * lw, k, &ctx.mc);
                            if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                                checkpoint_failed.store(1);
                            mont_ckpt_clear(&ck);
                        }
                    }
                    if (ecm_stage1_stop_requested()) break;      /* SIGINT: leave after saving */
                    start = bitnum;                /* autosave: continue from here */
                }
                if (rc == MONT_SOA_ERROR) { init_failed.store(1); break; }
                if (rc == MONT_SOA_PAUSED) { paused_flag.store(1); break; }

                uint32_t local = 0;
                for (uint32_t k = 0; k < tk.count; k++) {
                    const uint32_t c = cur[k];
                    lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                    if (mpz_cmp_ui(bg[k], 1) > 0 && mpz_cmp(bg[k], N) < 0) {
                        mpz_set(factors[c], bg[k]);
                        array_found[c] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[c], bg[k]);
                        hit[c] = 1;
                        local++;
                        if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[c],factors[c]))checkpoint_failed.store(1);
                    } else {
                        mpz_set(xs[c], bx[k]);
                    }
                    if (use_ckpt) {
                        /* DONE record: if the process dies before the shared .save
                           is written, this is what keeps the curve from being
                           recomputed by the resumed run. */
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_DONE;
                        ck.hit = hit[c];
                        ck.bitnum = nbits;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                        if (ck.hit) mpz_set(ck.factor, factors[c]);
                        else        mpz_set(ck.xout, xs[c]);
                        if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                }
                if (local) found_atomic.fetch_add(local);
            }
            for (unsigned k = 0; k < IFMA_LANES; k++) { mpz_clears(bx[k], bg[k], NULL); }
            mont_soa_clear(&ctx);
        } else {
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks) break;
                const MontTask &tk = task_list[t];
                const uint32_t c = pending[tk.first];
                mpz_t g, x, Qx, Qz, inv;
                mpz_inits(g, x, Qx, Qz, inv, NULL);

                mont_ladder_state_t st;
                mont_ladder_state_init(&st);
                size_t start = tk.start_bit;
                if (start > 0) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    std::string why;
                    if (mont_ckpt_read(mont_ckpt_path(ckpt_stem, c), N, B1, torsion, nbits,
                                       &ck, &why) && ck.bitnum == start) {
                        st.bitnum = ck.bitnum;
                        mpz_set(st.X0, ck.X0);
                        mpz_set(st.Z0, ck.Z0);
                        mpz_set(st.X1, ck.X1);
                        mpz_set(st.Z1, ck.Z1);
                    } else {
                        start = 0;
                    }
                    mont_ckpt_clear(&ck);
                }

                uint32_t lane[1] = { c };
                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = lane;
                pc.lanes = 1;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* autosave pause vs SIGINT: see the SIMD branch above */
                int rc = MONT_LADDER_MISS;
                for (;;) {
                    rc = mont_stage1_curve_bits_ex(g, Qx, Qz, N, sigmas[c], bits, nbits,
                                                   start, &st, mont_progress_cb_scalar,
                                                   &pc, ckpt_chunk);
                    if (rc != MONT_LADDER_PAUSED) break;
                    if (use_ckpt) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_INFLIGHT;
                        ck.bitnum = st.bitnum;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                        mpz_set(ck.X0, st.X0);
                        mpz_set(ck.Z0, st.Z0);
                        mpz_set(ck.X1, st.X1);
                        mpz_set(ck.Z1, st.Z1);
                        if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                    if (ecm_stage1_stop_requested()) break;      /* SIGINT: leave after saving */
                    start = st.bitnum;             /* autosave: continue from here */
                }
                if (rc == MONT_LADDER_ERROR) {
                    init_failed.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }
                if (rc == MONT_LADDER_PAUSED) {
                    paused_flag.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }

                lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                /* same normalisation as mont_stage1_curve_bits_x() */
                if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
                    mpz_mul(x, Qx, inv);
                    mpz_mod(x, x, N);
                } else {
                    mpz_set(x, Qx);
                }
                if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
                    mpz_set(factors[c], g);
                    array_found[c] = ECM_FACTOR_FOUND_STEP1;
                    mpz_set(xs[c], g);
                    hit[c] = 1;
                    found_atomic.fetch_add(1);
                    if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[c],factors[c]))checkpoint_failed.store(1);
                } else {
                    mpz_set(xs[c], x);
                }
                if (use_ckpt) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    ck.status = MONT_CKPT_DONE;
                    ck.hit = hit[c];
                    ck.bitnum = nbits;
                    mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                    if (ck.hit) mpz_set(ck.factor, factors[c]);
                    else        mpz_set(ck.xout, xs[c]);
                    if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                        checkpoint_failed.store(1);
                    mont_ckpt_clear(&ck);
                }
                mpz_clears(g, x, Qx, Qz, inv, NULL);
                mont_ladder_state_clear(&st);
            }
        }
    };

    indicators::ProgressBar *bar = stage1_bar_create();
    opt.stage1_t0_ms = stage1_now_ms();
    opt.stage1_curves = curves;
    stage1_progress_set((double)(curves - pending.size()), curves);

    const auto t0 = std::chrono::steady_clock::now();
    if (nthreads <= 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.emplace_back([&worker, &opt, t]() {
                apply_thread_affinity(opt.execution.affinity_cpus, t);
                worker();
            });
        for (auto &th : pool) th.join();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stage1_bar_destroy(bar);
    free(bits);

    if (init_failed.load() || checkpoint_failed.load()) {
        ecm_ts_fprintf(stderr, "ERROR: mont stage-1 %s\n",
                       checkpoint_failed.load() ? "checkpoint write failed" : "SIMD context init failed");
        mpz_clear(s);
        for (uint32_t i = 0; i < curves; i++) { mpz_clear(factors[i]); mpz_clear(xs[i]); }
        free(factors); free(array_found); free(hit);
        return ECM_ERROR;
    }

    /* Interrupted: every worker wrote its lane's mid-ladder state before leaving,
       and the curves it never reached keep their seed/older record, so rerunning
       the identical command line continues instead of restarting. */
    if (paused_flag.load()) {
        ecm_ts_fprintf(stdout,
                       "stage-1 paused (checkpoint saved under %s); rerun the same command "
                       "line to resume\n",
                       opt.persistence.tmp_dir.c_str());
        mpz_clear(s);
        for (uint32_t i = 0; i < curves; i++) { mpz_clear(factors[i]); mpz_clear(xs[i]); }
        free(factors); free(array_found); free(hit);
        out->ret = ECM_ERROR;
        out->factors = nullptr;
        out->array_found = nullptr;
        return ECM_ERROR;
    }

    stage1_progress_set((double)curves, curves);

    /* Hit lines are printed here, in curve order, rather than from inside the worker
       loop: with N workers the old placement interleaved lines at random. */
    const uint32_t found = found_atomic.load();
    for (uint32_t i = 0; i < curves; i++) {
        if (hit[i])
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                           i, (unsigned long long)sigmas[i]);
    }

    /* ONE shared save file for the whole task (same N, same B1), named with the
       CUDA/GPU convention: m{n}_{b1}.save */
    bool save_failed = false;
    if (use_ckpt || !savefilename.empty()) {
        const std::string fname = mont_format_save_name(opt.persistence.save_name_pattern, N, B1,&opt);
        const std::string path = savefilename.empty()?opt.persistence.tmp_dir+"/"+fname:
            opencl_ecm_resolve_data_path_buf(savefilename.c_str());
        if(!savefilename.empty() && !opencl_ecm_check_save_file_writable(path,saveappend))save_failed=true;
        if(!save_failed)save_failed = !ecm_append_save_lines_mont(
            path, N, B1, sigmas.data(), curves, xs.data(), hit,
            n_expr.empty() ? std::to_string(mpz_sizeinbase(N, 2)) : n_expr);
        if (save_failed) {
            ecm_ts_fprintf(stderr, "ERROR: could not publish Montgomery save %s\n", path.c_str());
        } else {
            ecm_ts_fprintf(stdout, "  save            : %s (%u curve lines, shared)\n",
                           path.c_str(), curves);
        }
        /* The .save above is now the durable artifact (and the interoperable one),
           so the resume scratch files are no longer needed: dropping them makes a
           repeated run a fresh run instead of an instant replay of this one. */
        if (!save_failed) {
            for (uint32_t i = 0; i < curves; i++)
                mont_ckpt_remove(mont_ckpt_path(ckpt_stem, i));
        }
    }

    if (n_done_ck || n_inflight_ck) {
        ecm_ts_fprintf(stdout, "  resumed from checkpoint: %u curve(s) already done, %u mid-ladder\n",
                       n_done_ck, n_inflight_ck);
    }
    ecm_ts_fprintf(stdout, "  curves=%u  hits=%u  wall=%.2fs  (%.3f s/curve)\n",
                   curves, found, elapsed, curves ? elapsed / curves : 0.0);

    for (uint32_t i = 0; i < curves; i++) mpz_clear(xs[i]);
    free(hit);
    mpz_clear(s);

    out->ret = save_failed ? ECM_ERROR :
               (found ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND);
    out->publish_failed = save_failed;
    out->factors = factors;
    out->array_found = array_found;
    return out->ret;
}


} // namespace ecm_cpu_stage1
