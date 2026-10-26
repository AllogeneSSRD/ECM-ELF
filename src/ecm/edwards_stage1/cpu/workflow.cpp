#include "../../../io/files.h"
#include "ecm/stage1/progress.h"
#include "ecm/stage1/cpu_support.h"
#include <future>
#include <memory>
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
#include <direct.h> /* _mkdir for the local save directory */
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h> /* __rdtsc() for the Prime95-style sigma entropy mix */
#endif

#include <gmp.h>

#include "ecm/stage1/cpu_execution.h"
#include "ecm/curve/edwards.h"
#include "ecm/stage1/preparation.h"
#include "ecm/io/save.h"
#include "ecm/montgomery_stage1/opencl/checkpoint.h"
#include "../../../io/stage1_checkpoint.h"
#include "ecm/montgomery_stage1/cpu/checkpoint.h"
#include "ecm/edwards_stage1/cpu/ecm_edwards_cpu.h"
#include "ecm/montgomery_stage1/cpu/ecm_mont_cpu.h"
#include "ecm/montgomery_stage1/cpu/simd_mont_curve.h"
#include "ecm/cpu/arithmetic/ifma/simd_mont_ifma.h"
#include "../../../io/prime95_save.h"
#include "ecm/io/save.h"
#include "ecm/platform/logging.h"
#include <iomanip>
#include <regex>
#include "ecm/edwards_stage1/cpu/simd_edwards.h" /* SoA 批量 Edwards 层 (AVX512-IFMA TU, 见 §13.8) */
#include <intrin.h>
#include <windows.h>

namespace ecm_cpu_stage1 {
using ecm_stage1::ComputationResult;
using ecm_stage1::compute_batch_s;
using ecm_stage1::CurveKind;
using ecm_stage1::CurveOutcome;
using ecm_io::ensure_dir;
using ecm_stage1::ExecutionOptions;
using ecm_stage1::fixed_sigma_batch_exceeds_prime95_range;
using ecm_save::format_name;
using ecm_stage1::random_sigma_u64;
using ecm_stage1::stage1_now_ms;
using ecm_stage1::Stage1Options;
using ecm_stage1::stdout_is_tty_local;
using ecm_stage1::validate_stage1_factor;

// Classify the computed value directly. Text exists only at checkpoint/final-save encoding.
static bool classify_edwards_outcome(CurveOutcome& out, mpz_srcptr n, uint64_t sigma, mpz_srcptr qx,
                                     mpz_srcptr qz, mpz_srcptr factor, bool hit) {
    ecm_edwards::Integer g, inv;
    out.sigma = sigma;
    if (hit) {
        if (!validate_stage1_factor(n, factor))
            return false;
        out.kind = CurveKind::Factor;
        mpz_set(out.value, factor);
    } else {
        mpz_gcd(g.v, qz, n);
        if (mpz_cmp(g.v, n) == 0) {
            ecm_ts_fprintf(stdout, "Edwards identity: sigma=%llu gcd(Qz,N)=N, no Stage2 point\n",
                           static_cast<unsigned long long>(sigma));
            out.kind = CurveKind::CompletedWithoutPoint;
            mpz_set_ui(out.value, 0);
            return true;
        }
        if (mpz_cmp_ui(g.v, 1) != 0 || !mpz_invert(inv.v, qz, n))
            return false;
        mpz_mul(out.value, qx, inv.v);
        mpz_mod(out.value, out.value, n);
        out.kind = CurveKind::Stage2Point;
    }
    return true;
}

// Local Stage1 text-save publication and batch checkpoint records.

// 创建目录 (已存在视为成功).

// 自我 checkpoint 上下文 + 进度回调 (写 STAGE1 存档)
struct EdwardsCheckpointCtx {
    ecm_stage1_checkpoint::Context* checkpoint;
    const ExecutionOptions* opt;
    uint64_t sigma;
    uint64_t B1, B2;
    std::string save_path;
    long long interval_ms;
    long long last_ms;
    mpz_srcptr N;
    mpz_srcptr s;
    bool write_failed = false;
};

static int edwards_checkpoint_progress(void* ctx, const edwards_checkpoint_t* cur) {
    EdwardsCheckpointCtx* c = (EdwardsCheckpointCtx*)ctx;
    auto& checkpoint = *c->checkpoint;
    const long long now = stage1_now_ms();
    if (c->save_path.empty())
        return ecm_stage1_stop_requested() ? 0 : 1;
    const bool due = c->interval_ms > 0 && (now - c->last_ms >= c->interval_ms);
    if (!due && !ecm_stage1_stop_requested())
        return 1; // 继续

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
    const bool written =
        ecm_edwards_write_stage1(checkpoint, c->save_path, cm, 2, expbuf, cur->bitnum, dict_size,
                                 Px, Py, cur->Rx, cur->Ry, cur->Rz);
    c->last_ms = now;
    mpz_clears(d, Px, Py, NULL);
    if (!written) {
        c->write_failed = true;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 checkpoint %s\n", c->save_path.c_str());
        return 0;
    }

    if (ecm_stage1_stop_requested()) {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stdout, "checkpoint written (stop): %s @ bitnum=%u\n", c->save_path.c_str(),
                       cur->bitnum);
        return 0; // 中止
    }
    return 1; // 继续
}

// ---------------------------------------------------------------------------
// 单条曲线的 stage-1 执行 (顺序/并行共用)
// ---------------------------------------------------------------------------
//
// 线程安全约定: N 与 s 只读共享 (GMP 允许同一 mpz_t 的并发只读访问);
// 输出 (factor_out / Qx,Qz / checkpoint 文件) 均按曲线索引独立, 无共享写.
// 返回: 1=因子, 0=无因子, -1=错误, 2=被 SIGINT 中止(已写 checkpoint).
/* 已完成曲线数 (文件作用域, 让"曲线内"的进度回调也能读到) */

/* 定义在后面 (需要 g_stage1_bar / stage1_progress_set), 这里先声明给调用点用 */
static int edwards_checkpoint_progress_bar(void* ctx, const edwards_checkpoint_t* cur);

// Keep DONE_V1 byte compatibility. Decode only at the recovery seam.
static bool edwards_completed_read(ecm_stage1_checkpoint::Context& checkpoint, uint32_t curve,
                                   uint64_t sigma, mpz_srcptr n, uint64_t b1, mpz_ptr factor,
                                   int& rc, CurveOutcome& outcome) {
    std::string bytes;
    if (!checkpoint.read_blob(ecm_stage1_curve_record(curve), bytes))
        return false;
    std::istringstream in(bytes);
    std::string tag, value, extra, line;
    uint64_t saved_sigma;
    if (!(in >> tag) || tag != "ECM_EDWARDS_DONE_V1" ||
        !(in >> saved_sigma >> rc >> value >> std::quoted(line)) || (in >> extra) ||
        saved_sigma != sigma || (rc != 0 && rc != 1) || mpz_set_str(factor, value.c_str(), 16))
        return false;
    try {
        outcome.sigma = sigma;
        if (line.empty()) {
            if (rc != 0)
                return false;
            outcome.kind = CurveKind::CompletedWithoutPoint;
            mpz_set_ui(outcome.value, 0);
            return true;
        }
        const auto saved = ecm_edwards::parse_saved_curve(line);
        if (saved.n != ecm_edwards::hex(n) || saved.sigma != sigma || saved.b1 != b1 ||
            saved.factor != (rc == 1) || mpz_set_str(outcome.value, saved.x.c_str(), 16))
            return false;
        if (rc == 1 && mpz_cmp(outcome.value, factor))
            return false;
        outcome.kind = rc ? CurveKind::Factor : CurveKind::Stage2Point;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static bool edwards_completed_write(ecm_stage1_checkpoint::Context& checkpoint, uint32_t curve,
                                    uint64_t sigma, mpz_srcptr n, uint64_t b1,
                                    const CurveOutcome& outcome) {
    const bool hit = outcome.kind == CurveKind::Factor;
    const auto line = outcome.kind == CurveKind::CompletedWithoutPoint
                          ? std::string{}
                          : ecm_edwards::save_line(n, outcome.value, sigma, b1, hit);
    std::ostringstream out;
    out << "ECM_EDWARDS_DONE_V1\n"
        << sigma << ' ' << int(hit) << ' ' << (hit ? ecm_edwards::hex(outcome.value) : "0") << ' '
        << std::quoted(line) << '\n';
    return checkpoint.write_blob(ecm_stage1_curve_record(curve), out.str());
}

static int edwards_run_one_curve(ecm_stage1_checkpoint::Context& checkpoint, mpz_srcptr N,
                                 mpz_srcptr s, const ExecutionOptions& opt, double B1, double B2,
                                 uint32_t idx, uint64_t sigma, uint32_t curves, mpz_t factor_out,
                                 const std::string& ckpt_path, CurveOutcome& outcome) {
    mpz_t Qx, Qz;
    mpz_inits(Qx, Qz, NULL);

    int rc;
    if (!ckpt_path.empty() &&
        edwards_completed_read(checkpoint, idx, sigma, N, uint64_t(B1), factor_out, rc, outcome)) {
        mpz_clears(Qx, Qz, NULL);
        return rc;
    }
    {
        EdwardsCheckpointCtx ck;
        ck.opt = &opt;
        ck.checkpoint = &checkpoint;
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
            uint64_t rsp;
            uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy, rex, rey, rez;
            mpz_inits(rdx, rdy, rex, rey, rez, NULL);
            if (ecm_edwards_read_stage1(checkpoint, ckpt_path, rcm, &rsp, &rebs, &rbn, &rds, rdx,
                                        rdy, rex, rey, rez)) {
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

        const uint32_t chunk_bits =
            static_cast<uint32_t>(std::clamp<size_t>(mpz_sizeinbase(s, 2) / 512, 64, 4096));
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
        if (!classify_edwards_outcome(outcome, N, sigma, Qx, Qz, factor_out, rc == 1))
            rc = ECM_ERROR;
        if (rc >= 0 && !ckpt_path.empty() &&
            !edwards_completed_write(checkpoint, idx, sigma, N, uint64_t(B1), outcome))
            rc = ECM_ERROR;
    }
    mpz_clears(Qx, Qz, NULL);
    return rc;
}

// ---------------------------------------------------------------------------
// 多曲线并行 (每线程一条曲线, 动态取号)
// ---------------------------------------------------------------------------

struct EdwardsWorkerCtx {
    ecm_stage1_checkpoint::Context* checkpoint;
    mpz_srcptr N;
    mpz_srcptr s;
    const ExecutionOptions* opt;
    double B1, B2;
    uint32_t curves;
    const uint64_t* sigmas;
    const std::string* ckpt_paths;
    mpz_t* factors;
    int* array_found;
    CurveOutcome* outcomes;
    std::atomic<uint32_t>* next;
    std::atomic<int>* aborted;
    /* SIMD 批量 (8 曲线/批) 相关 */
    int simd_ok = 0;
    std::atomic<uint32_t>* done_ctr = nullptr;
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

struct EdSimdProg {
    ecm_stage1_checkpoint::Context* checkpoint;
    std::atomic<uint32_t>* done_ctr;
    uint32_t curves, batch_first, batch_count;
    size_t s_bits;
    long long t0_ms;
    const ExecutionOptions* opt;
    double B1, B2;
    const uint64_t* sigmas;
    const ifma_ctx_t* mc;
    long long last_ms;
    long long interval_ms;
    bool write_failed = false;
};

static int edwards_simd_progress(void* p, size_t bits_done, size_t bits_total, const uint64_t* Rx,
                                 const uint64_t* Ry, const uint64_t* Rz) {
    EdSimdProg* pr = (EdSimdProg*)p;
    auto& checkpoint = *pr->checkpoint;
    const ifma_ctx_t* mc = pr->mc;

    /* --- checkpoint (与标量路径同一套写入器/路径/字段) --- */
    const long long now = stage1_now_ms();
    const bool due = !pr->opt->persistence.tmp_dir.empty() &&
                     ((pr->interval_ms > 0 && now - pr->last_ms >= pr->interval_ms) ||
                      ecm_stage1_stop_requested());
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
                if (ecm_edwards_write_stage1(checkpoint, path, cm, 2, expbuf, (uint32_t)bits_done,
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
                ecm_ts_fprintf(stdout,
                               "checkpoint written (simd batch %u): %zu lane(s) @ bitnum=%zu\n",
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
static int edwards_checkpoint_progress_bar(void* ctx, const edwards_checkpoint_t* cur) {
    if (ctx) {
        EdwardsCheckpointCtx* ck = (EdwardsCheckpointCtx*)ctx;
        if (ck->opt && ck->s && ck->opt->stage1_curves > 0) {
            const size_t s_bits = (size_t)mpz_sizeinbase(ck->s, 2);
            if (s_bits > 0) {
                const uint32_t base = g_stage1_done.load(std::memory_order_relaxed);
                double frac = (double)cur->bitnum / (double)s_bits;
                if (frac > 1.0)
                    frac = 1.0;
                const double done_f = (double)base + frac;
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                stage1_progress_set(done_f, ck->opt->stage1_curves);
            }
        }
    }
    return edwards_checkpoint_progress(ctx, cur);
}

static int edwards_run_batch_simd_impl(ecm_stage1_checkpoint::Context& checkpoint,
                                       const ExecutionOptions& opt, mpz_srcptr N, mpz_srcptr s,
                                       double B1, double B2, uint32_t first, uint32_t count,
                                       const uint64_t* sigmas, mpz_t* factors, int* array_found,
                                       std::atomic<uint32_t>* done_ctr, CurveOutcome* outcomes) {
    if (!opt.persistence.tmp_dir.empty()) {
        bool completed = true;
        std::vector<int> statuses(count);
        for (uint32_t k = 0; k < count; ++k)
            if (!edwards_completed_read(checkpoint, first + k, sigmas[first + k], N, uint64_t(B1),
                                        factors[first + k], statuses[k], outcomes[first + k])) {
                completed = false;
                break;
            }
        if (completed) {
            for (uint32_t k = 0; k < count; ++k)
                array_found[first + k] = statuses[k] ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
            if (done_ctr)
                done_ctr->fetch_add(count, std::memory_order_relaxed);
            return 0;
        }
    }
    // Exceptional setup lanes need scalar factor/error classification before
    // building an affine SIMD dictionary (which requires invertible inputs).
    bool exceptional = false;
    for (uint32_t k = 0; k < count; ++k) {
        ecm_edwards::Integer d, x, y, a24, f;
        if (ecm_edwards::construct(d.v, x.v, y.v, a24.v, f.v, sigmas[first + k], N) !=
            ecm_edwards::Setup::Ready)
            exceptional = true;
    }
    auto run_exceptional = [&]() {
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t i = first + k;
            const auto path =
                opt.persistence.tmp_dir.empty() ? std::string() : ecm_stage1_curve_record(i);
            const int rc = edwards_run_one_curve(checkpoint, N, s, opt, B1, B2, i, sigmas[i],
                                                 opt.stage1_curves, factors[i], path, outcomes[i]);
            if (rc == 2 || rc < 0)
                return rc;
            array_found[i] = rc == 1 ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
            if (rc == 1 && !validate_stage1_factor(N, factors[i]))
                return ECM_ERROR;
            if (done_ctr)
                done_ctr->fetch_add(1, std::memory_order_relaxed);
        }
        return 0;
    };
    if (exceptional)
        return run_exceptional();
    ed_soa_ctx_t ctx;
    /* 字典窗口跟随配置 (naf_w / ini), 不再硬编码 w=8:
       这样 SIMD 与标量用同一套 digit/字典, 存档逐字节可比, 也便于混合与续跑。
       代价是字典内存 = 3*2^(w-2)*8n 字节 (w=12/n=155 约 29 MB/批), 见启动信息。 */
    int w = edwards_get_naf_w();
    if (w < 3 || w > 12)
        w = 8;
    {
        const int rc =
            ed_soa_init_ex(&ctx, N, w, to_ifma_field_mode(opt.execution.selection.simd_reduction));
        if (rc != 0) {
            /* 保持与原来完全相同的控制流 (先尝试建 ctx, 失败就返回 -1), 只把
               "SIMD stage-1 internal error" 这句含糊的日志换成能看出原因的话:
               reduction=mersenne-fold 要求 N = 2^k-1 (k>=64), 这是最常见的误用。 */
            bool want_mers =
                (opt.execution.selection.simd_reduction == ecm_stage1::SimdReduction::MersenneFold);
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
                               "with dict w=%d.\n",
                               rc, mpz_sizeinbase(N, 2), w);
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
    for (uint32_t k = 0; k < 8; k++)
        sg[k] = sigmas[first + (k < count ? k : count - 1)];

    const int setup_rc = ed_soa_set_curves(&ctx, sg, 8);
    if (setup_rc != 0) {
        ed_soa_clear(&ctx);
        return setup_rc == -4 ? run_exceptional() : -1;
    }

    EdSimdProg pr;
    pr.done_ctr = done_ctr;
    pr.curves = (done_ctr && opt.stage1_curves) ? opt.stage1_curves : count;
    pr.batch_first = first;
    pr.batch_count = count;
    pr.s_bits = mpz_sizeinbase(s, 2);
    pr.t0_ms = opt.stage1_t0_ms;
    pr.opt = &opt;
    pr.checkpoint = &checkpoint;
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
        for (uint32_t k = 0; k < count; k++)
            mpz_inits(rxs[k], rys[k], rzs[k], NULL);
        uint32_t common_bitnum = 0;
        int all_ok = (count > 0);
        for (uint32_t k = 0; k < count && all_ok; k++) {
            const uint32_t i = first + k;
            const std::string path = ecm_stage1_curve_record(i);
            ecm_save_common rcm;
            uint64_t rsp;
            uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy;
            mpz_inits(rdx, rdy, NULL);
            const bool ok = ecm_edwards_read_stage1(checkpoint, path, rcm, &rsp, &rebs, &rbn, &rds,
                                                    rdx, rdy, rxs[k], rys[k], rzs[k]);
            mpz_clears(rdx, rdy, NULL);
            if (!ok || rbn == 0 || rcm.sigma != sigmas[i] || rcm.B1 != (uint64_t)B1) {
                all_ok = 0;
                break;
            }
            if (k == 0)
                common_bitnum = rbn;
            else if (rbn != common_bitnum) {
                all_ok = 0;
                break;
            }
        }
        if (all_ok && common_bitnum > 0) {
            const int rr = ed_soa_set_resume(&ctx, common_bitnum, (int)count, rxs.data(),
                                             rys.data(), rzs.data());
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "simd batch %u: resume from .ckpt @ bitnum=%u (%u lane(s))%s\n",
                           first, common_bitnum, count,
                           rr == 0 ? "" : " -- unusable, restarting from scratch");
            if (rr != 0)
                ctx.resume_digit = 0;
        }
        for (uint32_t k = 0; k < count; k++)
            mpz_clears(rxs[k], rys[k], rzs[k], NULL);
    }
    ed_soa_set_progress(&ctx, edwards_simd_progress, &pr);

    mpz_t Qx[8], Qz[8], fac[8];
    for (int k = 0; k < 8; k++)
        mpz_inits(Qx[k], Qz[k], fac[k], NULL);
    const int aborted = ed_soa_stage1(&ctx, s, 8, Qx, Qz, fac);
    if (aborted) { /* 中途停: 不写任何存档 */
        for (int k = 0; k < 8; k++)
            mpz_clears(Qx[k], Qz[k], fac[k], NULL);
        ed_soa_clear(&ctx);
        return pr.write_failed ? ECM_ERROR : 2;
    }

    int rc = 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = first + k;
        if (mpz_cmp_ui(fac[k], 1) > 0) {
            mpz_set(factors[i], fac[k]);
            array_found[i] = ECM_FACTOR_FOUND_STEP1;
            if (!validate_stage1_factor(N, factors[i]))
                rc = ECM_ERROR;
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n", i,
                           (unsigned long long)sigmas[i]);
        }
        if (!classify_edwards_outcome(outcomes[i], N, sigmas[i], Qx[k], Qz[k], fac[k],
                                      array_found[i] == ECM_FACTOR_FOUND_STEP1))
            rc = ECM_ERROR;
        if (rc == 0 && !opt.persistence.tmp_dir.empty() &&
            !edwards_completed_write(checkpoint, i, sigmas[i], N, uint64_t(B1), outcomes[i]))
            rc = ECM_ERROR;
    }
    for (int k = 0; k < 8; k++)
        mpz_clears(Qx[k], Qz[k], fac[k], NULL);
    ed_soa_clear(&ctx);

    if (done_ctr) {
        const uint32_t done = done_ctr->fetch_add(count, std::memory_order_relaxed) + count;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done, opt.stage1_curves);
    }
    (void)s;
    return rc;
}

static void edwards_worker(EdwardsWorkerCtx* ctx) {
    auto& checkpoint = *ctx->checkpoint;
    const int use_simd = (ctx->opt->execution.selection.cpu_implementation !=
                          ecm_stage1::CpuImplementation::GmpScalar) &&
                         ctx->simd_ok;
    for (;;) {
        if (ecm_stage1_stop_requested()) {
            ctx->aborted->fetch_or(1, std::memory_order_relaxed);
            break;
        }
        uint32_t i, n;
        if (use_simd) { /* 静态分批: 一批 8 条 */
            i = ctx->next->fetch_add(8, std::memory_order_relaxed);
            if (i >= ctx->curves)
                break;
            n = ctx->curves - i;
            if (n > 8)
                n = 8;
        } else {
            i = ctx->next->fetch_add(1, std::memory_order_relaxed);
            if (i >= ctx->curves)
                break;
            n = 1;
        }
        if (use_simd && n >= 2) {
            /* 批内至少 2 条才值得批量 (1 条走标量, 见 §13.6 的 hybrid 规则) */
            const int rc = edwards_run_batch_simd_impl(
                checkpoint, *ctx->opt, ctx->N, ctx->s, ctx->B1, ctx->B2, i, n, ctx->sigmas,
                ctx->factors, ctx->array_found, ctx->done_ctr, ctx->outcomes);
            if (rc == 2) {
                ctx->aborted->fetch_or(1, std::memory_order_relaxed);
                break;
            }
            if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  batch " << i << " -> SIMD stage-1 internal error" << std::endl;
                ctx->aborted->fetch_or(2, std::memory_order_relaxed);
                break;
            }
            continue;
        }
        if (use_simd && n == 1) { /* fallthrough to scalar for the tail curve */
        }
        {
            const uint64_t sigma = ctx->sigmas[i];
            const int rc = edwards_run_one_curve(checkpoint, ctx->N, ctx->s, *ctx->opt, ctx->B1,
                                                 ctx->B2, i, sigma, ctx->curves, ctx->factors[i],
                                                 ctx->ckpt_paths[i], ctx->outcomes[i]);
            if (rc == 1) {
                ctx->array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if (!validate_stage1_factor(ctx->N, ctx->factors[i])) {
                    ctx->aborted->store(1);
                    break;
                }
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                if (ctx->opt->execution.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigma << " -> factor found"
                              << std::endl;
                } else {
                    ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n", i,
                                   (unsigned long long)sigma);
                }
            } else if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  curve " << i << " sigma=" << sigma
                          << " -> Edwards stage-1 internal error" << std::endl;
                ctx->aborted->fetch_or(2, std::memory_order_relaxed);
                break;
            } else if (rc == 2) {
                ctx->aborted->fetch_or(1, std::memory_order_relaxed);
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
    if (hw == 0)
        hw = 1;
    uint32_t t = (uint32_t)hw;
    if (t > curves)
        t = curves;
    return t;
}

// Run one Edwards (Atkin-Morain) stage-1 batch on the CPU. One curve per sigma.
//
// curves == 1 or opt.execution.stage1_threads == 1 -> 顺序执行 (与历史行为一致);
// 否则按曲线并行 (每线程独立曲线, 动态取号), 吞吐随核数线性提升.
ComputationResult run_edwards_stage1(const ecm_stage1::Stage1Request& request,
                                     ExecutionOptions& opt,
                                     ecm_stage1_checkpoint::Context& checkpoint) {
    const auto N = request.modulus;
    const auto B1 = request.b1, B2 = request.b2;
    const auto curves = request.curves;
    ComputationResult result;
    result.failure = ecm_stage1::FailureSource::Preparation;
    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return result;
    }

    // NAF 窗口 (字典大小 2^(w-2)); 0 = 保持 ecm_edwards_cpu 的默认值.
    if (opt.execution.naf_w >= 2) {
        edwards_set_naf_w(opt.execution.naf_w);
    } else if (opt.execution.naf_w != 0) {
        std::cerr << "naf_w must be 0 (default) or >= 2" << std::endl;
        return result;
    }

    // s = 48 * lcm(1..B1). (GPU batch uses lcm(1..B1); Edwards adds the
    // lcm(12,16)=48 torsion factor, matching Prime95 ecm_calc_exp.)
    ecm_edwards::Integer scalar;
    auto* s = scalar.v;
    if (!compute_batch_s(s, B1, 1, opt.persistence.exponent_cache_dir)) {
        std::cerr << "Failed to compute Edwards stage-1 exponent" << std::endl;

        return result;
    }
    mpz_mul_ui(s, s, 48);

    ecm_edwards::Array factors_owner(curves);
    auto* factors = factors_owner.data();
    std::vector<int> statuses(curves, ECM_NO_FACTOR_FOUND);
    auto* array_found = statuses.data();
    result.curves.resize(curves);
    auto* outcomes = result.curves.data();

    const bool use_ckpt = !opt.persistence.tmp_dir.empty();
    if (use_ckpt) {
        if (!ensure_dir(opt.persistence.tmp_dir)) {
            std::cerr << "ERROR: cannot create tmp_dir: " << opt.persistence.tmp_dir << std::endl;

            return result;
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
            sigmas[i] = checkpoint.sigma(i, opt.identity.fixed_sigma64 + i);
        } else {
            sigmas[i] = checkpoint.sigma(i, random_sigma_u64());
            if (!opt.persistence.tmp_dir.empty()) {
                const std::string cp = ecm_stage1_curve_record(i);
                ecm_save_common rcm;
                uint64_t rsp;
                uint32_t rebs, rbn, rds;
                mpz_t dx, dy, rx, ry, rz;
                mpz_inits(dx, dy, rx, ry, rz, NULL);
                if (ecm_edwards_read_stage1(checkpoint, cp, rcm, &rsp, &rebs, &rbn, &rds, dx, dy,
                                            rx, ry, rz) &&
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
    if (nthreads == 0)
        nthreads = edwards_default_threads(curves);
    if (nthreads > curves)
        nthreads = curves;
    if (nthreads < 1)
        nthreads = 1;

    std::cout << "Using B1=" << B1 << ", B2=" << B2 << " (" << curves << " Edwards curves, CPU";
    if (nthreads > 1)
        std::cout << ", " << nthreads << " threads";
    std::cout << ")";
    if (use_ckpt)
        std::cout << " [saves -> " << opt.persistence.tmp_dir << "]";
    std::cout << std::endl;

    Stage1Progress progress;
    opt.stage1_t0_ms = stage1_now_ms();
    opt.stage1_curves = curves;
    g_stage1_done.store(0);
    stage1_progress_set(0, curves);
    bool aborted = false;
    bool failed = false;
    if (nthreads <= 1 && curves == 1) {
        for (uint32_t i = 0; i < curves; i++) {
            const int rc = edwards_run_one_curve(checkpoint, N, s, opt, B1, B2, i, sigmas[i],
                                                 curves, factors[i], ckpt_paths[i], outcomes[i]);
            if (rc == 1) {
                array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if (!validate_stage1_factor(N, factors[i])) {
                    failed = true;
                    aborted = true;
                    break;
                }
                if (opt.execution.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigmas[i] << " -> factor found"
                              << std::endl;
                }
            } else if (rc < 0) {
                failed = true;
                std::cerr << "  curve " << i << " sigma=" << sigmas[i]
                          << " -> Edwards stage-1 internal error" << std::endl;
                aborted = true;
                break;
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
        wctx.checkpoint = &checkpoint;
        /* ---- backend 选择 + 起始信息 (见 §13.6 第 7 条: simd 不允许静默降级) ---- */
        g_stage1_done.store(0, std::memory_order_relaxed);
        std::atomic<uint32_t>& done_ctr = g_stage1_done;
        {
            const int isa = driver_simd_isa_ok();
            if (opt.execution.selection.cpu_implementation ==
                    ecm_stage1::CpuImplementation::IfmaSimd &&
                !isa) {
                ecm_ts_fprintf(stderr, "ERROR: --impl simd requested but this CPU lacks "
                                       "AVX512-F/DQ/IFMA. Refusing to fall back silently; use "
                                       "--impl auto or scalar.\n");
                return result;
            }
            wctx.simd_ok = (opt.execution.selection.cpu_implementation ==
                            ecm_stage1::CpuImplementation::IfmaSimd)
                               ? 1
                           : (opt.execution.selection.cpu_implementation ==
                              ecm_stage1::CpuImplementation::Auto)
                               ? (isa && curves >= 8)
                               : 0;
            wctx.done_ctr = &done_ctr;
            char sbuf[160];
            snprintf(sbuf, sizeof(sbuf),
                     "simd (AVX512-IFMA: 8 curves per batch, 1 thread/batch, dict w=%d)",
                     edwards_get_naf_w() >= 3 ? edwards_get_naf_w() : 8);
            ecm_ts_fprintf(stdout, "CPU implementation : %s\n",
                           wctx.simd_ok ? sbuf : "gmp (scalar mpn, 1 curve/thread)");
            ecm_ts_fprintf(stdout, "N / s           : %zu bit N, %zu bit s\n", mpz_sizeinbase(N, 2),
                           mpz_sizeinbase(s, 2));
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
                    if (i)
                        a += ",";
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
        wctx.outcomes = outcomes;
        std::atomic<uint32_t> next(0);
        std::atomic<int> aborted_flag(0);
        wctx.next = &next;
        wctx.aborted = &aborted_flag;

        std::vector<std::future<void>> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.push_back(std::async(std::launch::async, [&wctx, t]() {
                apply_thread_affinity(wctx.opt->execution.affinity_cpus, t);
                edwards_worker(&wctx);
            }));
        for (auto& th : pool)
            th.get();
        const int stopped = aborted_flag.load(std::memory_order_relaxed);
        failed = (stopped & 2) != 0;
        aborted = stopped != 0;
    }

    if (!aborted)
        stage1_progress_set(curves, curves);
    progress.finish();
    if (aborted) {
        ecm_ts_fprintf(stdout, "stage-1 incomplete; checkpoints retained\n");

        if (!failed)
            result.stop();
        return result;
    }

    result.completion = ComputationResult::Completion::Completed;
    result.failure = ecm_stage1::FailureSource::None;
    result.legacy_code = ECM_NO_FACTOR_FOUND;
    result.validate(N);
    return result;
}

} // namespace ecm_cpu_stage1
