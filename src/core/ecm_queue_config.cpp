#include "ecm_queue_config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

void trim(std::string &s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
}

} // namespace

bool ecm_queue_config_load(const std::string &path, EcmQueueConfig &cfg) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    bool warned_legacy = false;   // legacy-key shim warns once per run
    while (std::getline(in, line)) {
        // Strip a UTF-8 BOM if present on the first line.
        if (!line.empty() && static_cast<unsigned char>(line[0]) == 0xEF &&
            line.size() >= 3 && static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key);
        trim(val);
        if (key.empty()) {
            continue;
        }

        const auto set_int = [&](int &dst) {
            try {
                dst = std::stoi(val);
            } catch (...) {
            }
        };
        const auto set_u32 = [&](uint32_t &dst) {
            try {
                const unsigned long long v = std::stoull(val);
                if (v <= 0xFFFFFFFFull) {
                    dst = static_cast<uint32_t>(v);
                }
            } catch (...) {
            }
        };
        const auto set_double = [&](double &dst) {
            try {
                dst = std::stod(val);
            } catch (...) {
            }
        };
        const auto set_bool = [&](bool &dst) {
            std::string v;
            for (char ch : val) v.push_back((char)std::tolower((unsigned char)ch));
            if (v == "1" || v == "true" || v == "yes" || v == "on") dst = true;
            else if (v == "0" || v == "false" || v == "no" || v == "off") dst = false;
        };
        const auto set_u64 = [&](uint64_t &dst) {
            try {
                dst = std::stoull(val);
            } catch (...) {
            }
        };
        // Scratch used by the legacy-key shim below: an integer whose value only feeds
        // an enumeration ("edwards = 1" -> method, "mont_torsion = 12" -> exponent).
        int fallback_int = 0;
        const auto warn_legacy = [&](const std::string &key, const std::string &old_form,
                                     const std::string &new_form) {
            if (warned_legacy) return;      // once per run, not once per key
            warned_legacy = true;
            fprintf(stderr,
                    "[ecm] NOTE: ecm.ini uses the pre-2026-09-24 key '%s' (%s); "
                    "please change it to '%s'. Legacy keys are still honoured.\n",
                    key.c_str(), old_form.c_str(), new_form.c_str());
        };

        // ---- [queue] ------------------------------------------------------
        if (key == "worktodo") cfg.worktodo = val;
        else if (key == "finished") cfg.finished = val;
        else if (key == "save_sync_dir_1") cfg.save_sync_dir_1 = val;
        else if (key == "save_sync_dir_2") cfg.save_sync_dir_2 = val;
        else if (key == "sync_mode") cfg.sync_mode = val;
        else if (key == "log_file") cfg.log_file = val;
        else if (key == "tmp_dir") cfg.tmp_dir = val;
        else if (key == "progress_color") cfg.progress_color = val;
        else if (key == "verbose") set_bool(cfg.verbose);

        // ---- [method] -----------------------------------------------------
        else if (key == "method") cfg.method = val;

        // ---- [cpu] --------------------------------------------------------
        else if (key == "backend") cfg.backend = val;
        else if (key == "field") cfg.field = val;
        else if (key == "stage1_threads") set_u32(cfg.stage1_threads);
        else if (key == "affinity") cfg.affinity = val;
        else if (key == "cpu_affinity") cfg.affinity = val;          // 已废弃别名 (见迁移表)
        else if (key == "save_name_pattern") cfg.save_name_pattern = val;
    else if (key == "exp_cache") cfg.exp_cache = val;

        // ---- [edwards] ----------------------------------------------------
        else if (key == "naf_w") set_int(cfg.naf_w);

        // ---- [mont] -------------------------------------------------------
        else if (key == "exponent") cfg.exponent = val;

        // ---- [task] -------------------------------------------------------
        else if (key == "sigma") set_u64(cfg.sigma);

        // ---- [handoff] ----------------------------------------------------
        else if (key == "p95_dir") cfg.p95_dir = val;

        // ---- [gpu] --------------------------------------------------------
        else if (key == "device") set_int(cfg.device);
        else if (key == "gpu_param") {
            set_int(cfg.gpu_param);
            if (cfg.gpu_param != 0 && cfg.gpu_param != 2 && cfg.gpu_param != 3) {
                fprintf(stderr, "[ecm] WARNING: gpu_param = %d is not supported (0 = Suyama "
                                "param0, 2 = param2 batch-2 / 6-torsion, 3 = gmp-ecm batch); "
                                "using 3.\n", cfg.gpu_param);
                cfg.gpu_param = 3;
            }
        }
        else if (key == "tpi") set_u32(cfg.tpi);
        else if (key == "wg_size") set_int(cfg.wg_size);
        else if (key == "ckpt_seconds") set_double(cfg.ckpt_seconds);
        else if (key == "gpuckpt_seconds") {
            warn_legacy(key, key + " = N", "ckpt_seconds = N");
            set_double(cfg.ckpt_seconds);
        }
        else if (key == "kernel_mul") cfg.kernel_mul = val;
        else if (key == "kernel_sqr") cfg.kernel_sqr = val;
        else if (key == "kernel_add") cfg.kernel_add = val;
        else if (key == "kernel_sub") cfg.kernel_sub = val;
        else if (key == "kernel_special_mult") cfg.kernel_special_mult = val;

        // ---- pre-2026-09-24 keys: mapped with a warning -------------------
        // These were one knob per method for the same concept (or a 0/1 flag for a
        // choice), which is exactly what the cleanup removed.  Fold them into the
        // single field so an existing ecm.ini keeps behaving as before.
        else if (key == "edwards" || key == "mont") {
            set_int(fallback_int);
            if (fallback_int != 0) cfg.method = (key == "mont") ? "mont" : "edwards";
            warn_legacy(key, key + " = N", "method = mont|edwards");
        }
        else if (key == "edwards_backend" || key == "mont_backend" || key == "edbackend") {
            cfg.backend = val;
            warn_legacy(key, key + " = <auto|simd|gmp>", "backend = <auto|simd|gmp>");
        }
        else if (key == "edwards_threads" || key == "mont_threads" || key == "edthreads") {
            set_u32(cfg.stage1_threads);
            warn_legacy(key, key + " = N", "stage1_threads = N");
        }
        else if (key == "edwards_mersenne" || key == "emersenne") {
            // auto | on/mersenne | off/montgomery  ->  field = auto|mersenne|montgomery
            std::string v;
            for (char ch : val) v.push_back((char)std::tolower((unsigned char)ch));
            if (v == "on" || v == "mersenne" || v == "mers" || v == "yes" || v == "1") cfg.field = "mersenne";
            else if (v == "off" || v == "montgomery" || v == "mont" || v == "no" || v == "0") cfg.field = "montgomery";
            else cfg.field = "auto";
            warn_legacy(key, key + " = <auto|on|off>", "field = <auto|mersenne|montgomery>");
        }
        else if (key == "edwards_naf_w" || key == "ednafw") {
            set_int(cfg.naf_w);
            warn_legacy(key, key + " = W", "naf_w = W");
        }
        else if (key == "mont_torsion") {
            set_int(fallback_int);
            cfg.exponent = (fallback_int == 12) ? "choose12" : "lcm";
            warn_legacy(key, key + " = <1|12>", "exponent = <lcm|choose12>");
        }
        else if (key == "mont_save_pattern") {
            cfg.save_name_pattern = val;
            warn_legacy(key, key + " = <pattern>", "save_name_pattern = <pattern>");
        }
        // Unknown keys are ignored.
    }
    return true;
}

bool ecm_queue_config_write_default(const std::string &path) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    out <<
"# ---------------------------------------------------------------------------\n"
"# ECM queue-manager configuration.\n"
"# ECM 队列管理器配置。\n"
"# Format: key = value; '#' starts a comment. Relative paths are resolved against\n"
"# the directory of the executable. Unknown keys are ignored (default kept).\n"
"# \n"
"# Keys are grouped by WHEN they apply: [method] selects the stage-1 engine, and\n"
"# the [cpu] / [edwards] / [mont] / [gpu] groups only matter for that method.\n"
"# 格式：key = value；'#' 起注释。相对路径相对可执行文件所在目录解析；\n"
"# 未知键忽略（保留默认值）。\n"
"# \n"
"# 键按\"何时生效\"分组：[method] 选 stage-1 引擎；\n"
"# [cpu] / [edwards] / [mont] / [gpu] 只在对应 method 下生效。\n"
"# \n"
"# 旧键仍可用（每次运行提示一次），对应关系：\n"
"#   edwards = 0/1, mont = 0/1        ->  method = gpu|edwards|mont\n"
"#   edwards_backend / mont_backend   ->  backend\n"
"#   edwards_threads / mont_threads   ->  stage1_threads\n"
"#   edwards_mersenne = auto|on|off   ->  field = auto|mersenne|montgomery\n"
"#   edwards_naf_w                    ->  naf_w\n"
"#   mont_torsion = 1|12              ->  exponent = lcm|choose12\n"
"#   mont_save_pattern                ->  save_name_pattern\n"
"# ---------------------------------------------------------------------------\n"
"# [queue] work queue, logging, saves\n"
"# [queue] 工作队列、日志、存档\n"
"# Work file: one ECMSTAGE2 (= k,b,n,c) line per task.\n"
"# 工作文件：每行一个 ECMSTAGE2（= k,b,n,c）任务。\n"
"worktodo = worktodo.txt\n"
"# Successfully completed tasks are appended here; the queue entry is removed.\n"
"# 成功完成的任务追加到此处，并从队列中移除。\n"
"finished = worktodo.finished.txt\n"
"# Log file (timestamped, append-only); empty = screen only.\n"
"# 日志文件（带时间戳、追加写）；留空 = 仅屏幕输出。\n"
"log_file = screen.log\n"
"# Stage-1 output directory. The CPU paths write m{n}_{b1}.save here; the Edwards\n"
"# path also writes e{n:07d}_c{k}.tmp/.ckpt for the stage-2 handoff and checkpoints.\n"
"# stage-1 输出目录：CPU 路径在此写 m{n}_{b1}.save；Edwards 路径另有\n"
"# e{n:07d}_c{k}.tmp/.ckpt，用于 stage-2 交接与自检查点。\n"
"tmp_dir = .\n"
"# Sync .save files into these directories after each task; empty = disabled.\n"
"# 任务完成后把 .save 同步到这两个目录；留空 = 关闭同步。\n"
"save_sync_dir_1 =\n"
"# (second sync target, same rules as save_sync_dir_1)\n"
"# （第二个同步目标，规则同 save_sync_dir_1）\n"
"save_sync_dir_2 =\n"
"# Sync mode after each task: incremental | full.\n"
"# 每个任务完成后的同步模式：incremental | full。\n"
"sync_mode = incremental\n"
"# Progress-bar colour: none|red|green|yellow|blue|magenta|cyan|white|grey.\n"
"# 进度条颜色：none|red|green|yellow|blue|magenta|cyan|white|grey。\n"
"progress_color = cyan\n"
"# Verbose output: true | false.\n"
"# 详细输出：true | false。\n"
"verbose = true\n"
"# ---------------------------------------------------------------------------\n"
"# [method] which stage-1 engine to use (exactly one)\n"
"# [method] stage-1 引擎（三选一）\n"
"#   gpu     : GPU batched stage 1      (the [gpu] group applies); WHICH GPU\n"
"#             implementation runs is decided by the executable, not by the ini:\n"
"#             ecm.exe = OpenCL, ecm_cuda.exe = CUDA/CGBN.  The startup banner\n"
"#             prints the real one ('gpu backend : ...').\n"
"#   edwards : CPU Edwards / Atkin-Morain (the [edwards] group applies);\n"
"#   mont    : CPU Suyama-sigma Montgomery -- same curve family as gmp-ecm\n"
"#             -param 0 and Prime95 sigma_type=1 (the [mont] group applies).\n"
"#   gpu     : GPU 批量 stage 1（[gpu] 组生效）；具体跑 OpenCL 还是 CUDA/CGBN 由\n"
"#             **可执行文件**决定，不由 ini 决定：ecm.exe = OpenCL，\n"
"#             ecm_cuda.exe = CUDA/CGBN。启动横幅会打印真实后端（'gpu backend : ...'）。\n"
"#   edwards : CPU Edwards / Atkin-Morain（[edwards] 组生效）；\n"
"#   mont    : CPU Suyama-σ 蒙哥马利，与 gmp-ecm -param 0 / Prime95\n"
"#             sigma_type=1 同曲线族（[mont] 组生效）。\n"
"method = gpu\n"
"# ---------------------------------------------------------------------------\n"
"# [cpu] method = edwards | mont\n"
"# [cpu] 两条 CPU 路径共用\n"
"# Modular-multiplication backend:\n"
"#   auto : AVX512-IFMA 8-curve batch when the CPU supports it, else scalar GMP;\n"
"#   simd : force the batch; hard error if the CPU lacks AVX512-F/DQ/IFMA;\n"
"#   gmp  : force the scalar mpn path (1 curve/task; also the A/B baseline).\n"
"# 模乘后端：\n"
"#   auto : CPU 支持时用 AVX512-IFMA 8 曲线批，否则标量 GMP；\n"
"#   simd : 强制批量（缺 AVX512-F/DQ/IFMA 直接报错，不静默降级）；\n"
"#   gmp  : 强制标量 mpn（1 曲线/任务，也用作对照基线）。\n"
"backend = auto\n"
"# Reduction domain of the SIMD field layer (ignored when backend = gmp).\n"
"# The domain actually chosen is printed on the \"field layer :\" line:\n"
"#   auto       : Mersenne fold for N = 2^k-1 (half the madds per multiply),\n"
"#                Montgomery reduction otherwise;\n"
"#   mersenne   : force the fold; N must be 2^k-1 or the run fails;\n"
"#   montgomery : force Montgomery CIOS even for N = 2^k-1 (A/B baseline).\n"
"# SIMD 域归约方式（backend = gmp 时忽略）；实际选中的域打印在 \"field layer :\" 行：\n"
"#   auto       : N=2^k-1 用 Mersenne 折叠（每模乘 madds 减半），否则蒙哥马利；\n"
"#   mersenne   : 强制折叠（N 形状不符则报错）；\n"
"#   montgomery : 强制蒙哥马利 CIOS（对照用）。\n"
"field = auto\n"
"# Stage-1 worker threads: 0 = auto = min(#tasks, #cores), 1 = serial, n = fixed.\n"
"# One task = one 8-curve SIMD batch (simd) or one curve (gmp), so keeping N cores\n"
"# busy needs at least 8*N curves per task (see -gpucurves).\n"
"# stage-1 工作线程：0 = 自动 = min(任务数, 核数)，1 = 串行，n = 指定。\n"
"# 一个任务 = 一个 8 曲线批（simd）或 1 条曲线（gmp）；想占满 N 核，\n"
"# 任务至少要有 8N 条曲线（见 -gpucurves）。\n"
"stage1_threads = 0\n"
"# Pin worker t to logical CPU list[t modulo len]: \"1,3,5,7\", \"0-7\", \"0-3,8,10-11\".\n"
"# Empty / none / auto = let the OS schedule.  On hybrid CPUs (Zen5 + Zen5c, P+E)\n"
"# measure first: leaving it unset was fastest on the test machine (README 4.1).\n"
"# 亲核性：worker t 绑定到 list[t modulo len]（支持 \"1,3,5,7\"、\"0-7\"、\"0-3,8,10-11\"）；\n"
"# 留空 / none / auto = 交给系统调度。混合核机器（Zen5 + Zen5c、大小核）请先实测：\n"
"# 测试机上不绑定最快（README 4.1）。\n"
"affinity =\n"
"# File-name template of the CPU paths: {n} = Mersenne exponent when N = 2^k-1\n"
"# (else the bit length); {b1} = compact bound (1e5, 110e6, 12345).\n"
"# The READING side ignores this template: it takes B1 from the last '_' token\n"
"# before the trailing .save, so keep that shape when editing.\n"
"# CPU 路径的存档名模板：{n} = N=2^k-1 时的 Mersenne 指数（否则位长）；\n"
"# {b1} = 紧凑界（1e5、110e6、12345）。\n"
"# 读取侧不依赖模板：只取文件名最后一个 '_' 与 .save 之间那段作为 B1，\n"
"# 改模板时请保持该形状。\n"
"save_name_pattern = m{n}_{b1}.save\n"
        "# exp_cache = <dir|off>           : validated cache for s = torsion*lcm(1..B1).\n"
        "#                                  B1=260e6: ~10 s to build, ~0.3 s to load;\n"
        "#                                  default = exe dir, off = do not cache.\n"
        "exp_cache = .\n"
"# ---------------------------------------------------------------------------\n"
"# [edwards] method = edwards only\n"
"# [edwards] 仅 method = edwards\n"
"# NAF window (dictionary = 2^(w-2) entries); 0 = built-in default (12).\n"
"# Larger w = fewer point additions but a bigger table (w >= 16 stops paying off).\n"
"# NAF 窗口（字典 = 2^(w-2) 项）；0 = 内置默认 12。\n"
"# w 越大点加法越少但表越大（w >= 16 起不再划算）。\n"
"naf_w = 0\n"
"# ---------------------------------------------------------------------------\n"
"# [mont] method = mont only\n"
"# [mont] 仅 method = mont\n"
"# Stage-1 exponent convention -- decides which stage-2 producer this run stays\n"
"# compatible with (docs/ECM_Montgomery_STAGE1.md 16.7):\n"
"#   lcm      = lcm(1..B1)      -- gmp-ecm -param 0; our acceptance metric;\n"
"#   choose12 = 12 * lcm(1..B1) -- Prime95 sigma_type=1; use it when Prime95\n"
"#                                 will run stage 2 on our point.\n"
"# stage-1 指数约定 —— 决定本次运行与哪个 stage-2 产出方兼容（文档 §16.7）：\n"
"#   lcm      = lcm(1..B1)      —— gmp-ecm -param 0，本项目验收口径；\n"
"#   choose12 = 12 * lcm(1..B1) —— Prime95 sigma_type=1，\n"
"#                                 交给 Prime95 做 stage 2 时使用。\n"
"exponent = lcm\n"
"# ---------------------------------------------------------------------------\n"
"# [task] per-run task parameters\n"
"# [task] 每次运行的任务参数\n"
"# Fixed first sigma (0 = random per run); curve i uses sigma + i.\n"
"# Keep it <= 2^63: Prime95 ECMSTAGE2 reads SIGMA with atoll(), and a larger value\n"
"# silently rebuilds a different curve (docs 16.7.1). gmp-ecm has no such limit.\n"
"# 固定起始 σ（0 = 每次随机）；第 i 条曲线用 sigma + i。\n"
"# 请保持 <= 2^63：Prime95 的 ECMSTAGE2 用 atoll() 读 SIGMA，更大的值会静默\n"
"# 重建出另一条曲线（文档 16.7.1）；交给 gmp-ecm 则无此限制。\n"
"sigma = 0\n"
"# ---------------------------------------------------------------------------\n"
"# [handoff] Prime95 stage-2 handoff\n"
"# [handoff] Prime95 stage-2 交接\n"
"# Informational only: ecm.exe never writes into the Prime95 directory.  The\n"
"# transfer program ecm_p95feeder reads the same key from its own feeder.ini.\n"
"# 仅作提示：ecm.exe 从不写 Prime95 目录；真正投递的是 ecm_p95feeder，\n"
"# 它从自己的 feeder.ini 读同名键。\n"
"p95_dir =\n"
"# ---------------------------------------------------------------------------\n"
"# [ckpt] mid-stage-1 checkpoint interval\n"
"# [ckpt] stage-1 中途检查点间隔\n"
"# Seconds between mid-stage-1 checkpoint writes, for every method that has one:\n"
"#   gpu        : GPU curve buffer + exponent offset (same v4 frame for OpenCL\n"
"#                and CUDA; the header records which parametrization it is)\n"
"#   edwards    : <tmp_dir>/e{n}_c{k}.ckpt\n"
"#   montgomery : <tmp_dir>/m{n}_{b1}_c{k}.ckpt\n"
"# 0 = no periodic autosave (Ctrl+C still saves one).\n"
"# 中途检查点写入间隔（秒），凡是有中途检查点的方法都使用：\n"
"#   gpu        : GPU 曲线缓冲 + 指数进度（OpenCL 与 CUDA 用同一套 v4 头，\n"
"#                头里记录了参数化，不会串）\n"
"#   edwards    : <tmp_dir>/e{n}_c{k}.ckpt\n"
"#   montgomery : <tmp_dir>/m{n}_{b1}_c{k}.ckpt\n"
"# 0 = 不做定时保存（Ctrl+C 仍会保存一次）。\n"
"ckpt_seconds = 600\n"
"# ---------------------------------------------------------------------------\n"
"# [gpu] method = gpu only\n"
"# [gpu] 仅 method = gpu\n"
"# Curve parametrization of the GPU stage-1 path.\n"
"#   gpu_param = 0 : Suyama param0 (Prime95 sigma_type=1 / gmp-ecm -param 0) -- the\n"
"#                   SAME curves as the CPU --method mont path, with a Z/12 torsion\n"
"#                   point.  Effective divisor D ~ 21-23 vs ~6.4-7.6 for the batch\n"
"#                   family (a ~3x ratio in D); the per-curve SUCCESS-RATE ratio is\n"
"#                   much smaller -- measured 1.30x-1.8x at B1=256 over bits 15-40\n"
"#                   (bit20: 30.47% vs 21.64%).  Save file is written in param0 form,\n"
"#                   so gmp-ecm -param 0 and Prime95 both accept it for stage 2.\n"
"#                   Implemented for the CUDA/CGBN build (ecm_cuda); OpenCL refuses 0.\n"
"#   gpu_param = 3 : gmp-ecm batch parametrization (P=(2:1), d = sigma/2^32) -- the\n"
"#                   historical GPU path, Z/4 torsion, save file carries PARAM=3.\n"
"# GPU stage-1 的曲线参数化：\n"
"#   0 = Suyama param0（Prime95 sigma_type=1 / gmp-ecm -param 0）：与 CPU --method mont\n"
"#       同一条曲线，带 Z/12 挠点。有效除子 D≈21–23（batch 族 ≈6.4–7.6，**D 的比值 ≈3×**），\n"
"#       但**单曲线成功率的比值远小于 3×**：B1=256 实测 bit15–40 为 1.30×–1.8×\n"
"#       （bit20：30.47% vs 21.64%）。存档写成 param0 形态，gmp-ecm -param 0 与 Prime95\n"
"#       都能接着做 stage 2。仅 CUDA/CGBN 版实现（ecm_cuda），OpenCL 后端会明确拒绝 0。\n"
"#   3 = gmp-ecm batch 参数化（P=(2:1)、d = sigma/2^32）：老的 GPU 路径，Z/4 挠点，\n"
"#       存档带 PARAM=3。\n"
"gpu_param = 0\n"
"# GPU device index: OpenCL device in ecm.exe, CUDA device in ecm_cuda.exe.\n"
"# GPU 设备索引：ecm.exe 下是 OpenCL 设备，ecm_cuda.exe 下是 CUDA 设备。\n"
"device = 0\n"
"# Threads per instance -- **OpenCL only**; the CUDA/CGBN path picks its own\n"
"# kernel tier (TPI is baked into the kernel it selects).\n"
"# 每实例线程数 —— **仅 OpenCL**；CUDA/CGBN 路径自己选内核档位（TPI 编在内核里）。\n"
"tpi = 8\n"
"# Explicit work-group size; 0 = auto (**OpenCL only**).\n"
"# 显式工作组大小；0 = 自动（**仅 OpenCL**）。\n"
"wg_size = 0\n"
"# Operator kernel override, one key per operator (id / alias / auto) -- **OpenCL\n"
"# only**; the CUDA/CGBN backend ignores them (a note is printed if set):\n"
"#   mul = modular multiplication,     sqr = modular squaring,\n"
"#   add = modular addition,           sub = modular subtraction,\n"
"#   special-mult = special multiplication kernel.\n"
"# 算子内核覆盖，每个算子一个键（id / 别名 / auto）—— **仅 OpenCL**，\n"
"# CUDA/CGBN 后端会忽略（设了会打印提示）：\n"
"#   mul = 模乘，        sqr = 模平方，\n"
"#   add = 模加，        sub = 模减，\n"
"#   special-mult = 特殊乘法内核。\n"
"kernel_mul =\n"
"# (sqr kernel override)\n"
"# （sqr 内核覆盖）\n"
"kernel_sqr =\n"
"# (add kernel override)\n"
"# （add 内核覆盖）\n"
"kernel_add =\n"
"# (sub kernel override)\n"
"# （sub 内核覆盖）\n"
"kernel_sub =\n"
"# (special-mult kernel override)\n"
"# （special-mult 内核覆盖）\n"
"kernel_special_mult =\n";
    out.close();
    return !out.fail();
}
