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

        if (key == "worktodo") cfg.worktodo = val;
        else if (key == "finished") cfg.finished = val;
        else if (key == "save_sync_dir_1") cfg.save_sync_dir_1 = val;
        else if (key == "save_sync_dir_2") cfg.save_sync_dir_2 = val;
        else if (key == "sync_mode") cfg.sync_mode = val;
        else if (key == "log_file") cfg.log_file = val;
        else if (key == "device") set_int(cfg.device);
        else if (key == "edwards") set_int(cfg.edwards);
        else if (key == "edwards_threads") set_int(cfg.edwards_threads);
        else if (key == "edwards_naf_w") set_int(cfg.edwards_naf_w);
        else if (key == "affinity") cfg.affinity = val;
        else if (key == "cpu_affinity") cfg.affinity = val;           // 别名
        else if (key == "edwards_backend") cfg.edwards_backend = val;
        else if (key == "edbackend") cfg.edwards_backend = val;       // 别名
        else if (key == "p95_dir") cfg.p95_dir = val;                 // 已废弃, 见 tmp_dir
        else if (key == "tmp_dir") cfg.tmp_dir = val;
        else if (key == "gpuckpt_seconds") set_double(cfg.gpuckpt_seconds);
        else if (key == "verbose") set_int(cfg.verbose);
        else if (key == "tpi") set_u32(cfg.tpi);
        else if (key == "wg_size") set_int(cfg.wg_size);
        else if (key == "kernel_mul") cfg.kernel_mul = val;
        else if (key == "kernel_sqr") cfg.kernel_sqr = val;
        else if (key == "kernel_add") cfg.kernel_add = val;
        else if (key == "kernel_sub") cfg.kernel_sub = val;
        else if (key == "kernel_special_mult") cfg.kernel_special_mult = val;
        else if (key == "sigma") set_u32(cfg.sigma);
        else if (key == "save_name_pattern") cfg.save_name_pattern = val;
        else if (key == "progress_color") cfg.progress_color = val;
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
"# ECM queue-manager configuration.\n"
"# ECM 队列管理器配置。\n"
"# All relative paths are resolved against the directory of the executable.\n"
"# 所有相对路径都相对于可执行文件所在目录解析。\n"
"\n"
"# Work file: one ECMSTAGE2 line per task.\n"
"# 工作任务文件：每行一个 ECMSTAGE2 任务。\n"
"worktodo = worktodo.txt\n"
"\n"
"# Successfully completed tasks are appended here.\n"
"# 成功完成的任务追加到此文件。\n"
"finished = worktodo.finished.txt\n"
"\n"
"# .save result sync target directories (empty = disabled).\n"
"# .save 结果同步目标目录（留空 = 关闭同步）。\n"
"save_sync_dir_1 =\n"
"save_sync_dir_2 =\n"
"\n"
"# Sync mode after each task: incremental | full.\n"
"# 每个任务完成后的同步模式：incremental | full。\n"
"sync_mode = incremental\n"
"\n"
"# Log file (empty = stdout only).\n"
"# 日志文件（留空 = 仅 stdout）。\n"
"log_file = screen.log\n"
"\n"
"# GPU device index.\n"
"# GPU 设备索引。\n"
"device = 0\n"
"\n"
"# Stage-1 backend: 0 = GPU, 1 = CPU Edwards (Atkin-Morain).\n"
"# Stage-1 后端：0 = GPU，1 = CPU Edwards（Atkin-Morain）。\n"
"edwards = 0\n"
"\n"
"# Edwards stage-1 worker threads: 0 = auto (min(curves, CPU cores)), 1 = sequential.\n"
"# Edwards stage-1 并行线程数：0 = 自动（min(曲线数, CPU 核数)），1 = 顺序执行。\n"
"edwards_threads = 0\n"
"\n"
"# CPU affinity for Edwards stage-1 worker threads (亲核性).\n"
"#   (empty) / none / auto : let the OS schedule (default)\n"
"#   list of logical CPU numbers, e.g.  Affinity = 1,3,5,7\n"
"# Worker t is pinned to list[t % count]; on Windows only the first 64 logical\n"
"# CPUs can be addressed this way.\n"
"# 留空/none/auto = 交给系统调度；或写逻辑 CPU 号列表，第 t 个线程绑到 list[t % n]。\n"
"affinity = \n"
"\n"
"# Edwards stage-1 backend:\n"
"#   auto : use the SIMD batch backend when the CPU has AVX512-F/DQ/IFMA and\n"
"#          curves >= 8, otherwise the scalar GMP path (default)\n"
"#   simd : force the AVX512-IFMA batch backend (8 curves per batch, dict w=8).\n"
"#          Hard error if the CPU lacks the ISA - never falls back silently.\n"
"#   gmp  : force the scalar mpn path (1 curve per thread)\n"
"# stage-1 后端：auto = 自动（有 AVX512-IFMA 且曲线 >= 8 时用批量，否则标量）；\n"
"# simd = 强制批量（无 ISA 直接报错，不静默降级）；gmp = 强制标量。\n"
"edwards_backend = auto\n"
"\n"
"# Edwards NAF window (dictionary = 2^(w-2) entries). 0 = built-in default.\n"
"# Larger w means fewer point additions but a bigger table; w=12 measured best\n"
"# across 347..4003 bit operands, w>=16 regresses (table no longer cache-resident).\n"
"# Edwards NAF 窗口（字典 = 2^(w-2) 项）。0 = 使用内置默认值。\n"
"edwards_naf_w = 0\n"
"\n"
"# Local directory for stage-1 results: writes e{n:07d}_c{curve:06d}.tmp (MIDSTAGE,\n"
"# state=2, for stage 2) and e{n:07d}_c{curve:06d} (STAGE1 self-checkpoint).\n"
"# ecm.exe never writes into the Prime95 directory; use ecm_p95feeder to transfer.\n"
"# stage-1 结果落盘目录：写 e{n:07d}_c{curve:06d}.tmp（MIDSTAGE，state=2，供 stage-2）\n"
"# 与 e{n:07d}_c{curve:06d}（STAGE1 自检查点）。ecm.exe 不再写 Prime95 目录，\n"
"# 交接请运行独立的 ecm_p95feeder。\n"
"tmp_dir = .\n"
"\n"
"# Deprecated: ecm.exe no longer writes to the Prime95 directory (feeder's job).\n"
"# 已废弃：ecm.exe 不再写 Prime95 目录（改由 feeder 负责）。\n"
"p95_dir =\n"
"\n"
"# GPU checkpoint interval in seconds.\n"
"# GPU 检查点间隔（秒）。\n"
"gpuckpt_seconds = 600\n"
"\n"
"# Verbose level.\n"
"# 详细程度。\n"
"verbose = 1\n"
"\n"
"# Threads per instance (OpenCL).\n"
"# 每实例线程数（OpenCL）。\n"
"tpi = 8\n"
"\n"
"# Explicit work-group size (0 = auto).\n"
"# 显式工作组大小（0 = 自动）。\n"
"wg_size = 0\n"
"\n"
"# OpenCL operator kernel path overrides (CUDA ignores these).\n"
"# OpenCL 算子内核路径覆盖（CUDA 忽略）。\n"
"kernel_mul =\n"
"kernel_sqr =\n"
"kernel_add =\n"
"kernel_sub =\n"
"kernel_special_mult =\n"
"\n"
"# Fixed sigma (0 = random per run).\n"
"# 固定 sigma（0 = 每次运行随机）。\n"
"sigma = 0\n"
"\n"
"# save_name pattern used to extract B1 (e.g. m{n}_{b1}.save).\n"
"# 用于提取 B1 的 save_name 模式（例如 m{n}_{b1}.save）。\n"
"save_name_pattern = m{n}_{b1}.save\n"
"\n"
"# Progress bar color: none|red|green|yellow|blue|magenta|cyan|white|grey.\n"
"# 进度条颜色：none|red|green|yellow|blue|magenta|cyan|white|grey。\n"
"progress_color = cyan\n";
    out.close();
    return !out.fail();
}
