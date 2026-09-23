# tools/ 目录索引

本目录按**用途**分子目录。历史上所有脚本都堆在 `tools/` 根下，现按下面分类整理；
根目录不再放散落脚本。

```
tools/
├── gen/        内核/参数代码生成器（Python，单源 → .cl/.py 产物）
├── refactor/   一次性迁移/重构/校验脚本（Python，历史工程脚本）
├── bench/      基准与 A/B 脚本（.cpp/.c/.ps1）
├── test/       单元测试 + 集成测试 + 夹具
├── disasm/     反汇编 / ISA 检查（含 Windows 工具链安装）
├── ecm_prob/   ECM 参数化概率分析套件（Python，自带 README）
├── ecm_report/ 进度数据库/图表（Python + bat，自带 README）
└── log_parser/ 日志解析（自带 README）
```

## test/ — 单元测试与集成测试

| 文件 | 内容 |
|---|---|
| `p95_worktodo_test.cpp` | Prime95 worktodo/prime.txt 解析与写回（35 项断言） |
| `ecm_worktodo_test.cpp` | `ECM=`/`ECM2=`/`ECMSTAGE2=` 行解析与 N 计算 |
| `ecm_edwards_save_test.cpp` | Prime95 ECM_VERSION=6 存档读写；与真实 `e0000347` 字节级比对 |
| `ecm_edwards_checkpoint_test.cpp` | 分块标量乘的中止/恢复等价性（Qx/Qz 一致） |
| `gen_ckpt.cpp` | 生成一个中途 STAGE1 存档，用于验证驱动恢复 |
| `test_feeder.ps1` | `ecm_p95feeder` 端到端集成测试（沙箱 p95 目录，7 个周期） |
| `fixtures/` | 测试派生文件（`_test_*.save`，由 `ecm_edwards_save_test` 写出） |

手工编译示例（GPE/zen3 前缀按需替换）：

```powershell
cl /nologo /O2 /EHsc /utf-8 /I third_party/gmp-zen3/dist/include /I src/cpu /I src/core ^
   tools/test/ecm_edwards_save_test.cpp src/cpu/ecm_edwards_cpu.cpp src/cpu/ecm_edwards_save.cpp ^
   /Fe:tools/test/ecm_edwards_save_test.exe /link third_party/gmp-zen3/dist/lib/gmp.lib
```

```powershell
# 纯 GMP 无关的单测
cl /nologo /O2 /EHsc /utf-8 /I src/core ^
   tools/test/p95_worktodo_test.cpp src/core/p95_worktodo.cpp /Fe:tools/test/p95_worktodo_test.exe
```

## bench/ — 基准与 A/B

| 文件 | 内容 |
|---|---|
| `ecm_edwards_speed.cpp` | Edwards stage-1 单曲线计时（bits/B1/w/reps 参数化） |
| `ecm_edwards_bench.cpp` | N × NAF 窗口扫描 + 字典内存统计 |
| `gmp_mpn_microbench.c` | `mpn_addmul_1`/`mpn_mul_n`/`mpn_sqr` 的 cycles/limb（内联汇编标定主频） |
| `mont_probe.cpp`、`gmp_limb_probe.c` | Montgomery/limb 行为探针 |
| `bench_threads.ps1` | 多曲线并行线程数扫描（吞吐） |
| `build_speed.ps1`、`sweep_flags.ps1` | MSVC 编译标签扫描（单标签 / 批量） |
| `abtest_gmp.ps1` | 两套 GMP（通用 vs zen3/BMI2）交替 A/B |
| `bench_wg_scaling.ps1` | OpenCL work-group 规模扫描 |

> 本机（笔记本）持续满载会降频，**<10% 的差异必须交替 A/B 测量**才可信；
> 单配置连跑取 min 会把降频趋势当成配置差异。见 `docs/ECM_EDWARDS_STAGE1.md` §10。

## disasm/ — 反汇编 / ISA

`DISASM_SETUP.md`（Windows 工具链说明）、`install_disasm_tools.{bat,ps1}`、
`verify_disasm_tools.{bat,ps1}`、`disasm_mont_isa.ps1`、`disasm_addsub_isa.ps1`。

## gen/ — 代码生成器

内核 `.cl` 与参数文件的单源生成器；产物在 `kernels/` 下。
`gen_all.py` 批量跑 mp_addsub 一组生成器；`mp_asm_block_gen.py` 是共享库
（被多个生成器 `import`，因此与它们同目录）。

生成器一律以 `Path(__file__).resolve().parents[N]` 定位仓库根（`tools/gen/*.py` 为
`parents[2]`），或以命令行参数/当前目录为输出根；**不要再把它们移出本目录而不改深度**。

## refactor/ — 一次性工程脚本

`migrate_ecm_stage1_kernel_layout.py`、`migrate_operators_v2.py`、
`refactor_ecm_stage1_macro_iface.py`、`split_ecm_stage1_kernel_tree.py`、
`patch_npu_addsub_mod.py`、`validate_stage1.py`、`_trace_4x2.py`。
保留供追溯/重跑，日常开发不需要。

## ecm_prob/ · ecm_report/ · log_parser/

各自独立，用法见其目录内 `README.md`：

- `ecm_prob/` — 参数化概率/`D_eff` 实测（含 `FindGroupOrder3_example.gp`）
- `ecm_report/` — 进度数据库与图表（`import_ecm.py` / `download_ecm.py`）
- `log_parser/` — `screen.log` 等日志解析
