# CUDA param0 Stage1：驻留 ladder 与 Prime95 PRAC 实施记录

后续：[TPI、168 寄存器与管理员 Nsight 采集实测](D:/code/MPA-OpenCl/docs/ECM_STAGE1_TPI_REGISTER_TUNING_20261006.md:1)。下表保留前一轮二进制的数据，不能与新轮测试混为同次运行。

日期：2026-10-06。承接 [PRAC 可行性与成本分析](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_FEASIBILITY_20261006.md:1)。

## 1. 范围与结论

实现三个同二进制可选路径：原有 ladder、驻留 Montgomery 域的 ladder、驻留域的三点 PRAC。默认仍为原有 ladder；新路径只接受 CUDA Suyama param0、非 Mersenne-fold 构建。

生产测试使用 **B1=10,000,000 和 260,000,000**，读取进度中的预计 `s/curve`，没有等待整批完成。以下生产数据是部分运行的投影，不能等同于完整曲线墙钟实测。

8191 位 PRAC 有明确短时收益；2203 位收益较小；4423 位 PRAC 寄存器压力使其弱于驻留 ladder。已加入自然寄存器策略，消除了该档位的 spill，但仍未胜过驻留 ladder。因此不全局切换默认算法。

## 2. 入口和边界

- [ecm_backend.h:52](D:/code/MPA-OpenCl/include/ecm_backend.h:52)：后端接口显式携带整数 B1 与 torsion=1/12，不能从标量反推来源。
- [ecm_driver.cpp:2904](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2904)：驱动传递 B1 与 `--exponent lcm|choose12` 的乘数。
- [cgbn_stage1.cu:1152](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1152)：读取 `ECM_GPU_STAGE1_ALGO`；非法值、param2/3 或 fold 构建明确拒绝新路径。
- [cgbn_stage1_prac_host.cuh:48](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:48)：档位选择、计划、分片、检查点和结果处理。

新路径接受 `2 <= B1 <= UINT32_MAX`、合法 64 位 sigma 批次范围，使用覆盖 `nbits(N)+CARRY_BITS` 的现有 param0 档位。没有为 param2、param3、OpenCL 或 fold 编写 PRAC 算术。

最终 save 保持现有曲线、N、B1、sigma、仿射 x 语义。PRAC 临时 B/C 是链工作点，不代表 ladder 的最终 `[s+1]P`；结果处理只使用 A 的 X/Z。

## 3. 主机计划及缓存

### 3.1 素数幂分解

设 q=B1、t=1 或 12，`s(q,t)=t×lcm(1,...,q)`。每个 p≤q 重复执行 `[p]P` 共 `floor(log_p q)` 次；choose12 另加两个 `[2]` 和一个 `[3]`。B1=2 的 choose12 也显式包含额外的 3。

[ecm_prac_plan.cpp:69](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:69) 使用奇数分段筛，段数组 512 KiB，小筛到 `floor(sqrt(q))`。

### 3.2 Prime95 风格种子

沿用 simplified PRAC 的十个种子比例，各搜索 `ceil(p×ratio)±3`，候选去重，等成本保留第一个。采用本项目 CGBN 成本 DBL=5、DADD=6，不直接复制 Prime95 FFT 的成本常数。

源代码：[十个比例](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:33)、[整数成本递推](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:38)、[搜索](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:51)。参考 [Prime95 lucas_cost](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2645) 和 [ell_mul](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2857)。

每条记录为四个 u32：`p,d,repetitions,work`，16 bytes。GPU 根据 d/e 执行链，没有展开成巨大指令表。批内所有曲线消费同一计划，链分支在 warp 内一致。

默认最多 8 个 CPU 线程，每次领取 1024 条记录。工作线程或线程创建失败时等待已启动线程，错误传回调用者。见 [ecm_prac_plan.cpp:142](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:142)。

### 3.3 缓存

文件名 `prac_v1_b{B1}_t{torsion}_s7.bin`，默认共用 exponent cache 目录，可用 `ECM_PRAC_PLAN_CACHE` 指定目录。计划跨 N 位宽复用。

64-byte header 记录版本、B1、torsion、search、标量身份、记录数、工作量、载荷校验和。检查长度、版本、身份、范围和工作量；坏缓存重建，缓存写入失败不阻止计算。详见 [ecm_prac_plan.cpp:107](D:/code/MPA-OpenCl/src/core/ecm_prac_plan.cpp:107)。

| B1 | 素数记录 | 数组 bytes | MiB | 模乘等价工作 W | 冷准备时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10,000,000 | 664,579 | 10,633,264 | 10.14 | 128,750,918 | 0.943 s |
| 260,000,000 | 14,195,860 | 227,133,760 | 216.61 | 3,369,476,895 | 28.677 s |

冷时间来自 v1 exe 和当时 CPU 负载，缓存读取时间见日志。B1=260m 准备时 GPU0 在其他生产任务中，GPU1 单独用于本轮。

## 4. GPU 每一步

### 4.1 初始化和入域

现有 CPU `set_p_2p_suyama` 生成 `N,a24,xdiff,AX,AZ,BX,BZ` 的 7-word 曲线记录。整批坐标与控制数组各上传一次。INIT 将六个域元素转成 Montgomery 形式，N 保持普通模数。

源代码：[主机初始化及上传](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:134)、[INIT/EXPORT](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:75)。

### 4.2 驻留 ladder

保持原 MSB-first 位序、P/2P 初始状态、交换规则和融合 `double_add_v2_suyama`。片间只加载/保存 Montgomery 坐标。原路径每片六次入域、四次出域，驻留路径消除重复转换。

见 [cgbn_stage1_prac_kernel.cuh:87](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:87)。本轮未加入指数字缓存或修改融合公式；实际对照差异还包含编译布局、占用率变化。

### 4.3 PRAC 单素数乘法

1. p=2：直接 DBL(A)。
2. 奇素数：C=A、B=2A，初始化 `e=p-d0,d=d0-e`。
3. d<e 时交换 A/B 和 d/e。
4. 按 Prime95 simplified PRAC 的减法、同奇偶、d 偶数、e 偶数规则更新点和 d/e。
5. d=e=1 时，以 C 为差点做最终 DADD，结果写回 A。
6. 完成该素数所有 repetitions，再执行下一条描述。

见 [素数循环](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:106)。非减法三条规则用固定角色交换，共享 ADD+DBL 实现，避免动态索引点数组和多个展开体。见 [规则实现](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:121)。

DBL=3M+2S，DADD=4M+2S。各用三个域临时量；DADD 覆盖输出前读完两个输入和差点 X/Z，允许别名。乘方/乘法保留 normalized 包装及条件减 N。

源代码：[DBL](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:13)、[DADD](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:40)。

### 4.4 分片及计时

按前一片时间调节长度，目标约 100 ms。PRAC 只在完成一个素数的全部 repetitions 后切片，无需持久化链内 d/e、B/C。

CUDA events 计每片纯 kernel 时间；50 片窗口取 `work/ms` 的算术平均。PRAC 进度按 `5D+6A` 工作量加权，resident 按位数，输出的 s/curve 使用各自的总工作量：

`projected seconds/curve = total_work / (1000 × mean(work/ms) × curves)`。

见 [计时和投影](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:185)。`kernel=` 为累计 event 时间；`execution-wall=` 包含主机片间调度、检查点、最终出域/传输。两者均不包括前置指数/计划生成、曲线准备、首次上传/INIT，也不包括后置 CPU 仿射化/save。端到端时间需另外计量。

### 4.5 收尾

EXPORT 还原六个域元素，回传整批 7-word 数据；现有 `process_results` 用 A 的 X/Z 检查因子并仿射化，驱动写兼容 save。完成且非错误时删检查点。见 [主机收尾](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:228)。

## 5. 检查点与部分运行

独立文件后缀 `.prac-v1` / `.resident-v1`，不读取原 ladder v5。112-byte header 含算法/域、B1/t、模数和标量 hash、计划身份、档位/TPI、批量、64 位 sigma、下一条 prime/bit offset、长度和校验和。载荷保持 Montgomery 域；先写临时文件再替换。FNV 检测意外损坏，不是认证。

源代码：[格式和写入](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:16)、[恢复验证](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:117)。

`ECM_GPU_STAGE1_SAMPLE_SECONDS>0` 使新路径在片边界到时后写检查点并返回不完整状态，不发布完整 Stage1 Q。原路径由采样工具终止其独立子进程。驱动可能预创建空 save，必须按 SIGMA/X 记录数判断完成，不能按文件存在判断。

源代码：[停止点](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:217)、[bench_cuda_prac.py](D:/code/MPA-OpenCl/tools/bench/bench_cuda_prac.py:1)。

## 6. 工作量、内存、显存和传输

约定 n=nbits(N)，K=覆盖 n+6 的档位，L=K/8 bytes，C=曲线数，P=π(B1)，b=bitlength(s)，J=片数，H=检查点写入次数。

### 6.1 算术

- ladder 每曲线名义 `10(b−1)` 模乘等价单位，另计初始化和转换。
- PRAC 每曲线 `W=Σ_p repetitions_p×(5D_p+6A_p)`，全批 C×W。
- 当前平方调用一般模乘，S=M。W 不包含加减、normalize、交换、控制、CGBN 通信或 spill。
- 固定位宽时可写 `T_kernel≈C×W/R(n,K,TPI,C,policy)`，R 必须测量；没有设备指令计数，不能给出确定周期数。

B1=10m 的 lcm 标量 b=14,424,844，PRAC 名义算术相对 `10(b−1)` 减少约 10.74%。

### 6.2 显式设备分配

PRAC 曲线与控制数组 `7CL+16P` bytes；resident 为 `7CL+4ceil(b/32)`。还存在错误报告、events、CUDA 上下文及可能的 local-memory backing，这不是进程显存总峰值。

| N bits | K/TPI | C=1536 的曲线数组 |
| ---: | --- | ---: |
| 2203 | 2560/16 | 3.28125 MiB |
| 4423 | 4608/16 | 5.90625 MiB |
| 8191 | 9216/32 | 11.8125 MiB |

PRAC 控制数组与 N/C 无关，本轮为 10.14 / 216.61 MiB。大 B1 下比 ladder 的指数 bitstream 更大，收益来自链算术和域驻留。

### 6.3 主机和磁盘

计划有效载荷 16P bytes，冷生成 vector 容量可能大于 P，扩容时旧、新数组短暂共存。初始化短暂持有两份 7CL 主机数组，随后保留一份。PRAC 仍持有驱动生成的 GMP 标量（约 b/8），尚未去除指数构造。筛附加约 `512 KiB+O(sqrt(B1))`。

计划缓存 `64+16P` bytes，检查点 `112+7CL` bytes。

### 6.4 数据移动

- PRAC 初始 H2D `7CL+16P`，最终 D2H `7CL`，每个检查点另 D2H `7CL`；显式 PCIe 数据合计 `14CL+16P+7HCL`。
- resident 将 16P 换成 `4ceil(b/32)`。
- 无检查点时，片间不回传曲线数组，仍有 event 同步、错误报告检查和日志。
- 曲线 global 读写按代码估计：PRAC 每片读 N/a24/A、写 A，约 6CL；resident 约 11CL。全程约 `6JCL` / `11JCL`，另加 INIT/EXPORT、控制读取、spill。实际缓存命中和 DRAM 流量需 profiler 测量。

## 7. 寄存器和 spill

sm89、CUDA 13.3。ptxas spill bytes 是编译统计，不是一次迭代确定的运行流量；48-byte stack 本身不等于存在 spill。

| K/TPI | 内核 | registers/thread | stack bytes | spill stores / loads bytes |
| --- | --- | ---: | ---: | ---: |
| 2560/16 | resident | 84 | 48 | 0 / 0 |
| 2560/16 | PRAC per-tier | 106 | 48 | 0 / 0 |
| 2560/16 | PRAC natural | 105 | 48 | 0 / 0 |
| 4608/16 | resident | 128 | 56 | 0 / 0 |
| 4608/16 | PRAC per-tier | 128 | 264 | 636 / 1520 |
| 4608/16 | PRAC natural | 172 | 48 | 0 / 0 |
| 9216/32 | resident | 126 | 48 | 0 / 0 |
| 9216/32 | PRAC per-tier | 163 | 48 | 0 / 0 |

`ECM_PRAC_REG_TARGET=255` 对中档宽度编译独立内核，在同一 exe 中 A/B。小档及现有自然寄存器的大档复用原 PRAC 实例。255 是上限，实际数由 ptxas 决定。日志报告所选 kernel 的实际 blocks/SM。

源代码：[maxnreg](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:63)、[实例分派](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:143)、[策略选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:55)。

## 8. 性能证据

GPU1=RTX 4060 Laptop、24 SM，GPU0 有其他任务。C=1536，sigma=26，param0，lcm，B2=0。新路径采样约 15 s，旧路径约 17 s 后终止；忽略前 5 s，取最后 5 s 的中位数。未锁频，小差值需重复确认。

### 8.1 首轮 per-tier

| B1 | N bits | 原 ladder s/curve | resident s/curve | PRAC s/curve | PRAC 相对 resident 耗时变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 2203 | 2.030 | 1.979 | 1.950 | −1.46% |
| 10m | 4423 | 5.640 | 5.365 | 6.443 | +20.09% |
| 10m | 8191 | 21.230 | 20.535 | 18.707 | −8.90% |
| 260m | 2203 | 52.650 | 51.465 | 51.263 | −0.39% |
| 260m | 4423 | 146.610 | 139.579 | 172.597 | +23.66% |
| 260m | 8191 | 552.140 | 533.995 | 489.761 | −8.28% |

全部是 **预计 s/curve**。完成比例不到 1%，260m 常规进度甚至显示 0.0%。PRAC 按总工作加权，但速率来自前部素数；后期链分支、交换和每片工作量可能改变效率。需后部素数采样或完整曲线进一步确认。

### 8.2 4423 位自然寄存器复测

| B1 | resident 两次 s/curve | natural PRAC 两次 s/curve | 按两次均值的 PRAC 耗时变化 |
| ---: | --- | --- | ---: |
| 10m | 5.365679 / 5.365995 | 5.613561 / 5.614201 | +4.62% |
| 260m | 139.535364 / 139.490109 | 146.951939 / 146.822837 | +5.29% |

自然策略消除 spill，将 PRAC 相对 per-tier 的预计耗时降低约 12.9% / 14.9%，但高寄存器减少驻留块数，仍不如 resident。该结果支持优先改善活跃区间/占用率，不支持继续扩大链搜索或强制全局 PRAC。

### 8.3 数据及版本

原始数据保留在按要求排除的 `docs/data/`：

- `prac_cuda_perf_20261006_v1/summary.json`：10m 首轮。
- `prac_cuda_perf_20261006_v1_b260m/summary.json`：260m 首轮。
- `prac_cuda_perf_20261006_natural/summary.json`：natural 重复 A/B。
- `prac_cuda_q_gate_20261006_v1/summary.json`：首轮 Q 门禁。
- `prac_cuda_q_gate_20261006_v2_natural/summary.json`：扩展 natural 门禁。
- `prac_cuda_q_gate_20261006_v4_default/summary.json`、`prac_cuda_q_gate_20261006_v4_natural/summary.json`：最终两种策略验收，含冷计划和坏缓存重建。

v1 保存为 `build_cuda_cmake/prac/v1/ecm_cuda.exe`；后续使用 `build_cuda_cmake/prac/ecm_cuda.exe`。采样 summary 有二进制 SHA256。首轮曾将预创建空 save 误标完成，已仅纠正完成元数据，保留 `summary.original.json`，速率未改。

版本 SHA256：

- v1/per-tier 性能：`9dc43bf04fe23ffb8f97ed9cc7349cbf5f215bf3d407a73a9cd22f7896526e21`。
- v2/natural 性能：`2795e7ff8d673716f13c547985583cb04777d1cf0efa0ee015b31cb076a2d270`。
- 最终交付 exe / v4 门禁：`57ef2e2dd83f7bbd69cd883eb1973c48cd73e205352e0c7d221fe0e40cd16ebc`。

v2 到最终 exe 只修改主机结果处理和 sigma 输出，没有修改被测 GPU kernel；v2 原路径不再位于当前 exe 路径，不能仅凭该路径重现旧二进制身份。

## 9. 正确性验收

[test_cuda_prac.py](D:/code/MPA-OpenCl/tools/test/test_cuda_prac.py:1) 使用同一 exe 中的 CPU Montgomery/GMP 作为独立域后端。逐条比较 SIGMA、X、N、B1、PARAM、CHECKSUM，忽略 WHO/TIME。

首轮通过 256 条 GPU 完整 Q 比较。最终 per-tier/natural 各通过 376 条 Q/因子 save 比较，共 **752 条比较**，另有 6 个退化结果拒绝检查。覆盖三个主力宽度、lcm/choose12、53/62 位 sigma、通用合数、B1=2/3/5 边界、原生冷计划、坏计划缓存重建、检查点续跑与损坏回退。实际执行均在 GPU1。

因子命中门禁使用 `N=1009×(2^127−1)`、B1=1000、8 曲线、62 位 sigma，逐条检查 CPU/GPU save 及因子输出的完整 sigma。额外退化案例 `N=1009×1000003` 的部分曲线得到 `gcd(Z,N)=N`。这可能是合法的全模数湮灭，不能把 N 当作非平凡因子，也没有可逆分母来生成仿射 save。

该案例复现两处旧主机问题：结果处理和驱动因子日志截断 sigma，以及把整个 N 接受为因子。修正后，[findfactor:304](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:304) 只接受严格介于 1 和 N 的 gcd；[process_results:782](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:782) 遇到退化结果使整个批次返回错误，不写最终 save；[驱动 sigma:2969](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2969) 保留后端返回的完整 sigma；[CLI 错误批次:4197](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:4197) 不发布因子记录。这个主机修正也适用于原有 CUDA ladder。

先用正式 CLI 门禁复现截断，再修复并对原案例回归。独立新目录排除了检查点覆盖 sigma 的假设；CPU、旧 ladder 与新 PRAC 都能出现退化分母，排除了将该现象仅归因于 PRAC 算术的解释。没有保留临时调试日志代码。

未覆盖全部链中退化点、全部因子命中情形或全部档位；未测完整生产批次耗时。当前退化处理是批次报错，没有实现单曲线回溯、缩小 B1 或在同批次中保留其他有效 save。

## 10. 构建和使用

### 10.1 独立构建

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage1_local.ps1 `
  -BuildDir build_cuda_cmake/prac -Arch 89 -Jobs 6 `
  -Tiers '512,1024,2560,4608,9216' -Extra '-DECM_NO_PARAM2=1'
```

PRAC 按 small/TPI16/TPI32 拆 TU，由 [parallel_nvcc.ps1](D:/code/MPA-OpenCl/tools/build/internal/parallel_nvcc.ps1:1) 并行编译，C++/CUDA17。首轮 CUDA 编译关键路径 207.8 s；增加 natural 实例后增量关键路径 371.2 s，另加 host/link。

当前实验 exe 仅含这五档；源码保留其他现有档位分派。本轮没有覆盖安装目录生产 exe。

### 10.2 选择算法

```powershell
$env:ECM_GPU_STAGE1_ALGO = 'prac'   # ladder / resident / prac
$env:ECM_PRAC_REG_TARGET = '255'   # 0: per-tier；255: natural；168: 仅 4608/TPI16
'(2^4423-1)' | .\build_cuda_cmake\prac\ecm_cuda.exe `
  -gpu -d 1 --gpu-param 0 -sigma 0:26 -gpucurves 1536 `
  --exponent lcm -savea completed.save 10000000 0
```

完整计算清除 `ECM_GPU_STAGE1_SAMPLE_SECONDS` 或设为 0；检查点间隔按现有驱动配置。PRAC 线程数可用 `ECM_PRAC_THREADS=1..32`，计划目录可用 `ECM_PRAC_PLAN_CACHE`。

### 10.3 采样和门禁

```powershell
python tools/bench/bench_cuda_prac.py --device 1 `
  --bits 2203 4423 8191 --b1 10000000 260000000 `
  --seconds 15 --warmup 5 --output docs/data/my_prac_sample

python tools/bench/bench_cuda_prac.py --device 1 --bits 4423 `
  --b1 10000000 260000000 --algorithms resident prac `
  --prac-registers 255 --repeats 2 --seconds 15 --warmup 5 `
  --output docs/data/my_prac_natural_sample

python tools/test/test_cuda_prac.py --device 1 --output docs/data/my_prac_gate
```

使用新目录，避免旧 save/检查点污染；生产 B1 不用于完整 Q 门禁。

## 11. 下一轮优先级

1. 4423 位：压缩点操作临时量生命周期，比较 160/168 寄存器预算或 TPI 变体。无 spill 不等于更快。
2. 后部素数微基准或等价状态采样，检验模乘等价速率是否随 p 变化。
3. 批量及准备开销：比较不足/充分填充；进一步去除 P/2P/xdiff 和 GMP 标量构造需新接口及身份校验设计。
4. Auto B2：将算法、构建、寄存器策略、批量、torsion 纳入 profile 身份后再更新 T1，不自动导入本轮部分运行投影。
5. Lucas 扩展后置；此前离线额外收益小，当前主要限制已是寄存器压力。

默认算法变更需要更多实证；本轮已提供同二进制对照、独立检查点和短时采样能力。

后续窗口校准和 compact DBL 候选记录：[PRAC 窗口与算术实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_WINDOWS_COMPACT_20261006.md:1)。本文性能数字保留原二进制与采样范围，源码链接更新到当前对应入口。
