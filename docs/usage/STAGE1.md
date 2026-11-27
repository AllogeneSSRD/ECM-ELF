# Stage1 使用

<a id="pm1-stage1"></a>

## P−1 分解

在 CUDA worker 的 worktodo 中加入显式上界任务，运行 `ecm_cuda --config ecm.toml`。例如：

```text
Pminus1=1,2,400381,-1,500,1000
```

完整语法为 `Pminus1=[AID,]1,2,p,-1,B1,B2[,how_far_factored][,B2_start][,"factors"]`。p 取 32 至 5,242,880 范围内的素数，B1 取 2 至 5,000,000,000。CUDA 加权 NTT 在完整 M=2ᵖ−1 上计算 3 的 2p·lcm(1,…,B1) 次幂。CPU GMP 对 x−1 和剥离原有已知因子后的余因子求 GCD。完成记录位于 worker 的 finished 文件，包括本轮找到的真因子或 `gcd=target`。

连续的 `Pminus1` 可按相同 NTT 长度并发执行。用 `ecm_cuda -tune "pm1,exp=400381,batch=1-4"` 测量本机的批量和执行形状。批次选择、计时口径和矩阵字段见 [P−1 Stage1 调优](TUNE.md#pm1-stage1-tune)。

批内每条任务完成后即可写入 `finished` 并尝试交接。同批其他任务仍在计算或因安全停止留下断点时，已完成的任务仍会提交。交接受阻的任务各自保留回执，重启后重试。

运行中，`pm1_stage1_progress` 报告整批已验证的 iter、总 iter、百分比、本次运行的 iter/s 和 ETA。每条任务的开始、断点、结果及交接另有日志。续算和多任务 ETA 的口径见 [P−1 进度显示](GUI_V2.md#pm1-progress)。

**存档。** 每个完成的任务都生成原生 Prime95 `m…` 端点，新写文件使用 v8。默认交接目录为 ecm.toml 同目录下的 `pm1/M<p>-B1-<B1>/`。从头执行时，`p1-<p>/` 存放 GpuOwl 断点，可与参考版 PRPLL（`6dd6dd5`）在相同 p、B1 下双向续算。定期保存和停止时使用通过 Gerbicz 校验的块末余数。

**调大 B1。** 直接增大 worktodo 原行的 B1，保留其余字段。worker 优先恢复目标 B1 的进度；有较小 B1 的未完成本地断点时，先完成断点记录的指数，再扩展到目标 B1。运行中修改同样生效：写入完成记录前，worker 重新读取所属队列，能唯一识别这次增大时保留旧端点，仅按新 B1 记录完成和交接。若在交接等待期间增大 B1，已写入的旧 B1 完成记录保留，仍待投递的旧任务由新 B1 取代。

没有未完成进度时，worker 从同一 p 的可用完成端点中选择不超过新 B1 的最大值。可把 Prime95 的 `m` 文件放在配置文件旁、任务目录或 `prime95_save_dir`，以往较小 B1 任务目录中的端点也可复用。导入支持 v7/v8 中保留 Stage1 余数的 MIDSTAGE 和 DONE 存档。写回同名文件前复核原文件指纹；损坏、不兼容或 B1 更高的目标文件会保留并报错。

记 `L(B)=lcm(1,…,B)`。扩展从旧余数 x 计算 `x^(L(新B1)/L(旧B1)) mod M`。例如从 7 增到 9，跨过 8 和 9，增量指数 K=2×3。区间没有新素数幂时直接复用余数。扩展中的 `extension.pm1` 保存旧 B1、底数、K 的二进制游标及已验证余数，可中断后续算，也可再次增大 B1。若需要先完成旧 B1，`extension.progress` 记录启动时旧阶段剩余的迭代数，供重启后的进度显示使用。完成后仍用原生 `m` 文件交接。

**Stage 2 交接。** Pminus1 使用 `[stage1]` 下的 `handoff_mode_pm1`，默认 `"prime95"`，即使通用 `handoff_mode` 为 `"off"` 也会按此设置交接。可在 `[worker.N]` 覆盖同名键。设为 `"off"` 时仍写 Prime95 存档，所有 Pminus1 任务均跳过 worktodo 发布。

使用 `"prime95"` 时，无真因子即发布 `worktodo.add`；有真因子时仅在产生存档的 worker 设置 `[stage2].stage2_after_factored=1` 时发布。交接保留原 AID、试除深度、B2_start 和已知因子列表，本轮发现的因子仅记入本机结果。任务 B2 非零时原样交接，为零时输出 B1 × `b2_ratio`，比例也为零则交接 B2=0，Prime95 自行选界。参考版 GpuOwl 将 B2=0 解释为仅运行 Stage 1。

要直接交给 Prime95，可在 `[stage1]` 指定 `prime95_save_dir` 和 `prime95_worktodo_file` 指向接收程序使用的存档目录与工作文件。投递时按接收队列的文件名生成对应的 `.add` 并按需创建父目录。接收端若要在 Stage 2 末尾重新发现 Stage 1 因子，可设置 `Stage1GCD=-1`。使用上述小界示例测试 Prime95 时设置 `AllowLowB1=1`。ECM-ELF 在存档和任务行发布后结束该任务。

worker 写出 Stage1 存档和 `finished` 记录后，在同一个 `.add` 末尾追加交接行。已有内容保留，重试时同一行只写一次，接收端可以从同步目录复制文件。接收流程直接使用 Prime95 的活动目录时，须协调导入和写入的时间，避免 Prime95 导入后删除本轮刚更新的文件。写入失败时保留完成回执和原任务行，重启后继续投递。旧版遗留的 P−1 暂存回执在重试时迁移到 `.add`。

## 程序与输入

正常计算/队列启动输出公共程序名和版本；`--version` 只输出版本后退出，不启动计算。`--gpu-info` 等机器查询保持原输出结构。

`-h`、`-help`、`--help` 和 `/?` 显示同一份帮助，按曲线与任务、队列与交接、GPU、CPU/NAF、存档及诊断排列。曲线构造统一使用 `--param e|m0|m2|m3`。

`ecm_cuda.exe` 按实际模数位宽和后端能力选择 CUDA NTT 或 CGBN。`ecm.exe` 提供 CPU 和 OpenCL 路径。CPU 支持 Montgomery/Suyama 与 Edwards，OpenCL 的设备和算子能力按实际构建确定。选项以程序帮助及 [TOML 参考](../ECM_TOML_REFERENCE.zh-CN.md) 为准。

历史覆盖、计时和建表诊断见 [Stage1 环境变量](ENVIRONMENT.md#stage1)。

CPU Stage1 的实际模数超过 10240 位时，`cpu_implementation = "auto"` 使用 IFMA，单曲线也适用。无可用 IFMA 时任务报错。Edwards 可用 `--impl scalar` 或在 `[stage1]` 设置 `cpu_implementation = "scalar"`，显式运行宽位 GMP `mpz` 二进制点乘。Montgomery CPU 标量的上限为 10240 位。

也可用 `cpu_implementation = "simd[:6144], scalar[:4096], gwnum[4096:], auto[]"`。规则按顺序检查剥除已知因子后的实际模数位宽和后端能力。区间左闭右开，省略边界表示最小值或无上限。`[]` 等价于 `[:]`，空字符串等价于 `auto`。

全局默认值与当前 worker 的覆盖合并后，按最终 `target` 应用运行选项。GPU 任务保留配置中的 CPU 实现、归约及点乘链预设，CPU 任务保留 GPU 算法预设，切换目标时可复用这些配置。TOML 仍校验当前 worker 的选项值。为 GPU 任务显式指定非默认 `--impl`、`--reduction` 或 `--cpu-mont-chain` 会报错。CPU 任务拒绝显式指定的 `--gpu-algorithm` 和非自动的 `--gpu-backend`。

Windows x64 worker 可通过 `--impl gwnum` 或 `[stage1]` 的 `cpu_implementation = "gwnum"` 为 CPU Montgomery PARAM0 加载同目录的 `gwnum.dll`。该路径沿用现有批次存档和结果格式。见 [构建与运行](BUILD.md#cpu-stage1-本地-gwnum-316)。

CPU Montgomery PARAM0 的 IFMA 路径默认使用二进制 ladder。可添加 `--cpu-mont-chain prac`，或在 `[stage1]` 设置 `cpu_montgomery_chain = "prac"`、`target = "cpu"`、`param = "m0"` 和 `cpu_implementation = "simd"`。`cpu_prac_search` 可设为 1～50，默认 7。相同 B1、标量倍数和搜索宽度共用 `exponent_cache_dir` 中的计划。末点分母与模数不互素时用 ladder 复核。构造遇到不可逆分母或奇异曲线时直接走 ladder。续跑检查点时沿用原点乘方式和搜索宽度。

## GPU 算法选择

默认配置为：

```toml
[stage1]
gpu_algorithm = "ntt[16384:], cgbn[]"
```

每条任务先构造实际模数，再从左到右选第一个位宽区间及后端能力都匹配的规则。位宽按剥除已知因子后的模数计算，区间左闭右开。`[]` 等价于 `[:]`，空边界表示不限该侧。默认从 16384 位起优先尝试 NTT，适用曲线为 Edwards 和 Montgomery PARAM0。NTT 可计算完整 `2^p−1`，也可为保留原始梅森元数据的余因子任务使用梅森承载。其他不符合条件的输入继续尝试 CGBN。CUDA CGBN 还要求当前构建含有适用的内核档。OpenCL worker 跳过 CUDA 专用的 NTT 和 thread，按原 OpenCL 路径处理 CGBN 规则命中的任务。

`[stage1] ntt_allow_known_factors = 1` 默认允许带已知因子的梅森任务。GPU 按完整梅森数选择 NTT 长度并计算，构造、求逆、因子检查、save 和结果发布使用余因子。0 禁用这条承载路径，自动规则继续尝试后续后端，强制 `ntt` 则报错。规则仍按余因子的实际位宽匹配。例如 M15919 剥除 1782929 后为 15899 位，`"ntt[14336:], cgbn[]"` 可以选用 15919 位承载。原有 CGBN 检查点继续恢复其原后端。见 [梅森承载原理](../architecture/NTT.md#stage1-已知因子与梅森承载)。

若要固定算法，可写单值 `"cgbn"`、`"ntt"`、`"thread"`（也可用 `1`、`2`、`3`）。算法名不区分大小写。单值按指定算法校验，资格不符时报错。CLI `--gpu-algorithm` 可以覆盖单值或规则列表。使用列表恢复任务时，匹配的检查点锁定原来的 GPU 算法。NTT 的调优选择只影响当前后端的计划和批量。

## 进度前的执行详情

支持标量速度输出的 CUDA Stage1 进度行附带 `iter_per_s`，表示每条曲线的标量推进速度，不乘以批量。CGBN ladder 为 bit/s，NAF 为列/s；PRAC 的加权 M-equiv 不作为 bit/s。NTT 的执行详情附带 `B1` 和 `iter_total`，进度行按当前 bits/columns 输出 `iter_per_s`，暂停恢复以保存游标为起点，不把已保存进度算作新完成工作；周期检查点写入不进入速度采样间隔。

Stage1 在控制台和启用的文件日志中用 `Stage1 execution:` 行记录当前计算采用的曲线、参数化和后端。后端完成选型后再输出，包含恢复检查点、资源限制和调优结果生效后的实际值。

- CGBN 输出内核容器位宽 `kernel_weight`、每个大整数的线程数 `tpi` 和适用的启动形状。Edwards 输出实际 NAF 窗口，Montgomery 输出所用点乘链，PRAC 路径同时输出实际搜索宽度 `prac`。
- NTT 输出本批模数位宽 `ntt_bits`、卷积类型 `ntt_type`、变换长度 `ntt_length`、平均 digit 位宽 `ntt_bpw_avg`、最大 digit 位宽 `ntt_bpw_max`、系数上界 `ntt_u` 和 Goldilocks 域模数 `ntt_q`。旧字段 `ntt_bpw` 等于 `ntt_bpw_max`。`ntt_shape` 与调优输出共用五段形状编码。`ntt_graph_bits` 和 `ntt_warp_tail` 分别记录 Graph 步数上限和 warp 尾处理状态。调试行 `stage1_ntt_plan` 另列 outer 上限。Edwards 的 `naf=0` 表示 x-only。[NTT 形状编码](TUNE.md#ntt-形状编码)
- CPU 输出实际选中的实现和线程数，以及适用的 NAF 窗口或 PRAC 搜索宽度。OpenCL 输出实际内核形状和工作组；由驱动自行决定的工作组显示为 `auto`。

GUI v2 从 Worker 输出读取这些数据，详情里的预计交接另按 Worker 的有效配置显示。见 [GUI v2 Worker 详情](GUI_V2.md)。

### CUDA CGBN Montgomery PARAM0

CUDA Stage1 本机和发布构建都包含 PARAM0 PRAC。默认点乘链为 ladder，可在 TOML 中为 PARAM0 启用 PRAC，例如：

```toml
[stage1]
target = "gpu"
stage1_gpu_backend = "cuda"
gpu_algorithm = "cgbn"
param = "m0"
cuda_montgomery_chain = "prac"
cuda_prac_search = 50
```

`cuda_prac_search` 可设为 1～50，默认 7。PRAC 计划支持整数 B1 2～4,294,967,295；设备控制表占用随 B1 增长。冷启动会生成计划，相同 B1、LCM 模式及搜索宽度可以复用缓存。两个键均可在 `[worker.N]` 下用相同的键名覆盖。使用 `m0_` 存档前缀的队列任务同样适用。混合队列中的 Edwards、PARAM2/3 及 NTT 任务各自使用其原点乘路径。省略 `cuda_montgomery_chain` 时，点乘链仍可由 `ECM_GPU_STAGE1_ALGO` 选择。

Stage1 save、结果和 Stage2 交接沿用 PARAM0 格式。正在运行的批次应以原点乘链完成；PRAC 恢复还需匹配原搜索宽度及计划。程序拒绝用不匹配的设置恢复检查点。完成当前批次后，可自由为下一批切换点乘链。见 [Stage1 实测](../performance/STAGE1.md#0637-优化前基线)。

验证入口为 `tools/test/test_cuda_prac_queue.py`，指定 `--exe`、`--device` 和一个新的 `--output` 目录；可通过 `--stage2` 同时验证 Stage2 消费。0.63.8 的 PRAC 接入在两张 sm89 卡验证了 TOML 优先级、混合曲线队列、断点恢复、命中因子及 Stage2 交接。CUDA 11.8 的 384 位发布样本通过 compute_50 PTX 在 4070 Ti 上的 Stage1 验证。原始验收日志保留于 `data/experiments/stage1_prac_production_0640/`。

### 预计算 PRAC 种子

用 `ECM_BUILD_TOOLS` 构建 `prepare_prac_cache`。它按 $B_1$、LCM 模式和搜索宽度生成计划，写入每个素数的赢家种子和两组 D/A/W。缺省宽度为 7，可显式选择 1～50。以下的 `1` 表示 LCM 模式，`6` 是规划线程上限。

```powershell
$build = 'build_dev/opencl_cpu_stage1_prac_06215'
cmake --build $build --config Release --target prepare_prac_cache
& "$build/prepare_prac_cache.exe" --search 50 cache 1 6 50000 100000 1000000 11000000 44000000 260000000
& "$build/prepare_prac_cache.exe" --summary cache/prac_v2_b260000000_t1_s50.bin
```

使用预计算计划时，将 `cpu_prac_search` 或 `cuda_prac_search` 设为建表时的宽度。各宽度的文件可共存于同一缓存目录。v1 的 7 档文件可在加载时迁移，原文件可供旧版本使用。

`exponent_cache_dir` 留空时，Stage1 从工作目录及可执行文件目录向上查找 ECM-ELF 仓库，使用仓库根的 `cache/`。独立部署使用可执行文件同级的 `cache/`。可用 `--exp-cache <目录>` 指定，CUDA PRAC 还可用 `ECM_PRAC_PLAN_CACHE` 指定计划目录。文件大小、读取方式和迁移身份见 [PRAC 计划存储](../architecture/PRAC.md#计划存储与复用)。

### OpenCL Edwards

桌面 `ecm.exe` 可用 OpenCL 在奇数模数上批量运行 Edwards，最高 1024 bit。通过 `param = "e"`、`target = "gpu"` 和 `stage1_gpu_backend = "opencl"` 选择后端。直接输入示例使用 OpenCL 编号 1：

```powershell
'(2^521-1)' | .\ecm.exe --target gpu --param e --gpu-backend opencl -d 1 --naf-w 6 -sigma 20260922 -gpucurves 256 -savea ed.save 1000 0
```

`edwards_naf_window_bits` 接受 2～12，自动从 6 开始。`edwards_doubling = 1` 用 2XY，`0` 用 HWCD。指定 `naf_memory_mb` 可压低设备预算。`opencl_workgroup_size` 为 0 时交由运行时决定。Edwards 每工作项处理一条曲线，`opencl_threads_per_integer` 作用于 OpenCL PARAM3。最终 Edwards 文本 save 可交给现有 Stage2。断点续跑沿用检查点中的窗口和倍点公式。 [设备算术与内存](../architecture/OPENCL.md#stage1-曲线执行)

OpenCL 和 CUDA 的一线程一曲线路径共用 `[stage1] gpu_init_threads`。设为 0 时，从 1024 条曲线起用最多 4 个主机线程构造曲线和打包初始坐标，更小的批次用 1 个。设为 1 可固定单线程，日志中的 `construction_threads` 显示实际值。

Edwards 默认使用 CIOS。模数不超过 512 bit 时，可在 `[stage1]` 为 `opencl_kernel_mul` 和 `opencl_kernel_sqr` 选便携 `"fips"` 或 NVIDIA 的 `"fips_ptx"`。例如，让 RTX 4060 同时使用 PTX 模乘和对称平方：

```toml
[stage1]
opencl_kernel_mul = "fips_ptx"
opencl_kernel_sqr = "fips_ptx"
```

平方设为 `"auto"` 时复用模乘，显式 `"cios"` 可固定 CIOS 平方。也可保留 CIOS 模乘，只把平方设为 `"fips"`。一个任务使用同一种 FIPS 实现。CLI 的 `--mul` 和 `--sqr` 可覆盖这两个键。恢复检查点时可以切换算术，日志用 `field_mul` 和 `field_sqr` 标出实际选择。 [4060 Laptop 实测](../performance/STAGE1.md#opencl-edwards-fips)

### CUDA Edwards 一线程一曲线

用 `ecm_cuda.exe` 在 GPU 上运行，支持最高 1024 bit 的奇数模数。以下示例只运行 Stage1：

```powershell
'(2^521-1)' | .\ecm_cuda.exe --target gpu --param e --gpu-algorithm thread -d 1 --wg 128 --naf-w 6 -sigma 20260922 -gpucurves 6144 -savea ed.save 1000 0
```

也可在 `[stage1]` 设置 `param = "e"`、`target = "gpu"`、`stage1_gpu_backend = "cuda"`、`gpu_algorithm = "thread"`（或 `3`）。`thread_block_size` 可取 32、64、128 或 256，默认 128，`--wg` 可覆盖。512 bit 以内使用 PTX FIPS 模乘与对称平方，更宽的容器使用 CIOS。日志的 `backend=cuda-thread` 后显示实际容器、TPB、模乘、平方、寄存器、每线程本地存储和 NAF 恢复位置。CUDA thread 检查点由同一路径恢复，完成后的 Edwards save 可交给现有 Stage2。 [设备与检查点布局](../architecture/OPENCL.md#cuda-原生一线程一曲线) · [4060 Laptop 对照](../performance/STAGE1.md#cuda-与-opencl-一线程一曲线)

### 队列与交接

`--go` 是命中后的可选群阶诊断：默认优先本地 PARI DLL，备选外部 GP，不自动使用 GMP 算法。`--go-backend auto|pari|gp|gmp` 可选择后端，`native` 保留为 `pari` 的兼容别名，`--go-gp-timeout 1..600` 设置 GP 截止秒数（默认 30）；诊断未完成不影响因子存档、结果和主计算退出码。输出区分旧群阶界限与新增点阶界限，资源和素性证明范围见[群阶模块](../architecture/GROUP_ORDER.md#共享后端选择与进程隔离)。GPU Edwards 的既有 `--go` 限制保留。

生产任务通常使用 Suyama PARAM0。sigma 是曲线种子，不是曲线序号；不同参数化的 sigma 不能互换。GPU 一批曲线共享 N、B1 和指数，分别维护点状态。`ECM=`/`ECM2=` 队列行没有 save 名称；CPU/GPU 的自动 save 共用 `save_name_pattern`，默认 `{curve}_{target}_{b1}.save`，写入 `save_dir`。curve 为 e、m0、m2、m3；target 保留梅森任务的原始指数，例如 m5351，即使已经剥离因子。普通整数使用 bit 加实际位宽。B1 使用紧凑写法，例如 `m0_m5351_260e6.save`、`e_m5351_260e6.save`。旧自定义模式自动补充曲线前缀；CLI 显式 `-save` 路径保持原样。保存失败时任务留在 worktodo，不会进入 finished。

Stage1 启动器也接受 `ECMSTAGE1=` 队列，从 save 文件名末尾的 `_B1.save` 提取 B1。文件名前缀 `e_`、`m0_`、`m2_`、`m3_` 分别指定 Edwards、Montgomery PARAM0/2/3，覆盖该任务的 TOML `param`。`ed_` 兼容识别为 Edwards，并规范为 `e_`。Edwards 自动采用固定的 48·lcm 标量，忽略该任务的 Montgomery `lcm` 设置。确定实际曲线后，才应用 Edwards NAF 窗口和 CPU Montgomery 点乘链等曲线专用设置。CPU 遇到 `m2_` 或 `m3_` 等当前配置无法执行的任务会报错并保留队列。只有任务行本身无效时才将其标为 `# ERROR`。每条任务独立应用前缀，无类型前缀的旧名称沿用 TOML，并自动补充类型前缀。父目录和剩余文件名保留。

控制台 `START:` 保留原始 worktodo 行；`finished` 和受支持的 Prime95 交接行使用实际 save 名称，其他字段保持原样。补前缀时，若新目标不存在或为空，将旧 save 中同曲线族、同参数化的记录复制到新文件后追加，保持原 `skip_curves` 编号；旧文件保留。记录类型冲突时停止并保留队列。新目标已有内容时不重复复制旧历史。`save_name_pattern` 只生成没有显式 save 名称的任务路径。

Montgomery 路径的 `lcm=1` 使用 lcm(1…B1)，`lcm=12` 在该标量上乘 12。后者会改变保存点和因子发现范围，比较程序或生成测试存档时必须明确口径。Edwards 路径固定使用 48·lcm(1…B1)。CUDA PARAM0 和 Edwards 可使用完整 64 位 sigma；PARAM2/3 的批次受 32 位窗口限制。

GPU Edwards 用 `--param e` 和不带参数化前缀的 sigma。已知素因子的 Edwards 群阶、点阶及分解通过 [独立 PARI 入口](DATASET.md#edwards群阶与点阶) 计算。CUDA CGBN 默认使用窗口 NAF，`--naf-w 0` 选择窗口 6，显式允许 2..12。CGBN 切片的设备时间目标为约 100 ms，OpenCL 为约 50 ms。CPU 负责构造曲线、检查结果及求 GCD。窗口越大，GPU 字典显存和准备成本越高。自动文件名使用共用模式中的 e 前缀，显式 `-save` 可指定路径。GPU 算术和检查点限制见 [GPU Edwards 实现](../architecture/STAGE1.md#gpu-edwards)。CPU/GPU Edwards 完成后统一生成多曲线文本 save，供 Stage2 读取。

`[stage1] random_sigma = true` 是默认值。Edwards 新任务混合高分辨率时间、worker 编号和进程内采种计数，抽取一个 53 位任务种子。单次命令使用保留编号 `4294967295`。默认按种子和任务内曲线索引派生 σ，同一任务内互不重复。设为 `false` 时，从同样抽取的随机起点逐条递增 1。跨任务仍可能发生概率性重叠。GPU Montgomery 批次随机选择起点后连续编号。worktodo、命令行或配置指定非零 σ 时，使用该值作为连续区间的起点。恢复时沿用检查点里的 σ 或任务种子。

`gpu_init_threads` 控制 GPU Edwards 在主机端构造曲线的并行度，涵盖 CUDA CGBN、CUDA NTT 与 CUDA/OpenCL 一线程一曲线路径。σ 分配在构造前串行完成。

## CUDA 全宽 NTT

在 `[stage1]` 设置 `gpu_algorithm = "ntt"` 或 `2` 可以显式启用 NTT。线性和加权卷积均支持梅森承载指数 `32 ≤ p ≤ 5242880`，包括完整 `2^p−1` 和保留原指数及已知因子元数据的合格余因子。在 5242880 位上，保守选长使用 262144 点加权 NTT 或 524288 点线性 NTT。Edwards 使用 `param = "e"` 和固定的 `48·lcm(1…B1)`。Montgomery PARAM0 的 `montgomery_lcm_multiplier` 可设为 `"1"` 或 `"12"`。队列任务通过 `e_` 或 `m0_` save 前缀选择曲线构造。[已知因子与分段进位](../architecture/NTT.md#stage1-已知因子与梅森承载)

Edwards NTT 的 `edwards_naf_window_bits = 0` 按调优表选择吞吐最高的可用 NAF 窗口，2～12 选择原生 2XY NAF 窗口。生产和调优共用 `[stage1] ntt_naf_gpu_min_bits = "auto"`。默认按梅森指数 p、NAF 窗口和曲线数选择 GMP 或 GPU；调优按一条曲线计算。也可填写整数，例如 `ntt_naf_gpu_min_bits = 20000` 表示 p ≥ 20000 优先用 GPU，否则用 GMP。0 等同 auto，1 让所有位宽优先用 GPU，5242881 让所有支持的位宽用 GMP。若仅 GPU 建表的临时缓冲超出显存预算，可自动改用主机。若驻留频谱也放不下，可为新批次减小窗口或批量。两条路径生成同格式的 Edwards save。见 [字典计算和显存口径](../performance/STAGE1_MERSENNE_NTT.md#奇数倍点字典与显存)。

暂停后可以修改 `edwards_naf_window_bits`。已有算术快照的子批次以保存的模式、窗口和游标继续，随后开始的子批次使用新设置。日志 `stage1_ntt_naf_resume` 在窗口不同时列出保存值、配置值和恢复位置，`0` 表示 x-only。已有快照仍需为原窗口提供足够内存。见 [NTT 快照格式](../architecture/CHECKPOINT.md#46-cuda-ntt-单批算术快照)。

两条 Edwards NTT 路径都在主机准备曲线与基点。可用 `gpu_init_threads = 1` 强制串行构造，或设置所需的主机线程数。默认 0 根据位宽和批量自动选择，实际线程数在 `stage1_ntt_edwards_setup` 中记录。CPU Stage1 自身的线程数使用 `stage1_cpu_threads`。见 [主机初始化流程](../architecture/STAGE1.md#cuda-edwards-ntt-主机曲线初始化)和 [worker CPU 资源](../architecture/CONFIGURATION.md#worker-cpu-资源)。

```toml
config_version = 4

[stage1]
target = "gpu"
param = "m0"
gpu_algorithm = "NTT"
stage1_gpu_device = 1
ntt_profile = 0
```

```powershell
'2^8191-1' | .\ecm_cuda.exe --config .\ecm.toml -sigma 26 -gpucurves 2 -save m0_m8191_256.save 256
```

### 任务曲线数与设备批量

`worktodo.txt` 中的曲线数是请求量。NTT 按 `[stage1] ntt_max_batch` 和可用内存分批，上限默认 512，可设为 1–512。直接运行时，`-gpucurves` 指定的曲线数在上限和单批预算内时一次执行，超出时自动拆批。省略 `-gpucurves` 时 NTT 运行 1 条。

linear 和 weighted 按各自 NTT 长度估算每条曲线的工作区和 NAF 字典，分别受显存、字典预算和主机内存限制。自动选批可在任一候选能容纳的范围内选取批量，之后只用能容纳该批量的计划查表或补测。`stage1_ntt_batch_selection` 的 `linear_capacity`、`weighted_capacity` 是各卷积模式估算可容纳的最大曲线数，`capacity` 还受剩余任务量限制。`Stage1 execution:` 记录实际选中计划的每曲线估算、配置和设备额度、NAF 字典预算，以及 `max_curves_in_memory`。

自动拆批按调优表和当前容量选批。见 [生产选型](TUNE.md#生产选型)。

`ntt_pad_batch = 0` 默认让新任务精确完成请求量。设为 `1` 时，自动拆批可以在资源和 sigma 范围允许时用新曲线补齐尾批。例如每批 75 条，请求 1000 条时，尾批从 25 条补到 75 条，共完成 1050 条。新增曲线沿用任务的 σ 分配规则，计入实际结果和汇总。交接范围以实际保存记录为准。

尾批小于首选批量时，生产优先复用同任务已验证的 NTT 计划，并按实际曲线数检查当前容量。断点恢复会沿用仍符合配置上限的首选批量。若进程内没有计划，可以用首选批量的调优记录确定形状，再对实际批量验证；没有可用形状时按实际批量选型。

拆批检查点保存已完成曲线、σ 分配规则及其种子或起点、当前批量和整数坐标。随机 Edwards 通过种子和曲线索引重建各 σ，同一任务内的 σ 互异。旧检查点继续使用其连续序列。恢复时沿用已保存的尾批范围，再为其余曲线选型。任务完成后统一追加存档和内部结果、汇总适用的 NF 提交记录，并按交接设置生成 Stage2 队列行。子批次得到的已核验因子可先单独发布。

暂停后可修改同一 worktodo 行的曲线数，再恢复 NTT 队列任务。已完成前缀保留，检查点中的当前批次按原 σ、宽度和游标完整执行；即使新上限小于已完成数量，也不丢弃当前批。该批完成后，累计数量达到或超过新上限就进入发布，否则按新上限、当前 `ntt_max_batch` 和预算规划后续批次。若检查点没有当前批，且完成前缀已达到新上限，则直接发布，不再计算。save、结果和 Stage2 交接均记录实际完成数量，因此缩小上限后可能超过新的请求量。模数、B1、曲线构造和指定 σ 的恢复校验保持有效。

### NTT 形状与调优

`ntt_profile = 0` 按设备、NTT 长度和活跃曲线数选型；1 选择大 tile，2 选择分块 tile。`ntt_length`、`ntt_bits_per_word`、`ntt_tile` 和 `ntt_outer_radix_limit` 的 0 值使用自动值。`ntt_window_bits = -1` 使用配置档的 Graph 窗口，0 使用直接提交；`ntt_warp_tail = "auto"` 使用配置档的 warp 尾层设置。显式参数覆盖所选档位；长度和数字宽度仍须通过精确性检查。

`ntt_convolution` 默认为 `"auto"`。程序按当前曲线、预算、实际算术分派和显式参数查共享表。可选形状包括线性、加权二次幂，以及能够缩短变换的加权 radix-5。开启运行时调优后可补测缺档。不同指数的兼容 NTT 形状可以共用记录。`"linear"` 固定使用定宽线性卷积，`"weighted"` 固定使用精确 IBDWT 加权循环卷积，并在 `ntt_length=0` 时使用满足约束的较短长度。`ntt_bits_per_word` 可限定最大 digit 位宽。

加权卷积支持的混合长度为 2560、5120、10240、20480、40960、81920 和 163840。七档均可自动选长或显式指定。固定加权、采用保守界和自动 tile 时，393217–491520 位选 20480 点，786433–942080 位选 40960 点；原生 Edwards NAF 开启收紧界时按单独的边界选长。显式 `ntt_tile` 以二次幂子变换的层数为上限。日志以 `r5` 表示 radix-5 pass。[完整尺寸表](../usage/TUNE.md#ntt-and-transform-sizes)

`[stage1]` 下的 `ntt_tight_weighted_bound` 默认为 `true`。启用原生 Edwards 2XY NAF 后，自动选型将通过收紧界的短加权计划和其他合法计划一同按 NAF 列吞吐比较。固定 `"weighted"` 且长度为 0 时，程序选择通过检查的最短长度。M65521 可用 2560 点，把 `ntt_bits_per_word` 限为 25 或把 `ntt_tight_weighted_bound` 设为 `false`，加权计划改用 4096 点。强制指定不合格长度时报错。x-only 和异常曲线的 x-only 重放按保守界准入。 [配置参考](../ECM_TOML_REFERENCE.zh-CN.md#ntt_tight_weighted_bound) · [系数界](../architecture/NTT.md#stage1-加权卷积的无符号系数界) · [实测](../performance/STAGE1_MERSENNE_NTT.md#双卡加权自动选长矩阵)

原生 Edwards 加权计划通过四倍系数界检查时，可在频域合并 2XY 倍点的第一层。启用后，`ntt_u` 显示四倍乘积系数上界。超过此界时，原长度仍可单独计算四个乘积。`ntt_tight_weighted_bound = false` 同时关闭频域融合。 [2XY 融合与准入](../architecture/NTT.md#edwards-2xy-第一层频域融合)

命令行筛选、采样和日志字段见 [Stage1 NTT 调优](TUNE.md)。

NTT 单批快照保存标量位置、活跃曲线位图和整数坐标，允许在同一路径内切换卷积后端和 NTT 参数续跑。拆批检查点将当前批快照与已完成曲线一起保存，CGBN 与 NTT 各用自己的检查点格式。见 [NTT 算术、选型和实测](../performance/STAGE1_MERSENNE_NTT.md)。

### 启动详情示例

0.63.13 在 RTX 4060 Laptop 上使用 M65521、Edwards 2XY NAF w6、单曲线和默认 NTT 选型，日志节选如下。

```text
stage1_ntt_selection: source=edwards_unsigned_bound p=65521 batch=1 convolution=weighted length=2560
Stage1 execution: … ntt_bits=65521 ntt_type=weighted ntt_length=2560 … ntt_bpw_avg=25.594141 ntt_bpw_max=26 ntt_u=13695445919225353215 ntt_q=18446744069414584321
stage1_ntt_plan: … bpw_avg=25.594141 bpw_max=26 U=13695445919225353215 q=18446744069414584321
```

平均 bpw 是实际模数位宽除以参与打包的 digit 数，加权计划的 digit 数等于变换长度。最大位宽是单个 digit 的位宽上限，本例为 26 位。`U` 和 `q` 以十进制整数输出，必须满足 $U<q$ 才能从一个域余数恢复整数系数。 [系数界推导](../architecture/NTT.md#stage1-加权卷积的无符号系数界)

## 共用 TOML 与 worker

Stage1 与 Stage2 共用 `ecm.toml` 和 `worktodo.txt`。新版 TOML 在首个表头前声明 `config_version = 4`。`--worker N` 选择 TOML 中的 `[worker.N]` 覆盖及 worktodo 中的 `[Worker #N]` 队列分区。覆盖项沿用原键名，例如 `stage1_gpu_device`。见 [配置参考](../ECM_TOML_REFERENCE.zh-CN.md) 和 [版本说明](../architecture/CONFIGURATION.md#配置版本)。

`finished_file`、`log_file` 和 `worktodo_file` 在 `[shared]` 设置，以 TOML 所在目录解析。`stage1_save_dir` 仍相对程序目录解析。

`ECM=`、`ECM2=` 和 `ECMSTAGE1=` 交给 Stage1，`ECMSTAGE2=` 和 `ECMSTAGE2N=` 交给 Stage2。队列的 Stage1 默认 Edwards，`curve` 或 `param` 可覆盖，`ECMSTAGE1` 的存档类型前缀优先。GUI 根据下一个可运行任务启动对应程序，CLI 只领取本阶段任务。`[shared]` 的 `queue_order` 可选 `"file_order"`、`"stage1_first"` 或 `"stage2_first"`。优先阶段无任务时运行另一阶段。`queue_skip` 可选 `"none"`、`"stage1"` 或 `"stage2"`。不同 worker 可以共用队列文件，每个 worker 同时由一个程序处理。

当前程序无符合阶段、优先级和跳过条件的任务时正常结束；队列不存在或读取失败时报错。Stage2 从命令行、任务和配置均未取得非零 B2 时，可用 `[stage2] b2_ratio` 按存档 B1 × ratio 计算上界。结果须满足 B2 > B1。[Stage2 B2 优先级](STAGE2.md#队列语法)

## 自动交接到 Stage2

`[stage1] handoff_mode` 默认为 `"ecm-elf"`。Stage1 无因子完成且有可用保存点时，将原任务写入完成记录并生成 `ECMSTAGE2=`。投递保留 B2、曲线范围、AID 和已知因子，save 使用实际输出文件的绝对路径。Stage2 的位宽、曲线和 sigma 等执行条件由接收端检查。CUDA Stage2 拒绝不支持的输入时，会以 `# ERROR ECMSTAGE2=…` 将原任务保留在队列中，便于调整接收后端或其他参数。只运行 Stage1 时可设 `handoff_mode = "off"`。

`handoff_workers` 留空时保留当前 worker。同一队列内移除已完成的 Stage1 行并追加到该 worker 最后一段末尾。指定外部队列时写入其中的同编号 worker。填写 `"2"`、`"1,3"` 或 `"2-4"` 可选择接收 worker，编号范围为 1…1000000；在候选中选择任务行较少者，并列时取较小编号，同时考虑本 worker 尚未投递的记录。目标缺少分区时创建标签，文件头未标号部分视为 worker 1。同队列的 `"auto"` 使用 `[shared] worker_count` 确定候选范围。候选必须读取此队列并允许 Stage2。未设数量时从队列分区选取，没有分区时默认从 worker 1 开始。外部队列优先采用已有分区，缺少分区时按配置数量生成候选。

`[stage1] handoff_partial = true` 默认启用 NTT 分批交接。目标为 ECM-ELF 时，Stage1 验证并追加每个非末批的记录，再交接 `ECMSTAGE2PART`。末批完成后投递普通 `ECMSTAGE2`。不同批次可以交给不同 worker。同一队列需要不同 worker 才能让两阶段同时运行。单个 Stage1 任务的各批需在同一存档内连续追加。设置为 `false` 时，整条 Stage1 完成后集中交接。Prime95 交接使用整项完成的方式。 [Stage2 分批消费](STAGE2.md#分批交接)

`handoff_worktodo_file` 留空时使用当前 worker 生效的 `[shared] worktodo_file`。可填写绝对路径或相对 TOML 所在目录的路径。接收端去重使用 `[shared] finished_file`。在 `[worker.N]` 中使用同名键可覆盖交接设置。

`handoff_workers = "2[:4096], 3[4096:]"` 将小于 4096 bit 的余因子交给 worker 2，将 4096 bit 及以上的交给 worker 3。切片使用 Stage2 实际处理的余因子位宽，Edwards 命中后若继续交接则使用新余因子。编号区间也可以带切片，如 `2-4[:4096]`。`"2, 3"`、`"2-3"`、`"2[], 3[:]"` 等价。位宽重叠时，所有匹配的 worker 一起比较队列长度。如果没有匹配项，保留源任务并报告位宽，修正配置后可重试。

例如，将无因子完成的 Stage1 任务交给当前队列的 worker 2 或 3：

```toml
config_version = 4

[stage1]
handoff_mode = "ecm-elf"
handoff_workers = "2-3"
handoff_worktodo_file = ""
```

改为 `handoff_worktodo_file = "stage2_tasks.txt"` 即投递到同目录另一文件。跨 worker 或跨文件投递先写入源队列旁的 `.stage2_pending_workerN`，再完成源队列行移除；目标不可写时保留待投递记录，下次启动重试。已经存在于目标活动队列或接收端 finished 的相同行不会重复加入。关闭总开关不丢弃已有待投递记录。

Edwards 命中后可通过 [余因子交接](STAGE2.md#edwards-命中后续算余因子) 续算剩余模数。

## 计算与输出

主机生成曲线和指数，GPU 分片推进点乘，主机验证末点、求 GCD 并写入结果。`gpucurves` 给出单次命令请求的曲线数，设备实际驻留的曲线数取决于位宽、TPI、TPB、SM 数和资源占用。NTT 请求超过单批上限时按 [设备批量](#任务曲线数与设备批量) 拆分。

CPU Edwards 单曲线、SIMD 和 GPU 路径均在分块边界刷新进度；重定向到文件或 GUI 管道时输出普通文本百分比，不依赖终端控件。

GPU Edwards 先输出 `GPU Edwards initialization` 和 `GPU Edwards dictionary` 耗时，再进入与 CUDA Montgomery 相同格式的 `GPU:` 实时进度条。NAF 的累计量和单片增量单位为 `columns`；binary 为 `bits`，二者不可互换。NAF 直接使用已有列计数，不为进度重新编码标量。终端原地刷新；GUI 管道保留完整文本行，常规刷新间隔约 200 ms，首片、末片及安全停止时立即输出。

这里的 `bits` 是 Stage1 标量 S 的二进制位置数，不是目标 N 的位宽。`columns` 是同一 S 的窗口 NAF 位置数，包含零列；每列仍对应一个 2 的幂次位置，窗口 6 不代表每列处理 6 bits。每列执行一次倍点，非零列再加减一个预计算奇数倍点。NAF 列数可能与二进制位数不同，也不等于非零项数或点加法次数。比较两种路径的耗时应看同任务的完成时间及 `s/curve`；数字表示与例子见 [NAF 原理](../reference/ECM_EDWARDS_MATH.md#3-naf-标量乘法)。

Edwards 的初始化涵盖指数、曲线构造或恢复、标量编码及设备分配/上传，设备字典构建单独计时。NAF 在周期检查点、安全停止或完成时导出坐标，`GPU Edwards NAF snapshots` 记录导出次数。进度条 `elapsed` 从准备完成后开始，包含当前主循环的坐标导出、输出和检查点耗时。速度取最近最多 50 个切片的推进量/时间，排除进度回调中的周期检查点导出、写入及输出。续算以已保存位置为采样基准。完整计时边界见 [实现说明](../architecture/STAGE1.md#gpu-edwards)。

普通进度中的 `s/curve` 是当前速度的每曲线摊销；部分运行的投影不是完成一条生产曲线的实测时间。CGBN 和 NTT 共用 `[stage1] progress_log_seconds` 控制文件日志中的中间进度行。正值为最短间隔（默认 60 秒），`0` 仅保留完成进度，负值记录每次更新。控制台输出和 GUI 的进度刷新频率独立保留。NTT 按当前批次的实际迭代游标判断完成。NTT 进度行中的 `eta` 是当前批次的预计剩余秒数，按剩余迭代数和实测速率估算，`-1` 表示尚无有效样本。

Auto B2的可复用Stage1成本可用[tune_stage1_cost.py](../../tools/bench/tune_stage1_cost.py)测量完整批次并独立复核保存点。当前限CUDA ladder PARAM0、自动TPI、关闭指数缓存；可使用已知梅森素数目录，或通过`--target-n <N...>`测量指定余因子。所有测量曲线必须完成且通过无因子/独立末点校验，不接受部分运行速度投影。等级、参数、计时边界及使用方式统一见[完整Stage1成本预计算](../architecture/AUTO_B2.md#stage1-成本)。

最终 save 保存可继续 Stage2 的点；checkpoint 保存尚未完成 Stage1 的状态，二者不能互用。save 的归一化 X 必须属于记录中的 N、参数化、sigma、B1 和指数模式；checksum 用于检查文件完整性，不重新证明点乘正确。

内部 JSONL 记录全部曲线的 F/NF，整批在内存聚合，按 uint64 sigma 升序导出；新 save 批次使用同一排序。整条任务完成后统一发布最终内部结果和提交文件，NF 合并为一行（B2=B1），因子逐条保留 sigma、curves=1。Stage1 不生成逐曲线磁盘回执，字段与双文件恢复规则见 [ECM 结果输出](ECM_RESULTS.md)。

## 检查点与交接

完整内部格式、偏移、校验范围和兼容条件见 [Checkpoint 架构](../architecture/CHECKPOINT.md)；完成点字段、校验公式、发布去重及消费者能力见 [Save 架构](../architecture/SAVE.md)。本节说明正常使用与恢复操作。

Stage1 支持按切片保存中间进度。恢复时检查任务身份、格式和实际计算布局，不能通过改名绕过检查。未完成采样只输出 checkpoint，不应当作有效最终 save。

Ctrl+C、Ctrl+Break 或 GUI 的正常停止请求会在当前后端的下一个安全边界触发检查点写入，然后结束当前任务。CUDA/OpenCL 在 GPU 批次边界保存，CPU 在曲线计算的分块边界保存；这不是在信号处理函数中直接写文件。队列模式会随即退出，当前未完成任务及后续行留在 `worktodo`，不会被标为 `# ERROR` 或写入 `finished`；重启时从检查点恢复。`checkpoint_interval_seconds=0` 只关闭定期写入，不关闭中断存档。GUI 超时后仍会强制终止 worker，因此应给大批次和慢速存储足够的 `graceful_stop_ms`。

每个任务只使用一个批次检查点，CPU 各曲线的进度和完成结果都封装在该文件中。主文件名严格为：

- Edwards：`.e_s1_m{n}_{B1}.ckpt` 或 `.e_s1_bit{bits}_{B1}.ckpt`。
- Montgomery：`.m{param}_s1_m{n}_{B1}.ckpt` 或 `.m{param}_s1_bit{bits}_{B1}.ckpt`，param 为 0、2、3；CPU 使用 0。

有 k·b^n+c 元数据的任务使用原始 n，即使剥离了已知因子；直接输入完整梅森整数也能识别 n。其他整数使用实际位宽 bits。B1 使用完整十进制整数。例如 `.e_s1_m127_1000.ckpt`、`.m3_s1_bit512_1000.ckpt`。不追加 worker 或曲线序号，同名任务需要独立的 `save_dir`。

当前写入 `ECMS1B2` 二进制容器，采用固定小端字段、逐记录长度与 CRC32C，以及完整文件 CRC32C。任务身份绑定完整模数、曲线族、参数化、B1、指数约定、曲线数及精确 sigma。CPU 另绑定实现，GPU 载荷检查对应后端和状态布局。Montgomery CUDA ladder 保留 V5 头和设备状态，Edwards GPU 载荷保存状态、算法和指数指纹。指数从 B1 重建或从已验证缓存取得。

Edwards 有算术进度的批次沿用检查点中的标量编码及适用的 NAF 窗口、倍点公式。修改这些配置后，恢复中的批次按原列位置续跑，新批次采用当前配置。旧 CPU Edwards 检查点的外层任务身份允许历史窗口值，实际窗口从曲线记录读取。恢复仍检查载荷完整性、曲线身份和当前内存条件。

文本批次 V1 及其旧 Edwards GPU 文本载荷仍可读取。CUDA Montgomery ladder 可以从同目录旧 `.ecm_ckpt_<bits>_<first8>_<last8>.dat` 及其备份导入带校验的 V5 状态：逐 lane 核对完整 N、参数化、曲线数、布局、指数位数和进度，保留原文件。没有待验证的批次容器且没有当前算术载荷时才尝试导入；已有算术载荷优先。已有仅种子批次保留自身 sigma 并从头计算，不用独立旧 GPU 档覆盖。随机 sigma 从有效旧档继承，固定 sigma 不得被不同曲线替换；V4 及更早版本不导入。旧 GPU Edwards 独立文件 `ed_<旧模式>.save.gpu.ckpt` 也可在没有现有算术载荷时导入默认或当前模式对应的文件，继续检查完整 N、B1、指数、sigma 及曲线状态，原文件保留。

保留的旧源档不会单凭存在就反复续算：同一内部结果已确认全部曲线完成时，自动导入跳过该来源。迁移检查点时应保留配对最终结果，以及旧历史仍依赖的账本；若只将旧档复制到全新工作目录，程序无法从旧文件本身得知它是否已在其他目录完成。

原子写入使用 `.new` 临时文件；`checkpoint_backups` 轮换备份，例如 `.e_s1_m127_1000.bak.ckpt`、`.bak2.ckpt`。恢复检查完整校验，主档损坏时尝试备份；有效主档属于不同任务时直接拒绝，不覆盖它。没有任何有效备份时保留损坏文件并报错。CRC 用于发现字节损坏，恢复依靠有效备份，不执行纠错码修复。只有 sigma、没有非空算术载荷的文件保留了曲线身份，但不包含曲线内进度；定期存档尚未到时可能处于该状态。

同一批次有进程独占锁，防止两个 worker 同时写同名文件。CPU Edwards 已完成曲线保留结果，CPU Montgomery 保留完成结果及尚未完成的 ladder 状态；SIMD Edwards 在各 lane 的恢复位号一致时继续该批，否则重算该 SIMD 批。GPU 存放整个设备批次的状态。只支持本容器中的对应后端载荷，不自动读取 Prime95 的检查点。Stage2 的曲线计数文件并非曲线内算术检查点；本节命名和续算保证适用于 Stage1。

单次模式在最终多曲线 save、内部 JSONL 和提交 JSON 均成功发布后删除检查点。队列模式继续保留 checkpoint，直至持久完成记录、finished 和源行提交成功；重启优先恢复收尾，完成状态不明时停止并保留证据。写入失败时保留批次状态和原队列任务；恢复发布不会重复增加数学工作回执或重复追加相同 save 记录。内部与提交文件的配置、字段和校验见 [ECM 结果输出](ECM_RESULTS.md)。

### Prime95 持久交接

交接资格按每条任务应用文件名前缀后的实际曲线判断。Edwards 与 GPU Montgomery PARAM2 忽略 Prime95 转移路径，仅输出说明，不访问转移路径或账本。受支持的 Montgomery PARAM0/3 使用 `prime95_worktodo_file` 指定外部队列，另须配置 `prime95_save_dir` 指定 save 目标目录；`save_sync_dir_1/2` 只用于备份，不作为 P95 文件已就绪的依据。P95 save 读取的 sigma 上限为 2⁶³−1；更宽 sigma 可以进行本地 Stage1，但不能交给该读取协议。

成功批次生成规范化的 `ECMSTAGE2` 行，引用 P95 目录中的实际 save 与本批区间。同步先持久保存交接责任和原文件版本，再验证 N、B1、param、sigma、checksum、范围和既有目标前缀，完整发布并落盘目标文件，最后发布并落盘 `worktodo.add`。接收队列和父目录可以尚未创建，投递器会按需创建 `worktodo.add` 及其父目录。目标历史只允许延长，不能回退或重排。不同本地历史若使用同一目标文件名，会报前缀冲突并保留待交接记录；应为这些 worker 选择独立 p95_sync_dir 或保存名，不自动拼接并重映射 skip。任务引用的绝对路径须能被 Prime95 进程访问。文件失败不会发布对应任务；文件已就绪而任务写入失败时保留待交接记录，后续合资格任务或初始空队列启动可重试，无须重算 Stage1。关闭 `prime95_worktodo_file` 暂停新的 P95 重试，保留的交接记录在重新启用后仍按原目标和路由投递。

待交接记录位于程序目录的 `p95_add_pending.txt.handoff/`，`.intent` 表示仍由本程序承担交付责任，`.done` 保留批次去重依据。不要删除这些记录来清除 GUI 红色提示。旧 `p95_add_pending.txt` 中引用绝对 save 且范围、校验有效的 `ECMSTAGE2` 行会先转为新记录；无法证明保存范围的旧 `ECM/ECM2` 行保持原样并报错，需人工核对，不会猜测文件后继续投递。

“投递成功”表示本程序已持久发布 drop box，不代表 Prime95 已完成接纳。Prime95 不参与本程序的写锁/完成确认，不能承诺跨进程恰好一次。崩溃若发生在发布与确认之间，恢复会检查 add、活动队列和发送账本；仍不能判定时保留 `.sending` 和 intent 并停止该交接，避免盲目重复。计算结果仍保留。`.ready` 保存本次实际尝试发布的完整行集；恢复会先修复发送账本再确认 done。责任文件不可读时保留材料并报错，不能当成空文件。不要让两个独立消费者同时修改 Prime95 活动 `worktodo.txt`。

在 `[stage1]` 设 `handoff_mode = "prime95"` 启用 Prime95 交接。Prime95 协议不支持的任务按 `handoff_unsupported = "ecm-elf"` 投递到本地 Stage2 队列。可设 `"keep-save"` 保留在原地。

如果进程在记录交接意图后、完成源队列回执前中断，先沿用原路由恢复该任务，再修改 `handoff_mode`。此时源任务仍可能留在 worktodo 中。

跨 worker 投递前先将记录写入队列旁的 `.stage2_pending_workerN`；目标不可写时保留待投递记录并在下次启动时重试。已经存在于活动队列或 Stage2 finished 的相同行不会重复入队。已有 pending 记录在关闭交接开关后仍会重试。

待投递文件的新增、接收端选择和重试确认使用同一跨进程锁；新增责任持久发布后才允许推进源行，重试确认通过持久替换更新文件。目标队列已写入但确认失败时，原记录保留供去重重放，不覆盖并发新增的交接。

## 共享文件锁与升级

Stage1、Stage2、feeder 和 GUI 使用共同的短写入 gate，保护读改写全过程；Windows 关闭后删除短锁载体，进程退出会释放内核拥有权。owner/checkpoint 属于长期锁，计算期间继续持有，磁盘载体残留不代表仍被占用。不要按文件年龄清理活动锁。具体身份、锁顺序与等待规则见[文件并发与持久责任](../architecture/OVERVIEW.md#文件并发与持久责任)。

升级到 0.53.3 时先停止共享资源的写入程序，再成组更新 Stage1、Stage2、feeder 与 GUI；不支持新旧程序同时写相同资源，旧业务文件继续读取兼容。最终文件的符号链接与多硬链接写入别名会被拒绝，请使用实际文件路径；父目录别名可归一化。旧纯 `worktodo.add.lock` marker 不再参与互斥；pending/intent/ready/sending/done 是恢复材料，不能当作残留锁删除。

## 算法与构建范围

CUDA 默认路径为 ladder。resident/PRAC 必须在构建时启用，再显式选择；发布入口默认不编译这些实验模块。TPI 和寄存器实验的适用条件见 [Stage1 实现](../architecture/STAGE1.md) 与 [性能](../performance/STAGE1.md)。

Stage1 与 Stage2 仍为独立可执行文件。GUI 按队列行自动选择并在当前阶段任务耗尽后切换；直接启动 CLI 时应分别运行两个程序。

## 代码入口

- [main.cpp](../../src/main.cpp)：命令行、TOML、队列、进度和交接。
- [runtime.cu](../../src/ecm/stage1/montgomery/cuda/runtime.cu)：CUDA 设备调度与资源；曲线构造与结果转换在同目录 host_math.cpp，进度和诊断输出在 output.cpp，checkpoint codec 在 checkpoint.cpp。
- [ecm_stage1_exp.cpp](../../src/ecm/stage1/scalar.cpp)：指数生成与缓存。
- CPU 实现：[Montgomery](../../src/ecm/stage1/montgomery/cpu/)、[Edwards](../../src/ecm/stage1/edwards/cpu/)；各自拥有曲线流程和数学 checkpoint codec。

## 输入归一化与运行中编辑

`src/io/cli.cpp`、`config.cpp`、`worktodo.cpp` 分别解释三个来源，`normalize.cpp` 应用默认、优先级和组合检查。CLI 每次出现只转换一次，TOML 一次启动快照保留已解析值，默认/覆盖仅生成一次有效启动设置，不回填 CLI。原始输入保留显式值及来源位置；配置生成默认仍只以 `config/ecm_options.json` 为准。`ECM/ECM2` 省略曲线数时在归一化时补为 Prime95 的 100 条，空/零 sigma 表示未指定。无 B1/B2 位置参数时只有 worktodo 能提供任务；单次模式继续从 stdin 读取 N 或整数表达式。

队列模式拒绝 CLI 的 `--param`、`--curve`、`--lcm`、`-gpucurves`、`-save/-savea` 和单次群阶诊断选项；`--bits` 需用于 `--gpu-info` 或 `--tune ntt`。任务结构由 worktodo/typed save 决定。显式设备、target、impl、reduction、线程、NAF、affinity、checkpoint、目录和内核偏好可覆盖 worker 配置。sigma 优先级为 worktodo > 显式 CLI > 配置 > 原有随机策略；队列 CLI 仅接受不带 param 前缀的 sigma。运行时调试/缓存标志继续作用于进程。`--p95-dir` 为旧兼容参数，不启用交接。

CLI 或当前 Stage1 worker 的非法配置会停止启动，诊断给出来源、键、原值和允许类型或范围。CLI 覆盖仍需通过配置值校验。TOML 语法和重复键按整份文件检查，其他阶段和其他 worker 的选项值由对应的运行入口检查。选中的非法 worktodo 行标为 `# ERROR` 并继续。后端能力、运行失败、文件或锁错误会停止本次执行，保留任务供排查或重试。

计算期间冻结当前任务，其他行允许新增、删除、修改或重排。完成时重新读取文件，在当前 worker 中从上到下匹配第一条完全相同的任务；行号只用于诊断。找不到当前任务时保留结果和完成记录并停止推进，不删除其他行。任意编辑器不遵守文件编辑锁时，无法提供与合作写入方相同的并发保证。

启动只检查当前 worker 的 `ECMSTAGE1` 且 skip=0 的行，按有效 save 的 METHOD 记录数修复；同一文件在该轮扫描中只统计一次，已有 `ECMSTAGE2` 和其他 worker 不改动。实际领取时重新统计以处理前序任务增长；收尾再强制重读并采用发布证明的本批区间。例如历史 150 条、新完成 300 条，最终为 skip=150、count=300。去重或恢复时以实际连续记录为准，不用总数减请求数猜测；无法证明连续范围则停止收尾。
