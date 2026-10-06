# 原生 Auto B2：CLI、INI 与队列接入

日期：2026-10-05。延续 [设计](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_DESIGN.md) 和 [阶段成本标定](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_PHASE_CALIBRATION.md)。本阶段已实现并运行原生 Auto B2；当前发布的成本 profile 仅覆盖 **GPU1 / 精确 M8191 / B1=1000 / lcm / 指定后端与预算范围**。这仍不是任意生产 B1、模数和设备可直接使用的通用自动选择器。

2026-10-06更新：同4acc二进制的多端点重标定已发布三个宽度/六路径范围，独立盲测最大误差6.99%，见 [最新范围与chain实验](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_MULTIANCHOR.md)。本页保留10月5日的原生接口验收与当时profile记录。

## 1. 当前可用构建与命令

独立可执行文件：`build_cuda_cmake/_auto_b2_native_20261005/native/ecm_cuda_stage2.exe`，SHA256 `4acc15d26593eb9d8e66ff8f62ad8a64e12a18b04a4d01966dc1399580c4d9e2`。sm89、固定 Goldilocks PTX3、outer-unroll=0，25 个原始依赖冻结在同目录 `sources/`。完整 CUDA 编译约374.4秒，最终主程序编译约5.2秒。成本 profile 与二进制精确绑定；重新编译后必须重新标定。

GPU1 的运行 profile：[gpu1.cprof](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_gpu1.cprof)，SHA256 `0e62ea83ec8c0cdf4ba1cb72cc1b9225e6cd0f7433c1cb085f7176c4ba1e6639`。使用环境 CUDA runtime/driver=13030，设备 UUID=`8a67b1f8ef1c3177a822813a7ac2224d`。

以下命令在仓库根目录执行。先查看选择结果：

```powershell
$exe = 'build_cuda_cmake/_auto_b2_native_20261005/native/ecm_cuda_stage2.exe'
$profile = 'docs/data/ecm_auto_b2_native_20261005_gpu1.cprof'
$save = 'build_cuda_cmake/_auto_b2_native_20261005/study/m8191.save'
& $exe --device 1 --save $save --auto-b2 --cost-profile $profile --plan-only
```

实际执行：

```powershell
& $exe --device 1 --save $save --auto-b2 --cost-profile $profile `
  --results run/auto_b2/results.jsonl --log run/auto_b2/screen.log
```

程序本身不依赖 Python；Python 仅用于离线标定、审计和 profile 导出。本 profile 的命名配置是 factor-only，选择器会绑定 `NTT_NAME_HITS=0` 并记录 `requested_factor_only=true`；保留原始因子和必需算术检查。若要证明素性或拆解复合因子，可组合现有 `--factorize-hits --gp PATH`。

## 2. 参数、INI 和零 B2 语义

新增 CLI：

- `--auto-b2 --cost-profile FILE`：对有效 B2 为零的曲线自动选择。
- `--auto-min-b2 B2 --auto-max-b2 B2`：缩小搜索区间，不能超出实测范围。
- `--d D`：将候选限制为已标定的一个 D。
- `--arena-mb MB --owner-budget-mb MB`：选择相匹配的 arena 标定范围和 owner 上限；owner=0 强制使用已标定的回退路径。
- `--stage1-batch N`：使用相应批量的 Stage1 进程耗时摊销，当前有1与12。
- `--stage1-seconds-per-curve S`：用户给出的有限正数，总流程 Stage1 每曲线成本；读取 save 时也计入。
- `--stage2-ratio-adjust R`：有限正数，对预测 Stage2 成本乘 R，不代表目标 T2/T1。
- `--cost-device-info --device N`：仅返回设备与后端身份，供导出工具核验。

配置例子，profile 路径相对于 ini 所在目录：

```ini
[gpu]
device=1
[queue]
worktodo=worktodo.txt
finished=worktodo.finished.txt
tmp_dir=.
[stage2]
stage2_auto_b2=1
stage2_cost_profile=gpu1.cprof
stage2_arena_mb=4096
stage2_fold_mb=640
stage1_batch=12
stage2_ratio_adjust=1
```

其他可用键：`stage2_auto_min_b2`、`stage2_auto_max_b2`、`stage1_seconds_per_curve`。CLI 显式参数优先。队列可使用：

```text
ECMSTAGE2=1,2,8191,-1,"m8191.save",0,0,1,""
```

规则：

1. 显式非零 CLI/worktodo/INI B2 保持固定值；INI 开启 auto 不改变它，且不消费成本 profile。
2. 显式同时传 `--auto-b2` 和非零 `--b2` 是参数冲突。
3. 仅当最终有效 B2=0 且已开启 auto 时自动选择。没有开启 auto 的零值沿用原配置/错误语义。
4. `--plan-only` 查询设备并规划，不执行曲线、不写 result、不推进队列；`--dry-run` 只做输入检查，不验证实时 GPU 准入与完整成本 profile。
5. 每个实际 curve worker 根据当前 free VRAM 重新选择，结果写入真实 B2。失败不写该曲线成功结果，队列保留；全部曲线成功后追加原始任务并推进队列。
6. 多条直接输入的保存点分别按自身 B1 规划；队列仍要求同一任务保存点的 B1 相同。这里没有新增并发调度或总显存租约。

队列尾部的 quoted 字段表示已知因子；任务标识 AID 放在 k 前面。测试使用 `ECMSTAGE2=auto-test,1,2,8191,-1,"m8191.save",0,0,1,""`，没有将标识放入因子字段。

## 3. 原生算法与成本

[成本 reader/solver](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:39) 读取版本化 `.cprof`，包含 binary/device/backend/accounting/naming 身份、Stage1 摊销和分阶段率。限制大小1MiB；拒绝缺失 END、重复记录、非法整数、NaN/Inf、负数、越界规模与未知格式。Windows 读锁覆盖文件指纹及内容读取，避免在规划期间替换 profile。

[设备和 packing 接口](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:48) 查询实际 `ntt_shape_query`，主程序没有另写近似的 NTT 长度公式。UUID 不匹配在设置 CUDA device 前拒绝；identity 包括设备、SM、CUDA runtime/driver、固定后端、outer 和二进制 SHA。Mersenne 限制直接检查实际 N 的位数及 popcount，余因子/泛型 N 不外推。

[联合选择](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:105)：

\[
P=\varphi(D)/2,\quad I=\lfloor B2/D\rfloor+2,\quad G=\lceil I/P\rceil.
\]

用33个对数 B2 点，加入32768点 giant chain/ladder 切换边界与附近 G 树边界的整数邻点。遍历实测 D 和驻留/回退路径，校验 P/G/B2、arena、owner 的范围，再最大化：

\[
a=1.96617-0.06781\log_{10}(B1),\quad
K=0.11343+0.88657(\log_{10}(B2/B1)/2)^a,
\quad score=\frac{K}{T1+R(T_{engine}+T_{cold})}.
\]

K 是沿用 Prime95 的相对收益近似，不是绝对成功概率。engine 包含 baby、affine、F树、G树、fold、descent、inverse、accum、glue、giant；giant 复用实际256MiB坐标分块及32768点阈值的 chain/ladder 分率与 chain 固定成本。阶段公式详见前一标定报告。cold 为实测进程耗时减 engine 的训练中位数，受主机负载/启动影响；没有声称完整进程 wall 具有同样的10%误差保证。

内存准入要求实际 geometry 的 arena 估算不超 scope，驻留 owner 满足预算，并要求实测 arena cap 能容纳于 free VRAM−768MiB。owner payload 为 `8*ceil(bits/64)*(9P+8)+48` 字节。它是保守的组件准入，**不是全进程同时峰值保证**，输出 `process_peak_guaranteed=false`。

[主程序绑定及检查](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:493) 校验已登记的算法/内核环境配置，设置选中的 arena/owner/D_MODEL。profile 命名配置绑定后，工作进程直接执行选中的 D/B2。结果记录原始 `requested_B2`、`requested_D`，新增 `auto_b2`、`auto_plan` 和 `auto_planning_seconds`；worker 的 `seconds` 包含规划和初始化，可选 GP 拆解时间仍分开。

## 4. 当前实测范围与失败的模型

新二进制重新标定18批 Stage1 /117个独立参考核验点；Stage2 36拟合+12留出检查，再冻结预测后运行36次 B2=45亿盲测。84条曲线全部算术检查通过，472264个 GMP 抽样系数，21组同几何跨路径/重复 leaf 哈希一致。另有本阶段4条真实 Auto/manual/queue 验收曲线。

六组模型在60亿留出检查的误差均在10%以内，但新盲测发现：

- 2203 bits：驻留最大绝对误差25.125%，回退21.384%。
- 4423 bits：驻留23.230%，回退20.951%。
- 8191 bits：驻留3.058%，回退2.402%。

三个宽度的预测最快 D/path 都与实测一致，但正确排名不能代替时长标定。审计分开给出算术通过、每个范围的预测门限与合格范围的排名；导出器只发布通过独立盲测的范围。**2203/4423被排除，原生 auto 会明确报没有实测范围；手动 B2/D 仍可运行。** 失败原始预测与日志保留，不修改10%门限来通过。

当前可发布范围：精确 M8191，B1=1000、lcm、Stage1 batch1/12，B2=30亿～60亿，D=30030/60060/120120，arena=4096MiB，owner驻留与预算回退；P/G还必须满足 profile 实测矩形范围，因此不是区间内的所有 D/B2 组合均可用。不能用于 choose12、高 B1、G=1、任意位宽或其他 GPU。

原生默认选择 B2=30亿、D=60060、P=5760、I=49952、G=9，驻留 owner=50.633MiB，fold NTT length=2^24。arena 形状估算约768.176MiB，而匹配实测 cache cap 仍为4096MiB。模型 T1=0.827374s（batch1进程摊销），engine=4.974032s，cold=1.516060s，T2=6.490092s。选择位于下界，`range_limited=true`，不能据此宣布通用最优 B2。指定实际 Stage1 batch=12 时使用对应摊销值。

对失败模型进行一次不发布的按D分率诊断，偏差没有消失；这尚未确认是主机负载、形状特征还是其他固定开销的原因。下一步需交叉顺序采样、记录 CPU/GPU 状态，并细化小位宽的 launch/尾树成本，再使用新的盲测 B2；已使用的45亿数据只能作为诊断，不能重复宣称独立验证。

## 5. 验收和证据

[原生验收脚本](D:/code/MPA-OpenCl/tools/test/test_stage2_auto_b2_native.py:21)：33次调用、213项断言通过。原生各 phase、packing、K/score 与 Python 模型逐项一致；固定 D、owner=0、Stage1 batch/显式成本、ratio、范围参数生效。缺失/损坏/重复/NaN profile、错误 binary/UUID/driver、配置冲突、不支持位宽/B1/泛型 N 均拒绝。

实际 auto、manual、INI queue-auto、INI auto开启但非零queue B2共4条GPU曲线，raw因素一致，leaf hash均 `7271918632011950804`，必需检查全通过。plan-only不推进，worker失败保留队列且不产生成功结果，成功只消费原任务，finished保留两条原始任务。前两次验收脚本运行因预期错误文本和测试任务字段写法不正确而中止，保留 `acceptance/acceptance2`，修正测试后最终 `acceptance3` 通过；没有把它们报为算术失败。

另有4项导出保护检查：没有合格范围、合格范围排名慢于5%门限、旧格式审计整体误差超过10%、算术审计失败，均拒绝发布且不生成目标 `.cprof`。输入及结果包含在 evidence 的 `export_guard` 和对应文件内。

可移植证据：[原始实验及日志](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_evidence.json)、[完整模型](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_profile.json)、[审计](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_audit.json)、[原生验收](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_acceptance.json)。

重新导出：

```powershell
python tools/bench/export_ecm_cost_profile.py `
  --profile build_cuda_cmake/_auto_b2_native_20261005/profile.json `
  --audit build_cuda_cmake/_auto_b2_native_20261005/audit.json `
  --stage2 $exe --device 1 --output run/gpu1.cprof
```

完整验收：

```powershell
python tools/test/test_stage2_auto_b2_native.py --exe $exe --profile $profile `
  --model docs/data/ecm_auto_b2_native_20261005_profile.json `
  --save-dir build_cuda_cmake/_auto_b2_native_20261005/study --output run/auto_b2_acceptance
```

## 6. 后续推进

优先补小位宽可靠成本、低 B2/较小 D/G=1，确认最优点是否仍位于边界；扩展生产 B1 与 choose12 的 Stage1 摊销及逆元/GCD成本。之后才可为更宽输入提供默认 Auto B2，不能只插值当前 B1=1000。

总显存活跃集/lease、多曲线并发、在线成本反馈以及 GPU seed后 giant chain阈值、NTT小层 launch/同步优化仍待推进。本阶段交付是原生受限选择与可复查的门限保护，没有将整个长期优化目标标为完成。
