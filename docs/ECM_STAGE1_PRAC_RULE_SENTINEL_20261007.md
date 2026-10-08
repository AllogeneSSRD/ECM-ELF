# Stage1 param0：PRAC 终结哨兵与配对版本实验（2026-10-07）

## 1. 实验身份与问题

上一轮输出不别名 ADD 的完整机器码不变，已提交为 `2ff2875`。本轮在其上仅改变 `single-compact` 的循环控制：把跨过大段 xADD 的 `finish` 标记编码为 `rule=4`。四种普通规则仍为0..3；公共算术、点角色、PRAC计划、归一化和切片/检查点语义保持原实现。

- before：`2ff2875` 的输出不别名 ADD，沿用上一轮 v2；exe SHA256 `bcfa5aa8b6b5da23f218619b2a897241a2676fc897dd5fc310d0e6a29c7b45d0`。
- after：本轮终结哨兵，称为 v3；exe SHA256 `464f124878ab755fba5e698f26962c15a1425c7bc0f4de83ed85e8df773d2b88`。
- 同名 `ECM_PRAC_VARIANT=single-compact` 在两个冻结版本中指代不同候选，必须记录二进制 SHA。
- 仍只有4608/TPI16、register255/168支持本候选。默认配置未修改。
- 专用 body SHA256 `521327f138899206f3bf98b271dda421fd81e0e78ab321a4412c63f3d52cd1d9`；公共算术头 SHA256仍为 `67dc71128495378991dd750d73f7d8226ebedd255e267abd435dc6331a03e0a6`。

冻结旧 exe、DLL、对象、编译配置及私有头位于本地 `build_cuda_cmake/prac/before_rule_sentinel_20261007/`。原始数据在忽略的 `docs/data/` 中。

## 2. 源码及预测

改动位于 [私有单点链](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:48)：

1. COMPACT=true：初始 `rule = d==e ? 4 : 0`，规则计算只在 `rule!=4` 时执行。
2. 仍统一调用一次输出不别名 ADD，终结时 `rule==4` 将 T 复制给 A 并退出。
3. 普通规则0..3的点交换、DBL和d/e更新完全沿用此前流程。
4. COMPACT=false 保留单独 finish，保持旧 `single-add` 的编译路径。

排名预测：减少控制状态的存活区间可能减少寄存器/spill；编译器也可能已完成同等优化；另一个可能是新增规则比较增加寄存器/指令。CPU模型不是寄存器预测工具，先检查机器码，再决定实测。

结果：自然策略寄存器**增加170→174**，未实现自然168分配的目标；cap168的stack/spill减少。静态结果支持进一步测试cap168吞吐，也允许检验自然策略的调度变化；不能从源码标量数量直接推断硬件寄存器数。

## 3. 构建与静态资源

专用单 TU 构建165.5秒，随后仅链接。之前两次197.8/205.9秒是各自构建记录，本轮不把差异宣称为新编译算法的加速。

- MODE10/11：完整函数 SASS 和资源行与 before 相同，保留旧 single-add 对照。
- MODE12/natural：170→174 registers；stack48、spill0/0不变；36,128→36,096条静态指令，text578,048→577,536 bytes（−512）。
- MODE13/cap168：168 registers不变；stack88→80 bytes；spill store/load48/40→36/28 bytes；36,000→36,056条静态指令，text576,000→576,896 bytes（+896）。
- 两个新 entry 的 CONSTANT[2]68→92 bytes。无新增 outlined callee 或设备调用 ABI。
- normal TPI16、host及dispatcher对象SHA与冻结 before 快照相同。

比较工具 [compare_stage1_sass.py](D:/code/MPA-OpenCl/tools/bench/compare_stage1_sass.py:1) 逐函数哈希完整导出行，包含第二行调度编码；只归一化换行和函数尾部空行。摘要包含导出文件SHA及旧/新资源行。工具计数包含 `Function` 标题，较上一轮不含标题的行数多1，不影响比较结论。它提供静态证据，不替代 GPU 正确性或吞吐验证。

工具的六项fixture检查覆盖相同函数、CRLF/尾空行、单独调度编码变化、资源变化、缺失及重复函数。首轮换行fixture把已有CRLF再次替换为CRCRLF而失败，修正fixture后通过；没有修改比较逻辑以绕过失败。配对计时工具的mock覆盖两种模式、反序、二进制身份及聚合；这些不是原生性能数据。

## 4. 正确性门禁

[五模型链角色对照](D:/code/MPA-OpenCl/tools/test/test_prac_single_add.py:35)：8,533 prime/d×两sigma=17,066数学输入链，baseline、single-add、compact-v1、disjoint-v2、sentinel-v3共85,330求值。264,538规则步骤的全部中间/终结XZ、操作计数及独立ladder一致。五种规则全部覆盖。

独立 normalized Montgomery/别名模型804正例、4,824公共别名比较通过；差分点输出违规786项错误，归一化反例仍存在。两个CPU模型均记录本轮私有头SHA；CPU结果不代替GPU证明。

同一after SHA、GPU1：natural184条和cap168/50ms/C384的3,568条完整Q/save对照通过；各27类完整门禁案例包括lcm/choose12、64bit sigma、因子、退化、损坏/恢复检查点及目标边界。生产窗口门禁1,352窗口Q、8恢复Q、20配置拒绝；B1=10m/260m的prefix/middle/tail、count32、chunk7/32通过。oracle184miss/1,168hit，每个输出仍检查。

完整XZ CSV逐字节比较128项通过，包含48项跨切片，不重复累加；窗口门禁169条结果全部完成。

本地证据：`stage1_rule_sentinel_{roles,montgomery}_20261007.json`、`stage1_rule_sentinel_{natural,cap168,windows}_q_gate_20261007/`。

## 5. 性能实验口径

[配对版本工具](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_versions.py:1) 用三种身份 before/after/baseline，baseline取after二进制的未改动内核。两重复的配置顺序互相反转；每次执行前后检查exe SHA，输入计划缓存必须命中，记录每一行版本/SHA。

GPU1 RTX4060 Laptop、N4423/容器4608/TPI16/TPB128、sigma26/lcm，GPU任务串行执行；GPU0生产任务保持其原状态。性能计时期间没有编译、大SASS导出或NCU重放。

- 固定子乘积：B1=260m的tail32，三C384/768/1536×register255/168×chunk4/32×三身份×两反序，共72样本；每轮恢复同一seed，排除两个warmup轮，采样6秒。
- 普通前缀：C1536/register168、B1=10m/260m、目标50/100ms×三身份×两反序，共24样本；采样15秒，warmup5秒，取最后5秒的精确s/curve中位；正常采样限时退出，不能视为完整Stage1耗时。
- 所有数据都是子乘积/短时投影，不发布Auto B2 T1。

`grid=ceil(C/(TPB/TPI))`。日志blocks/SM是资源允许的驻留容量，不能据此假定C384也填满3blocks/SM；GPU1共24SM，C384/TPI16只有48个提交blocks。寄存器上限不自动修改TPB。

### 5.1 固定窗口：72样本完成

以两反序的墙钟投影中位为主，顺序为before/after/baseline，单位s/curve。投影公式 `W_full * measured_wall_ms / (W_window * measured_rounds * 1000 * C)`；事件kernel计时排除seed恢复，墙钟包含每轮恢复及launch/synchronize。tail32的W_window=8,053，W_full=3,369,476,895；不是完整曲线计时。

cap168短片chunk4：

- C384：151.318965 / 150.459692 / 150.963202；after相对before−0.568%，相对baseline−0.334%。
- C768：149.069511 / 147.889291 / 148.508088；−0.792% / −0.417%。
- C1536：133.316032 / 132.527838 / 132.965714；−0.591% / −0.329%。

cap168长片chunk32：

- C384：149.717381 / 148.805272 / 149.540132；−0.609% / −0.491%。
- C768：153.084557 / 152.082774 / 183.957911；−0.654% / −17.327%。
- C1536：139.788492 / 139.205975 / 174.199776；−0.417% / −20.088%。

自然策略六组相对before减少2.014%..2.212%，相对baseline减少1.754%..1.820%。C1536短/长分别148.032198→144.757580、147.568403→144.309309。自然仍不如C1536 cap168短片。

长片相对baseline的17%..20%主要继承此前单点ADD的收益，**本轮新增收益只有相对before的0.417%..0.654%**。不能把累计改善全部归因于哨兵。

严格重建三C×两reg×两chunk×三身份×两重复，72/72无重复或缺项；计划/指数全部缓存命中，每条first/count/prime/work/full_work相同，无最终save及检查点修改。独立分析包含事件和墙钟两种口径，位于本地 `stage1_rule_sentinel_fixed_20261007/paired_analysis.json`。

GPU采样1,060点，869点利用率≥70%，这些点SM时钟均1800MHz；整个采样温度48..65°C，设备总memory.used采样最大323MiB，均非进程完整峰值。

### 5.2 普通前缀与独立50ms确认

首批24样本结束后，再运行只含50ms的三身份×两B1×两反序12样本；不丢弃首批。合计36条正常前缀，50ms每身份/每B1四次，100ms两次。以下对每个配置的运行级s/curve投影取中位，顺序before/after/baseline：

- B1=10m、50ms：5.080760 / **5.050395** / 5.063071；after相对before−0.598%，相对baseline−0.250%。
- B1=260m、50ms：132.941849 / **132.178987** / 132.510655；−0.574% / −0.250%。
- B1=10m、100ms：5.104009 / 5.087464 / 5.243030；−0.324% / −2.967%。
- B1=260m、100ms：134.406514 / 134.322842 / 138.787888；−0.062% / −3.217%。

50ms的四次范围：10m after5.049905..5.051005、baseline5.062621..5.063572；260m after132.162809..132.274678、baseline132.457980..132.540196。本批范围没有交叠，收益小而一致；仍限于这组硬件/配置的短时前缀。10m/260m前缀有共同的早期素数和规则，不能将它们当作完全独立的完整曲线样本。

100ms的baseline波动更大，260m范围138.009215..139.566560；本轮只声称哨兵相对before的变化，较baseline的累计收益包含既有单点ADD和compact DBL。50ms仍是本批最佳目标，100ms没有提高最佳吞吐。

全部108个计时样本（72固定+24前缀+12确认）严格从controls重建矩阵，检查无遗漏/重复、exe SHA、缓存命中、几何及原始日志；前缀均有“sample limit reached; incomplete Stage1 saved only as checkpoint”，正常exit1，无强制结束或最终save，独立目录不恢复旧checkpoint。源码和二进制在计时期间不变。

前缀752采样点/703忙点、确认376点/354忙点（利用率≥70%），忙时SM均1800MHz；温度分别57..66°C、49..66°C，memory.used采样最大均317MiB。完整计时审计本地文件为 `stage1_rule_sentinel_timings_audit_20261007.json`。

### 5.3 管理员NCU：五份采集完成

五份均管理员exit0、19passes、CSV导出exit0；捕获一次tail32、grid192/TPB128、Device1、4608/TPI16，与计时分开。逐项核对exe SHA、实际demangled MODE、三维grid/block及相同first/prime/work/warmup；clock-control/cache-control为none，跳过前两次匹配launch后捕获一份恢复窗口。重放时长不用作吞吐。

首份采集后的collect-only命令漏必填`--exp-cache`，CLI拒绝；补参数导出已存在报告，未重新采集。只读解析器先按mangled/一维尺寸假定及错误DRAM字段名拒绝实际CSV；读取真实列格式后改用demangled名称、三维尺寸和`gpu__dram_throughput...`，全部核对通过，未重跑GPU。

cap168 before→after，baseline作为第三项：

- 硬件分配168、寄存器限制容量3blocks/SM均不变。
- local load sectors983,040→688,128；store820,048→525,568。
- `32*(load+store)/2^20`：55.025879→37.039062MiB，减少17.986816MiB（**32.688%**）；baseline37.028931MiB。
- local load L1命中率99.872233→99.904669%；store96.325093→95.375670%，访问以L1命中为主。
- eligible warps/scheduler/cycle0.461668→0.461933，几乎相同；issue active32.848125→33.117792%。
- wait/issue-active4.356346→4.211369；no_instruction0.639560→0.720858、short_scoreboard0.252706→0.349937反而增加，不能简化为全部stall都下降。

自然策略before→after：实际170→174、硬件分配均176、寄存器限制均2blocks/SM；local load仍0，store21,556→21,580。eligible0.356790→0.365991，issue active30.685756→31.266529%，wait3.965871→3.849777，short_scoreboard0.190068→0.167711。指标方向与自然窗口改善相符，未证明单一调度原因。

该恢复窗口的DRAM throughput仅0.000053%..0.000151% of peak；没有DRAM吞吐饱和证据，不能据此推广整段Stage1的CPU准备成本。local代理不是显存容量/DRAM/PCIe字节，也不乘19passes；stall比值不是墙钟占比。

本地证据：`stage1_rule_sentinel_ncu_{before168,after168,baseline168,before255,after255}_20261007/`，汇总为 `stage1_rule_sentinel_ncu_analysis_20261007.json`。

## 6. 计算量、内存与传输

计划中ADD次数A、DBL次数D对应 `4A+3D` 模乘和 `2A+2D` 平方；本CGBN平方路径按模乘等价计，仍为 `W=6A+5D`。本轮没有减少椭圆曲线操作或依赖模乘；变化来自控制流和编译调度/溢出。

B=容器位数，C=曲线数，P=计划记录数：显式curve和seed各 `7CB/8` bytes，计划 `16P` bytes，切片边界逻辑量 `6CB/8 * launches`，均未改变。C1536/B4608时，data/seed各6,193,152 bytes；tail32/chunk4有8launch，边界逻辑量42,467,328 bytes；chunk32一launch为5,308,416 bytes。这些是设备逻辑数组读写量，不能称为新增PCIe传输。

stack/spill是每线程静态编译记录，不能简单乘C当作进程峰值显存。NCU local sectors×32若用于累计访问代理，须区分L1缓存流量、显存容量及DRAM/PCIe；不得乘重放pass数。GPU采样memory.used也只提供采样时设备总占用，不是完整进程分配峰值。

## 7. 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/internal/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac_single_add\.cu$' -Jobs 6

python tools/bench/compare_stage1_sass.py `
  --old docs/data/stage1_disjoint_add_sass_20261007 `
  --new docs/data/stage1_rule_sentinel_sass_20261007 `
  --output docs/data/sentinel_sass_comparison.json

python tools/bench/bench_stage1_prac_versions.py `
  --before build_cuda_cmake/prac/before_rule_sentinel_20261007/ecm_cuda.exe `
  --after build_cuda_cmake/prac/ecm_cuda.exe --mode windows `
  --curves 384 768 1536 --registers 255 168 --b1 260000000 `
  --chunks 4 32 --seconds 6 --exp-cache build_cuda_cmake/prac --output docs/data/sentinel_fixed

python tools/bench/bench_stage1_prac_versions.py `
  --before build_cuda_cmake/prac/before_rule_sentinel_20261007/ecm_cuda.exe `
  --after build_cuda_cmake/prac/ecm_cuda.exe --mode prefix `
  --curves 1536 --registers 168 --b1 10000000 260000000 --target-ms 50 100 `
  --seconds 15 --warmup 5 --exp-cache build_cuda_cmake/prac --output docs/data/sentinel_prefix

python tools/bench/bench_stage1_prac_versions.py `
  --before build_cuda_cmake/prac/before_rule_sentinel_20261007/ecm_cuda.exe `
  --after build_cuda_cmake/prac/ecm_cuda.exe --mode prefix `
  --curves 1536 --registers 168 --b1 10000000 260000000 --target-ms 50 `
  --seconds 15 --warmup 5 --repeats 2 --exp-cache build_cuda_cmake/prac `
  --output docs/data/sentinel_confirm50
```

需要保留完整before快照及同轮SASS导出；未来若after编译覆盖当前路径，先冻结本轮exe/DLL/source再开展新实验。输出目录必须新建，门禁和生产检查点使用独立目录。

## 8. 采用决定与下一项

保留v3为显式候选；在**本机GPU1/N4423、C1536、lcm、cap168、50ms的已测范围**，它取代baseline cap168/50ms成为暂定最佳实验配置。50ms四重复的短时投影相对baseline改善约0.25%，相对v2改善约0.57%..0.60%。这不是完整B1曲线的实测加速，也没有覆盖其他N位数、硬件或生产Auto B2 T1；默认算法/寄存器/TPI/TPB128/100ms不变。

显式复现该配置：

```powershell
$env:ECM_GPU_STAGE1_ALGO='prac'
$env:ECM_PRAC_VARIANT='single-compact'
$env:ECM_PRAC_REG_TARGET='168'
$env:ECM_PRAC_TARGET_MS='50'
$env:ECM_STAGE1_TPI='16'
```

本轮增加的收益与更少spill/local访问及控制/寄存器调度变化并存，不能把32.688%的local代理下降解释为32.688%总时间改善。自然策略174/176仍未达到168分配门槛；最佳配置仍依赖寄存器cap。

下一项优先试验**共享私有链的seed DBL与循环DBL调用体**：当前二者分别inline，尝试统一到一次固定点变量的DBL，目标是减少重复text及存活集合；保持归一化、同一ADD/DBL计数、全部中间/终结XZ和无设备调用ABI。先CPU点角色模型及单TU SASS检查，防止编译器重新拆分或增加spill；出现可解释的机器码变化后，再跑同样的固定窗口/生产前缀配对。公共p2分支保留作为后续独立因素。本轮尚未实现此项，不宣称会加速。
