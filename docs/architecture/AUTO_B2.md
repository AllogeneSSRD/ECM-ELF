# Auto B2 与 tune

## 目标与生产入口

Auto B2 选择连续生成、处理新曲线的单位时间收益，计入 Stage1 成本，即使当前读取已完成的 save。固定 B2 时，性能文件选择 D 和合法梅森承载；最终 B2=0 且启用 Auto B2 时联合选择 B2、D、承载。

默认性能文件为 INI 所在目录的 `stage2_ecm_tune.toml`，可由 `stage2_tune_profile` 或 `--tune-profile FILE.toml`覆盖。`--auto-b2` 对应 `stage2_auto_b2=1`。显式同时提供非零 `--b2` 与 `--auto-b2` 报冲突；任务或 INI 已有非零 B2 时保持固定值。

性能数据可跨实际位宽、B1 复用，允许插值、外推。复用从不要求 B1 相等；实际 B1 仍用于区间、成功概率、Stage1 成本和真实运算。模型估算与实测在输出中分别标识。

`stage2_tune_ignore` 默认 `gpu,driver,cuda,backend,environment`，可选项另含 `memory`；空值要求这些条件匹配。默认 memory 按 fold/frontier 执行路径匹配，不要求预算数值相等。忽略条件不会取消 save、算术、整除关系和当前显存检查，也不表示其他设备的秒数已经在本机验证。

非零 `--d` 锁定 D。`--carrier-exponent 0` 锁定普通运算，非零 p 锁定承载。未锁定时，worktodo `ECMSTAGE2=1,2,p,-1,...`先按已知因子重建实际 N，核对 save，再验证 **N 整除 2ᵖ−1**，比较普通与承载。只读 save 而没有指数来源时不猜测承载指数。

## 收益与搜索

P=φ(D)/2，I=⌊B2/D⌋+2，G=⌈I/P⌉。搜索目标为：

**收益 = Pr(实际 B1、B2，目标素因子位数)/(T1 + R·T2_guard)**。

- T1 是已摊销的 Stage1 秒/曲线，不是 save 读取或 tune 点生成时间。
- R 对应正的 `stage2_ratio_adjust`。
- T2_guard = 预测引擎时间·(1+相对误差余量)+2·MAD；包含 shape、init、main，排除 Stage1 和父进程启动。
- 默认目标因子位数从 B1 推荐式反推并受实际 N 约束；`stage2_target_factor_bits`可覆盖。因子规模不能直接用余因子总位宽代替。
- 原生 Dickman/半光滑概率与 `tools/ecm_prob` 对拍；当前 poly 不计入未实现的 Brent–Suyama 收益。

默认 B1<B2≤2.6×10¹²；`stage2_auto_min_b2`、`stage2_auto_max_b2`及对应 CLI 可修改。候选包含区间内49个对数间距点及已记录 B2，D从14项表选择。搜索不受 tune 网格边界限制，但有限候选搜索不证明连续范围全局最优。

先按收益或固定 B2 成本排序，再逐项要求原生计划 `valid=true`、`finished=true`、`initial_free_snapshot_fits=true`。规划后 fold/frontier 路径变化时不借用原路径排名。实际执行仍检查 free/headroom及处理物理分配失败。无可用成本时固定 B2 返回原因并保留原生选型；Auto B2 明确失败，队列保持未完成。

### Stage1 成本

优先级：正的 `stage1_seconds_per_curve` → `stage1_cost_csv` → `stage1_tune_profile`。秒/曲线不再除以 `stage1_batch`。

CSV 必填 `b1,mhz,tpi,curves,seconds_per_curve`，另提供 `container_bits` 或 `target_bits`。可选 `tpb`（默认128）、`exponent`（空值表示未指定，否则 `lcm|choose12`）。空耗时表示缺失测量，同 TPI 不接受混杂提交批量/TPB。参考表为 [stage1_cost_4060_2000mhz.csv](../../config/stage1_cost_4060_2000mhz.csv)，不自动启用。

用户表 N 明确是容器位宽。实际目标按 **target_bits+6** 选择 PARAM0 容器及 TPI；8192-bit容器覆盖实际位数≤8186，实际8192 bits进入9216-bit/TPI32。表的条件是4060 Laptop、2000 MHz、B1=110×10⁶，TPI8提交768曲线、TPI16提交384曲线，统一提交2 blocks/SM；不等于全部容器同时驻留2 blocks/SM。算法、指数模式及独立核验未给出，来源是用户报告。

初始近似 T1∝B1·container_bits²/f。归一化系数在同 TPI 内插值；范围外邻近锚点外推，无同 TPI 数据时允许跨档外推并标注。`stage1_cost_mhz=0`采用锚点参考频率，正值使用指定目标频率，不读取瞬时时钟。跨 TPI、高 B1 与不同功耗尚未独立校准。

完整 Stage1 TOML采用邻近容器锚点，线性缩放 B1、平方缩放容器；保留 batch、模数类型、指数模式匹配，不要求 B1 相等。预计算工具见 [Stage1成本工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。

## 完整 Stage2 tune

`--tune ecm --tune-level 1..10 --tune-file FILE.toml`生成完整候选网格，预算内测少量完整引擎锚点及 NTT。默认 `stage2_tune_budget_seconds=1800`秒；初次设备查询在预算外，点准备、预热、正式测量在预算内。软预算到达后不发起新测量，正在执行的测量结束并保留有效结果。

每级额外包含512 bits。位宽 a:b表示b、2b、…、ab；B2 a:r从上界 U=2.6×10¹²按 r 倍递减取 a 项，再升序排列。非零 `--auto-max-b2`可改变 U。

- level1：位宽3:4096，D4项，B2 3:10，正式重复2次。
- level2：位宽4:3072，D6项，B2 3:10，正式重复2次。
- level3：位宽5:2048，D8项，B2 4:10，正式重复2次。
- level4：位宽5:2048，D10项，B2 4:10，正式重复2次。
- level5：位宽5:2048，D10项，B2 6:5，正式重复3次。
- level6、7：位宽8:1536，D12项，B2 6:5，正式重复3次。
- level8：位宽10:1024，D12项，B2 8:5，正式重复3次。
- level9：位宽15:1024，D12项，B2 8:5，正式重复5次。
- level10：位宽15:1024，D14项，B2 8:5，正式重复5次。

D在 `30030,60060,120120,180180,210210,360360,570570,690690,810810,1021020,1141140,1381380,1711710,2282280` 中均匀选取并含端点。P随D确定，不是独立参数。

`--tune-exponents bits,...`在此入口表示**实际位数**；`--tune-d`、`--tune-b2`、`--tune-repeats`覆盖对应网格。默认构造精确位宽的 GMP probable prime，sigma26、B1=20，不称为已证明素数。`--tune-save FILE`使用验证后的真实点并替代位宽列表；比较承载须同时给 `--tune-carrier-exponent p`并检查整除。旧 `--tune-tail-samples`只接受0。

完整测量先预热，再正式采样，要求 mandatory自检、GMP检查、clean通过、bad=0、hits=0。预算中途结束保留已完成正式重复及真实次数。优先缺项、未校验、误差超限或重复不足的候选；预测超过剩余预算的长任务暂不启动。再次 tune 复用摘要，模型网格按新增证据更新。

### NTT 与配对阶段组合

特征分别计数 NTT形状/调用、系数归约、传输、baby/giant点和积累。多G驻留请求按原生 packing 分块；giant区分chain、ladder、短尾。普通归约用limb²、梅森归约用limb初始工作量。NTT缺项以邻近长度/批量的N·log₂N比例估计，不假设NTT本身平方增长。

完整曲线的互斥阶段提供秒数尺度。giant、G树、fold嵌套计时仅用于权重，不与包含它们的主循环重复相加。同算术/位宽/D/内存形状的已测B2区间可插值；区间外走组件估算。

稀疏数据尚不能识别所有系数。当前多项式阶段权重为NTT0.55、归约0.40、传输0.03、调用0.02，是初始估算，不是独立拟合的硬件定律。未校验相对余量默认25%，随形状距离增加；有校验证据时使用实际误差并保留估算余量。

`validation_max_relative_error=|预测−实测中位数|/实测中位数`来自纳入新样本前形成的预测。无独立校验时不写零误差。默认8%门限用于补测优先级，允许未收敛模型参与生产并明确输出状态，不保证全网格达标。`independently_validated`表示该形状有独立误差证据，不等同误差低于门限。

## 启动短校准

默认 `stage2_short_calibration=1`，独立预算10秒。没有合适文件、必要NTT组件缺失，或当前worktodo范围完全不在可复用已测位宽覆盖内时触发；部分覆盖时不因首项宽度外推反复校准。直接读save以该目标作为单项范围。

在已验证目标上重新生成sigma26/B1=20有效点，测小D/B2完整引擎及少数NTT点；所需承载缺摘要时补测。结果供后续tune利用，不改用户save、不消费队列，不代表生产B2精度验收。plan-only也可能触发校准并更新性能文件。`--short-calibration 0`可关闭；关闭后固定B2缺文件可走原生规划，Auto B2仍要求可用成本和T1。

## NTT tune

`--tune ntt --length-log2 a:b --tune-slices s,... --tune-repeats n --tune-file FILE.toml`测长度和批量，也可用等级预设。共享软预算和增量摘要规则。每秒域卷积数=batch/median，每秒批调用数=1/median，不是ECM曲线吞吐，不含大整数carry、归约或完整Stage2。

GMP参考验证每个输出项；摘要保留中位数、MAD、重复数及检查量。内存不足可跳过，验证失败报错并保留证据。TOML用格式5，显式JSONL调试输出保留逐次接口。

## 汇集预计算结果

格式5只含profile、condition、sample、ntt的最终标量摘要，不含采样数组、逐轮运行或导入历史。保留来源、条件、离散度、检查量和独立误差。旧ECM2/3/4、NTT2先经原严格reader，再导入摘要。

同条件同候选仅在正式中位数**严格更低**时替换；相等或更慢保持原记录，不同条件分别保存。估算不覆盖实测；模型仅在证据变化时更新。复用忽略项不把不同条件的秒数混入同一原始最快记录。写入用独占锁，临时文件重新校验后原子替换。

合并：`--tune ecm --tune-merge A.toml --tune-merge B.toml --tune-ntt-profile NTT.toml --tune-file combined.toml`。不运行曲线或查询GPU。只有NTT的文件尚无完整引擎锚点，需要短校准或ECM tune。

原始日志、计划、回执、二进制SHA256与可用构建清单位于工作目录的`data/experiments/`，不写入性能TOML。队列身份用性能文件路径，增量tune不重置已完成编号；Stage1成本输入身份另行保护。

## 验证范围与限制

sm89/CUDA13.3/GMP Zen3，二进制SHA256 `1fb0a06e500a34232dc673669fc5768ca542bb2899df20649887f3386f59af39`，4060 Laptop设备1。未修改时钟/功耗，未采完整负载频率；以下是功能检查，不是速度提升报告：

- CPU概率377组对拍，最大绝对误差3.44×10⁻¹⁵；10级网格、最快更新、5类异常profile、CSV推断及旧ECM导入通过。
- GPU18项流程通过：512/4423 bits、D210、B2=60000、正式重复2次的驻留/fold预算0测量，跨B1=20→1000、短校准及关闭、NTT摘要/合并、一条4423-bit完整曲线、部分覆盖队列和M503余因子318 bits承载推断/锁定。
- 证据：`data/experiments/stage2_portable_tune_20261010/{probability_01,core_11,gpu_06}/`。进程墙钟独立记录，不与引擎阶段相加。

稀疏跨位宽/跨B2预测已出现显著误差，早期功能样本最高约343%和211%，原始摘要保留。不能作为8%收敛、生产收益提高或最优D证明。待完成孤立组件校准、非负系数拟合与生产B2/高位宽独立排名验证。

G=1与baby预算回退缺完整联合生命周期，成本可估算但原生准入不放行。host fold/frontier采用受限arena的保守上界；部分借用设备NTT的cap失败回退尚未完成执行验证。见 [内存](MEMORY.md)及 [TODO](../TODO.md)。旧`.cprof`接口保持独立资格合同，不套用上述模型。

## 代码入口

- [ecm_stage2_portable_driver.inl](../../src/core/ecm_stage2_portable_driver.inl)：短校准、候选、测量、合并与发布。
- [ecm_stage2_portable_model.h](../../src/core/ecm_stage2_portable_model.h)：工作量、成本、插值与误差余量。
- [ecm_stage2_tune_portable.h](../../src/core/ecm_stage2_tune_portable.h)：网格、格式与更新。
- [ecm_stage1_cost_csv.h](../../src/core/ecm_stage1_cost_csv.h)、[ecm_stage2_probability.h](../../src/core/ecm_stage2_probability.h)：容器模型及概率。
- [ecm_cuda_stage2_main.cpp](../../src/core/ecm_cuda_stage2_main.cpp)：配置、验证与队列。
- [配置参考](../ECM_INI_REFERENCE.md)、[使用](../usage/STAGE2.md)、[工具](../../tools/bench/README_STAGE2_CARRIER_PLAN.md)。
