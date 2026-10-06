# Auto B2 多端点标定与 giant chain 阈值实验

日期：2026-10-06。上一阶段 [原生接入](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_NATIVE.md) 只发布了 M8191 范围。本阶段使用同一4acc二进制重新采样，**2203、4423、8191 bits 的六个范围全部通过新的独立盲测**；另完成 giant chain 的同二进制性能与点坐标对照。

本阶段没有将长期目标标为完成。生产 B1、低 B2/G=1 的 Auto B2 成本、精确树成本、总显存租约、并发和后续 NTT 优化仍须推进。

## 1. 可用范围及命令

可执行文件保持 `build_cuda_cmake/_auto_b2_native_20261005/native/ecm_cuda_stage2.exe`，SHA256 `4acc15d26593eb9d8e66ff8f62ad8a64e12a18b04a4d01966dc1399580c4d9e2`。本阶段没有重编译或修改该二进制的算术源码。

新运行 profile：[gpu1.cprof](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_multianchor_20261006_gpu1.cprof)，SHA256 `fbe6613b82ba3685a55101f2b9c94e3730a515e607579187b3ac7551262e6ffd`。精确输入 M2203/M4423/M8191，B1=1000、lcm，Stage1 batch1/12，B2=30亿～60亿，D=30030/60060/120120，arena=4096MiB，驻留与预算回退两路径。P/G仍受每个scope的实测界限约束；不是任意位宽、B1、D或GPU可通用的 profile。

```powershell
$exe = 'build_cuda_cmake/_auto_b2_native_20261005/native/ecm_cuda_stage2.exe'
$profile = 'docs/data/ecm_auto_b2_multianchor_20261006_gpu1.cprof'
$save = 'build_cuda_cmake/_auto_b2_multianchor_20261006/study/m4423.save'
& $exe --save $save --device 1 --auto-b2 --cost-profile $profile --plan-only
& $exe --save $save --device 1 --auto-b2 --cost-profile $profile `
  --stage1-batch 12 --results run/auto_b2/results.jsonl --log run/auto_b2/screen.log
```

INI/队列语义沿用原生接入报告，只替换 `stage2_cost_profile` 路径；显式非零 B2 保持固定。新profile仍绑定GPU1 UUID、SM、runtime/driver=13030、PTX3/outer0/accounting2、factor-only和准确binarySHA。命名/GMP拆解及预算参数规则保持。

原生规划中，batch1的M2203选择B2=3459335879/D60060，为搜索区间内的整数树边界邻点；M4423/M8191仍选择下界30亿/D60060。Stage1进程摊销分别约1.702384/1.036335/1.062569秒，batch12成本另有实测行。cold和Stage1进程样本波动明显，用户可传实际 `--stage1-seconds-per-curve`。**`range_limited=false`只表示没有选中区间端点，不证明全局最优。** 当前树工作特征还存在下述已识别的近似误差，边界附近实际排名需补对照。

## 2. 诊断及采样协议

新增 [交叉复测工具](D:/code/MPA-OpenCl/tools/bench/diagnose_ecm_cost_drift.py:45)：同4acc、两个小位宽、D60060/120120、30/45/60亿、owner640/0、各两次，共48条曲线。每个重复块按固定seed随机交叉，保留全程NVML和系统CPU负载观察。

大多数配置的两次full时间相差0～2.5%左右，但旧模型仍持续高估约10～37%。用本批30/60亿端点诊断拟合，45亿中间点偏差约5%以内。这说明重新覆盖B2依赖与同一运行状态下的率有帮助；没有证明旧数据漂移的单一原因。

随后 [标定工具](D:/code/MPA-OpenCl/tools/bench/measure_ecm_costs.py:29) 增加：

- `--train-b2` 多个拟合端点；
- `--shuffle-seed` 固定随机交叉顺序；
- `--monitor-state` 在实际子进程计时期间采样，CPU参考工作在计时之外；
- 原有resume继续核验二进制、工具、输入和控制参数，修改配置需新目录。

新实验：18批Stage1、117保存点独立参考核验；72条Stage2拟合（30/60亿 × 三宽度 × 三D × 两路径 × 两重复）；12条37.5亿留出检查。再冻结模型与预测，运行 **36条52.5亿盲测**。之前45亿数据仅作为诊断，不复用为新独立验证。

部分训练样本有descent等阶段的秒级等待：例如M4423/D60060/B2=60亿驻留，full=2.638165与5.606842，descent=0.248与3.182秒；M8191/D120120/B2=30亿回退，full=10.188397与14.165960，descent阶段出现约4秒差异。算术、阶段工作量及oracle检查保持一致。样本全部保留，没有因误差大而删除。

观察中的NVML值也可能异常，功耗有少数明显不可信读数，未据此得出因果结论。采集时旧字段 `sm_mhz` 使用的是NVML clock type0（graphics）；冻结源码随证据保存，后续采样器改为分别读取graphics=0与SM=1。系统CPU负载是整机区间观察，不是曲线进程CPU占用，也不直接解释CUDA等待。

## 3. 稳健估计及独立门限

[拟合器](D:/code/MPA-OpenCl/tools/bench/fit_ecm_costs.py:17) 增加可选 `--estimator phase_medians`。对某阶段的相同工作特征值取重复样本中位数，再做原有非负率拟合；inverse/descent等在相同P、不同B2端点拥有重复观测，可避免单次外部等待被解释为全部形状的MAC成本。giant按chain工作、chain分块和ladder工作组合分组。原始样本不被修改，模型附原始训练误差。

这估计的是**典型阶段成本**，不是平均长尾耗时、P99或时限保证。普通最小二乘拟合另行保存用于比较；发布模型和盲测前确定使用phase_medians，不在看过盲测后择优更换模型。10%预测、5%排名门限保持。

CPU回归先复现缺失稳健估计接口，然后实现：两个测试通过。构造固定工作量的一次100倍等待，普通拟合率明显增大，而重复特征中位数恢复原率；无污染样本保持原率。见 [回归脚本](D:/code/MPA-OpenCl/tools/test/test_ecm_cost_fit.py:26)。

新盲测结果：

- M2203驻留/回退最大绝对误差5.085% / 5.036%。
- M4423驻留/回退4.435% / 5.279%。
- M8191驻留/回退6.986% / 4.255%。
- 总误差范围−6.986%～+5.279%，六个scope均发布。
- M2203/M8191选择本批最快D60060/驻留；M4423选择D60060，比本批实际最快D120120慢1.893%，小于5%门限。没有宣称三个宽度全部精确选中最快。

120条标定/留出/盲测曲线全clean，761568个GMP抽样系数，30组同几何跨路径/重复叶哈希一致。随后原生验收37调用、255断言和8条真实GPU曲线通过，三个宽度auto/manual、INI与队列结果分别一致。详见 [审计](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_multianchor_20261006_audit.json) 和 [原生验收](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_multianchor_20261006_acceptance.json)。

## 4. giant chain 的实际性能对照

[实验工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_chain_threshold.py:21) 使用同一4acc、相同有效Stage1保存点、D120120、B2=10亿/30亿、三个宽度；每个配置按ladder/chain/chain/ladder交叉，共24次性能测量。强制ladder使用极大 `NTT_GIANT_CHAIN_MIN`，强制chain设0，block保持64；两端保留算术门禁。

另6条chain运行开启 `NTT_GIANT_CHAIN_CHECK=1`，逐点用ladder对比仿射X/Z，8327和24977点 × 三宽度，**99912点失配0**。这些完整点对照不计入性能样本。两路径projective代表与叶哈希可以不同，不能以原始projective X/Z或叶哈希相等作为门禁；相同配置raw因子集合相同，proper divisor及GMP/NTT检查通过。

每模式两次测量的giant中位数：

- M2203：10亿0.1595→0.0655秒（−58.93%）；30亿0.4125→0.068秒（−83.52%）。
- M4423：10亿0.5900→0.2465秒（−58.22%）；30亿1.5395→0.262秒（−82.98%）。
- M8191：10亿2.2960→1.5520秒（−32.40%）；30亿5.4570→1.5995秒（−70.69%）。

整段full的两样本中位数减少约10.46%～48.46%。M4423 ladder样本有较大长尾（例如30亿3.423571/4.662099秒），所以整段百分比受它影响；没有置信区间，不把48.46%当作普遍可复现的加速。相比之下giant本身的差异较稳定。M8191/30亿full=9.844801→5.928964秒（−39.78%）。

这证明原32768点阈值对本批8327/24977点过于保守，符合GPU seed/6MAC xADD之后应重新测拐点的预期。尚未测得最小拐点，也没据此更改全输入默认。需要补更小点数、阈值边界和退化/因子点，然后将策略接入成本profile身份与原生规划。

当前可在相同实测条件下手动复现，例如：

```powershell
$env:NTT_GIANT_CHAIN_MIN = '0'
& $exe --device 1 --save $save --b2 3e9 --d 120120 --arena-mb 4096 `
  --factor-only --results run/chain/results.jsonl --log run/chain/screen.log
Remove-Item Env:NTT_GIANT_CHAIN_MIN
```

该实验开关与当前32768阈值的Auto B2 profile不兼容，auto会拒绝配置冲突。要启用自动选择下的新策略，必须标定新策略并保存其身份。

## 5. G=1 与树成本的两个明确后续修正

本批B2=10亿/D120120为 **G=1**，实际Stage2和chain/ladder点门禁已运行通过，但还不属于Auto B2发布范围。

G=1没有外层fold；[预先inverse](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9705) 仅在loops>0时计算。可是 [scaled descent](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:7842) 在没有cached inverse时自行构造逆元，长度P。因此不能简单把G=1的inverse工作全部删去；需要重新划分阶段，并区分P与P+1的真实packing。驻留fold owner在G=1也没有同样的活跃条件。

其次，旧树特征 `floor((n+h)/(2h))` 会对部分单边节点多算一次乘法。真实 [设备G树](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:4474) 在一侧degree=0时只复制；有效乘法对数为：

\[
c(n,h)=\lfloor n/(2h)\rfloor + [n\bmod(2h)>h]
       =\lfloor(n+h-1)/(2h)\rfloor,\qquad h=1,2,4,\ldots<n.
\]

单棵树合计n−1次乘法；G批树为I−G。本批84条runtime `gdevice.pairs`全部等于此公式。例如M4423/D60060/B2=30亿，实际49943、旧特征计数49952；D30030实际99867、旧特征99902。该证据包含在 `tree_census`。

当前成本率是以旧经验特征拟合，不能只替换公式却继续使用同一profile。下一版同时修正Python与原生树特征、处理G=1，并对新binary/模型重新标定和盲测；还需计入复制/分组/launch成本，不能把复制的成本当作零。M2203当前内点选择尤其需要树边界邻点验证。

## 6. 复现与证据

```powershell
python tools/bench/measure_ecm_costs.py --stage2 $exe `
  --save-dir build_cuda_cmake/_auto_b2_native_20261005/study `
  --output run/new_cost_study --train-b2 3000000000 6000000000 `
  --holdout-b2 3750000000 --shuffle-seed 20261006 --monitor-state
python tools/bench/fit_ecm_costs.py --measurements run/new_cost_study/measurements.json `
  --output run/new_profile.json --estimator phase_medians
python tools/bench/validate_ecm_costs.py --profile run/new_profile.json --stage2 $exe `
  --save-dir run/new_cost_study --output run/new_blind --b2 5250000000
```

```powershell
python tools/test/test_stage2_auto_b2_native.py --exe $exe --profile $profile `
  --model docs/data/ecm_auto_b2_multianchor_20261006_profile.json `
  --save-dir build_cuda_cmake/_auto_b2_multianchor_20261006/study `
  --expected-bits 2203 4423 8191 --output run/three_width_acceptance
```

GPU状态采样器及异常清理在采集完成后做了小修正；当时6个测量工具的原始源码/SHA已经冻结，随证据保存。成本数学模块和二进制保持与本轮profile一致。后续复测会记录新工具SHA，不能冒充完全相同的采集工具。

[模型](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_multianchor_20261006_profile.json)、[完整原始数据/日志/冻结工具](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_multianchor_20261006_evidence.json)、[chain对照结果](D:/code/MPA-OpenCl/docs/data/ecm_giant_chain_crossover_20261006_results.json)。新profile保留精确device/binary门禁；本阶段完成的是三宽度可靠范围扩展和性能候选验证，生产B1、G=1、默认chain策略、并发/lease和后续NTT优化继续推进。
