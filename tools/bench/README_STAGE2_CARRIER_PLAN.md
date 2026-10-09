# Stage2 carrier / memory planning analysis

用于比较目标余因子 `N` 与梅森承载模数 `M=2^p−1` 的几何代价，以及不同D/P在
现有布局下的显存下界。只做离线计算，不运行ECM或CUDA。

研究结论与实现设计见
[研究报告](D:/code/MPA-OpenCl/docs/STAGE2_MERSENNE_CARRIER_MEMORY_PLAN_20261009.md)。

## 使用

在仓库根目录运行，Python 3；分析脚本使用标准库及同目录
`calibrate_stage2_d.py` 中的整数 `phi/shape` 计算。绘图需要matplotlib和numpy。

```powershell
python tools/bench/analyze_stage2_carrier_plan.py --output data/stage2_carrier_research_20261009
python tools/bench/plot_stage2_carrier_plan.py --input data/stage2_carrier_research_20261009/analysis.json --output-prefix docs/figures/stage2_carrier_plan_20261009
```

指定输入与预算：

```powershell
python tools/bench/analyze_stage2_carrier_plan.py --bits 7995 --carrier-bits 8011 --b2 2600000000000 --d 690690 810810 1021020 --free-mib 7106 --reserve-mib 768 --arena-mib 6300 --fold-mib 640 --output data/stage2_carrier_custom
```

- `--bits`：目标N位宽；默认7995。
- `--carrier-bits`：承载位宽p；默认8011，必须大于等于目标位宽，最大16384。
- `--b2`：正整数；默认2600000000000。
- `--d`：一个或多个不小于6的偶数；默认报告中的7个候选。
- `--free-mib`：默认7106，来自历史日志，**不会查询当前GPU**。
- `--reserve-mib`、`--arena-mib`、`--fold-mib`：默认768、6300、640。
- `--gpu-analysis`：可选的已完成位宽扫描分析JSON。未指定时，若本地存在
  `docs/benchmarks/stage2_n_scaling_20261008_analysis.json` 则自动使用；不存在
  仍可计算本次几何，省略跨位宽比较图。
- `--output`：结果目录，必填。

本工具接收位宽，不能验证实际整数的整除关系。生产启用承载前必须另行验证 `N|M`。

## 输出口径

`analysis.json`：输入预算、源码SHA256、NTT临界P、候选数据；`measured=false`。

`geometry.csv`：每个目标/承载位宽、D组合一行，主要字段：

- `P/I/G`：baby多项式度数、巨点数、G树数，使用现有GPU覆盖公式。
- `fold_length/packing_bpw/slot_words`：fold的NTT长度、digit位宽和每系数digit数。
- `nominal_tree_length/padded_tree_length`：名义P/2子树与补齐后实际最大子树的NTT。
- `legacy_arena_mib`：现有保守arena估计，便于与共享池比较。
- `fold_big_mib`：单次最大fold所需A/B/Q池 `24L`。
- `owner_mib/legacy_owner_mib`：当前reuse=3与默认reuse=0布局的占用。
- `raw_g_mib/coord_mib`：compact G raw A/B与当前256MiB向上取整策略的坐标chunk。
- `concurrent_lower_mib`：上面四项并存的下界，不含表、S4输出、seed等。
- `owner_budget_fits`：当前owner布局是否在fold预算内。
- `legacy_arena_fits`：旧arena估计是否在arena预算内。
- `lower_fits_free/lower_fits_after_reserve`：下界是否低于输入free及free−reserve。

`carrier_cases.csv`：仅提供/找到既有分析时生成。原D不变，比较输入N与原梅森p
的fold长度和owner变化。`ntt_length_ratio`不是速度比。

PNG/SVG：候选并存下界、G数量随P变化，以及可选的跨位宽NTT长度比。

绘图可加 `--canvas <绝对路径.canvas.tsx>` 输出自包含交互看板；全部数据内嵌，
可切换目标/承载位宽，不联网。

## 适用限制

- 同时存活下界仅建模 `I>P`、当前compact raw、resident owner reuse=3和当前巨点
  chunk策略；其它运行策略需要独立生命周期模型。
- 大池只计算最大单fold形状；若某个批量树调用的 `L×batch` 更大，实际峰值会增加。
- 下界超预算可以排除当前布局；通过预算不能证明运行时能申请成功。
- 生产 `ecm_cuda_stage2_shape_query` 仍为最终形状依据；源码策略更新后应同步公式。
- `data/`、`docs/figures/` 按仓库规则不提交，保留脚本即可重建。
