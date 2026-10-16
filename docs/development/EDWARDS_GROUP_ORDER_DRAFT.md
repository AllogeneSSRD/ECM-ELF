# Edwards 群阶与点阶分析草稿

临时开发记录，保留待用户处理。目标是用PARI/GP重建本仓库及Prime95的Atkin-Morain Edwards曲线，计算并分解完整群阶和初始点阶，使用用户提供的17组mersenne.ca数据独立验证。

## 2026-10-16：范围与数学入口

现有数据集的两表schema及最优sigma字段专属于Montgomery PARAM0，指数入口限制为9999；不能把本次Edwards样本写入这些字段。新增独立GP及分析入口，支持完整uint64 sigma和本次大指数，保留每个样本的原始GP输出及模型身份。

在模素数p上由sigma构造 `x²+y²=1+d·x²y²`，再经 `u=(1+y)/(1-y)`、`v=u/x` 映射到 `Bv²=u³+Au²+u`，其中 `A=2(1+d)/(1-d)`、`B=4/(1-d)`；使用 `X=B·u`、`Y=B²·v` 得到PARI支持的Weierstrass模型。保留B可直接映射给定基点，无需另选平方根；一般Edwards曲线省略B可能改变扭转身份，本Atkin-Morain族的B则可证明为平方，见正式数学说明。

群阶使用ellcard独立计算，再分解；点阶使用ellorder并逐个素因子检查最小性。用户表格按行配对保存为输入证据，尚不预设其“GO”一定等于完整群阶。当前正在验证输入整除关系、素性、Hasse界、群阶与点阶对应关系及现有C++构造的一致性。

## 2026-10-16：完成与验收

独立GP入口及Python CLI已完成，支持uint64 sigma、可选梅森整除核对和批量fixture，不修改PARAM0数据库。输出完整群阶/原始点阶及分解、48LCM下的有效点阶和幂光滑前沿。主程序 `--go`、数据库扩展与Edwards扫描不在本阶段实现范围；使用方法统一见[正式说明](../usage/DATASET.md#edwards群阶与点阶)，数学与模型身份见[固定参考](../reference/ECM_EDWARDS_MATH.md#41-在有限素域计算群阶与点阶)。

PARI/GP 2.17.3对17行数据均完成素性证明、独立ellcard、点阶最小性及完整分解。15行与给定完整群阶一致；M13883给定乘积少一个因子3，完整群阶为 `190998088435897266796654466340963036240192`；M15173给定群阶及所有分解项由Montgomery PARAM3精确复现，与同sigma的Edwards结果不同。用户原始输入仍原样保留，`complete`不代替外部参考匹配标记。

最终原始采集为 `data/experiments/edwards_group_order_20261016/reference_03/`，验证为 `validation_02/`，CLI检查为 `cli_01/`。17组d/x/y与直接调用 `src/core/ecm_edwards_curve.h` 的CPU/GMP探针完全一致；独立Python仿射算术核对每个原始点阶及去掉每个不同素因子的非零性。24组小素域/不同sigma组合包含完整uint64上界，22组有效并通过逐点群阶计数，2组双方判退化；6种无效输入及5种缺失证据被拒绝。直接单曲线、PATH/bare GP、复合模数失败状态和已有目录拒绝覆盖均通过，PARAM0数据库摘要前后一致。

汇总和orders.csv为 `data/benchmarks/edwards_group_order_20261016/qualification_01/`；GP、libpari、探针/GMP、源码及日志摘要均有记录。较早采集及验证保留于同实验根目录；正式结论以最终目录为准。正式使用说明、入口索引、TODO及群阶/点阶数学口径已同步；本临时草稿保留待用户处理。
