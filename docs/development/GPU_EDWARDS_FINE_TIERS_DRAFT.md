# GPU Edwards 细分实例与并行构建草稿

临时开发记录，完成后保留待用户处理。正式状态维护于 Stage1 架构、构建与性能说明。

## 2026-10-16：范围与实现

目标是为 Edwards binary/NAF 补齐 Montgomery PARAM0/3 的容器网格，并保留现有17408档上限。Edwards继续要求目标位数小于容器位数，Montgomery继续预留六位；不为统一图表而改变算术约束。32档为128/192/256/384/512、768..2048间隔256、2560..8192间隔512、9216..17408间隔1024。

共享实例清单驱动两个模式的分派和七组CUDA文件；`ECM_TIERS`限制实际编入的实例，未编入时选择更大可用容器，无足够容器时报错。切片、检查点与标量语义保持原接口。构建采用现有并行nvcc引擎，增加共享头依赖跟踪，并记录实际队列墙钟与各TU起止，避免把最长单个TU当作总构建时间。

用户明确要求内核位于 `kernels/cuda/`，核函数与实例化文件均已置于该目录，`src/cuda/`只保留接口、校验、分派及主机驱动。首次编译发现NAF头先于点公式头包含，已修正顺序；默认沙箱内首次配置停滞及失败编译日志保留于 `data/experiments/edwards_fine_tiers_build_01.log`、`edwards_fine_tiers_build_02.log`。完整本机工具链重建输出为 `edwards_fine_tiers_build_03.log`，不以失败轮次的对象作为Edwards验收依据。

待验证：全32档CPU末点、部分block、停止/恢复、旧检查点跨容器恢复；完整worker和Stage2交接；GPU1上同exe的Edwards NAF、Montgomery PARAM0、PARAM3三曲线族性能。正式测量采用既有六个位宽、B1=10000、批量768、三次正式样本与一次排除的预热，保存构建和原始日志身份。未经测量不声明加速比例。

## 2026-10-16：实现与构建阶段门禁

完整sm89构建通过：exe SHA256=`fa8b903e9186dada7055e61176fff237411b575c4707a04541f5cc50e15ad3eb`，25,714,176字节，21个cubin、无PTX；资源转储确认全部32个Edwards `(容器,TPI)` 对。证据为 `data/experiments/edwards_fine_tiers_20261016/build_identity_01/` 和 `full_resources.txt`。编译轨迹确认最多8个进程重叠，七组Edwards实例均独立编译；时间轨迹仅作构建诊断，未采集CPU功耗，不作为串行/并行性能对比。

384单档构建只含该档GPU实例，三个模式（binary、NAF HWCD、NAF 2XY）各33曲线的CPU末点、17/113切片停止恢复均通过，共99条组合；同时核对缺档回退和无足够容器时的查询结果。原始日志、资源及清单为 `restricted_384*`。独立探针复用CMake对象时暴露运行库不一致，已使编译与链接显式使用 `/MD`，保留 `probe_link_01.log`。最终探针构建采用 `probe_build_03.log`。

边界测试输入生成与算术验收分离，同位宽的三个模式复用同一边界整数。首次探针轮次 `validation_01` 因重复素数生成而中断，原始输出与二进制保留，不标完成；当前轮次为 `validation_02`。实现阶段提交时，128..16384的31档边界算术及恢复已与CPU一致；17408档、原有梅森/复合目标用例及完整流水线仍在运行。性能正式样本数仍为0。正式状态与验收范围将在完整门禁和性能采集后同步。

## 2026-10-16：完整正确性门禁

最终验收为 `data/experiments/edwards_fine_tiers_20261016/validation_03/`，主worker仍为上述 `fa8b903e...`。131组探针覆盖全部32档、537条逻辑曲线组合，全部核对CPU投影末点、GCD、点不变量，以及17列停止/113列继续的逐坐标一致性；包含原有梅森/复合输入、多窗口、两种NAF倍点和部分block。pipeline 29、边界9、恢复23及小素数9共70条CLI检查全部通过，包含旧binary V1和旧七档NAF V2恢复；V2的M677实际从1024档恢复到768档，归一化记录一致。Stage2交接与叶值独立核对通过。

边界补充用例使用接近容器上界的奇数模数，只按曲线构造 `Setup::Ready` 选择输入，不依据GPU或点乘结果筛选；CPU参考使用现有实现的w2，保留全部末点/GCD比较。素数搜索与默认CPU w12字典对短标量验收没有必要，故 `validation_01/02` 的未完成输出保留，最终完整范围以 `validation_03` 为准。最终probe构建日志为 `probe_build_04.log`，SHA256=`c1b17d070be375a8858c7dc098fd5cdaaf50321b47dbb92ef7a615bcd7ab5f9f`。

最终 `/MD` 单档入口再次通过384档的3组、99条逻辑曲线组合，证据为 `restricted_384_final.log` 和 `final_restricted_manifest.json`。汇总 `data/benchmarks/edwards_fine_tiers_20261016/qualification_01.json` 核对所有清单/日志摘要；原始工作程序、探针及各自GMP运行库在 `validation_03/bin/` 独立冻结。完整三曲线族性能矩阵另行采集，不把正确性运行的时间计入正式性能。

## 2026-10-16：三曲线族性能与图表验收

阶段目标已完成：32档实例、七组内核CU和8路并行构建经过完整正确性门禁；同一完整worker在GPU1上完成Edwards NAF w6/HWCD、Montgomery PARAM0和PARAM3的六个位宽对照。输入为M347/M677/M991/M2203/M4003/M8191、B1=10000、768条/批、sigma=20260922；18个预热排除，54个正式完整批次全部保留，原始证据为 `data/experiments/edwards_fine_tiers_20261016/bench_01/`。新旧两版六个位宽的4608条Edwards归一化记录逐字节相同，见 `data/benchmarks/edwards_fine_tiers_20261016/endpoint_identity_01.json`。

审计汇总为 `data/benchmarks/edwards_fine_tiers_20261016/comparison_01/summary.json`，SHA256=`158da9ec62537d18a2a59f9fb0ee3c8b2d48e9420d32aad0e961640fdd0f8665`；数据、计时边界、硬件功耗及结论统一维护在[正式性能说明](../performance/STAGE1.md#gpu-edwards-细分实例三曲线族对照)。M8191完整进程中位数为NAF 16.6265秒、PARAM0 20.3712秒、PARAM3 16.5450秒；前五个位宽的NAF/PARAM3成本比为1.423…1.719，短任务有明显新进程离散，不能宣称统一领先或细分实例的独立加速比例。

图表由现有绘图入口的 `--comparison` 模式生成到 `data/figures/edwards_fine_tiers_20261016/comparison.png`、`comparison.svg`，身份见 `plots_01.json`。`plot_validation_01.json` 核对54个原始样本、两面板108个观测点、CSV和正式文档数值；PNG/SVG可解析、重生成逐字节相同。私有内存中的五类错误元数据（缺样本、错误exe、错误wall、错误Edwards/Montgomery几何）均被审计拒绝，原始证据未修改；旧144样本审计回归与冻结汇总除审计脚本身份外一致。

正式架构、构建、性能说明和TODO已同步。后续生产B1/B2、其他设备、常驻队列、批量和TPI/窗口标定仍待独立验收。本临时草稿保留，待用户决定如何处理。
