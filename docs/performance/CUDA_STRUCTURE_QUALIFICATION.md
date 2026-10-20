# CUDA 模块验收

本文记录0.44.6程序的位宽、构建、行为与性能覆盖。算法、配置默认值、存档协议和输出语义保持；主程序源码仅增加修订号。生产模块职责见[架构总览](../architecture/OVERVIEW.md)、[Stage1](../architecture/STAGE1.md)、[Stage2](../architecture/STAGE2.md)与[NTT](../architecture/NTT.md)。

验收范围为本机LOCAL编译、完整位宽功能与性能。完整位宽与行为门禁通过，9个代表样例的配对性能检查通过。多架构发布按用户要求暂缓，部分构建证据保留，不构成完整发布矩阵或其他实体GPU的验收结论。

## 程序与环境

完整重构的对照基线为0.44.0、源码`36171f43f5737d525d7b5d013bd9b04d0131468c`，使用预先冻结的完整程序。验收根目录为`data/experiments/cuda_structure_p5_20261020/`，`baseline/identity.json`绑定程序、匹配构建记录和源归档。保留原始失败、日志、存档与采样，不覆盖失败轮次。

当前功能程序为LOCAL、CUDA13.3、MSVC14.51.36231、sm89、Zen3 GMP；生产算术默认保持PTX GL backend、fixed mode3、add/sub mask1、outer unroll0。Stage1完整32档、PARAM2启用、TPB128、Mersenne fold关闭、PRAC默认关闭。共享位宽政策以[权威清单](../../include/ecm/cuda/tiers.def)为准，Stage1容器与Stage2容量保持不同语义。

运行仅使用GPU1：NVIDIA GeForce RTX4060 Laptop GPU、sm89、驱动610.88。主机AMD Ryzen AI9 HX370、32GiB、24逻辑处理器；不更改GPU功耗、频率或其他进程。驱动报告GPU1功耗上限不可用，按不可用记录，不写成0。设备与工具链依据为`devices.csv`、`toolchains.json`。

本机两张实体GPU均为sm89。性能验收使用GPU1，构建、镜像检查、兼容SASS、PTX JIT与目标原生硬件运行分别记录。

实际程序SHA-256：0.44.0 Stage1为`849a5a18a5a7d948c06cdcfa674f425afcfd07295aeacc24aa4cf101097ea2d8`，Stage2为`98fd5495707808db6b06aae69d57be6945e518c91157194fff1cef0b689323dd`；0.44.6 Stage1为`baaae49fc5cdee1e32ab5f6fa81506dae4c2a386167545000e99aa476c15b078`，Stage2为`8ff2b546c0d3e9c1e17eb730bb27a88880762834048a1570efe7fe7a90408d76`。两代程序使用同一GMP DLL，摘要为`969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`；构建选项和工具链相等性见`profile_match/result.json`。

`final_local_identity.json`复核Stage1的196项源绑定和31个CUDA对象、Stage2的149项源绑定和25个编译对象，以及完整功能证据。程序构建时基于`5d15a93`工作树加0.44.6版本头；最终绑定以源摘要和实际程序摘要为准。


## 已通过的位宽与行为门禁

- Stage1实际工厂、余量、上下边界及TPI/TPB：260项检查。Montgomery全部31档各PARAM0/2/3，共93组、每组2曲线，完整末点存档与冻结0.44.0一致；只排除WHO和TIME。
- Edwards全部32档binary/NAF4/NAF5及窗口、复合输入扩展，共131组537曲线。独立GMP末点、点不变量、17/113停止恢复和自适应重启全部通过。
- 四种TPI分别补充33曲线批次，覆盖三种Montgomery参数化与Edwards三模式，共24组、每侧792曲线，完整存档相同。PARAM0/3与Edwards首末点另由独立整数参考核对；这些批次中的PARAM2以冻结基线为对照。几何查询同时验证实际IPB和TPB。
- Stage2实际W=1…256、通用/梅森模数和三种尾部边界，共1536例、152064次独立GMP比较，bad0。最小W=1的下边界使用有效的2位模数，其余字数覆盖1/63/64位尾部。七档容量均从真实生产算术对象调用；Montgomery radix使用实际W。
- 完整交接5组Stage1与5组Stage2，另有14组Stage2流程对照、4组规划和20组损坏拒绝。目标叶值、非计时公开结果和检查工作量一致，错误情况下不发布部分结果。
- 16种位宽编译配置、40项构建身份/故障门禁通过。32个生产模块清理、27项迁移/138个CUDA消费者和5个探针接口约束复核通过。
- CPU资源模型：初始树4556、owner6580、S4 299592、giant265638项检查通过，另核对对应分配事件、生命周期和错误尺寸拒绝；这些模型计数不是进程显存峰值。

完整功能绑定见`native/result.json`，包含程序、实际工作对象的探针清单、GMP、输入和各门禁输出摘要。额外批次依据为`macro_batches_02/result.json`。

受限开发构建（384/1024/1280、关闭FULL_BUILD与PARAM2）的28项注册及拒绝查询通过。Mersenne fold显式构建五档完成10组PARAM0/2完整点对照，并确认5种宽度的PARAM3拒绝及PARAM0/2非梅森输入拒绝，共7组，无存档发布。依据为`macro_dev/result.json`、`macro_fold_02/result.json`。折叠构建保持只支持PARAM0/2的既有边界。

TPB256/寄存器目标64的显式构建完成四种TPI、三种Montgomery参数化共12组对照，每侧780条完整存档相同。每组65曲线均超过实际IPB（64/32/16/8），覆盖自定义TPB下的跨block边界；工厂几何查询同时通过，见`macro_geometry/result.json`。构建并行度为1，不改变该显式配置的算法或生产默认。

## 指令与可选路径

`stage1_sass_nested/result.json`核对189个完整函数的指令正文，规范化NVCC匿名命名空间文件身份后无增删或指令差异。嵌套命名空间只替换两个已核对长度的文件哈希，保留文件名、模板几何和指令。`sass_identity/result.json`的10项CPU门禁验证指令/几何变化仍失败，符号碰撞及错误ABI长度仍拒绝。该结论仅限指令正文；cubin字节、调度编码和性能使用各自证据。

Stage2为176个函数，161个指令正文相同，15个算术函数不同，与拆分时记录的范围相同；严格比较器仍返回不一致，见`stage2_sass_nested/result.json`。差异包括Montgomery自检NW8…256、七档plain scale及NW128的baby/block product。全部容量有实际对象GMP覆盖，性能结论必须另行测量。

显式PRAC选档384/2560/4608/9216/16384单独构建，不进入默认生产性能结论。16组CPU/GMP、暂停恢复和双代损坏拒绝通过；46组策略完整点比较与4组非法策略拒绝通过，覆盖替代TPI32、168/255寄存器、128 single-compact和三种常量策略。依据为`optional_runtime/result.json`、`optional_policies_03/result.json`。

最高Montgomery GPU目标为16378位，Edwards为17407位。CPU标量MPN参考的固定容量为160 limbs、10240位，源码依据为[mont上下文](../../src/cpu/ecm_edwards_mont.h:55)。旧／新版在16378位均同样拒绝，相关CPU源码与0.44.0相同；超过该参考边界时，使用独立任意精度整数ladder并与冻结旧版GPU完整点交叉核对，包含LCM和12LCM两种已有模式。证据见`cpu_oracle_diagnosis/result.json`。未扩展或修改生产CPU边界。

显式Graph探针完成368次GMP卷积、14779392个数字检查，缓存捕获、重放、失效及释放通过；故障模式bad1并按预期退出3。另有5组真实曲线双版本对照和一组双版本错误拒绝，覆盖owner回退、单槽校验队列、较大树、合法承载及8193位输入。依据为`graph/result.json`、`graph_curves/result.json`。日志可能重复打印同一arena累计计数，不将记录之和当作独立调用总数。

已有M127活动点上下文的合成baby诊断仍同样拒绝；通用8193位输入上的该诊断通过。实验worker缺少曲线内存模型的限制保持，不能把独立NTT实验当作生产曲线验收。只用于合成操作混合、会产生错误结果的PROBE_CHAIN模式不属于生产正确性或性能矩阵；正式发布检查要求其关闭。

## 发布与硬件覆盖边界

多架构发布暂缓，本机验收不要求生成全部发布版本。已完成的sm60/70两阶段源/对象/程序/工具链身份、cubin/PTX策略和各8项隔离ZIP载荷检查保留在`release_legacy_static/result.json`；Stage2另有sm75/86/89构建及打包产物。完整六架构联合静态门禁和发布镜像运行均未完成，不能据此声明全部发布版本通过。中断范围和自有进程身份见`release_deferred_by_user.json`。

发布检查工具保留为显式入口：`test_cuda_p5_release.py`核对默认宏、工具链、源/对象身份、镜像和隔离载荷；`test_cuda_p5_images.py`要求完整发布静态门禁及匹配本机参考程序后，才执行sm89原生、sm86兼容SASS和可运行旧目标PTX JIT。`image_identity/result.json`的4项CPU负测试验证未完成本机门禁、参考替换、发布程序替换和架构重复均在GPU初始化前拒绝。本轮没有执行该发布镜像矩阵。

TODO：恢复多架构发布时，先完成六架构构建、联合身份与ZIP检查，再运行可用镜像；其他架构的原生算术及性能需要对应实体GPU和冻结程序。兼容或JIT结果不能代替目标实体GPU验收。

## 本机抽样性能

性能对照限定为相同LOCAL构建规则、GMP、设备及固定输入下的0.44.0与0.44.6。按用户要求只检查少量代表范围，不完成全位宽、长B2或多架构性能矩阵；功能检查仍保持前述完整覆盖。生产默认与显式实验策略分别记录，功能检查中的时间不进入性能统计。

实际选取9例：Stage1以σ26、B1=10000，分别对M127/128曲线、M607/64曲线、M2203/16曲线运行Edwards NAF4与Montgomery PARAM0，共6例；Stage2使用独立整数ladder核对的PARAM0完成点，B1=20、B2=13230、D=210，在M127、M2203、M16381运行普通曲线，共3例，覆盖算术容量4/64/256。最宽Stage2点来自已通过独立GMP检查的CPU夹具，避免超出Stage1 GPU Montgomery的目标上限。输入准备和独立点校验均在计时外，Stage2保留强制GMP检查，并核对完整结果及工作量。

每例每侧预热2次，然后交替AB/BA采集21对。进程墙钟从启动到观察到退出；内部边界分别为Montgomery CUDA event gputime、Edwards arithmetic API墙钟及Stage2 full_wall.total。各边界独立比较，不相加嵌套计时。

固定门限为新／旧配对耗时比中位数及每例95% bootstrap区间上界均≤1.05，两个计时边界都须通过。bootstrap10000次、固定种子20261020，不丢弃慢样本、不修改门限。这是逐例置信区间，不代表其他输入或硬件的资格。

`performance_protocol/result.json`的13项CPU门禁验证两个计时边界、置信区间、慢样本保留及固定5%边界；未就绪、缺失/未完成前置门禁、样本对数不足和改动门限均在GPU初始化之前拒绝。已就绪计划绑定22项通过的本机功能、身份及协议门禁，采样前复核25项本机输入和2114项功能输出。实际短B2三例使用独立的`performance_plan_sampled.json`，输入与冻结程序保持相同身份。

完整计划停止记录保留在`performance_reduced_by_user.json`。`performance_01/result.json`整体保持未完成，只使用其中已完成的6例Stage1结果；中断例的原始观测保留，不纳入上述9例统计。Stage2抽测单独保存于`performance_sampled_02/`。不将缩减范围后的通过结论写成原26例完整矩阵通过。

采样前已完成本机功能与身份门禁，并停止本任务全部编译。记录主机、GPU/驱动、程序和GMP摘要、输入、样本数、计时边界、功耗限制或不可用状态，以及实际GPU功耗/温度/时钟；按所选GPU保存前后计算进程快照。NVML观测表示所选整张GPU，包括验证和空闲时段，不作为进程峰值或能耗。

9例全部通过：共189对、378次正式测量、36次预热。进程墙钟配对比中位数范围为0.9922～1.0237，逐例95%区间上界最高1.0328；内部计时配对比中位数范围为0.9880～1.0201，上界最高1.0308。两种边界的最高上界均出现在M127 Stage2，低于固定1.05门限；最宽Stage2的内部比中位数为1.0002。所选范围未发现超过门限的回退，不能外推到未测长B2、其他树规模、承载路径或其他GPU。

统计只包含上述9个完整样例；中断例另有35条有效观测保留，按未完成样例整体排除，未按耗时过滤。完整逐例旧／新中位耗时、配对比和区间见`data/benchmarks/cuda_structure_p5_local_sampled_20261021/paired_times.csv`，程序、计划、采集器、原始报告及统计摘要见同目录`summary.json`。两轮原始数据保持各自身份，第一轮整体仍标记未完成。

设备观测共2093条，覆盖两轮采样、部分中断例、验证和空闲时段：整卡功耗1.820～10.503W（中位9.823W）、温度53～59°C、SM时钟210～1500MHz；功耗上限仍为不可用。该范围描述所选整卡的观测条件，不是各样例的平均能耗或进程显存峰值。

复用入口为`prepare_cuda_refactor_qualification.py`和`qualify_cuda_refactor.py`；后者拒绝缺失或未通过的前置门禁。后续如扩大范围，应另行明确输入与门禁，不把本次抽测结论作为完整矩阵资格。
