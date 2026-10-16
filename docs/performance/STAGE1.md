# Stage1 性能依据

## 口径与当前策略

CUDA Montgomery 默认 ladder、TPB128、按容器选择 TPI，分档寄存器；PRAC 默认不编译。GPU Edwards 默认窗口 NAF。已测局部最佳不能扩大到其他位宽/设备或直接替换默认。

完整曲线 wall、GPU kernel 时间、批量摊销 s/curve 和短时投影分别报告。生产 B1=10e6…260e6 的短时采样按完整/窗口工作比投影，排除冷准备、最终归一化/save；checkpoint 不算已完成曲线。

计时先预热，再按相反/交错顺序重复，保留完整矩阵、失败和原始日志。管理员 NCU 会重放，采集时间不作吞吐；Nsight Systems 的自身事件间隙不等于整卡空闲比例。

## GPU Edwards 细分实例三曲线族对照

当前Edwards使用[32档最小可用容器](../architecture/STAGE1.md#gpu-edwards)，核函数和七组实例化文件位于 `kernels/cuda/`，保留17408档上限。以下为同一新exe的Edwards NAF、Montgomery PARAM0与PARAM3完整批次成本，不比较曲线族的因子发现概率。

### 条件与计时

GPU1 RTX4060 Laptop、24 SM、UUID `GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d`，驱动610.88、CUDA13.3、MSVC14.51.36231、sm89、无PTX。exe SHA256=`fa8b903e9186dada7055e61176fff237411b575c4707a04541f5cc50e15ad3eb`，产品源码对应实例扩展提交 `ec47b0c`。构建为通用Montgomery域、TPB128、resident/PRAC关闭，保留当前分档寄存器策略。

输入为完整 `M_p=2^p−1`，B1=10000、每批768条、sigma起点20260922。Edwards采用48·lcm、w6、HWCD；Montgomery采用lcm与ladder。每组排除一次设备预热，再交错/反序采集三次正式样本；每次是新进程，CUDA上下文/模块加载仍计入。完整进程wall包含启动、标量/曲线准备、GPU、GCD/归一化及save，排除独立CPU末点核对。此边界不能与嵌套的算术入口或CUDA event区间相加。

硬件设置保持原值。利用率≥50%的457个离散遥测点中，SM均为1500MHz，观测功耗8.90…26.51W、温度57…68°C，功耗上限查询N/A；不声明锁频或55W。利用率窗口与瞬时功耗不同步；这一批的温度与采样条件不能直接合并到下文冻结参考。

### 完整进程成本

![GPU1细分实例同exe三曲线族完整进程成本](../../data/figures/edwards_fine_tiers_20261016/comparison.png)

左面板为秒/批的对数轴；右面板把每个样本除以同位宽PARAM3的进程中位数，越低表示成本越小。实心点为中位数，空心点保留全部三次正式样本，误差线仅是实测最小值至最大值，不是置信区间。[矢量图](../../data/figures/edwards_fine_tiers_20261016/comparison.svg)。

| 输入 | Edwards NAF 秒/批 | PARAM0 秒/批 | PARAM3 秒/批 | NAF/PARAM3成本比 | Edwards容器/TPI | Montgomery容器/TPI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| M347 | 0.9537 | 0.6228 | 0.6062 | 1.573 | 384/4 | 384/4 |
| M677 | 1.1098 | 0.6719 | 0.6457 | 1.719 | 768/8 | 768/8 |
| M991 | 1.1686 | 0.8531 | 0.7538 | 1.550 | 1024/8 | 1024/8 |
| M2203 | 2.5931 | 2.6388 | 1.8226 | 1.423 | 2560/16 | 2560/16 |
| M4003 | 5.7542 | 4.6686 | 3.7624 | 1.529 | 4096/16 | 4096/16 |
| M8191 | 16.6265 | 20.3712 | 16.5450 | 1.005 | 8192/16 | 9216/32 |

前五个输入的容器/TPI已经一致；8191位按Edwards一位余量使用8192/TPI16，Montgomery按六位余量使用9216/TPI32。不同曲线构造、状态坐标、切片和准备开销仍存在，不能把整进程比值解释为纯点公式代价。表值除以768仅是批量摊销值，不是单曲线串行耗时。

本矩阵前五个位宽的NAF中位成本为PARAM3的1.423…1.719倍。M8191为1.005倍，NAF范围16.4140…16.7420秒与PARAM3范围16.5397…16.5563秒重叠，所测完整成本接近。M347/M677的NAF范围分别为0.6381…1.0394、0.7084…1.6228秒，新进程离散明显；不把三次中位数视为稳定算法倍数。M2203的NAF与PARAM0范围也重叠，不能据中位数微小差异宣布稳定领先。

本表不与下文不同exe的七档数据相除来宣称细分容器的独立加速比例，也不外推生产B1、常驻队列吞吐、其他设备或全局最优TPI/窗口。

### 验收、证据与复现

完整[验收记录](../development/GPU_EDWARDS_FINE_TIERS_DRAFT.md#2026-10-16完整正确性门禁)覆盖32档、131组探针/537条逻辑曲线组合、70条CLI检查、旧V1/V2跨容器恢复及Stage2交接。正式性能采集还核对每条记录和重复稳定性，首尾曲线用独立整数梯形计算；六个位宽共4608条Edwards归一化记录与冻结七档NAF逐字节一致，汇总为 `data/benchmarks/edwards_fine_tiers_20261016/endpoint_identity_01.json`。

正式采集共72个完整批次：54个正式样本、18个排除的预热。原始证据为 `data/experiments/edwards_fine_tiers_20261016/bench_01/`，审计汇总为 `data/benchmarks/edwards_fine_tiers_20261016/comparison_01/summary.json` 和 `comparison.csv`；汇总SHA256=`158da9ec62537d18a2a59f9fb0ee3c8b2d48e9420d32aad0e961640fdd0f8665`。程序/GMP、源与工具快照、编译命令、设备遥测、日志和save摘要均保留。审计同时核对新Edwards实例清单、实际编入档位及两种Montgomery的原日志几何信息。

采集入口为[批次基准](../../tools/bench/bench_edwards_gpu_stage1.py)，完整参数保存在原始 `summary.json` 的 `command`。审计输出须使用新目录；以下示例重新审计同份原始证据并重绘，不重新运行GPU：

```powershell
python tools/bench/analyze_edwards_gpu_stage1.py `
  --evidence data/experiments/edwards_fine_tiers_20261016 --suite bench_01 `
  --output data/benchmarks/edwards_fine_tiers_20261016/comparison_new
python tools/bench/plot_edwards_gpu_stage1.py --comparison `
  --input data/benchmarks/edwards_fine_tiers_20261016/comparison_new/summary.json `
  --output-dir data/figures/edwards_fine_tiers_20261016 `
  --manifest data/benchmarks/edwards_fine_tiers_20261016/plots_new.json
```

图表以Python/matplotlib生成，输入、脚本和图表摘要记录于 `plots_01.json`。图和汇总按规则保存在Git忽略的 `data/`，新工作区需要取得相同证据后生成；Git维护脚本与文档。

## GPU Edwards 初始基准与 NAF 对照

以下为冻结七档容器版本的初始基准和NAF/窗口对照；当前32档构建及三曲线族成本见上节。它们保留各自构建与计时条件，不表示当前程序的绝对性能。

### 输入、设备与计时

GPU1 RTX4060 Laptop、24 SM、UUID `GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d`，驱动 610.88、CUDA13.3、MSVC14.51.36231、sm89、无 PTX。输入均为完整 `M_p=2^p−1`，B1=10000、sigma 起点 20260922。主矩阵每批 768 条；窗口/倍点矩阵每批 128 条。Edwards 用 48·lcm，Montgomery PARAM0/3 用 lcm；同 sigma 表示各自曲线族的确定性种子，不表示同一条曲线。这些数据比较实现成本，不比较找到因子的概率或单因子总成本。

每组先运行一次不计入统计的设备预热，再交错/反序运行三次正式样本。每次均为新进程，CUDA 上下文及模块加载仍计入；整批完成后才采样结束。进程 wall 包含进程启动、指数/曲线构造、GPU、GCD/归一化及 save，排除独立 CPU 末点核对。Edwards 的 `arithmetic api_seconds` 另覆盖完整点乘入口，边界见[实现](../architecture/STAGE1.md#gpu-edwards)。CUDA event 区间不是 profiler 的纯 kernel 时间，不能与上述嵌套 wall 相加。

保持原硬件设置。利用率≥50%的 1132 个离散采样点中，SM 均为1500MHz，观测功耗8.71…27.90W、温度70…78°C；功耗上限查询返回 N/A，不声明55W或锁频。利用率采样窗口与瞬时功耗并非严格同步。这批数据不与历史1800MHz/55W样本合并。

### 优化前的三曲线族成本

下表为完整进程秒/批的三次中位数，批量768。每曲线摊销值为表值÷768；不是单曲线串行耗时或峰值吞吐。

![GPU1 三曲线族完整 Stage1 进程耗时与全部样本](../../data/figures/edwards_naf_20261016/initial.png)

图中横轴按输入分组，纵轴为对数秒/批。实心点是中位数，空心点保留三次正式样本，误差线仅表示实测最小值至最大值，不是置信区间；后续耗时图沿用此约定。M8191 灰底组使用独立补建的原算法程序，不能与其他构建拼成优化加速比。[矢量图](../../data/figures/edwards_naf_20261016/initial.svg)。

| 输入 | Edwards 二进制 | Montgomery PARAM0 | Montgomery PARAM3 | Edwards 容器/TPI | Montgomery 容器/TPI |
| --- | ---: | ---: | ---: | --- | --- |
| M347 | 1.0412 | 0.7407 | 0.7488 | 512/4 | 384/4 |
| M677 | 1.2414 | 0.9110 | 0.8507 | 1024/8 | 768/8 |
| M991 | 2.0503 | 1.0725 | 1.0091 | 1024/8 | 1024/8 |
| M2203 | 7.8158 | 2.3166 | 1.9861 | 4096/16 | 2560/16 |
| M4003 | 8.3948 | 4.9636 | 4.0369 | 4096/16 | 4096/16 |
| M8191 | 26.5794 | 20.6843 | 17.4349 | 8192/32 | 9216/32 |

前五行使用优化前 exe SHA256=`3777416c9026027c88d2d56f78e6fbfa28a3886a7d78aa25ce9a500b66a534cb`。M8191 使用同一冻结原算法源码、包含所需9216实例的 exe=`cfa6c4d675a33fddca72797254c02ff2170593c2a7507d1d78c5e8000cbb5dde`。Montgomery 需预留六位进位，8191 位不能使用8192容器。两份构建及样本分别保存在 `bench_initial_01/`、`bench_initial_wide_01/`：前者原报告complete=false，审计明确只采用五个完整范围；8191位的完整资格由后者提供。未完成记录不计正式样本。

容器和 TPI 不同，尤其 M2203 的4096与2560，不能将成本差距全部归因于曲线公式。原算法跨构建/采样批次也存在时间差异，优化比例另用下面的同一 exe 配对数据。

### 同一 exe 的 binary/NAF 配对

exe SHA256=`d853b4ce80c78853a4fdafa5f3ad3abc51cbe267cd5231c64d2a523e299f9a89`，原13个CUDA对象与前五行基线逐字节相同，新增独立NAF模块。仅切换 Edwards binary/NAF，曲线、标量、容器、批量和保存合同相同；NAF 为窗口6、HWCD倍点。每批所有归一化记录在两种模式间相同。计时变化包含稀疏加法、按需生成T、设备算术域跨片保留及只回传四坐标的共同收益。

![同一 exe 的 binary 与 NAF 进程和算术入口耗时比](../../data/figures/edwards_naf_20261016/paired_naf.png)

两面板分别将每个样本除以**同位宽、同一计时边界的 binary 中位数**，虚线为1，越低表示耗时越小；标注为中位数耗时变化。归一化让各位宽的收益与离散可直接比较，绝对秒数见下表。算术入口嵌套在完整进程内，两个面板不能相加；M991 的范围重叠在图中保留。[矢量图](../../data/figures/edwards_naf_20261016/paired_naf.svg)。

| 输入 | binary 秒/批 | NAF 秒/批 | wall 降幅 | 算术入口秒 binary→NAF |
| --- | ---: | ---: | ---: | --- |
| M347 | 1.0060 | 0.8466 | 15.8% | 0.8389→0.6839 |
| M677 | 2.0284 | 1.7999 | 11.3% | 1.7931→1.5595 |
| M991 | 1.9627 | 1.8670 | 4.9% | 1.7031→1.5952 |
| M2203 | 8.3400 | 5.3243 | 36.2% | 7.7581→4.7173 |
| M4003 | 8.3451 | 5.3988 | 35.3% | 7.1778→4.2313 |
| M8191 | 26.2992 | 17.1940 | 34.6% | 23.0279→13.9288 |

这些是三次中位数。M991的binary范围1.2384…2.1979秒、NAF范围1.7704…1.8745秒重叠，微小收益尚未超出实测离散；没有剔除偏快或偏慢样本。M2203/M4003/M8191 的完整批次为1.53…1.57倍、算术入口为1.64…1.70倍；不将两者相乘或相加。

### 窗口与倍点形式

同一新版exe，B1=10000、批量128、HWCD倍点。下表为字典与主循环 CUDA event 区间之和的三次中位数（秒），包含可能的首次加载/发射间隙。不同窗口组分别采集，完整进程排序受准备及冷启动离散影响。

| 输入 | w=6 | w=8 | w=10 | w=12 |
| --- | ---: | ---: | ---: | ---: |
| M347 | 0.4064 | 0.3945 | 0.3901 | 0.4100 |
| M2203 | 1.4017 | 1.3506 | 1.3405 | 1.4018 |
| M8191 | 3.3671 | 3.2583 | 3.2202 | 3.3745 |

![窗口对完整进程、CUDA event 区间及字典分配的影响](../../data/figures/edwards_naf_20261016/windows.png)

左、中面板按同位宽的 w6 中位数归一化，各用独立纵轴；event 面板放大局部差异，完整进程面板显示准备及启动离散。右面板为 M8191、128条曲线的**字典逻辑分配量**，由容器字节数×曲线数×4坐标×2^(w−2) 得到，依据为 [`run_naf`](../../src/cuda/ecm_cuda_edwards_naf.inl) 第11行起的分配公式，不是进程显存峰值。窗口组分别采集，折线仅连接所测离散配置。[矢量图](../../data/figures/edwards_naf_20261016/windows.svg)。

这批event区间以w10较小，不能据此宣称所有B1/批量的最佳窗口或全局改默认。M8191的w6/w10完整进程中位数为4.7432/4.5841秒，字典从8MiB增长到128MiB；w12为512MiB且完整进程升至5.1838秒。默认仍为6，允许显式2..12；生产B1和更多批次的标定仍待验证。

倍点配对使用w8、批量128。HWCD（GPU内部编号4）与采用Prime95算法5代数形式的2XY路径（编号5）交错运行：M347/M2203/M8191 的event区间中位数为0.4031→0.3758、1.3490→1.3362、3.2551→3.2511秒，分别降低约6.8%/0.95%/0.12%。完整进程样本范围重叠，未据此切换全局默认。Prime95 `undoc.txt` 的算法6说明过期，同版源码已废弃6；CGBN没有模拟算法5的FFT复合指令，依据见[数学参考](../reference/ECM_EDWARDS_MATH.md#21-prime95-倍点算法编号与-fft-条件)。

![HWCD 与 2XY 倍点形式的进程和 CUDA event 区间耗时比](../../data/figures/edwards_naf_20261016/doubling.png)

各样本除以同位宽、同一计时边界的 HWCD 中位数。右面板放大 event 比值变化：大位宽的局部差异很小，M8191 两路径的 event 范围也重叠；左面板完整进程范围三个位宽均重叠。百分比不是不确定性估计，不能由此宣称稳定的端到端收益。[矢量图](../../data/figures/edwards_naf_20261016/doubling.svg)。

### 证据与复现

原始证据根目录为 `data/experiments/edwards_naf_20261016/`；汇总为 `data/benchmarks/edwards_naf_20261016/final_02/summary.json` 和 `initial.csv`。共192个完整批次：144个正式样本、48个预热；采集器冒烟和缺档失败不计入。每份数据保存exe/GMP摘要、源快照、编译命令、设备信息、遥测、日志及save摘要；采集核对每条记录/重复稳定性，首尾曲线用独立整数梯形核对，同曲线的binary/NAF整批记录逐条相等。审计重查范围、样本数、文件/源身份、几何形状、窗口/倍点选择、原日志计时与嵌套关系；四种私有元数据损坏均拒绝，原始证据未修改。

复现工具为[完整批次采集](../../tools/bench/bench_edwards_gpu_stage1.py)、[冻结原算法构建](../../tools/build/test/build_edwards_initial_baseline.ps1)及[证据审计](../../tools/bench/analyze_edwards_gpu_stage1.py)。NAF全档、V1/V2恢复、大B1及Stage2交接正确性见[验收记录](../development/GPU_EDWARDS_STAGE1_NAF_DRAFT.md)。B1=1e6/1e7只有正确性验收，本表不外推生产B1、其他设备、连续队列吞吐或Auto B2成本资格。

四组图由[绘图脚本](../../tools/bench/plot_edwards_gpu_stage1.py)直接读取上述审计汇总生成，没有重新采集 GPU 数据。需要 Python 3.10+ 和 matplotlib，从仓库根目录执行：

```powershell
python tools/bench/plot_edwards_gpu_stage1.py `
  --input data/benchmarks/edwards_naf_20261016/final_02/summary.json `
  --output-dir data/figures/edwards_naf_20261016 `
  --manifest data/benchmarks/edwards_naf_20261016/plots_01.json
```

脚本先核对144个正式样本、48个排除的预热、矩阵范围、缓存统计、曲线参数和配对构建身份，再输出4组 PNG/SVG。输入汇总 SHA256=`edcc591d52c683c10447386e5b2a12e8d00cc6249be40d6b7a027f9e49acbe53`；`plots_01.json` 记录完整输入/脚本/图表摘要、各组 exe 身份、环境版本及归一化口径。绘图核对记录 `plot_validation_01.json` 验证全部样本点在坐标范围内、归一化比值与审计一致、36个窗口正式样本的字典分配与原日志一致、7项损坏输入均拒绝，以及8个输出重复生成时摘要相同；四组图均已查看排版。原始证据不作修改。图和汇总位于 Git 忽略的 `data/`；新工作区需取得同份审计汇总后再生成，Git 维护脚本与说明。

## 已测 PRAC 范围

RTX4060 Laptop、24 SM、sm89、N=2^4423−1、4608/TPI16、lcm、TPB128 的显式实验中，批量1536、cap168、目标切片50ms 为已测有效配置。TPI32/cap128 能改善某些受限批量的资源或局部吞吐，但没有超过该范围最佳吞吐。

在终结规则专用体的冻结对照中，目标50ms、每配置4次短时前缀，得到：

| B1 | 参考 PRAC s/curve | 专用体 s/curve | 时间变化 |
| --- | ---: | ---: | ---: |
| 10e6 | 5.063071 | 5.050395 | −0.250% |
| 260e6 | 132.510655 | 132.178987 | −0.250% |

这是速度投影的中位数，未计完整曲线墙钟。冻结专用二进制 SHA=`464f124878ab755fba5e698f26962c15a1425c7bc0f4de83ed85e8df773d2b88`；108 条采样矩阵完整，GPU 忙时 SM=1800MHz。后续共享 DBL 的配对测试保持 TPI16/cap168/50ms 为最佳范围，不能把此表当作不同二进制的最新完整性能。

当前可复现选项、源码和采样脚本见 [实现](../architecture/STAGE1.md)、[bench_cuda_prac.py](../../tools/bench/bench_cuda_prac.py)、[bench_stage1_tpi_matrix.py](../../tools/bench/bench_stage1_tpi_matrix.py)。证据在 `data/experiments/stage1_rule_sentinel_*`、`stage1_shared_dbl_*`；基准必须同时记录实际 exe SHA、variant、TPI、寄存器、C 和切片策略。

## 资源与瓶颈

4608/TPI16 PRAC 的一个自然分配样本为172 regs、实际176，TPB128 时寄存器最多容纳2 blocks/SM；cap168 可容纳3。强制 TPI32 样本为109/112 regs、4 blocks/SM，但同时减小每 block 曲线数、增加 padding/跨 lane 成本，端到端未因此更快。

已采集的 natural/168/TPI32 achieved occupancy 为16.34%/19.37%/27.12%，issue active约30.60%/30.93%/38.93%。这些 profiler 值说明资源权衡，不能独立证明瓶颈成因或加速比例。DRAM 吞吐很低，主要成本仍在依赖多精度算术、发射与局部存储/调度。

normalized PRAC 按 ADD=A、DBL=D 统计：ADD=4M+2S，DBL=3M+2S；若平方与乘法同价，工作权重为6A+5D。该权重用于投影，不代表固定 GPU 周期。

## 限制与后续

PRAC表对应1800MHz/55W条件；GPU Edwards本轮条件见上文，保持分开记录。不同批次、驱动、采样窗口和构建的点不能直接拼成新的加速比。

对通用默认的改变仍需更多位宽、曲线族、设备和完整 B1 验证；现阶段优化优先级及其他未完成工作见 [TODO](../TODO.md)。
