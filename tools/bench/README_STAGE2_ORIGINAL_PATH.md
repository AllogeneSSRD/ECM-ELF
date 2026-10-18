# 原路径 GPU Stage2 优化测量

当前算法见 [Stage2 管线](../../docs/architecture/STAGE2.md)，数学布局见 [G 树固定资料](../../docs/reference/ECM_STAGE2_G_TREE_LAYOUT.md)，有效性能结论见 [Edwards 性能](../../docs/performance/EDWARDS_STAGE2.md)。实验数据均在忽略的 `data/` 下，脚本要求新输出目录，不覆盖已有原始调用。

## 构建和正确性

使用 `tools/build/test/build_edwards_gpu_workers.ps1 -Stage stage2 -Build <候选目录> -SplitCompile 1` 构建生产引擎，再用 `build_stage2_relocation_probe.ps1 -Build <候选目录>/stage2` 链接同一引擎对象。正式构建入口和依赖签名与本地构建共用，不能以开发原型代替生产候选。

- `tools/test/test_stage2_original_tree.py --exe <候选EXE> --output data/experiments/<G树门禁> --device 1`：48组独立GMP根/节点、多项式故障注入；普通M127和其8255-bit合法承载分别验证保留/均衡两种运行策略，带258叶尾批的独立密集请求枚举、实际请求及S4路由有序签名。
- `tools/test/test_edwards_stage2_relocation.py --exe <候选EXE> --probe <同对象API探针> --output data/experiments/<Edwards门禁> --wide-reference <既有精确输入计划>`：复用完整Edwards因子、普通/承载和非单位门禁，含gcd=N与baby零。该门禁需要已有的数学反例证据及输入计划；它也检查共享内核对可选relocation的兼容性，计时矩阵仍全部关闭relocation。
- `tools/test/test_stage2_ntt_memory.py` 和 `test_stage2_s4_memory.py` 各以 `--output <新目录>` 运行CPU模型验证。前者将生产arena替换成不调用GPU的数组分配账本；Graph禁用且对象句柄不在此模型范围。后者验证真实S4分配语句、完整请求程序及错误模型的拒绝。

`build_stage2_giant_point_probe.ps1` 与 `tools/test/stage2_giant_point_probe.cu` 用于选择阶段的短反馈。探针参数为 `nw 模板容量 点数 chain点数 CUDA线程块 轮换缓冲区 设备`。直接提取生产模乘/xADD，以独立八乘法GMP公式比较所有chain末点和首末chain全部坐标；一次预热后取三次CUDA event内核均值，不含GMP检查。它不是完整曲线性能测试。测试架构以sm89为默认，实际设备须显式指定。

## 冻结和运行

`bench_stage2_original_path.py prepare` 接受 `--baseline`、`--candidate`、`--reference`、`--gates`、`--output`、`--device 1`、`--repeats 3`。reference是已完成的Edwards relocation研究计划，复用其六份独立Stage1输入；gates是与候选EXE哈希一致且已通过的Edwards门禁摘要。还须单独完成上述G树门禁。prepare会冻结两个EXE、GMP、构建清单、编译源闭包、输入、采集器及辅助脚本。

加 `--rounding` 时，只为10232-bit低/高两档增加独立的完整G实验。它以固定D/P计算 `B2'=max(B2,(ceil((floor(B2/D)+2)/P)*P-2)*D)`，真实地增加覆盖范围，不改写请求B2下的任何调用。其他单元格不增加该实验臂。

随后运行 `bench_stage2_original_path.py run --study <冻结目录>`。每侧预热一次、正式三次，按轮次和单元格交替顺序；每次是独立进程、单条完整Edwards曲线，固定D、普通目标N、relocation=0、arena6300/owner640/batch256 MiB、factor-only、NTT计数开启、短标定与runtime tune关闭。清除继承的NTT环境，只指定非阻塞CUDA启动。不得同时在GPU1运行其他诊断或构建后另换EXE。

基准主指标为 `stage2_full_wall.total=init+main`，包含引擎内自检、树、逆、主循环、下降、GCD及检查收尾；shape已在init中，不再相加。Stage1、进程启动、计时前点算术预检和外层选D不在此总计时内。另存进程墙钟和200ms整卡遥测，退出轮询最多多计200ms；遥测不是进程显存峰值。

中断后只恢复已入账且哈希匹配的调用；发现未入账目录会停止供检查。改变采集器、源、输入或协议时建立新研究，保留旧的成功和失败证据。

## 审计和图表

`tools/test/check_stage2_original_path.py --study <目录> --output <新审计JSON>` 独立核对声明的调用（六格、两档补齐、三次正式重复时共56次）、全部原始文件哈希、输入/版本/几何/实际B2、计时守恒、因子整除、相同B2跨构建完整叶一致、Gamma唯一覆盖以及NTT长度表和重复工作量。`--partial` 仅检查已入账调用，输出不能作为完成证明。

`plot_stage2_original_path.py --study <目录> --audit <完成审计> --output data/benchmarks/<新目录> --figures data/figures/<新目录>` 使用Matplotlib生成NF、因子和显式B2补齐三组PNG/SVG，另写统计JSON、计时CSV和逐长度NTT计数CSV。只从匹配的完成审计绘图；中位数和min/max按正式样本计算，预热保留但不入统计。图不把增加B2的实验与原覆盖的加速混为一类。
