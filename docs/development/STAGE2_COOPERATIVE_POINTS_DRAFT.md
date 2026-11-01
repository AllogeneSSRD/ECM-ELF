# Stage2 协作点算术实验草稿

基线0.57.2，点专用平方已默认启用。正式复杂度与全宽基线见[位宽报告](../performance/STAGE2_BIT_SCALING.md)。本草稿保留探索和失败样本，完成后由用户决定是否清理。

## 首轮设计

- 只做独立原语，不先替换生产点内核。比较现有64-bit单线程实现与CGBN的4／8／16线程协作，保留相同实际模数、64W-bit Montgomery radix、输入与依赖递推。
- 普通模数使用CGBN Montgomery；精确梅森保留完整乘积＋线性折叠／旋转，避免把普通REDC与梅森折叠混为协作差异。本地CGBN的wide平方／Montgomery平方调用一般乘法，不假设其已有Karatsuba或对称部分积优势。
- GMP核对所有返回词，覆盖随机奇数、接近radix、精确梅森、不对齐位宽、别名、单步和递推；保留故障注入。先测试常见完整容器，再补实际W非整TPI的边界。
- 性能同时比较少量链的依赖延迟与充足实例的吞吐量，记录寄存器／local静态资源和GPU遥测；不修改显卡设置，不把原语收益写成完整Stage2收益。
- 原语有稳定优势后才实现完整xADD／链／ladder适配，固定D／布局／链长做小B2对照，保留0.57.2作为默认性能保护。

入口为`tools/build/test/build_stage2_point_coop_probe.ps1`与`tools/test/stage2_point_coop_probe.cu`。当前字段／链适配器已位于`src/ecm/stage2/cuda/device/point_coop_field.cuh`、`point_coop_chain.cuh`，0.57.3接入可选giant生产分派；下面保留初版探索顺序。实验输出放`data/experiments/stage2_point_coop_20261101/`。

## 构建记录

初版遇到CGBN旧load接口不接受const指针的编译错误，修正适配器指针签名；未修改第三方库，输入仍只读。证据保留`compiler_first/`。随后split=1的单CU模板展开在约7.4分钟／445 CPU秒时因实验预算主动停止；后续nvlink缺失obj是此次停止的结果，不是算术故障。日志和命令保留`compiler_single_budget/`。

构建工具新增显式SplitCompile参数，纳入命令与清单身份；共享实验构建默认1，本探针默认8，必须在沙盒外使用8，或在沙盒内显式选择1。当前使用新的`build_dev/stage2_point_coop_0_57/parallel/`目录，不覆盖前述证据。

全TPI组合的split=8构建前端完成，用时328.613s；ptxas在约11分钟总预算内仍未结束，主动停止并保留`compiler_parallel_budget/`。改为每个二进制一种TPI及合理位宽上限，三路并行：TPI4仅4k、TPI8至8k、TPI16至16k，分别321.77／351.06／357.84s。构建宏记录在manifest，其他TPI不会静默替代。当前生产路径未变化。

## 规范余数合同诊断

`correct_tpi16/call_0040`在4096-bit随机奇数、37实例、一次普通模乘中失败：192词不符。冻结二进制重复、改别名、改为32实例均在21／23／30号实例复现，现有标量原语通过。按本机CUDA13.3 CUPTI示例，在自有探针进程内采集只读host-visible memcpy，确认每个差值恰好为N，全部同余，排除别名与尾warp问题；证据在`capture/repro/`。采集器仅用于诊断，不参与性能统计。

本地`core_mont_wmad.cu`的Montgomery输出只在radix溢出时减N，允许非规范余数。适配器补一次比较／条件减N，保持既有[0,N)合同，第三方库不变，GMP按原始词严格比较。原适配器及探针保留`unreduced_source/`；修正后需重新完整验证后才能计时。

## 规范化修正后的独立结果

修正后三个TPI二进制重新构建并冻结。`canonical_correct16`为168次合法＋1次故障、1,022,976词；TPI8为96＋1、340,992词；TPI4为48＋1、113,664词，合计312次合法、1,477,632词全部GMP一致，3次故障均被拒绝。三类模数包括接近radix、精确梅森与随机奇数；保留原地／独立输出和1／3次递推。

`perf4/8/16`共288次正常性能进程，两个相反顺序、每次全部5事件；实例1／2048／8192、递推8次。count1生成器首元素为0边界，不代表随机单点延迟。正式结论只以大批量作吞吐量参考，初步`exploratory_mersenne_during_build`不进入合格成绩。

## 真实链与实际radix

在16k固定CGBN容器上支持实际8193…16384 bits，保留标量R=2^(64*nw)。普通宽乘积左移容器与实际radix的差，再做容器REDC；梅森先按容器归约，再作差值旋转。乘法体设为out-of-line共享，控制xADD多个调用点的编译展开；全局坐标仍只传实际nw词，不引入容器补齐缓冲。

生产头验证的`native_chain_edges/deep/single`共294次、76,908,204词全部GMP一致，覆盖实际W255等非整容量、1／2点尾、seed stride1／2、长度32／64；两侧共同形状完整SHA相同。较早102次原型验证与`chain_perf_first`独立保留，不替代生产头验证。

`native_chain_perf`使用p16319、W255、2048链×32点、seed stride2、完整65536点。标量／TPI16两进程中位数1495.813／306.247ms，约4.88倍；首末链GMP一致，完整33423360词输出SHA256为`73cef7c9e50ff67dca1c05095ab32147f927e5e3872859709467d79261d047db`。

## 0.57.3生产接入与验收

`NTT_GIANT_COOP=0|16`默认0；只替换W>128、W<=256、长度2…64的giant链。非单位ladder恢复及其他点阶段不变。CGBN额外64组乘法＋64组平方复用原有四个设备数组，默认／小位宽不触发。环境随子进程继承并记入ECM tune，旧cost-profile拒绝该模式；未改D模型、未新增TOML键。

通过正式`build_stage2_local.ps1`在受限环境外Jobs8／SplitCompile8构建。45单元依赖检查、24单元重编译，编译墙钟142.6s，总196.6s；协作CU76.49s。输出EXE SHA256 `1083a51c3fa568b202ec33f1a314bf7146de85f7782126001ff2717f70215db0`，版本0.57.3；NTT对象复用核对通过。未向外部生产目录部署。

测试结果：

- 初始化内存50组／4556检查、容量变异拒绝；原有四个显式自检设备数组不变。
- `native_switch`9例与`square_regression`7例通过，含非法值、真实队列与小位宽回退。
- 目标7995／16272 bits、普通／承载、旧版／off／on共12条完整诊断；D2310/P240/B2=26000000，目标全叶指纹、xADD、S4一致。此B2确保实际触发giant链，诊断不计入速度。
- `curves_fixed`30次＋最高位逆序复测4次，共5位宽／10格。固定B2=27015978990与原优选D、布局、链长；全部NF、算术正常、工作量相同。按各格中位数合计：0.57.2为197.199s，新off196.629s、TPI16为188.028s。较旧版−4.65%，较off−4.37%；最高位约−6.2%，10k接近持平。因此默认仍关闭。

## 管理员NCU及后续目标

NCU2026.2.1对上述生产头链探针的第二kernel采集，clock-control/cache-control均none，未更改显卡设置。`ncu_scalar/coop`保留19／17 passes、工具哈希、原报告及CSV；输出与未剖析参考完全一致。指标：活跃warp5.571%→16.665%，issue active11.590%→38.093%；local load sectors10,270,990,020→6,642,204,672，store5,241,958,971→96,199,234。后端寄存器64→255、栈26640→1504 B/thread；协作实际受4个64-thread block/SM的寄存器上限限制。

解释及公开图以[正式报告第11节](../performance/STAGE2_BIT_SCALING.md#11-可选tpi16协作giant链)为准。原始数据不改写。本草稿保留失败、预算中断与过程记录；完成后提示用户决定清理，不自动删除。下一项优先比较贴近实际位宽的容器／TPI32，再评估baby与seed，不凭原语倍率承诺全流程收益。
