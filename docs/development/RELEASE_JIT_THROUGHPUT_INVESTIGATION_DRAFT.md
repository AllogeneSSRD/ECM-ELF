# 旧包 PTX50 在 Ada 上的吞吐调查草稿

## 范围与状态

2026-10-25 用户报告同一 RTX 4070 Ti：cu11.8 发布 JIT 为112.00秒/曲线、265W，本地cu13.3 sm89为48.78秒/曲线、212W。这些数值来自用户观察，功耗采样区间及完整运行参数尚未确认，不将它们直接解释为平均能耗。

本轮已用私有输入复现约2.3倍的持续计算差距，并用同编译器、同目标、只改变CGBN后端的实验定位原因。调查不修改生产代码、业务队列、GPU功耗/频率或正式发布矩阵；不存在生产补丁的修复后验收。原始试验及编译失败样本保留在 `data/experiments/release_jit_perf_20261025/`。

## 反馈回路与冻结身份

将三个实际EXE及各自GMP DLL复制到证据目录，身份见 `inputs.json`：旧包0.52.3、现代包0.52.3、本地sm89实际为0.52.2。实际包路径分别为 `release/ecm-elf-0.52.3-cu11.8/`、`release/ecm-elf-0.52.3-cu13.3/`，本地为 `build_stage1_local/`。保留程序版本差异，不将本地结果冒充同版本比较。

完整程序缩短任务使用 `(2^5323-1)/2372950817`（5292位，5632位容器），Montgomery param3、sigma26、960曲线、B1=20000、B2=0、GPU0，关闭指数缓存和定时checkpoint，所有配置、缓存及存档位于私有目录。按本地、旧包、现代包、旧包、本地、现代包串行运行；每个程序第二次运行复用自己的私有CUDA缓存。`repro.py`、每次 `invocation.json`、`stdout.log` 与 `repro.json` 保留命令及原始输出。第二次运行阶段返回的gputime分别为：本地3.76254秒、旧包8.74327秒、现代包3.74783秒。该字段是程序阶段计时，不等同于纯内核CUDA event；旧包首轮含JIT的26.9593秒不用于持续吞吐比较。

`math_comparison.json` 核对每次960条保存记录，数学字段完全相同，仅TIME不同；规范化数学记录SHA256为 `876d3264a6b0d99f54beccf8d5821bf40d18eba1396df51f54ec8f9f87573874`。`source_comparison.json` 核对旧包对应源码归档与工作区的CGBN选路、两种Montgomery实现及生产算术模板，四者字节相同。

## 假设与隔离结果

优先顺序为：架构条件编译选错当前设备的算术路径；寄存器/溢出；编译器版本；配置、频率或后台负载。一次性JIT启动等待由热运行反馈回路单独排除，继续调查持续计算差距。

选路依据在 `third_party/cgbn/include/cgbn/cgbn.h:67`：未强制后端时，低于compute50选择IMAD，compute50到compute6x选择XMAD，compute70起选择WMAD。`core/core.cu:315` 据此包含各自乘法和Montgomery实现。发布目标由 `tools/build/internal/release_profiles.ps1` 固定；旧包的compute50 PTX在C++条件编译阶段已经选择XMAD。JIT输入中没有被预处理移除的另一套模板，驱动不能重新选择WMAD。NVCC的宏定义依据见[CUDA 11.8虚拟架构说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-compiler-driver-nvcc/index.html#virtual-architecture-identification-macro)。compute50是计算能力目标，不是PTX ISA版本号。

私有 `arch_probe.cu` 使用实际CGBN Montgomery乘法，TPI16、TPB128、MAX_ROTATION=1、5632位容器、同一5292位模数、960组固定随机全宽操作数、每组50000次依赖乘法。只用CUDA11.8.89/MSVC14.16.27023编译，所有构建采集nvcc `--time`。每个进程先加载内核、预热，再用CUDA event采5次内核时间，不含JIT、主机传输或GMP核对；最后4组结果均用独立GMP幂模公式核对，错误0。

- 默认PTX50：设备编译宏500，实际binary89，XMAD；124寄存器，local=0，中位1701.755毫秒，范围1686.796—1737.778。
- 默认sm89：编译宏890，WMAD；80寄存器，local=0，中位726.152毫秒，范围691.796—732.483。
- sm89强制XMAD：编译宏890，XMAD；123寄存器，local=0，中位1645.600毫秒，范围1614.773—1654.177。与上一项同编译器/目标/源码，只改后端宏，足以复现主要差距。
- 默认PTX75：编译宏750，实际binary89，WMAD；94寄存器，local=0，中位704.768毫秒，范围692.033—722.710。
- PTX50强制WMAD：编译宏500，实际binary89，WMAD；94寄存器，local=0，中位709.212毫秒，范围689.393—732.206。另成功生成原生sm50 SASS；这证明默认sm70分界不是该算法不可跨越的编译下限，未验证sm50实体硬件性能。

前四项证据为 `probe_results.json` 及同名日志，最后一项为 `probe_wmad_result.json`。探针中没有local memory或静态编译溢出；不能将这一结论扩展到未剖析的全部生产内核。根因是旧目标选择的算术实现及其寄存器/指令组织，CUDA11.8本身可以生成快速WMAD，不必更换到13.3才能使用。

## MAD实现的支持语义

当前CGBN有IMAD、XMAD、WMAD三种后端。上游默认覆盖分别为Kepler（构建目标sm35、选路小于sm50）、Maxwell/Pascal（sm50起）、Volta及以后（sm70起）。上游README要求Kepler/Maxwell/Pascal使用CUDA9.0+、Volta及以后使用CUDA10+；这是CGBN上游要求，不是ECM-ELF完整工程的已验证最低工具链。[上游说明](https://github.com/NVlabs/CGBN)

默认架构分界、PTX指令支持、编译通过及实体硬件正确性/性能是不同条件。本轮证明WMAD可为sm50编译，且PTX50在4070 Ti上运行正确、恢复吞吐；未证明其在Maxwell/Pascal上比XMAD更快，不能据此全局强制WMAD并宣布旧显卡性能不回退。

## 条件、结论和待处理项

设备为RTX4070 Ti、sm89、60 SM、驱动610.88；设备UUID/功耗上限/时钟快照见 `device-after.csv`。未锁定功耗和时钟，保留桌面上下文；串行运行试验，不干预其他进程。完整程序每个二进制有1次热缓存样本；算术探针每个变体5次event样本。上述比值只用于定位这套输入、设备和二进制身份的原因，不作为其他位宽、曲线或GPU的稳定加速比。

4070 Ti可以直接使用现有cu13.3现代发布包：本次同版本现代包已达到本地sm89相近吞吐。若需要cu11.8旧包同时高效服务现代GPU，可考虑额外现代目标镜像或独立后端分派；其余方案包括显式WMAD，但须先验证旧硬件数学与吞吐。正式PTX50/SASS60矩阵保持当前授权状态，不把实验变体写成已发布功能。

同sm89目标WMAD/XMAD交替各3轮，每轮5次event，中位数再取三轮中位为728.510/1680.275毫秒。每轮通过nvidia-smi轮询power、SM/显存时钟、利用率及温度，只汇总GPU利用率不低于95%的忙时样本，WMAD/XMAD分别22/51条；每轮平均功耗再取三轮均值为202.134/261.866W，SM时钟约2802.946/2788.529MHz。原始遥测及每轮汇总见 `power_*/telemetry.json` 和 `power_ab.json`。这是同编译器、同实体目标、只改后端的功耗方向验证，不直接换算用户完整任务能耗。功耗/时钟未锁定，桌面活动和轮询延迟保留。

调查结论同步到正式[发布/JIT说明](../performance/RELEASE_JIT.md#旧目标-ptx-在-ada-上的持续吞吐)。TODO：如需要正式改变旧包策略，补做Maxwell/Pascal实体硬件数学与吞吐验证；其他曲线/位宽及驱动不由本例获证。草稿完成后交由用户处理，不自动删除。

## CUDA13.0—13.2环境的补充核对

用户进一步询问cu13.3包能否在CUDA13.0—13.2环境运行。实际Stage1/Stage2 PE导入表没有外部cudart DLL依赖，运行不要求安装本机13.3 Toolkit。若指驱动仅支持13.0—13.2，则官方13.x小版本兼容允许R580及以上驱动上的受支持SASS/既有功能；新版PTX JIT与新驱动功能不在保证内。当前包包含sm80 SASS，因此Ampere/Ada有直接镜像可用；需要回退到PTX75的设备不能一概承诺在旧驱动运行。未更换驱动，不将610.88实测写成R580/R590/R595已验。[官方兼容条件](https://docs.nvidia.com/deploy/cuda-compatibility/610/minor-version-compatibility.html)

`modern_sass_no_observer/result.json` 记录实际cu13.3程序设置 `CUDA_DISABLE_PTX_JIT=1` 后正常退出，完成960曲线且数学字段与原对照一致。额外CUPTI观测在生成保存记录后出现120秒退出超时，保留在 `modern_sass_observed/diagnostic_status.json`；其中实际模块为sm80，仅作为镜像路径观测，不计入完整退出验收或吞吐样本。去掉观察器的对照已通过；该附加观察器异常不作为生产性能差距的根因。
