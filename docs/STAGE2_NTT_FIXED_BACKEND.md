# ECM CUDA Stage2：固定 Goldilocks 后端

日期：2026-10-05，起点 `8cbd3be`。接续 [PTX 归约实验](D:/code/MPA-OpenCl/docs/STAGE2_GOLDILOCKS_PTX_REDUCTION.md)。6 模乘 xADD 与既有 D 重标定已完成，本阶段继续改善 NTT 热路径。

## 1. 改动与边界

运行时模式每次 `gl_reduce` 读取 volatile constant，再检查 PTX/short。上一阶段的普通短归约基线因此变慢；固定 PTX 后端能同时去掉读取和模式判断。

新增编译选项 `NTT_GL_FIXED_MODE`：−1 保留运行时 A/B，0 固定四fold，1 固定普通短归约，3 固定 PTX 短归约。host 参考保持原四fold。设备端固定版本直接调用相同的已验证归约器，未改变余数范围、NTT 长度、位宽、根、检查或数据流。

两套构建入口增加 `-GlBackend runtime|fold|short|ptx`，默认 runtime。构建签名包含后端，防止换选项时复用旧 exe。固定版本缺省环境变量继承编译后端；显式配置与有效后端冲突会退出2并输出 `ntt_gl_backend_conflict`，不会将固定 PTX 误记成普通短归约。

固定 PTX 即使没有设置 `NTT_GL_PTX_REDUCE`，D 选择器仍识别实际后端并拒绝沿用旧经验成本。原已发布 xADD6/GPU baby D 模型继续用于匹配的普通短归约路径；固定 PTX 的新 D 拟合尚未接入。

### 当前实现索引

- [后端合同与配置校验](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:95)、[有效模式](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:110)。
- [编译期归约选择](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:174)，固定 PTX 的设备路径不引用 mode symbol。
- [生产入口缺省配置](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:15)、[实际 D 后端匹配](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10788)。
- [NTT 构建](D:/code/MPA-OpenCl/tools/build/build_ntt_coop_probe.ps1:1)、[独立 native 构建](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:9)。
- [固定后端完整卷积探针](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:24)、[交叉驱动](D:/code/MPA-OpenCl/tools/bench/bench_ntt_fixed_backend.py:1)。
- [完整 Stage2 交叉驱动](D:/code/MPA-OpenCl/tools/bench/bench_stage2_fixed_backend.py:1)、[固定后端 D scope 门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_production_scope.py:19)。

## 2. 计算、容量和传输

设每次实际 NTT 调用 `N_i=2^k_i`、slice 数 `J_i`。三次变换和 pointwise/scale 的通用模乘数保持 `Σ_i J_i[(3k_i/2−1)N_i+T_outer,i]`（warp tile 最低两层根特化的统计约定）。PTX 短归约算术与上一阶段相同；本轮改善来自去掉每次归约的 mode load/判断，而非减少模乘。

NTT 数据数组、twiddle、GPU owner、baby 临时空间、pinned oracle、CPU准备以及 H2D/D2H/D2D 数据 payload 的算法增量均为0B。固定版不声明原4B device mode symbol，也不执行 mode 的4B `cudaMemcpyToSymbol`。这是配置开销变化，不能说曲线 PCIe 数据传输明显下降。constant 缓存 load 的消除也不能按每次4B算作外部显存带宽节省。

GPU scratch、owner 与 pinned 生命周期继续使用原预算。标准 outer pass 的 global 数据流量为16N B/slice；两个 forward tile 加 pointwise/scale/inverse tile为56N B/slice。本轮没有新增可用周期、stall或实际 PCIe/NVML 峰值测量；不能以0B算法增量替代完整进程 VRAM 账本。

## 3. 验证与纯 NTT 结果

GPU1 RTX4060 Laptop/sm89/CUDA13.3，串行。固定 short/PTX 各自通过：200000 device arithmetic cases、263357 inverse-scale/fallback word、cooperative 96组合/27131904 word与cached切换4次/3145728 word；legacy shared/warp 各216组合/6854400 word及缓存/容量/生命周期检查；故障拒绝；两种矛盾配置拒绝；显式一致配置确认。驱动 **16组/0失败**。另重新编译当前 runtime，PTX0/1的独立门禁 **8组/0失败**。

固定 short/PTX 在每个 k 下按 short/PTX/PTX/short 进程顺序：每进程8次、每次 warm+3 event均值，因此每后端16个目标样本。全部 N 输出在 event 外检查，pass/M/coop一致。

- k16：0.131884 → 0.120813 ms，快8.3948%。
- k24：12.788353 → 12.234177 ms，快4.3334%。
- k25：27.860523 → 26.801644 ms，快3.8006%。
- k26：55.300907 → 53.644779 ms，快2.9948%。
- k27：117.366955 → 112.381333 ms，快4.2479%。

再与上一阶段冻结的 runtime/PTX1 探针交叉：runtime/fixed/fixed/runtime；runtime进程只取PTX1的4次，固定进程取8次，两个目标进程/后端，分别8/16个event均值。

- k16：0.157824 → 0.122517 ms，快22.3710%。
- k24：12.513451 → 12.230763 ms，快2.2591%。
- k25：28.314544 → 26.804417 ms，快5.3334%。
- k26：56.364075 → 53.731243 ms，快4.6711%。
- k27：122.626007 → 112.697942 ms，快8.0962%。

两组比较分别保留，不将不同序列的均值相加或乘成生产收益。每后端只有两个进程，尚无置信区间；微小尺寸更受启动开销影响。

所选 warp forward/inverse REG40、LOCAL0、max_blocks3；M8 forward/inverse REG48/46、shared36096/36864B、max_blocks2，与 runtime 相同。未宣称整个 binary 无 stack；未选中的 legacy 模板仍可能 spill。

静态 SASS 另比较本轮重新编译的 runtime 与 fixed/PTX 探针：mode symbol在runtime存在、在fixed消失；所选四个kernel的constant bank3引用共111→0。M8 forward静态指令4848→3272、inverse4792→3208；warp tile forward3528→1696、inverse3992→1952。统计包含展开、循环、未执行后端分支和padding，证明编译后移除了模式及其他后端代码，不能当成动态执行指令数或固定周期减少。[原始指令与符号复算](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/sass_quantitative.json)。

证据：[固定与冻结runtime交叉数据](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/results/measurements.json)、[当前runtime门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/runtime_gates/measurements.json)、[固定short构建](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/short/manifest.json)、[固定PTX构建](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/ptx/manifest.json)。

## 4. 后续 D 重标定顺序

固定 PTX 必须建立独立成本依据，不能直接照搬 GPU baby 加普通短归约的 profile4。先冻结全部 k16..27 的完整卷积相对权重，再使用同 binary、同 save/Q/B1/检查的多 D 曲线拟合各阶段；拟合排除独立 B2 留出，排名同时执行 arena、owner、baby临时payload约束。最后复核 C++ 实际 shape/features、自动D scope及独立曲线，再决定是否发布。

点 Montgomery MAC、CPU准备与共享NTT scratch的多曲线lease仍是后续方向。旧Tensor真实tile/双流实验为负，本轮只改善CUDA整数路径。相同Q和实际B2的Prime95公平对照仍待完成。


## 5. 完整 native Stage2 与生产 A191 交叉

固定 PTX native SHA256 `9abd6ec69286d95971467924bdc5434e09b98af5ecce760588d0a8ae6494215f`，3709440 B，CUDA编译367.4s，C++对象3.9/2.4/3.1/2.9s；编译前冻结18个原始依赖、编译与运行后复核。构建参数明确为 `-GlBackend ptx`；生产A191未覆盖。

同一M4423 Stage1 save，sigma26、B1=1000、B2=2011326186870、D1381380、P126720、GPU1，Stage1跳过，baby1/xADD6/short1/outer2/warp1/resident/check保持。save SHA256 `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`。存档X为4423bit大整数；`restored_X=1`只表示恢复已启用。显式D且D_MODEL0，逆移位归一化0。

按 old/new/new/old/new/old/old/new 跑8条，每后端4条：

- 已发布A191：51.593827/51.255202/51.237514/51.375301 s，均值 **51.365461s**。
- 固定PTX：49.559370/49.539397/49.599690/49.602582 s，均值 **49.575260s**。
- 完整墙钟快 **3.4852%**；ABBA/BAAB两组改善分别 **3.6464%/3.3237%**。每侧4条，未建立置信区间，不与上一阶段不同序列拼接。
- init 10.998690→10.912117s，main 40.366771→38.663143s。
- `ntt_seconds` 调用侧墙钟 25.587000→24.000250s，占full约 49.81%→48.41%。包含准备/同步，非Systems纯kernel池，不能与其他阶段随意相加。
- baby ladder 7.894250→7.896000s；affine 0.259250→0.257250s。算术未改，差异属当前测量。

八条叶hash4244971527793015097、oracle c85031f6149bae11、因子集合一致；每条S4 397launches/1836241poly_muls/36615543coefficients、2400selftest、60474GMP样本、3 full_checks；oracle1010selected/queued/compared、60474samples、pending0，GMP bad0/clean1。原生产和候选检查覆盖一致。

固定PTX native缺省配置另过D selector **48/0**（观察决策后早停，不计完整曲线），所有旧模型匹配路径因实际PTX被拒绝。实际native在普通模数frozen save上正确发现因子59649589127497217，并通过真实warp GMP/cache/lifecycle和LOCAL0检查；两种配置冲突用直接curve worker均exit2，无完成曲线/结果记录。

两侧arena统计峰均 6149/6149 MiB；这是逻辑arena计数，非NVML VRAM总峰。本轮没有额外测量完整CPU私有内存/VRAM/PCIe峰，算法数组及payload无增量的结论由相同数据流得出。

证据：[8条实际完整曲线](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/stage2_ab/measurements.json)、[复算阶段与覆盖](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/quantitative.json)、[原始源码与save](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/native_provenance.json)、[scope](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/scope/summary.json)、[实际native因子/冲突](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_fixed_20261005/native_gates.json)。

## 6. 复现与发布状态

```powershell
tools/build/build_ntt_coop_probe.ps1 -Build build_cuda_cmake/fixed_short -GlBackend short
tools/build/build_ntt_coop_probe.ps1 -Build build_cuda_cmake/fixed_ptx -GlBackend ptx
python tools/bench/bench_ntt_fixed_backend.py --short build_cuda_cmake/fixed_short/ntt_coop_outer_probe.exe --ptx build_cuda_cmake/fixed_ptx/ntt_coop_outer_probe.exe --output <fresh-directory> --device 1
tools/build/build_ecm_cuda_stage2.ps1 -Build build_cuda_cmake/fixed_ptx_native -GlBackend ptx
```

完整Stage2交叉驱动要求baseline有 `frozen_manifest.json`、两侧有对应 `sources/` 原始快照，candidate构建manifest为fixed_mode3；验证每条运行前后exe/save/source哈希。日常runtime构建和默认开关保持，固定后端是独立可选编译。fold0逻辑保留，但本阶段独立编译门禁覆盖的是short1/PTX3/runtime。

当前仍生产A191。候选固定PTX默认不会沿用旧D模型；下一阶段按§4重新标定后，才评估自动D及生产发布。长期Prime95和多曲线吞吐目标尚未完成。


## 7. 后续发布状态（2026-10-05）

固定PTX已完成新D标定、独立holdout、原生入口与发布包验收，生产由A191更新为DCF；最终8条A/B为49.9000155→48.33592125s（−3.1345%）。本报告之前的测量和当时发布状态保留。详细证据与最新source/line见[固定PTX D报告](D:/code/MPA-OpenCl/docs/STAGE2_FIXED_PTX_D_CALIBRATION.md)。
