# Stage1 param0：共享 seed/loop DBL 实验（2026-10-07）

## 1. 实验身份与范围

基线 `829b8e3`（终结哨兵v3）。本轮称为v4：仅演进显式 `ECM_PRAC_VARIANT=single-compact` MODE12/13，把私有奇素数链的seed DBL与循环DBL统一到一个inline调用点。公共p2分支、公共算术、计划、归一化、切片和checkpoint语义保持原实现；旧single-add MODE10/11保留。

- before exe SHA256：`464f124878ab755fba5e698f26962c15a1425c7bc0f4de83ed85e8df773d2b88`。
- after exe SHA256：`fe2b74c7f6b17a401c256b351e55004382bc67ab4923e59356c3ea5dbda90ef5`。
- 本轮私有头SHA256：`51d10f79284dfc813f1478f9ae437c9d5ce6254ed73109e9b07837468bf7e26f`。
- 公共算术头仍为 `67dc71128495378991dd750d73f7d8226ebedd255e267abd435dc6331a03e0a6`。
- before冻结文件在本地 `build_cuda_cmake/prac/after_rule_sentinel_20261007/`；计时、门禁、SASS与NCU在忽略的 `docs/data/` 中。

仍仅支持容器4608/TPI16的此候选，register255/168。N4423默认选4608/TPI16；TPI32此前为显式对照，本轮未改变默认TPI、TPB128或默认算法/寄存器/目标100ms。

### 1.1 TPI、TPB 与 blocks/SM 的约定

TPI是协作计算一条曲线的线程数，TPB是一个block的线程数；二者不能混用。默认容器2560..8192使用TPI16，9216..16384使用TPI32，见[类型定义](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1201)。容器须覆盖`N bits + 6`，所以8191-bit输入会选9216/TPI32，不能简单按输入是否超过8192判断。

此前4423-bit/TPI32的对照使用显式`ECM_STAGE1_TPI=32`，见[覆盖选项](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:89)及[4608/TPI32实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_alternate32.cu:15)；4423-bit默认仍为4608/TPI16。本轮专用候选不支持TPI32。

[当前TPB默认128](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:104)，源码记录2026-09-25由256调整。限制寄存器不会自动改变TPB。提交块数为`ceil(C*TPI/TPB)`；TPI32将曲线数减半，可以匹配TPI16的提交grid，但实际驻留量仍由各内核资源和grid供给决定。

GPU1每SM有65,536个32-bit寄存器。仅考虑寄存器约束：TPB128、硬件分配176/168寄存器每线程时，容量分别为2/3blocks；若TPB256则二者均为1block。这里须采用分配粒度后的数值，不能只用ptxas实际寄存器数；还须检查线程、warp、shared memory等其他限制。本文“容量3”不表示每个采样时刻每个SM都实际驻留3块。

## 2. 状态机与点角色

源码：[新共享helper](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:36)、[COMPACT分派](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cuh:110)、[公共alias-safe DBL](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:16)。本报告提交后的行号变化应结合commit读取。

原流程：`C=P; B=DBL(P)`位于循环外；规则1..3在循环内另有一份 `A=DBL(A)`。本轮改为：

1. 设置 `A=P; C=P; rule=5`，d/e仍为原始 `e=p−r; d=2r−p`。
2. rule5跳过规则计算和ADD；进入共同DBL：`A=DBL(A)`。
3. 将加倍结果复制给B，再用C恢复A，得到原来的 `(A,B,C)=(P,2P,P)`；d/e不变，rule置0。
4. 后续每轮计算0..3或终结4，一次统一ADD产生T。rule0执行减法规则；1..3再次进入同一个DBL，再按原流程更新点与d/e；rule4复制T给A并退出。

seed时不读B或T。B由初始化DBL定义，T由普通ADD定义后才被规则更新读取；先前素数留下的B/C值不成为新链的输入。DBL始终使用A原位输出，公共compact DBL先读输入，保持别名安全。ADD仍使用与全部输入分离的tx/tz，公共alias-safe ADD不改。

仅添加rule5及seed所需的固定点复制，没有动态bn指针、取地址、设备函数调用ABI。`#pragma unroll 1`约束该循环的展开，但是否实际合并代码由编译后的SASS检验。

## 3. 假设、编译与静态证据

预先预测：共享DBL可能缩小text/存活集合；编译器可能重新拆开seed而保留两份；共享控制也可能增加spill并降低吞吐。CPU模型先验证数学，再用单TU编译反馈区分。

专用TU构建171.1秒，最终CMake仅链接。公共header未改，无其他CUDA TU重编。normal TPI16、host、dispatcher对象SHA与before冻结快照相同。

- MODE10/11完整SASS（含指令/调度编码）和资源记录与before一致。
- MODE12/natural：36,096→27,752条指令；text577,536→444,032 bytes；寄存器174→**161**；stack48不变，spill0/0。
- MODE13/cap168：36,056→27,784条指令；text576,896→444,544 bytes；实际寄存器168→**162**；stack80→48，spill36/28→**0/0**。
- CONSTANT[2]92→68 bytes；没有新增outlined callee或设备调用ABI。

自然与cap策略的实际寄存器数不同，NCU确认两者均按168硬件分配。TPB128下寄存器限制 `floor(65536/(128*168))=3blocks/SM`；自然从176分配/2blocks的档位降至168/3，cap容量仍3。

text减少支持共同调用体实际减少了重复算术代码的解释，不将静态指令条数当作动态执行量。源文件中共享helper只有一份DBL及一份ADD调用；公共p2分支仍独立。SASS完整比较工具采用上一轮已验证的 [compare_stage1_sass.py](D:/code/MPA-OpenCl/tools/bench/compare_stage1_sass.py:1)。本地目录 `stage1_shared_dbl_sass_20261007/`。

## 4. 数学与原生GPU门禁

[独立rule5模型](D:/code/MPA-OpenCl/tools/test/test_prac_single_add.py:76) 将B/T初始设None，seed中误读会失败；seed恰好执行一次，d/e不动，随后所有角色trace与原链逐步一致。8,533 prime/d×两sigma=17,066输入链；六模型共102,396求值，264,538规则步骤、全部中间/终结XZ、ADD/DBL计数及独立ladder一致。

独立normalized Montgomery/别名模型804正例和4,824公共别名对照通过；违规覆盖差分点786项错误，raw REDC≥N的归一化反例保留。CPU结果不代替原生GPU正确性。

同一after SHA、GPU1：

- natural：184完整Q/save；cap168/50ms/C384：3,568完整Q/save；各27类案例包括lcm/choose12、64bit sigma、因子/退化、损坏/恢复checkpoint与目标边界。
- 窗口门禁1,352 Q、8恢复Q、20拒绝；B1=10m/260m的prefix/middle/tail、count32、chunk7/32全部通过。
- 169条窗口结果全部完成，oracle184miss/1,168hit；每份GPU输出仍检查。
- 完整XZ CSV逐字节128项通过，其中48跨切片，包含在128中。

本地证据：`stage1_shared_dbl_{roles,montgomery}_20261007.json`、`stage1_shared_dbl_{natural,cap168,windows}_q_gate_20261007/`。

## 5. 性能实验

GPU1 RTX4060 Laptop/24SM/sm89，N4423/4608/TPI16/TPB128、sigma26/lcm；GPU任务串行，计时期间不编译、不导出大SASS、不运行NCU。before和after不同冻结SHA，baseline来自after的未改内核。每个样本核对前后exe SHA及指数/PRAC缓存命中，两重复反转顺序。

- 固定窗口：B1=260m的tail32，C384/768/1536×register255/168×chunk4/32×三身份×两反序=72样本，6秒采样、两个排除的warmup轮。
- 普通前缀：C1536、B1=10m/260m×register255/168×目标50/100ms×三身份×两反序=48样本，15秒采样、warmup5秒、最后5秒精确s/curve中位。

自然策略新旧驻留容量不同。本轮匹配相同C与提交grid，不声称实际驻留blocks/SM相同；C384/TPI16只有48提交blocks，对24SM而言最多平均2个，无法填满3容量。C768有96、C1536有192，尾波与驻留容量共同影响投影。

事件投影 `W_full*kernel_ms/(W_window*measured_rounds*1000*C)`；墙钟同式替换为measured_wall_ms，包含每轮D2D恢复、launch/synchronize。W_window=8,053，W_full=3,369,476,895。均为子乘积或短时前缀投影，不是完整B1曲线墙钟，不发布Auto B2 T1。

### 5.1 固定窗口：72样本完成

以下是两次反序的墙钟投影中位，单位s/curve，顺序before / after / baseline：

- C384/natural：chunk4为146.281133 / 146.412592 / 148.841212；chunk32为144.848969 / 145.072415 / 147.538268。相对before增加0.090%/0.154%，未改善。
- C768/natural：chunk4为145.406637 / 144.921384 / 148.131541；chunk32为144.592193 / 147.180377 / 147.245066。短片减少0.334%，长片反而增加**1.790%**。
- C1536/natural：chunk4为144.756476 / 132.442854 / 147.436747；chunk32为144.322388 / 135.851131 / 146.979453。相对before减少**8.506%/5.870%**。
- C384/cap168：chunk4为150.268039 / 147.885434 / 150.902103；chunk32为148.802759 / 146.280890 / 149.562467。相对before减少1.586%/1.695%，相对baseline减少1.999%/2.194%。
- C768/cap168：chunk4为147.852827 / 145.595121 / 148.528797；chunk32为152.056962 / 147.680287 / 185.018829。相对before减少1.527%/2.878%，相对baseline减少1.975%/20.181%。
- C1536/cap168：chunk4为132.541905 / **131.390051** / 132.954020；chunk32为139.356660 / 135.021028 / 172.543195。相对before减少**0.869%/3.111%**，相对baseline减少1.176%/21.747%。

本批最佳仍是C1536/cap168短片；不能把自然策略的大幅改善当作相对上一轮最佳的改善。长片相对baseline的累计收益包含此前单点ADD等优化，不全部归因于本轮。事件口径同样保留在原始summary中，和墙钟口径不混合。

C384只有48blocks，容量增加不能填满3blocks/SM。C768有96blocks，容量2可安排2+2波，容量3可能形成3+1的尾波；C1536有192blocks，供给更充分。这是对批量敏感性的解释，实际分配和调度仍须核对NCU，不能当作精确执行时间公式。原生配对审计核对72/72矩阵、每条SHA/缓存/几何及完整first/count/prime/work签名，见本地`stage1_shared_dbl_fixed_20261007/audit.json`。

### 5.2 普通前缀：48样本完成

全部48个普通前缀完成；每配置两次反序，以下为运行级投影中位，单位s/curve，顺序before / after / baseline。

cap168：

- B1=10m、50ms：5.049758 / **5.005864** / 5.063091；相对before减少**0.869%**，相对baseline减少1.130%。
- B1=260m、50ms：132.172455 / **130.988540** / 132.484001；相对before减少**0.896%**，相对baseline减少1.129%。
- B1=10m、100ms：5.084091 / 5.026121 / 5.227701；相对before减少1.140%，相对baseline减少3.856%。
- B1=260m、100ms：134.516425 / 132.078452 / 139.372222；相对before减少1.812%，相对baseline减少5.233%。

natural：

- B1=10m、50ms：5.521315 / 5.044261 / 5.618663；相对before减少8.640%，相对baseline减少10.223%。
- B1=260m、50ms：144.460665 / 132.023499 / 147.028305；相对before减少8.609%，相对baseline减少10.205%。
- B1=10m、100ms：5.515144 / 5.077240 / 5.612085；相对before减少7.940%，相对baseline减少9.530%。
- B1=260m、100ms：144.293517 / 133.669895 / 146.831063；相对before减少7.363%，相对baseline减少8.963%。

自然策略跨过驻留门槛有明显改善，但仍未超过本轮cap168/50ms。50ms的after两次范围为10m的5.005760..5.005967、260m的130.972075..131.005006，before分别5.049666..5.049851、132.154239..132.190671；本批范围不交叠。相对上一轮最佳只改善约0.9%，不能宣称8.6%。100ms没有提高本批最佳。两档B1共用早期素数/规则，不能当作两种独立完整曲线验证。

全部120计时样本（72固定+48前缀）完整重建controls矩阵，无遗漏/重复；每条核对SHA、缓存、几何及原始日志。前缀正常exit1、采样limit/checkpoint-only，无强制终止和最终save；固定窗口不改checkpoint。31项源码/工具SHA留档，只private候选header和独立角色模型与v3不同。

固定采样1,058点/876忙点；前缀1,505点/1,404忙点（利用率≥70%），忙时SM时钟全部1800MHz。温度分别47..67°C/58..67°C，设备总memory.used采样最大323/317MiB；不能当作进程完整分配峰值。本地完整审计为`stage1_shared_dbl_timings_audit_20261007.json`。

### 5.3 管理员NCU：五份采集完成

五份全部管理员exit0、19passes、CSV导出exit0；一次tail32/grid192/TPB128/Device1/4608/TPI16采集，与计时分开。逐项核对冻结exe SHA、wrapper输入与选项、实际demangled MODE及grid/block、相同first/prime/work/warmup。clock/cache-control为none；跳过两个warmup launch后捕获一份。重放时长不用于吞吐。

cap168 before→after：

- 实际寄存器168→162，硬件分配仍168，寄存器限制容量仍3blocks/SM。
- local load sectors688,128→**0**；store525,212→21,468；`32*(load+store)/2^20`为37.028198→**0.655151MiB**，减少约98.23%。baseline仍37.027100MiB。
- eligible warps/scheduler/cycle0.460015→0.482076；issue active33.137713→33.806918%。
- wait/issue-active4.212218→4.157930；no_instruction0.709788→0.425990；short_scoreboard0.349743→0.326006。
- before local load L1命中99.904088%、store95.410615%；after没有local load，工具的load hit rate=0不能解释为cache miss。剩余store命中42.835849%，分母/访问来源已不同，不能单看命中率称为回退。

natural before→after：

- 实际174→161、硬件分配176→168、寄存器限制2→3blocks/SM；达到了此前未达到的门槛。
- local load仍0，store21,288→21,456；累计代理0.649658→0.654785MiB，略增加。
- eligible0.365985→0.490585；issue active31.266621→33.416761%。
- wait3.849825→4.181511、no_instruction0.055955→0.492342、short_scoreboard0.167709→0.218061均增加。容量/eligible提高不表示所有stall减少，单个stall比值也不是墙钟占比。

cap的local访问减少支持spill消失，但对应最佳吞吐只改善约0.9%；不能等比例换算成98%总时间收益。自然策略的驻留容量增加支持大批量收益的解释，仍保留C384无改善及C768长片回退。该恢复窗口DRAM throughput仅0.000068%..0.000138% of peak，没有DRAM吞吐饱和证据；不推广为整段Stage1 CPU准备成本结论。累计local代理不是显存容量/DRAM/PCIe字节，不乘19passes。

本地证据：`stage1_shared_dbl_ncu_{before168,after168,baseline168,before255,after255}_20261007/`及`stage1_shared_dbl_ncu_analysis_20261007.json`。

### 5.4 采用范围与下一项

v4作为GPU1/N4423/C1536/lcm、显式single-compact/cap168/50ms的暂定最佳实验选择；默认TPI/TPB/算法/寄存器/目标不改。原生门禁通过，128份完整XZ逐字节一致；性能仅为短时投影，不代表完整10m/260m批次时间或Auto B2 T1。

下一项优先将此候选以显式选项扩展至4608/TPI32，并按TPI16 C384/768/1536对应TPI32 C192/384/768匹配提交grid。先明确候选分派与正确性门禁，再比较寄存器分配/驻留容量和吞吐；匹配grid不宣称驻留blocks/SM相等。4423-bit默认TPI16保持原规则。

## 6. 计算量与存储量

每个奇素数重复仍有一次seed DBL及原有循环DBL；ADD=4M+2S，DBL=3M+2S。本CGBN平方按模乘等价计，计划总量 `W=6A+5D` 不变。**减少的是代码体复制和寄存器/溢出代价，没有少算任何模乘。**

共享初始化源码额外复制A到B和C到A，合计四个bn坐标赋值；容器B=4608时逻辑值每bn576 bytes，即每次seed2,304 bytes的源码逻辑复制。它们是线程私有寄存器值，不能称为GPU显存或PCIe拷贝，也不能保证实际产生同等数量MOV；编译器可重命名/复用。

显式curve/seed各 `7CB/8` bytes、计划 `16P` bytes、边界逻辑读写 `6CB/8*launches`，全部未变。C1536/B4608，data/seed各6,193,152 bytes；tail32/chunk4的8launch边界42,467,328 bytes，chunk32单launch为5,308,416 bytes。没有新增host/device传输量。

stack48 bytes不是spill48 bytes；本轮两策略spill均0，但仍有静态stack和可能的local访问。local sectors×32若用于累计访问代理，须区分L1、显存容量与DRAM/PCIe，不乘NCU重放pass。GPU采样memory.used只记录采样设备总占用，不是进程完整分配峰。

## 7. 复现

```powershell
python tools/test/test_prac_single_add.py --output docs/data/shared_roles.json
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac_single_add\.cu$' -Jobs 6

python tools/bench/bench_stage1_prac_versions.py `
  --before build_cuda_cmake/prac/after_rule_sentinel_20261007/ecm_cuda.exe `
  --after build_cuda_cmake/prac/ecm_cuda.exe --mode windows --curves 384 768 1536 `
  --registers 255 168 --b1 260000000 --chunks 4 32 --seconds 6 `
  --exp-cache build_cuda_cmake/prac --output docs/data/shared_fixed
python tools/bench/bench_stage1_prac_versions.py `
  --before build_cuda_cmake/prac/after_rule_sentinel_20261007/ecm_cuda.exe `
  --after build_cuda_cmake/prac/ecm_cuda.exe --mode prefix --curves 1536 `
  --registers 255 168 --b1 10000000 260000000 --target-ms 50 100 `
  --seconds 15 --warmup 5 --exp-cache build_cuda_cmake/prac --output docs/data/shared_prefix
```

本轮仍须指定single-compact和register255/168；源码版本和exe SHA必记。下一轮覆盖exe之前保留after快照。门禁/计时使用独立新目录，不恢复生产checkpoint或发布最终save。
