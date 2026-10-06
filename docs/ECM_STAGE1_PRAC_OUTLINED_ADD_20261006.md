# Stage1 PRAC：共享 xADD 函数实验

日期：2026-10-06。基线 `1bb2124`，接续 [固定子乘积切片](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_FIXED_SLICING_20261006.md:1)。目标是检验减少点运算重复展开是否改善指令供给，保留长片与短片对照。

## 1. 动机与假设

上一阶段，N4423/4608/TPI16/TPB128 的 cap168 长片比短片慢；同一32-record算术的 no_instruction / issue_active 为2.534028，而自然策略为0.059075。单个PRAC全局函数的静态文本跨度约0.86MiB。计数支持检查指令供给与调度，但不能将no_instruction全部认定为指令缓存miss。

本轮三个可检验的假设：共享xADD缩小重复代码后，长片计数/吞吐应改善；调用ABI新增局部存储、寄存器或访存成本可能抵消收益；若片长收益主要来自执行同步，缩小代码也可能无法消除退化。只提取xADD，DBL沿用baseline，以保持单一主要变更。

## 2. 实现

`ECM_PRAC_VARIANT=outline-add` 选择新MODE8（自然寄存器上限255）或MODE9（上限168）；仅提供4608/TPI16，要求已有 `ECM_PRAC_REG_TARGET=255|168`。正常最小容器与TPI分档规则沿用；候选缺失时报错，不改用更大容器。

使用 `__noinline__` 共享函数包装原有 `prac_add`，三处PRAC调用通过编译期选择进入共享函数。原normalized 4M+2S（当前CGBN S=M）及输出别名规则沿用，输出可以覆盖输入或difference。不是减少模乘次数，也没有改变PRAC计划、标量、域表示或slice/checkpoint格式。

源文件：[共享函数及选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:63)、[PRAC调用点](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:143)、[主机参数](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:61)。

## 3. 容量与访存模型

全局曲线数据仍为 `7*C*B/8`，计划为 `16*P` bytes。片间逻辑读N/a24/AX/AZ、写AX/AZ为 `6*C*B/8`；窗口seed额外 `7*C*B/8`，每轮恢复一次。提取函数不会增加这些显式分配或PCIe传输。

CUDA调用ABI可能把寄存器中的多精度值放入thread local memory，并引入额外加载/写回。设ptxas栈帧L、线程数C*TPI，`L*C*TPI`是按每线程栈帧计算的逻辑容量代理，不是cudaMalloc清单、物理显存承诺或进程峰值；栈帧、spill字节和运行时local访存需分别记录。4608/TPI16每个多精度值每线程有9个32位limb，地址可取的引用参数可能影响标量化。

静态代码必须记录全局kernel及可达共享callee，不能仅报告kernel变小而遗漏callee。本机构建中，callee是入口text内的局部函数，已包含在该text跨度内；记录其ELF符号，避免重复相加。SASS静态条数不是动态执行条数或缓存工作集；cycles和访存收益由运行计数验证。

## 4. 构建与身份

6个TU并行编译完成；critical path为4608/TPI16，643.0s，serial sum1097.8s。新增两个实例增加该TU工作，主机TU17.2s。旧exe、DLL、CMake配置及原PRAC头保留于本地 `build_cuda_cmake/prac/before_outline_20261006/`。

新exe SHA256：

```text
eec7b2e06696ebb2e677196c7a7c454a3771878569762b6de7152d05f8a57fc9
```

GPU1、RTX4060 Laptop、sm89、24SM、TPB128；本轮候选N4423/4608/TPI16。ptxas资源与完整text清点如下：

| 策略 | register/thread | stack bytes/thread | spill store/load bytes | 静态指令条数 | text bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline natural MODE4 | 172 | 48 | 0 / 0 | 56352 | 901632 |
| baseline cap168 MODE5 | 168 | 80 | 36 / 28 | 55928 | 894848 |
| outline-add natural MODE8 | 189 | 304 | 0 / 0 | 47960 | 767360 |
| outline-add cap168 MODE9 | 168 | 344 | 48 / 40 | 47472 | 759552 |

自然策略text缩小14.89%，cap168缩小15.12%。cuobjdump显示两个 `$_Z20kernel...$_Z17prac_add_outlined...` 局部函数符号，分别内嵌于MODE8/9；上述text已经包含相应函数及内部子程序，没有另加callee字节。原MODE4/5的条数、跨度及ptxas资源与基线相同。

C1536/TPI16时，栈帧逻辑代理分别1.125/1.875/7.125/8.0625MiB；相对baseline，候选每线程多256/264bytes。ptxas的spill字节是编译报告量，不是运行时累计流量。设备实际分配寄存器和驻留由CUDA/NCU核验，不能仅用源码上限作结论。

本地证据：`build_cuda_cmake/prac/par_nvcc/cgbn_stage1_prac_tpi16.log`、`docs/data/stage1_outline_sass_full_20261006/`。工具：[完整text与ELF符号清点](D:/code/MPA-OpenCl/tools/bench/inventory_stage1_sass.py:17)。导出时检查对象SHA未改变，独立旧对象fixture复现原56352/55928条。

## 5. 正确性

同一SHAeec7二进制的完整Q/save门禁：自然策略184条比较，cap168/target50ms/C384为3568条比较，均通过。包含CPU/GMP、lcm/choose12、64位sigma、checkpoint正常恢复与损坏重建、退化因子、合法50/500ms目标及非法目标拒绝。计数包含ladder/resident对照及部分baseline边界检查，不全部是outline-add运行。

证据：本地 `stage1_outline_natural_q_gate_20261006/summary.json` 和 `stage1_outline_cap168_q_gate_20261006/summary.json`。窗口独立整数ladder oracle及逐字节比较另行记录。

完整窗口重跑exit0：1064条窗口Q、8条checkpoint恢复Q、13个非法参数拒绝全部通过。覆盖2203/4423/8191、显式TPI32对照、lcm/choose12、64位sigma、B1=10m/260m前/中/后32-record子乘积以及chunk7/32。同策略跨切片36次逐字节相等；额外审计按相同输入/容器/TPI分组，跨variant/register/chunk共有92次完整X/Z CSV逐字节比较通过，包含前述36次，不相加计数。窗口隔离检查也确认既有checkpoint不被改变、没有最终save。

证据：本地 `stage1_outline_windows_final_q_gate_20261006/summary.json`、`bitwise_audit.json`。审计工具：[输入分组与逐字节验证](D:/code/MPA-OpenCl/tools/test/audit_cuda_prac_windows.py:10)。

首轮窗口门禁完成数值循环后，在新增resident拒绝用例的错误信息断言失败：外层首先返回 `PRAC window/variant requires ECM_GPU_STAGE1_ALGO=prac`，测试期待内层 `variant requires PRAC`。修正测试期望与实际调用边界一致，不修改native拒绝行为；保留失败日志，完整重跑使用新目录。拒绝检查同时清除继承的chunk/target环境变量，避免外部配置污染用例。[拒绝门禁](D:/code/MPA-OpenCl/tools/test/test_cuda_prac_windows.py:98)

## 6. 固定范围与生产采样

### 6.1 固定尾部，48个样本

GPU1、N4423/4608/TPI16/TPB128、B1=260m、sigma26、lcm。同一32-record tail：prime259999307..259999991、1193次xADD与179次DBL，W=8053；完整计划F=3369476895。C384/768/1536 × baseline/outline-add × natural/cap168 × chunk4/32 ×两次反序。全部48样本完成并命中指数/PRAC缓存。每次窗口至少6s，排除前两恢复轮，关闭坐标导出；同一C的所有配置使用相同子乘积和seed。采样投影不是完整生产曲线墙钟。

以下为两次中位数，每格为CUDA event / 有效轮墙钟投影，单位s/curve：

| C / registers | baseline chunk4 | baseline chunk32 | outline-add chunk4 | outline-add chunk32 |
| --- | ---: | ---: | ---: | ---: |
| 384 / natural | 147.860657 / 148.865670 | 147.290130 / 147.513279 | 209.046782 / 210.160006 | 208.454302 / 208.689208 |
| 384 / cap168 | 149.924796 / 151.022671 | 149.324722 / 149.571633 | 208.742998 / 209.831541 | 208.198279 / 208.440741 |
| 768 / natural | 147.482000 / 148.062030 | 147.130129 / 147.244421 | 208.739061 / 209.308382 | 208.406734 / 208.511162 |
| 768 / cap168 | 147.875286 / 148.500078 | 185.185868 / 185.297141 | 270.951290 / 271.536210 | 311.899711 / 312.009244 |
| 1536 / natural | 147.088183 / 147.386181 | 146.895041 / 146.952445 | 208.600877 / 208.905569 | 208.406562 / 208.463125 |
| 1536 / cap168 | 132.663191 / 132.976617 | 172.969246 / 173.023646 | 243.965303 / 244.258859 | 284.147431 / 284.202552 |

所有同配置比较都变慢。C1536 cap168短片墙钟增加83.686%，长片增加64.256%。候选自然策略在短/长片均约208s/curve；候选cap168长片仍明显退化，没有消除批量/片长敏感性。长片cap168两次候选为288.131357/280.163505 event s/curve，不能只报告最快一次。

显式seed/恢复量仍为 `7*C*4608/8` bytes；C1536为6193152bytes（5.90625MiB）。chunk4有8次launch，边界逻辑量42467328bytes/round（40.5MiB）；chunk32一次，5308416bytes（5.0625MiB）。该逻辑量不是PCIe传输量。两variant的这些量相同，新增成本需检查调用栈与实际local访存。

证据：本地 `stage1_outline_slicing_matrix_20261006/summary.json`、`gpu.csv`。[同子乘积矩阵工具](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_slicing.py:39)

### 6.2 普通生产前缀

同二进制C1536/grid192、B1=10m/260m、resident及natural/cap168的baseline/outline-add，PRAC目标50/100ms，两次反序，共36样本。15s采样、排除前5s、最后5s报告中位数；所有指数缓存命中，全部PRAC计划缓存命中。所有样本均到达sample limit后保存checkpoint、exit1，无强制终止及最终save。

每格为两次投影中位数，单位s/curve，列名为B1 / target：

| 配置 | 10m / 50ms | 10m / 100ms | 260m / 50ms | 260m / 100ms |
| --- | ---: | ---: | ---: | ---: |
| resident | — | 5.365666 | — | 139.515564 |
| baseline natural | 5.619232 | 5.613139 | 147.074992 | 146.840805 |
| baseline cap168 | 5.066622 | 5.212626 | 132.525922 | 138.880261 |
| outline-add natural | 7.909046 | 7.890837 | 206.549131 | 206.374953 |
| outline-add cap168 | 8.891762 | 9.516921 | 231.220463 | 249.092285 |

resident使用自己的bit切片，并不使用PRAC target，放在100ms列只是对照展示。两档B1的候选最佳均为natural/100ms，较既有最佳baseline cap168/50ms投影分别增加55.742%/55.724%，没有生产收益。固定同子乘积与普通生产前缀结论一致，不采用该候选为默认。

本地证据：`stage1_outline_prefix_matrix_20261006/summary.json`；[生产前缀工具](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_variants.py:48)。未完成真实生产中/后段点或整条曲线，因此这些数据不能发布为Auto B2的T1实测值。

### 6.3 设备遥测

GPU1每500ms整卡采样。固定矩阵754行，其中GPU utilization≥90%的571行，SM频率均1800MHz、温度55..62°C，记录显存最大323MiB；普通前缀1159行，其中忙时1058行，频率也全部1800MHz、温度56..65°C，显存最大317MiB。每份记录有3个未知数值（初始采样），按字段单独排除；未把未知值当0。utilization采样窗口有滞后，部分忙时used_MiB为0，不能把整卡遥测当逐分配或进程峰值证明，也不能将其与模块逻辑容量相加。没有观察到忙时降频证据。

证据：本地矩阵`gpu.csv`、`stage1_outline_prefix_matrix_20261006_telemetry/gpu.csv`、`stage1_outline_telemetry_20261006.json`。计时期间没有编译或NCU重放；采集在全部基准结束后另行串行运行。

### 6.4 管理员NCU：调用边界与local访存

四份采集/CSV导出均exit0，baseline各19passes、候选各20passes。相同SHAeec7、GPU1、grid192、TPB128、C1536、sigma26、tail32、W8053、一次launch覆盖相同子乘积；跳过INIT后捕获第一恢复轮，clock-control/cache-control均none。kernel模板MODE4/5/8/9及native几何分别核验。以下时间相关计数属于重放，不能取代前述未剖析计时。

| 指标 | baseline natural | baseline cap168 | outline natural | outline cap168 |
| --- | ---: | ---: | ---: | ---: |
| 实际分配register/thread | 176 | 168 | 192 | 168 |
| register-limited blocks/SM | 2 | 3 | 2 | 3 |
| eligible warps/scheduler/cycle | 0.357321 | 0.352005 | 0.514377 | 0.429885 |
| issue active % | 30.661264 | 27.060711 | 42.990247 | 34.025764 |
| wait / issue_active | 3.889475 | 4.006981 | 1.839728 | 1.862053 |
| no_instruction / issue_active | 0.058959 | 2.879844 | 0.807287 | 3.812227 |
| short_scoreboard / issue_active | 0.274087 | 0.364417 | 0.418907 | 0.258579 |
| long_scoreboard / issue_active | 0.000063 | 0.000115 | 0.055404 | 0.024083 |
| local load sectors | 0 | 688128 | 3336255756 | 3337238796 |
| local store sectors | 21360 | 525004 | 315624844 | 315008988 |
| local load L1 hit % | — | 99.913388 | 99.581969 | 98.992501 |
| local store L1 hit % | 68.314607 | 95.442320 | 96.847565 | 86.173323 |
| 32×(load+store sectors)，MiB | 0.651855 | 37.021851 | 111446.551514 | 111457.757080 |
| DRAM throughput，峰值百分比 | 0.000095 | 0.000090 | 0.000035 | 0.000038 |

这里sectors是 `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st.sum`，32bytes/sector。字节乘积表示所捕获kernel的累计local sector请求代理，重复访问与缓存请求都计入，**不是唯一数据字节、DRAM流量、PCIe传输或显存容量**；也不乘19/20重放pass数。natural没有local load，hit值0不代表所有加载miss，所以表中写“—”。

候选约108.83/108.85GiB累计local sector字节，大部分load命中L1；baseline仅0.65/37.02MiB。自然候选的ptxas spill仍0，但引用参数跨device-call需要地址可取存储：静态LDL由baseline natural的0增至候选1937条，与动态local请求大增一致。不能把“0 spill”误读为没有local-memory成本，亦不能仅看DRAM低吞吐排除访存/地址依赖。

两种寄存器策略的blocks/SM分别保持2/3，自然候选并未因寄存器189→实际192而减少驻留block。代码缩小15%也没有改善no_instruction比值。候选issue active更高、wait比值更低，却更慢：指令组成、动态访存/地址操作和该比值的分母都改变，不能把发射率提高当作单位子乘积效率提高。这些stall比值也不是墙钟时间百分比，不能相加作耗时归因。

结论：本轮数据拒绝“通用引用参数的整xADD outline能通过缩小代码改善吞吐”的假设；支持调用ABI/local访问新增成本抵消体积收益。只能确认本实现、此位宽/设备/批量范围的结论，未直接测得指令缓存miss数，也没有量化local访问独自贡献了多少墙钟增量。

本地证据：四个 `stage1_outline_ncu_{baseline|outline-add}_{255|168}_20261006/` 的 `command.json`、`app.log`、`run.log`、`trace.ncu-rep`、`metrics.csv`、`quantitative.json`。[采集与显式local counters](D:/code/MPA-OpenCl/tools/bench/profile_cuda_stage1.py:75)、[导出归纳](D:/code/MPA-OpenCl/tools/bench/summarize_stage1_profile.py:28)。

## 7. 使用与复现

```powershell
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 255 `
  --variant outline-add --device 1 --output docs/data/my_outline_q
python tools/test/test_cuda_prac_windows.py --outline-add --production `
  --production-counts 32 --production-chunks 7 32 --device 1 `
  --output docs/data/my_outline_windows
python tools/bench/bench_stage1_prac_slicing.py --curves 1536 `
  --variants baseline outline-add --registers 255 168 --chunks 4 32 `
  --count 32 --b1 260000000 --seconds 6 --warmup 2 --repeats 2 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_outline_slicing
python tools/bench/bench_stage1_prac_variants.py `
  --configs resident natural cap168 outline outline168 --target-ms 50 100 `
  --b1 10000000 260000000 --curves 1536 --seconds 15 --warmup 5 `
  --repeats 2 --device 1 --exp-cache build_cuda_cmake/prac `
  --output docs/data/my_outline_prefix
python tools/bench/profile_cuda_stage1.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --b1 260000000 --curves 1536 --tpi 16 --registers 168 `
  --variant outline-add --window tail --window-count 32 --window-chunk 32 `
  --window-warmup 2 --launch-skip 1 --device 1 --seconds 6 --memory `
  --exp-cache build_cuda_cmake/prac --prepare-only --output docs/data/my_outline_ncu
# 上一步创建admin.ps1；执行时需要管理员权限，然后以--collect-only导出。
```

## 8. 采用决定

不采用outline-add为默认或生产推荐。保留4608/TPI16显式实验入口和可复现证据，用于区分代码展开与CUDA调用边界成本。默认算法、TPI、寄存器策略及100ms目标沿用原配置。

下一候选优先统一内联xADD调用点及固定输入/输出角色，减少重复展开时避免地址可取引用跨调用。统一点角色可能增加临时多精度值与寄存器压力，不能预设它一定更快；先比较静态资源，再用相同子乘积、跨切片逐字节门禁和普通生产前缀验证。也可进一步缩小共享算术边界，但不再直接推广本轮通用引用参数的整xADD函数。
