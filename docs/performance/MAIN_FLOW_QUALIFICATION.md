# 主流程重构验收

当前源码与local程序版本0.45.24；对照为P0冻结的0.44.9。职责、算法和文件合同以[架构总览](../architecture/OVERVIEW.md)、[Stage1](../architecture/STAGE1.md)、[Stage2](../architecture/STAGE2.md)为准。本文说明实际验证覆盖，不替代功能规范。

## 环境、程序与证据

本机LOCAL，CUDA13.3/MSVC14.51.36231/sm89、相同Zen3 GMP；Stage1完整32档、PARAM0/2/3，PRAC默认关闭；Stage2生产引擎。全位宽复用经验证的原local叶build_cuda_cmake，未复制旧对象或混用release；新默认目录仍按role/profile隔离。

GPU1：NVIDIA GeForce RTX 4060 Laptop GPU，UUID GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d；主机 AMD Ryzen AI 9 HX 370 w/ Radeon 890M，24逻辑处理器。GPU功耗上限状态按原始NVML记录：None mW，status=3。不可用不记为0；不更改功耗或频率。设备遥测每250ms采样，其他进程快照、驱动/环境在原始证据保留。

完整[资格清单](../../data/experiments/main_flow_refactor_implementation_20261021/p7b/qualification.json)绑定源码、工具、程序和证据；原始日志/样本在data/experiments/main_flow_refactor_implementation_20261021/p7b，按仓库规则忽略，须随本机证据保留。

| 程序 | 最终SHA256 |
| --- | --- |
| full32/ecm_cuda.exe | 4bfd667b8ac4d76eecd5581846a7acbe5ef478ebb1f0281993d5a8206dab6f71 |
| stage2/ecm_cuda_stage2.exe | 7e373437e61bd5a695b8e2e4168735c87bcedd1cb013200294815e51f5440f62 |
| opencl/ecm.exe | b5649020cc17bf15e84abb1bd2116ec2849e88dc56bca05f2b607c5b04fc5af4 |
| gui/ecm_gui.exe | 7688b3939633aa749efdf2f91391a3b4d380c86c7d1505622856544c8ee6d5bb |
| cpu2/ecm_cpu_stage2.exe | 007e2fe52a69186fab325e7f412800785ac844b14e6b31ffc6fb98da183dd4cf |

## 功能与编译边界

完整32档注册260项；Montgomery93项对照；Edwards131项独立GMP用例，共537曲线及96个边界模式，包含停止/恢复和自适应重启。Stage2独立算术1536模数、152064比较bad=0，14对完整运行、4规划及20拒绝项。[完整原生门禁](../../data/experiments/main_flow_refactor_implementation_20261021/p7b/native/result.json)。

配置读/写、CPU曲线数学/编码、checkpoint、save、data、result、tune.toml、screen九类正文加入无运行影响的C++编译探针，经过正式local脚本；不改设备合同或CUDA runtime实现。保存并恢复原始源码字节，CUDA对象31+2的SHA保持。[实际增量日志](../../data/experiments/main_flow_refactor_implementation_20261021/p7b/incremental_03/result.json)：

| 构建阶段 | 程序 | CXX | CU |
| --- | --- | --- | --- |
| edited | stage1 | 10 | 0 |
| edited | stage2 | 9 | 0 |
| edited | gui | 6 | 0 |
| restored | stage1 | 10 | 0 |
| restored | stage2 | 9 | 0 |
| restored | gui | 6 | 0 |
| repeat | stage1 | 0 | 0 |
| repeat | stage2 | 0 | 0 |
| repeat | gui | 0 | 0 |

重复阶段同时要求零链接。每轮探针使用独立正文身份，要求真实编译；随后逐文件恢复并校验原始字节，任何未完成恢复均拒绝通过。原始证据保留全部失败与成功记录。

353个生产源码/接口文件、14个公开头、753条项目内include边无环，公开头无src依赖。17个实际主机CPP无CUDA SDK独立编译；真实设备同步/回读和显存查询保持SDK依赖。身份55项、MSVC退役依赖9项、task8项、queue13项、process30项、storage18项通过。

OpenCL172份资源、2缓存/4数学输出；CPU Stage2互操作27项、结果23项、Prime95交接83项；Stage2规划8项、工作流12项及停止恢复通过。GUI配置/生命周期/日志/结果/生成分别33/54/85/26/110项；Controller10场景44检查及真实两阶段混合队列，逐阶段2条内部结果和2批NF/CRC通过。GUI是无窗口验证，不包括绘制和交互资格。

修改探针前的7个实际测试程序保存在pre_incremental_binaries；最终完整CUDA门禁和性能绑定恢复后程序，GUI恢复后自检另留日志。各证据自身记录测试程序身份，不把不同链接产物混为同一二进制。

## 代表性能范围

仅8个代表样例：M2203的Edwards NAF/binary及Montgomery PARAM0/2/3，Stage2的M127/M2203/M16381、B1=20、B2=13230、D=210普通运算。Stage1 B1=10000，16曲线，除PARAM2首sigma=1000000外首sigma=26；完整输入、环境及独立点oracle在冻结[计划](../../data/experiments/main_flow_refactor_implementation_20261021/p7b/performance_plan_02.json)。未测试的性能形状不外推为已验收。

每侧2次预热，每项21对，AB/BA交替；每个样本验证点/结果和工作量。两种计时边界分别统计，不相加：墙钟为subprocess.run前后；内部为Montgomery CUDA事件、Edwards算术API墙钟或Stage2 full_wall.total。阈值保持1.05；配对比值中位数及10000次固定seed bootstrap的95%区间上界均须≤1.05。全部慢样本保留。

| 样例 | 墙钟中位比 | 墙钟95%上界 | 内部中位比 | 内部95%上界 |
| --- | --- | --- | --- | --- |
| stage1_m2203_ed | 1.0053 | 1.0098 | 1.0011 | 1.0087 |
| stage1_m2203_ed-binary | 1.0022 | 1.0106 | 0.9999 | 1.0032 |
| stage1_m2203_mont0 | 0.9964 | 1.0156 | 1.0001 | 1.0003 |
| stage1_m2203_mont2 | 0.9981 | 1.0085 | 0.9997 | 1.0006 |
| stage1_m2203_mont3 | 1.0028 | 1.0046 | 1.0006 | 1.0010 |
| stage2_m127_ordinary | 0.9971 | 1.0214 | 0.9971 | 1.0030 |
| stage2_m2203_ordinary | 0.9832 | 1.0329 | 0.9912 | 1.0012 |
| stage2_m16381_ordinary | 0.9988 | 1.0081 | 1.0000 | 1.0008 |

[性能结果](../../data/experiments/main_flow_refactor_implementation_20261021/p7b/performance_02/result.json)包含每项样本、程序、helper、输入和遥测摘要。功能验证中的耗时不作为性能样本；不声明全矩阵提升或把组件/嵌套耗时相加。

## 覆盖限制

- 仅GPU1/sm89本机local；未运行release、多架构或其他实体GPU/PTX-JIT资格。
- 可选PRAC保留0.45.16受限构建资格；最终全位宽默认不编译PRAC。
- CPU Stage2本轮验证libecm，未重建GWPoly；OpenCL不新增便携发布资格。
- 每进程仍限一个活跃任务/引擎；未新增跨后端checkpoint互换、配置热重载或同进程多会话。
- 群阶诊断仍有已说明的sigma/点阶/最小界限限制；组织重构不修订其数学能力。
