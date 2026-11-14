# CUDA Stage1 PRAC 构建拆分计划

本机、开发与发布构建默认包含 PARAM0 resident／PRAC，默认点乘链为 ladder。可在 TOML 中选择 PRAC。0.63.7 优化前基线使用标准 PRAC、搜索级别 50 和三片滚动提交。构建命令及覆盖范围见 [构建说明](../usage/BUILD.md)，测试口径见 [Stage1 基线](../performance/STAGE1.md#0637-优化前基线)。下文为进一步裁剪构建体积和依赖范围的计划。

## 哪些内核需要实例化

标准 PRAC 的容器位宽和 TPI 由 CUDA 模板固定。全位宽支持需要给每个受支持的 PARAM0 容器生成标准实例。实例中有域导入、域导出、resident ladder 和 PRAC 四种基础模式。当前还在 2560–8192 位实例化自然寄存器档，在 4608/TPI16 实例化多种寄存器额度和点公式，并生成部分 alternate TPI、single-add 和常量专版。依据见 [生成清单](../../tools/build/internal/cuda_modules.cmake#L62)、[模板分派](../../src/ecm/stage1/montgomery/cuda/prac/cgbn_stage1_prac_kernel.cuh#L181) 和 [运行时路由](../../src/ecm/stage1/montgomery/cuda/prac/dispatch.cu#L37)。

PRAC 搜索级别产生的是运行时读取的链描述符，切片深度控制主机提交队列。两者使用同一组已编译的设备内核。扩大构建覆盖范围时只需新增缺失的 PRAC 容器；现有流程从 PRAC 关闭切换到打开会选择另一构建 profile，且 `ECM_CUDA_ENABLE_PRAC` 目前是目标级宏，因此普通模板也会重编。这里的额外编译来自构建隔离和编译命令变化。

## 裁剪顺序

1. **先减少构建选项的失效范围。** 将 PRAC 能力宏限制在引用它的分派、恢复和日志单元，保持普通 Edwards、PARAM0/2/3 ladder 和 NTT 的编译命令稳定。验证跨 profile 复用前核对编译器、架构、源闭包、对象依赖和目标链接合同。不以跳过校验换取编译速度。
2. **每档保留一条通用 PRAC 主实现。** 对照 0.63.7 的同位宽、B1、sigma、批量和主循环边界。候选版本在两张卡及适用位宽有稳定收益，再用它替换旧标准实现。旧 PRAC 继续留在 Git 和冻结基准中。若两个方案分别在不同寄存器档或设备上胜出，才考虑按实测分派并随本机构建保留两份。
3. **将实验变体移到单独构建开关。** 优先考虑单模数常量内核、替代 TPI 及尚无生产优势的寄存器变体。保留常规非 PRAC ladder。resident ladder 也不应与普通 ladder 混为一类，移出默认构建前需核对已有任务开关和检查点恢复。验证全位宽 device 覆盖、明确拒绝未编译的实验变体、两卡同 sigma 末点与因子，以及中断续跑。
