# CUDA Stage1 PRAC 构建拆分计划

本机、开发与发布构建默认包含 PARAM0 resident／PRAC，默认点乘链为 ladder。可在 TOML 中选择 PRAC。0.63.7 优化前基线使用标准 PRAC、搜索级别 50 和三片滚动提交。构建命令及覆盖范围见 [构建说明](../usage/BUILD.md)，测试口径见 [Stage1 基线](../performance/STAGE1.md#0637-优化前基线)。下文为进一步裁剪构建体积和依赖范围的计划。

## 哪些内核需要实例化

标准 PRAC 的容器位宽和 TPI 由 CUDA 模板固定。全位宽支持需要给每个受支持的 PARAM0 容器生成标准实例。实例中有域导入、域导出、resident ladder 和 PRAC 四种基础模式。当前还在 2560–8192 位实例化自然寄存器档，在 4608/TPI16 实例化多种寄存器额度和点公式，并生成部分 alternate TPI、single-add 和常量专版。依据见 [生成清单](../../tools/build/internal/cuda_modules.cmake#L62)、[模板分派](../../src/ecm/stage1/montgomery/cuda/prac/cgbn_stage1_prac_kernel.cuh#L181) 和 [运行时路由](../../src/ecm/stage1/montgomery/cuda/prac/dispatch.cu#L37)。

PRAC 搜索级别产生的是运行时读取的链描述符，切片深度控制主机提交队列。两者使用同一组已编译的设备内核。普通内核的复用规则见[构建说明](../usage/BUILD.md#默认策略)。

## 裁剪顺序

1. **已完成 PRAC 开关的构建隔离。** 默认 local 固定输出，PRAC 通过链接选择实现，普通 CUDA 对象继续经过原有校验。关闭时保留已生成的 PRAC 分发表和对象，便于下次开启复用。
2. **每档保留一条通用 PRAC 主实现。** 对照 0.63.7 的同位宽、B1、sigma、批量和主循环边界。候选版本在两张卡及适用位宽有稳定收益，再用它替换旧标准实现。旧 PRAC 继续留在 Git 和冻结基准中。若两个方案分别在不同寄存器档或设备上胜出，才考虑按实测分派并随本机构建保留两份。
3. **将实验变体移到单独构建开关。** 优先考虑单模数常量内核、替代 TPI 及尚无生产优势的寄存器变体。保留常规非 PRAC ladder。resident ladder 也不应与普通 ladder 混为一类，移出默认构建前需核对已有任务开关和检查点恢复。验证全位宽 device 覆盖、明确拒绝未编译的实验变体、两卡同 sigma 末点与因子，以及中断续跑。

本轮构建验收使用 Windows NMake、CUDA 13.3.73、MSVC 14.51.36231、sm89。384 位完整切换中，普通 CUDA 的 9 个对象及编译记录在切换后保持相同内容和时间戳。首次开启仅编译 3 个 PRAC CUDA 单元和 PRAC 主机 checkpoint 实现。关闭后再次开启复用 12 个 CUDA 对象。各次最终 NMake 均只处理需要的主机编译和链接。另对全档配置比较 125 个普通实例的编译命令、NMake 标志和响应文件，三次配置一致。全档未执行编译，本轮未执行 GPU 曲线。原始日志位于 `build_dev/s1_prac_cycle_61784f4d/` 和 `build_dev/s1_prac_all_rules_c5919cab/`。
