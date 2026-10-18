# GUI worker 输出与 worktodo 生成器开发草稿

版本：0.39.0。本文保留本次开发的判断和验证记录；当前使用方式以[GUI 使用说明](../usage/GUI.md)为准。功能完成后由用户决定如何处理本草稿，不自动删除。

## 需求与依据

- 每个 worker 输出窗展示应用程序版本。Stage1 已输出 `ECM-ELF x.y.z (Stage1)` 横幅，但 GUI 原先把它归为原始输出；CUDA Stage2 的 GUI 运行原先没有对应横幅。GUI 版本和工作程序版本须区分，避免所选二进制不是当前源码构建时误报。
- Stage2 状态已有 B1、B2、S、D、P；`stage2_modulus` 另输出本曲线实际 `carrier_exponent`。0 表示未使用额外梅森承载；读取前状态未知，不等同于 0。
- 生成器输出是 ECMSTAGE2 worktodo 项。当前 Stage1 save 命名及队列前缀约定见[Save 架构](../architecture/SAVE.md)，`e`、`m0`、`m2`、`m3` 是曲线构造身份；执行目标由 INI 决定。Stage2 按 save 内的 METHOD/PARAM 计算，不能仅靠重命名文件更换曲线。

## 实现轮次

GUI 在每个输出窗固定显示自身版本，并从 worker 实际启动横幅提取其版本。Stage2 仅在 GUI 事件模式补充横幅，不改变普通 tune/规划输出。状态解析从 `stage2_modulus` 取得实际承载指数，并在新曲线开始时清除上一条曲线的承载状态。

worktodo 生成器增加曲线类型选择，默认从目标 worker 的 Stage1 配置读取。默认格式改为 `{curve}_{target}_{b1}.save`，自定义格式规范为所选曲线前缀。梅森任务 target 使用原始指数；其他任务使用现有无 GMP 位宽估算，生成器发出核对提示。`{b1}` 保留 PrimeNet 输入或 GUI 覆盖值的拼写。核心 Stage1 自行生成 save 时另对数值 B1 做压缩，二者原先被文档混写；现按各自实际行为说明。

Montgomery GPU 推荐批量按所选 param 查询实际 worker/INI 的 `--gpu-info`；Edwards 和 CPU 目标不使用该 Montgomery 档位，保留任务原曲线数。Stage2 源码仅读取 Edwards 和 Montgomery PARAM0 save，CPU Stage1 不支持 PARAM2/3；生成器按目标 worker 限制曲线类型。

代码入口：[GUI 面板](../../src/gui/app.cpp)、[日志解析](../../src/gui/log_parse.cpp)、[生成器](../../src/gui/worktodo_gen.cpp)、[Stage2 worker](../../src/core/ecm_cuda_stage2_main.cpp)、[公共版本](../../src/core/ecm_program_version.h)。

## 验证

GUI 构建完成，自检 36/36；生成器单元测试 104/104，日志解析 84/84，假 worker 的进程/版本捕获自检 54/54。旧 `m{n}_{b1}.save` 自定义格式与 Python 参考生成器的 5 组输出逐字节一致；完整生成器验收 32/32，通过并留下输入/输出夹具于 `data/experiments/gui_worktodo_v039_test/`。首次验收因系统 Python 路径失效中止，换用工作区 Python 后完整通过。

CUDA Stage2 正式源码在已有 CUDA 对象上进行 host-only 重编译并链接成功。用空 worktodo 的 GUI 事件 dry run 输出 `ECM-ELF 0.39.0 (Stage2)`，退出码 0。初次生成器测试暴露了 B1 压缩误用；已修正为保留输入拼写。未进行真实曲线算术或性能测试。
