# GUI 群阶分析合并开发草稿

生产目录具备 `ecm_group_order_ui.exe`、`gmp-10.dll`、`libpari.dll` 和 SEA 数据，但缺少 `ecm_diagnostics_host.exe`。对 p=1009、sigma=26 的显式 PARI 请求返回 `pari_launch_failed; pari_protocol_failed`。请求和原始输出保存在 `data/experiments/gui-pari-integration/`。

主 GUI 自身提供 `--group-order <请求> <输出>` 和 `--pari-host <请求>` 两个子进程入口，复用现有请求协议、群阶协调逻辑及共享 PARI 适配层。分派发生在创建窗口和读取工作流之前。GMP 延迟加载，外部 GP 入口继续支持仅 GUI 加语言文件的部署。计算程序的独立共享诊断入口继续存在。

本轮验证使用隔离目录，仅复制主 GUI、GMP、PARI DLL 与 SEA 数据，目录中没有两个旧助手 EXE。测试目标包含小素数、用户提供的大素数、数学结果与 GP 对照，以及超时和取消流程。最终使用和依赖说明见 [GUI v2](../usage/GUI_V2.md) 与 [群阶模块](../architecture/GROUP_ORDER.md)。

已验证的本机候选为 `build_dev/gui_v2_embedded_pari/ecm_gui_v2.exe`，版本 0.69.19。隔离目录测试覆盖 p=1009、用户样本 3436193553215096358758503 和 192 位素数；三组均完成群阶和点阶验证。无数学 DLL 的目录仍可执行主程序版本查询。常规 DX11 窗口运行未加载 GMP/PARI，缩放、最小化和恢复测试通过。

GUI-only 候选包位于 `data/experiments/gui-pari-integration/package-final/`，不包含 `ecm_group_order_ui.exe`。迁移到中文路径、限制 PATH 为 System32 后，包内全部 98 项文件摘要、群阶分析及 192 位 PARI 计算、配置重启、皮肤和缺失依赖路径均通过检查；报告位于 `package-smoke/result.json`。原生产目录未替换。

GP 数学对照通过 15 组素域和 15 组合数输入；GroupOrderJob 回归覆盖 GP-only、DLL 优先、显式 GMP、部分结果、超时和取消。共享诊断层 44 项检查通过，保留 Stage1/Stage2 的默认独立诊断入口。发布布局单元测试 9 项通过，GUI 应用模型回归通过。原始结果分别位于实验目录的 `embedded-final/`、`gp-comparison/`、`shared-diagnostics/`、`gui-app/`、`gui-host/`；目录均由 Git 忽略。测试不代表任意 192 位输入都可在五秒内完成。
