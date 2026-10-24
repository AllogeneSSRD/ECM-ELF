# 共享 PARI 诊断接入草稿

日期：2026-10-23。授权：将 GUI 的 PARI DLL 能力接入 Stage1/Stage2 共享诊断层，默认 DLL，备选 GP，GMP 保留但不自动使用。版本统一为 0.52.0；开发与验收已完成，用户于 2026-10-24 授权按 [0.52.0] 提交 Git。

## 实现

- PARI C 接口迁至 `service/diagnostics/pari_native.h/.cpp`，独立 `ecm_diagnostics_host.exe` 承载每个数学请求。worker/UI 不加载 DLL，Job 管理取消、512 MiB 进程内存上限与 5 秒截止。
- `Session` 默认 DLL → GP，`pari`/`native` 只 DLL，`gp` 只外部 GP，`gmp` 为显式原算法。GP 默认 128 MiB 栈，原始因子发布先于诊断。
- 新增协议 2 的逐行检查点，接收方重验乘积、Hasse 界、点阶消去和最小性；超时保留最后完整记录。GUI Auto/本地复用 Session，显式 GP 保留无助手路线；合数先处理便宜小分量。
- 共用 CMake 目标、诊断依赖打包与源码/运行时身份清单。Stage2 独立构建复用同一目标；Stage1/Stage2 ZIP 增加受限相对目录的依赖资源。配置说明只修改权威 schema 后重新生成。

## 验证与证据

根目录：`data/experiments/shared_pari_0520_01/`。初次新 CMake 编译器探测使用 Debug/PDB 时遇到本机 C1902；改为 Release 配置的真实编译器探测后通过，没有跳过探测。原始日志仍保留。

- `shared-01/result.json`：44 项通过；含用户 82 bit 样本及 127/160/192 bit × 五种曲线的 16 组 DLL/GP 对照，群阶、点阶、分解相同。默认 DLL 路线未启动故障 GP；验证缺失助手/DLL、显式 GMP、原 `native` 别名、无效输入、超时部分因子、错误乘积和中文路径移动。
- 当前对照为功能验收，不发布 DLL/GP 加速比例，也不承诺所有 192 bit 输入能在 5 秒完成。
- `stage1-contract-01/summary.json` / `stage1-gpu-01/summary.json`：实际完整位宽 sm89 Stage1 的 CPU、GPU 各 24 项通过。包含原始因子、提交文件、诊断失败隔离、显式 GP/GMP；CPU 对照因子存档，GPU 使用真实计算命中。
- `stage2-contract-02/summary.json`：实际 Stage2 队列 10 项通过，覆盖 DLL 优先、无助手时 GP 故障、超时、协议/乘积错误、大整数证素、困难余因子及日志不可写；回执和原始命中保留。输入为合成 saved-X 快捷路径，不额外声称验证 GPU Stage2 算术。
- `gui-contract-03.log`、`gui-backend-01/result.json`：合同、合数分量、取消、uint64 sigma、GP-only 和显式 GMP 通过；15 组素数与 15 组合数的两后端对照及困难余因子保留通过。
- `gui-192-01/result.json`：共享进程路线的 16 组最高 192 bit 输入完成且证素，与 GP 对照一致。此处不引用计时作为性能结论。
- `gui-ui-02/result.json`：中英文、15/24 字号、900px 窄窗口、选择复制、导入和 Toast 通过。初次 GUI 与队列测试被沙箱拒绝 weakly_canonical，正常权限复测通过；保留原失败证据，未放宽产品检查。
- `worker-package-test-final/result.json`：ZIP 的嵌套 SEA/许可资源、用户配置/队列排除、路径越界拒绝和助手篡改拒绝通过。
- `gui-relocated-final/result.json`、`worker-relocated-final/summary.json`、`worker-stage1-relocated-final/result.json`：最终包移动部署通过；GUI 在中文路径、仅 System32 的 PATH 下完成 192 bit 分析并重启，Stage1/Stage2 从同目录加载 DLL，未依赖 GP。
- `source-rebuild-01/result.json`：分发的诊断源码包脱离 GUI/worker 工程重建助手与 probe，复制单独分发的 DLL/SEA 后完成用户样本，PATH 中无 GP。
- `stage1-build-final.log` / `stage2-build-final.log` / `gui-build-final.log`：正式 worker HostOnly 验证 CUDA 对象后构建，共享助手源与依赖在编译前冻结、打包后复核。配置、侧车和 XML 生成检查及 diff 检查通过。
- 汇总：`verification-final.json`。版本均为 0.52.0；本机 Windows x64 / sm89 候选，不代表多架构发布或独立干净 OS 资格。

本草稿完成后保留，提示用户决定整理；当前工作方式以 `docs/architecture/GROUP_ORDER.md` 为权威说明。

## 最终二进制与候选

- stage1 SHA256：`c0d87a09af866fd5a009c12ddf9c76fcef1e03ff9c4e2fdf8dd2a2ce4b83e023`。
- stage2 SHA256：`55a81960123b226a8ec0f81e29395eb8f598114806992349196d995727512c78`。
- gui SHA256：`2fc9a6a538d626bd2d12b7f08a55d9db741dfec0aa364dcc9c8460a4518b380a`。
- gui_helper SHA256：`874cf9793fa0a204ecbc45dc87a5ca4adbe3e16e12c0422cf2612556c9c4e3e1`。
- gui_pari_host SHA256：`e0928be8f7dd952cabb9bf12aaf3a7d4bd7111df7ec09d5ef434a0c901cffe8e`。
- worker_pari_host SHA256：`26707bada8cb8a2b2244311714002741ffe3be7a2a7a4a38c79771cab204debc`。
- workers 候选：`data/experiments/shared_pari_0520_01/worker-package-final/ECM-ELF-Workers-0.52.0-sm89.zip`，ZIP SHA256：`735b782877c78481ddbea24a4764df2b0c171f84e33e748ade1e5c9be291d38e`。
- gui 候选：`data/experiments/shared_pari_0520_01/gui-package-final/ECM-ELF-GUI-v2-0.52.0.zip`，ZIP SHA256：`902acd7d0afcc2ae8f4fca4b1d3edda1964e24f2d750e84f1a990efd1cbfc2ea`。

TODO：其他物理机器/CPU、独立干净 OS、多架构 release。5 秒是本地进程预算，不是任意 192 bit 群阶分解的完成保证；Auto 转 GP 可超过 5 秒。草稿继续保留，等待用户决定整理。
