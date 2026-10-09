# Windows 图形前端

`ecm_gui.exe` 使用 Dear ImGui docking、Win32/DX11，管理 Stage1 worker 进程、配置、队列生成、GPU 监控及结果展示。GUI 不执行数论算法，也不链接 GMP/OpenCL/CUDA 引擎；计算能力来自所选 worker exe。

## Worker 与配置

一个 worker 对应一个 `ecm_cuda.exe -ini <file> --worker N` 进程，并有独立输出窗口和进程监管。INI/worktodo 的 `[Worker #N]` 使用同一编号，exe 必须支持该接口。不同 GPU 的队列和批量独立配置。

GUI 专用设置位于 `[GUI]`，worker 设置位于相应分区；公共键放在任何分区前。读写保留注释、未知键和顺序，保存使用原子替换。全部默认/范围见 [INI 参考](../ECM_INI_REFERENCE.md)。

## 队列生成与交接

生成器解析 PrimeNet ECM/ECM2 任务，过滤、去重、排序、检查 save 名称，并按所选设备档位建议曲线批量。写入前展示预览，追加前重新核对目标队列大小/修改时间，避免覆盖外部编辑。

批量建议基于 blocks/SM、SM 数及每 block 实例数，不等于填满所有理论寄存器槽位。生成器不使用 GMP，剥离已知因子后的位宽为估计下界，精确整数仍由驱动校验。

Prime95 交接状态显示 pending、失败或成功；真实 save/worktodo.add 写入由驱动负责。GUI 不是 Prime95 活动队列的第二个消费者。

## 日志、结果与监控

管道进度用于百分比、速度和 ETA，事件行用于开始、因子、checkpoint 和错误。结果 JSONL 为原始结果，文本汇总合并同因子的 sigma；两者不可互当 checkpoint。

GPU 面板动态加载 NVML，显示利用率、显存、SM/graphics 时钟、功耗、温度和限频原因。驱动不支持的指标显示不可用；NVML 利用率不是 SM occupancy。监控不改变 GPU 功率/频率配置。

NVML 采样时不持有 UI 读取设备/历史的锁；初始化仍同步，没有进程隔离或超时保证。`--trace` 可记录初始化与窗口恢复。

## 窗口与退出

启动检查普通窗口矩形及标题栏是否位于可用显示器；异常尺寸、最小化哨兵或屏幕外位置恢复到工作区，并重建停靠布局。退出保存正常窗口位置，最小化时保留有效布局，避免下次只有任务栏图标。

停止/关闭按 worker 的 checkpoint/进程策略执行；图形窗口关闭不代表未完成曲线已有最终 save。状态与兼容要求由当前 worker 协议确定。

## 代码入口

- [main_win32.cpp](../../src/gui/main_win32.cpp#L87)：窗口矩形验证、恢复和保存。
- [app.cpp](../../src/gui/app.cpp)、[worker_proc.cpp](../../src/gui/worker_proc.cpp)：界面和 worker 生命周期。
- [worktodo_gen.cpp](../../src/gui/worktodo_gen.cpp)、[results.cpp](../../src/gui/results.cpp)、[log_parse.cpp](../../src/gui/log_parse.cpp)：生成器、结果和进度。
- [gpu_monitor.cpp](../../src/gui/gpu_monitor.cpp)、[ini_file.cpp](../../src/gui/ini_file.cpp)：监控和配置。
- 构建入口：`tools/build/dev/build_gui.ps1`；[构建说明](BUILD.md)。
