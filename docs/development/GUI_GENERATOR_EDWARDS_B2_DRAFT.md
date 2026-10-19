# GUI worktodo 生成器：Edwards 推荐与 B2 覆盖开发草稿

## 问题与依据

- GUI 原先在选择 Edwards 时关闭并拒绝推荐曲线数；`ecm_cuda --gpu-info --curve ed` 仍返回 Montgomery 档位。生成器因此无法按 Edwards GPU 的实际批量规模生成任务。
- CPU Stage1 没有 GPU 档位，界面虽然自动关闭推荐选项，但仍允许用户重新勾选，预览阶段才报错。
- 生成器核心已支持 `set_b2` 覆盖，GUI 没有输入控件或默认值。

## 本轮设计

- `--gpu-info` 根据 CLI `--curve` 或 worker INI 的 `curve` 输出实际曲线类型。Edwards 档位来自与运行分派共享的 `src/cuda/ecm_cuda_edwards_tiers.inc`，容器余量为一位，TPB 为 128；当前无法从此查询获得 Edwards 内核占用率，报告为未知。GUI 推荐按用户设置的块/SM 计算。
- GUI 为 Edwards GPU 请求 Edwards 档位，并核对返回档位的曲线类型；CPU Stage1 的推荐控件禁用并保持原有曲线数。
- GUI 的 B2 覆盖值默认为 `0`，留空保留输入任务 B2。非空值必须是非负有限数字。

## 验证

- `ecm_gui` 与 `ecm_gui_gen_test` 编译成功；生成器单元测试 110 通过、0 失败。
- CUDA 可执行文件在本机链接成功；RTX 4070 Ti 上 `--gpu-info --curve ed --bits 512` 返回 Edwards、`carry_bits=1`、768-bit/TPI8 档位。GPU 查询回归 59 通过、0 失败，包括 INI/CLI 曲线优先级。
- 当前工作区缺少构建配置引用的 `cgbn/include`。本机 `D:/code/CUDA/cgbn` 的头文件又与已有 Edwards 对象包含的定义冲突；本轮构建使用 `data/experiments/gui_generator_build_shim_20261019/` 中的临时头文件，并通过 `NVCC_PREPEND_FLAGS` 注入。该构建依赖问题需单独整理，以保证洁净环境可重复构建。

此草稿在功能完成后由用户决定是否清理。
