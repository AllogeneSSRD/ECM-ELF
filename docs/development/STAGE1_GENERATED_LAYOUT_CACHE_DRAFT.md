# Stage1 generated 布局与旧缓存身份冲突修复

状态：已完成修复、回归及真实入口配置验证；草稿保留待用户处理。正式使用规则见[构建与发布](../usage/BUILD.md#默认策略)。不修改主程序版本。

2026-11-01，Stage1 删除 grouped 路径后，正式构建脚本同时删除了显式 `ECM_CUDA_ISOLATE_TIERS=ON`。无参数 local 的 profile 因而重新与最早 grouped 构建的 `ffdf5b532231282b` 相同。该目录的 CMakeCache 仍为 OFF；脚本显示 isolated，但 CMake 正确拒绝旧布局。载入原 MSVC 环境后，仅运行 CMake 配置即可复现原报错。

local/dev 共用入口和 release 阶段入口均恢复显式 ON，使 CMake 有确定值，且布局进入原有配置身份。既有目录选择器为默认 local 选择相容的独立目录；显式旧目录仍拒绝 profile 不匹配。不改写旧 marker、对象或旧 CMakeCache，不恢复已移除的 grouped 实现，也不放宽 CMake 检查。

验证证据保存在 `data/experiments/stage1_layout_cache_20261101/`：

- `tools/test/test_stage1_layout_cache.py`：两项测试先失败后通过。运行真实 local 入口和真实 profile 助手，以工具链夹具停止于 CMake 边界；验证默认入口显式传 ON、选择独立目录、保留旧缓存，以及显式旧目录在配置前被拒绝。
- `repro_vc.log` 保存原配置失败；`configure-fixed.log` 保存真实无参数入口成功配置及进入并行编译器之前的受控停止。实际选择 `build_stage1_local_c869d1f184d3b595`，profile 和 CMakeCache 均为 ON；旧目录的 profile 与 CMakeCache 内容摘要不变。
- 没有启动项目 CUDA 单元重编译或 GPU 数学/性能验收，本次变更仅修复构建配置与目录身份。
