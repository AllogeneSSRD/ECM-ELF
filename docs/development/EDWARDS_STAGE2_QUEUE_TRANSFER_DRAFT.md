# Edwards Stage1 → Stage2 队列交接开发草稿

## 目标与边界

Stage1 的 Edwards `ECM=`/`ECM2=` 与 `ECMSTAGE2=` 任务完成后，可由配置投递到本仓库 Stage2 队列。默认接收相同 worker，也可指定其他 worker。Stage1 和 Stage2 仍各自运行；Montgomery 的 Prime95 交接保持独立。

## 当前设计

- `config/ecm_options.json` 定义 `edwards_stage2_transfer` 与 `edwards_stage2_worker`；生成器同步 INI、配置头和参考表。
- `src/core/ecm_driver.cpp` 在实际曲线构造确认后选定交接端。ECM2 输入转为 ECMSTAGE2，实际 save 使用绝对路径；复用 save 时跳过交接前已存在的 Edwards 记录。ECMSTAGE2 输入保留原 B2、skip、曲线数及因子字段，只改写 save 路径。
- `src/core/ecm_stage2_transfer.cpp` 将目标路径、worker、任务行写入 Stage1 队列旁的待投递文件。下次启动先重试，配置关闭也不丢弃已有待投递项。
- `src/core/ecm_worktodo.cpp` 使用 `.edit.lock` 序列化 Stage1/Stage2 的队列改写与新投递；投递时对照活动队列及 Stage2 finished。

## 已验证

- `ecm_cuda.exe` 在 Windows sm89 构建通过；`stage2_transfer_test.exe` 覆盖跨 worker 投递、活动/完成或输入错误回执去重、目标队列不可写后的恢复；既有 `worktodo_sections_test.exe` 30 项通过。
- 隔离目录中运行一条 CPU Edwards `ECM2=1,2,127,-1,20,100,1`：Stage1 finished 保留原行，Stage2 worker 2 收到含绝对 save 路径、B2=100、skip=0、count=1 的 ECMSTAGE2 行。现有 Stage2 0.39.3 对该行 `--dry-run` 验证 checksum，识别为 Edwards，可读取一条曲线。
- 另用 `ECMSTAGE2=` 输入、`ed_` 旧前缀和初始 INI `curve=mont` 验证按实际曲线选择交接。Stage1 将 save 规范为 `e_` 并投递到 worker 2；现有 Stage2 0.39.3 完整执行一条曲线，发布结果及 Stage2 finished，移除队列行，退出码 0。证据位于忽略目录 `data/experiments/stage2_transfer_typed_20261019_232203/`。
- 将 worker 2 的 `stage2_worktodo` 覆盖为专属路径后，实际 Stage1 任务只写入该路径，未创建全局队列路径；证据位于 `data/experiments/stage2_transfer_override_20261019_232756/`。
- 当前源码的 `ecm.exe`、`ecm_cuda.exe` 以及两个队列测试目标均编译成功；配置生成物 `--check` 和 `git diff --check` 通过。

## 待验证

- 扩大 Windows/CUDA 实际生产位宽和多任务并发队列测试，尤其是 Stage2 正在完成前一任务时的新投递。
- 将同一交接场景加入持续集成的隔离端到端测试。
- 本机 CUDA 13.3 + MSVC 18 编译新 `ecm_cuda_stage2` 时，`nvcc` 的宿主 `cl.exe` 在 `ecm_cuda_stage2.cu` 返回 `0xC0000005`；共享队列源文件已由其他目标编译，但新的 Stage2 可执行文件尚未生成。完整 Stage2 实跑使用现有 0.39.3 二进制，尚不能据此声称新 Stage2 二进制已验收。编译日志见 `data/experiments/stage2_transfer_stage2_final_build.log`。
- GPU Edwards 的单曲线小任务在该本机测试中进入字典准备后未在 90 秒内完成，已停止；本轮未将它计为通过的交接验收。样本位于 `data/experiments/stage2_transfer_gpu_20261019_232937/`。
