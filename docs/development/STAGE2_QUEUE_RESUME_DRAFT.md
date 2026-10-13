# Stage2 队列续跑与生产配对验收草稿

临时开发记录，保留至用户决定处理。正式规则见[Stage2使用](../usage/STAGE2.md#自动续跑与退出)，性能证据见[M12011配对](../performance/STAGE2.md#m12011-真实-save-配对与默认无预热)。

## 问题与定位

- 原生产进度为schema1、done3/total144，第340条尝试未完成。旧身份绑定整份队列哈希、路径及性能设置；只交换未来两项即可复现致命退出码2，尚未启动曲线。原始输入及复现位于`data/experiments/stage2_resume_tune_20261013/production_snapshot/`与`reorder_before/`。
- 多个候选调用公开规划API，每次触发同一阶段输出并更新阶段时钟。公开规划作用域现在静默正常日志及阶段时钟，保留错误、规划JSON和独立调试记录。
- 旧调优证据为通用12288-bit完整梅森输入，生产实际N为11694-bit余因子；短校准的D210/B242020也不构成当前B2配对基线。当前真实worktodo调优已补齐默认D240240的普通/承载各3次数据。

## 实现与验证

- 每任务SHA256身份与`.ecm_ckpt_stage2_<bits>_<id>.dat`，schema2仅保留计数和回执元数据；独占任务锁、原子刷新和成功回执对账保留。
- 匹配旧状态后先发布新文件，再归档原字节。非当前任务的旧进度保留。改变队列顺序、输出路径或性能配置不重置计数。
- GUI的Stage1安全退出观察器排除`.ecm_ckpt_stage2_`，避免将曲线尝试计数误判为Stage1算术存档；本机GUI目标已编译通过，未替换生产GUI或进行GUI运行验收。
- CPU原生夹具18项通过：任务键、SHA256、旧导入、schema2、小文件、结果尾部撕裂保护、回执恢复及嵌套日志静默。原生旧文件匹配由调用方解析AID/N；夹具第一次错误地省略了AID匹配，修正夹具后通过，未放宽生产匹配。
- 先用隔离的生产队列/save执行重排续跑，第340条完成后单次Ctrl+C正常停止，done4/144，新文件116字节。再将一个真实但已耗尽save范围的其他任务移到队首：零曲线推进该任务，原部分进度与不相关旧状态保持不变。
- 生产目录发布前核对原EXE、INI、queue、progress、profile与冻结SHA一致，备份至`publication_backup/`；GMP DLL身份相同。发布新EXE、合格配对profile及生成INI参考，未改INI、queue、save或GPU设置。
- 生产验收仅执行第340条，承载12011/D240240，自检零错误，退出0；done4/144，下次从第341条继续。规划阶段恰好一行；队列和INI的SHA不变。旧进度归档，当前程序未持续运行。

两次收集器问题均保留失败证据后修正，不重新生成计时来改善结论：tune收集器起初将两份plan JSON当作曲线回执；续跑收集器起初假设生产回执有嵌套`result`，实际字段为顶层。按真实格式重新核验已完成结果。生产验收使用修正后的收集器一次通过。

## 构建与证据

完整sm89/CUDA13.3生产编译127秒；主机帮助同步后复用经校验CUDA对象重编33秒。最终EXE SHA256=`64103a3051fcf127744576de063e07849dd3a0479978d7d809a3124cb5b340bb`。首次主机编译失败的摘要另保存在`build_host_failure.log`，失败原因是SHA流函数提取后残留文件路径变量，已修复。

源码入口：`src/core/ecm_stage2_queue_state.h::Task/State/import_legacy`、`src/core/ecm_cuda_stage2_main.cpp::run`、`src/core/ecm_stage2_logging.h::PlanningScope`、`src/cuda/ecm_cuda_stage2.cu::ecm_cuda_stage2_plan`。工具入口为`tools/test/stage2_resume_fixture.cpp`、`test_stage2_resume_stop.py`、`test_stage2_task_tune.py`。全部原始证据位于忽略目录`data/experiments/stage2_resume_tune_20261013/`。

TODO：扩大显式预热1、输入错误/worker/预算截断及生产内存边界验证；任务运行过程中修改当前行仍属于一致性保护范围。本文完成后提示用户处理草稿，不自动删除。
