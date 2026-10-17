# Stage1 检查点与自动 save 命名草稿

本草稿记录正在实施的修复与验收，未通过的项目不得视为生产保证。

## 已确认问题

2026-10-17，生产目录三份原文件的只读快照位于 `data/experiments/stage1_ckpt_audit_20261017/`。原文件未修改；完整校验结果见 `snapshots.json`，独立解析见 `analysis.json`。

- Mont 旧档 4,300,880 bytes：V5，960 曲线，5120-bit/TPI16，PARAM0。载荷恰为 7×960×5120/8=4,300,800 bytes，另加72-byte头和8-byte校验。进度387326/375102575，原校验通过。
- Mont 新档20,513 bytes：文本批次V1，960条sigma、零条非空算术记录；外层校验通过。它只能保留种子，不包含可继续的运算进度。生产INI定期存档间隔600秒，不能仅凭该体积推断算术状态被截断。
- Edwards 新档103,265,174 bytes：一条103,244,791-byte GPU载荷。完整指数的十六进制文本占93,775,646 bytes，即文件的90.8105%；状态和内层尾部约9.47 MB。该指数写入方式在引入批次外壳之前已经存在。
- `b55f47d` 将旧Mont文件访问切换成批次内部记录，改变了外壳及查找入口；Mont的V5头和设备数组本身仍被保留。当前入口缺少旧`.dat`导入，所以只改名不能恢复。
- GPU队列自动save使用内置默认pattern，未读取已加载的`save_name_pattern`；Edwards另行拼接`ed_`。旧formatter只看到余因子N，无法保留worktodo的原始指数。

## 用户选定方向

统一新二进制容器，兼容读取现存档。Mont保留V5算术载荷；Edwards精简为二进制载荷，不保存完整指数。校验失败恢复有效备份；本任务不引入纠错码。

## 实施合同

- 新批次外壳使用`ECMS1B2`标记、版本2、小端整数、完整任务身份及逐记录长度。每条记录和全文件各有CRC32C；边界和长度在分配前检查。
- 新Edwards GPU载荷使用版本3，保存N、B1、指数位数及指纹、NAF窗口/倍点算法、当前列、每条sigma/结果及9个状态量。恢复后重建指数核对指纹，并保留曲线重建、状态恒等式和范围检查。
- 文本批次V1与旧Edwards文本GPU载荷仍可读取；新写入使用二进制。不同后端载荷保留自身版本和布局校验。
- 旧CUDA Mont V5 `.dat`及轮换备份只读导入：检查原校验、曲线数、参数化、布局、指数位数、进度和每条lane的完整N。仅在新批次没有算术状态时导入，已有算术状态优先。旧源文件不删除、不覆盖。V4及更早不作为安全V5状态导入。
- 随机种子任务采用有效旧档中的sigma；显式sigma与任务不匹配时不能偷换曲线。
- 自动save默认pattern为`{curve}_{target}_{b1}.save`。curve=e/m0/m2/m3；target使用原始梅森指数，例如m5351，其他整数使用bit4916。旧自定义pattern自动补充曲线前缀；CLI显式save路径保留，CPU与GPU均使用指定目标。
- Stage1的ECMSTAGE2队列从save末尾提取B1；类型前缀覆盖每条任务的INI曲线/param，Ed固定48LCM，无前缀沿用INI。混合队列不传播前一任务的覆盖。队列可规范旧save前缀并复制相同曲线历史、保留旧文件及skip编号；START保留输入，finished和受支持Prime95交接仅替换save字段。CPU拒绝PARAM2/3，交接资格按实际任务判断。
- 检查点名称保持`.e_s1_m5351_260000000.ckpt`、`.m0_s1_m5351_260000000.ckpt`等；save使用`e_m5351_260e6.save`、`m0_m5351_260e6.save`等。配置源唯一为`config/ecm_options.json`，引用文档由生成器同步。
- 原子替换、备份轮换、任务锁和最终save/结果成功发布后才删除检查点的顺序继续保留。校验用于检测字节损坏，不能证明数学运算正确；备份恢复不等于纠错码修复。

## 验收结果

离线容器探针已验证CRC32C标准向量、逐记录/整文件校验、长度边界、截断、身份拒绝、有效备份恢复与文本V1读取。生产Edwards快照全部960条坐标和sigma在旧→新→恢复后相等，GPU载荷由103,244,791缩为4,774,277 bytes，新外壳合计4,798,593 bytes；这是存档体积，不是运行内存或性能测量。Mont生产快照的4,300,880-byte V5载荷原样导入，并逐lane核对完整N。

测试发现并修复秒级同步边界：增量同步需包含与任务启动标记同一秒写出的save。`compat_full/`保留失败证据；`compat_full_sync/`随后确认同步修复，但测试将PARAM0必须写出PARAM字段作为错误断言。现按现有GMP-ECM约定将省略PARAM视为0，最终结果记录在独立目录，不改写失败样本。

完整构建包含21个sm89算术编译单元、未嵌入PTX、未编入实验resident/PRAC；验证设备为GPU1 RTX4060 Laptop，CUDA13.3/MSVC14.51.36231。本轮未修改GPU功耗或频率，不采集正式性能数据。程序位于`build_cuda_cmake/stage1_checkpoint_v2_full_20261017/stage1/ecm_cuda.exe`，SHA256为`176777374166d28ef50c6346610f99cfde38304ff51028ec9e2d29563568c77a`。仅验证本机sm89构建，未执行多架构发布验证。

最终二进制完整运行、故障注入和停止恢复24项均通过，见`usability_final/summary.json`；旧格式、命名与队列33项均通过，见`compat_full_final/summary.json`。后者覆盖冲突INI、混合前缀与无前缀回归、CPU拒绝PARAM2/3、历史复制、含逗号绝对路径、AID和参数保留、类型冲突拒绝及交接。Prime95路由、pending、锁、AID及恢复66项通过，见`p95_final.log`。GUI文件名/结果存储50项和检查点扫描5类主档、3类排除通过，见`gui_results_final.log`、`gui_final_scan.log`。

生产规模短续算使用原始快照副本，M5351余因子4916 bits、B1=260,000,000、960曲线，并故意将INI曲线设为相反类型来验证队列前缀覆盖。Mont进度387326→387526 bits，新文件4,325,206 bytes；Edwards进度648448→648704列，新文件4,798,588 bytes。两例均正常响应停止、保留worktodo且不产生完成回执，源快照hash不变。Mont预检留下0-byte save占位文件，不能当作已完成曲线。见`real_gpu_resume/summary.json`及两份日志；记录的wall包含准备、导入与验证，不能当作完整Stage1耗时。

- [x] 新容器边界、CRC损坏、截断、任务身份拒绝及备份恢复。
- [x] CPU/GPU Edwards与Mont各PARAM的停止/恢复和完整运行结果一致。
- [x] 文本批次V1与GPU Edwards旧载荷兼容。
- [x] CUDA Mont V5旧档导入、sigma保持、完整N拒绝及原档不变。
- [x] 生产原始快照离线导入及Edwards压缩体积测量，不在生产目录试写。
- [x] CPU/GPU自动save、旧pattern、自定义pattern、余因子原指数与显式路径。
- [x] 生成物核对、正式使用文档同步、独立测试构建身份归档。

证据入口为`data/experiments/stage1_ckpt_audit_20261017/final_validation.json`，程序与源码快照在同目录`frozen_final/`。未替换外部生产程序、未提交Git。

完成后提示用户处理本草稿，不自动删除。
