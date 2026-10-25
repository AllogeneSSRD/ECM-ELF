# OpenCL Stage1 与算子路径

OpenCL 实现 Stage1 的曲线批处理，使用主机设备配置与动态拼装内核。它不提供独立 CUDA Stage2 的 NTT 引擎；两条路径的支持范围和基准分别核对。

## 算子与选择

算子族包含 Montgomery 乘法/平方、模加/减和 special_mult。描述符维护 id、别名、内核文件、函数名、位宽/TPI/容器限制、厂商掩码与 auto 优先级。

固定位宽算子需要精确匹配容器；generic 算子按声明范围使用。手动选择和 auto 都经相同的设备/位宽资格检查，不能让别名绕过约束。`mp_` 前缀区分公共多精度助手与专用算子接口。

## 内核组装

主机根据路径计划确定 limbs、TPI、工作组和算子。组装顺序为配置宏、公共头、算子文件、统一接口和主 kernel；相同算子源只加载一次。内核缓存身份包含选择、源和版本信息，避免配置变更后复用错误程序。

普通、local 和 cooperative 路线有不同私有/LDS/协作状态。某个微基准快不代表完整 ladder 快；厂商汇编和 limb24 路线只有在对应声明与验证范围内使用。

## 算术与资源

通用 CIOS 乘法主项 O(limbs²)，每一步包括乘累加、Montgomery 消去及条件校正。寄存器/private spill、LDS 容量、同步和实际 wave 占用共同影响时间。算子正确性应检查边界/进位/借位、规范范围与 GMP 对照，再进行完整 Stage1 验证。

OpenCL 编译器对整数乘法和内联汇编的生成方式可能随驱动/设备变化，ISA 导出工具用于核对实际指令。环境中的时钟/功率条件不是跨设备性能保证。

## 代码入口

- [opencl_ecm_path_registry.h](../../src/ecm/backend/opencl/arithmetic/path_registry.h)、[实现](../../src/ecm/backend/opencl/arithmetic/path_registry.cpp)：路径元信息与选择。
- [opencl_ecm_runtime_config.cpp](../../src/ecm/application/configuration/runtime.cpp)：设备和运行配置。
- [opencl_ecm_stage1.cpp](../../src/ecm/backend/opencl/stage1/montgomery/engine.cpp)、[impl_opencl.cpp](../../src/ecm/backend/opencl/runtime/runtime.cpp)：主机/内核接线。
- [生产资源](../../src/ecm/backend/opencl/resources/)：公共原语、算子与 Stage1 kernel；纯工具资源仍在 kernels/opencl/bench。
- [opencl_ecm_selftest.cpp](../../src/ecm/backend/opencl/diagnostics/selftest.cpp)、[opencl_mont_isa_export.cpp](../../tools/disasm/opencl_mont_isa_export.cpp)：验证和 ISA 入口。

## 资源与 local 构建

桌面入口为 [build_opencl_local.ps1](../../tools/build/build_opencl_local.ps1)，默认输出到 build_opencl_local/stage1/&lt;profile-key&gt;。公开 include 只承载共享主机接口；OpenCL 的 context/program/cache、设备枚举和曲线实现均为 src/ecm/backend/opencl 内的私有实现。

local 构建以普通 CXX 编译，关闭 CUDA、GUI、工具及独立 CPU Stage2 目标；profile 核对工具链、依赖与用途，拒绝不兼容或未标记的已有目录。CMake 将生产资源同步到叶目录 opencl_resources，修改资源不依赖 CXX 对象重编；build_manifest.json 记录程序及资源摘要。该入口不生成 release 包。

资源加载保留显式 --kernel-root、当前目录相对查找及编译默认根的既有优先关系。默认根指向构建叶目录随附资源；不含默认宏的实现使用 src/ecm/backend/opencl/resources 回退。生产自检源在 diagnostics/mont.cl；旧逻辑名称 bench/mont.cl 先按旧方式查找，缺失时再加载 diagnostics/mont.cl，因此已有自定义根仍可使用原布局。实际算术、源拼装次序和设备/驱动/完整源/编译选项的缓存身份保持。

完整发布入口使用独立release用途构建并分发 `ecm.exe`、生产资源及OpenSSL运行库，GMP固定使用与GUI/CUDA worker相同的vcpkg通用版本。该构建启用 `ECM_OPENCL_PORTABLE_RESOURCES`，默认资源根由EXE所在目录派生，显式 `--kernel-root` 仍优先；local默认仍保留上述构建目录资源根。分发与身份验证以[构建与发布](../usage/BUILD.md#完整发布与按需加载)为准。OpenCL内核由设备驱动统一JIT，两套CUDA包不改变其编译策略。

Android CMake 消费同一主机源码，Gradle 从新生产根和原纯 bench 根组合资源，保留 assets/kernels/opencl 逻辑根。本轮只核对其路径合同，未进行 Android 实体构建或设备验收。
