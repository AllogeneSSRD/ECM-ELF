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

- [opencl_ecm_path_registry.h](../../src/opencl_ecm_path_registry.h)、[实现](../../src/opencl_ecm_path_registry.cpp)：路径元信息与选择。
- [opencl_ecm_runtime_config.cpp](../../src/opencl_ecm_runtime_config.cpp)：设备和运行配置。
- [opencl_ecm_stage1.cpp](../../src/opencl_ecm_stage1.cpp)、[impl_opencl.cpp](../../kernels/opencl/impl_opencl.cpp)：主机/内核接线。
- [kernels/opencl](../../kernels/opencl/)：公共接口、算子与 Stage1 kernel。
- [opencl_ecm_selftest.cpp](../../src/opencl_ecm_selftest.cpp)、[opencl_mont_isa_export.cpp](../../src/opencl_mont_isa_export.cpp)：验证和 ISA 入口。
