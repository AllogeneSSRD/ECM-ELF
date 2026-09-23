# 算子路径注册表 —— 去耦合可扩展架构（开发者指南）

Stage-1 的 **Montgomery mul/sqr** 与 **add/sub-mod** 算子，其「选择 → 注入 → 拼装」全流程
集中在单一注册表中。主内核与算子实现 **完全解耦**：主内核只调用宏别名，Host 端通过宏注入
把选定算子绑定到统一接口。桌面与 Android 共用同一份 `src/opencl_ecm_path_registry.cpp`。

> 本文档同时是「如何新增 / 删除一个算子」的操作手册，见 [§5](#5-如何新增一个算子)。

---

## 1. 架构总览

```
Host (C++)                                  Kernel (OpenCL)
──────────────────────────────────────────  ────────────────────────────
opencl_ecm_path_registry.cpp                注入的汇编头：
  ├─ ECM_MONT_OPERATORS  (单一数据源)         #define ECM_STAGE1_MUL_IMPL mont_mul_unroll_384b
  ├─ ECM_ADDSUB_OPERATORS(单一数据源)         #define ECM_STAGE1_SQR_IMPL mont_sqr_unroll_384b
  │                                           #define ECM_STAGE1_ADD_IMPL add_mod_unroll_384b
  ├─ resolve_mont_side()  通用解析            #define ECM_STAGE1_SUB_IMPL sub_mod_unroll_384b
  ├─ resolve_addsub_side()通用解析                       ↓
  └─ assemble_kernel_source()  拼装           common/operator_iface.h.cl:
                                                #define mont_mul ECM_STAGE1_MUL_IMPL
                                                #define add_mod  ECM_STAGE1_ADD_IMPL ...
                                                         ↓
                                              ecm_stage1.cl:  只调用 mont_mul / mont_sqr /
                                              add_mod / sub_mod，无任何 #ifdef 算子分支。
```

**解耦的三层边界**

| 层 | 职责 | 改一个算子是否需要动它 |
|----|------|----------------------|
| 主内核 `ecm_stage1.cl` | ladder 逻辑，只用宏别名 | 否 |
| 接口 `operator_iface.h.cl` | 宏别名 → 注入符号 | 否 |
| 注册表 `opencl_ecm_path_registry.cpp` | 描述符表 + 解析 + 拼装 | 是（仅一行） |
| 算子 `.cl` 文件 | 具体实现 | 是（新增/删除一个文件） |

---

## 2. 源文件布局（本次重构后）

| 文件 | 角色 |
|------|------|
| `include/opencl_ecm_path_registry.h` | **唯一公共头**。描述符、枚举、全部对外函数声明。 |
| `src/opencl_ecm_path_registry.cpp` | **唯一实现**。注册表数据源 + 通用解析器 + 拼装 + 兼容封装。 |

> 历史上的 `opencl_ecm_mont_path.{h,cpp}` 与 `opencl_ecm_addsub_path.{h,cpp}` 已**合并删除**，
> 其全部公共接口迁入上面两个文件，调用方无需改动函数名。

内核目录：

| 目录 | 角色 |
|------|------|
| `kernels/opencl/common/` | 配置、limb 原语、ladder 辅助、算子接口、asm 公共块 |
| `kernels/opencl/mont_mul/` | Montgomery mul + sqr 实现（mul/sqr 同文件，sqr 内部调 mul） |
| `kernels/opencl/add_mod/` | 模加算子 |
| `kernels/opencl/sub_mod/` | 模减算子 |
| `kernels/opencl/special_mult/` | 单 limb Montgomery 乘算子（固定位宽 + generic 通用版） |
| `kernels/opencl/ecm_stage1.cl` | 主 ladder 入口 |
| `kernels/opencl/ecm_stage1_coop.cl` | 4096 位协作工作组补充（仅 `COOP_WG>1` 时加载） |

---

## 3. 单一数据源（X-macro）

去冗余的核心：**mul 与 sqr 共用同一 `.cl` 文件**（仅函数名前缀不同），**add 与 sub 是约束完全
一致的镜像族**。因此不再维护四张几乎重复的表，而是两张「单一数据源」宏，按算子族各展开两次：

```cpp
// Montgomery：每行一个算子，展开出 kMontMulRegistry / kMontSqrRegistry
#define ECM_MONT_OPERATORS(X)                                                  \
    X(unroll_384b, unroll_384b, unroll384, 10, kMontNoMinLimbs,            \
      kMontUnroll384MaxLimbs, kMontUnroll384MaxLimbs, ECM_OS_ANY, ECM_GPU_ANY, \
      OS_ANDROID, true, 1, 0)                                              \
    ... /* 其余算子各一行 (idt, stem, al, prio, min_l, max_l,                 \
           container_l, os_mask, gpu_mask, gpu_excl, fixed_width, coop, scratch) */

#define ECM_MONT_MUL_ROW(idt, stem, al, ...) \
    {#idt, "mont_mul_" #stem, kMontAliases_mul_##al, "mont_mul/mont_mul_" #stem ".cl", __VA_ARGS__},
#define ECM_MONT_SQR_ROW(idt, stem, al, ...) \
    {#idt, "mont_sqr_" #stem, kMontAliases_sqr_##al, "mont_mul/mont_mul_" #stem ".cl", __VA_ARGS__},

constexpr EcmMontPathDescriptor kMontMulRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_MUL_ROW)};
constexpr EcmMontPathDescriptor kMontSqrRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_SQR_ROW)};
```

### 3.1 别名数组的生成与引用（连锁宏展开详解）

别名字符串数组不是硬编码的，而是通过**三阶段连锁宏展开**把两个宏的参数连接起来：

#### 第一阶段：ECM_MONT_ALIAS_TABLE 定义数组变量

```cpp
#define ECM_MONT_ALIAS_TABLE(side, S)                                          \
    static const char *const kMontAliases_##side##_unroll768[] = {             \
        "unroll_768b", "unroll_only_768b", "mont_" S "_priv_unroll_only_768b", nullptr};
```

调用 `ECM_MONT_ALIAS_TABLE(mul, "mul")` 展开得到：

```cpp
static const char *const kMontAliases_mul_unroll768[] = {
    "unroll_768b", "unroll_only_768b", "mont_mul_priv_unroll_only_768b", nullptr};
```

- `##side##` 处 token-paste `mul` → 变量名含 `_mul_`
- `S` 处字符串字面量 `"mul"` → `"mont_mul_priv_unroll_only_768b"`（side-prefixed 兼容键）
- sqr 同理调用 `ECM_MONT_ALIAS_TABLE(sqr, "sqr")`，生成 `kMontAliases_sqr_unroll768[]`

#### 第二阶段：ECM_MONT_OPERATORS 提供别名令牌

```cpp
#define ECM_MONT_OPERATORS(X)
    X(unroll_768b, unroll_768b, unroll768, ...)
//                ^^^^^^^^^^   ^^^^^^^^
//                idt + stem   al（一个裸 token，不是字符串）
```

第三个参数 `al` = `unroll768` 是**裸 C 预处理器 token**（无引号、无 `#`），它作为变量名的"后缀片
段"传递。同理 manual 变体传 `unroll768manual`。

#### 第三阶段：ECM_MONT_MUL_ROW token-paste 引用

```cpp
#define ECM_MONT_MUL_ROW(idt, stem, al, ...)                                   \
    {#idt, "mont_mul_" #stem, kMontAliases_mul_##al, ...}
//                           ^^^^^^^^^^^^^^^^^^^^^^^^
//                           把 al 粘贴到变量名尾部
```

当 `X = ECM_MONT_MUL_ROW` 时，`X(unroll_768b, unroll_768b, unroll768, ...)` 展开为：

| 参数 | 形参 | 展开后值 | 方式 |
|------|------|----------|------|
| `unroll_768b` | `idt` | `"unroll_768b"` | 字符串化 `#idt` → `id` 字段 |
| `unroll_768b` | `stem` | `"mont_mul_unroll_768b"` | `"mont_mul_" #stem` → `cl_name` |
| `unroll768` | `al` | `kMontAliases_mul_unroll768` | `##al` token-paste → `aliases` 字段 |
| `__VA_ARGS__` | ... | `10, 0, ...` | 逐字展开为 `auto_priority` 等剩余字段 |

最终生成一条描述符：

```cpp
{
    "unroll_768b",                           // id
    "mont_mul_unroll_768b",                  // cl_name
    kMontAliases_mul_unroll768,              // aliases → 指向第一阶段数组的指针
    "mont_mul/mont_mul_unroll_768b.cl",      // kernel_path
    22, 0, 24, 24, ECM_OS_ANY, ECM_GPU_ANY, OS_ANDROID, true, 1, 0
}
```

#### 关键点总结

| 概念 | 传递形态 | 最终用途 |
|------|---------|---------|
| `idt` | 宏参数 token → `#idt` 字符串化 | 描述符 `id` 字段（字符串），用于 UI 显示、调试日志、legacy 查找 |
| `stem` | 宏参数 token → `#stem` 字符串化拼接 | 描述符 `cl_name`（如 `mont_mul_unroll_768b`）和 `kernel_path` |
| `al` | 宏参数 token → `##al` token-paste | 拼出变量名 `kMontAliases_mul_unroll768`，该变量由 ECM_MONT_ALIAS_TABLE 定义 |
| `"unroll_768b"` | 第一阶段数组内的字符串字面量 | 别名表 [0] 项，如 `--mul unroll_768b` CLI 参数匹配 |

`ECM_MONT_ALIAS_TABLE` 定义的变量名后缀必须与 `ECM_MONT_OPERATORS` 传入的 `al` token 字面一致，
否则 token-paste 后得到的变量名不存在 → 编译错误。这是一种**编译期契约**：没有中间注册步骤，
错误会在编译期直接暴露。

```cpp
// add/sub：id 即文件/函数 stem，展开出 kAddModRegistry / kSubModRegistry
#define ECM_ADDSUB_OPERATORS(X)                                                \
    X(asm_384b, 26, kAddSubNoMinLimbs, kAddSub384MaxLimbs, 12u,                 \
      ECM_OS_ANY, ECM_GPU_AMD, 0)                                               \
    ... /* 其余算子各一行 */

#define ECM_ADD_ROW(idt, ...) {#idt, "add_mod_" #idt, kAddAliases_##idt, "add_mod/add_mod_" #idt ".cl", __VA_ARGS__},
#define ECM_SUB_ROW(idt, ...) {#idt, "sub_mod_" #idt, kSubAliases_##idt, "sub_mod/sub_mod_" #idt ".cl", __VA_ARGS__},
```

**命名铁律（X-macro 依赖它）**

- mont：`cl_name == "mont_<side>_" + stem`，`kernel_path == "mont_mul/mont_mul_" + stem + ".cl"`
  （mul/sqr 共用 mul 文件）。
- add/sub：`cl_name == "<fam>_" + id`，`kernel_path == "<fam>/<fam>_" + id + ".cl"`。
- `cl_name` == OpenCL 函数名 == 文件名主干。位宽后缀带 `b`（如 `384b`）。
- mont `al` token 必须与 `ECM_MONT_ALIAS_TABLE` 定义的数组名后缀字面一致（参见 §3.1）。

只要遵守命名铁律，新增算子就只是宏里加一行。

---

## 4. 描述符字段

```cpp
struct EcmMontPathDescriptor {           // special_mult 版去掉最后三个字段
    const char *id;            // CLI/别名匹配键（也是 JNI 下拉框的显示名）
    const char *cl_name;       // OpenCL 函数名
    const char *const *aliases;// 以 nullptr 结尾的别名数组指针（指向 ECM_MONT_ALIAS_TABLE 定义的静态数组）
    const char *kernel_path;   // 相对 kernels/opencl/
    int8_t  auto_priority;     // 自动选择优先级，越小越优先；-1=仅手动
    uint32_t min_limbs;        // 最小 limbs 数（0=不限）
    uint32_t max_limbs;        // 最大 limbs 数（0=不限）
    uint32_t max_container_limbs; // Exact-Fit Container 的 limbs 数；0=无容器限制
    uint32_t os_mask;          // OS 白名单（ECM_OS_*，低 16 位；ANY=不限）
    uint32_t gpu_vendor_mask;  // GPU 厂商白名单（ECM_GPU_*，高 16 位；ANY=不限）
    uint32_t gpu_vendor_exclude_mask; // 黑名单：按【完整 runtime mask = OS|GPU】判定，
                                      // 可排除 OS（如 OS_ANDROID）或 GPU（如 ECM_GPU_AMD），任意组合
    bool     fixed_width;        // 固定位宽算子
    uint8_t  coop_work_group_size;    // 协作工作组大小：==1 单线程，>1 多线程
    uint16_t local_scratch_u32;       // 本地内存占用（4096 专用）
};
```

选择逻辑（`resolve_*_side`）：

1. `auto`/空/`default` → 按 `auto_priority` 升序取第一个 `*_fits()` 通过者；全不通过则回退
   （mont 回退 `priv_opt`→`unroll_only_512`；addsub 回退 `fused_unroll`）。
2. 显式别名 → 命中后若 `*_fits()` 通过即用；否则在不低于该算子优先级的范围内自动回退。
3. 别名无法匹配 → 置 `unknown_path`（mont）/ 返回 `nullptr`（addsub）。

`*_fits()` 依次校验：N 位宽区间、容器位宽、OS 掩码、GPU 厂商白/黑名单、fixed_width 容器约束。

**关于固定位宽算子（fixed_width）**：384b / 512b / 4096 都是同一类固定位宽算子，没有「4096 专用」
的特殊判定。它们之间唯一的额外区别是 `coop_work_group_size`：==1 为单线程（在普通内核里直接
运行），>1 为多线程协作算子（加载 `ecm_stage1_coop.cl`）。协作 scratch / 整型 path 等也一律按
`coop_work_group_size` 与 `fixed_width` 判定，不再有 `is_4096_fixed_width` 之类的特例函数。

**别名单一数据源**：mul / sqr 的别名数组由 `ECM_MONT_ALIAS_TABLE(side, S)` 宏按 side 展开两次
生成（side-prefixed 兼容键如 `mont_mul_priv_*` / `mont_sqr_priv_*` 通过字符串字面量拼接自动产生），
不再手工维护两份。

---

## 5. 如何新增一个算子

以新增一个 add-mod 算子 `myvariant`（适用 ≤256 位、全平台）为例：

1. **写内核文件** `kernels/opencl/add_mod/add_mod_myvariant.cl`，导出
   `add_mod_myvariant(uint *r, const uint *a, const uint *b, const uint *N, uint limbs)`。
   sub 同理放 `sub_mod/sub_mod_myvariant.cl`，导出 `int sub_mod_myvariant(...)`。
   - 必须 **自包含**：只能依赖 `common/` 里始终加载的原语（`mp_priv.h.cl` 等），
     不能调用其他算子文件里的函数（拼装时只会加载被选中的算子文件）。
2. **加别名数组**（若只有默认别名，数组就一项 + `nullptr`）：
   ```cpp
   static const char *const kAddAliases_myvariant[] = {"myvariant", nullptr};
   static const char *const kSubAliases_myvariant[] = {"myvariant", nullptr};
   ```
3. **在 `ECM_ADDSUB_OPERATORS(X)` 加一行**（id 必须等于文件/函数 stem）：
   ```cpp
   X(myvariant, 18, kAddSubNoMinLimbs, 8u, 8u, ECM_OS_ANY, ECM_GPU_ANY, 0)
   ```
   优先级 18 表示比现有 128b(20) 更优先被自动选中（按需调整）。各字段依次为：
   id, priority, min_limbs, max_limbs, max_container_limbs, os_mask, gpu_vendor_mask, gpu_vendor_exclude_mask。
4. 若该 id 需要走 4096 协作整型分发或被 Android legacy 解析，再到 `addsub_id_kernel_path()`
   / `ecm_stage1.cl` 的 `ECM_ADDSUB_PATH_*` 同步一个枚举值；普通算子无需此步。
5. **重新构建并验证**（见 [§7](#7-验证)）。

新增 mont 算子类似：写 `mont_mul/mont_mul_<stem>.cl`（同时导出 `mont_mul_<stem>` 与
`mont_sqr_<stem>`），加 `kMulAliases_*`/`kSqrAliases_*`，在 `ECM_MONT_OPERATORS` 加一行。
4096 固定宽算子需正确设置 `fixed_width/coop_work_group_size/local_scratch_u32`。

新增 special_mult 算子类似：写 `special_mult/special_mult_<stem>.cl`，加 `kSpecialMultAliases_*`，
在 `ECM_SPECIAL_MULT_OPERATORS` 加一行。固定位宽版由 `tools/gen/gen_special_mult_unroll.py` 单源生成，
新增宽度只需在该脚本的 `WIDTHS` 加一行并重新生成。

### 删除一个算子

1. 删除 `ECM_*_OPERATORS` 中对应行与其别名数组。
2. 删除对应 `.cl` 文件。
3. 若它出现在 `addsub_id_kernel_path()` / `mont_id_kernel_path()` / coop 整型分发中，一并清理。
4. 重新构建验证。

---

## 6. 汇编加载顺序

`opencl_ecm_stage1_assemble_kernel_source(plan, load_file)` 顺序拼接：

1. 注入 `#define ECM_STAGE1_{MUL,SQR,ADD,SUB}_IMPL <cl_name>`
2. `common/stage1_config.h.cl` → `common/mp_priv.h.cl` → `common/ladder_helpers.h.cl`
3. `common/asm_common.h.cl`（仅当所选 add/sub 路径名含 `_asm_`）
4. 所选算子文件（mul, sqr, add, sub —— 按 `kernel_path` 去重）
5. `common/operator_iface.h.cl`（宏别名）
6. `ecm_stage1_coop.cl`（仅当 4096 且 `coop_work_group_size > 1`）
7. `ecm_stage1.cl`（主入口）

根目录 `kernels/opencl/`，环境变量 `ECM_KERNEL_ROOT` 可覆盖。

---

## 7. 验证

C++ 注册表逻辑可独立于 GPU 做**行为等价**校验（本次重构即如此验证：合并前后对
~38 个 path 字符串 × 12 个位宽 × 7 个容器 × 3 个 OS × 4 个 GPU 的解析结果、注册表全量字段、
构建选项与源文件列表，逐字节一致）。

GPU 端冒烟测试：

```powershell
cd d:\code\MPA-OpenCl
D:\code\vcpkg\downloads\tools\cmake-4.3.2-windows\cmake-4.3.2-windows-x86_64\bin\cmake --build build --config Debug --target ecm
echo '(2^151-1)' | .\build\Debug\ecm.exe -v -d 1 -gpu -sigma 3:2026     -gpucurves 16  1e4 0
echo '(2^421-1)' | .\build\Debug\ecm.exe -v -d 1 -gpu -sigma 3:20260611 -gpucurves 256 1e5 0
echo '(2^347-1)' | .\build\Debug\ecm.exe -v -d 1 -gpu -sigma 3:561219477 -gpucurves 32 1e5 0
echo '(2^641-1)' | .\build\Debug\ecm.exe -v -d 1 -gpu -sigma 3:20260611 -gpucurves 32  1e4 0
```

---

## 8. 已完成的合并与后续优化

以下项目已在 REV 14 中完成：
- Host 端注册表合并（三文件→一文件，四表→两单源，删除 `force_macro`）
- `add_mod_unroll_{128,192,256,384,512}b.cl` 与 `sub_mod_*` 由 `tools/gen/gen_mp_addsub_bits_stage1.py` 统一生成
- `special_mult` 固定位宽版由 `tools/gen/gen_special_mult_unroll.py` 统一生成
- 容器分配改为 Exact-Fit Container（按位宽精确匹配，移除 `stage1_need_512_container`）
- OpenCL 后端源文件（`cgbn_opencl.h`、`impl_opencl.cpp`）移至 `kernels/opencl/`

后续可考虑：
- `add_mod` 与 `sub_mod` 互为镜像，可考虑由同一模板宏生成两族，进一步减少重复。
- 各位宽 unroll 与通用 fused_unroll 语义接近，差异在于位宽专用版本只对活跃 limbs 做运算，属性能优化。
  合并前需对比各位宽 kernel 实测吞吐。

---

## 9. common 命名约定

`common/` 下全部文件统一为 `*.h.cl` 后缀（配置、limb 原语、ladder 辅助、算子接口、asm 公共块），
不再混用 `.cl` / `.inc.cl`。`asm_common.h.cl` 由 `tools/gen/gen_mp_addsub_bits_stage1.py` 生成。

> Android assets（`Android/ECM/app/src/main/assets/kernels/`）是独立副本，重命名后需重新同步资源
> （`tools/refactor/split_ecm_stage1_kernel_tree.py` / kernel_assets 流程）。

## 10. auto / manual 双实现与平台门控

每个展开宽度（192/256/384/512/768/1024）都有两份实现，**共用 `unroll_*b` 别名，id 不同**：

| 变体 | id | stem | 文件 | 形态 | 平台 |
|------|----|------|------|------|------|
| auto | `unroll_<W>b` | `unroll_<W>b` | `mont_mul/mont_mul_unroll_<W>b.cl` | `#pragma unroll` 循环 | 桌面/非安卓 |
| manual | `unroll_manual_<W>b` | `unroll_manual_<W>b` | `mont_mul/mont_mul_unroll_manual_<W>b.cl` | 手工直线标量变量 | 安卓 |

### 编译性能差异

auto 版（REV 15+）仅内层 `for j` 循环使用 `#pragma unroll`，外层 `for i` 保持为普通
循环。

**实测结果（Adreno 830, 768b）**：
- 编译时间：100s → 5s（**20× 加速**）
- 运行性能：比原双层 `#pragma unroll` **快 20%**

这是反直觉的结果——减少展开反而提速。根因是**过度展开导致寄存器溢出**：

```
双层 #pragma unroll (REV<15):
  展开后 IR: 1152 条操作在一个扁平作用域
  → 寄存器分配器面对 ~80 个跨迭代 live range
  → 寄存器文件仅 128×32bit，必然产生 spill
  → 每条 spill store+load ≈ 100 cycles (local memory latency)
  → 推测 ~50 次 spill → 5000 cycles 额外开销

仅内层 #pragma unroll (REV 15+):
  外循环 i=0..23 保持循环结构
  → 每次迭代 IR 仅 ~50 条操作
  → 寄存器分配器在独立迭代作用域内优化，仅需处理当前迭代的 live ranges
  → 寄存器文件内足够容纳 t[0..25] + 临时变量 → 零溢出
  → 消除全部 spill 开销 → +20% 性能
```

**关键机制三条**：

1. **寄存器压力塌缩**：外层展开后 24 次迭代的 `ai`/`carry`/`uv` 同时存在于 SSA 中，
   寄存器分配器被迫做出全局 spill 决策。外层不展开后，每次迭代的临时变量作用域独立，
   寄存器分配近似最优。

2. **VLIW 调度质量**：Adreno 的 3-way VLIW 调度器面对 1152 条操作时，搜索空间巨大，
   启发式算法退化为局部贪心，打包效率下降。面对 50 条操作时，调度接近全局最优。

3. **Loop-carried dependency 显式化**：外层展开使 `t[j]` 在 SSA 中出现 24 个独立版本
   （`t_j_0`, `t_j_1`...），内存别名分析保守推断可能导致 false dependency。
   外层不展开时，循环结构显式传达了迭代间的 RAW 依赖，SROA 能更激进地优化。

**双向优化策略**：manual 版走手工标量化路径（v4 累计 +12%，距基线 -28%）；
auto 版走编译器优化路径（内层 `#pragma unroll` + SRA，编译 5s，性能最优）。两者
互补：低端设备用 manual 保障兼容性，高端设备用 auto 享受最佳性能。

| 版本 | 形态 | Android 编译 | Android 相对基准 | 说明 |
|------|------|-------------|-----------------|------|
| auto (REV<15) | 双层 `#pragma unroll` | ~100s | **基准** | 过度展开，寄存器溢出严重 |
| auto (REV 15+) | 仅内层 `#pragma unroll` | ~5s | **+20%** | 零溢出，VLIW 调度最优 |
| manual v4 | 标量变量 + 消 D 数组 | ~10s | -8% | 手工寄存器控制 |

> 注：REV 15+ 后 auto 版性能超越原版，manual 版仅作兼容性备选。

v3→v4 优化原理：CIOS 最终减法阶段原本存储 24 个 `D_i = t_i - N[i] - borrow` 中间结果，
再统一 mask-select 输出。改为第一遍减法链只计算 `need_sub`（不存中间值），第二遍重新
计算减法并直接 inline mask-select 输出。代价是额外 24 次减法操作（对比内层 576 次 mul
微不足道），收益是节省 24 个 `uint` 寄存器变量，显著缓解 Adreno 的寄存器溢出。

**别名共享机制**：auto 和 manual 变体的别名数组**都**以 `"unroll_<W>b"` 为首项：
...（以下不变）

```cpp
// auto 变体
static const char *const kMontAliases_mul_unroll768[] = {
    "unroll_768b", "unroll_only_768b", ..., nullptr};

// manual 变体
static const char *const kMontAliases_mul_unroll768manual[] = {
    "unroll_768b", "unroll_manual_768b", ..., nullptr};
```

`resolve_mont_side` 在别名匹配循环中按注册表顺序扫描：桌面端 auto 先匹配且 `*_fits()` 通过
→ 返回 auto；安卓端 auto 匹配但被 `OS_ANDROID` 排除 mask 拦截 → 继续扫描 → manual
匹配且通过 → 返回 manual。

**下拉框区分**：Android JNI `build_mont_list` 用 `id` 字段（而非 `aliases[0]`）生成下拉列表，
因此 auto/prolog/manual 显示为 `unroll_768b` / `unroll_manual_768b` 两个独立可区分选项。

两份实现由 `tools/gen/gen_mont_unroll.py` 单源生成，算法逐位等价。

> 注：768b/1024b 位宽较大，其 manual 变体代码量可能超出部分 Adreno 编译器的单文件处理
> 能力上限。遇到编译失败时可退回 `--mul unroll_768b`（auto 变体），实测 Qualcomm Adreno 830
> 能成功编译 auto 版 768b。

## 11. 内核版本

`ECM_STAGE1_KERNEL_REV = 14` —— 容器按 Exact-Fit Container 动态分配（不再统一 16 limbs）；addsub 算子
入口 guard 改为对应位宽；special_mult 扩展为 192/256/384/512/768/1024b 固定位宽家族；`stage1_need_512_container`
及 `kStage1Container512Limbs` 已移除；`stage1_container_limbs` 仅按 add/sub 的 max_container_limbs 升级；
OpenCL 后端源文件（`cgbn_opencl.h`、`impl_opencl.cpp`）移至 `kernels/opencl/`；注册表字段从 `min_n_bits`/`max_n_bits`
统一为 `min_limbs`/`max_limbs`；auto unroll 外层循环去掉 `#pragma unroll`（仅保留内层 SRA 展开），
编译 100s→5s，运行 +20%（消除寄存器溢出）；manual unroll 保留标量变体（标量变量 + 两遍减法 pass，累计 +12%）。
