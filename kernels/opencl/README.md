# ECM Stage1 OpenCL kernels

Stage1 sources live under `kernels/opencl/`. Host assembles kernel source via
`opencl_ecm_stage1_assemble_kernel_source()`: injects `ECM_STAGE1_*_IMPL` macros, then
concatenates only the files from `opencl_ecm_stage1_kernel_source_paths()`.

算子选择/注入/拼装全部在单一文件 `src/opencl_ecm_path_registry.cpp` 中（mont_path /
addsub_path 已合并进来）。新增/删除算子的完整指南见
[`docs/architecture/OPENCL.md`](../../docs/architecture/OPENCL.md)。

> 注意：拼装时**只加载被选中的算子文件**，因此每个算子 `.cl` 必须自包含——只能依赖
> `common/` 中始终加载的原语，不能调用其他算子文件里的函数。

## Directory layout

| Directory | Role |
|-----------|------|
| `common/` | Config, limb primitives, ladder helpers, operator interface |
| `mont_mul/` | Montgomery mul + sqr implementations (paired in same file) |
| `add_mod/` | Modular addition operators |
| `sub_mod/` | Modular subtraction operators |
| `ecm_stage1.cl` | Ladder logic only (`double_add_v2`, non-coop `kernel_double_add`) |
| `ecm_stage1_coop.cl` | Optional 4096-bit cooperative WG supplement |

## Naming convention

**Rule:** `cl_name` == OpenCL function name == filename stem (bits suffix with `b`).

| Family | Example `id` | `cl_name` | `kernel_path` |
|--------|--------------|-----------|---------------|
| Montgomery mul | `unroll_only_384` | `mont_mul_unroll_384b` | `mont_mul/mont_mul_unroll_384b.cl` |
| Montgomery sqr | (paired) | `mont_sqr_unroll_384b` | same file as mul |
| Add mod | `unroll_384b` | `add_mod_unroll_384b` | `add_mod/add_mod_unroll_384b.cl` |
| Sub mod | `unroll_384b` | `sub_mod_unroll_384b` | `sub_mod/sub_mod_unroll_384b.cl` |

## Operator ABI

Host injects before `common/operator_iface.h.cl`:

```c
#define ECM_STAGE1_MUL_IMPL mont_mul_unroll_384b
#define ECM_STAGE1_SQR_IMPL mont_sqr_unroll_384b
#define ECM_STAGE1_ADD_IMPL add_mod_unroll_384b
#define ECM_STAGE1_SUB_IMPL sub_mod_unroll_384b
```

Ladder code calls `mont_mul`, `mont_sqr`, `add_mod`, `sub_mod` (macros alias to selected impl).

## Load order

1. Injected `ECM_STAGE1_*_IMPL` macros
2. `common/stage1_config.h.cl`, `common/mp_priv.h.cl`, `common/ladder_helpers.cl`
3. Selected operator files (mul, sqr, add, sub — deduped by path)
4. `common/operator_iface.h.cl`
5. `ecm_stage1_coop.cl` (only when `ECM_STAGE1_COOP_WG > 1`)
6. `ecm_stage1.cl`
