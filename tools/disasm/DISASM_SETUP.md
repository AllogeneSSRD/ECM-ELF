# Disassembly Tool Setup (Windows)

## 1) Install tools

From repository root:

```powershell
.\tools\install_disasm_tools.ps1
```

This script installs:

- LLVM (`llvm-objdump`, `llvm-readobj`) via `winget`
- Tries to install Radeon GPU Analyzer (`rga`) via `winget` if available

If `rga` is not available in `winget`, install manually from:

- https://gpuopen.com/rga/

## 2) Verify tools

```powershell
.\tools\verify_disasm_tools.ps1
```

## 3) Export AMD OpenCL binary (4096-bit mont kernels)

```powershell
.\build\Debug\opencl_mont_isa_export.exe
```

Output binary:

- `bench/mont_isa_4096_amd.bin`

## 4) Disassemble (examples)

Tool command line varies by version. Typical pattern:

```powershell
rga -s opencl -c gfx1150 --binary "bench\mont_isa_4096_amd.bin" --isa "bench\mont_isa_4096_amd.isa.txt"
```

If your `rga --help` differs, adapt the arguments according to your installed version.
