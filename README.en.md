# ECM-ELF

[中文](README.md) | English

Elliptic Curve Method tools with CUDA/CGBN and OpenCL GPU Stage1, GMP and optional AVX-512 IFMA CPU Stage1, a standalone CUDA polynomial Stage2, and Windows GUI, queue and factor-dataset utilities.

## Programs

- `ecm_cuda.exe`: NVIDIA GPU Stage1, with the curve families and bit-width tiers included in the build, batching and checkpoints.
- `ecm.exe`: CPU Montgomery/Edwards and OpenCL build entry points.
- `ecm_cuda_stage2.exe`: normalized PARAM0 Stage1 saves, product trees and exact NTT; a separate queue resumes completed curves.
- `ecm_gui.exe`: configuration, Stage1 workers, queue generation, logs, GPU monitoring and results.
- `ecm_p95feeder`: Prime95 handoff for local CPU saves; the driver also has completed-task handoff settings.

Save/checkpoint compatibility depends on the input format and parameterization. Manual Stage2 supports 2…16384-bit inputs. Auto B2 requires a matching audited cost profile; the current production combination needs an explicit B2.

## Getting started

1. Select a release matching the GPU architecture and CPU dependencies, or use the launchers in `tools/build/`.
2. Edit the shared `ecm.ini`. Stage1 uses `worktodo.txt`; Stage2 uses its separate `stage2_worktodo.txt`.
3. Generate completed Stage1 saves, then run Stage2 directly or through its queue. The two executables remain separate.
4. Consult executable help and the bilingual [INI reference](docs/ECM_INI_REFERENCE.md) for options, defaults and paths.

Stage1/Stage2 each have development, local production and release production PS1/BAT entry points. Release builds target60,70,75,86,89,120 with CUDA12.6/13.3 as supported and package automatically. Compilation does not establish runtime validation on every architecture.

## Documentation

- [Documentation index](docs/README.md): current usage, implementation, performance evidence and TODO.
- [Build and release](docs/usage/BUILD.md), [build launchers](tools/build/README.md).
- [Stage1](docs/usage/STAGE1.md), [Stage2](docs/usage/STAGE2.md), [GUI](docs/usage/GUI.md).
- [Tools](tools/README.md), [factor dataset](tools/ecm_dataset/README.md), [Android](Android/ECM/README_ECM_FACTORIZATION.md).

Experiments, benchmark summaries and generated figures belong in ignored `data/` directories. External sources belong in `.refactor/`; papers and translations remain in `docs/paper/`. Project status documents describe current behavior; [fixed references](docs/reference/README.md) preserve mathematical and third-party implementation analysis.

## Dependencies and license

The project builds on GMP-ECM, uses CGBN for CUDA big integers and Dear ImGui for the GUI. Builds need the appropriate CUDA/OpenCL, C++ toolchain and GMP. Releases default to generic x64 GMP; a local Zen3 library is not a guarantee of compatibility with every CPU.

See [LICENSE](LICENSE) and retain dependency licenses when distributing binaries.
