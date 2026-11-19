#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
arch=${1:-89}
if [[ $# -gt 1 || ! $arch =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 [CUDA compute capability, e.g. 89]" >&2
    exit 2
fi
nvcc=${ECM_NVCC:-/usr/local/cuda/bin/nvcc}
if [[ ! -x $nvcc ]]; then
    echo "Linux nvcc not found at $nvcc. Set ECM_NVCC to its full path." >&2
    exit 2
fi
nvcc=$(realpath -- "$nvcc")

build="$repo/build_stage1_wsl_sm$arch"
mkdir -p -- "$build"
printf '*\n' > "$build/.gitignore"
cmake -S "$repo" -B "$build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DECM_ENABLE_CUDA=ON -DECM_CUDA_ARCHITECTURES="$arch" \
    -DECM_TIERS= -DECM_CUDA_EMBED_PTX=OFF -DECM_CUDA_PTX_ARCH= -DECM_CUDA_COMPRESS=ON \
    -DECM_CUDA_FULL_BUILD=ON -DECM_CUDA_ENABLE_PRAC=ON \
    -DECM_BUILD_GUI=OFF -DECM_BUILD_TOOLS=OFF
cmake --build "$build" --target ecm_cuda --parallel "${ECM_BUILD_JOBS:-6}"
"$build/ecm_cuda" --version
