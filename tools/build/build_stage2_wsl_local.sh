#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
upstream=${1:-"$repo/.refactor/gmp-ecm"}
if [[ $# -gt 1 || ! -f $upstream/configure.ac ]] ||
   ! revision=$(git -C "$upstream" rev-parse --verify HEAD); then
    echo "Usage: bash $0 [path to a GMP-ECM Git checkout]" >&2
    exit 2
fi

# Use Ubuntu's native development package for both Autotools and CMake.
multiarch=$(cc -print-multiarch)
gmp_include="/usr/include/$multiarch"
gmp_lib="/usr/lib/$multiarch"
if [[ ! -f $gmp_include/gmp.h || ! -f $gmp_lib/libgmp.so ]]; then
    echo "Install Ubuntu's native libgmp-dev package." >&2
    exit 2
fi

build="$repo/build_stage2_wsl_cpu"
mkdir -p -- "$build"
printf '*\n' > "$build/.gitignore"
source_dir="$build/libecm-src-$revision"
library_dir="$build/libecm-$revision"
mkdir -p -- "$source_dir" "$library_dir"

if [[ ! -f $library_dir/Makefile ]]; then
    # A Git archive keeps Autotools scripts in LF on a Windows checkout.
    git -C "$upstream" archive "$revision" | tar -xf - -C "$source_dir"
    (cd -- "$source_dir" && autoreconf -fi)
    (cd -- "$library_dir" && "$source_dir/configure" \
        --enable-openmp --disable-asm-redc --enable-static --disable-shared \
        --with-gmp-include="$gmp_include" --with-gmp-lib="$gmp_lib" CFLAGS=-O2)
fi
make -C "$library_dir" -j "${ECM_BUILD_JOBS:-6}" libecm.la

cmake -S "$repo" -B "$build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DECM_ENABLE_CUDA=OFF \
    -DECM_BUILD_GUI=OFF -DECM_BUILD_TOOLS=OFF -DECM_BUILD_CPU_STAGE2=ON \
    -DGMP_INCLUDE_DIR="$gmp_include" -DGMP_LIBRARY="$gmp_lib/libgmp.so" \
    -DECM_LIBECM_SOURCE_DIR="$source_dir" -DECM_LIBECM_BUILD_DIR="$library_dir"
cmake --build "$build" --target ecm_cpu_stage2 --parallel "${ECM_BUILD_JOBS:-6}"
"$build/ecm_stage2" --version
