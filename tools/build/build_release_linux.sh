#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
arch=89
ptx=89
jobs=${ECM_BUILD_JOBS:-8}
build_root="$repo/build_linux_release"
out="$repo/release"
upstream="$repo/.refactor/gmp-ecm"
skip_build=false
skip_package=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) arch=$2; shift ;;
        --ptx-arch) ptx=$2; shift ;;
        --jobs) jobs=$2; shift ;;
        --build-root) build_root=$2; shift ;;
        --out) out=$2; shift ;;
        --libecm-source) upstream=$2; shift ;;
        --skip-build) skip_build=true ;;
        --skip-package) skip_package=true ;;
        -h|--help)
            echo "Usage: bash $0 [--arch 89] [--ptx-arch 89|none] [--jobs 8]"
            echo "    [--build-root DIR] [--out DIR] [--libecm-source DIR] [--skip-build] [--skip-package]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ||
      ! $arch =~ ^[0-9]+$ || ! $ptx =~ ^([0-9]+|none)$ ||
      ! $jobs =~ ^([1-9]|[1-5][0-9]|6[0-4])$ ]]; then
    echo "Use Linux x86-64 with numeric CUDA architectures and 1–64 jobs." >&2
    exit 2
fi
nvcc=$(realpath -- "${ECM_NVCC:-/usr/local/cuda/bin/nvcc}")
toolkit=$("$nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+),.*/\1/p')
[[ $toolkit ]] || { echo "Cannot identify Linux nvcc: $nvcc" >&2; exit 2; }
"$nvcc" --list-gpu-code | grep -Fx "sm_$arch"
ptx_arg=
if [[ $ptx != none ]]; then
    "$nvcc" --list-gpu-arch | grep -Fx "compute_$ptx"
    ptx_arg=$ptx
fi

profile="cu$toolkit-sm$arch-ptx$ptx"
build_root=$(realpath -m -- "$build_root")
out=$(realpath -m -- "$out")
upstream=$(realpath -- "$upstream")
if [[ $build_root == "$repo" || $out == "$repo" || $out/ == "$build_root/"* || $build_root/ == "$out/"* ||
      $build_root/ == "$repo/.git/"* || $out/ == "$repo/.git/"* ]]; then
    echo "Use separate build and output directories." >&2
    exit 2
fi
if [[ $build_root/ == "$repo/"* ]] &&
   [[ $(git -C "$repo" ls-files -- "${build_root#"$repo/"}") ]]; then
    echo "Build root contains tracked repository files: $build_root" >&2
    exit 2
fi
build="$build_root/$profile"
for directory in "$build_root" "$build"; do
    mkdir -p -- "$directory"
    printf '*\n' > "$directory/.gitignore"
done
exec 9>"$build/.release.lock"
flock -n 9 || { echo "Release profile already in use: $build" >&2; exit 1; }
cd -- "$repo"

multiarch=$(/usr/bin/gcc -print-multiarch)
gmp_include="/usr/include/$multiarch"
system_lib="/usr/lib/$multiarch"
gmp="$system_lib/libgmp.so"
for dependency in "$gmp_include/gmp.h" "$gmp" "$system_lib"/lib{ssl,crypto,OpenCL}.so; do
    [[ -f $dependency ]] || { echo "Missing Ubuntu development dependency: $dependency" >&2; exit 1; }
done
revision=$(git -C "$upstream" rev-parse --verify HEAD)
dependency_inputs=("$gmp_include/gmp.h" "$gmp" /usr/bin/gcc)
for tool in cc1 as ld; do
    path=$(/usr/bin/gcc "-print-prog-name=$tool")
    dependency_inputs+=("$(command -v "$path")")
done
dependency_key=$( { echo "$revision"; sha256sum "${dependency_inputs[@]}"; } | sha256sum | cut -d ' ' -f1 )
dep="$build/libecm-$dependency_key"
if ! (cd -- "$dep" 2>/dev/null && sha256sum --status --check verified.sha256); then
    if $skip_build; then
        echo "No verified GMP-ECM dependency for $profile" >&2
        exit 1
    fi
    mkdir -p -- "$dep"
    printf '*\n' > "$dep/.gitignore"
    # Stage the clean Git revision in a new directory after any failed verification.
    generation=$(mktemp -d "$dep/generation.XXXXXXXX")
    mkdir -p -- "$generation/source" "$generation/build"
    git -C "$upstream" archive "$revision" > "$generation/source.tar"
    tar -xf "$generation/source.tar" -C "$generation/source"
    (cd -- "$generation/source" && autoreconf -fi)
    (cd -- "$generation/build" && ../source/configure \
        --enable-openmp --disable-asm-redc --enable-static --disable-shared \
        --with-gmp-include="$gmp_include" --with-gmp-lib="$system_lib" CC=/usr/bin/gcc CFLAGS=-O2)
    make -C "$generation/build" -j "$jobs" libecm.la
    (cd -- "$dep" && sha256sum "${generation#"$dep/"}"/{source.tar,source/ecm-ecm.h,build/ecm.h,build/.libs/libecm.a} > verified.sha256.tmp &&
        mv -- verified.sha256.tmp verified.sha256)
fi
generation="$dep/$(head -n1 "$dep/verified.sha256" | cut -d ' ' -f3 | cut -d / -f1)"
ecm_source="$generation/source"
ecm_build="$generation/build"

if ! $skip_build; then
    cmake -S "$repo" -B "$build" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
        -DECM_ENABLE_CUDA=ON -DECM_CUDA_ARCHITECTURES="$arch" -DECM_TIERS= \
        -DECM_CUDA_EMBED_PTX=OFF -DECM_CUDA_PTX_ARCH="$ptx_arg" -DECM_CUDA_COMPRESS=ON \
        -DECM_CUDA_FULL_BUILD=ON -DECM_CUDA_ENABLE_PRAC=ON \
        -DECM_BUILD_GUI=OFF -DECM_BUILD_GUI_V2=OFF -DECM_BUILD_TOOLS=OFF \
        -DECM_BUILD_CPU_STAGE2=ON -DECM_OPENCL_PORTABLE_RESOURCES=ON \
        -DGMP_INCLUDE_DIR="$gmp_include" -DGMP_LIBRARY="$gmp" \
        -DECM_LIBECM_SOURCE_DIR="$ecm_source" -DECM_LIBECM_BUILD_DIR="$ecm_build"
fi
verify=(--build "$build" --nvcc "$nvcc" --jobs "$jobs")
for path in "$gmp" "$system_lib"/lib{ssl,crypto,OpenCL}.so "$generation/source.tar" \
        "$ecm_source/ecm-ecm.h" "$ecm_build/ecm.h" "$ecm_build/.libs/libecm.a"; do
    verify+=(--input "$path")
done
if $skip_build; then verify+=(--skip-build); fi
python3 tools/build/internal/verified_linux_build.py "${verify[@]}"
if ! $skip_package; then
    python3 tools/packaging/package_release_linux.py --build "$build" --out "$out" \
        --nvcc "$nvcc" --profile "$profile" --arch "$arch" --ptx-arch "$ptx" \
        --libecm-source "$generation/source.tar"
fi
