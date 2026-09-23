#!/bin/sh
# Build GMP 6.3.0 with the x86_64/zen3 (BMI2: mulx + adcx/adox) assembly kernels,
# targeting x86_64-pc-mingw32 with MSVC (cl.exe) + clang as the assembler.
#
# Mirrors the vcpkg gmp port configuration, except that the CPU path is forced to
#   path_64="x86_64/zen3 x86_64/zen2 x86_64/zen x86_64"
# instead of the generic default "x86_64/k8 x86_64".
#
# Run from cmd after vcvars64.bat so cl.exe / lib.exe / dumpbin.exe are on PATH.
set -e

MSYS=/d/code/vcpkg/downloads/tools/msys2/634f555cfe9f8889
CLANG=/d/code/vcpkg/downloads/tools/clang/clang-15.0.6/bin
ROOT=/d/code/MPA-OpenCl/third_party/gmp-zen3
SRC=$ROOT/src
BUILD=$ROOT/build
PREFIX=$ROOT/dist
SRC_REL=../src

# MSYS paths are APPENDED so `nm`/`link` resolution matches the vcpkg build.
export PATH="$PATH:/usr/share/automake-1.16:/usr/bin:/bin:$CLANG"

mkdir -p "$BUILD"
cd "$BUILD"

if [ ! -f Makefile ]; then
  echo "=== configure ==="
  # NOTE: configure MUST be invoked through a relative path. GMP's configure
  # bakes $srcdir into `#include "$srcdir/gmp-h.in"` probes, and cl.exe cannot
  # resolve MSYS-style absolute paths such as /d/code/...
  CC="compile cl.exe" CXX="compile cl.exe" \
  CFLAGS="-Xcompiler -nologo -Xcompiler -utf-8 -Xcompiler -MP -Xcompiler -MD -Xcompiler -O2 -Xcompiler -Oi -Xcompiler -Gy" \
  LDFLAGS="-Xlinker -Xlinker -Xlinker -machine:x64 -Xlinker -Xlinker -Xlinker -nologo -Xlinker -Xlinker -Xlinker -DEBUG -Xlinker -Xlinker -Xlinker -INCREMENTAL:NO -Xlinker -Xlinker -Xlinker -OPT:REF -Xlinker -Xlinker -Xlinker -OPT:ICF" \
  CCAS="clang.exe" ASMFLAGS="-c --target=x86_64-pc-windows-msvc" \
  "$SRC_REL/configure" --build=x86_64-pc-mingw32 \
      ac_cv_func_memset=yes gmp_cv_asm_w32=.word gmp_cv_check_libm_for_build=no \
      gmp_cv_prog_exeext_for_build=.exe ac_cv_prog_ac_ct_STRIP=: \
      gl_cv_double_slash_root=yes ac_cv_func_memmove=yes \
      --disable-silent-rules --enable-shared --disable-static \
      --with-readline=no --prefix="$PREFIX" > configure-out.log 2>&1 \
    || { echo "CONFIGURE FAILED"; tail -80 configure-out.log; exit 1; }
  echo "=== configure ok; CPU path ==="
  grep -E "^path_64|^ABI|^CC|^CCAS|^ASMFLAGS" Makefile || true
  grep -E "x86_64/zen3|zen3" config.m4 2>/dev/null | head -5 || true
fi

echo "=== make ==="
make -j24 > make-out.log 2>&1 || { echo "MAKE FAILED"; tail -80 make-out.log; exit 1; }
echo "=== make install ==="
make install > install-out.log 2>&1 || { echo "INSTALL FAILED"; tail -40 install-out.log; exit 1; }

echo "BUILD-DONE"
ls -la "$PREFIX/bin" 2>/dev/null || true
ls -la "$PREFIX/lib" 2>/dev/null || true
