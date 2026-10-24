# Group Order Factor helper dependency

`ecm_gui_v2.exe` calls the configured or PATH-discovered PARI/GP directly for
order analysis. Only local CPU/GMP execution (explicit or Auto fallback) starts
`ecm_group_order_ui.exe`. The helper dynamically links `gmp-10.dll` and can load the optional local
`libpari.dll` plus `pari-data/seadata/`. The GUI itself loads neither math
library. A working external GP does not require any local helper/library.
The GP executable is configured separately and is not redistributed; the
native PARI library and SEA tables are included in the full candidate.

The local candidate uses the existing `third_party/gmp-zen3/dist` installation.
Its build selects x86_64 Zen3 assembly/BMI2 kernels; do not qualify this DLL for
all Windows x64 CPUs from this candidate's tests. The package records the DLL
SHA256 and includes the actual local GMP source tree and build configuration in
its source archive. GMP may be rebuilt and the external DLL replaced; a matching
header/import library/runtime set is required when rebuilding the helper.

GMP is copyright the Free Software Foundation and contributors. Its source
README offers LGPL v3 or later, or GPL v2 or later. This distribution includes
the LGPL v3 and GPL v3 texts, all original source notices, and the corresponding
local source. The DLL remains external and replaceable. Consult GMP's COPYING
files for the complete terms; this notice does not replace them.

The source snapshot includes `build/config.status`, `config.h`, and `config.m4`
as build evidence. Paths in those generated files describe the original local
build; select a new prefix and toolchain when rebuilding GMP on another machine.
The GUI's MSVC/CMake build consumes the installed GMP artifacts and does not run
GMP's autotools build. Compute workers and their GMP installations are separate.

## Local PARI C interface

The optional local engine is PARI 2.17.3, Windows x64, single-threaded MinGW,
with 64-bit PARI words. The adapter resolves typed C entry points from the DLL;
it does not run gp.exe, GP scripts or the GP interpreter. The exact short
Weierstrass curve and original point come from the existing GMP construction.
The library supplies group counting (including SEA), integer factorization,
primality proofs and point order. GMP independently checks products, Hasse and
point annihilation/minimality. This capability currently covers prime moduli
through 192 bits; five seconds is the request budget, not a universal completion
guarantee for arbitrary 192-bit factorizations.

Local dependencies are prepared with
`tools/build/test/prepare_gui_v2_pari.py --installed <PARI-2.17.3-directory> --archive <source-tar.gz>`.
The original source archive SHA256 is
`8d9c4fcd584c468d27e0f23c36836587284452094c4b1c404c20c4b810462dcb`.
Upstream: https://pari.math.u-bordeaux.fr/pub/pari/unix/pari-2.17.3.tar.gz
Source mirror: https://deb.debian.org/debian/pool/main/p/pari/pari_2.17.3.orig.tar.gz
The prepared ignored tree is `third_party/pari-2.17.3/`; its provenance records
all DLL, configured-header, data and archive hashes. The tested DLL SHA256 is
`50420f175ce1753870d17188e1330f8a58b5d46ac1468c5d32a6edeb2aa3dccd`.
This is an existing upstream runtime from the local PARI installation; its
original build flags are not independently reconstructed. Configured headers
identify GCC 12-posix / Win64 / GMP / single-threaded configuration. Rebuilding
uses the original archive's Configure, config/ and Windows documentation;
exact original binary reproducibility has not been qualified.

PARI is copyright the PARI group and contributors and is licensed under GPL v2
or later, compatible with this project's GPL v3 distribution. The candidate
includes PARI's COPYING, the complete unmodified source archive, configured
headers, matching SEA data and provenance in its source snapshot. The ECHIDNA
modular-polynomial tables are credited to David R. Kohel in the included
`pari-data/seadata/README`; preserve that notice when redistributing.
The DLL remains separate and replaceable by an ABI-compatible Win64 2.17.3
build. No PARI executable is included. Missing or incompatible optional files
return an explicit diagnostic in PARI-only mode; Auto may try external GP.
The bounded GMP engine is available only when explicitly selected.

The default GUI-v2 build requires the prepared runtime. A deliberate build without PARI
 can use `-WithoutLocalPari` with the test builder in a fresh directory, or
`-DECM_GUI_V2_LOCAL_PARI=OFF` with standalone CMake. Runtime DLL removal is not
part of that build switch; never reuse a directory containing retained PARI
files to claim a no-PARI qualification. Stage1, Stage2 and GUI now share
`service/diagnostics/` and the same DLL-first policy; a no-PARI build never
enables GMP automatically. Shared host dependencies, source archives and
frozen identities are produced by `tools/packaging/diagnostics_runtime.py`.
