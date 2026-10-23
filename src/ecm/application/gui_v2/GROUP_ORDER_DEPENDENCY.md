# Group Order Factor helper dependency

`ecm_gui_v2.exe` starts `ecm_group_order_ui.exe` for explicit order analysis.
The helper dynamically links `gmp-10.dll`; the GUI itself does not load GMP.
PARI/GP is optional, is configured separately, and is not redistributed.

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
