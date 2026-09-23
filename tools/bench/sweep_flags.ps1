# Sequential MSVC flag sweep for the Edwards speed benchmark (M991 B1=1e6 w=8).
$ErrorActionPreference = "Continue"
$root = "D:\code\MPA-OpenCl"
$gmp = "D:\code\vcpkg\installed\x64-windows"

$sets = @(
    @{ Tag="O2";            Flags="/O2" },
    @{ Tag="O2_AVX2";       Flags="/O2 /arch:AVX2" },
    @{ Tag="O2_AVX512";     Flags="/O2 /arch:AVX512" },
    @{ Tag="O2_GL";         Flags="/O2 /GL" },
    @{ Tag="O2_AMD64";      Flags="/O2 /favor:AMD64" },
    @{ Tag="Ox";            Flags="/Ox" },
    @{ Tag="O2_GL_AVX2";    Flags="/O2 /GL /arch:AVX2" }
)

$env:PATH = "$gmp\bin;" + $env:PATH
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

foreach ($s in $sets) {
    $Tag = $s.Tag
    $Flags = $s.Flags
    $exe = Join-Path $root "tools\bench\_speed_$Tag.exe"
    $cline = "cl /nologo $Flags /EHsc /utf-8 /I `"$gmp\include`" /I `"$root\src\cpu`" `"$root\tools\bench\ecm_edwards_speed.cpp`" `"$root\src\cpu\ecm_edwards_cpu.cpp`" /Fe:`"$exe`" /link `"$gmp\lib\gmp.lib`""
    $build = "@echo off`r`ncall `"$vcvars`" >nul`r`n$cline`r`n"
    $tmp = Join-Path $env:TEMP "build_speed_$Tag.bat"
    Set-Content -Path $tmp -Value $build -Encoding ASCII
    cmd /c $tmp | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "=== $Tag : COMPILE FAILED ==="; continue }
    Write-Host "=== $Tag ($Flags) ==="
    & $exe 991 1000000 8 3
    Write-Host ""
}
Write-Host "DONE"
