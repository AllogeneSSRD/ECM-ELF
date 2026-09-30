#Requires -Version 5.1
<#
.SYNOPSIS
    Build and run the gwnum IBDWT probe (tools/gwnum_probe/gwnum_probe.cpp).

.DESCRIPTION
    Links prime95's prebuilt x64 gwnum library (gwnum64.lib) plus this repo's own
    Zen3-optimised GMP, so the AVX2 / AVX-512 IBDWT multiply can be compared with
    the GMP arithmetic the ECM program uses today. Read-only with respect to the
    prime95 tree: it only compiles against it.

.PARAMETER Prime95Source
    Root of an extracted prime95 source tree (the directory that contains gwnum\).
    Defaults to the newest D:\code\GIMPS\p95v*.source directory.

.PARAMETER Quick
    Skip the >20,000-bit sizes (fast smoke run of the harness itself).

.PARAMETER VcVars
    Explicit path to vcvars64.bat when the automatic search fails.
#>
param(
    [string]$Prime95Source = '',
    [switch]$Quick,
    [string]$VcVars = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

if (-not $Prime95Source) {
    $cand = Get-ChildItem 'D:\code\GIMPS' -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'p95v*.source' -and (Test-Path (Join-Path $_.FullName 'gwnum\gwnum64.lib')) } |
        Sort-Object Name -Descending
    if (-not $cand) { throw 'no prime95 source tree with gwnum\gwnum64.lib found (pass -Prime95Source)' }
    $Prime95Source = $cand[0].FullName
}
$gwnum = Join-Path $Prime95Source 'gwnum'
if (-not (Test-Path (Join-Path $gwnum 'gwnum64.lib'))) { throw "no gwnum64.lib under $gwnum" }

$gmpRoot = Join-Path $repoRoot 'third_party\gmp-zen3\dist'
if (-not (Test-Path (Join-Path $gmpRoot 'include\gmp.h'))) { throw "no gmp.h under $gmpRoot" }

if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) { throw 'vcvars64.bat not found (install Desktop development with C++)' }
    $VcVars = $cand[0].FullName
}

$outDir = Join-Path $PSScriptRoot 'bin'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host ("prime95 source : " + $Prime95Source)
Write-Host ("gmp            : " + $gmpRoot)
Write-Host ("vcvars         : " + $VcVars)

# /MT, not /MD: the prebuilt gwnum64.lib objects were compiled with the static release
# runtime, and /MD produces LNK2038 ("RuntimeLibrary mismatch") followed by a cascade of
# bogus unresolved externals (measured 2026-09-29). advapi32 is needed by gwnum's
# large-page support (gwutil.obj: OpenProcessToken/AdjustTokenPrivileges).
$log = Join-Path $outDir 'build.log'
Remove-Item $log -ErrorAction SilentlyContinue
$cmd = @(
    ('cd /d "{0}"' -f $outDir),
    ('cl /c /nologo /O2 /EHsc /MT /D_CRT_SECURE_NO_WARNINGS /I"{0}" /I"{1}" "{2}"' -f `
        $gwnum, (Join-Path $gmpRoot 'include'), (Join-Path $PSScriptRoot 'gwnum_probe.cpp')),
    ('link /nologo /OUT:gwnum_probe.exe gwnum_probe.obj "{0}" "{1}" advapi32.lib' -f `
        (Join-Path $gwnum 'gwnum64.lib'), (Join-Path $gmpRoot 'lib\gmp.lib'))
) -join ' && '
$wrapped = "call `"$VcVars`" >nul 2>&1 && ($cmd)"
cmd /c $wrapped 2>&1 | Tee-Object -FilePath $log
# Exit codes from a cmd chain are unreliable here (the whole line is parsed before it runs,
# so %ERRORLEVEL% expands to the PREVIOUS value) -- check the artifact instead.
$exePath = Join-Path $outDir 'gwnum_probe.exe'
$linkErrors = @(Get-Content $log | Select-String -Pattern 'error LNK|error C')
if (-not (Test-Path $exePath) -or $linkErrors.Count -gt 0) {
    throw ("build failed -- " + $linkErrors.Count + " compiler/linker errors, see " + $log)
}

# gmp-10.dll must sit next to the exe.
Copy-Item (Join-Path $gmpRoot 'bin\gmp-10.dll') $outDir -Force -ErrorAction SilentlyContinue
if (-not (Test-Path (Join-Path $outDir 'gmp-10.dll'))) {
    $dll = Get-ChildItem (Join-Path $repoRoot 'third_party\gmp-zen3') -Recurse -Filter 'gmp-10.dll' |
        Select-Object -First 1
    if ($dll) { Copy-Item $dll.FullName $outDir -Force }
}

$exe = Join-Path $outDir 'gwnum_probe.exe'
$argv = @()
if ($Quick) { $argv += '--quick' }
Write-Host ""
& $exe @argv
exit $LASTEXITCODE
