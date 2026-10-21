#Requires -Version 5.1
param(
    [string]$BuildDir = 'build_gui_v2_test',
    [string]$VcVars = '',
    [string]$VulkanRoot = '',
    [string]$BackendRoot = '',
    [switch]$TestHelpers
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
foreach ($legacy in @($repo, (Join-Path $repo 'build_gui'), (Join-Path $repo 'build_gui_local'))) {
    if ($BuildDir.TrimEnd('\') -eq $legacy.TrimEnd('\')) { throw 'GUI v2 requires an independent build directory' }
}
if (-not $VcVars) {
    $candidate = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
        Where-Object FullName -NotMatch 'BuildTools' | Select-Object -First 1
    if (-not $candidate) { throw 'MSVC vcvars64.bat not found; pass -VcVars' }
    $VcVars = $candidate.FullName
}
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
$defs = @('-DCMAKE_BUILD_TYPE=Release')
if ($VulkanRoot) { $defs += "-DECM_GUI_V2_VULKAN=`"$VulkanRoot`"" }
if ($BackendRoot) { $defs += "-DECM_GUI_V2_BACKEND=`"$BackendRoot`"" }
$configure = "cmake -S `"$repo/src/ecm/application/gui_v2`" -B `"$BuildDir`" -G `"NMake Makefiles`" " + ($defs -join ' ')
function Invoke-Compiler([string]$Command, [string]$Log) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & cmd.exe /d /c "call `"$VcVars`" >nul 2>&1 && $Command" 2>&1 | Out-String
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $saved }
    [IO.File]::WriteAllText((Join-Path $BuildDir $Log), $output)
    if ($code) { Write-Host $output; throw "GUI v2 command failed ($code), log $Log" }
    $output -split "`r?`n" | Where-Object { $_ -match 'gui_v2:|Built target|error|warning' } | Write-Host
}
Invoke-Compiler $configure 'configure.log'
$targets = if ($TestHelpers) { 'ecm_gui_v2 ecm_gui_v2_fake_worker ecm_gui_v2_owner_probe ecm_gui_v2_state_test ecm_gui_v2_app_test' } else { 'ecm_gui_v2' }
Invoke-Compiler "cmake --build `"$BuildDir`" --target $targets" 'build.log'
Write-Host "GUI v2 executable: $(Join-Path $BuildDir 'ecm_gui_v2.exe')"
