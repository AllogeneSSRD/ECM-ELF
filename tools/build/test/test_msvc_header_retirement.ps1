# Reproduce old->new compiler depfiles using the actual formal host toolchain.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path -LiteralPath $out){throw 'Use a fresh output directory'}
$m=Get-Content -LiteralPath (Join-Path $worker 'build_manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $m.host_compiler -or -not $m.compile_setup){throw 'Formal host compiler identity required'}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $out
$fixture=Join-Path $out 'fixture';Initialize-EcmBuildDirectory $fixture
$source=Join-Path $fixture 'main.cpp';$header=Join-Path $fixture 'retired header.h';$obj=Join-Path $fixture 'main.obj';$exe=Join-Path $fixture 'main.exe'
$command='"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MT /c "'+$source+'" /Fo"'+$obj+'" /sourceDependencies "'+$obj+'.ecm-msvc.json"'
$entry=[pscustomobject]@{file=$source;directory=$fixture;command=$command;output=$obj}
function Write-Text([string]$Path,[string]$Text){[IO.File]::WriteAllText($Path,$Text,[Text.UTF8Encoding]::new($false))}
function Compile([string]$Name){
    $batch=Join-Path $out ($Name+'.cmd')
    Write-Text $batch ("@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`nif errorlevel 1 exit /b 1`r`n`"$($m.host_compiler)`" /nologo `"$obj`" /Fe`"$exe`"`r`n")
    & cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $out ($Name+'.log')) -Encoding utf8
    if($LASTEXITCODE){throw "MSVC fixture failed: $Name"}
}
$checks=New-Object 'System.Collections.Generic.List[string]'
function Check([bool]$Yes,[string]$Name){if(-not $Yes){throw "FAIL: $Name"};$checks.Add($Name)}
function Reject([scriptblock]$Action,[string]$Name){
    $errorText='';try{& $Action}catch{$errorText=$_.Exception.Message}
    Check ($errorText -like '*Source changed during build*') $Name
}
Write-Text $header 'constexpr int fixture_value=0;'
Write-Text $source "#include <array>`n#include `"retired header.h`"`nint main(){return fixture_value;}`n"
Compile 'before'
$old=Get-EcmCompileState $fixture $entry $m.compile_setup;Save-EcmCompileState $old
Copy-Item -LiteralPath "$obj.ecm-msvc.json" -Destination (Join-Path $out 'before_dependencies.json')
Write-Text $source "int main(){return 0;}`n"
$before=Get-EcmCompileState $fixture $entry $m.compile_setup
Check (-not (Test-EcmCompileState $before)) 'source edit invalidates the old object'
Compile 'after'
$after=Get-EcmCompileState $fixture $entry $m.compile_setup
Copy-Item -LiteralPath "$obj.ecm-msvc.json" -Destination (Join-Path $out 'after_dependencies.json')
$retired=@($before.source_hashes.Keys|Where-Object {-not $after.source_hashes.Contains($_)})
Check ($retired.Count -gt 1 -and @($retired|Where-Object {$_ -match '/array$'}).Count -eq 1) 'real compiler retires the STL array dependency'
Check ($retired -contains 'retired header.h') 'real compiler retires relative local dependency'
Assert-EcmCompileInputs $before $after
Check $true 'real include removal accepted without false source mutation'
Write-Text $header 'changed after compiler retired this dependency'
Reject {Assert-EcmCompileInputs $before $after} 'retired local header mutation rejected'
Remove-Item -LiteralPath $header
Reject {Assert-EcmCompileInputs $before $after} 'retired local header deletion rejected'
Write-Text $header 'constexpr int fixture_value=0;'
Write-Text $source "int main(){return 1;}`n"
$mutated=Get-EcmCompileState $fixture $entry $m.compile_setup
Reject {Assert-EcmCompileInputs $before $mutated} 'active source mutation rejected'
Write-Text $source "int main(){return 0;}`n"
Save-EcmCompileState $after
Check (Test-EcmCompileState (Get-EcmCompileState $fixture $entry $m.compile_setup)) 'new dependency/object record is reusable'
& $exe
Check ($LASTEXITCODE -eq 0) 'actual executable succeeds'
[ordered]@{complete=$true;checks=$checks.Count;tests=@($checks);compiler=$m.host_compiler;retired_dependencies=$retired;
    actual_cxx_compilations=2;cuda_compilations=0;guard_or_production_cache_modified=$false;
    helper_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'tools/build/internal/cuda_build_state.ps1')).Hash}|
    ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding utf8
Write-Host "PASS: $($checks.Count) actual MSVC dependency-retirement checks"
