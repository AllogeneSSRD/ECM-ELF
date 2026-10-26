# CPU-only real-header, multiple-TU construction/destruction regression.
param([Parameter(Mandatory=$true)][string]$Output,[string]$CudaRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit '60' $CudaRoot -Version '11.8'
$vc=Resolve-EcmVcEnvironment $cuda
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path $out){throw 'Use a fresh evidence directory'}
New-Item -ItemType Directory $out | Out-Null
$headers='#include "'+($repo+'/src/ecm/platform/compute_log.h').Replace('\','/')+'"'+"`n"+
    '#include "'+($repo+'/src/ecm/backend/cuda/runtime/device_memory_ledger.h').Replace('\','/')+'"'+"`n"
foreach($name in @('a','b')){[IO.File]::WriteAllText((Join-Path $out "$name.cpp"),$headers+"int $name(){stage2_memory::Session session;stage2_memory::fixture();return 0;}`n")}
[IO.File]::WriteAllText((Join-Path $out 'main.cpp'),"int a();int b();int main(){return a()+b();}`n")
$line=$vc.Setup+' && cd /d "'+$out+'" && "'+$vc.Compiler+'" /nologo /std:c++17 /EHsc /O2 /MT /I "'+$cuda.Root+'/include" a.cpp b.cpp main.cpp /Fe:ledger_lifecycle.exe'
& cmd.exe /d /c $line
if($LASTEXITCODE){throw 'Ledger lifecycle fixture compilation failed'}
$env:NTT_MEMORY_LEDGER='1'
& (Join-Path $out 'ledger_lifecycle.exe')
$code=$LASTEXITCODE
[ordered]@{complete=($code -eq 0);exit_code=$code;compiler=$vc.Compiler;compiler_sha256=(Get-FileHash $vc.Compiler).Hash;
    header_sha256=(Get-FileHash (Join-Path $repo 'src/ecm/backend/cuda/runtime/device_memory_ledger.h')).Hash} |
    ConvertTo-Json | Set-Content (Join-Path $out 'result.json') -Encoding utf8
if($code){throw "Ledger lifecycle exit failed: $code"}
Write-Host 'PASS: shared ledger real-header multi-TU lifecycle'
