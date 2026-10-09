<# CPU/CUDA compile-only regression for compiler subprocess output capture.
   Does not query a GPU, link a runtime program or execute a CUDA kernel. #>
param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit 'sm_89'
$vc=Resolve-EcmVcEnvironment $cuda
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path -LiteralPath $out){throw 'Use a fresh output directory'}
New-Item -ItemType Directory -Path $out | Out-Null
foreach($kind in @('cpp','cu')) {
    $source=Join-Path $out "tiny.$kind"
    $object=Join-Path $out "tiny_$kind.obj"
    $command=Join-Path $out "compile_$kind.cmd"
    $log=Join-Path $out "compile_$kind.log"
    $text=if($kind -eq 'cpp'){'int main(){return 0;}'}else{'__global__ void capture_test(){}'}
    [IO.File]::WriteAllText($source,$text,[Text.UTF8Encoding]::new($false))
    $compile="`"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -arch=sm_89 -Xcompiler /utf-8 -c `"$source`" -o `"$object`""
    [IO.File]::WriteAllText($command,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$compile`r`n",[Text.UTF8Encoding]::new($false))
    & cmd.exe /d /c $command 2>&1 | Out-File -LiteralPath $log -Encoding utf8
    if($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $object)) {
        Get-Content -LiteralPath $log
        throw "Compiler capture failed for $kind"
    }
}
Write-Host 'PASS: piped NVCC/MSVC output capture, C++ and CUDA compile-only'
