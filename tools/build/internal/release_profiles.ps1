. (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')

function Get-EcmReleaseTarget([ValidateSet('cu11.8','cu13.3')][string]$Name,
        [ValidateSet('grouped','isolated')][string]$Stage2KernelLayout = 'grouped') {
    if ($Name -eq 'cu11.8') { return [pscustomobject]@{ Name = $Name; Version = '11.8'; Ptx = '50'; Sass = '60'; SplitCompile = 1 } }
    return [pscustomobject]@{ Name = $Name; Version = '13.3'; Ptx = '75'; Sass = '80'; SplitCompile = $(if ($Stage2KernelLayout -eq 'isolated') { 1 } else { 8 }) }
}

function Resolve-EcmReleaseToolchain($Target, [string]$CudaRoot = '', [string]$VcVars = '') {
    $cuda = Resolve-EcmCudaToolkit -Archs $Target.Sass -CudaRoot $CudaRoot -Version $Target.Version
    $virtual = @(& $cuda.Nvcc --list-gpu-arch)
    $real = @(& $cuda.Nvcc --list-gpu-code)
    if ("compute_$($Target.Ptx)" -notin $virtual -or "sm_$($Target.Sass)" -notin $real) { throw "Toolkit lacks release targets: $($Target.Name)" }
    return [pscustomobject]@{ Target = $Target; Cuda = $cuda; Vc = (Resolve-EcmVcEnvironment $cuda $VcVars) }
}
