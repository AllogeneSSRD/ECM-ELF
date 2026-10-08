# Shared CUDA architecture/toolkit and supported MSVC selection for build entries.
function Resolve-EcmCudaToolkit([string]$Archs, [string]$CudaRoot = '') {
    $architectures = @($Archs.Split(',') | ForEach-Object { [int]($_ -replace '^sm_', '') })
    $legacy = @($architectures | Where-Object { $_ -lt 75 }).Count -gt 0
    if ($legacy -and @($architectures | Where-Object { $_ -ge 75 }).Count) {
        throw 'Legacy and modern CUDA architectures require separate build directories'
    }
    $version = if ($legacy) { '12.6' } else { '13.3' }
    if (-not $CudaRoot) {
        $CudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH_V' + $version.Replace('.','_'))
        if (-not $CudaRoot) { $CudaRoot = Join-Path $env:ProgramFiles ('NVIDIA GPU Computing Toolkit/CUDA/v' + $version) }
    }
    $CudaRoot = [IO.Path]::GetFullPath($CudaRoot).TrimEnd('\','/')
    $nvcc = Join-Path $CudaRoot 'bin/nvcc.exe'
    if (-not (Test-Path -LiteralPath $nvcc)) { throw "CUDA $version not found: $nvcc; specify the corresponding CUDA root" }
    $versionText = (& $nvcc --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Cannot identify CUDA toolkit: $nvcc" }
    $supported = @(& $nvcc --list-gpu-arch)
    if ($LASTEXITCODE -ne 0) { throw "Cannot list CUDA architectures: $nvcc" }
    foreach ($arch in $architectures) {
        if ("compute_$arch" -notin $supported) { throw "$nvcc does not support sm_$arch" }
    }
    return [pscustomobject]@{ Root = $CudaRoot; Nvcc = $nvcc; VersionText = $versionText; Legacy = $legacy }
}

function Resolve-EcmVcEnvironment($Cuda, [string]$VcVars = '') {
    $hostConfig = Get-Content -LiteralPath (Join-Path $Cuda.Root 'include/crt/host_config.h') -Raw
    $limits = [regex]::Match($hostConfig, '#if\s+_MSC_VER\s*<\s*(\d+)\s*\|\|\s*_MSC_VER\s*>=\s*(\d+)')
    if (-not $limits.Success) { throw "Cannot read supported MSVC versions from $($Cuda.Root)" }
    $minimum = [int]$limits.Groups[1].Value
    $maximum = [int]$limits.Groups[2].Value
    $candidates = if ($VcVars) { @(Get-Item -LiteralPath $VcVars) } else {
        @(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue |
          Sort-Object FullName -Descending)
    }
    foreach ($candidate in $candidates) {
        $vcRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $candidate.FullName))
        $versions = @(Get-ChildItem -LiteralPath (Join-Path $vcRoot 'Tools/MSVC') -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^14\.[0-9]+\.[0-9]+$' } | Sort-Object { [version]$_.Name } -Descending)
        foreach ($tools in $versions) {
            $version = [version]$tools.Name
            $msc = 1900 + $version.Minor
            $compiler = Join-Path $tools.FullName 'bin/Hostx64/x64/cl.exe'
            if ($msc -ge $minimum -and $msc -lt $maximum -and (Test-Path -LiteralPath $compiler)) {
                return [pscustomobject]@{
                    VcVars = $candidate.FullName; Version = $tools.Name; Compiler = $compiler
                    Setup = 'call "' + $candidate.FullName + '" -vcvars_ver=' + $tools.Name + ' >nul 2>&1'
                }
            }
        }
    }
    throw "No installed MSVC toolset supported by $($Cuda.Root) (_MSC_VER=$minimum..$($maximum-1)); install a compatible x64 C++ toolset"
}

function Assert-EcmCmakeToolchain([string]$BuildDir, $Cuda, $Vc) {
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cache)) { return }
    $content = Get-Content -LiteralPath $cache -Raw
    foreach ($entry in @(@('CMAKE_CUDA_COMPILER',$Cuda.Nvcc), @('CMAKE_CXX_COMPILER',$Vc.Compiler))) {
        $match = [regex]::Match($content, '(?m)^' + $entry[0] + ':[^=]+=(.+)\r?$')
        if ($match.Success -and [IO.Path]::GetFullPath($match.Groups[1].Value.Trim()).Replace('\','/') -ine $entry[1].Replace('\','/')) {
            throw "Existing $BuildDir uses another $($entry[0]); choose a fresh build directory for this toolkit"
        }
    }
}
