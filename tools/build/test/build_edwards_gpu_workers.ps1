#Requires -Version 5.1
param([ValidateSet('stage1','stage2','both')][string]$Stage='both',
      [string]$Build='build_cuda_cmake/edwards_gpu_workers',
      [string]$Arch='89',
      [string]$Tiers='1024',
      [int]$Jobs=8,
      [ValidateRange(1,64)][int]$SplitCompile=8,
      [switch]$RelinkOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if($Stage -ne 'stage2') {
    Set-Location (Split-Path -Parent (Split-Path -Parent $root))
    . (Join-Path $root 'internal/cuda_build_state.ps1')
    $repo=(Get-Location).Path
    $inventory=Get-EcmCudaInventory $repo
    $edwardsSources=@($inventory.groups.STAGE1_EDWARDS)
    $sourcePaths=@(Get-EcmSourceClosure $repo $edwardsSources @('include','src','third_party/cgbn/include'))+@('CMakeLists.txt')
    $hashes=Get-EcmSourceHashes $repo $sourcePaths
    if($RelinkOnly){
        & (Join-Path $root 'build_stage1_local.ps1') -BuildDir "$Build/stage1" -Arch $Arch -Tiers $Tiers -Jobs $Jobs -RelinkOnly
    } else {
        & (Join-Path $root 'build_stage1_local.ps1') -BuildDir "$Build/stage1" -Arch $Arch -Tiers $Tiers -Jobs $Jobs -Incremental
    }
    if($LASTEXITCODE -ne 0){throw 'Edwards Stage1 worker build failed'}
    foreach($source in $sourcePaths){if($hashes[$source] -ne (Get-FileHash -LiteralPath $source).Hash){throw "source changed during build: $source"}}
    $objects=[ordered]@{}
    $commands=Get-Content "$Build/stage1/compile_commands.json" -Raw|ConvertFrom-Json
    foreach($entry in $commands|Where-Object{(Get-EcmSourceKey $repo $_.file) -in $edwardsSources}){
        $obj=Get-EcmObjectPath $entry (Resolve-EcmSourcePath $repo "$Build/stage1")
        $objects[$entry.file]=[ordered]@{path=$obj;sha256=(Get-FileHash -LiteralPath $obj).Hash}
    }
    [ordered]@{arch="sm_$Arch";tiers=$Tiers;sources=$hashes;objects=$objects;
        exe_sha256=(Get-FileHash "$Build/stage1/ecm_cuda.exe").Hash;
        compile_commands_sha256=(Get-FileHash "$Build/stage1/compile_commands.json").Hash
    }|ConvertTo-Json -Depth 6|Set-Content "$Build/stage1/edwards_build_manifest.json" -Encoding utf8
}
if($Stage -ne 'stage1') {
    & (Join-Path $root 'build_stage2_local.ps1') -Build "$Build/stage2" -Arch "sm_$Arch" -Engine production -SplitCompile $SplitCompile
    if($LASTEXITCODE -ne 0){throw 'Edwards Stage2 worker build failed'}
}
