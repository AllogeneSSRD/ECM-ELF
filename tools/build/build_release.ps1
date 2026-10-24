#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('all','cu11.8','cu13.3')][string]$Profile='all',
    [ValidateSet('all','stage1','stage2')][string]$Component='all',
    [ValidateSet('grouped','isolated')][string]$Stage2KernelLayout='grouped',
    [string]$BuildRoot='build_cuda_release', [string]$OutDir='release',
    [ValidateRange(1,64)][int]$Jobs=16,
    [string]$Gmp='', [string]$Cuda118Root='', [string]$Cuda133Root='', [string]$VcVars='',
    [string]$OpenSslRoot='', [string]$OpenClRoot='',
    [switch]$SkipBuild, [switch]$SkipPackage, [switch]$ReinspectCuda
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'internal/release_profiles.ps1')
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
. (Join-Path $PSScriptRoot 'internal/stage1_release_sources.ps1')
. (Join-Path $PSScriptRoot 'internal/cuda_build_profile.ps1')
$dependency=Resolve-EcmGmp $repo $Gmp -Release
foreach($name in @('BuildRoot','OutDir')) {
    $path=Get-Variable $name -ValueOnly
    if(-not [IO.Path]::IsPathRooted($path)){$path=Join-Path $repo $path}
    Set-Variable $name ([IO.Path]::GetFullPath($path))
}
Assert-EcmSeparatePackageDirectory $BuildRoot $OutDir
$names=if($Profile -eq 'all'){@('cu11.8','cu13.3')}else{@($Profile)}
# Preflight every requested toolchain before constructing any component.
$toolchains=@{}
foreach($name in $names) {
    $target=Get-EcmReleaseTarget $name $Stage2KernelLayout
    $root=if($name -eq 'cu11.8'){$Cuda118Root}else{$Cuda133Root}
    $toolchains[$name]=Resolve-EcmReleaseToolchain $target $root $VcVars
}
& python -c "import sys;sys.path.insert(0,sys.argv[1]);from gmp_payload import distribution;distribution(sys.argv[2])" (Join-Path $repo 'tools/packaging') $dependency.Root
if($LASTEXITCODE){throw 'vcpkg GMP source/provenance preflight failed'}
$records=Join-Path $BuildRoot 'components'
New-Item -ItemType Directory -Force $records | Out-Null
$gui=Join-Path $BuildRoot 'gui-v2-release'
$opencl=Join-Path $BuildRoot 'opencl-release'
if($Component -eq 'all'){
    $openclArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'build_opencl_local.ps1'),
        '-BuildRole','release','-BuildDir',$opencl,'-Gmp',$dependency.Root)
    foreach($option in @('OpenSslRoot','OpenClRoot','VcVars')){
        $value=Get-Variable $option -ValueOnly
        if($value){$openclArgs+=@('-'+$option,$value)}
    }
    if($SkipBuild){$openclArgs+='-SkipBuild'}
    & powershell @openclArgs
    if($LASTEXITCODE){throw 'Release OpenCL worker build/verification failed'}
}
if(-not $SkipBuild -and $Component -eq 'all') {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'test/build_gui_v2.ps1') `
        -BuildDir $gui -GmpRoot $dependency.Root -BasePetOnly
    if($LASTEXITCODE){throw 'Release GUI v2 build failed'}
}
foreach($name in $names) {
    $tools=$toolchains[$name];$target=$tools.Target
    $stage1Record=Join-Path $records "$name.stage1.json"
    $stage2Record=Join-Path $records "$name.stage2.json"
    if(-not $SkipBuild) {
        if($Component -in @('all','stage1')) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'internal/build_stage1_release_group.ps1') `
                -BuildRoot $BuildRoot -OutDir $OutDir -Archs $target.Sass -PtxArch $target.Ptx -Jobs $Jobs `
                -CudaRoot $tools.Cuda.Root -VcVars $tools.Vc.VcVars -Gmp $dependency.Root -SkipSplit -ResultFile $stage1Record
            if($LASTEXITCODE){throw "Release Stage1 failed: $name"}
        }
        if($Component -in @('all','stage2')) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'build_stage2_local.ps1') `
                -BuildRoot $BuildRoot -BuildRole release -Arch ('sm_'+$target.Sass) -PtxArch $target.Ptx `
                -KernelLayout $Stage2KernelLayout `
                -SplitCompile $target.SplitCompile -Jobs $Jobs -CudaRoot $tools.Cuda.Root -VcVars $tools.Vc.VcVars `
                -Gmp $dependency.Root -ResultFile $stage2Record
            if($LASTEXITCODE){throw "Release Stage2 failed: $name"}
        }
    }
    foreach($stage in @('stage1','stage2')) {
        if($Component -ne 'all' -and $stage -ne $Component){continue}
        Write-Host "== Verify $name/${stage}: sources, compiler, GMP, objects and executable =="
        $record=Get-Content -LiteralPath (Join-Path $records "$name.$stage.json") -Raw | ConvertFrom-Json
        $manifest=Get-Content -LiteralPath $record.manifest -Raw | ConvertFrom-Json
        $profileIdentity=Get-Content -LiteralPath (Join-Path $record.build 'profile.json') -Raw | ConvertFrom-Json
        if($manifest.role -ne 'release' -or $manifest.cuda_root -ine $tools.Cuda.Root -or
            $manifest.toolkit -ne $tools.Cuda.VersionText -or $manifest.ptx_architecture -ne $target.Ptx -or
            $profileIdentity.profile_sha256 -ne $manifest.profile_sha256 -or
            (Get-EcmProfileHash $profileIdentity.fields) -ne $manifest.profile_sha256) {throw "Release component/profile identity mismatch: $name/$stage"}
        $fields=$profileIdentity.fields
        if($fields.stage -ne $stage -or $fields.role -ne 'release' -or $fields.engine -ne 'production' -or
            $fields.cuda_root -ine $tools.Cuda.Root -or $fields.toolkit -ne $tools.Cuda.VersionText.Replace("`r`n","`n") -or
            $fields.architectures -ne $target.Sass -or
            $fields.host_compiler -ine $tools.Vc.Compiler -or $fields.host_version -ne $tools.Vc.Version -or
            $manifest.host_compiler -ine $tools.Vc.Compiler -or $manifest.host_version -ne $tools.Vc.Version -or
            $fields.nvcc_sha256 -ne (Get-FileHash -LiteralPath $tools.Cuda.Nvcc).Hash -or
            $fields.host_compiler_sha256 -ne (Get-FileHash -LiteralPath $tools.Vc.Compiler).Hash -or
            $fields.gmp_root -ine $dependency.Root) {throw "Release compiler/GMP identity mismatch: $name/$stage"}
        foreach($file in $dependency.Hashes.Keys) {
            if($fields.('gmp/'+$file) -ne $dependency.Hashes[$file]) {throw "Release GMP input mismatch: $name/$stage/$file"}
        }
        if($stage -eq 'stage1') {
            if($manifest.architectures -ne $target.Sass -or $manifest.tiers -or $manifest.enable_prac){throw 'Complete release requires full production tiers, PRAC disabled'}
            Assert-Stage1BuildIdentity $repo $record.build $manifest
        } else {
            if($manifest.architecture -ne ('sm_'+$target.Sass) -or $manifest.engine -ne 'production' -or
                $manifest.gl_fixed_mode -ne 3 -or $manifest.add_sub_mask -ne 1 -or $manifest.outer_unroll_u -ne 0 -or
                $manifest.kernel_layout -ne $Stage2KernelLayout -or $fields.'option/kernel_layout' -ne $Stage2KernelLayout -or
                [bool]$manifest.module_directory -ne ($Stage2KernelLayout -eq 'isolated') -or
                $manifest.split_compile -ne $target.SplitCompile -or
                $fields.'option/split_compile' -ne [string]$target.SplitCompile -or
                $manifest.sha256 -ne (Get-FileHash -LiteralPath $record.exe).Hash){throw 'Stage2 production identity mismatch'}
            $inventory=Get-EcmCudaInventory $repo (Join-Path $record.build 'cuda_sources.json')
            $includeDirs=@((Join-Path $dependency.Root 'include'),'include','src','src/ecm/backend/cuda')
            $set=Get-EcmStage2SourceSet $repo $inventory production $includeDirs $manifest.module_directory
            Assert-EcmHashes $manifest.source_hashes (Get-EcmSourceHashes $repo $set.Dependencies) 'Stage2 source identity changed'
            Assert-EcmBuildUnits $repo $record.build $manifest $set.Sources $includeDirs -Production
        }
        if((Get-FileHash -LiteralPath (Join-Path $record.build 'gmp-10.dll')).Hash -ne $dependency.Hashes['bin/gmp-10.dll']){throw 'Release runtime GMP mismatch'}
        Write-Host "Verified $name/$stage"
    }
}
if($Component -ne 'all' -or $SkipPackage){Write-Host "Release components verified: $Component, $($names -join ','); packaging skipped";exit 0}
$plan=Join-Path $records 'package-plan.json'
[ordered]@{profiles=@($names);build_root=$BuildRoot;out=$OutDir;gui=$gui;opencl=$opencl;gmp=$dependency.Root;stage2_kernel_layout=$Stage2KernelLayout;reinspect_cuda=[bool]$ReinspectCuda} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $plan -Encoding UTF8
& python (Join-Path $repo 'tools/packaging/package_release.py') --plan $plan
if($LASTEXITCODE){throw 'Complete release packaging failed'}
Write-Host "Release complete: $($names -join ',') -> $OutDir (runtime qualification is recorded separately)"
