function Build-EcmStage2CpuHost([string]$Build, [string]$Gmp, $Vc, [int]$Jobs, [switch]$Rebuild) {
    $dir = Join-Path $Build 'cpu-host'
    Initialize-EcmBuildDirectory $dir
    $project = Join-Path $PSScriptRoot 'stage2_cpu'
    $batch = Join-Path $dir 'build.cmd'
    $configure = "cmake -S `"$project`" -B `"$dir`" -G Ninja -DCMAKE_BUILD_TYPE=Release `"-DECM_GMP_ROOT=$($Gmp.Replace('\','/'))`""
    $compile = "cmake --build `"$dir`" --target ecm_stage2_cpu_host --parallel $Jobs"
    if ($Rebuild) { $compile += ' --clean-first' }
    [IO.File]::WriteAllText($batch, "@echo off`r`n$($Vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$configure`r`nif errorlevel 1 exit /b 1`r`n$compile`r`n", [Text.UTF8Encoding]::new($false))
    $log = Join-Path $dir 'build.log'
    & cmd.exe /d /c $batch 2>&1 | Out-File -LiteralPath $log -Encoding utf8
    if ($LASTEXITCODE) { Get-Content $log -Tail 40 | Write-Host; throw 'Stage2 CPU host build failed' }
    return @('ecm_stage2_cpu_host.lib','ecm_cpu_library.lib' | ForEach-Object { Join-Path $dir $_ })
}
