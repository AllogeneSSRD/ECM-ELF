# Inspect actual parameter ASTs; formal worker inputs keep their stable paths.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$formal = @('build_dev','build_cuda','build_cuda_dev','build_cuda_local','build_cuda_release',
    'build_opencl_local','build_opencl_release','build_cpu_local','build_gui','build_gui_local',
    'build_gui_v2_local','build_stage1_local','build_stage2_local','build_android_local','build_android_release')
$checked = 0
$scriptDirectories = @('tools/build/test','tools/build/dev','tools/test') |
    ForEach-Object { Join-Path $repo $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
$scripts = @(Get-ChildItem -LiteralPath $scriptDirectories -Filter '*.ps1' -File) +
    @(Get-Item (Join-Path $repo 'build_and_run.ps1'))
foreach ($file in $scripts) {
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$tokens,[ref]$errors)
    if ($errors.Count) { throw "Invalid PowerShell syntax: $($file.FullName)" }
    foreach ($parameter in $ast.ParamBlock.Parameters) {
        $value = $parameter.DefaultValue
        if ($value -isnot [Management.Automation.Language.StringConstantExpressionAst] -or
            $value.Value -notmatch '^build(?:_|$)') { continue }
        $root = $value.Value.Split([char[]]'\/')[0]
        if ($root -notin $formal -and $root -notlike 'build_stage1_local_*') {
            throw "Test build default must be inside build_dev: $($file.Name) $($parameter.Name)=$($value.Value)"
        }
        $checked++
    }
}
Write-Host "PASS: $checked actual build-path defaults use formal directories or build_dev"
