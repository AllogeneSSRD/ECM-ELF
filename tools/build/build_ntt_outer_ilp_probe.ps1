#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_outer_ilp_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [ValidateSet(0,1,2,4,8)][int]$UnrollU=0,
      [switch]$IntegratedSchedule)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if($IntegratedSchedule -and $UnrollU -notin @(0,4)){throw 'Integrated schedule supports only 0 or 4'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
$deps=@('tools/test/ntt_coop_outer_probe.cu','tools/bench/ntt_poly_probe.cu',
        'tools/bench/ntt_coop_outer.cuh','tools/bench/ntt_goldilocks_reduce.cuh',
        'tools/bench/ntt_goldilocks_ptx.cuh','tools/build/build_ntt_outer_ilp_probe.ps1')
$original=[ordered]@{};$generated=[ordered]@{}
$encoding=New-Object System.Text.UTF8Encoding($false)
foreach($dep in $deps){
    $original[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash
    if($dep.EndsWith('.ps1')){continue}
    $relative=Join-Path '_src' $dep;$dest=Join-Path $Build $relative
    New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    $raw=[System.IO.File]::ReadAllBytes((Join-Path $repo $dep))
    if($dep -eq 'tools/bench/ntt_coop_outer.cuh' -and $UnrollU -ne 0 -and -not $IntegratedSchedule){
        $text=$encoding.GetString($raw)
        $needle='            for(int u=row;u<d;u+=ROWS) {'
        if(($text.Split(@($needle),[StringSplitOptions]::None).Count-1) -ne 1){throw 'Outer loop extraction differs'}
        $newline=if($text.Contains("`r`n")){"`r`n"}else{"`n"}
        $text=$text.Replace($needle,"#pragma unroll $UnrollU$newline$needle")
        $raw=$encoding.GetBytes($text)
    }
    if($dep -eq 'tools/test/ntt_coop_outer_probe.cu'){
        $text=$encoding.GetString($raw)
        $needle='    CK(cudaSetDevice(device));'
        if(($text.Split(@($needle),[StringSplitOptions]::None).Count-1) -ne 1){throw 'Probe entry extraction differs'}
        $newline=if($text.Contains("`r`n")){"`r`n"}else{"`n"}
        $ack='    std::printf("ntt_outer_variant: unroll_u='+$UnrollU+' device=%d\n",device);'
        if($IntegratedSchedule){$ack+=$newline+'    std::printf("ntt_outer_compiled: unroll_u=%d\n",NTT_OUTER_UNROLL_U);'}
        $raw=$encoding.GetBytes($text.Replace($needle,$needle+$newline+$ack))
    }
    [System.IO.File]::WriteAllBytes($dest,$raw)
    $generated[$relative.Replace('\','/')]=(Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash
}
$exe=Join-Path $Build 'ntt_outer_ilp_probe.exe';$log=Join-Path $Build 'build.log'
$src=Join-Path $Build '_src/tools/test/ntt_coop_outer_probe.cu'
$compiledU=if($IntegratedSchedule){$UnrollU}else{0}
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=3 -DNTT_OUTER_UNROLL_U=$compiledU -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 `"$src`" -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Outer ILP build failed'}
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $original[$dep]){throw "Source changed during build: $dep"}}
foreach($relative in $generated.Keys){if((Get-FileHash -LiteralPath (Join-Path $Build $relative) -Algorithm SHA256).Hash -ne $generated[$relative]){throw "Generated source changed: $relative"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe).Hash;
    architecture=$Arch;gl_fixed_mode=3;unroll_u=$UnrollU;integrated_schedule=[bool]$IntegratedSchedule;
    compiled_outer_u=$compiledU;build_seconds=$watch.Elapsed.TotalSeconds;
    sources=$original;generated_sources=$generated;toolkit=(& nvcc --version | Out-String).Trim()} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built unroll_u={0} ({1:N1}s) {2}" -f $UnrollU,$watch.Elapsed.TotalSeconds,$exe)
