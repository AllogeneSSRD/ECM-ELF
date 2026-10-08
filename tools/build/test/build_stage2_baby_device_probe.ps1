#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/baby_device_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
New-Item -ItemType Directory -Force $Build | Out-Null
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
$deps=@('tools/bench/stage2_tree_gpu.cu','tools/bench/stage2_baby_device.cuh',
    'tools/test/stage2_baby_device_probe.cu','tools/build/test/build_stage2_baby_device_probe.ps1')
$hashes=[ordered]@{}
foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
# Extract verbatim arithmetic from the actual point implementation. The fixture is
# generated in the ignored build directory and cannot drift from production.
$tree=[IO.File]::ReadAllText((Join-Path $repo $deps[0]))
$lo=$tree.IndexOf('/* (hi, lo) = x*y + z + c, exactly */')
$hi=$tree.IndexOf('/* r = 2p: the reference''s xdbl, in Montgomery images */')
if($lo -lt 0 -or $hi -le $lo){throw 'Arithmetic extraction boundary changed'}
$arithmetic=Join-Path $Build 'stage2_baby_arithmetic_fixture.cuh'
[IO.File]::WriteAllText((Join-Path $repo $arithmetic),$tree.Substring($lo,$hi-$lo))
$exe=Join-Path $Build 'stage2_baby_device_probe.exe'
$log=Join-Path $Build 'build.log'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -I `"$Build`" -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 tools/test/stage2_baby_device_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$timer=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $line
if($LASTEXITCODE -ne 0){Get-Content $log -Tail 35;throw 'Baby device probe build failed'}
$timer.Stop()
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $hashes[$dep]){throw "Source changed during build: $dep"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash $exe -Algorithm SHA256).Hash;
    build_seconds=$timer.Elapsed.TotalSeconds;architecture=$Arch;sources=$hashes;
    arithmetic_sha256=(Get-FileHash $arithmetic -Algorithm SHA256).Hash} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host "built $exe in $($timer.Elapsed.TotalSeconds)s"
