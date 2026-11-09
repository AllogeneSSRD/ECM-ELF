param([string]$Build='build_dev/stage1_ntt_auto_probe', [string]$ModuleBuild='',
      [string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -ModuleBuild $ModuleBuild `
    -Source 'tools/test/stage1_ntt_auto_probe.cpp' -Name 'stage1_ntt_auto_probe' `
    -ExtraSources @('src/ecm/stage1/montgomery/cuda/ntt/ladder.cu', `
                    'src/ecm/stage1/montgomery/cuda/ntt/tune.cpp', 'src/io/files.cpp', `
                    'src/ecm/platform/logging.cpp', 'src/ecm/platform/stop.cpp', `
                    'src/ecm/application/configuration/runtime.cpp')
