# CPU-only gates for ordered, fresh verification at multiple worker counts.
param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_build_state.ps1')
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path $out){throw 'Use a fresh output directory'}
New-Item -ItemType Directory $out | Out-Null
function Put($Path,$Text){[IO.File]::WriteAllText($Path,$Text)}
function Check($Value,$Name){if(-not $Value){throw "FAIL: $Name"};$script:checks++}
$checks=0
$header=Join-Path $out 'shared.h';Put $header '// original'
$requests=@(0..15 | ForEach-Object {
    $src=Join-Path $out "unit$_.cu";$obj=Join-Path $out "unit$_.obj"
    Put $src '#include "shared.h"';Put $obj "object $_"
    $entry=[pscustomobject]@{file=$src;directory=$out;output=$obj;command="nvcc -c `"$src`" -o `"$obj`""}
    Save-EcmCompileState (Get-EcmCompileState $out $entry)
    [pscustomobject]@{Entry=$entry}
})
$baseline=@($requests | ForEach-Object {Get-EcmCompileState $out $_.Entry})
foreach($width in @(1,4,8)){
    $result=@(Invoke-EcmCompileVerification $out $requests -RequireReusable -VerifyJobs $width)
    for($i=0;$i -lt 16;$i++){
        Check ($result[$i].Index -eq $i -and $result[$i].State.fingerprint -eq $baseline[$i].fingerprint) "ordered identical fingerprint $width/$i"
    }
    $stamp=(Get-Item $header).LastWriteTimeUtc
    Put $header '// changed!';(Get-Item $header).LastWriteTimeUtc=$stamp
    $changed=@(Invoke-EcmCompileVerification $out $requests -VerifyJobs $width)
    Check (@($changed | Where-Object {-not $_.Reason}).Count -eq 0) "same-timestamp content mutation $width"
    $rejected=$false;$continued=$false
    try{Invoke-EcmCompileVerification $out $requests -RequireReusable -VerifyJobs $width | Out-Null;$continued=$true}catch{$rejected=$true}
    Check ($rejected -and -not $continued) "phase barrier rejects mutation $width"
    Put $header '// original'
    $obj=$requests[9].Entry.output;Put $obj 'tampered'
    $changed=@(Invoke-EcmCompileVerification $out $requests -VerifyJobs $width)
    Check ($changed[9].Reason -eq 'object content changed') "object mutation $width"
    Put $obj 'object 9'
    $entry=$requests[3].Entry;$entry.command+=' -DCHANGED=1'
    $changed=@(Invoke-EcmCompileVerification $out $requests -VerifyJobs $width)
    Check ($changed[3].Reason -eq 'compile command or setup changed') "command mutation $width"
    $entry.command=$entry.command.Replace(' -DCHANGED=1','')
    $changed=@(Invoke-EcmCompileVerification $out $requests -Setup 'changed setup' -VerifyJobs $width)
    Check (@($changed | Where-Object {$_.Reason -ne 'compile command or setup changed'}).Count -eq 0) "setup mutation $width"
    $nested=Join-Path $out 'new.h';Put $nested '// added dependency'
    Put $header '#include "new.h"'
    $changed=@(Invoke-EcmCompileVerification $out $requests -VerifyJobs $width)
    Check (@($changed | Where-Object {-not $_.Reason}).Count -eq 0) "new transitive dependency $width"
    Remove-Item -LiteralPath $nested
    $rejected=$false
    try{Invoke-EcmCompileVerification $out $requests -VerifyJobs $width | Out-Null}catch{$rejected=$_.Exception.Message -like '*Unresolved local include*'}
    Check $rejected "deleted transitive dependency $width"
    Put $header '// original'
    $again=@(Invoke-EcmCompileVerification $out $requests -RequireReusable -VerifyJobs $width)
    Check ($again.Count -eq 16) "fresh phase after restoration $width"
    $request=[pscustomobject]@{Entry=$requests[5].Entry;ValidateFingerprint=$true;ExpectedFingerprint=('0'*64)}
    $rejected=$false
    try{Invoke-EcmCompileVerification $out @($request) -ManifestOnly -VerifyJobs $width | Out-Null}catch{$rejected=$_.Exception.Message -like '*compile dependency changed*'}
    Check $rejected "manifest fingerprint mismatch $width"
    $request=[pscustomobject]@{Entry=$requests[5].Entry;ValidateObject=$true;ExpectedObjectHash=('0'*64)}
    $rejected=$false
    try{Invoke-EcmCompileVerification $out @($request) -ManifestOnly -VerifyJobs $width | Out-Null}catch{$rejected=$_.Exception.Message -like '*object changed*'}
    Check $rejected "manifest object mismatch $width"
    Remove-Item -LiteralPath $request.Entry.output
    $request.ExpectedObjectHash=''
    $rejected=$false
    try{Invoke-EcmCompileVerification $out @($request) -ManifestOnly -VerifyJobs $width | Out-Null}catch{$rejected=$_.Exception.Message -like '*object missing*'}
    Check $rejected "missing object cannot match an empty manifest hash $width"
    Put $request.Entry.output 'object 5'
}
[ordered]@{checks=$checks;workers=@(1,4,8);gpu_runs=0}|ConvertTo-Json|Set-Content (Join-Path $out 'result.json')
Write-Host "PASS: $checks parallel verification gates"
