# Each call owns a fresh PS 5.1 runspace pool. Workers only read inputs;
# fingerprints and object bytes are never reused between verification phases.
function Invoke-EcmCompileVerification([string]$Repo, [object[]]$Requests,
        [string]$Setup='', [string[]]$IncludeDirs=@(), [switch]$Production,
        [switch]$RequireReusable, [switch]$ManifestOnly,
        [ValidateRange(1,64)][int]$VerifyJobs=4, [string]$Phase='CUDA verification') {
    if(-not $Requests.Count){return @()}
    $stateFile=Join-Path $PSScriptRoot 'cuda_build_state.ps1'
    $worker={param($StateFile,$Repo,$Request,$Setup,$IncludeDirs,$Production,$RequireReusable,$ManifestOnly,$Index)
        $ErrorActionPreference='Stop'
        try {
            if(-not (Test-Path Function:\Get-EcmCompileState)){. $StateFile}
            $state=Get-EcmCompileState $Repo $Request.Entry $Setup $IncludeDirs -Production:$Production
            $objectHash=if(Test-Path -LiteralPath $state.object -PathType Leaf){(Get-FileHash -LiteralPath $state.object -Algorithm SHA256).Hash}else{''}
            if($ManifestOnly -and -not $objectHash){throw 'Build object missing'}
            $reason=if($ManifestOnly){''}else{Get-EcmCompileInvalidationReason $state -ObjectHash $objectHash}
            if($RequireReusable -and $reason){throw $reason}
            if($Request.ValidateFingerprint -and $state.fingerprint -ne $Request.ExpectedFingerprint){throw 'Build compile dependency changed'}
            if($Request.ValidateObject -and $objectHash -ne $Request.ExpectedObjectHash){throw 'Build object changed'}
            [pscustomobject]@{Index=$Index;State=$state;Reason=$reason;ObjectHash=$objectHash;Error=$null}
        } catch {
            [pscustomobject]@{Index=$Index;Error=$_.Exception.Message}
        }
    }
    $width=[Math]::Min($VerifyJobs,$Requests.Count)
    Write-Host ("== {0}: {1} units, {2} verification workers ==" -f $Phase,$Requests.Count,$width)
    $watch=[Diagnostics.Stopwatch]::StartNew()
    $pool=[RunspaceFactory]::CreateRunspacePool(1,$width);$pool.Open()
    $running=New-Object Collections.ArrayList
    $results=New-Object object[] $Requests.Count
    $next=0;$completed=0
    try {
        while($next -lt $Requests.Count -or $running.Count){
            while($next -lt $Requests.Count -and $running.Count -lt $width){
                $ps=[PowerShell]::Create();$ps.RunspacePool=$pool
                [void]$ps.AddScript($worker.ToString()).AddArgument($stateFile).AddArgument($Repo).AddArgument($Requests[$next]).AddArgument($Setup).AddArgument($IncludeDirs).AddArgument([bool]$Production).AddArgument([bool]$RequireReusable).AddArgument([bool]$ManifestOnly).AddArgument($next)
                [void]$running.Add([pscustomobject]@{PowerShell=$ps;Handle=$ps.BeginInvoke();Index=$next})
                $next++
            }
            foreach($job in @($running)){
                if(-not $job.Handle.IsCompleted){continue}
                $output=@($job.PowerShell.EndInvoke($job.Handle))
                # Dependency-set changes are caught by the identity helper and
                # returned as invalidation reasons, but still set HadErrors.
                # The worker catches unhandled failures in its Error field.
                if($output.Count -ne 1){throw "${Phase}: worker failed: $($job.PowerShell.Streams.Error)"}
                $result=$output[0]
                if($result.Error){throw "${Phase}: $(Get-EcmSourceKey $Repo $Requests[$job.Index].Entry.file): $($result.Error)"}
                $results[$job.Index]=$result
                $job.PowerShell.Dispose();[void]$running.Remove($job);$completed++
                if($completed -eq 1 -or $completed%8 -eq 0 -or $completed -eq $Requests.Count){
                    Write-Host ('   Checked {0}/{1} ({2:N1}s elapsed)' -f $completed,$Requests.Count,$watch.Elapsed.TotalSeconds)
                }
            }
            if($running.Count){Start-Sleep -Milliseconds 20}
        }
    } finally {
        foreach($job in @($running)){
            try{$job.PowerShell.Stop()}finally{$job.PowerShell.Dispose()}
        }
        $pool.Close();$pool.Dispose()
    }
    return $results
}
