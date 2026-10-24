function Invoke-EcmReleaseLogged([string]$Repo,[string]$Script,[Collections.IDictionary]$Parameters,
        [string]$Profile,[string]$OutDir) {
    $versionText=[IO.File]::ReadAllText((Join-Path $Repo 'include/ecm/platform/program_version.h'))
    $version=[regex]::Match($versionText,'version\[\]\s*=\s*"([0-9.]+)"').Groups[1].Value
    if(-not $version){throw 'Cannot identify release version for log filename'}
    if(-not [IO.Path]::IsPathRooted($OutDir)){$OutDir=Join-Path $Repo $OutDir}
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $log=Join-Path $OutDir "release-v$version-$Profile.log"
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$Script,'-LogWorker')
    foreach($name in $Parameters.Keys){
        if($name -eq 'LogWorker'){continue}
        $value=$Parameters[$name]
        if($value -is [Management.Automation.SwitchParameter]){if($value){$arguments+='-'+$name}}
        elseif([string]$value -ne ''){$arguments+=@(('-'+$name),[string]$value)}
    }
    Write-Host "Release log: $log"
    $writer=[IO.StreamWriter]::new($log,$false,[Text.UTF8Encoding]::new($true));$writer.AutoFlush=$true
    $savedPreference=$ErrorActionPreference;$code=1
    try {
        $writer.WriteLine(('Release v{0} / {1} started {2:o}' -f $version,$Profile,[DateTimeOffset]::Now))
        $ErrorActionPreference='Continue'
        & powershell @arguments 2>&1 | ForEach-Object {
            $line=$_.ToString();$writer.WriteLine($line);Write-Host $line
        }
        $code=$LASTEXITCODE
    } finally {
        $ErrorActionPreference=$savedPreference
        $writer.WriteLine(('Release exit code: {0}; finished {1:o}' -f $code,[DateTimeOffset]::Now))
        $writer.Dispose()
    }
    return $code
}
