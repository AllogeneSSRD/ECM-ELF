#Requires -Version 5.1
<#
.SYNOPSIS
    gmp-ecm GPU work queue manager -- Windows port of gpu/work_manager.sh

.DESCRIPTION
    Runs the task lines of a worktodo file one at a time. After every task:
      1. the finished line is appended to worktodo.finished.txt;
      2. that line is removed atomically from the worktodo file (the file is
         re-read every round, so new lines can be appended while it runs);
      3. all *.save files are synced to the two sync folders.

    Task line format is identical to the Linux version, only the executable
    path changes (./ecm  ->  .\ecm_cuda.exe):

        echo '<N expression>' | .\ecm_cuda.exe -v -savea m8237_110e6.save -gpu --ckpt 300 -gpucurves 384 110e6 0

    Pasting Linux style lines is fine as well: when the executable token of a
    line cannot be resolved on this machine (./ecm, ecm.exe, ...) the local
    executable (-ExePath) is used instead and a NOTE is written to the log.
    The N expression is written to a temp file and fed on stdin, so shell
    quoting of ^ ( ) * | in the expression never reaches cmd.exe.

.EXAMPLE
    .\work_manager.bat

.EXAMPLE
    .\work_manager.bat --run-log --sync-mode full

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\work_manager.ps1 -RunLog -SyncMode incremental
#>

[CmdletBinding()]
param(
    # Work list file (positional argument). Relative paths resolve against this folder.
    [Parameter(Position = 0)]
    [string]$WorkFile = 'worktodo.txt',

    # File the finished task lines are appended to (never truncated).
    [string]$FinishedFile = 'worktodo.finished.txt',

    # ECM executable. Lines naming an executable that does not exist here are
    # mapped to this one -- that is how './ecm' becomes '.\ecm_cuda.exe'.
    [string]$ExePath = '.\ecm_cuda.exe',

    # Do not add -v. By default every task is run with -v so the log always has
    # the detailed driver output; a line that already carries -v keeps it once.
    [switch]$NoVerbose,

    # .save sync target 1 (empty = built-in default, see $DefaultSaveSyncDir).
    [string]$SaveSyncDir = '',

    # .save sync target 2 (empty = built-in default, see $DefaultSaveP95Dir).
    [string]$SaveP95Dir = '',

    # full = copy every *.save ; incremental = only *.save written by this task.
    [ValidateSet('full', 'incremental')]
    [string]$SyncMode = 'incremental',

    # Write one log file per task into .\log (default: off).
    [switch]$RunLog,

    # How often (ms) new task output is flushed to the console and the log files.
    [int]$PollMs = 1000,

    # Show help and exit.
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------- setup
$Root = $PSScriptRoot
if ([string]::IsNullOrEmpty($Root)) { $Root = (Get-Location).Path }
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location -LiteralPath $Root

# The second sync folder name contains CJK characters; they are assembled from code
# points so this file stays pure ASCII (Windows PowerShell 5.1 reads a BOM-less
# script with the ANSI code page, which would mangle non-ASCII literals).
$DefaultSaveSyncDir = 'D:\code\GIMPS\GIMPS_' + [char]0x540C + [char]0x6B65 + '\ECM'
$DefaultSaveP95Dir  = 'D:\code\GIMPS\p95v3104'
if ([string]::IsNullOrEmpty($SaveSyncDir)) { $SaveSyncDir = $DefaultSaveSyncDir }
if ([string]::IsNullOrEmpty($SaveP95Dir))  { $SaveP95Dir  = $DefaultSaveP95Dir }

$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$MainLogPath = Join-Path $Root 'screen.log'
$LogDirPath  = Join-Path $Root 'log'
$Script:RunLogPath = $null

function Resolve-LocalPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

$WorkFileFull = Resolve-LocalPath $WorkFile
$FinishedFull = Resolve-LocalPath $FinishedFile
$ExeFull      = Resolve-LocalPath ($ExePath -replace '/', '\')

# --------------------------------------------------------------------------- logging
function Write-Log {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray,
        [switch]$NoTimestamp
    )
    if ($NoTimestamp) { $line = $Message }
    else { $line = '[' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '] ' + $Message }
    Write-Host $line -ForegroundColor $Color
    $payload = $line + "`r`n"
    try { [System.IO.File]::AppendAllText($MainLogPath, $payload, $Utf8NoBom) } catch { }
    if ($Script:RunLogPath) {
        try { [System.IO.File]::AppendAllText($Script:RunLogPath, $payload, $Utf8NoBom) } catch { }
    }
}

function Show-Usage {
    @'
work_manager.ps1 - gmp-ecm GPU work queue manager (Windows port of work_manager.sh)

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File work_manager.ps1 [options] [WORK_FILE]
  work_manager.bat [options] [WORK_FILE]

Options:
  -WorkFile <file>       work list file, default worktodo.txt (positional)
  -FinishedFile <file>   finished lines file, default worktodo.finished.txt
  -ExePath <path>        ECM executable, default .\ecm_cuda.exe
  -NoVerbose             do not add -v (by default every task runs with -v)
  -SaveSyncDir <dir>     .save sync target 1, default D:\code\GIMPS\GIMPS_<cjk>\ECM
  -SaveP95Dir <dir>      .save sync target 2, default D:\code\GIMPS\p95v3104
  -SyncMode <mode>       full | incremental, default incremental
  -RunLog                write one log file per task into .\log (default off)
  -PollMs <ms>           output flush interval, default 1000
  -Help                  show this help

work_manager.bat also accepts the old shell style options:
  --run-log              same as -RunLog
  --sync-mode <mode>     same as -SyncMode
  --no-verbose           same as -NoVerbose
  --pause                wait for a key press before the window closes
  -h, --help             same as -Help

Task line format (same as the Linux version, executable path updated):
  echo '<N expression>' | .\ecm_cuda.exe -v -savea m8237_110e6.save -gpu --ckpt 300 -gpucurves 384 110e6 0

-v is added automatically when the line does not have it yet.

Lines starting with '#' and empty lines are ignored. Finished lines are appended
to the finished file and removed from the work file. Log file: screen.log.
'@ | Write-Host
    return
}

# --------------------------------------------------------------------------- file helpers
function Read-AllLinesSafe {
    param([Parameter(Mandatory = $true)][string]$Path)
    $lastError = $null
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            $lines = [System.IO.File]::ReadAllLines($Path, [System.Text.Encoding]::UTF8)
            # ',': keep an empty result an empty array instead of unrolling it to $null
            return , $lines
        }
        catch {
            $lastError = $_.Exception.Message
            Start-Sleep -Milliseconds 200
        }
    }
    Write-Log -Message ('WARN : cannot read ' + $Path + ': ' + $lastError) -Color Yellow
    return $null
}

function Get-FirstTaskLine {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $lines = Read-AllLinesSafe -Path $Path
    if ($null -eq $lines) { return $null }
    foreach ($raw in $lines) {
        $t = $raw.Trim()
        if ($t.Length -eq 0) { continue }
        if ($t.StartsWith('#')) { continue }
        return $t
    }
    return $null
}

function Remove-FirstTaskLine {
    param([Parameter(Mandatory = $true)][string]$Path)
    $lines = Read-AllLinesSafe -Path $Path
    if ($null -eq $lines) { return }
    $kept = [System.Collections.Generic.List[string]]::new()
    $removed = $false
    foreach ($raw in $lines) {
        $t = $raw.Trim()
        if (-not $removed -and $t.Length -gt 0 -and -not $t.StartsWith('#')) {
            $removed = $true
            continue
        }
        $kept.Add($raw)
    }
    if (-not $removed) { return }
    $tmp = $Path + '.tmp'
    [System.IO.File]::WriteAllLines($tmp, $kept.ToArray(), $Utf8NoBom)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

# --------------------------------------------------------------------------- task line parsing
# Splits a command tail into arguments, honouring single and double quotes.
function ConvertTo-ArgumentList {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text)
    $list = [System.Collections.Generic.List[string]]::new()
    $sb = [System.Text.StringBuilder]::new()
    $inQuote = $false
    $quoteChar = [char]0
    $started = $false
    for ($i = 0; $i -lt $Text.Length; $i++) {
        $ch = $Text[$i]
        if ($inQuote) {
            if ($ch -eq $quoteChar) { $inQuote = $false } else { [void]$sb.Append($ch) }
        }
        elseif ($ch -eq [char]39 -or $ch -eq [char]34) {
            $inQuote = $true
            $quoteChar = $ch
            $started = $true
        }
        elseif ([char]::IsWhiteSpace($ch)) {
            if ($started) {
                $list.Add($sb.ToString())
                [void]$sb.Clear()
                $started = $false
            }
        }
        else {
            [void]$sb.Append($ch)
            $started = $true
        }
    }
    if ($started) { $list.Add($sb.ToString()) }
    return $list.ToArray()
}

# True when the token looks like the executable of the line (not a flag).
function Test-ExeToken {
    param([AllowEmptyString()][string]$Token)
    if ([string]::IsNullOrWhiteSpace($Token)) { return $false }
    if ($Token.StartsWith('-')) { return $false }
    if ($Token -match '(?i)\.exe$') { return $true }
    if ($Token -match '[\\/]') { return $true }
    if ($Token -match '(?i)^ecm$') { return $true }
    return $false
}

# Parses one worktodo line into the N expression, the executable and its arguments.
function Parse-WorkLine {
    param([Parameter(Mandatory = $true)][string]$Line)

    $text = $Line.Trim()
    $nExpr = $null
    $cmdText = $null

    # Legacy shell form: echo '<N>' | <exe> <args...>
    $m = [regex]::Match($text, '^\s*echo\s+(?<q>[''"])(?<n>.*?)\k<q>\s*\|\s*(?<cmd>.*)$')
    if ($m.Success) {
        $nExpr = $m.Groups['n'].Value
        $cmdText = $m.Groups['cmd'].Value
    }
    else {
        # Plain form: <N> | <exe> <args...>
        $idx = $text.IndexOf('|')
        if ($idx -lt 0) { throw ('cannot parse task line (no | separator): ' + $Line) }
        $nExpr = $text.Substring(0, $idx).Trim()
        if ($nExpr.Length -ge 2) {
            $first = $nExpr[0]
            $last = $nExpr[$nExpr.Length - 1]
            if (($first -eq [char]39 -and $last -eq [char]39) -or ($first -eq [char]34 -and $last -eq [char]34)) {
                $nExpr = $nExpr.Substring(1, $nExpr.Length - 2)
            }
        }
        $cmdText = $text.Substring($idx + 1).Trim()
    }

    if ([string]::IsNullOrWhiteSpace($nExpr)) { throw ('task line has no N expression: ' + $Line) }

    $tokens = @(ConvertTo-ArgumentList -Text $cmdText)
    if ($tokens.Count -eq 0) { throw ('task line has no ECM arguments: ' + $Line) }

    $exeToken = $null
    $argTokens = $tokens
    if (Test-ExeToken -Token $tokens[0]) {
        $exeToken = $tokens[0]
        if ($tokens.Count -gt 1) { $argTokens = $tokens[1..($tokens.Count - 1)] } else { $argTokens = @() }
    }

    # -v is on by default: prepend it unless the line already asks for it.
    $verboseAdded = $false
    if (-not $NoVerbose) {
        $hasVerbose = $false
        foreach ($a in $argTokens) {
            if ($a -eq '-v') { $hasVerbose = $true; break }
        }
        if (-not $hasVerbose) {
            $argTokens = @('-v') + @($argTokens)
            $verboseAdded = $true
        }
    }

    $exe = $ExeFull
    $exeDisplay = $ExePath
    $modified = $true
    if (-not [string]::IsNullOrWhiteSpace($exeToken)) {
        $candidate = Resolve-LocalPath ($exeToken -replace '/', '\')
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $exe = $candidate
            $exeDisplay = $exeToken
            $modified = $false
        }
    }
    if ($verboseAdded) { $modified = $true }

    return [pscustomobject]@{
        N          = $nExpr
        Exe        = $exe
        ExeDisplay = $exeDisplay
        Args       = @($argTokens)
        Modified   = $modified
    }
}

# --------------------------------------------------------------------------- task execution
# Reads the growing output file and writes every complete new line to the log.
function Write-NewTaskOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$Offset,
        [switch]$Flush
    )
    $text = $null
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $fs = $null
        $sr = $null
        try {
            $fs = [System.IO.FileStream]::new($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $sr = [System.IO.StreamReader]::new($fs, [System.Text.Encoding]::UTF8)
            $text = $sr.ReadToEnd()
        }
        catch { $text = $null }
        finally {
            if ($sr) { $sr.Dispose() } elseif ($fs) { $fs.Dispose() }
        }
    }
    if ($null -eq $text) { return $Offset }
    if ($text.Length -le $Offset) { return $Offset }

    $chunk = $text.Substring($Offset)
    $cut = $chunk.LastIndexOf("`n")
    if ($cut -ge 0) {
        $complete = $chunk.Substring(0, $cut)
        $newOffset = $Offset + $cut + 1
    }
    elseif ($Flush) {
        $complete = $chunk
        $newOffset = $text.Length
    }
    else {
        return $Offset
    }

    if ($complete.Length -gt 0) {
        foreach ($l in ($complete -split "`n")) {
            $t = $l.TrimEnd("`r")
            if ($t.Length -eq 0) { continue }
            # the ECM driver already stamps its own lines: do not duplicate the stamp
            if ($t -match '^\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\]') { Write-Log -Message $t -NoTimestamp }
            else { Write-Log -Message $t }
        }
    }
    return $newOffset
}

# Runs one task: stdin from a temp file, stdout+stderr to a temp file which is
# tailed live. Returns the exit code of the ECM process.
function Invoke-EcmTask {
    param(
        [Parameter(Mandatory = $true)]$Task
    )
    $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ('ecm_wm_' + [Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Path $tmpDir -Force)
    $proc = $null
    try {
        $stdinPath  = Join-Path $tmpDir 'stdin.txt'
        $stdoutPath = Join-Path $tmpDir 'stdout.txt'
        $batPath    = Join-Path $tmpDir 'run.bat'

        [System.IO.File]::WriteAllText($stdinPath, $Task.N, [System.Text.Encoding]::ASCII)

        $argText = ''
        foreach ($a in $Task.Args) { $argText += ' "' + $a + '"' }
        $batLines = @(
            '@echo off',
            ('"' + $Task.Exe + '"' + $argText + ' < "' + $stdinPath + '" > "' + $stdoutPath + '" 2>&1'),
            'exit /b %ERRORLEVEL%'
        )
        [System.IO.File]::WriteAllText($batPath, (($batLines -join "`r`n") + "`r`n"), [System.Text.Encoding]::ASCII)

        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $env:ComSpec
        if ([string]::IsNullOrEmpty($psi.FileName)) { $psi.FileName = 'cmd.exe' }
        $psi.Arguments = '/d /c "' + $batPath + '"'
        $psi.WorkingDirectory = $Root
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardInput = $false
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError = $false

        $proc = [System.Diagnostics.Process]::Start($psi)
        $offset = 0
        while (-not $proc.HasExited) {
            $offset = Write-NewTaskOutput -Path $stdoutPath -Offset $offset
            Start-Sleep -Milliseconds $PollMs
        }
        $proc.WaitForExit()
        [void](Write-NewTaskOutput -Path $stdoutPath -Offset $offset -Flush)
        return $proc.ExitCode
    }
    finally {
        if ($proc) {
            if (-not $proc.HasExited) { try { $proc.Kill() } catch { } }
            $proc.Dispose()
        }
        Remove-Item -LiteralPath $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# --------------------------------------------------------------------------- .save sync
function Sync-SaveFiles {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('full', 'incremental')][string]$Mode,
        [datetime]$Since
    )

    foreach ($dir in @($SaveSyncDir, $SaveP95Dir)) {
        if (-not (Test-Path -LiteralPath $dir)) {
            try { [void](New-Item -ItemType Directory -Path $dir -Force) }
            catch { Write-Log -Message ('WARN : cannot create sync folder ' + $dir + ': ' + $_.Exception.Message) -Color Yellow }
        }
    }

    $files = @(Get-ChildItem -LiteralPath $Root -Filter '*.save' -File -ErrorAction SilentlyContinue)
    if ($Mode -eq 'incremental') {
        if (-not $PSBoundParameters.ContainsKey('Since')) {
            Write-Log -Message 'SYNC : incremental mode requires a valid marker file' -Color Yellow
            return
        }
        $files = @($files | Where-Object { $_.LastWriteTime -gt $Since })
    }

    if ($Mode -eq 'full' -and $files.Count -eq 0) {
        Write-Log -Message 'SYNC : no .save files found'
        return
    }

    $names = [System.Collections.Generic.List[string]]::new()
    foreach ($f in $files) {
        foreach ($dir in @($SaveSyncDir, $SaveP95Dir)) {
            try { Copy-Item -LiteralPath $f.FullName -Destination $dir -Force -ErrorAction Stop }
            catch { Write-Log -Message ('WARN : copy failed ' + $f.Name + ' -> ' + $dir + ': ' + $_.Exception.Message) -Color Yellow }
        }
        $names.Add($f.Name)
    }

    Write-Log -Message ('SYNC : mode=' + $Mode + ', copied ' + $files.Count + ' .save file(s) -> ' + $SaveSyncDir + ', ' + $SaveP95Dir + ';') -Color Cyan
    if ($names.Count -gt 0) { Write-Log -Message ('files=' + ($names.ToArray() -join ',')) }
}

# --------------------------------------------------------------------------- main
if ($Help) { Show-Usage; exit 0 }

if (-not (Test-Path -LiteralPath $LogDirPath)) { [void](New-Item -ItemType Directory -Path $LogDirPath -Force) }

# Make sure the finished file exists (existing content is kept).
$finishedDir = Split-Path -Parent $FinishedFull
if (-not [string]::IsNullOrEmpty($finishedDir) -and -not (Test-Path -LiteralPath $finishedDir)) {
    [void](New-Item -ItemType Directory -Path $finishedDir -Force)
}
if (-not (Test-Path -LiteralPath $FinishedFull)) { [System.IO.File]::WriteAllText($FinishedFull, '', $Utf8NoBom) }

if (-not (Test-Path -LiteralPath $ExeFull -PathType Leaf)) {
    Write-Log -Message ('ERROR: ECM executable not found: ' + $ExeFull) -Color Red
    exit 1
}

$runLogEnabled = 0
if ($RunLog) { $runLogEnabled = 1 }

Write-Log -Message ('===== RUN WORK_FILE: ' + $WorkFile + ' =====') -Color Cyan
Write-Log -Message ('CONFIG: SYNC_MODE=' + $SyncMode + ', RUN_LOG_ENABLED=' + $runLogEnabled) -Color Cyan
Write-Log -Message ('EXE   : ' + $ExeFull)
Write-Log -Message ('SYNC  : ' + $SaveSyncDir + ' , ' + $SaveP95Dir)

Sync-SaveFiles -Mode 'full'

$i = 0
try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $WorkFileFull -PathType Leaf)) {
            Write-Log -Message ('WORK_FILE not found: ' + $WorkFile) -Color Yellow
            break
        }

        $line = Get-FirstTaskLine -Path $WorkFileFull
        if ([string]::IsNullOrWhiteSpace($line)) { break }

        $i++

        # An unparsable line cannot be executed: report it, keep it visible in the
        # finished file and drop it from the work file so the queue keeps moving.
        $task = $null
        try { $task = Parse-WorkLine -Line $line }
        catch {
            Write-Log -Message ('ERROR: ' + $_.Exception.Message) -Color Red
            [System.IO.File]::AppendAllText($FinishedFull, ('# ERROR unparsed: ' + $line + "`r`n"), $Utf8NoBom)
            Remove-FirstTaskLine -Path $WorkFileFull
            continue
        }

        $Script:RunLogPath = $null
        if ($RunLog) {
            $Script:RunLogPath = Join-Path $LogDirPath ('run_' + (Get-Date).ToString('yyyy-MM-dd_HHmmss') + '_' + $i + '.log')
        }

        # Marker for the incremental sync: .save files touched from now on (including
        # --ckpt checkpoints written during the task) belong to this task.
        $marker = (Get-Date).AddSeconds(-2)

        Write-Log -Message ('START: ' + $line) -Color Green
        if ($task.Modified) {
            Write-Log -Message ('EXEC : ' + $task.ExeDisplay + ' ' + ($task.Args -join ' ')) -Color DarkGray
        }

        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        $exitCode = $null
        try { $exitCode = Invoke-EcmTask -Task $task }
        catch { Write-Log -Message ('ERROR: ' + $_.Exception.Message) -Color Red }
        $watch.Stop()

        Write-Log -Message ('DONE : (elapsed ' + [int]$watch.Elapsed.TotalSeconds + 's) ' + $line) -Color Green
        if ($null -ne $exitCode -and $exitCode -ne 0) {
            Write-Log -Message ('WARN : task exit code ' + $exitCode) -Color Yellow
        }

        # Append the finished line, then remove it from the work file.
        [System.IO.File]::AppendAllText($FinishedFull, ($line + "`r`n"), $Utf8NoBom)
        Remove-FirstTaskLine -Path $WorkFileFull

        Sync-SaveFiles -Mode $SyncMode -Since $marker
    }

    Write-Log -Message ('===== ALL DONE, ' + $i + ' task(s) =====') -Color Cyan
}
finally {
    $Script:RunLogPath = $null
}

exit 0
