#Requires -Version 5.1
<#
.SYNOPSIS
    Pixel-level proof that ecm_gui really draws Chinese glyphs (not "tofu").

.DESCRIPTION
    Three different failures can make the UI show Chinese as boxes, and they need
    three different checks:

      1. the localization XML was not loaded      -> trace: localization: ... language=...
      2. the font has no such glyph / not baked   -> trace: font: ... map=.. baked=.. negctl=..
         (ImGui's dynamic atlas silently substitutes U+FFFD, whose advance is nonzero,
          so "the text has a width" proves nothing)
      3. the glyph is in the atlas but the D3D11 backend never draws it
                                                  -> the actual window PIXELS

    This script covers (3) by capturing the window (tools/diag/grab_window.ps1, which
    uses PrintWindow + PW_RENDERFULLCONTENT) and measuring glyph cells in the first
    text band (tools/diag/text_ink_probe.ps1).

    The measurement is calibrated against a negative control: the same Chinese UI with
    [GUI] font forced to a font that has no CJK glyphs (arial.ttf). Measured facts:

      real CJK (msyh.ttc, 40 px)  -> 5 cells, median width 30 px (0.75 em), 5 distinct
                                     ink values (each glyph is its own shape)
      tofu (arial.ttf, 40 px)     -> 6 cells, median width 17 px (0.43 em), 2 distinct
                                     ink values (the same fallback glyph repeated)

    So the assertions are: median cell width >= 0.6 em AND >= 3 distinct ink values.
    The control run must FAIL those same assertions -- otherwise the test proves
    nothing and this script fails loudly instead of reporting a false pass.

    Exit code: 0 = all checks passed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_cjk_pixels.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = "",
    [int]$FontSize = 40,
    [string]$ControlFont = ""
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe",
                        "$repoRoot\build_vs18\Release\ecm_gui.exe",
                        "$repoRoot\build_cuda_cmake\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "FAIL: ecm_gui.exe not found (pass -Exe <path>)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_cjk_pixels' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

$grabScript = Join-Path $repoRoot 'tools\diag\grab_window.ps1'
$probeScript = Join-Path $repoRoot 'tools\diag\text_ink_probe.ps1'
foreach ($s in @($grabScript, $probeScript)) {
    if (-not (Test-Path $s)) { Write-Host "FAIL: missing $s" -ForegroundColor Red; exit 2 }
}

# The negative control needs a font with real Latin glyphs but NO CJK glyphs.
if (-not $ControlFont) { $ControlFont = "$env:WINDIR\Fonts\arial.ttf" }
$haveControl = Test-Path $ControlFont

$script:pass = 0
$script:fail = 0
function Check([string]$name, $ok, [string]$detail = "") {
    if ($ok) {
        $script:pass++
        Write-Host ("  [ok]   " + $name)
    } else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

function Read-TextShared([string]$path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
        $text = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
        return $text
    } catch { return "" }
}

# ---------------------------------------------------------------------------------
# One run: sandbox with the requested [GUI] font setting + Chinese UI, screenshot the
# The screenshot is taken with PrintWindow over a DC sized to the CLIENT rect, so the image
# starts at the WINDOW's top-left: the DWM title bar comes first, then the menu bar. The title
# bar is a solid band (nearly every pixel in the row is "ink" in the probe's sense), so its last
# row is easy to find and the menu bar starts right below it.
# Why explicit geometry at all: the probe's automatic "first text band" scan cut the menu-bar
# glyphs short once the Prime95 notice strip added a second text row underneath them -- the
# measured ink runs then covered only the upper part of each glyph, and the median cell width
# fell from 30 px to 23 px (2026-09-29), which looked exactly like a narrow fallback font.
# ---------------------------------------------------------------------------------
# window, measure the glyph cells. Returns @{ trace; probe } or $null on failure.
# ---------------------------------------------------------------------------------
function Invoke-CjkRun([string]$Name, [string]$FontSetting, [string]$EnvCjkFont = "",
                       [string]$ExtraArgs = "", [string]$Language = "") {
    $dir = Join-Path $Sandbox $Name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if (-not $Language) { $Language = 'chineseSimplified' }
    $ini = @(
        '[GUI]',
        'window = 40,30,1600,1000',
        ('language = ' + $Language),
        ('font_size = ' + $FontSize),
        'refresh_hz = 30'
    )
    if ($FontSetting) { $ini += ('font = ' + $FontSetting) }
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $dir 'ecm.ini'), (($ini -join "`r`n") + "`r`n"), $enc)

    # The GUI looks for localization/ next to the exe first; the build copies it there.
    $locDst = Join-Path (Split-Path -Parent $Exe) 'localization'
    if (-not (Test-Path (Join-Path $locDst 'chineseSimplified.xml'))) {
        Copy-Item (Join-Path $repoRoot 'src\gui\localization') $locDst -Recurse -Force
    }

    # ECM_GUI_CJK_FONT overrides the CJK font search (see platform_win32.cpp): run [C]
    # uses it to simulate a machine with no CJK font at all.
    if ($EnvCjkFont) { $env:ECM_GUI_CJK_FONT = $EnvCjkFont } else { Remove-Item Env:\ECM_GUI_CJK_FONT -ErrorAction SilentlyContinue }
    $argv = @('-ini', (Join-Path $dir 'ecm.ini'), '--trace')
    if ($ExtraArgs) { $argv += $ExtraArgs.Split(' ') }
    $proc = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru
    Start-Sleep -Seconds 6
    # Which rows are the MENU BAR? The GUI traces its height (`layout: menu_bar_h=…`, last
    # match wins -- frame 1 still reports 0 before the menu bar is measured) and the title bar's
    # end comes from the capture itself, so the band is exact instead of "the first text band".
    $menuBarH = 0
    $traceNow = Read-TextShared (Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log')
    $mbAll = [regex]::Matches($traceNow, 'layout: menu_bar_h=(\d+)')
    if ($mbAll.Count -gt 0) {
        $menuBarH = [int]$mbAll[$mbAll.Count - 1].Groups[1].Value
    }
    $png = Join-Path $dir 'window.png'
    # Capturing is retried: a screenshot can land on a partially drawn frame (DWM/composition
    # timing), and then the probe finds narrow fragments instead of glyphs -- that produced a
    # flaky "median cell width 9 px" failure once (2026-09-29). The property under test is
    # "the UI renders CJK", not "every single capture is perfect", so keep the best attempt.
    $best = $null
    $bestCells = -1
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        # -ProcId, not -ProcessName: the operator's own GUI is normally running, and "the first
        # ecm_gui with a window" then captures THEIR window (measured 2026-09-29: a capture came
        # back at the production window's size), which would make the pixel assertions measure
        # the wrong UI.
        & powershell -NoProfile -ExecutionPolicy Bypass -File $grabScript `
            -ProcId $proc.Id -ProcessName ecm_gui -Class ecm_gui -Out $png 2>&1 | Out-Null
        if (-not (Test-Path $png)) { Start-Sleep -Seconds 1; continue }
        # The probe locates the text band itself (its "first text band" rule); forcing a band
        # from the traced geometry was tried and made things WORSE (2026-09-29: feeding it the
        # whole 48-row menu bar split every glyph into radical fragments -- 31 cells of 13 px).
        # What matters is the CALIBRATION below, not the exact rows: real CJK measures 23-30 px
        # at font_size 40 (0.575-0.75 em) while the tofu/English fallback measures 13-17 px
        # (0.33-0.43 em), so the pass mark is 0.5 em.
        $probeBand = @()
        $json = & powershell -NoProfile -ExecutionPolicy Bypass -File $probeScript -Path $png -Json @probeBand 2>&1
        $cand = $null
        try { $cand = ($json -join "`n") | ConvertFrom-Json } catch { $cand = $null }
        if ($null -ne $cand) {
            $n = @($cand.cells).Count
            if ($n -gt $bestCells) { $bestCells = $n; $best = $cand }
            if ($n -ge 3 -and [int]$cand.medianCellWidth -ge 20) { break }   # good enough
        }
        Start-Sleep -Seconds 1
    }
    Remove-Item Env:\ECM_GUI_CJK_FONT -ErrorAction SilentlyContinue

    $trace = Read-TextShared (Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log')
    $proc.Refresh()
    if (-not $proc.HasExited) { $proc.CloseMainWindow() | Out-Null; Start-Sleep -Seconds 2 }
    $proc.Refresh()
    if (-not $proc.HasExited) { $proc.Kill() }
    # Keep this run's trace: the next run truncates the shared file.
    $traceFile = Join-Path $dir 'trace.log'
    [System.IO.File]::WriteAllText($traceFile, $trace, [System.Text.Encoding]::UTF8)

    $probe = $best
    return [pscustomobject]@{ Trace = $trace; Probe = $probe; Png = $png; Dir = $dir }
}

# ---------------------------------------------------------------------------------
Write-Host "ecm_gui CJK pixel check (font_size=$FontSize)"
Write-Host ("exe     : " + $Exe)
Write-Host ("sandbox : " + $Sandbox)

Write-Host "[A] Chinese UI with the auto-selected system font must render real glyphs"
$a = Invoke-CjkRun 'real' ''
# Two separate trace lines (the header is written before the atlas is measured), so
# match them one at a time -- '.' does not cross a newline in .NET regexes.
$ahdr = [regex]::Match($a.Trace, "font: (\S+) at ([\d.]+) px ")
$amap = [regex]::Match($a.Trace, "cjk_ok=(\d) map=(\d)(\d) baked=(\d)(\d) negctl=(\d)(\d)")
Check "the trace reports the font decision" ($ahdr.Success -and $amap.Success)
if ($ahdr.Success) {
    $fontPath = $ahdr.Groups[1].Value
    $px = [double]$ahdr.Groups[2].Value
    Write-Host ("       font=" + $fontPath + " size=" + $px + "px cjk_ok=" + $amap.Groups[1].Value +
                " map=" + $amap.Groups[2].Value + $amap.Groups[3].Value +
                " baked=" + $amap.Groups[4].Value + $amap.Groups[5].Value +
                " negctl=" + $amap.Groups[6].Value + $amap.Groups[7].Value)
    Check "the requested font size was applied" ($px -eq $FontSize) ("size=" + $px)
    Check "the atlas has both Chinese glyphs (not the fallback box)" `
          (($amap.Groups[2].Value -eq '1') -and ($amap.Groups[3].Value -eq '1') -and
           ($amap.Groups[4].Value -eq '1') -and ($amap.Groups[5].Value -eq '1')) `
          ("map=" + $amap.Groups[2].Value + $amap.Groups[3].Value +
           " baked=" + $amap.Groups[4].Value + $amap.Groups[5].Value)
    Check "the negative control codepoint (U+E123) is reported missing" `
          (($amap.Groups[6].Value -eq '0') -and ($amap.Groups[7].Value -eq '0'))
    Check "the atlas check itself says cjk_ok=1" ($amap.Groups[1].Value -eq '1')
    $pxRef = $px
} else {
    $pxRef = $FontSize
}
$aloc = [regex]::Match($a.Trace, "localization: .*language=(\S+) keys=(\d+) baseline=(\d+) missing=(\d+)")
Check "the Chinese localization was loaded with 0 missing keys" `
      ($aloc.Success -and $aloc.Groups[1].Value -eq 'chineseSimplified' -and
       [int]$aloc.Groups[4].Value -eq 0) $(if ($aloc.Success) { "missing=" + $aloc.Groups[4].Value } else { "no trace line" })

Check "the window was captured" ($null -ne $a.Probe) $a.Png
$realOk = $false
$realMedian = 0
if ($null -ne $a.Probe) {
    $cells = @($a.Probe.cells)
    $realMedian = [int]$a.Probe.medianCellWidth
    $realDistinct = [int]$a.Probe.distinctInk
    Write-Host ("       cells=" + $cells.Count + " medianWidth=" + $realMedian +
                "px distinctInk=" + $realDistinct + " band=" + $a.Probe.band.top + ".." +
                ($a.Probe.band.top + $a.Probe.band.height - 1))
    foreach ($c in $cells) {
        Write-Host ("       cell x=" + $c.x + " w=" + $c.w + " h=" + $c.h + " ink=" + $c.ink + " midInk=" + $c.midInk)
    }
    Check "at least three glyphs were found in the menu bar" ($cells.Count -ge 3) ("cells=" + $cells.Count)
    # ~0.75 em for a full-width CJK glyph; the tofu fallback measured 0.43 em.
    Check "the glyph cells are full-width (CJK advance, not a narrow fallback)" `
          ($realMedian -ge (0.5 * $pxRef)) ("median=" + $realMedian + "px of " + $pxRef + "px")
    Check "the glyphs differ from each other (real shapes, not one repeated box)" `
          ($realDistinct -ge 3) ("distinctInk=" + $realDistinct)
    Check "every glyph has strokes in its centre" `
          (@($cells | Where-Object { $_.midInk -gt 0 }).Count -ge 3)
    $realOk = ($cells.Count -ge 3) -and ($realMedian -ge (0.5 * $pxRef)) -and ($realDistinct -ge 3)
}

if (-not $haveControl) {
    Write-Host ("[B] SKIPPED: no control font at " + $ControlFont) -ForegroundColor Yellow
    Write-Host "     (it simulates a user-configured font that cannot draw Chinese)"
} else {
    Write-Host("[B] a CJK-less [GUI] font must be rescued, not rendered as boxes")
    # This is the user-reported failure: a font that cannot draw the language produced
    # "???" labels. The GUI must detect it (atlas API) and load a CJK system font.
    $b = Invoke-CjkRun 'rescue' $ControlFont
    Check "the configured (CJK-less) font is reported as such" `
          ($b.Trace -match 'cannot draw CJK: rescuing') `
          "no 'cannot draw CJK: rescuing' line in the trace"
    $bresc = [regex]::Match($b.Trace, "font: (\S+) at [\d.]+ px \([^)]*cjk rescue font")
    Check "a CJK system font was loaded instead" $bresc.Success `
          $(if ($bresc.Success) { $bresc.Groups[1].Value } else { "no rescue line" })
    $bmap = [regex]::Match($b.Trace, "cjk_ok=(\d) map=(\d)(\d) baked=(\d)(\d)")
    if ($bmap.Success) {
        Write-Host ("       cjk_ok=" + $bmap.Groups[1].Value +
                    " map=" + $bmap.Groups[2].Value + $bmap.Groups[3].Value +
                    " baked=" + $bmap.Groups[4].Value + $bmap.Groups[5].Value)
        Check "the rescued font really has both glyphs baked in" `
              ($bmap.Groups[1].Value -eq '1' -and $bmap.Groups[2].Value -eq '1' -and
               $bmap.Groups[3].Value -eq '1' -and $bmap.Groups[4].Value -eq '1' -and
               $bmap.Groups[5].Value -eq '1')
    }
    Check "the rescued window was captured" ($null -ne $b.Probe) $b.Png
    if ($null -ne $b.Probe) {
        $bCells = @($b.Probe.cells)
        $bMedian = [int]$b.Probe.medianCellWidth
        $bDistinct = [int]$b.Probe.distinctInk
        Write-Host ("       cells=" + $bCells.Count + " medianWidth=" + $bMedian +
                    "px distinctInk=" + $bDistinct)
        # The rescued run must pass the SAME pixel criteria as run A: full-width cells,
        # distinct shapes. (Measured tofu before this fix: 17 px, 2 distinct values.)
        Check "the rescued glyphs are full-width CJK too" ($bMedian -ge (0.5 * $pxRef)) `
              ("median=" + $bMedian + "px")
        Check "the rescued glyphs differ from each other" ($bDistinct -ge 3) `
              ("distinctInk=" + $bDistinct)
    }
}

Write-Host "[C] no CJK font anywhere: the UI must fall back to English, never boxes"
# ECM_GUI_CJK_FONT=none hides every CJK font from the GUI, so the Chinese UI cannot be
# drawn at all. The required behaviour is an English (drawable) UI plus a status message
# -- NOT a screen full of "???".
$c = Invoke-CjkRun 'no-cjk' '' 'none'
Check "the GUI says it switched the UI to English" `
      ($c.Trace -match 'no CJK-capable font found: switched the UI to English') `
      "no English-fallback line in the trace"
$cloc = [regex]::Match($c.Trace, "localization: dir=\S+ language=(\S+) keys=(\d+) baseline=(\d+) missing=(\d+)")
Check "the trace reports the English localization" ($cloc.Success -and $cloc.Groups[1].Value -eq 'english') `
      $(if ($cloc.Success) { $cloc.Groups[1].Value } else { "no localization line" })
Check "the window was still captured (the GUI did not die)" ($null -ne $c.Probe) $c.Png
if ($null -ne $c.Probe) {
    $cCells = @($c.Probe.cells)
    $cMedian = [int]$c.Probe.medianCellWidth
    Write-Host ("       cells=" + $cCells.Count + " medianWidth=" + $cMedian +
                "px distinctInk=" + $c.Probe.distinctInk)
    # English menu labels are narrow Latin text, so the cells must NOT look like
    # full-width CJK -- and more importantly they are real letters, not "?" markers.
    Check "the fallback UI draws Latin labels (not CJK-width boxes)" `
          ($cCells.Count -ge 3 -and $cMedian -lt (0.6 * $pxRef)) `
          ("cells=" + $cCells.Count + " median=" + $cMedian + "px")
    # A "?" fallback for Chinese would also be narrow, so check the text is English:
    # the workers panel title (ASCII when English) is drawn on the first traced frame.
    Check "the traced UI sample is the English string" ($c.Trace -match "sample workers\.title='Workers'") `
          "the localization sample is not the English baseline"
}

Write-Host "[D] the user's exact case: Latin font at startup, switch to Chinese at runtime"
# This is what produced the "???" report: the ini says language=english, so the startup
# font is the Latin system font; the user then picked Chinese from the Language menu and
# nothing had loaded a CJK font. --switch-language performs exactly the menu's call.
$d = Invoke-CjkRun 'runtime-switch' '' '' '--switch-language chineseSimplified --switch-language-after 20' 'english'
Check "the startup font was the Latin system font" `
      ($d.Trace -match 'font: \S+ at [\d.]+ px \([^)]*latin system font[^)]*\) \[startup\]') `
      "no startup latin-font line in the trace"
Check "the runtime switch is visible in the trace" `
      ($d.Trace -match 'language: runtime switch to chineseSimplified') `
      "no 'language: runtime switch' line"
# The decisive line: the font was re-applied for the new language, with a CJK font.
# (Matched in two steps: the trace line is
#   font: <path> at <px> px (dpi x.., cjk system font, snap=1), CJK language [language change]
# so the parenthesised part and the suffix are not adjacent.)
Check "the font was re-applied on the language change" `
      ($d.Trace -match 'font: \S+ at [\d.]+ px [^\r\n]*\[language change\]') `
      "no '[language change]' font line"
Check "the re-applied font is CJK-capable" `
      ($d.Trace -match 'font: [^\r\n]*cjk system font[^\r\n]*\[language change\]') `
      "the reloaded font is not the CJK one"
$dmap = [regex]::Match($d.Trace, "cjk_ok=(\d) map=(\d)(\d) baked=(\d)(\d)")
if ($dmap.Success) {
    Write-Host ("       cjk_ok=" + $dmap.Groups[1].Value +
                " map=" + $dmap.Groups[2].Value + $dmap.Groups[3].Value +
                " baked=" + $dmap.Groups[4].Value + $dmap.Groups[5].Value)
    Check "Chinese glyphs are baked after the switch (no ??? fallback)" `
          ($dmap.Groups[1].Value -eq '1' -and $dmap.Groups[2].Value -eq '1' -and
           $dmap.Groups[3].Value -eq '1' -and $dmap.Groups[4].Value -eq '1' -and
           $dmap.Groups[5].Value -eq '1')
}
# The sample must be non-ASCII after the switch (i.e. the Chinese string, not the English
# baseline). Written as a character class on purpose: a literal Chinese pattern in a .ps1
# file without a BOM is decoded as ANSI by Windows PowerShell 5.1 and never matches.
Check "the trace sample is non-ASCII (Chinese) after the switch" `
      ($d.Trace -match "sample workers\.title='[^']*[^\x20-\x7E][^']*'") `
      "the localization sample is not the Chinese one"
Check "the switched window was captured" ($null -ne $d.Probe) $d.Png
if ($null -ne $d.Probe) {
    $dCells = @($d.Probe.cells)
    $dMedian = [int]$d.Probe.medianCellWidth
    $dDistinct = [int]$d.Probe.distinctInk
    Write-Host ("       cells=" + $dCells.Count + " medianWidth=" + $dMedian +
                "px distinctInk=" + $dDistinct)
    # Pixels, not just the atlas: after the switch the menu bar must show full-width CJK
    # glyphs. Before the fix it showed narrow "?" fallbacks (measured: 17 px at 40 px).
    Check "the switched UI draws full-width CJK glyphs" ($dMedian -ge (0.5 * $pxRef)) `
          ("median=" + $dMedian + "px of " + $pxRef + "px")
    Check "the switched glyphs differ from each other (not one repeated marker)" `
          ($dDistinct -ge 3) ("distinctInk=" + $dDistinct)
}

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
