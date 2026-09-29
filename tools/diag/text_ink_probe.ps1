# text_ink_probe.ps1 -- decide "real CJK glyphs" vs "tofu boxes" from window pixels.
#
# Why: an ImGui atlas check (App::trace_font_metrics) proves the glyph was baked into
# the atlas, but not that the D3D11 backend uploaded/rendered it. This probe looks at
# the actual captured pixels.
#
# Method: locate a text band (by default the first one = the ImGui menu bar), split it
# into per-glyph cells by empty columns, and measure each cell's ink.
#
# What "tofu" actually looks like (measured -- see tools/test/test_gui_cjk_pixels.ps1):
#   real CJK -> cell width ~0.75 em (full-width), cells all DIFFERENT (ink/midInk)
#   tofu     -> every unmapped char is the SAME fallback glyph, so the cells are
#               byte-identical (same w/h/ink) and only ~0.43 em wide
# Hence the discriminating statistics are (a) median cell width / font size, and
# (b) the number of distinct ink values -- NOT "is the centre hollow": the U+FFFD/'?'
# fallback is not hollow either (it has strokes).
#
# "Ink" is auto-detected: the dominant luminance in the image is the background (the
# GUI uses a dark theme, so text is *lighter* than the background) and a pixel counts
# as ink when it departs from that background by more than -Delta.
#
# Usage:
#   powershell -File tools\diag\text_ink_probe.ps1 -Path shot.png             # auto band
#   powershell -File tools\diag\text_ink_probe.ps1 -Path shot.png -Top 55 -Height 32
#   powershell -File tools\diag\text_ink_probe.ps1 -Path shot.png -Json   # machine readable
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Top = -1,        # -1 = locate the first text band automatically
    [int]$Height = 0,      # 0 = adopt the located band's height
    [int]$Delta = 140,     # ink = luminance differs from the background by more than this
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$img = [System.Drawing.Bitmap]::FromFile((Resolve-Path $Path).Path)
try {
    $w = $img.Width
    $imgH = $img.Height

    # --- what is the background? (dominant luminance over the whole image) --------
    $hist = New-Object 'int[]' 256
    for ($y = 0; $y -lt $imgH; $y += 2) {
        for ($x = 0; $x -lt $w; $x += 2) {
            $c = $img.GetPixel($x, $y)
            $hist[[int]((0.299 * $c.R) + (0.587 * $c.G) + (0.114 * $c.B))]++
        }
    }
    $bg = 0
    for ($i = 1; $i -lt 256; $i++) { if ($hist[$i] -gt $hist[$bg]) { $bg = $i } }

    function Test-InkPixel {
        param([System.Drawing.Color]$c, [int]$Background, [int]$Threshold)
        $lum = (0.299 * $c.R) + (0.587 * $c.G) + (0.114 * $c.B)
        return ([Math]::Abs($lum - $Background) -gt $Threshold)
    }

    # --- locate the first text band (skips the title bar and panel borders) -------
    # A text row has some ink but never a near-full-width run of it.
    $bandH = 0
    if ($Top -lt 0) {
        $limit = [int]($imgH * 0.5)
        $rowInk = New-Object 'int[]' $limit
        for ($y = 0; $y -lt $limit; $y++) {
            $n = 0
            for ($x = 0; $x -lt $w; $x += 2) {
                if (Test-InkPixel $img.GetPixel($x, $y) $bg $Delta) { $n++ }
            }
            $rowInk[$y] = $n * 2                  # sampled every 2nd column
        }
        $rowHi = [int]($w * 0.75)
        $y = 0
        $Top = -1
        while ($y -lt $limit) {
            if (-not ($rowInk[$y] -ge 6 -and $rowInk[$y] -le $rowHi)) { $y++; continue }
            $start = $y
            while ($y -lt $limit -and $rowInk[$y] -ge 6 -and $rowInk[$y] -le $rowHi) { $y++ }
            $end = $y - 1
            if (($end - $start + 1) -ge 8) { $Top = $start; $bandH = $end - $start + 1; break }
        }
        if ($Top -lt 0) {
            Write-Error "no text band found in the top half of '$Path' (nothing rendered?)"
            exit 2
        }
        if ($Height -le 0) { $Height = $bandH }
    }
    else {
        $bandH = $Height
    }

    $y0 = [Math]::Max(0, $Top)
    $h = [Math]::Min($Height, $imgH - $y0)

    # --- rasterize the band into a boolean ink map -------------------------------
    $ink = New-Object 'bool[,]' $h, $w
    $colInk = New-Object 'int[]' $w
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            if (Test-InkPixel $img.GetPixel($x, $y + $y0) $bg $Delta) {
                $ink[$y, $x] = $true
                $colInk[$x]++
            }
        }
    }

    # --- split the band into glyph cells on empty columns ------------------------
    # A run of >= 2 empty columns separates two glyphs; cells wider than ~1.5 em are
    # merged neighbours (reported as-is, the caller can see the width).
    $cells = New-Object System.Collections.ArrayList
    $x = 0
    while ($x -lt $w) {
        if ($colInk[$x] -eq 0) { $x++; continue }
        $start = $x
        $gap = 0
        while ($x -lt $w -and $gap -lt 2) {
            if ($colInk[$x] -eq 0) { $gap++ } else { $gap = 0 }
            $x++
        }
        $end = $x - $gap                       # last column holding ink
        if (($end - $start) -ge 3) {
            [void]$cells.Add([pscustomobject]@{ Start = $start; End = $end })
        }
    }

    # --- per-cell ink statistics -------------------------------------------------
    $rows = foreach ($c in $cells) {
        $cw = $c.End - $c.Start + 1
        $top = -1; $bot = -1
        for ($y = 0; $y -lt $h; $y++) {
            for ($xx = $c.Start; $xx -le $c.End; $xx++) {
                if ($ink[$y, $xx]) { if ($top -lt 0) { $top = $y }; $bot = $y; break }
            }
        }
        if ($top -lt 0) { continue }
        $ch = $bot - $top + 1

        # central region: middle 40 % in both axes
        $cx0 = $c.Start + [Math]::Floor($cw * 0.30)
        $cx1 = $c.Start + [Math]::Ceiling($cw * 0.70) - 1
        $cy0 = $top + [Math]::Floor($ch * 0.30)
        $cy1 = $top + [Math]::Ceiling($ch * 0.70) - 1

        $total = 0; $interior = 0
        for ($y = $top; $y -le $bot; $y++) {
            for ($xx = $c.Start; $xx -le $c.End; $xx++) {
                if (-not $ink[$y, $xx]) { continue }
                $total++
                if ($xx -ge $cx0 -and $xx -le $cx1 -and $y -ge $cy0 -and $y -le $cy1) { $interior++ }
            }
        }
        $area = $cw * $ch
        $interiorArea = [Math]::Max(1, ($cx1 - $cx0 + 1) * ($cy1 - $cy0 + 1))
        [pscustomobject]@{
            x       = $c.Start
            w       = $cw
            h       = $ch
            ink     = $total
            fillPct = [Math]::Round(100.0 * $total / $area, 1)
            midInk  = $interior
            midPct  = [Math]::Round(100.0 * $interior / $interiorArea, 1)
        }
    }
    $rows = @($rows)
    $widths = @($rows | ForEach-Object { $_.w } | Sort-Object)
    $median = 0
    if ($widths.Count -gt 0) { $median = $widths[[int][Math]::Floor($widths.Count / 2)] }
    $distinct = @($rows | Select-Object -ExpandProperty ink -Unique).Count

    if ($Json) {
        [pscustomobject]@{
            path            = (Resolve-Path $Path).Path
            width           = $w
            height          = $imgH
            background      = $bg
            band            = @{ top = $y0; height = $h }
            cells           = $rows
            cellCount       = $rows.Count
            medianCellWidth = $median
            distinctInk     = $distinct
        } | ConvertTo-Json -Depth 4
        return
    }

    Write-Host ("image {0}x{1}  background luminance={2}  band y={3}..{4}  cells={5}" -f `
            $w, $imgH, $bg, $y0, ($y0 + $h - 1), $rows.Count)
    Write-Host ("{0,6} {1,5} {2,5} {3,7} {4,8} {5,7} {6,8}" -f 'x', 'w', 'h', 'ink', 'fill%', 'midInk', 'mid%')
    foreach ($r in $rows) {
        Write-Host ("{0,6} {1,5} {2,5} {3,7} {4,8} {5,7} {6,8}" -f `
                $r.x, $r.w, $r.h, $r.ink, $r.fillPct, $r.midInk, $r.midPct)
    }
    Write-Host ("median cell width: {0}   distinct ink values: {1}   cells with interior ink: {2}/{3}" -f `
            $median, $distinct, @($rows | Where-Object { $_.midInk -gt 0 }).Count, $rows.Count)
}
finally {
    $img.Dispose()
}
