# row_ink_profile.ps1 -- locate text rows in a screenshot (where is the ink?).
#
# Prints, for every row, the count of dark pixels, then groups consecutive rows with
# "text-like" ink into bands. Used to aim text_ink_probe.ps1 at real text instead of
# guessing the band (the ImGui menu bar is not always at y=0 in a captured window).
#
# "Ink" is auto-detected from the dominant luminance (background); the GUI's dark
# theme means text is *lighter* than the background, so a fixed dark threshold would
# mark the whole window as ink.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Delta = 45,
    [int]$MinInk = 8
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Bitmap]::FromFile((Resolve-Path $Path).Path)
try {
    $hist = New-Object 'int[]' 256
    for ($y = 0; $y -lt $img.Height; $y += 2) {
        for ($x = 0; $x -lt $img.Width; $x += 2) {
            $c = $img.GetPixel($x, $y)
            $hist[[int]((0.299 * $c.R) + (0.587 * $c.G) + (0.114 * $c.B))]++
        }
    }
    $bg = 0
    for ($i = 1; $i -lt 256; $i++) { if ($hist[$i] -gt $hist[$bg]) { $bg = $i } }

    Write-Host ("image {0}x{1}  background luminance={2}" -f $img.Width, $img.Height, $bg)
    $counts = New-Object 'int[]' $img.Height
    for ($y = 0; $y -lt $img.Height; $y++) {
        $n = 0
        for ($x = 0; $x -lt $img.Width; $x++) {
            $c = $img.GetPixel($x, $y)
            $lum = (0.299 * $c.R) + (0.587 * $c.G) + (0.114 * $c.B)
            if ([Math]::Abs($lum - $bg) -gt $Delta) { $n++ }
        }
        $counts[$y] = $n
    }
    $max = ($counts | Measure-Object -Maximum).Maximum
    Write-Host ("max ink in a row: {0}" -f $max)
    # First 60 rows verbatim: the menu bar lives there.
    Write-Host "--- first 60 rows (y : ink)"
    for ($y = 0; $y -lt [Math]::Min(60, $img.Height); $y++) {
        Write-Host ("{0,4} : {1,6} {2}" -f $y, $counts[$y], ('#' * [Math]::Min(60, [int]($counts[$y] / [Math]::Max(1, $max) * 60))))
    }
    # Text-like rows: some ink, but not a filled panel.
    $lo = [Math]::Max(1, $MinInk)
    $hi = [Math]::Max($lo + 1, [int]($img.Width * 0.9))
    Write-Host ("--- text-like bands (ink {0}..{1})" -f $lo, $hi)
    $y = 0
    while ($y -lt $img.Height) {
        $isText = ($counts[$y] -ge $lo -and $counts[$y] -le $hi)
        if (-not $isText) { $y++; continue }
        $start = $y
        while ($y -lt $img.Height -and $counts[$y] -ge $lo -and $counts[$y] -le $hi) { $y++ }
        $end = $y - 1
        if (($end - $start) -ge 4) {
            $peak = ($counts[$start..$end] | Measure-Object -Maximum).Maximum
            Write-Host ("band y={0,4}..{1,-4} h={2,-3} peakInk={3}" -f $start, $end, ($end - $start + 1), $peak)
        }
    }
}
finally { $img.Dispose() }
