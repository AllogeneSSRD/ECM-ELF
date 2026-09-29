# Scan src/gui/*.cpp for ImGui::Text-style calls whose format specifiers do not match the
# number of arguments (the class of bug that crashed the GUI: a %s fed an integer makes
# vsnprintf dereference it).
$files = Get-ChildItem 'src\gui' -Filter *.cpp
$problems = 0
foreach ($f in $files) {
    $text = [System.IO.File]::ReadAllText($f.FullName)
    # join continuation lines: a call spanning lines is joined into one logical line
    $lines = $text -split "`r?`n"
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -notmatch 'ImGui::Text') { continue }
        $call = $lines[$i]
        $j = $i
        while (($call.ToCharArray() | ForEach-Object { $_ } | Where-Object { $_ -eq '(' }).Count -gt
               ($call.ToCharArray() | ForEach-Object { $_ } | Where-Object { $_ -eq ')' }).Count -and
               $j + 1 -lt $lines.Count) {
            $j++
            $call += ' ' + $lines[$j].Trim()
        }
        # format string = first quoted literal
        $m = [regex]::Match($call, '"((?:[^"\\]|\\.)*)"')
        if (-not $m.Success) { continue }
        $fmt = $m.Groups[1].Value
        # count conversion specifiers (ignore %%)
        $specs = ([regex]::Matches($fmt, '%[-+ #0]*[\d\.\*]*[hlLjzt]*[diouxXeEfgGaAcspn%]') |
                  Where-Object { $_.Value -ne '%%' }).Count
        # arguments = everything after the format literal, split on top-level commas
        $rest = $call.Substring($m.Index + $m.Length)
        $rest = $rest.TrimStart().TrimStart(',').Trim()
        $depth = 0; $args = 0; $cur = ''
        foreach ($ch in $rest.ToCharArray()) {
            switch ($ch) {
                '(' { $depth++; $cur += $ch }
                ')' { if ($depth -eq 0) { break } else { $depth--; $cur += $ch } }
                ',' { if ($depth -eq 0) { if ($cur.Trim()) { $args++ }; $cur = '' } else { $cur += $ch } }
                default { $cur += $ch }
            }
        }
        if ($cur.Trim()) { $args++ }
        # A call without conversion specifiers takes just the string (TextUnformatted, or a
        # Text("literal") with a %-free string), so 0 specifiers is always fine.
        if ($specs -eq 0) { continue }
        if ($specs -ne $args) {
            $problems++
            "{0}({1}): specs={2} args={3}" -f $f.Name, ($i + 1), $specs, $args
            "    " + $lines[$i].Trim()
        }
    }
}
"checked $($files.Count) files, $problems mismatch(es)"
