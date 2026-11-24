# Builds slides/<name>.pptx from build/<name>/deck.json (written by export.mjs), using PowerPoint on Windows.
# Each slide: the rendered background (graphics, charts, photos) as a picture, and every text block as an editable text box.
#   pwsh make_pptx.ps1 midterm
param([Parameter(Mandatory = $true)][string]$Name)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here "build\$Name"
$out = Join-Path (Split-Path -Parent $here) "$Name.pptx"
$deck = Get-Content (Join-Path $src 'deck.json') -Raw -Encoding utf8 | ConvertFrom-Json
$K = 0.5                                   # 1920 px -> 960 pt

function ConvertTo-Rgb([string]$css) {
    $m = [regex]::Match($css, 'rgba?\((\d+),\s*(\d+),\s*(\d+)')
    if (-not $m.Success) { return 0 }
    return [int]$m.Groups[1].Value + 256 * [int]$m.Groups[2].Value + 65536 * [int]$m.Groups[3].Value
}

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Add($false)     # no window
try {
    $pres.PageSetup.SlideWidth = 960
    $pres.PageSetup.SlideHeight = 540
    $n = 0
    foreach ($s in $deck) {
        $n++
        $sl = $pres.Slides.Add($n, 12)     # blank layout
        $pic = $sl.Shapes.AddPicture((Join-Path $src $s.bg), $false, $true, 0, 0, 960, 540)
        $pic.AlternativeText = "Slide graphics: $($s.title)"
        $pic.Name = 'Background'
        foreach ($b in $s.boxes) {
            $slack = 24                     # room so that PowerPoint never wraps where Chrome did not
            $w = [double]$b.w * $K + $slack
            $x = [double]$b.x * $K
            if ($b.align -eq 'right') { $x -= $slack } elseif ($b.align -eq 'center') { $x -= $slack / 2 }
            $y = [double]$b.y * $K
            $h = [double]$b.h * $K
            $t = $sl.Shapes.AddTextbox(1, $x, $y, $w, $h)
            $f = $t.TextFrame
            $f.AutoSize = 0
            $f.WordWrap = -1
            $f.MarginLeft = 0; $f.MarginRight = 0; $f.MarginTop = 0; $f.MarginBottom = 0
            $f.VerticalAnchor = 1
            $text = ($b.lines | ForEach-Object { (($_ | ForEach-Object { $_.t }) -join '').TrimEnd() }) -join [string][char]11
            $r = $f.TextRange
            $r.Text = $text
            $r.ParagraphFormat.TextDirection = 1      # left to right, also when PowerPoint itself is installed in Hebrew
            $r.ParagraphFormat.Alignment = @{ left = 1; center = 2; right = 3 }[$b.align]
            $r.ParagraphFormat.SpaceWithin = [math]::Round([double]$b.lh / ([double]$b.fs * 1.2), 3)
            $pos = 1
            foreach ($ln in $b.lines) {
                $lineText = (($ln | ForEach-Object { $_.t }) -join '').TrimEnd()
                $left = $lineText.Length
                foreach ($run in $ln) {
                    $len = [math]::Min($run.t.Length, $left)
                    if ($len -le 0) { continue }
                    $c = $r.Characters($pos, $len)
                    $c.Font.Size = [math]::Round([double]$run.s * $K, 1)
                    $c.Font.Color.RGB = ConvertTo-Rgb $run.c
                    $c.Font.Name = if ($run.b) { 'Bahnschrift SemiBold' } else { 'Bahnschrift' }
                    $c.LanguageID = 1033
                    if ($run.i) { $c.Font.Italic = -1 }
                    $pos += $len; $left -= $len
                }
                $pos += 1
            }
            $t.Left = $x; $t.Top = $y; $t.Width = $w; $t.Height = $h
        }
        if ($s.notes) { $sl.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = $s.notes }
    }
    if (Test-Path $out) { Remove-Item $out -Confirm:$false }
    $pres.SaveAs($out, 24)
    $chk = Join-Path $src 'pptx_check'
    if (Test-Path $chk) { Remove-Item $chk -Recurse -Force -Confirm:$false }
    $pres.Export($chk, 'PNG', 960, 540)    # rendered back, for a visual check
    Write-Output "$out : $n slides"
}
finally {
    $pres.Close()
    if ($ppt.Presentations.Count -eq 0) { $ppt.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
}
