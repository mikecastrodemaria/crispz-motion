# crispz-motion lot 2 - the key tools on the card, through the command line, timed.
# Run after tools/bench_lot2.py levers (it uses the projects that one made):
#   powershell -ExecutionPolicy Bypass -File tools\bench_lot2_tools.ps1
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$out = "bench\lot2"
function Step($name, [scriptblock]$b) {
    $t = Get-Date
    & $b
    "{0}: {1:N1} s (exit {2})" -f $name, ((Get-Date) - $t).TotalSeconds, $LASTEXITCODE | Tee-Object -Append "$out\tools_times.txt"
}
Remove-Item "$out\tools_times.txt" -ErrorAction SilentlyContinue
$edit = (Get-ChildItem projects -Directory -Filter "*lot2-c-edit*" | Sort-Object Name | Select-Object -Last 1).Name
$d625 = (Get-ChildItem projects -Directory -Filter "*lot2-a-d625*" | Sort-Object Name | Select-Object -Last 1).Name

# 1. assisted middle key on the edit pair the guard flagged (distance 39.6)
Step "mid $edit k000 k001" { & $py cm_cli.py mid $edit k000 k001 }
& $py cm_cli.py guard $edit | Tee-Object -Append "$out\tools_times.txt"

# 2. an edit that asks to keep the style
Step "edit keep-style" { & $py cm_cli.py edit $d625 "turn his head to the left, keep the black and white pencil drawing style" --from k000 --at 3 }

# 3. inpaint of the smoke area (upper right quarter of the frame)
& film\.venv\Scripts\python.exe -c "from PIL import Image, ImageDraw; m=Image.new('L',(1088,1920),0); ImageDraw.Draw(m).rectangle((560,80,1088,700),fill=255); m.save(r'$out\mask_smoke.png')"
Step "inpaint smoke" { & $py cm_cli.py inpaint $d625 --mask "$out\mask_smoke.png" --prompt "thick swirling pencil-drawn smoke clouds" --from k002 }

# 4. regeneration with cascade: k001 with a new seed, k002 rebuilt from it
Step "regen k001 cascade" { & $py cm_cli.py regen $d625 k001 --seed 99 --cascade }
& $py cm_cli.py info $d625 | Tee-Object -Append "$out\tools_times.txt"

# 5. order a in 4K (the engine upscales the keys), on the 3 chain keys
$a = (Get-ChildItem projects -Directory -Filter "*lot2-a-d750*" | Sort-Object Name | Select-Object -Last 1).Name
Step "render keys_first 4k" { & $py cm_cli.py render $a --res 4k --order keys_first --loop pingpong }
Copy-Item "projects\$a\video.mp4" "$out\keys_first_4k.mp4" -Force
Step "render film_first 4k" { & $py cm_cli.py render $a --res 4k --order film_first --loop pingpong }
Copy-Item "projects\$a\video.mp4" "$out\film_first_4k.mp4" -Force
& $py cm_cli.py engine stop
