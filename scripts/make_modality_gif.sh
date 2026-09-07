#!/usr/bin/env bash
# Render docs/images/cuhk-x-modalities.gif from docs/videos/all_modality.mp4.
#
# Re-tiles the 7-panel demo video (RGB on top, IR/Thermal/Depth and
# Radar/Skeleton/IMU below) into a README banner: RGB on the left, the six
# sensor modalities in a 3x2 grid on the right. Needs ffmpeg and Python 3
# with Pillow (labels are rendered with Pillow because Homebrew ffmpeg ships
# without the drawtext filter).
#
# Usage: scripts/make_modality_gif.sh [start_sec] [duration_sec]
set -euo pipefail
cd "$(dirname "$0")/.."

START="${1:-40}"; DUR="${2:-7}"
SRC=docs/videos/all_modality.mp4
OUT=docs/images/cuhk-x-modalities.gif
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# Panel crops measured on the 1440x1080 source: x y w h.
RGB="443 0 554 415"
IR="0 462 432 280";     THERMAL="504 459 432 271";  DEPTH="1019 452 421 278"
RADAR="9 809 395 231";  SKELETON="496 798 403 242"; IMU="1011 799 426 241"

W=224; H=168; G=8                 # grid cell size and gutter
RH=$((2*H+G)); RW=$((RH*554/415))  # RGB panel spans both rows

python3 - "$TMP" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 17)
for name in ['RGB', 'IR', 'Thermal', 'Depth', 'Radar', 'Skeleton', 'IMU']:
    w, h = font.getbbox(name)[2:]
    im = Image.new('RGBA', (w + 14, h + 12), (0, 0, 0, 150))
    ImageDraw.Draw(im).text((7, 4), name, font=font, fill='white')
    im.save(f'{out}/{name}.png')
PY

cell() { set -- $1; echo "crop=$3:$4:$1:$2,scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:white"; }
FC="[0:v]split=7[a][b][c][d][e][f][g];
[a]crop=554:415:443:0,scale=${RW}:${RH}[a1];[a1][1:v]overlay=8:H-h-8[rgb];
[b]$(cell "$IR")[b1];[b1][2:v]overlay=8:H-h-8[ir];
[c]$(cell "$THERMAL")[c1];[c1][3:v]overlay=8:H-h-8[th];
[d]$(cell "$DEPTH")[d1];[d1][4:v]overlay=8:H-h-8[dp];
[e]$(cell "$RADAR")[e1];[e1][5:v]overlay=8:H-h-8[ra];
[f]$(cell "$SKELETON")[f1];[f1][6:v]overlay=8:H-h-8[sk];
[g]$(cell "$IMU")[g1];[g1][7:v]overlay=8:H-h-8[imu];
[rgb][ir][th][dp][ra][sk][imu]xstack=inputs=7:fill=white:layout=0_0|w0+${G}_0|w0+w1+2*${G}_0|w0+w1+w2+3*${G}_0|w0+${G}_h1+${G}|w0+w1+2*${G}_h1+${G}|w0+w1+w2+3*${G}_h1+${G}[tile];
[tile]fps=12,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
FC="$(echo "$FC" | tr -d '\n')"

ffmpeg -y -v error -ss "$START" -t "$DUR" -i "$SRC" \
  -i "$TMP/RGB.png" -i "$TMP/IR.png" -i "$TMP/Thermal.png" -i "$TMP/Depth.png" \
  -i "$TMP/Radar.png" -i "$TMP/Skeleton.png" -i "$TMP/IMU.png" \
  -filter_complex "$FC" -loop 0 "$OUT"
ls -la "$OUT"
