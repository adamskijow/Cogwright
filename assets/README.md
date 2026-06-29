<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Banner assets

`cogwright_banner.gif` is the README banner. The scene is `banner.html`, a
self-contained WebGL animation of a query lighting up the nearest chunks in an
embedding cloud.

To regenerate it you need [uv](https://docs.astral.sh/uv/) and `ffmpeg`:

```sh
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright python capture.py    # renders frames/ (one loop period)

ffmpeg -y -framerate 30 -i frames/f%04d.png \
  -vf "scale=1280:360:flags=lanczos,format=yuv420p" -c:v libx264 -crf 20 \
  -movflags +faststart cogwright_banner.mp4

ffmpeg -y -framerate 30 -i frames/f%04d.png \
  -vf "fps=20,scale=760:-1:flags=lanczos,palettegen=stats_mode=diff" /tmp/pal.png
ffmpeg -y -framerate 30 -i frames/f%04d.png -i /tmp/pal.png \
  -lavfi "fps=20,scale=760:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a" \
  cogwright_banner.gif
```

The frames are rendered deterministically over one loop period, so the GIF and
MP4 loop seamlessly.
