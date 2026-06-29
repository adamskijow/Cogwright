# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors

import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
FRAMES = HERE / "frames"
FRAMES.mkdir(exist_ok=True)
for old in FRAMES.glob("*.png"):
    old.unlink()

W, H, P, FPS = 1280, 360, 8.0, 30
n = int(P * FPS)

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
        ],
    )
    page = browser.new_context(
        viewport={"width": W, "height": H}, device_scale_factor=2
    ).new_page()
    page.goto((HERE / "banner.html").as_uri())
    page.wait_for_function("window.__ready === true", timeout=15000)
    gl = page.evaluate(
        "(() => { const c = document.createElement('canvas');"
        " const g = c.getContext('webgl'); return g ? g.getParameter(g.VERSION) : 'NO-WEBGL'; })()"
    )
    print("webgl:", gl)
    if gl == "NO-WEBGL":
        sys.exit(2)
    page.evaluate("window.__capture = true")
    for i in range(n):
        t = i * P / n
        page.evaluate("(t) => window.renderAt(t)", t)
        page.locator("#cwc").screenshot(path=str(FRAMES / f"f{i:04d}.png"))
    browser.close()

print("frames:", len(list(FRAMES.glob("*.png"))))
