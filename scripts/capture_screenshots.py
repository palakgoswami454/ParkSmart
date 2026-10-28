"""Capture full-page screenshots of every ParkSmart page for the report.

Requires a running app (streamlit run app.py --server.headless true) and
Playwright (`pip install playwright && playwright install chromium`).

    python scripts/capture_screenshots.py [--url http://localhost:8501] [--chromium /path/to/chrome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {"01_overview": "", "02_availability": "availability", "03_forecasting": "forecasting",
         "04_model_performance": "models", "05_insights": "insights"}
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--chromium", default=None, help="optional path to a Chromium binary")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if a.chromium:
            kw["executable_path"] = a.chromium
            kw["args"] += ["--single-process", "--no-zygote", "--disable-gpu"]
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1440, "height": 2600}, device_scale_factor=1)
        for name, path in PAGES.items():
            page.goto(f"{a.url}/{path}", wait_until="networkidle")
            page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
            # wait until Streamlit has finished running the script
            page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
            page.wait_for_timeout(2500)
            box = page.locator("[data-testid='stMainBlockContainer']").bounding_box()
            height = int(min(page.viewport_size["height"], box["y"] + box["height"] + 30))
            page.screenshot(path=str(OUT / f"{name}.png"),
                            clip={"x": 0, "y": 0, "width": page.viewport_size["width"], "height": height})
            print("saved", OUT / f"{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
