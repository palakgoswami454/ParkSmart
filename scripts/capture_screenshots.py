"""Capture screenshots of every ParkSmart page in light and dark mode.

Requires a running app (streamlit run app.py --server.headless true) and
Playwright (`pip install playwright && playwright install chromium`).
Dark mode is captured by emulating the browser's colour-scheme preference,
which Streamlit follows when both [theme.light] and [theme.dark] are defined.

    python scripts/capture_screenshots.py [--url http://localhost:8501] [--chromium /path/to/chrome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {
    "01_overview": "",
    "02_explore": "explore?facility=airport-mlcp",
    "03_find_parking": "find?dest=fc-road&vehicle=car",
    "04_city_insights": "insights",
    "05_forecast": "forecast",
    "06_model_performance": "models",
    "07_data_sources": "about",
}
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--chromium", default=None, help="optional path to a Chromium binary")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--plain-map", action="store_true", help="use the offline map background (no tile download)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if a.chromium:
            kw["executable_path"] = a.chromium
            kw["args"] += ["--single-process", "--no-zygote", "--disable-gpu"]
        browser = p.chromium.launch(**kw)
        for theme in a.themes.split(","):
            page = browser.new_page(viewport={"width": 1440, "height": 2400}, color_scheme=theme, locale="en-IN")
            for name, path in PAGES.items():
                url = f"{a.url}/{path}"
                if a.plain_map:
                    url += ("&" if "?" in url else "?") + "basemap=plain"
                page.goto(url, wait_until="networkidle")
                page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
                page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
                if name == "03_find_parking":  # submit the form so results show
                    page.get_by_role("button", name="Show parking options").click()
                    page.wait_for_timeout(1500)
                    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
                                           timeout=120000)
                page.wait_for_timeout(2500)
                box = page.locator("[data-testid='stMainBlockContainer']").bounding_box()
                height = int(min(page.viewport_size["height"], box["y"] + box["height"] + 30))
                out = OUT / f"{name}_{theme}.png"
                page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1440, "height": height})
                print("saved", out)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
