# ParkSmart 2.0: Pune upgrade notes, verification log and rubric checklist

## 1. What changed and why

ParkSmart 1.0 was a Birmingham-only forecasting app. ParkSmart 2.0 makes Pune the focus **without** pretending Birmingham data describes Pune.

| Area | Before | After |
|---|---|---|
| Geography | Birmingham car parks only | Pune directory (19 sourced facilities, 16 destinations, dated policy timeline); Birmingham kept as the clearly labelled model-training dataset |
| User question | "How full will this Birmingham car park be?" | "Where should I park in Pune, and what can I expect?" |
| Decision support | — | *Find parking*: transparent score (0.6 × proximity + 0.4 × evidence), vehicle, budget and tariff pricing, explicit "still unknown" list |
| Uncertainty | one global ±width | time-of-day split-conformal widths (better conditional coverage) |
| Transfer to new places | unseen-car-park experiment only | park-agnostic model + **test-on-your-own-data** tool: forecasts are only shown if the model beats persistence on the user's data |
| UI | single-file app, light only, static PNG charts | 7-page app, custom **light and dark** palettes via Streamlit's native theming, interactive theme-aware Plotly charts, offline map mode, bundled fonts |
| Tests | 11 | 40 (directory integrity, ranking, tariffs, invalid input, intervals, transfer, every page, form end-to-end) |

### Files

**New:** `pages/` (7 pages), `src/data_loader.py`, `src/location_services.py`, `src/recommendations.py`, `src/intervals.py`, `src/transfer.py`, `src/ui.py`, `src/visualizations.py`, `data/pune/{facilities,destinations,policy,sources}.json` + README, `data/samples/sample_unseen_carpark.csv`, `models/parksmart_transfer_h{30,60}.joblib`, `outputs/metrics/transfer_summary.json`, `assets/` (logo), `static/fonts/` (Barlow, OFL), `tests/test_pune.py`, `notebooks/04_pune_directory_and_transfer.ipynb`, this document.

**Modified:** `app.py` (now the router + sidebar), `.streamlit/config.toml` (light/dark design system), `src/train.py` (stores time-of-day interval table), `src/predict.py` (uses it), `src/evaluate.py` (band coverage report), `src/data_preprocessing.py` (`clean(min_obs=…)` for uploads), `models/parksmart_h{30,60}.joblib` (interval table added; the fitted models are unchanged), `outputs/metrics/*`, `scripts/capture_screenshots.py` (light + dark), README, report, presentation script, viva prep.

**Unchanged:** the Birmingham data pipeline, features, split, candidate models, tuning grids and all reported test scores.

## 2. Pune data: what was searched, found and used

**Searched for occupancy data:** PMC open-data portal, data.gov.in, OpenCity, Kaggle and GitHub listings, news. **Nothing public exists** for any Pune car park. The data.gov.in catalog "Parking lots: Pune" exists but blocks automated access. The two links supplied in the brief (PMC portal, Smart City PDF) could not be fetched from the build environment and are not relied on.

**Used (all in `data/pune/sources.json`):**

* Pune Metro parking announcement, 15 Feb 2024 (Republic World, Punekar News, Free Press Journal): 9 stations, rates, 25% metro-ticket discount.
* Contradicting reports: Dec 2024 (Punekar News, PMC ₹5 crore plan; Swargate restricted) and Sep 2025 (FPJ, no parking in the PCMC area).
* Central Railway Pune Division pay-and-park register, 15 Jul 2025 (official PDF).
* Pune airport multi-level car park (Punekar News, Sep 2022; Infrastructure Today) and the 15-minute terminal rule (Navbharat Live, Jun 2026).
* Pune Cantonment Board rates at Camp (Punekar News, Jul 2026, quoting the CEO).
* PMC Public Parking Policy 2016 (official PDF); on-street pilot tender (Mar 2026); proposed zone rates (Sep 2026); BOT multi-level plans (Jun 2026).
* Coordinates: 20 Wikipedia articles (stations, landmarks, localities), magicpin listing (Narayan Peth lot), SpotMyPark plus code (Mandai lot).

## 3. Verification log (Part 8 of the brief)

| # | Check | Result |
|---|---|---|
| 1 | Run the application | Started with `streamlit run app.py`; health endpoint OK; no tracebacks in the server log after visiting every page in light and dark mode |
| 2 | Every navigation page | `tests/test_pune.py::test_page_runs_without_errors` runs all 7 pages headlessly. Screenshots of all 7 in both themes are in `reports/screenshots/` |
| 3 | Theme switching | Dark/light both render from `[theme.light]`/`[theme.dark]`; captured by emulating the browser colour scheme. Switching is via ⋮ → Settings → Theme (persisted by the browser). A separate in-page toggle was **not** built: Streamlit has no supported API to change the theme from Python, and CSS/JS hacks break across versions |
| 4 | UI in both themes | Reviewed screenshots: charts, tables, badges, alerts and maps readable in both |
| 5 | Pune directory traceable | Loader rejects any fact without a source; test checks every facility; Data sources page lists all 40 with dates |
| 6 | Search and filters | Explore search/type/vehicle/status/charges/mapped filters exercised manually; destination search and ranking filters unit-tested |
| 7 | Predictions and invalid input | Forecast replay, transfer upload (good sample, missing columns, too little data, zero capacity) and recommendation input validation tested |
| 8 | Metrics and charts use actual outputs | Charts read `outputs/metrics` and `outputs/predictions`; `test_reported_test_metrics_match_saved_model` recomputes the headline MAE from the saved model |
| 9 | Automated tests | `pytest -q` → **40 passed** |
| 10 | No Birmingham location shown as Pune | `test_no_birmingham_names_in_pune_directory`; Pune and Birmingham pages carry different mode badges; every forecast page shows a "This is not Pune data" notice |
| 11 | No unverified availability shown as live | Facility schema has no occupancy field (tested); sidebar shows "Live data: not connected"; facility pages state "Occupancy forecast: not available" |
| 12 | Unverifiable features | Street map tiles and the deployment URL could not be verified from the build machine (tiles blocked; no Streamlit account). Offline map mode was verified instead |

## 4. Actual ML results (unchanged pipeline, new uncertainty and transfer results)

* 60-min test MAE: HGB **2.02 pp** (≈ 25 spaces) vs persistence 7.07, historical profile 7.24, weekly-shape rule 2.79, Ridge 2.83, RF 2.29. 30-min: **1.26 pp**.
* Status (thresholded HGB, 60 min): accuracy 0.932, macro-F1 0.915, FULL recall 0.880, precision 0.932.
* 90% range, time-of-day bands (60 min): half-widths ±5.19 / ±4.72 / ±3.99 / ±4.13 pp for 08:30–10:30 / 11:00–12:30 / 13:00–14:30 / 15:00–16:30. Test coverage 89.8% overall; per band 87.6–90.9%, against 86.4–93.0% with one global width.
* Transfer model (car-park ID hidden), held-out car park Others-CCCPS8: 60 min 1.75 pp vs persistence 5.28 pp; 30 min 1.10 vs 2.69 pp. The held-out park's range coverage was 97.7% (conservative, because widths come from car-park-level CV). Caveat: the held-out car park is new to the model but its dates overlap the training period, so shared calendar effects (e.g. the pre-Christmas rise) make this check easier than a new city in a new year would be.

## 5. Rubric checklist (AUCML ESE, 30 marks)

| Component | Marks | Where ParkSmart 2.0 shows it |
|---|---|---|
| Problem refinement & use case | 3 | User question reframed for Pune; stakeholders; explicit split of documented vs predicted vs unavailable (README, About page, report §2–4, §17) |
| Dataset understanding & preprocessing | 4 | Birmingham audit and cleaning (unchanged); Pune data provenance, evidence statuses, coordinate cross-checks, validation rules (`data/pune/README.md`, notebook 04) |
| Methodology & pipeline | 5 | Leakage-safe pipeline, walk-forward selection; conformal intervals by time band; park-agnostic transfer with car-park-level CV; transparent ranking formula |
| Implementation | 5 | Modular `src/`, 7-page app, saved models, 40 tests, executed notebooks, reproducible commands, offline-capable |
| Results & comparison | 5 | Model comparison vs three baselines, classification metrics, confusion matrices, segment/park errors, coverage by band, transfer results |
| Report & documentation | 3 | Updated report (.md/.docx) with Pune chapter and new results; this document; data READMEs |
| Presentation | 2 | Updated script with Pune demo flow and offline backup |
| Viva | 3 | `reports/viva_preparation.md`, now with a Pune / ParkSmart 2.0 section |

## 6. Remaining limitations

* Directory coverage is small and dated; many real Pune car parks are absent because no verifiable source was found.
* Metro-station parking status is uncertain (announced, never confirmed, partly contradicted).
* No routing: distances are straight-line between cited points.
* The transfer check is validated on one held-out Birmingham car park and by car-park CV; it has never seen Pune data.
* The app has not been deployed; the street-map tiles were not visible on the build machine.
