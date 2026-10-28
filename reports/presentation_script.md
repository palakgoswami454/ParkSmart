# ParkSmart — presentation outline, speaker script and demo plan

> **ParkSmart 2.0 update.** The talk below is still valid for the ML part. Add the Pune slides and demo steps in the next section; they replace the old 3-minute demo.

## ParkSmart 2.0 additions (insert after slide 3; demo replaces slide 8)

**New slide 3b — "Pune: what data exists" (1 min).** "I wanted this for Pune. I searched PMC open data, data.gov.in and news: there is no public occupancy data for any Pune car park. So I split the problem. For Pune, ParkSmart documents what can be verified: 19 facilities, each with a dated source and an evidence label. For forecasting, I built and tested the model where data exists, Birmingham. A transfer check decides when it may be used on Pune counts."

**Demo (4 min):**
1. *Overview*: map coloured by evidence, not occupancy; "what's live, historical, missing" panel. Switch to dark mode (⋮ → Settings → Theme).
2. *Find parking*: FC Road, car, 2 h → ranked options with score breakdown and ₹ cost; note the FC Road pay-and-park pilot; open *How the ranking works*.
3. *Explore*: Airport multi-level car park → capacity, sources, "Occupancy forecast: not available".
4. *City insights*: metro parking announced (9) vs confirmed (0).
5. *Occupancy forecast*: Broad Street replay (19 free, actual +0.4 pp), then *Test on your own data* → sample → model 1.75 pp vs persistence 5.28 pp → forecasts shown only because it wins.
6. *Model performance*: comparison chart and time-of-day coverage chart.

**Backup:** if the venue has no internet, turn off **Street map** in the sidebar (fonts are bundled). If the app fails, use `reports/screenshots/*_light.png` / `*_dark.png` and run `pytest -q` (40 passed).


Target length: **10 minutes talk + 3 minutes live demo**, then viva. All numbers below come from `outputs/metrics/` and match the report and the app.

| # | Slide | Time | Visual |
|---|---|---|---|
| 1 | The problem | 0:45 | Photo-free title slide: "Will there be a space when I arrive?" + one line on cruising (Shoup, 2006) |
| 2 | Why current information is not enough | 0:45 | Two small panels: "sign shows now" vs "you arrive in 60 min"; persistence vs actual line from `eval_example_day_h60.png` |
| 3 | Objectives and contribution | 0:45 | 5 objectives, 4 research questions |
| 4 | Dataset and preprocessing | 1:30 | Data table (source, licence, size), cleaning table, `eda_06_missing_data_map.png` |
| 5 | What the data shows | 1:00 | `eda_02_hour_of_day_profile.png`, `eda_05_daily_trend.png` |
| 6 | ML pipeline and algorithm choice | 1:30 | Pipeline diagram (report §10), feature table, model table |
| 7 | Results and comparison | 1:45 | `eval_model_comparison_h60.png`, Table 1 (60 min), `eval_confusion_h60.png` |
| 8 | Live demo | 3:00 | Streamlit app |
| 9 | Insights, limitations, future scope | 1:00 | `eval_error_by_hour_h60.png`, limitations bullets |
| 10 | Conclusion | 0:30 | Three headline numbers: 2.02 pp (≈ 25 spaces), 71% better than persistence, 90.1% range coverage |

## Speaker script

**Slide 1 — Problem.** "When you drive into a city centre you pick a car park based on what the sign says now. But you arrive half an hour or an hour later, and by then it may be full. Drivers then circle and queue — studies reviewed by Shoup found cruising for parking can be a substantial share of city-centre traffic. ParkSmart asks: *how full will this car park be when I get there?*"

**Slide 2 — Why existing information is insufficient.** "There are two common answers today. A live sign shows the present — that is our *persistence* baseline. A 'popular times' chart shows the average — our *historical profile* baseline. Here is Broad Street on 6 December: the sign's value, the dashed line, lags the real morning rise by a full hour. Neither approach combines what is happening today with the usual pattern, which is what a model can do."

**Slide 3 — Objectives.** "We set five measurable goals: beat both baselines on unseen weeks; also beat a strong hand-made rule, otherwise ML isn't justified; detect FULL car parks reliably; give an honest uncertainty range; and build a reproducible app. One clarification up front: our data is from 2016, so the app *replays* history — it never claims to be live."

**Slide 4 — Dataset.** "We use the UCI Parking Birmingham dataset: 35,717 readings from 30 council car parks, 4 October to 19 December 2016, CC BY 4.0. It has occupancy counts *and* capacity, which is what lets us talk about free spaces. It has no empty cells, but it does have problems: 216 duplicates, 12 negative counts, 373 counts above capacity, four days with no data and two car parks that barely report. We removed duplicates, clipped impossible values, dropped the two sparse car parks, and snapped readings onto a 30-minute grid. We never interpolated — a made-up value could end up as a training target. 5% of the grid is missing, and this map shows where."

**Slide 5 — EDA.** "Occupancy climbs from 8 a.m. to a peak at 2 p.m. — 62% on average — then falls. Weekdays average 53%, weekends 39%. And demand rises towards Christmas: the test fortnight is six points busier than training. That's why we split by time: a random split would hide this drift."

**Slide 6 — Pipeline.** "Each example is: a car park at a moment *t*; the target is its occupancy 30 or 60 minutes later on the same day. Features use only what's known at *t*: current level, the last two readings and the trend, time of day, weekday, car-park ID, and last week's values at the same times. We have an automated test that corrupts all future data and checks the features don't change — that's our leakage guarantee. We tune on the validation fortnight, *select* using four walk-forward weeks, and touch the final test fortnight exactly once. We compare three baselines with Ridge, Random Forest and HistGradientBoosting — linear versus two kinds of tree ensemble."

**Slide 7 — Results.** "Sixty minutes ahead, persistence is off by 7.1 percentage points — about 97 spaces. Our strong rule, 'now plus last week's change', gets 2.8. HistGradientBoosting gets 2.0 points — about 25 spaces — 71% better than persistence and 27% better than the rule. Notice Ridge does *not* beat the rule: the gain comes from non-linear interactions. For status, thresholding the regression forecast beat every direct classifier: 93% accuracy, macro-F1 0.915. Of 734 truly FULL cases, only 2 were called LOW or MODERATE; the rest of the misses were 'HIGH', a mild error. And our ±4.5-point range covered 90.1% of test outcomes, so it's honest."

**Slide 8 — Demo.** (see sequence below)

**Slide 9 — Insights and limits.** "The hardest times to forecast are 10 to 11:30, when car parks fill fastest, and weekends. Three car parks are full in over a quarter of slots. The model also works on car parks it has never seen: 2.7 points. Limits: no live feed, one autumn of data, no events or weather, and it needs a current reading to forecast."

**Slide 10 — Conclusion.** "ParkSmart turns a public dataset into a 60-minute forecast accurate to about 25 spaces, with a calibrated range and a clear app — and it's fully reproducible from the GitHub repository with one test command proving the reported numbers come from the saved model."

## Live demo sequence (3 minutes)

Start the app before the session: `streamlit run app.py`, open http://localhost:8501, zoom the browser to 110%.

1. **Overview (20 s).** Point at the replay banner: "historical, predicted, never live". Show 28 car parks, 73 days, the Christmas rise on the daily chart.
2. **Parking availability (70 s).** Car park *Broad Street*, date *2016-12-06*, time *11:00*, look-ahead *60 min*. Read the sign: "19 spaces expected free of 690, FULL". Show the calculation line and the replay check (+0.4 pp). Scroll to the table: "if Broad Street is full, Others-CCCPS119a is forecast to have 2,126 free". Switch to *30 min* to show the horizon changes the forecast.
3. **Forecasting (40 s).** Broad Street, *Wed 14 Dec*: the 60-min forecast tracks the black actual line inside the shaded range; show the weekday × hour heatmap.
4. **Model performance (40 s).** Table: baselines vs models; confusion matrix; "FULL missed as LOW/MODERATE: 2". Toggle to 30 min.
5. **Insights (10 s).** Data-generated recommendations and "what ParkSmart cannot tell you".

## Backup plan

* **App will not start:** run `pytest -q` (shows the pipeline works) and `python -m src.predict --park "Broad Street" --at "2016-12-14 12:00" --horizon 60` in a terminal — it prints the same forecast the app would show.
* **No laptop / projector issue:** use the five screenshots in `reports/screenshots/` (insert them as hidden backup slides).
* **Questions about a number:** every figure is in `outputs/metrics/evaluation_summary.json`; notebook 03 recomputes the headline MAE from the saved model.
