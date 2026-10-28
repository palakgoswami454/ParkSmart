# Phase 1 & 2 — Rubric analysis, problem refinement and dataset selection

## 1. Rubric-to-implementation matrix

The ESE (TE7483, 30 marks, 60% weight) awards *Excellent* (100%) for a response that is "complete, correct, technically sound, well-justified, and clearly presented".

| # | Component | Marks | What *Excellent* requires (rubric wording, condensed) | What in ParkSmart satisfies it | Evidence to show in the evaluation |
|---|---|---|---|---|---|
| 1 | Problem refinement & use-case relevance | 3 | Problem clearly refined; objective, scope, application domain, users/stakeholders and real-world relevance strongly explained | §2 below; report ch. 1–3; app *Overview* explains historical vs predicted vs live | Problem statement slide; stakeholder table; the "what ParkSmart cannot tell you" list |
| 2 | Dataset understanding, preprocessing & feature handling | 4 | Source, size, attributes, target, data types, missing values, preprocessing steps, feature selection/engineering **and validation strategy** | `data/README.md` (source, licence, schema, hash); `data_quality_report.json`; `src/data_preprocessing.py`; `src/feature_engineering.py` (16 documented features); chronological split + walk-forward folds | Data-quality numbers (216 duplicates, 12 negative counts, 373 over-capacity, 5.0% missing slots); coverage map; feature table; leakage test |
| 3 | Methodology & ML pipeline design | 5 | Complete pipeline: data prep, model selection, training strategy, evaluation method, **justification** of algorithms | 12-stage pipeline (README diagram); 3 baselines + 3 ML regressors, 2 baselines + 3 ML classifiers; each with a written justification in `src/models.py`; tune-on-validation → select-on-walk-forward → report-on-test | Pipeline diagram; "why this model" column; explanation of why test data never influences choices |
| 4 | Implementation / project execution | 5 | Complete prototype aligned with methodology; code, workflow, model execution, outputs, structure clear and **reproducible** | Modular `src/` package, CLI commands, saved models, 5-page Streamlit app, executed notebooks, `pytest` suite (11 tests incl. leakage and "reported metrics match saved model") | Live run of `pytest -q`; live app demo; repository tree |
| 5 | Results, metrics & comparative analysis | 5 | Accurate, clearly presented, analysed with suitable metrics (accuracy, P/R/F1, confusion matrix, RMSE, MAE…) | MAE/RMSE/R²/MAE-in-spaces for regression; accuracy, macro & weighted P/R/F1, per-class FULL/HIGH recall, confusion matrices for classification; walk-forward mean±std; peak/off-peak, weekday/weekend, per-car-park and unseen-car-park analysis; latency; interval coverage | Model comparison tables & charts (`outputs/metrics`, `outputs/figures`, app *Model performance*) |
| 6 | Final report & documentation | 3 | Professional, well-structured, technically clear, complete: problem, methodology, dataset, results, conclusion, future scope, **references** | `reports/ParkSmart_Report.md` / `.docx` (23 sections, verified references, real screenshots); README; data/models/outputs READMEs | The report itself |
| 7 | Communication & presentation | 2 | Clear, confident, structured, time-managed, effective visuals | `reports/presentation_script.md`: 10-slide flow with timings, speaker script, demo sequence, backup plan | Rehearsed 10–12 minute talk |
| 8 | Viva / project defence | 3 | Deep individual understanding of project, dataset, model, methodology, results, limitations, future scope | `reports/viva_preparation.md`: 40 questions with answers, incl. hard follow-ups on leakage, availability maths, model choice, what the system cannot guarantee | Your answers |

## 2. Refined problem statement

**Real-world problem.** Drivers arriving in a city centre cannot see how full a car park will be when they get there. Static signs show the situation *now*; by the time a driver arrives 30–60 minutes later the picture has changed. Drivers who find a full car park circle and queue, which wastes fuel and time and adds congestion. Shoup (2006) reviewed studies in which cruising for kerb parking accounted for a substantial share of traffic in congested central districts.

**Why historical data can predict the future.** Car-park occupancy is highly regular: it rises through the morning, peaks in the early afternoon and falls in late afternoon, with a strong weekly rhythm (our EDA shows weekday mean 53% vs weekend 39%). The current level and its recent trend, combined with what happened at the same time last week, carry most of the information about the next hour.

**Refined statement.** *Given the occupancy counts of a Birmingham car park up to time t, forecast its occupancy rate (and hence free spaces) at t + 30 min and t + 60 min, and classify the expected status as LOW / MODERATE / HIGH / FULL, for 28 council-reported car parks between 08:00 and 16:30.*

**Prediction vs real-time detection.** Real-time detection answers "how many spaces are free right now?" and needs sensors, cameras or a live feed. ParkSmart answers "how many will be free when I arrive?" from historical counts. Our data stopped in December 2016, so the app *replays* history: the user picks a moment, the model sees only data up to that moment, and the true outcome is shown alongside for checking. Nothing in ParkSmart is a live reading.

**Users and stakeholders.**

| Stakeholder | Need | What ParkSmart provides |
|---|---|---|
| Drivers / shoppers / commuters | Choose a car park and arrival time | Expected free spaces and status 30/60 min ahead, plus less-busy alternatives |
| Car-park operators | Staffing, opening overflow decks, switching "FULL" signs early | Forecasts with a 90% range; which car parks fill most often; when forecasts are least reliable |
| City council / traffic managers | Reduce cruising and queues, plan capacity | Network-wide view; peak windows; car parks persistently near capacity |
| Navigation / smart-city apps | An occupancy forecast API | `src/predict.py` returns a structured forecast |

**Limitations of conventional parking information.** Variable-message signs and apps show the *current* count (stale by arrival) or a fixed "typical busy times" chart that ignores today's conditions. Our two corresponding baselines — persistence and the historical profile — are exactly those approaches, so beating them is a direct test of added value.

**Why machine learning.** The future rate depends non-linearly on several interacting signals (current level, short-term trend, time of day, weekday, car-park identity, last week's shape). Hand-writing rules for 28 car parks × 18 slots × 7 weekdays is impractical; supervised learning fits these interactions from 2.5 months of examples and is evaluated objectively on unseen weeks.

**Scope and boundaries.**
* In scope: 28 car parks, 08:00–16:30, 4 Oct–19 Dec 2016; horizons 30 and 60 minutes; occupancy-rate regression (primary) and status classification; replay app.
* Out of scope: live data, overnight or pre-08:00 forecasts, events, weather, prices, on-street parking, routing.

**Objectives (measurable).**
1. Beat persistence and the historical-profile baseline in MAE on an unseen 2-week test period, for both horizons.
2. Also beat a strong hand-crafted rule (persistence + last week's shape) — otherwise ML is not justified.
3. Achieve high recall for FULL status (missing a full car park is the costliest error).
4. Provide an uncertainty range whose empirical coverage on the test period is close to its nominal 90%.
5. Deliver a working, reproducible app whose numbers come only from the saved model and evaluation files.

**Research questions.**
* RQ1: How much better than naïve and profile baselines can ML forecast car-park occupancy 30 and 60 minutes ahead?
* RQ2: Which inputs matter most — current occupancy, recent trend, time of day, or last week's pattern?
* RQ3: Is it better to predict the status class directly or to threshold a regression forecast?
* RQ4: Where (which car parks, which times) are forecasts least reliable, and does the model transfer to car parks it has not seen?

## 3. Dataset selection

| Candidate | Occupancy + capacity? | Facility ID & timestamps? | Documentation / licence | Fit for this project |
|---|---|---|---|---|
| **UCI Parking Birmingham** (ID 482) | Yes, both | Yes; ~30-min readings | UCI page, DOI 10.24432/C51K5Z, CC BY 4.0; used in published research (Stolfi et al., 2017; 2020) | **Selected.** Small enough to run on a laptop, large enough (35,717 readings) for proper temporal validation, has the capacity needed for "available spaces" |
| SFpark (San Francisco) | Block-level occupancy from the 2011–13 pilot | Yes | Pilot-era data; harder to obtain and document reliably today | Rejected: access and documentation uncertain; on-street blocks rather than car parks |
| City of Melbourne on-street bay sensors | Bay-level present/absent events | Yes | Open data portal | Rejected for a semester project: needs heavy aggregation to facility level, no "car park capacity" concept |
| Moscow parking occupancy (GitHub, 2025–) | Half-hourly snapshots, 210 lots | Yes | CC BY 4.0 (per repository description) | Promising future-scope dataset; newer and less established in the literature |

**Verification performed.** UCI page facts (35,717 instances, 4 features, 2016-10-04 → 2016-12-19, CC BY 4.0, "has missing values: yes") were read from the official page. Because the UCI download host was blocked from the build environment, the CSV was obtained from a public mirror and then checked against every one of those facts (see `data/README.md` and `tests/test_pipeline.py`). The file's SHA-256 is recorded so the user can compare with a fresh UCI download.

## 4. Prediction task design

| | Option A — classification | Option B — regression |
|---|---|---|
| Target | LOW / MODERATE / HIGH / FULL | occupancy rate in [0, 1] |
| Pros | Directly matches a sign or traffic-light display | Keeps all information; converts to free spaces; thresholds can be changed later without re-training |
| Cons | Throws away distance from thresholds; class imbalance (FULL ≈ 9%) | Needs conversion for a status display |

**Decision:** regression is the **primary** task (it gives free spaces, which the brief requires and which capacity makes possible); classification is a **complementary experiment**, comparing direct classifiers with thresholding the best regressor (RQ3).

* **Horizons:** 30 min (1 slot) and 60 min (2 slots). Readings arrive about every 30 min, so finer horizons are not supported by the data.
* **Target construction:** occupancy rate at slot t+h on the *same day* (16:30 → next 08:00 is not 30 minutes, so the overnight gap is never crossed).
* **Features available at prediction time:** only values observed at or before t, or on earlier days; plus calendar facts about the target time (known in advance). Enforced by a test that corrupts all future data and checks the features do not change.
* **Status thresholds:** LOW < 50%, MODERATE 50–75%, HIGH 75–90%, FULL ≥ 90%. FULL starts at 90% rather than 100% because drivers struggle to find the last few bays in a large multi-storey long before it is literally 100% full, and the raw counts include values above capacity (queuing cars), so "exactly 100%" is not a reliable signal.
* **Availability:** `estimated free spaces = max(0, capacity − round(predicted rate × capacity))`, capacity is known for every car park and never changes in the data.
