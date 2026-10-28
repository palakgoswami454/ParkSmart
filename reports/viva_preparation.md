# ParkSmart — viva preparation (40 questions)

Numbers refer to the 60-minute horizon unless stated; all come from `outputs/metrics/`.

## A. Problem statement and use case

**1. What problem does ParkSmart solve?**
It forecasts how full each of 28 Birmingham car parks will be 30 and 60 minutes ahead, converts that into expected free spaces using known capacity, and labels the status LOW/MODERATE/HIGH/FULL, so drivers and operators can act before a car park fills.

**2. How is that different from a "real-time parking availability" system?**
Real-time detection counts free spaces now using sensors or cameras. ParkSmart predicts the *future* from historical counts. It has no live feed; the 2016 data is replayed, and every number is labelled as an estimate.

**3. Who benefits?**
Drivers (choose car park/arrival time), operators (staffing, overflow, FULL signs), the council (reduce queues and cruising), and navigation apps (an API-style forecast).

**4. Why is ML appropriate rather than fixed rules?**
Future occupancy depends non-linearly on interacting factors — current level, trend, time, weekday, car-park identity, last week's shape. We tested this directly: a hand-crafted rule (persistence + weekly shape) reached 2.79 pp MAE, a linear model 2.83 pp, while HistGradientBoosting reached 2.02 pp. The gain comes from learned non-linear interactions.

**5. How did you refine the Phase-I problem?**
Availability became a derived quantity (capacity − predicted occupancy); the horizon was fixed by the 30-minute data frequency; prediction was separated from live detection; and success was defined relative to baselines that represent current practice.

## B. Dataset and preprocessing

**6. Describe the dataset.**
UCI *Parking Birmingham* (ID 482), Stolfi 2017, CC BY 4.0, DOI 10.24432/C51K5Z. 35,717 rows; 4 columns: SystemCodeNumber, Capacity, Occupancy, LastUpdated; 30 car parks; 4 Oct – 19 Dec 2016; readings about every 30 minutes, 08:00–16:30.

**7. Were there missing values? UCI says yes but the file has no empty cells.**
Correct — no empty cells, but missing *readings*: 4 whole days with no data, sensor gaps in some car parks, and two car parks with very few readings. After building the regular grid, 5.02% of slots are missing.

**8. What cleaning did you do and why?**
Removed 216 exact duplicates; clipped 12 negative counts to 0; capped the rate at 1.0 for 373 over-capacity readings; dropped two car parks with < 500 readings; snapped readings to the nearest 30-minute slot (within 10 minutes), averaging 17 collisions.

**9. Why not interpolate the missing slots?**
An interpolated value could become a training *target*, so we would be training and scoring on invented data. Tree models handle missing lag features natively; for Ridge we use median imputation plus missing-indicator columns.

**10. Why convert to a 30-minute grid?**
Raw timestamps alternate 27/33 minutes. A regular grid makes "one step ahead" always mean 30 minutes, lets lags line up exactly and exposes gaps explicitly.

**11. How did you verify the dataset since UCI was blocked?**
The copy came from a public mirror; we checked row count, columns, number of car parks and date range against the UCI page, recorded the SHA-256, and an automated test repeats the check. Instructions to re-download from UCI are in `data/README.md`.

## C. Feature engineering and leakage

**12. List your main features.**
rate_now, rate_lag1, rate_lag2, rate_delta1, rolling mean/std of the last three slots, origin/target slot and hour, day_of_week, is_weekend, park_code, capacity, rate at the target time yesterday, and rates at the origin and target times on the same weekday last week.

**13. How do you guarantee there is no leakage?**
(1) Lags are computed within a day, never across the overnight gap. (2) The target is a later slot on the same day. (3) Day-lags only look at earlier days. (4) The chronological split keeps all test dates after all training dates. (5) A test overwrites every value after a cut-off with a constant and asserts that features of earlier origins are identical. (6) The historical-profile baseline is fitted on training rows only.

**14. Isn't using `target_hour` leakage, since it's about the future?**
No — it is the clock time we are forecasting for, which is known in advance. Leakage means using the future *value*, not the future *time*.

**15. Why "same weekday last week" features?**
EDA showed a strong weekly cycle. Last week's change between the origin and target times tells the model the usual shape of the next hour; it is the basis of our strongest baseline.

**16. Why is `park_code` treated as categorical?**
IDs have no order. HistGradientBoosting handles it as a native categorical; Ridge gets one-hot encoding; Random Forest uses the integer, which trees can still split on.

## D. Classification versus regression

**17. Why is regression the primary task?**
It keeps all information, gives free spaces directly, and thresholds can change without re-training. Classification would discard how close a car park is to a boundary.

**18. Then why also do classification?**
Users and signs think in statuses. We compared direct classifiers with thresholding the best regressor (RQ3).

**19. Which won and why?**
Thresholding the HGB regressor: macro-F1 0.915 vs 0.899 for the best direct classifier (Random Forest). The regressor is trained on the continuous target, so it uses information that the four labels throw away, and one model serves both outputs consistently.

**20. Why 90% for FULL, not 100%?**
Drivers struggle to find the last few bays well before 100%, and the counts include over-capacity values (queuing), so exactly 100% is unreliable. Thresholds are in one place (`src/config.py`) and can be changed.

**21. How are available spaces calculated?**
occupied = round(predicted rate × capacity); free = max(0, capacity − occupied). E.g. Broad Street, 690 spaces, predicted 97.2% → 671 occupied → 19 free. Only possible because capacity is in the data and constant.

## E. Algorithm selection

**22. Why these models?**
Three baselines representing current practice and a strong rule; Ridge as an interpretable linear model; Random Forest (bagging) and HistGradientBoosting (boosting) as the two main tree-ensemble families for tabular data. A small, justified set rather than many algorithms.

**23. Why HistGradientBoosting in the end?**
Lowest walk-forward MAE (2.06 pp) — that is the selection criterion. It also had the lowest test MAE (2.02), fits in ~4 s, predicts ~21 ms per 1,000 rows, handles missing values and categoricals natively, and the saved model is under 2 MB.

**24. Difference between Random Forest and gradient boosting?**
RF trains many deep trees independently on bootstrap samples and averages them (reduces variance). Boosting adds shallow trees sequentially, each fitting the residual errors of the current ensemble (reduces bias); learning rate and number of iterations control it.

**25. Why no deep learning (LSTM)?**
~25k training rows and short, structured inputs; tree ensembles are the right first choice, and a 2 pp MAE leaves little headroom. Deep learning would add cost and tuning without clear benefit — listed as future work once more data exists.

## F. Training and validation

**26. Describe your split.**
Train 4 Oct – 21 Nov, validation 22 Nov – 5 Dec, test 6 – 19 Dec 2016. Chronological, no shuffling.

**27. What is walk-forward validation and why use it?**
Four expanding-window folds: each week from 6 Nov to 5 Dec is predicted by a model trained only on earlier dates. It mimics deployment, gives a mean ± sd, and is used to *select* the model so the test set is never used for decisions.

**28. Why not k-fold cross-validation with shuffling?**
It would train on the future to predict the past, and neighbouring 30-minute slots are almost identical, so scores would be inflated.

**29. How were hyper-parameters tuned?**
Grid search: fit on training, score on validation (MAE or macro-F1). E.g. HGB: learning rate {0.03, 0.1}, iterations {300, 600}, leaves {15, 31, 63}, L2 {0, 1} — 24 combinations. Best: 0.03, 600, 63, 0.

**30. How did you handle class imbalance?**
FULL is ~9% of slots. We tuned `class_weight` (balanced was chosen for every classifier) and report macro-F1 and per-class FULL/HIGH recall instead of only accuracy.

## G. Evaluation metrics

**31. Interpret MAE = 2.02 pp.**
On average the forecast occupancy rate is 2.02 percentage points off — about 25 spaces when weighted by each car park's capacity.

**32. Why report RMSE as well?**
RMSE penalises large errors more. HGB has RMSE 3.00 vs 4.85 for the weekly-shape rule, although their MAEs differ less: the rule makes occasional big mistakes.

**33. Which classification error is most costly?**
Predicting a FULL car park as LOW/MODERATE — the driver goes there and queues. Of 734 FULL cases, the model called only 2 LOW/MODERATE; 86 were called HIGH, a mild error. False FULL alarms (47) only nudge drivers to alternatives.

**34. How is the 90% range computed and is it valid?**
Half-width = 90th percentile of absolute out-of-fold walk-forward residuals (each from a model trained only on earlier data) — a split-conformal-style method. ±4.50 pp at 60 min. On the unseen test period it covered 90.1% of outcomes, so it is calibrated. Limitation: it is the same width at every time.

## H. Results and comparison

**35. Summarise the comparison.**
60 min test MAE: persistence 7.07, historical profile 7.24, persistence + weekly shape 2.79, Ridge 2.83, RF 2.29, HGB 2.02. 30 min: 3.67, 7.10, 1.71, 1.62, 1.37, 1.26.

**36. Why is the historical profile as bad as persistence?**
It ignores today. Daily levels shift (e.g., the Christmas rise), so an average from earlier weeks is biased. Persistence ignores the shape. Combining both is what the weekly-shape rule and the ML models do.

**37. Where does the model struggle?**
10:00–11:30 (fast filling), weekends (2.50 vs 1.84 pp), small often-full car parks (BHMBCCPST01 3.25 pp, BHMBRCBRG01 3.08 pp) and unusual surges (99th-percentile error: a pre-Christmas Sunday where occupancy rose 23 points in an hour).

**38. What does feature importance tell you, and what not?**
Shuffling `rate_now` raises MAE by 25 pp; time of origin (+2.2), trend (+1.6), car park (+1.1) and last week's value (+1.0) follow. `target_hour` looks unimportant only because `origin_slot` carries the same information — correlated features share importance. Importance shows reliance, not causation.

## I. Application architecture

**39. How does the app use the model? Could the numbers be hard-coded?**
`app.py` loads `models/parksmart_h{30,60}.joblib` via `src/predict.py`, runs it on the feature table and reads metrics from `outputs/metrics`. Nothing is hard-coded: `pytest` includes a test that recomputes the test MAE from the saved model and checks it equals the reported value, and notebook 03 shows the same check.

## J. Limitations and future work

**40. What can ParkSmart *not* guarantee?**
That a space will actually be free: it is an estimate with ~25-space average error and a 90% (not 100%) range. It has no live data, no events or weather, only 08:00–16:30, needs a current reading, and was trained on one autumn in 2016. Future work: live feed, a full year with events and weather, time-varying (quantile) intervals, full-day forecasts, sequence models when data allows, and new cities.

## Hard follow-ups to rehearse

* *"If I shuffle your data randomly, will accuracy go up?"* — Probably yes, and that is exactly why we don't: adjacent slots leak into each other and the future would be used to predict the past.
* *"Your R² is 0.987 — isn't that suspicious?"* — R² is high because occupancy varies a lot between car parks and times of day; persistence alone has R² 0.85. The honest comparison is MAE against strong baselines on future weeks, plus the corrupt-the-future leakage test.
* *"Why didn't you tune on the test set?"* — Then the test score would no longer estimate performance on unseen data.
* *"What happens if a car park's sensor fails?"* — No current reading → no forecast; the app says so instead of guessing.
* *"How would you deploy it?"* — Pull the council feed every 30 min, append to the grid, compute features, predict with the saved bundle, re-train weekly, monitor coverage of the 90% range as a drift alarm.
