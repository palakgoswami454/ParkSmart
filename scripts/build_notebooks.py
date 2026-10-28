"""Generate and execute the three project notebooks.

The notebooks call the same functions as the CLI pipeline, so their outputs are
real results, not copies. Training itself (about 15 minutes) is run with
`python -m src.train`; notebook 03 loads its saved outputs and re-checks them.

    python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
from IPython.display import Image, display
from src import config as C"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


nb1 = [
    md("# 01 · Data understanding\n\nGoal: describe the raw UCI *Parking Birmingham* file exactly as it is, before any cleaning, "
       "and confirm it matches the official description (35,717 rows, 4 attributes, 4 Oct – 19 Dec 2016)."),
    code(SETUP),
    code("from src.data_preprocessing import load_raw, quality_report\nraw = load_raw()\nprint(raw.shape)\nraw.head()"),
    code("raw.dtypes"),
    md("## Audit\nMissing cells, duplicates, impossible values and coverage, computed directly from the file."),
    code("q = quality_report(raw)\n{k: v for k, v in q.items() if k != 'per_park'}"),
    code("per_park = pd.DataFrame(q['per_park']).T.sort_values('observations')\nper_park"),
    code("rate = raw['Occupancy'] / raw['Capacity']\nrate.describe()"),
    code("print('readings per hour of day');print(raw['LastUpdated'].dt.hour.value_counts().sort_index())"),
    md("## What this tells us\n"
       "* The file matches the UCI description: 35,717 rows, the four documented columns, 30 car parks, no empty cells.\n"
       "* *No empty cells* does not mean *no missing data*: 216 rows are exact duplicates, four calendar days have no readings "
       "at all (20–21 Oct, 3–4 Dec), and two car parks (BHMBRTARC01 with 88 readings, NIA North with 162) report only rarely.\n"
       "* 12 readings have negative occupancy (physically impossible → sensor error) and 373 exceed capacity (up to 104%; "
       "cars queuing or counter drift).\n"
       "* Capacity never changes for a car park, so it can be used to turn a predicted rate into predicted free spaces.\n"
       "* Readings arrive about every 27–33 minutes between ~08:00 and 16:30, which fixes a natural 30-minute time step and "
       "means horizons shorter than 30 minutes cannot be supported."),
]

nb2 = [
    md("# 02 · Preprocessing and exploratory data analysis"),
    code(SETUP),
    md("## Cleaning and the 30-minute grid\nEach step and its row count is logged by `src/data_preprocessing.py`."),
    code("from src.data_preprocessing import load_raw, clean, to_slot_grid\nraw = load_raw()\ncleaned, clean_log = clean(raw)\n"
         "grid, grid_log = to_slot_grid(cleaned)\nclean_log, grid_log"),
    code("grid.head(20)"),
    md("Decisions and why:\n"
       "* **Drop exact duplicates** (216) – they would double-count a reading.\n"
       "* **Clip negative occupancy to 0** (12 rows) – a count cannot be negative.\n"
       "* **Cap the rate at 1.0** (373 rows) – a car park cannot be more than full; the raw count is kept separately.\n"
       "* **Drop car parks with < 500 readings** (2 parks) – too sparse to provide training days.\n"
       "* **Snap readings to the nearest 30-minute slot if within 10 minutes**, average if two land in one slot (17 cases) – "
       "turns irregular timestamps into a regular series so that 'one slot ahead' always means 30 minutes.\n"
       "* **Never interpolate** missing slots (5.0% of the grid) – an invented value could become a training target."),
    md("## Exploratory analysis\n`src/eda.py` draws every figure and stores the quoted numbers in `outputs/metrics/eda_summary.json`."),
    code("from src.eda import run as run_eda\neda = run_eda()"),
    code("display(Image(ROOT / 'outputs/figures/eda_01_occupancy_distribution.png'))"),
    md("The distribution is wide (mean ≈ 49%) with a spike at 100%: some car parks fill completely. About 9% of slots are FULL "
       "(≥ 90%), so FULL is a minority class – plain accuracy would hide poor FULL detection, hence macro-F1 and FULL recall later."),
    code("display(Image(ROOT / 'outputs/figures/eda_02_hour_of_day_profile.png'))\neda['peak_time'], eda['peak_mean_rate']"),
    md("Occupancy climbs from 08:00 to a peak around 14:00 and then falls. Time of day must be a feature, and because the curve is "
       "steepest in the morning, persistence ('no change') will be worst there."),
    code("display(Image(ROOT / 'outputs/figures/eda_03_weekday_hour_heatmap.png'))\neda['weekday_mean']"),
    md("Weekends, especially Sundays, are quieter and start later. This motivates `day_of_week`, `is_weekend` and the "
       "same-weekday-last-week features."),
    code("display(Image(ROOT / 'outputs/figures/eda_04_facility_comparison.png'))"),
    md("Car parks behave very differently (mean occupancy from ~16% to ~74%; some are FULL in a quarter of slots, others never). "
       "A car-park identifier is needed so one model can serve them all."),
    code("display(Image(ROOT / 'outputs/figures/eda_05_daily_trend.png'))\neda['mean_rate_by_split']"),
    md("There is a clear weekly cycle and a rise towards Christmas: the test fortnight (mean 53.6%) is busier than the training "
       "period (47.6%). A chronological split therefore tests the model under realistic drift – a random split would hide it."),
    code("display(Image(ROOT / 'outputs/figures/eda_06_missing_data_map.png'))\neda['parks_most_missing']"),
    md("Missing data is concentrated in a few car parks and days, not spread randomly. Tree models handle missing lag features "
       "natively; the linear model gets median imputation plus missing-indicator columns."),
    code("display(Image(ROOT / 'outputs/figures/eda_07_feature_target_correlation.png'))\neda['spearman_with_target_h60']"),
    md("Current occupancy and last week's value at the target time correlate most with occupancy 60 minutes later. Correlation is "
       "computed on the training period only, and it measures association, not cause."),
    code("display(Image(ROOT / 'outputs/figures/eda_08_daily_curves_examples.png'))"),
    md("## Feature engineering\nEvery feature is computed from information available at the forecast origin."),
    code("from src.feature_engineering import build_features, FEATURE_DESCRIPTIONS, split_by_date\n"
         "pd.Series(FEATURE_DESCRIPTIONS, name='description').to_frame()"),
    code("f60 = build_features(grid, 60)\nsp = split_by_date(f60)\n"
         "pd.DataFrame({k: {'rows': len(v), 'from': v['date'].min().date(), 'to': v['date'].max().date()} for k, v in sp.items()}).T"),
    code("f60[['SystemCodeNumber','slot_time','target_time','rate_now','rate_lag1','target_slot_rate_7d_ago','target_rate','target_status']].head(8)"),
    code("f60['target_status'].value_counts(normalize=True).round(3)"),
]

nb3 = [
    md("# 03 · Model training and evaluation\n\nTraining is run from the command line because the hyper-parameter search takes about "
       "15 minutes:\n\n```bash\npython -m src.train      # tune on validation, walk-forward CV, refit, save models\n"
       "python -m src.evaluate   # test-period metrics, figures, importance, unseen-car-park test\n```\n\n"
       "This notebook loads those saved outputs and **re-computes the headline test score from the saved model** to prove the "
       "reported numbers are real."),
    code(SETUP),
    code("tr = json.loads((C.METRICS_DIR / 'training_summary.json').read_text())\n"
         "ev = json.loads((C.METRICS_DIR / 'evaluation_summary.json').read_text())\n"
         "for h in ('30', '60'):\n    print(h, 'min | rows', tr[h]['rows'], '| folds', tr[h]['walk_forward_folds'])"),
    md("## Hyper-parameter tuning (validation fortnight)"),
    code("rows = []\nfor h in ('30', '60'):\n    for m, d in tr[h]['regression'].items():\n"
         "        rows.append({'horizon': h, 'model': m, 'best_params': d['best_params'], 'val_MAE_pp': d['validation_MAE']*100, "
         "'grid_size': len(d['tuning_log'])})\npd.DataFrame(rows)"),
    md("## Walk-forward validation (4 expanding weekly folds) – used to select the model"),
    code("pd.DataFrame([{'horizon': h, 'model': m, **{r['fold']: round(r['MAE']*100, 3) for r in d['walk_forward']}, "
         "'mean': d['walk_forward_MAE_mean']*100} for h in ('30','60') for m, d in tr[h]['regression'].items()])"),
    md("## Test fortnight (never used for any decision)"),
    code("for h in ('60', '30'):\n    print(f'--- {h} minutes ahead ---')\n"
         "    display(pd.DataFrame(ev[h]['regression_table']).drop(columns=['Selected']).round(3))"),
    code("display(Image(ROOT / 'outputs/figures/eval_model_comparison_h60.png'))"),
    md("### Verify: recompute the selected model's test MAE from the saved file"),
    code("from src.data_preprocessing import load_grid\nfrom src.feature_engineering import build_features, split_by_date, FEATURES\n"
         "from src.predict import load_bundle\nfrom sklearn.metrics import mean_absolute_error\ngrid = load_grid()\n"
         "for h in (30, 60):\n    b = load_bundle(h)\n    test = split_by_date(build_features(grid, h, codes=b['park_codes']))['test']\n"
         "    mae = mean_absolute_error(test['target_rate'], np.clip(b['regressor'].predict(test[FEATURES]), 0, 1)) * 100\n"
         "    rep = [r for r in ev[str(h)]['regression_table'] if r['Selected']][0]['Test MAE (pp)']\n"
         "    print(f'{h} min: recomputed {mae:.4f} pp | reported {rep:.4f} pp | model {b[\"regressor_name\"]}')"),
    md("## Status classification"),
    code("for h in ('60', '30'):\n    print(f'--- {h} minutes ahead ---')\n"
         "    display(pd.DataFrame(ev[h]['classification_table'])[['Model','accuracy','macro_precision','macro_recall','macro_F1',"
         "'weighted_F1','FULL_precision','FULL_recall','HIGH_recall','Walk-forward macro F1']].round(3))"),
    code("display(Image(ROOT / 'outputs/figures/eval_confusion_h60.png'))\n"
         "{k: ev['60'][k] for k in ('full_actual','full_missed','full_missed_as_low_or_moderate','false_full_alarms')}"),
    md("## Error analysis"),
    code("pd.DataFrame(ev['60']['segments']).T.round(3)"),
    code("display(Image(ROOT / 'outputs/figures/eval_error_by_hour_h60.png'))\ndisplay(Image(ROOT / 'outputs/figures/eval_per_park_mae_h60.png'))"),
    code("display(Image(ROOT / 'outputs/figures/eval_actual_vs_pred_h60.png'))\ndisplay(Image(ROOT / 'outputs/figures/eval_example_day_h60.png'))"),
    code("pd.read_csv(C.METRICS_DIR / 'prediction_examples_h60.csv')"),
    md("## Uncertainty and generalisation"),
    code("{h: {'halfwidth_90_pp': ev[h]['interval_halfwidth_90_pp'], 'test_coverage': ev[h]['interval_test_coverage_90']} for h in ('30','60')}"),
    code("ev['unseen_park_experiment']"),
    md("## Interpretation"),
    code("display(Image(ROOT / 'outputs/figures/eval_permutation_importance_h60.png'))\npd.DataFrame(ev['60']['permutation_importance']).head(8)"),
]

INTERP3 = """## Conclusions from the experiments
* **ML beats every baseline at both horizons.** 60 min ahead, HistGradientBoosting has test MAE 2.02 pp (≈ 25 spaces) vs 7.07 pp
  for persistence, 7.24 pp for the historical profile and 2.79 pp for the strong *persistence + weekly shape* rule.
* **The strong rule matters.** Ridge regression (2.83 pp) does *not* beat it on test MAE, and Random Forest only beats it
  modestly. A project that compared ML only with persistence would overstate the value of ML.
* **Thresholding the regressor gives the best status classifier** (60 min: macro-F1 0.915, FULL recall 0.88, FULL precision 0.93),
  better than training a classifier directly; the regression output contains more information than the four labels.
* **Uncertainty is calibrated:** the ±4.5 pp range (from walk-forward residuals only) covered 90.1% of test outcomes.
* **Hardest cases:** the morning fill-up (10:00–11:30) and weekends, and a few car parks with irregular behaviour.
* **Generalisation:** for car parks never seen in training (ID hidden) MAE rises from 2.02 to 2.73 pp – still far better than
  persistence, so the learned patterns are not just memorised car-park identities.
* **Importance:** `rate_now` dominates; time of day (`origin_slot`), recent trend (`rate_delta1`), car-park ID and last week's value
  add the rest. `target_slot`/`target_hour` show ~0 importance only because they carry the same information as `origin_slot`
  (correlated features share importance) – not because time does not matter.
"""
nb3.append(md(INTERP3))

nb4 = [
    md("# 04 · Pune directory, recommendations and model transfer\n\n"
       "ParkSmart 2.0 focuses on Pune. No public Pune occupancy data exists, so this notebook (1) audits the hand-curated, "
       "sourced Pune facility directory, (2) shows the transparent recommendation logic, and (3) evaluates whether the "
       "Birmingham-trained model *transfers* to a car park it has never seen, which is the evidence needed before any "
       "Pune forecast could be shown."),
    code(SETUP),
    md("## 1. The Pune directory and its evidence"),
    code("from src.data_loader import load_pune, summary_counts, STATUS\nd = load_pune()\nsummary_counts(d)"),
    code("d.facilities[['name','locality','type_label','status_label','vehicles_label','has_tariff','has_location','best_source_kind']]"),
    code("d.facilities['status_label'].value_counts()"),
    code("import pandas as pd\npd.Series({k: v['kind'] for k, v in d.sources.items()}).value_counts()"),
    md("Every row cites at least one dated source; `null` fields mean the sources are silent. Status labels describe the "
       "evidence on the source date. Nine of the 19 facilities are metro-station lots announced in Feb 2024 whose "
       "opening was never confirmed, two of them contradicted by later reports. That is why the app treats them as "
       "*worth checking* and never as available."),
    md("### Coordinate sanity check\nEvery mapped facility should lie within a few km of a cited destination. "
       "Wikipedia locality coordinates for Kothrud, Aundh and Pune Camp failed this kind of check and were not used."),
    code("from src.location_services import haversine_km\ndests = d.destinations.dropna(subset=['lat'])\n"
         "rows = []\nfor _, f in d.facilities[d.facilities.has_location].iterrows():\n"
         "    dist, name = min((haversine_km(f.lat, f.lon, x.lat, x.lon), x['name']) for _, x in dests.iterrows())\n"
         "    rows.append({'facility': f['name'], 'nearest destination': name, 'km': round(dist, 2)})\npd.DataFrame(rows)"),
    md("## 2. Recommendations\nScore = 0.6 × proximity + 0.4 × evidence; hard filters only for known reasons."),
    code("from src.recommendations import recommend\ndest = d.destinations.set_index('id').loc['fc-road'].to_dict() | {'id': 'fc-road'}\n"
         "rec = recommend(d, dest, 'car', hours=3, budget=None, radius_km=5)\n"
         "rec.ranked[['name','distance_km','proximity','evidence','score','cost','missing']].round(2)"),
    code("rec.not_suitable[['name','distance_km','reason']].round(2).head(10)"),
    md("Budget and vehicle filters exclude only facilities where the *known* price or vehicle list rules them out:"),
    code("r = recommend(d, d.destinations.set_index('id').loc['shivajinagar'].to_dict() | {'id': 'shivajinagar'}, 'car', 3, budget=40)\n"
         "r.not_suitable[['name','reason']].head(6)"),
    md("## 3. Uncertainty by time of day\nSplit-conformal ranges from walk-forward residuals, one width per band of the "
       "forecast time."),
    code("ev = json.loads((C.METRICS_DIR / 'evaluation_summary.json').read_text())\n"
         "pd.DataFrame(ev['60']['interval_band_report']['by_band']).T.round(3)"),
    md("A single width under-covers the morning fill-up and over-covers the afternoon; band widths bring every band "
       "closer to the 90% target while keeping overall coverage near 90%."),
    md("## 4. Does the model transfer to an unseen car park?\nA park-agnostic model (ID hidden) is trained on 27 car "
       "parks and tested on the 28th, exactly as an uploaded Pune file would be processed."),
    code("t = json.loads((C.METRICS_DIR / 'transfer_summary.json').read_text())\npd.DataFrame({h: t[h] for h in ('30', '60')})"),
    code("from src.transfer import evaluate_upload, SAMPLE_CSV\nres = evaluate_upload(pd.read_csv(SAMPLE_CSV), 60)\n"
         "{'model_MAE_pp': res.model_mae_pp, 'persistence_MAE_pp': res.persistence_mae_pp, 'coverage': res.coverage, "
         "'beats_persistence': res.beats_persistence, 'rows_scored': res.n_scored}"),
    md("**Interpretation.** On a car park it never saw, the park-agnostic model is about three times more accurate than "
       "persistence. This supports *testing* it on Pune counts, and nothing more: the held-out car park shares the "
       "training period and city, which makes it easier than Pune in a different year. The upload tool therefore "
       "measures error on the user's own data and only shows forecasts if the model beats persistence there."),
]


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)
    print("executed", NB / name)


ALL = {"01_data_understanding.ipynb": nb1, "02_preprocessing_and_eda.ipynb": nb2,
       "03_model_training_and_evaluation.ipynb": nb3, "04_pune_directory_and_transfer.ipynb": nb4}

if __name__ == "__main__":
    import sys
    NB.mkdir(exist_ok=True)
    wanted = sys.argv[1:] or list(ALL)  # e.g. python scripts/build_notebooks.py 04_pune_directory_and_transfer.ipynb
    for name in wanted:
        build(name, ALL[name])
