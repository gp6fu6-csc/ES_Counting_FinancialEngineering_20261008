# ESCounting V4 autoresearch program

The research daemon (`escount research`, `escount/research/daemon.py`) searches extraction and model
settings per case. These rules are fixed; a research run must not change them.

## Rules

1. **Fixed evaluator.** `escount/research/evaluate.py` is never edited by research. Every trial is scored
   by `evaluate()` with grouped folds: LOIO (all rows of one image held out) for every case; the General
   cases also record LOPO (one project held out).
2. **Single metric.** Exact hit rate only: an image is a hit iff `round(prediction) == GT`. No MAE, no ±1.
   Images for which a geometry yields no ROI row count as misses.
3. **Rank key.** `(-hit_rate, roi_count, latency_ms)`: higher hit rate first, then fewer ROIs, then lower
   latency. `latency_ms` is measured wall-clock time (`time.perf_counter`) of `predict` on cached
   features, so a trial tied on hit rate and ROI count is kept or discarded by timing jitter: that
   last tie-break, and therefore `status` in `results.tsv`, is not reproducible between runs.
   Hit rate and ROI count are.
4. **Fixed budget per trial** (`TrialBudget`, identical for all trials and cases): at most 200 training
   rows per image (deterministic subsample), 200 boosting rounds for XGBoost / LightGBM, at most 3000 SVR
   training rows. Each trial evaluates exactly one configuration.
5. **Keep / discard.** Each trial appends one row to `research/<case>/results.tsv`
   (`trial, case, config_hash, hit_rate, hits/total, roi_count, latency_ms, status, timestamp`).
   `status = keep` iff its rank key is strictly better than the current candidate; then the complete,
   reusable configuration is written to `candidates/<case>/best_params.json` (+ `per_image.csv`).
   Failed trials are recorded as `discard` with `hit_rate = nan`.
6. **Manual adoption (record only).** Production (`outputs/projects/<case>/`) is untouched by research.
   `escount adopt <case>` copies the candidate to `outputs/projects/<case>/research/` as a record;
   inference does not read it and keeps using `model/manifest.json` from `escount train` +
   `escount select-rois`. Research hit rates describe the research procedure, not the deployed model
   (its LOIO figures are in `model/manifest.json`).
7. **ROI geometry.** Every strip row stays ≥ 50 px from the left/right edge of the background_BW valid
   region; adjacent strips overlap by less than w/3 (`D > 2w/3`, enforced by `AugmentGrid`); strips span
   the whole stack vertically.
8. **No vertical smoothing.** Projection averages across the strip width only (mean / gaussian / median);
   smoothing happens only inside the indicators.
9. **Local hardware.** Features are computed with a local process pool; XGBoost uses `device="cuda"` with
   automatic CPU fallback. Nothing leaves the machine.
10. **Determinism / resume.** One Optuna study per case in `research/study.db` (TPE, seed = 0 + number of
    trials already in the study); `load_if_exists=True` keeps trial numbers contiguous after a restart.
    Cases are visited round-robin: every project not in `excluded_projects` of `configs/general.json`
    (Ring-type is excluded by user decision: LOIO 3/6) and `general` (`general_no_ring` only while
    Ring-type is active). A resumed case must use the same project set (else use fresh storage/roots).

## Search space

| Parameter | Values | Notes |
|---|---|---|
| `w` | 6–40 px, step 2 | strip width |
| `D` | 6–90 px, step 2 | pitch; raised to `floor(2w/3) + 1` when `D ≤ 2w/3` (overlap < w/3) |
| `x_offset_frac` | 0–0.95, step 0.05 | grid shift `x_offset = int(frac · D)` |
| `projection` | mean, gaussian, median | width-direction projection |
| `sigma_ratio` | 0.10–0.60, step 0.05 | gaussian σ = ratio · w (used only for gaussian) |
| `use_kd`, `use_macd`, `use_peaks`, `use_bollinger` | True / False | indicator families (18 fixed configs incl. their gates) fed to the model |
| `directions` | both, up_down, all | count directions of the indicator columns |
| `model` | baseline, ridge, svr, xgb, lgbm | baseline = best single count column on the training fold; ML = δ = GT − period_count regression |
| `roi_cap` | 1–8 | ROIs kept per image, ranked on the training fold by exact hits |
| `aggregation` | median, trimmed | image prediction from the kept ROIs |
| `ridge_alpha` | 1e-3–1e3 (log) | ridge only |
| `svr_C`, `svr_epsilon` | 0.1–100, 0.01–1 (log) | SVR only |
| `xgb_max_depth`, `xgb_learning_rate`, `xgb_subsample`, `xgb_colsample_bytree` | 2–8, 0.01–0.3 (log), 0.5–1, 0.5–1 | XGBoost only |
| `lgbm_num_leaves`, `lgbm_learning_rate`, `lgbm_min_child_samples`, `lgbm_subsample`, `lgbm_colsample_bytree` | 7–63, 0.01–0.3 (log), 5–50, 0.5–1, 0.5–1 | LightGBM only |

Project cases use absolute features and exact strip slots (`slot_key`); General cases use
scale-normalised features and x_norm-bin units (`pos_key`) shared across projects. The side margin
is fixed at 50 px.

## Report

`escount report` writes `outputs/research_report/<timestamp>/` and `latest/` with, per case,
`best_params.json, results.tsv, progress.png, param_importance.png (fANOVA), param_sensitivity.png,
per_image.csv`, the `comparison/` figures and tables, `summary.md` and a self-contained `report.html`.
