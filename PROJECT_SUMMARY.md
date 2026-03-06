# Project Summary: Alloy Property Prediction

This document summarizes the **overall architecture**, **notebook-by-notebook** flow, **models used**, and **typical run times** for the alloy forward/backward modeling pipeline.

---

## Overall Architecture

The project has two main tracks (wrought and cast) that share the same pipeline pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONFIG: hyperparams_config.json                                             │
│  (by_target: best forward model+params per property; backward: GMM params)   │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │ save                               │ save
┌────────┴────────┐                  ┌────────┴────────┐
│  01 Forward     │                  │  02 Backward     │
│  Tuning         │                  │  GMM Tuning      │
│  (per-target    │                  │  (BIC on        │
│   XGB/RF/GB)    │                  │   composition)   │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         │ by_target                          │ GMM params
         ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  06 Generate Synthetic Pool (wrought)    06 Generate Synthetic Pool (cast)  │
│  GMM sample → per-target forward models → synthetic_wrought.csv (50k rows)   │
│  GMM sample → per-target forward models → synthetic_cast.csv (50k rows)      │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│  05 Backward Wrought     │        │  05 Backward Cast       │
│  Load pool, set TARGETS, │        │  Load pool, set TARGETS,│
│  filter/sort → top-k    │        │  filter/sort → top-k    │
└─────────────────────────┘        └─────────────────────────┘

Optional (use config, do not create synthetic pool):
┌─────────────────────────┐        ┌─────────────────────────┐
│  03 Forward Wrought      │        │  04 Forward Cast        │
│  Composition → predict  │        │  Composition → predict  │
│  properties (per-target)│        │  properties (per-target)│
└─────────────────────────┘        └─────────────────────────┘
```

- **Forward**: composition (12 elements) → property (UTS, Yield, Conductivity, etc.). One regression model per target; model type and hyperparameters are chosen per target in notebook 01.
- **Backward**: desired properties → candidate compositions. The pipeline does **not** invert the forward model; it builds a large **synthetic pool** of (composition, properties) via GMM sampling + forward prediction, then **searches** that pool (filter/sort) for rows matching your targets.
- **Config** (`hyperparams_config.json`): stores per-target best model and params (`wrought.by_target`, `cast.by_target`) and GMM params for composition space (`backward.wrought.GMM`, `backward.cast.GMM`). All notebooks that need hyperparameters read from here (or fall back to defaults).

---

## Models Used (Short)

| Role | Model | Description |
|------|--------|-------------|
| **Forward (per property)** | **XGBoost** / **Random Forest** / **Gradient Boosting** | Tree-based regressors. For each target (e.g. UTS, conductivity), notebook 01 picks the single best of these three and its hyperparameters via `RandomizedSearchCV` (R²). 03, 04, 06 use that choice from config. |
| **Backward (composition space)** | **GMM (Gaussian Mixture Model)** | Models the distribution of alloy compositions (12 elements). Tuned by BIC in notebook 02 (e.g. `n_components`, `covariance_type`). Used only to **sample** new compositions; no property prediction. |
| **Backward search** | **None (rule-based)** | 05_backward_wrought and 05_backward_cast do **not** train a model. They load the precomputed synthetic CSV and filter/sort by your `TARGETS` to return top-k candidate alloys. |

---

## Execution Order and Run Times

Run notebooks in this order. Times are **pipeline timeouts** (max allowed per notebook when using `run_full_pipeline.py`); actual runs may be shorter.

| Step | Notebook | Timeout (max) | What it does |
|------|----------|----------------|--------------|
| 1 | **01_hyperparameter_tuning_forward.ipynb** | **30 min** (1800 s) | For each property (UTS, Yield, Conductivity, etc.) on wrought and cast data: runs `RandomizedSearchCV` over XGBoost, Random Forest, and Gradient Boosting; picks the best model type and hyperparameters by R²; writes per-target results to `hyperparams_config.json` under `wrought.by_target` and `cast.by_target`. |
| 2 | **02_hyperparameter_tuning_backward.ipynb** | **5 min** (300 s) | Fits GMM on composition data only (no properties). Tunes `n_components` and `covariance_type` via BIC, separately for wrought and cast. Saves best GMM params to `backward.wrought.GMM` and `backward.cast.GMM` in `hyperparams_config.json`. |
| 3 | **06_generate_synthetic_wrought.ipynb** | **10 min** (600 s) | Loads wrought data and config. Fits GMM on real wrought compositions (using params from 02), samples 50k compositions; for each target, builds the per-target forward model from config, trains it on real data, predicts on the 50k compositions. Writes **synthetic_wrought.csv** (composition + predicted properties). |
| 4 | **06_generate_synthetic_cast.ipynb** | **10 min** (600 s) | Same as 06_wrought but for cast: GMM (cast params) → 50k samples → per-target forward models from config → **synthetic_cast.csv**. |
| 5 | **03_forward_wrought_alloys.ipynb** *(optional)* | **5 min** (300 s) | Forward prediction for wrought: loads config, builds per-target models (XGB/RF/GB from `by_target`), trains on wrought data, evaluates and can predict properties for new compositions. Uses `load_hyperparams_for_target('wrought', target)`. |
| 6 | **04_forward_cast_alloys.ipynb** *(optional)* | **2 min** (120 s) | Same as 03 but for cast alloys; uses cast `by_target` from config. |
| 7 | **05_backward_wrought.ipynb** | **1 min** (60 s) | Loads **synthetic_wrought.csv**. You set `TARGETS` (e.g. UTS, Yield, Conductivity). Notebook filters/sorts the pool by those targets and returns top-k candidate alloys (no training). |
| 8 | **05_backward_cast.ipynb** | **1 min** (60 s) | Same as 05_wrought but uses **synthetic_cast.csv** and cast property column names. |

**Core pipeline only (01 → 02 → 06_wrought → 06_cast):** about **40 minutes** max.  
**Full pipeline (all 8):** about **64 minutes** max (timeouts only; actual wall time may be less).

---

## Notebook-by-Notebook Summary

### 01_hyperparameter_tuning_forward.ipynb
- **Purpose:** Choose, for each property and each dataset (wrought/cast), the best regression model (XGBoost, Random Forest, or Gradient Boosting) and its hyperparameters.
- **Models:** XGBoost, RandomForestRegressor, GradientBoostingRegressor; `RandomizedSearchCV` (e.g. n_iter=6, cv=3); best by R².
- **Output:** `hyperparams_config.json` → `wrought.by_target`, `cast.by_target` (each key = target name, value = `{ "model": "...", "params": {...} }`).
- **Time:** Longest step; up to 30 min.

### 02_hyperparameter_tuning_backward.ipynb
- **Purpose:** Tune GMM for the composition space only (no property data), so that 06 can sample realistic compositions.
- **Models:** `GaussianMixture`; grid over `n_components` and `covariance_type`; best by BIC.
- **Output:** `hyperparams_config.json` → `backward.wrought.GMM`, `backward.cast.GMM`.
- **Time:** Up to 5 min.

### 06_generate_synthetic_wrought.ipynb
- **Purpose:** Build the search pool for backward wrought: 50k synthetic (composition, properties) rows.
- **Models:** GMM (from 02) to sample compositions; per-target forward models (from 01) trained on real data and used to predict properties on the 50k rows.
- **Output:** `synthetic_wrought.csv`.
- **Time:** Up to 10 min.

### 06_generate_synthetic_cast.ipynb
- **Purpose:** Same as 06_wrought but for cast; produces the cast search pool.
- **Models:** GMM (cast); per-target forward models (cast by_target).
- **Output:** `synthetic_cast.csv`.
- **Time:** Up to 10 min.

### 03_forward_wrought_alloys.ipynb (optional)
- **Purpose:** Train per-target forward models on wrought data (using config) and predict/evaluate on wrought alloys.
- **Models:** Same as 01 (XGB/RF/GB per target from config); no tuning, just train and predict.
- **Output:** Metrics, plots, and ability to predict properties for new compositions.
- **Time:** Up to 5 min.

### 04_forward_cast_alloys.ipynb (optional)
- **Purpose:** Same as 03 for cast dataset.
- **Models:** Per-target models from cast `by_target`.
- **Time:** Up to 2 min.

### 05_backward_wrought.ipynb
- **Purpose:** Given desired properties (TARGETS), find top-k wrought alloys from the synthetic pool.
- **Models:** None; loads `synthetic_wrought.csv`, filters/sorts by TARGETS, returns top candidates.
- **Output:** Table (and optional plots) of best-matching compositions.
- **Time:** Up to 1 min.

### 05_backward_cast.ipynb
- **Purpose:** Same as 05_wrought for cast pool (`synthetic_cast.csv`).
- **Models:** None; filter/sort only.
- **Time:** Up to 1 min.

### 05_backward_universal_lab.ipynb
- **Status:** **Deprecated.** Use **05_backward_wrought.ipynb** or **05_backward_cast.ipynb** instead; those use the synthetic pools and per-target config.

---

## Data and Config

- **Input data:** `wrought_alloys_final.csv`, `cleaned_cast_dataset.csv` (composition columns: Al, Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ga, V, Ti).
- **Config:** `hyperparams_config.json` (by_target + backward GMM); created/updated by 01 and 02.
- **Generated:** `synthetic_wrought.csv`, `synthetic_cast.csv` (by 06).
- **Scripts:** `run_pipeline.py` (short run), `run_full_pipeline.py` (all notebooks in order with timeouts and logs); `utils.py` (load/save hyperparameters, GMM params).

This summary reflects the current design: per-target forward tuning, separate backward GMM tuning, synthetic pool generation, and backward search by filter/sort on that pool.
