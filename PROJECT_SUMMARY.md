# Project Summary: Alloy Property Prediction (Wrought Only)

This document summarizes the **overall architecture**, **notebook-by-notebook** flow, **models used**, and **typical run times** for the wrought alloy forward/backward pipeline.

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONFIG: hyperparams_config.json                                            │
│  (wrought.by_target: best forward model+params per property; backward hybrid) │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │ save                               │ save
┌────────┴────────┐                  ┌────────┴────────┐
│  01 Forward     │                  │  02 Backward     │
│  Tuning         │                  │  GMM Tuning      │
│  (per-target    │                  │  (BIC on        │
│   XGB/RF/GB;    │                  │   composition)   │
│   HGB/ET hard   │                  │                    │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         │ by_target                          │ GMM params
         ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  06 Generate Synthetic Pool (wrought)                                        │
│  GMM+BGMM sample → forward prediction → balanced selection → synthetic_wrought.csv │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  07 Consistency Report   │
│  Multi-seed GMM vs BGMM │
│  win stability summary  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  05 Backward Wrought     │
│  Load pool, set TARGETS, │
│  filter/sort → top-k    │
└─────────────────────────┘

Optional (use config, do not create synthetic pool):
┌─────────────────────────┐
│  03 Forward Wrought      │
│  Composition → predict  │
│  properties (per-target)│
└─────────────────────────┘
```

- **Forward**: composition (12 elements) → property (UTS, Yield, Conductivity, etc.). One regression model per target; model type and hyperparameters are chosen per target in notebook 01.
- **Backward**: desired properties → candidate compositions. The pipeline does **not** invert the forward model; it builds candidate pools via GMM and BGMM sampling + forward prediction, then selects the better pool with balanced scoring before searching top candidates.
- **Config** (`hyperparams_config.json`): stores per-target best model and params (`wrought.by_target`), generator settings (`backward.wrought.GMM`, `backward.wrought.BGMM`), and scoring weights (`backward.wrought.synthetic_selection`).

---

## Models Used (Short)

| Role | Model | Description |
|------|--------|-------------|
| **Forward (per property)** | **XGBoost** / **Random Forest** / **Gradient Boosting** / **HistGradientBoosting** / **ExtraTrees** | Tree-based regressors. Notebook 01 searches XGB, RF, and GB for every target; for harder targets (YS and Fatigue) it also searches **HistGradientBoosting** and **ExtraTrees**, with more CV folds and iterations. 03 and 06 rebuild the saved model type from config. |
| **Backward (composition space)** | **GMM + Bayesian GMM (BGMM)** | Both models sample wrought alloy compositions (12 elements). Notebook 02 persists both generators (`backward.wrought.GMM` and `backward.wrought.BGMM`) using dataset-level tuning. Both are sampling-only (no direct property prediction). |
| **Backward search** | **None (rule-based)** | **05_backward_wrought** does **not** train a model. It loads the precomputed synthetic CSV and filter/sorts by your `TARGETS` to return top-k candidate alloys. |

---

## Execution Order and Run Times

Run notebooks in this order. Times are **pipeline timeouts** (max allowed per notebook when using `run_full_pipeline.py`); actual runs may be shorter.

| Step | Notebook | Timeout (max) | What it does |
|------|----------|----------------|--------------|
| 1 | **01_hyperparameter_tuning_forward.ipynb** | **30 min** (1800 s) | Per property: expanded `RandomizedSearchCV` (XGB/RF/GB; plus HistGradientBoosting & ExtraTrees for YS and Fatigue). Standard targets use fewer iterations than hard targets. Writes `wrought.by_target` to `hyperparams_config.json`. |
| 2 | **02_hyperparameter_tuning_backward.ipynb** | **5 min** (300 s) | Tunes both generators on wrought composition data: GMM via BIC and BGMM via holdout log-likelihood; saves `backward.wrought.{GMM,BGMM}`. |
| 3 | **06_generate_synthetic_wrought.ipynb** | **10 min** (600 s) | GMM and BGMM each sample compositions; forward models predict properties; balanced scoring picks the best pool and writes **synthetic_wrought.csv** (+ score table). |
| 4 | **07_generator_consistency_report.ipynb** *(optional)* | **10 min** (600 s) | Runs multi-seed generator comparison and writes `generator_consistency_runs.csv` + `generator_consistency_summary.csv`. |
| 5 | **03_forward_wrought_alloys.ipynb** *(optional)* | **5 min** (300 s) | Forward prediction: loads config, trains per-target models on wrought data, evaluates / predicts new compositions. |
| 6 | **05_backward_wrought.ipynb** | **1 min** (60 s) | Loads **synthetic_wrought.csv**; set `TARGETS`; returns top-k candidate alloys. |

**Core pipeline (01 → 02 → 06_wrought):** about **45 minutes** max (timeouts).  
**Full pipeline with consistency report (core + 07 + 03 + 05):** about **61 minutes** max (timeouts only; actual wall time may be less).

---

## Notebook-by-Notebook Summary

### 01_hyperparameter_tuning_forward.ipynb
- **Purpose:** Best regression model and hyperparameters per wrought property among XGBoost, Random Forest, Gradient Boosting; for YS and Fatigue, also **HistGradientBoosting** and **ExtraTrees**, with heavier search.
- **Output:** `hyperparams_config.json` → `wrought.by_target` (`model` string must match `build_model` in 03, 06, and `run_pipeline.py`).

### 02_hyperparameter_tuning_backward.ipynb
- **Purpose:** Tune both GMM and BGMM on wrought compositions for synthetic sampling in 06.
- **Output:** `hyperparams_config.json` → `backward.wrought.GMM` and `backward.wrought.BGMM`.

### 06_generate_synthetic_wrought.ipynb
- **Purpose:** Build GMM and BGMM synthetic pools, score both with balanced criteria, and keep the best pool for backward search.
- **Output:** `synthetic_wrought.csv`, `synthetic_wrought_generator_scores.csv`.

### 07_generator_consistency_report.ipynb (optional)
- **Purpose:** Evaluate GMM vs BGMM winner stability over multiple random seeds.
- **Output:** `generator_consistency_runs.csv`, `generator_consistency_summary.csv`.
- **Recommendation:** Run periodically (for example after retuning or dataset updates) to verify generator winner stability.

### 03_forward_wrought_alloys.ipynb (optional)
- **Purpose:** Train/predict per-target models on wrought data using config.

### 05_backward_wrought.ipynb
- **Purpose:** Target properties → top-k candidates from synthetic pool.
- **Models:** None (filter/sort only).

### 05_backward_universal_lab.ipynb
- **Status:** **Deprecated.** Use **05_backward_wrought.ipynb** and the synthetic pool from **06_generate_synthetic_wrought.ipynb**.

---

## Data and Config

- **Input data:** `wrought_alloys_final.csv` (composition columns: Al, Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ga, V, Ti).
- **Config:** `hyperparams_config.json` (from 01 and 02, plus optional BGMM and selection settings).
- **Generated:** `synthetic_wrought.csv`, `synthetic_wrought_generator_scores.csv` (from 06), `generator_consistency_runs.csv`, `generator_consistency_summary.csv` (from 07).
- **Scripts:** `run_pipeline.py`, `run_full_pipeline.py`, `utils.py`.
