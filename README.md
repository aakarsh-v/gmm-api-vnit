# Alloy Property Prediction: Forward and Backward Modeling (Wrought)

This project uses machine learning to predict **wrought** aluminum alloy properties from chemical composition (forward modeling) and to discover candidate compositions for target properties (backward modeling). Hyperparameter tuning saves settings to a shared config file.

---

## Prerequisites

- Python 3.8+
- Jupyter (or VS Code with Jupyter support)

### Choose Python environment

Before running, select the correct kernel:

- **Jupyter:** Kernel → Change Kernel → Python 3 (or your conda/venv)
- **VS Code:** Top-right kernel selector → Python 3.x

### Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
```

---

## Data Files

Place the wrought dataset in the project directory:

| File | Description |
|------|-------------|
| `wrought_alloys_final.csv` | Wrought alloy compositions and properties |

---

## How to Run: Execution Order

### Step 1: Forward hyperparameter tuning (01)

Tunes per-target best model and hyperparameters for wrought (XGBoost, Random Forest, Gradient Boosting; plus **HistGradientBoosting** and **ExtraTrees** for YS and Fatigue). Saves to `hyperparams_config.json` under `wrought.by_target`.

```bash
jupyter nbconvert --to notebook --execute --inplace 01_hyperparameter_tuning_forward.ipynb
```

Or open the notebook and run all cells. May take several minutes.

---

### Step 2: Backward generator tuning (02)

Tunes both generators on wrought composition data:
- `GMM`: BIC-based search
- `BGMM`: holdout average log-likelihood search

Saves both to `backward.wrought.{GMM,BGMM}` in `hyperparams_config.json`.
Selection weights are read from `backward.wrought.synthetic_selection` (or defaults when absent).

```bash
jupyter nbconvert --to notebook --execute --inplace 02_hyperparameter_tuning_backward.ipynb
```

---

### Step 3: Generate synthetic pool (06)

`06_generate_synthetic_wrought.ipynb` → produces `synthetic_wrought.csv` (search pool for backward step).

Fits both GMM and BGMM on real compositions, samples pools, labels them with per-target forward models from config, scores both pools with a balanced objective, and saves:
- `synthetic_wrought.csv` (best pool selected automatically)
- `synthetic_wrought_generator_scores.csv` (per-generator score breakdown)

---

### Step 4 (optional): Generator consistency report (07)

`07_generator_consistency_report.ipynb` evaluates generator stability across multiple seeds and saves:
- `generator_consistency_runs.csv` (one row per seed with GMM vs BGMM scores)
- `generator_consistency_summary.csv` (win rates and score statistics)

Recommended: run this periodically (for example weekly or after retraining/tuning updates) to confirm the current winner remains stable.

---

### Step 5: Backward search (05)

**05_backward_wrought.ipynb** — load `synthetic_wrought.csv`, set `TARGETS` (e.g. UTS, Yield), run to get top candidate alloys.

---

### Optional: Forward prediction (03)

**03_forward_wrought_alloys.ipynb** — predict properties from composition using `by_target` from config.

---

### Run pipeline without Jupyter

```bash
python run_pipeline.py
```

Shortened 01 → 02 → 06 (wrought) → 05 (wrought) for a quick check.

**Full notebook pipeline:**

```bash
python run_full_pipeline.py
```

Optional: `python run_full_pipeline.py --core-only` runs only 01, 02, and 06_wrought.
Optional: `python run_full_pipeline.py --with-consistency-report --consistency-seeds 10 --consistency-samples 5000` runs notebook 07 after 06.

---

## One-line sequence (notebooks, run in order)

```bash
cd e:\vnit-intern\project_arch

jupyter nbconvert --to notebook --execute --inplace 01_hyperparameter_tuning_forward.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_hyperparameter_tuning_backward.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_generate_synthetic_wrought.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_generator_consistency_report.ipynb
```

Then open **05_backward_wrought.ipynb**, set `TARGETS`, and run.

---

## Quick demo (without full tuning)

```bash
python run_demo.py
```

Loads wrought data, trains on a few targets, prints R² and MAE.

---

## Project structure

```
project_arch/
├── README.md
├── utils.py                        # Hyperparameter load/save, by_target, backward GMM/BGMM helpers
├── hyperparams_config.json         # wrought.by_target + backward.wrought.{GMM,BGMM,synthetic_selection}
├── run_demo.py
├── run_pipeline.py                 # Optional: 01→02→06→05 without Jupyter
├── run_full_pipeline.py            # Execute notebooks in order with timeouts
├── 01_hyperparameter_tuning_forward.ipynb
├── 02_hyperparameter_tuning_backward.ipynb
├── 03_forward_wrought_alloys.ipynb
├── 05_backward_wrought.ipynb
├── 05_backward_universal_lab.ipynb # Deprecated; use 05_backward_wrought
├── 06_generate_synthetic_wrought.ipynb
├── 07_generator_consistency_report.ipynb
├── synthetic_wrought.csv           # Best synthetic pool selected by 06
├── synthetic_wrought_generator_scores.csv # Generator score table from 06
├── generator_consistency_runs.csv  # Per-seed generator comparison from 07
├── generator_consistency_summary.csv # Aggregated consistency metrics from 07
└── wrought_alloys_final.csv
```

---

## Interactive use

Open any notebook in Jupyter or VS Code and run cells top to bottom. Notebooks 01 and 02 should be run before 06 and 05 so `hyperparams_config.json` is populated; if it is missing, other notebooks fall back to default hyperparameters.
