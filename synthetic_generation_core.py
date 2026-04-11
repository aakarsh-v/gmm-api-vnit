"""Shared dual-generator synthetic data utilities for wrought pipeline."""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


INPUT_COLS = ["Al", "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ga", "V", "Ti"]


def load_wrought(path: str, input_cols=None):
    """Load wrought CSV/XLS(X) with robust encoding handling."""
    input_cols = input_cols or INPUT_COLS
    with open(path, "rb") as f:
        head = f.read(2)
    if path.endswith(".xlsx") or path.endswith(".xls") or head == b"PK":
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    exclude = input_cols + ["Series", "Parent Alloy", "Alloy Name", "AlloyNumber", "Temper", "El (%)"]
    targets = [c for c in df.columns if c not in exclude]
    for col in input_cols + targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[input_cols] = df[input_cols].fillna(0.0)
    return df, targets


def normalize_compositions(df: pd.DataFrame, input_cols=None):
    """Clip negatives and normalize compositions to 100%."""
    input_cols = input_cols or INPUT_COLS
    out = df.copy()
    out[out < 0] = 0
    out["_sum"] = out.sum(axis=1)
    out = out[out["_sum"] > 0.1]
    for c in input_cols:
        out[c] = (out[c] / out["_sum"]) * 100
    out = out.drop(columns=["_sum"]).fillna(0.0).reset_index(drop=True)
    return out


def build_model(name: str, params: dict, get_default_hyperparams):
    """Instantiate forward model by config name."""
    p = params.copy() if params else get_default_hyperparams(name)
    if name == "XGBoost":
        return xgb.XGBRegressor(objective="reg:squarederror", **{k: v for k, v in p.items()})
    if name == "RandomForest":
        return RandomForestRegressor(**{k: v for k, v in p.items()})
    if name == "GradientBoosting":
        return GradientBoostingRegressor(**{k: v for k, v in p.items()})
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**{k: v for k, v in p.items()})
    if name == "ExtraTrees":
        return ExtraTreesRegressor(**{k: v for k, v in p.items()})
    if name == "AdaBoost":
        return AdaBoostRegressor(**{k: v for k, v in p.items()})
    if name == "Bagging":
        return BaggingRegressor(**{k: v for k, v in p.items()})
    if name == "SVR":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(**{k: v for k, v in p.items() if k != "random_state"}))]
        )
    if name == "KNN":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(**{k: v for k, v in p.items() if k != "random_state"})),
            ]
        )
    if name == "MLP":
        return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(**{k: v for k, v in p.items()}))])
    if name == "CatBoost" and CatBoostRegressor is not None:
        return CatBoostRegressor(verbose=False, **{k: v for k, v in p.items()})
    if name == "LightGBM" and LGBMRegressor is not None:
        return LGBMRegressor(verbose=-1, **{k: v for k, v in p.items()})
    return None


def label_with_forward_models(gen_df: pd.DataFrame, df_real: pd.DataFrame, by_target: dict, get_default_hyperparams):
    """Train per-target forward models and label synthetic pool."""
    out = gen_df.copy()
    for target, spec in by_target.items():
        if target == "El (%)":
            continue
        if target not in df_real.columns or df_real[target].notna().sum() < 10:
            continue
        model = build_model(spec.get("model"), spec.get("params"), get_default_hyperparams)
        if model is None:
            continue
        df_t = df_real.dropna(subset=[target])
        model.fit(df_t[INPUT_COLS], df_t[target])
        out[target] = model.predict(out[INPUT_COLS])
    return out


def build_eval_targets(df_real: pd.DataFrame, property_cols):
    """Create generic backward targets for generator quality scoring."""
    targets = {}
    if "UTS (MPa)" in property_cols:
        targets["UTS (MPa)"] = 550.0
    for col in property_cols:
        series = pd.to_numeric(df_real[col], errors="coerce").dropna()
        if len(series) < 10:
            continue
        if col not in targets:
            targets[col] = float(series.median())
    return targets


def backward_metric(pool_df: pd.DataFrame, targets: dict, top_k: int):
    """Average top-k normalized target error (lower is better)."""
    if not targets:
        return 1.0
    err = np.zeros(len(pool_df))
    for col, target_val in targets.items():
        if col not in pool_df.columns:
            continue
        scale = max(abs(target_val), 1.0)
        err += np.abs(pool_df[col].values - target_val) / scale
    if len(err) == 0:
        return 1.0
    k = max(1, min(top_k, len(err)))
    return float(np.sort(err)[:k].mean())


def realism_metric(real_df: pd.DataFrame, gen_df: pd.DataFrame, cols):
    """Moment-based realism metric over selected columns (lower is better)."""
    eps = 1e-8
    total = 0.0
    used = 0
    for col in cols:
        if col not in real_df.columns or col not in gen_df.columns:
            continue
        r = pd.to_numeric(real_df[col], errors="coerce").dropna()
        g = pd.to_numeric(gen_df[col], errors="coerce").dropna()
        if len(r) < 10 or len(g) < 10:
            continue
        mean_gap = abs(float(g.mean()) - float(r.mean())) / (abs(float(r.mean())) + eps)
        std_gap = abs(float(g.std()) - float(r.std())) / (abs(float(r.std())) + eps)
        total += mean_gap + std_gap
        used += 1
    return float(total / max(used, 1))


def normalize_metric_scores(raw_scores: dict):
    """Map raw metrics to [0,1] where higher is better."""
    vals = np.array(list(raw_scores.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if abs(mx - mn) < 1e-12:
        return {k: 1.0 for k in raw_scores}
    return {k: 1.0 - ((float(v) - mn) / (mx - mn)) for k, v in raw_scores.items()}


def _seeded_params(base_params: dict, random_seed: int):
    """Override random_state when applicable."""
    params = (base_params or {}).copy()
    if random_seed is not None and "random_state" in params:
        params["random_state"] = int(random_seed)
    return params


def run_dual_generator_once(
    df_real: pd.DataFrame,
    by_target: dict,
    gmm_params: dict,
    bgmm_params: dict,
    selection_cfg: dict,
    get_default_hyperparams,
    n_samples: int = 50000,
    random_seed: int = 42,
):
    """Generate and score GMM/BGMM pools; return selected pool and score table."""
    gmm_p = _seeded_params(gmm_params, random_seed)
    bgmm_p = _seeded_params(bgmm_params, random_seed)
    if "random_state" not in gmm_p:
        gmm_p["random_state"] = int(random_seed)
    if "random_state" not in bgmm_p:
        bgmm_p["random_state"] = int(random_seed)

    weights = (selection_cfg or {}).get("weights") or {}
    w_backward = float(weights.get("backward", 0.5))
    w_forward = float(weights.get("forward_realism", 0.25))
    w_comp = float(weights.get("composition_realism", 0.25))
    top_k = int((selection_cfg or {}).get("top_k", 200))

    X_real = df_real[INPUT_COLS]
    gmm = GaussianMixture(**gmm_p)
    gmm.fit(X_real)
    bgmm = BayesianGaussianMixture(**bgmm_p)
    bgmm.fit(X_real)

    sample_gmm, _ = gmm.sample(n_samples)
    sample_bgmm, _ = bgmm.sample(n_samples)

    pool_gmm = normalize_compositions(pd.DataFrame(sample_gmm, columns=INPUT_COLS))
    pool_bgmm = normalize_compositions(pd.DataFrame(sample_bgmm, columns=INPUT_COLS))
    pool_gmm["_generator"] = "GMM"
    pool_bgmm["_generator"] = "BGMM"

    pools = {
        "GMM": label_with_forward_models(pool_gmm, df_real, by_target, get_default_hyperparams),
        "BGMM": label_with_forward_models(pool_bgmm, df_real, by_target, get_default_hyperparams),
    }

    prop_cols = [c for c in pools["GMM"].columns if c not in INPUT_COLS and c not in ["El (%)", "_generator"]]
    eval_targets = build_eval_targets(df_real, prop_cols)

    raw_backward = {name: backward_metric(df, eval_targets, top_k) for name, df in pools.items()}
    raw_forward = {name: realism_metric(df_real, df, prop_cols) for name, df in pools.items()}
    raw_comp = {name: realism_metric(df_real, df, INPUT_COLS) for name, df in pools.items()}

    norm_backward = normalize_metric_scores(raw_backward)
    norm_forward = normalize_metric_scores(raw_forward)
    norm_comp = normalize_metric_scores(raw_comp)

    total_scores = {
        name: (w_backward * norm_backward[name]) + (w_forward * norm_forward[name]) + (w_comp * norm_comp[name])
        for name in pools
    }
    best_generator = max(total_scores, key=total_scores.get)
    best_df = pools[best_generator].drop(columns=["_generator"], errors="ignore")

    score_table = pd.DataFrame(
        [
            {
                "generator": name,
                "backward_raw": raw_backward[name],
                "forward_realism_raw": raw_forward[name],
                "composition_realism_raw": raw_comp[name],
                "backward_score": norm_backward[name],
                "forward_realism_score": norm_forward[name],
                "composition_realism_score": norm_comp[name],
                "total_score": total_scores[name],
                "sample_rows": int(len(pools[name])),
                "random_seed": int(random_seed),
                "n_samples_requested": int(n_samples),
            }
            for name in pools
        ]
    ).sort_values("total_score", ascending=False)

    return {
        "best_generator": best_generator,
        "best_df": best_df,
        "pools": pools,
        "scores": score_table,
    }


def env_int(name: str, default_value: int):
    """Read integer environment variable with fallback."""
    val = os.getenv(name)
    if val is None:
        return int(default_value)
    try:
        return int(val)
    except ValueError:
        return int(default_value)
