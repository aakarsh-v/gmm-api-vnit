"""Backward search: target properties -> candidate alloy compositions."""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from synthetic_generation_core import INPUT_COLS

POOL_PATH = os.path.join(PROJECT_ROOT, "synthetic_wrought.csv")

_pool = None
_loaded = False


def normalize_composition_from_row(row) -> dict:
    comp = {el: float(row[el]) for el in INPUT_COLS}
    total = sum(comp.values())
    if total < 100.0:
        comp["Al"] = comp["Al"] + (100.0 - total)
    elif total > 100.0:
        scale = 100.0 / total
        comp = {el: comp[el] * scale for el in INPUT_COLS}
    return comp


def format_recipe(composition: dict) -> str:
    return ", ".join(
        f"{el}={composition[el]:.2f}%" for el in INPUT_COLS if composition[el] > 0.1
    )


def row_to_candidate(row, targets_in_pool: dict, total_error: float = None) -> dict:
    composition = normalize_composition_from_row(row)
    properties = {col: float(row[col]) for col in targets_in_pool}
    out = {
        "composition": composition,
        "properties": properties,
        "recipe": format_recipe(composition),
    }
    if total_error is not None:
        out["total_error"] = float(total_error)
    return out


def load_pool(pool_path: str = None) -> pd.DataFrame:
    global _pool, _loaded
    path = pool_path or POOL_PATH
    if not os.path.exists(path):
        _pool = None
        _loaded = False
        return None
    _pool = pd.read_csv(path)
    _loaded = True
    return _pool


def get_pool() -> pd.DataFrame:
    if not _loaded or _pool is None:
        raise RuntimeError(
            "Synthetic pool not loaded. Run 06_generate_synthetic_wrought.ipynb first "
            "to create synthetic_wrought.csv."
        )
    return _pool


def is_loaded() -> bool:
    return _loaded and _pool is not None


def search_targets(targets: dict, top_k: int = 3) -> list[dict]:
    pool = get_pool()
    targets_in_pool = {k: v for k, v in targets.items() if k in pool.columns}
    if not targets_in_pool:
        available = [c for c in pool.columns if c not in INPUT_COLS and c != "El (%)"]
        raise ValueError(
            f"None of the target keys found in pool. Available property columns: {available}"
        )

    df = pool.copy()
    total_error = np.zeros(len(df))
    for col, target_val in targets_in_pool.items():
        scale = max(abs(target_val), 1.0) if target_val != 0 else 1.0
        total_error += np.abs(df[col].values - target_val) / scale
    df["Total_Error"] = total_error
    winners = df.sort_values("Total_Error").head(top_k)

    candidates = []
    for _, row in winners.iterrows():
        candidates.append(
            row_to_candidate(row, targets_in_pool, total_error=row["Total_Error"])
        )
    return candidates


def pair_search(
    col_a: str,
    target_a: float,
    col_b: str,
    target_b: float,
    tolerance: float = 0.05,
) -> dict | None:
    pool = get_pool()
    missing_cols = [c for c in (col_a, col_b) if c not in pool.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in pool: {missing_cols}")

    low_a, high_a = target_a * (1 - tolerance), target_a * (1 + tolerance)
    low_b, high_b = target_b * (1 - tolerance), target_b * (1 + tolerance)

    pass_mask = pool[col_a].between(low_a, high_a) & pool[col_b].between(low_b, high_b)
    passing = pool.loc[pass_mask].copy()

    if passing.empty:
        return None

    err_a = np.abs(passing[col_a].values - target_a) / max(abs(target_a), 1.0)
    err_b = np.abs(passing[col_b].values - target_b) / max(abs(target_b), 1.0)
    passing["Pair_Error"] = err_a + err_b
    best = passing.sort_values("Pair_Error").iloc[0]

    composition = normalize_composition_from_row(best)
    properties = {col_a: float(best[col_a]), col_b: float(best[col_b])}
    extra_cols = [
        "YS (MPa)",
        "Fatigue Strength (MPa)",
        "TC (W/m-K)",
        "TE Coeff",
    ]
    for col in extra_cols:
        if col in pool.columns:
            properties[col] = float(best[col])

    return {
        "composition": composition,
        "properties": properties,
        "recipe": format_recipe(composition),
        "pair_error": float(best["Pair_Error"]),
        "tolerance": tolerance,
        "targets": {col_a: target_a, col_b: target_b},
    }
