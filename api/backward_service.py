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


def row_to_candidate(row, prop_cols: list | dict, total_error: float = None) -> dict:
    composition = normalize_composition_from_row(row)
    if isinstance(prop_cols, dict):
        prop_cols = list(prop_cols.keys())
    properties = {}
    for col in prop_cols:
        if col in row:
            val = float(row[col])
            properties[col] = val
            col_strip = col.strip()
            if col_strip != col:
                properties[col_strip] = val
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
    pool_col_map = {c.strip(): c for c in pool.columns}

    # Normalize targets mapping to exact pool column names (handling trailing spaces like 'Thermal Diffusivity ')
    normalized_targets = {}
    for k, v in targets.items():
        if v is None:
            continue
        try:
            val_float = float(v)
        except (ValueError, TypeError):
            continue
        k_clean = str(k).strip()
        if k in pool.columns:
            normalized_targets[k] = val_float
        elif k_clean in pool_col_map:
            normalized_targets[pool_col_map[k_clean]] = val_float

    if not normalized_targets:
        available = [c.strip() for c in pool.columns if c not in INPUT_COLS and c != "El (%)"]
        raise ValueError(
            f"None of the target keys found in pool. Available property columns: {available}"
        )

    df = pool.copy()
    total_error = np.zeros(len(df))
    for col, target_val in normalized_targets.items():
        scale = max(abs(target_val), 1.0) if target_val != 0 else 1.0
        total_error += np.abs(df[col].values - target_val) / scale
    df["Total_Error"] = total_error
    winners = df.sort_values("Total_Error")

    all_prop_cols = [c for c in pool.columns if c not in INPUT_COLS and c != "El (%)"]

    candidates = []
    seen_recipes = set()

    # Pass 1: Select distinct candidates with composition diversity (at least 0.25 wt% element difference)
    for _, row in winners.iterrows():
        cand = row_to_candidate(row, all_prop_cols, total_error=row["Total_Error"])
        recipe_key = cand["recipe"]
        if recipe_key in seen_recipes:
            continue

        is_dup = False
        for existing in candidates:
            max_elem_diff = max(
                abs(cand["composition"][el] - existing["composition"][el])
                for el in INPUT_COLS
            )
            if max_elem_diff < 0.25:
                is_dup = True
                break

        if is_dup:
            continue

        seen_recipes.add(recipe_key)
        candidates.append(cand)
        if len(candidates) >= top_k:
            break

    # Pass 2: If diversity filter was too strict and len(candidates) < top_k, fill with remaining unique recipes
    if len(candidates) < top_k:
        for _, row in winners.iterrows():
            cand = row_to_candidate(row, all_prop_cols, total_error=row["Total_Error"])
            if cand["recipe"] not in seen_recipes:
                seen_recipes.add(cand["recipe"])
                candidates.append(cand)
                if len(candidates) >= top_k:
                    break

    # Pass 3: If still under top_k (extreme edge case), fill from winners
    if len(candidates) < top_k:
        for _, row in winners.iterrows():
            cand = row_to_candidate(row, all_prop_cols, total_error=row["Total_Error"])
            candidates.append(cand)
            if len(candidates) >= top_k:
                break

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
