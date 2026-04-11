"""Run pipeline steps in order: 01 -> 02 -> 06_wrought -> 05_wrought (quick check, wrought only)."""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_step(name, fn):
    print(f"\n{'='*60}\nSTEP: {name}\n{'='*60}")
    try:
        fn()
        print(f"OK: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name}\n{e}")
        import traceback
        traceback.print_exc()
        return False

# --- 01: Forward tuning config sanity check (no El target) ---
def step01():
    from utils import load_hyperparams, save_hyperparams

    config = load_hyperparams() or {}
    wrought = config.get('wrought') or {}
    by_target = wrought.get('by_target') or {}

    if not by_target:
        print("01: wrought.by_target missing. Run 01 notebook to generate per-target models.")
        return

    if 'El (%)' in by_target:
        by_target.pop('El (%)', None)
        save_hyperparams({'wrought': {'by_target': by_target}})
        print("01: removed El (%) from wrought.by_target.")
    else:
        print("01: no El (%) target present in wrought.by_target.")

# --- 02: Backward GMM ---
def step02():
    from utils import load_backward_gmm_params
    w = load_backward_gmm_params('wrought')
    print("02: GMM params loaded.", "wrought OK" if w else "missing backward.wrought.GMM — run 02 notebook")

# --- 06: Generate synthetic (wrought) ---
def step06_wrought():
    import pandas as pd
    import numpy as np
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
    from utils import (
        load_hyperparams,
        load_backward_gmm_params,
        load_backward_generator_params,
        load_backward_selection_config,
        get_default_hyperparams,
    )

    INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']
    WROUGHT_PATH = 'wrought_alloys_final.csv'
    N_SAMPLES = 1000  # small for quick run

    def load_wrought(path):
        with open(path, 'rb') as f:
            head = f.read(2)
        if head == b'PK':
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin-1')
        exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper', 'El (%)']
        targets = [c for c in df.columns if c not in exclude]
        for col in INPUT_COLS + targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
        return df, targets

    def normalize_compositions(df):
        out = df.copy()
        out[out < 0] = 0
        out['_sum'] = out.sum(axis=1)
        out = out[out['_sum'] > 0.1]
        for c in INPUT_COLS:
            out[c] = (out[c] / out['_sum']) * 100
        out = out.drop(columns=['_sum']).fillna(0.0).reset_index(drop=True)
        return out

    def build_model(name, params):
        p = params.copy() if params else get_default_hyperparams(name)
        if name == 'XGBoost':
            return xgb.XGBRegressor(objective='reg:squarederror', **{k: v for k, v in p.items()})
        if name == 'RandomForest':
            return RandomForestRegressor(**{k: v for k, v in p.items()})
        if name == 'GradientBoosting':
            return GradientBoostingRegressor(**{k: v for k, v in p.items()})
        if name == 'HistGradientBoosting':
            return HistGradientBoostingRegressor(**{k: v for k, v in p.items()})
        if name == 'ExtraTrees':
            return ExtraTreesRegressor(**{k: v for k, v in p.items()})
        if name == 'AdaBoost':
            return AdaBoostRegressor(**{k: v for k, v in p.items()})
        if name == 'Bagging':
            return BaggingRegressor(**{k: v for k, v in p.items()})
        if name == 'SVR':
            return Pipeline([('scaler', StandardScaler()), ('model', SVR(**{k: v for k, v in p.items() if k != 'random_state'}))])
        if name == 'KNN':
            return Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(**{k: v for k, v in p.items() if k != 'random_state'}))])
        if name == 'MLP':
            return Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(**{k: v for k, v in p.items()}))])
        if name == 'CatBoost' and CatBoostRegressor is not None:
            return CatBoostRegressor(verbose=False, **{k: v for k, v in p.items()})
        if name == 'LightGBM' and LGBMRegressor is not None:
            return LGBMRegressor(verbose=-1, **{k: v for k, v in p.items()})
        return None

    def sample_with_generator(name, model, n_samples):
        samples, _ = model.sample(n_samples)
        sampled = pd.DataFrame(samples, columns=INPUT_COLS)
        sampled = normalize_compositions(sampled)
        sampled['_generator'] = name
        return sampled

    def label_with_forward_models(gen_df, df_real, by_target):
        out = gen_df.copy()
        for target, spec in by_target.items():
            if target == 'El (%)':
                continue
            if target not in df_real.columns or df_real[target].notna().sum() < 10:
                continue
            model = build_model(spec.get('model'), spec.get('params'))
            if model is None:
                continue
            df_t = df_real.dropna(subset=[target])
            model.fit(df_t[INPUT_COLS], df_t[target])
            out[target] = model.predict(out[INPUT_COLS])
        return out

    def build_eval_targets(df_real, property_cols):
        targets = {}
        if 'UTS (MPa)' in property_cols:
            targets['UTS (MPa)'] = 550.0
        for col in property_cols:
            series = pd.to_numeric(df_real[col], errors='coerce').dropna()
            if len(series) < 10:
                continue
            if col not in targets:
                targets[col] = float(series.median())
        return targets

    def backward_metric(pool_df, targets, top_k):
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

    def realism_metric(real_df, gen_df, cols):
        eps = 1e-8
        total = 0.0
        used = 0
        for col in cols:
            if col not in real_df.columns or col not in gen_df.columns:
                continue
            r = pd.to_numeric(real_df[col], errors='coerce').dropna()
            g = pd.to_numeric(gen_df[col], errors='coerce').dropna()
            if len(r) < 10 or len(g) < 10:
                continue
            r_mean, g_mean = float(r.mean()), float(g.mean())
            r_std, g_std = float(r.std()), float(g.std())
            mean_gap = abs(g_mean - r_mean) / (abs(r_mean) + eps)
            std_gap = abs(g_std - r_std) / (abs(r_std) + eps)
            total += mean_gap + std_gap
            used += 1
        return float(total / max(used, 1))

    def normalize_metric_scores(raw_scores, higher_better=False):
        values = np.array(list(raw_scores.values()), dtype=float)
        mn, mx = float(values.min()), float(values.max())
        if abs(mx - mn) < 1e-12:
            return {k: 1.0 for k in raw_scores}
        out = {}
        for k, v in raw_scores.items():
            scaled = (float(v) - mn) / (mx - mn)
            out[k] = float(scaled if higher_better else (1.0 - scaled))
        return out

    df_real, _ = load_wrought(WROUGHT_PATH)
    X_real = df_real[INPUT_COLS]
    by_target = (load_hyperparams('wrought') or {}).get('by_target') or {}
    if not by_target:
        raise RuntimeError("wrought.by_target is empty. Run 01 first.")

    gmm_params = load_backward_gmm_params('wrought') or get_default_hyperparams('GMM')
    bgmm_params = load_backward_generator_params('wrought', 'BGMM') or get_default_hyperparams('BGMM')
    selection_cfg = load_backward_selection_config('wrought') or get_default_hyperparams('BACKWARD_SYNTHETIC_SELECTION')
    weights = (selection_cfg or {}).get('weights') or {}
    w_backward = float(weights.get('backward', 0.5))
    w_forward = float(weights.get('forward_realism', 0.25))
    w_comp = float(weights.get('composition_realism', 0.25))
    top_k = int((selection_cfg or {}).get('top_k', 200))

    gmm = GaussianMixture(**gmm_params)
    gmm.fit(X_real)
    bgmm = BayesianGaussianMixture(**bgmm_params)
    bgmm.fit(X_real)

    pools = {
        'GMM': sample_with_generator('GMM', gmm, N_SAMPLES),
        'BGMM': sample_with_generator('BGMM', bgmm, N_SAMPLES),
    }
    pools = {name: label_with_forward_models(df_pool, df_real, by_target) for name, df_pool in pools.items()}

    property_cols = [c for c in pools['GMM'].columns if c not in INPUT_COLS and c not in ['El (%)', '_generator']]
    eval_targets = build_eval_targets(df_real, property_cols)

    raw_backward = {name: backward_metric(df_pool, eval_targets, top_k) for name, df_pool in pools.items()}
    raw_forward = {name: realism_metric(df_real, df_pool, property_cols) for name, df_pool in pools.items()}
    raw_comp = {name: realism_metric(df_real, df_pool, INPUT_COLS) for name, df_pool in pools.items()}

    norm_backward = normalize_metric_scores(raw_backward, higher_better=False)
    norm_forward = normalize_metric_scores(raw_forward, higher_better=False)
    norm_comp = normalize_metric_scores(raw_comp, higher_better=False)

    total_scores = {}
    for name in pools:
        total_scores[name] = (
            w_backward * norm_backward[name]
            + w_forward * norm_forward[name]
            + w_comp * norm_comp[name]
        )
    best_name = max(total_scores, key=total_scores.get)
    gen_df = pools[best_name].drop(columns=['_generator'], errors='ignore')

    score_table = pd.DataFrame(
        [
            {
                'generator': name,
                'backward_raw': raw_backward[name],
                'forward_realism_raw': raw_forward[name],
                'composition_realism_raw': raw_comp[name],
                'backward_score': norm_backward[name],
                'forward_realism_score': norm_forward[name],
                'composition_realism_score': norm_comp[name],
                'total_score': total_scores[name],
            }
            for name in pools
        ]
    ).sort_values('total_score', ascending=False)
    score_table.to_csv('synthetic_wrought_generator_scores.csv', index=False)

    gen_df.to_csv('synthetic_wrought.csv', index=False)
    print("06_wrought: generated pools with sizes", {k: len(v) for k, v in pools.items()})
    print("06_wrought: selected generator =", best_name)
    print("06_wrought: score table")
    print(score_table.to_string(index=False))
    print("06_wrought: saved synthetic_wrought.csv with", len(gen_df), "rows, cols", list(gen_df.columns))

# --- 05 backward wrought (quick check) ---
def step05_wrought():
    import pandas as pd
    import numpy as np
    pool = pd.read_csv('synthetic_wrought.csv')
    INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']
    prop_cols = [c for c in pool.columns if c not in INPUT_COLS and c != 'El (%)']
    TARGETS = {}
    if 'UTS (MPa)' in pool.columns:
        TARGETS['UTS (MPa)'] = 550
    if prop_cols:
        first_prop = prop_cols[0]
        TARGETS[first_prop] = pool[first_prop].median()
    if not TARGETS:
        print("05_wrought: no property columns in pool")
        return
    df = pool.copy()
    total_error = np.zeros(len(df))
    for col, target_val in TARGETS.items():
        scale = max(abs(target_val), 1.0) if target_val != 0 else 1.0
        total_error += np.abs(df[col].values - target_val) / scale
    df['Total_Error'] = total_error
    winners = df.sort_values('Total_Error').head(3)
    print("05_wrought: top 3 candidates for", TARGETS)
    for i, (_, row) in enumerate(winners.iterrows()):
        print("  Candidate", i+1, ":", {c: row[c] for c in TARGETS})

if __name__ == '__main__':
    run_step("01 Forward tuning", step01)
    run_step("02 Backward GMM", step02)
    run_step("06 Synthetic wrought", step06_wrought)
    run_step("05 Backward wrought", step05_wrought)
    print("\nPipeline run complete.")
