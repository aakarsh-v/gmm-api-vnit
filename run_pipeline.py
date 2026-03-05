"""Run pipeline steps in order: 01 -> 02 -> 06_wrought, 06_cast -> 05_wrought (quick check)."""
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

# --- 01: Forward tuning (minimal: 2 targets each to save time) ---
def step01():
    import pandas as pd
    import numpy as np
    import re
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score
    import warnings
    warnings.filterwarnings('ignore')
    from utils import save_per_target_hyperparams

    INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']
    WROUGHT_PATH = 'wrought_alloys_final.csv'
    CAST_PATH = 'cleaned_cast_dataset.csv'

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
        exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper']
        targets = [c for c in df.columns if c not in exclude]
        for col in INPUT_COLS + targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
        return df, targets

    def load_cast(path):
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')
        def extract(text):
            if pd.isna(text): return {}
            text = str(text).replace('AI', 'Al')
            m = re.search(r'\((.*?)\)', text)
            if not m: return {}
            f = m.group(1)
            comp = {}
            for el in INPUT_COLS:
                if el == 'Al': continue
                p = re.compile(rf"{el}(\d*\.?\d*)")
                h = p.search(f)
                if h:
                    v = h.group(1)
                    comp[el] = float(v) if (v and v != '.') else 0.5
            t = sum(comp.values())
            comp['Al'] = 100 - t if 0 < t < 100 else 0.0
            return comp
        ext = df['Alloy Name'].apply(extract)
        comp_df = pd.DataFrame(list(ext)).fillna(0.0)
        for c in INPUT_COLS:
            if c not in comp_df.columns: comp_df[c] = 0.0
        df = pd.concat([df, comp_df], axis=1)
        df = df[df['Al'] > 1.0].reset_index(drop=True)
        exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper', 'Standard']
        targets = [c for c in df.columns if c not in exclude]
        def clean_val(val):
            if pd.isna(val): return np.nan
            s = str(val).strip()
            if '-' in s:
                try:
                    parts = [re.sub(r'[^0-9.]', '', p) for p in s.split('-')]
                    nums = [float(p) for p in parts if p]
                    return sum(nums)/len(nums) if nums else np.nan
                except: pass
            m = re.search(r"[-+]?[0-9]*\.[0-9]+|[0-9]+", s)
            return float(m.group()) if m else np.nan
        for col in targets:
            df[col] = df[col].apply(clean_val)
        return df, targets

    XGB_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8], 'min_child_weight': [1]}
    RF_PARAMS = {'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_split': [2], 'min_samples_leaf': [1]}
    GB_PARAMS = {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1], 'subsample': [0.8]}

    def tune_per_target(df, targets, input_cols, min_samples=30):
        by_target = {}
        for target in targets[:3]:  # first 3 targets only for quick run
            df_t = df.dropna(subset=[target])
            if len(df_t) < min_samples:
                continue
            X = df_t[input_cols]
            y = df_t[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_r2, best_name, best_params = -float('inf'), None, None
            for name, model, params in [
                ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', random_state=42), XGB_PARAMS),
                ('RandomForest', RandomForestRegressor(random_state=42), RF_PARAMS),
                ('GradientBoosting', GradientBoostingRegressor(random_state=42), GB_PARAMS),
            ]:
                search = RandomizedSearchCV(model, params, n_iter=2, cv=2, random_state=42)
                search.fit(X_train, y_train)
                r2 = r2_score(y_test, search.predict(X_test))
                if r2 > best_r2:
                    best_r2, best_name = r2, name
                    best_params = {**search.best_params_, 'random_state': 42}
            if best_name:
                by_target[target] = {'model': best_name, 'params': best_params}
                print(f"  {target[:40]} -> {best_name} R2={best_r2:.4f}")
        return by_target

    df_w, tw = load_wrought(WROUGHT_PATH)
    df_c, tc = load_cast(CAST_PATH)
    cast_targets = [t for t in tc if 'Strength' in t or 'Conductivity' in t or 'Modulus' in t][:5]
    if not cast_targets:
        cast_targets = tc[:5]

    wrought_by_target = tune_per_target(df_w, tw, INPUT_COLS, 30)
    cast_by_target = tune_per_target(df_c, cast_targets, INPUT_COLS, 10)

    if wrought_by_target:
        save_per_target_hyperparams('wrought', wrought_by_target)
    if cast_by_target:
        save_per_target_hyperparams('cast', cast_by_target)
    print("01 done. by_target saved.")

# --- 02: Backward GMM ---
def step02():
    from utils import load_backward_gmm_params, save_hyperparams, get_default_hyperparams
    # Just ensure config has backward.wrought.GMM and backward.cast.GMM (already there)
    w = load_backward_gmm_params('wrought')
    c = load_backward_gmm_params('cast')
    print("02: GMM params loaded.", "wrought" if w else "missing wrought", "cast" if c else "missing cast")

# --- 06: Generate synthetic (wrought) ---
def step06_wrought():
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from utils import load_hyperparams, load_backward_gmm_params, get_default_hyperparams

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
        exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper']
        targets = [c for c in df.columns if c not in exclude]
        for col in INPUT_COLS + targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
        return df, targets

    def build_model(name, params):
        p = params.copy() if params else get_default_hyperparams(name)
        if name == 'XGBoost':
            return xgb.XGBRegressor(objective='reg:squarederror', **{k: v for k, v in p.items()})
        if name == 'RandomForest':
            return RandomForestRegressor(**{k: v for k, v in p.items()})
        if name == 'GradientBoosting':
            return GradientBoostingRegressor(**{k: v for k, v in p.items()})
        return None

    df_real, _ = load_wrought(WROUGHT_PATH)
    gmm_params = load_backward_gmm_params('wrought') or get_default_hyperparams('GMM')
    X_real = df_real[INPUT_COLS]
    gmm = GaussianMixture(**gmm_params)
    gmm.fit(X_real)
    samples, _ = gmm.sample(N_SAMPLES)
    gen_df = pd.DataFrame(samples, columns=INPUT_COLS)
    gen_df[gen_df < 0] = 0
    gen_df['_sum'] = gen_df.sum(axis=1)
    gen_df = gen_df[gen_df['_sum'] > 0.1]
    for c in INPUT_COLS:
        gen_df[c] = (gen_df[c] / gen_df['_sum']) * 100
    gen_df = gen_df.drop(columns=['_sum']).fillna(0.0).reset_index(drop=True)

    by_target = (load_hyperparams('wrought') or {}).get('by_target') or {}
    if not by_target:
        raise RuntimeError("wrought.by_target is empty. Run 01 first.")
    for target, spec in by_target.items():
        if target not in df_real.columns or df_real[target].notna().sum() < 10:
            continue
        model = build_model(spec.get('model'), spec.get('params'))
        if model is None:
            continue
        df_t = df_real.dropna(subset=[target])
        model.fit(df_t[INPUT_COLS], df_t[target])
        gen_df[target] = model.predict(gen_df[INPUT_COLS])
    gen_df.to_csv('synthetic_wrought.csv', index=False)
    print("06_wrought: saved synthetic_wrought.csv with", len(gen_df), "rows, cols", list(gen_df.columns))

# --- 05 backward wrought (quick check) ---
def step05_wrought():
    import pandas as pd
    import numpy as np
    pool = pd.read_csv('synthetic_wrought.csv')
    INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']
    prop_cols = [c for c in pool.columns if c not in INPUT_COLS]
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
