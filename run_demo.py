"""Quick demo run - loads wrought and cast data, runs minimal training, shows output."""
import pandas as pd
import numpy as np
import re
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))
INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']

# Load wrought (handle PK/xlsx-as-csv)
def load_wrought(path):
    with open(path, 'rb') as f:
        if f.read(2) == b'PK':
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='latin-1', errors='replace')
    exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper']
    targets = [c for c in df.columns if c not in exclude]
    for col in INPUT_COLS + targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
    return df, targets

# Load wrought
print("="*60)
print("LOADING WROUGHT ALLOYS: wrought_alloys_final.csv")
print("="*60)
df_w, targets_w = load_wrought('wrought_alloys_final.csv')
print(f"Shape: {df_w.shape}, Targets: {len(targets_w)}")
print(f"Sample targets: {targets_w[:5]}")

# Quick train on first 2 targets
for target in targets_w[:2]:
    df_t = df_w.dropna(subset=[target])
    if len(df_t) < 30:
        continue
    X = df_t[INPUT_COLS]
    y = df_t[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n  {target}: R2={r2_score(y_test, pred):.4f}, MAE={mean_absolute_error(y_test, pred):.2f}")

# Load cast
print("\n" + "="*60)
print("LOADING CAST ALLOYS: cleaned_cast_dataset.csv")
print("="*60)
df_c = pd.read_csv('cleaned_cast_dataset.csv')
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
ext = df_c['Alloy Name'].apply(extract)
comp_df = pd.DataFrame(list(ext)).fillna(0.0)
for c in INPUT_COLS:
    if c not in comp_df.columns: comp_df[c] = 0.0
df_c = pd.concat([df_c, comp_df], axis=1)
df_c = df_c[df_c['Al'] > 1.0].reset_index(drop=True)
exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper', 'Standard']
targets_c = [c for c in df_c.columns if c not in exclude]

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
for col in targets_c:
    df_c[col] = df_c[col].apply(clean_val)

print(f"Shape: {df_c.shape}, Targets: {len(targets_c)}")

# Quick train
for target in [t for t in targets_c if 'Strength' in t][:2]:
    df_t = df_c.dropna(subset=[target])
    if len(df_t) < 10: continue
    X = df_t[INPUT_COLS]
    y = df_t[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n  {target}: R2={r2_score(y_test, pred):.4f}, MAE={mean_absolute_error(y_test, pred):.2f}")

print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)
