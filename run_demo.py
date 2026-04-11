"""Quick demo run - loads wrought data, runs minimal training, shows output."""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))
INPUT_COLS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']


def load_wrought(path):
    with open(path, 'rb') as f:
        if f.read(2) == b'PK':
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='latin-1', errors='replace')
    exclude = INPUT_COLS + ['Series', 'Parent Alloy', 'Alloy Name', 'AlloyNumber', 'Temper', 'El (%)']
    targets = [c for c in df.columns if c not in exclude]
    for col in INPUT_COLS + targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[INPUT_COLS] = df[INPUT_COLS].fillna(0.0)
    return df, targets


print("=" * 60)
print("LOADING WROUGHT ALLOYS: wrought_alloys_final.csv")
print("=" * 60)
df_w, targets_w = load_wrought('wrought_alloys_final.csv')
print(f"Shape: {df_w.shape}, Targets: {len(targets_w)}")
print(f"Sample targets: {targets_w[:5]}")

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

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
