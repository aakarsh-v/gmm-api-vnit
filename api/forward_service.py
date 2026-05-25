"""Forward prediction: composition -> alloy properties."""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from synthetic_generation_core import INPUT_COLS, build_model as _build_model_core
from utils import get_default_hyperparams, load_hyperparams, load_hyperparams_for_target

WROUGHT_PATH = os.path.join(PROJECT_ROOT, "wrought_alloys_final.csv")

_trainer = None
_loaded = False


def build_model(name: str, params: dict):
    return _build_model_core(name, params, get_default_hyperparams)


class AlloyDataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        self.clean_data = None
        self.input_cols = list(INPUT_COLS)
        self.target_cols = []

    def load_and_clean(self):
        try:
            if self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
                self.raw_data = pd.read_excel(self.file_path)
            else:
                with open(self.file_path, "rb") as f:
                    if f.read(2) == b"PK":
                        self.raw_data = pd.read_excel(self.file_path)
                    else:
                        try:
                            self.raw_data = pd.read_csv(self.file_path, encoding="utf-8")
                        except UnicodeDecodeError:
                            self.raw_data = pd.read_csv(self.file_path, encoding="latin-1")
        except Exception as e:
            raise FileNotFoundError(f"Could not load wrought data: {e}") from e

        exclude = self.input_cols + [
            "Series",
            "Parent Alloy",
            "Alloy Name",
            "AlloyNumber",
            "Temper",
            "El (%)",
        ]
        self.target_cols = [c for c in self.raw_data.columns if c not in exclude]
        df = self.raw_data.copy()
        for col in self.input_cols + self.target_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_value)
        df[self.input_cols] = df[self.input_cols].fillna(0.0)
        self.clean_data = df
        return self.clean_data

    @staticmethod
    def _parse_value(val):
        if pd.isna(val) or val == "":
            return np.nan
        val = str(val).strip()
        if "-" in val:
            try:
                parts = [re.sub(r"[^\d\.]", "", p) for p in val.split("-")]
                return (float(parts[0]) + float(parts[1])) / 2
            except Exception:
                pass
        m = re.search(r"[-+]?\d*\.\d+|\d+", val)
        return float(m.group()) if m else np.nan


class ForwardModelTrainer:
    def __init__(self, data, input_cols, hyperparams=None):
        self.data = data
        self.inputs = input_cols
        self.best_models = {}
        self.metrics = {}
        self.hp = hyperparams or load_hyperparams("wrought")

    def train_all_targets(self, target_list):
        for target in target_list:
            valid_rows = self.data.dropna(subset=[target])
            if len(valid_rows) < 50:
                continue
            X = valid_rows[self.inputs]
            y = valid_rows[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            spec = load_hyperparams_for_target("wrought", target)
            if spec and spec.get("model") and spec.get("params"):
                name = spec["model"]
                params = spec["params"]
                model = build_model(name, params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                best_r2 = r2_score(y_test, preds)
                winner_name = name
                winner_model = model
            else:
                best_r2 = -float("inf")
                winner_name = None
                winner_model = None
                for name in [
                    "RandomForest",
                    "XGBoost",
                    "GradientBoosting",
                    "HistGradientBoosting",
                    "ExtraTrees",
                ]:
                    params = (
                        self.hp.get(name) if self.hp else get_default_hyperparams(name)
                    )
                    model = build_model(name, params)
                    if model is None:
                        continue
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    if r2 > best_r2:
                        best_r2 = r2
                        winner_name = name
                        winner_model = model
            if winner_model is None:
                continue
            self.best_models[target] = winner_model
            mae = mean_absolute_error(y_test, winner_model.predict(X_test))
            rmse = np.sqrt(mean_squared_error(y_test, winner_model.predict(X_test)))
            self.metrics[target] = {
                "Model": winner_name,
                "R2": best_r2,
                "MAE": mae,
                "RMSE": rmse,
            }


def normalize_composition(composition: dict) -> dict:
    """Fill missing input columns with 0."""
    return {col: float(composition.get(col, 0.0) or 0.0) for col in INPUT_COLS}


def predict_profile(composition: dict, trainer: ForwardModelTrainer) -> dict[str, float]:
    comp = normalize_composition(composition)
    input_df = pd.DataFrame([comp])[trainer.inputs]
    predictions = {}
    for target, model in trainer.best_models.items():
        if model is not None:
            predictions[target] = float(model.predict(input_df)[0])
    return predictions


def load_models(wrought_path: str = None) -> ForwardModelTrainer:
    global _trainer, _loaded
    path = wrought_path or WROUGHT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Wrought dataset not found at {path}. Place wrought_alloys_final.csv in project root."
        )
    processor = AlloyDataProcessor(path)
    df_clean = processor.load_and_clean()
    trainer = ForwardModelTrainer(df_clean, processor.input_cols)
    trainer.train_all_targets(processor.target_cols)
    _trainer = trainer
    _loaded = True
    return trainer


def get_trainer() -> ForwardModelTrainer:
    if not _loaded or _trainer is None:
        raise RuntimeError("Forward models not loaded. API startup may have failed.")
    return _trainer


def is_loaded() -> bool:
    return _loaded and _trainer is not None
