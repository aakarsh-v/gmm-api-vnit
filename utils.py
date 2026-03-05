"""
Shared utilities for hyperparameter management.
Used by tuning and model notebooks to load/save best hyperparameters.
"""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hyperparams_config.json")


def load_hyperparams(dataset: str = None, model_name: str = None):
    """
    Load hyperparameters from config.

    Args:
        dataset: 'wrought', 'cast', or 'backward'
        model_name: 'XGBoost', 'RandomForest', 'GradientBoosting', 'GMM',
                    or for backward: 'wrought' / 'cast' to get { 'GMM': {...} }

    Returns:
        dict of params, or None if not found
    """
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        if dataset is None:
            return config
        if dataset not in config:
            return None
        if model_name is None:
            return config[dataset]
        if model_name in config[dataset]:
            return config[dataset][model_name]
        return None
    except (json.JSONDecodeError, IOError):
        return None


def load_hyperparams_for_target(dataset: str, target: str):
    """
    Load per-target best model and hyperparameters (from by_target).

    Args:
        dataset: 'wrought' or 'cast'
        target: target column name (e.g. 'UTS (MPa)', 'EC Volume (% IACS)')

    Returns:
        dict with 'model' and 'params', or None if not found
    """
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        if dataset not in config:
            return None
        by_target = config[dataset].get("by_target") or {}
        return by_target.get(target)
    except (json.JSONDecodeError, IOError):
        return None


def save_per_target_hyperparams(dataset: str, by_target_dict: dict):
    """
    Save per-target best model and params under dataset.by_target.
    Merges into existing config; does not remove other dataset sections.

    Args:
        dataset: 'wrought' or 'cast'
        by_target_dict: e.g. { 'UTS (MPa)': { 'model': 'XGBoost', 'params': {...} }, ... }
    """
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    if dataset not in config:
        config[dataset] = {}
    if "by_target" not in config[dataset]:
        config[dataset]["by_target"] = {}
    config[dataset]["by_target"].update(by_target_dict)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def load_backward_gmm_params(dataset: str):
    """
    Load GMM hyperparameters for backward pipeline (wrought or cast).

    Args:
        dataset: 'wrought' or 'cast'

    Returns:
        dict of GMM params (e.g. n_components, covariance_type, random_state), or None
    """
    section = load_hyperparams("backward", dataset)
    if section and isinstance(section, dict) and "GMM" in section:
        return section["GMM"]
    # Legacy: single backward.GMM
    return load_hyperparams("backward", "GMM")


def save_hyperparams(config_updates: dict):
    """
    Update and save hyperparameters config.
    
    Args:
        config_updates: Nested dict e.g.
            {'wrought': {'XGBoost': {...}, 'RandomForest': {...}},
             'backward': {'GMM': {'n_components': 12}}}
    """
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    for dataset, models in config_updates.items():
        if dataset not in config:
            config[dataset] = {}
        for model_name, params in models.items():
            config[dataset][model_name] = params
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_default_hyperparams(model_name: str):
    """Fallback defaults when config is missing."""
    defaults = {
        "XGBoost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "RandomForest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "GradientBoosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
        "GMM": {
            "n_components": 12,
            "covariance_type": "full",
            "random_state": 42,
        },
    }
    return defaults.get(model_name, {})
