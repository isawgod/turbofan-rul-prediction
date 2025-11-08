#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict RUL for NASA CMAPSS FD001.

- Works with RAW CMAPSS input (unit, cycle, s01..s21) or pre-computed FEATURES.
- Exposes a single function: predict_rul(df, model=None, auto_fe=False) -> pd.Series
"""

from __future__ import annotations
from typing import List
from pathlib import Path
import sys

import joblib
import pandas as pd

# --- Import feature pipeline (robust in all run modes) ---
try:
    # Case A: executed as a package -> python -m src.predict
    from .features import FEATURES as MODEL_FEATURES, make_features_from_raw
except Exception:
    try:
        # Case B: imported from project root -> from src.predict import ...
        from src.features import FEATURES as MODEL_FEATURES, make_features_from_raw
    except Exception:
        # Case C: last resort, running directly inside src/
        sys.path.append(str(Path(__file__).resolve().parent))
        from features import FEATURES as MODEL_FEATURES, make_features_from_raw


# ---------------- Paths & helpers ---------------- #

MODEL_FILENAME = "rul_gb_fd001.joblib"

def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]

def load_model(model_name: str = MODEL_FILENAME):
    root = project_root_from_file()
    model_path = root / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure you saved the model to 'models/{model_name}'."
        )
    return joblib.load(model_path)

def has_raw_schema(df: pd.DataFrame) -> bool:
    sensors = [f"s{i:02d}" for i in range(1, 22)]
    return ("unit" in df.columns and "cycle" in df.columns and all(c in df.columns for c in sensors))

def has_feature_schema(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in MODEL_FEATURES)


# ---------------- Public API ---------------- #

def predict_rul(df: pd.DataFrame, model=None, auto_fe: bool = False) -> pd.Series:
    """
    Predict RUL for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Either RAW CMAPSS rows (must contain: unit, cycle, s01..s21),
        or pre-computed feature columns exactly as in MODEL_FEATURES.
    model : fitted sklearn regressor or None
        If None, the function will load models/rul_gb_fd001.joblib.
    auto_fe : bool
        If True and RAW is detected, compute features automatically.

    Returns
    -------
    pd.Series
        Predicted RUL values, name='RUL_pred'
    """
    if model is None:
        model = load_model()

    if has_feature_schema(df):
        feats = df[MODEL_FEATURES].copy()
    elif has_raw_schema(df):
        if not auto_fe:
            raise ValueError(
                "Input looks like RAW CMAPSS (unit, cycle, s01..s21). "
                "Set auto_fe=True to build features automatically, or pass feature columns."
            )
        feats = make_features_from_raw(df)
    else:
        raise ValueError(
            "Input schema not recognized. Provide RAW CMAPSS (unit, cycle, s01..s21) "
            "or a DataFrame with feature columns:\n"
            f"{MODEL_FEATURES}"
        )

    preds = model.predict(feats)
    return pd.Series(preds, name="RUL_pred")


# ---------------- Optional CLI ---------------- #

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Predict RUL for CMAPSS FD001.")
    p.add_argument("--input", "-i", type=Path, required=True, help="CSV path (RAW or features).")
    p.add_argument("--output", "-o", type=Path, default=None, help="Where to save predictions CSV.")
    p.add_argument("--auto-fe", action="store_true", help="Build features from RAW automatically.")
    p.add_argument("--model", "-m", type=str, default=MODEL_FILENAME, help="Model filename in 'models/'.")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    m = load_model(args.model)
    y = predict_rul(df, model=m, auto_fe=args.auto_fe)
    out = df.copy()
    out["RUL_pred"] = y.values

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"[OK] Saved -> {args.output}")
    else:
        print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    _cli()
