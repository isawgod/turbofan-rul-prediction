
"""

Input raw schema (per row):
    unit, cycle, setting_1..3, s01..s21

This module produces the feature set expected by the trained model:
    cycle, cycle_norm,
    s04, s11, s12, s15, s17,
    s04_rollmean, s11_rollmean, s12_rollmean, s17_rollmean

Notes:
- cycle_norm is computed per-unit = cycle / max(cycle in that unit batch)
- Rolling statistics are computed per-unit with window=5, sorted by cycle.
- If the input has mixed/unsorted units, we will sort within each group.
"""

from __future__ import annotations
from typing import List
import pandas as pd

# Final features expected by model
FEATURES: List[str] = [
    "cycle", "cycle_norm",
    "s04", "s11", "s12", "s15", "s17",
    "s04_rollmean", "s11_rollmean", "s12_rollmean", "s17_rollmean",
]

SENSOR_BASE = ["s04", "s11", "s12", "s15", "s17"]
ROLL_WINDOW = 5


def _normalize_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_norm per unit: cycle / max(cycle) within the provided batch."""
    if "unit" not in df.columns or "cycle" not in df.columns:
        raise ValueError("Expected columns 'unit' and 'cycle' in raw input.")
    max_cycle = df.groupby("unit")["cycle"].transform("max").astype(float)
    df = df.copy()
    df["cycle_norm"] = df["cycle"] / max_cycle.clip(lower=1.0)
    return df


def _rolling_means(df: pd.DataFrame, cols: List[str], window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Compute per-unit rolling means for selected sensor columns."""
    if "unit" not in df.columns or "cycle" not in df.columns:
        raise ValueError("Expected columns 'unit' and 'cycle' in raw input.")
    df = df.copy()
    # sort by unit-cycle to ensure correct rolling order
    df = df.sort_values(["unit", "cycle"])
    for col in cols:
        roll_name = f"{col}_rollmean"
        df[roll_name] = (
            df.groupby("unit")[col]
              .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )
    return df


def make_features_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw CMAPSS rows into the exact features needed by the model.
    Raw must contain:
        unit, cycle, setting_1..3 (optional), s01..s21 (required sensors)

    Returns a DataFrame with columns:
        FEATURES (see above), aligned row-wise to input.
    """
    required_sensors = [f"s{i:02d}" for i in range(1, 22)]
    missing = [c for c in ["unit", "cycle"] + required_sensors if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    df = raw.copy()

    # keep only relevant sensors for this model
    needed = ["unit", "cycle"] + SENSOR_BASE
    df = df[needed]

    # add normalized cycle
    df = _normalize_cycle(df)

    # add rolling means per-unit
    df = _rolling_means(df, SENSOR_BASE, window=ROLL_WINDOW)

    # arrange final feature order
    keep_cols = FEATURES.copy()
    return df[keep_cols]
