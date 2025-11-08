# scripts/train_and_export.py

from pathlib import Path
import sys
import pandas as pd
import joblib

# --- project roots & import path ---
ROOT = Path(__file__).resolve().parents[1]   # repo root
SRC  = ROOT / "src"
DATA = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))  # pozwala: from src.features import ...

# --- now imports from our package ---
from src.features import make_features_from_raw, FEATURES
from sklearn.ensemble import GradientBoostingRegressor

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # 1) load raw (FD001)
    train_raw = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    cols = ["unit","cycle"] + [f"setting_{i}" for i in range(1,4)] + [f"s{i:02d}" for i in range(1,22)]
    train_raw.columns = cols[:train_raw.shape[1]]

    # 2) target
    max_cycle = train_raw.groupby("unit")["cycle"].transform("max")
    train_raw["RUL"] = max_cycle - train_raw["cycle"]

    # 3) features + train
    raw_cols = ["unit","cycle"] + [f"s{i:02d}" for i in range(1,22)]
    X = make_features_from_raw(train_raw[raw_cols])
    y = train_raw.loc[X.index, "RUL"].reset_index(drop=True)

    model = GradientBoostingRegressor(random_state=42).fit(X, y)

    # 4) export
    out_path = MODEL_DIR / "rul_gb_fd001.joblib"
    joblib.dump(model, out_path)
    print("[OK] Saved:", out_path)

if __name__ == "__main__":
    main()
