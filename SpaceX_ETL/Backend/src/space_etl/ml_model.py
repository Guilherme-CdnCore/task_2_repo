from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_DIR = BASE_DIR / "data" / "clean"
MODEL_DIR = BASE_DIR / "out"
MODEL_PATH = MODEL_DIR / "launch_success_model.joblib"


def _load_data() -> pd.DataFrame:
    launches = pd.read_csv(DATA_DIR / "launches.csv")
    rockets = pd.read_csv(DATA_DIR / "rockets.csv")
    pads = pd.read_csv(DATA_DIR / "launchpads.csv")
    rockets_small = rockets[["id", "name"]].rename(columns={"id": "rocket", "name": "rocket_name"})
    pads_small = pads[["id", "name", "region"]].rename(columns={"id": "launchpad", "name": "pad_name", "region": "pad_region"})
    df = launches.merge(rockets_small, on="rocket", how="left").merge(pads_small, on="launchpad", how="left")
    return df


def _prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Target: success (drop NaNs)
    df = df.copy()
    df = df[df["success"].isin([True, False])]
    y = df["success"].astype(int)


    df["details_len"] = df.get("details", pd.Series([""] * len(df))).fillna("").astype(str).str.len()
    df["reuse_count"] = df.get("core_0_flight", pd.Series([0] * len(df))).fillna(0)

    X = df[["reuse_count", "details_len", "rocket_name", "pad_region"]].fillna({
        "rocket_name": "Unknown",
        "pad_region": "Unknown",
    })
    return X, y


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def train_and_save_model() -> str:
    df = _load_data()
    X, y = _prepare(df)
    cat = ["rocket_name", "pad_region"]
    num = ["reuse_count", "details_len"]
    pipe = build_pipeline(cat, num)
    pipe.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return str(MODEL_PATH)


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model was not found. Train it first.")
    return joblib.load(MODEL_PATH)


def predict_success_probability(reuse_count: int, details_len: int, rocket_name: str, pad_region: str) -> float:
    model = load_model()
    X = pd.DataFrame([{ 
        "reuse_count": reuse_count,
        "details_len": details_len,
        "rocket_name": rocket_name or "Unknown",
        "pad_region": pad_region or "Unknown",
    }])
    proba = model.predict_proba(X)[0, 1]
    return float(np.clip(proba, 0.0, 1.0))


