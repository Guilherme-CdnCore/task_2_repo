from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib  # For saving/loading trained models
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # Combines different preprocessing steps
from sklearn.linear_model import LogisticRegression  # Our ML model for binary classification
from sklearn.pipeline import Pipeline  # Chains preprocessing + model together
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Data preprocessing tools


BASE_DIR = Path(__file__).resolve().parents[2]  # Points to SpaceX_ETL/Backend/
DATA_DIR = BASE_DIR / "data" / "clean"  # Where our clean CSV files are stored
MODEL_DIR = BASE_DIR / "out"  # Where we save the trained model
MODEL_PATH = MODEL_DIR / "launch_success_model.joblib"  # Full path to saved model file


def _load_data() -> pd.DataFrame:
    """Load and merge launches, rockets, and launchpads data for training."""
    launches = pd.read_csv(DATA_DIR / "launches.csv")
    rockets = pd.read_csv(DATA_DIR / "rockets.csv")
    pads = pd.read_csv(DATA_DIR / "launchpads.csv")
    # Join tables: launches + rocket names + launchpad info
    rockets_small = rockets[["id", "name"]].rename(columns={"id": "rocket", "name": "rocket_name"})
    pads_small = pads[["id", "name", "region"]].rename(columns={"id": "launchpad", "name": "pad_name", "region": "pad_region"})
    df = launches.merge(rockets_small, on="rocket", how="left").merge(pads_small, on="launchpad", how="left")
    return df


def _prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features (X) and target (y) for machine learning."""
    # Target: success (True/False) - what we want to predict
    df = df.copy()
    df = df[df["success"].isin([True, False])]
    y = df["success"].astype(int)  # Convert to 0/1 for ML


    df["details_len"] = df.get("details", pd.Series([""] * len(df))).fillna("").astype(str).str.len()
    df["reuse_count"] = df.get("core_0_flight", pd.Series([0] * len(df))).fillna(0)

    X = df[["reuse_count", "details_len", "rocket_name", "pad_region"]].fillna({
        "rocket_name": "Unknown",
        "pad_region": "Unknown",
    })
    return X, y


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """Create ML pipeline: preprocess data then train logistic regression."""
    # Preprocessing: handle different data types differently
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),  # Convert text to numbers
            ("num", StandardScaler(), numeric_features),  # Scale numbers to same range
        ]
    )
    clf = LogisticRegression(max_iter=200)  # Our ML model: predicts success probability
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])  # Chain: preprocess → train model
    return pipe


def train_and_save_model() -> str:
    """Train the model on historical data and save it to disk."""
    df = _load_data()
    X, y = _prepare(df)
    cat = ["rocket_name", "pad_region"]  # Text features that need one-hot encoding
    num = ["reuse_count", "details_len"]  # Number features that need scaling
    pipe = build_pipeline(cat, num)
    pipe.fit(X, y)  # Train the model on historical launches
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)  # Save trained model to file
    return str(MODEL_PATH)


def load_model() -> Pipeline:
    """Load the previously trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model was not found. Train it first.")
    return joblib.load(MODEL_PATH)


def predict_success_probability(reuse_count: int, details_len: int, rocket_name: str, pad_region: str) -> float:
    """Use trained model to predict launch success probability for given inputs."""
    model = load_model()
    # Format input data same way as training
    X = pd.DataFrame([{ 
        "reuse_count": reuse_count,
        "details_len": details_len,
        "rocket_name": rocket_name or "Unknown",
        "pad_region": pad_region or "Unknown",
    }])
    proba = model.predict_proba(X)[0, 1]  # Get probability of success (class 1)
    return float(np.clip(proba, 0.0, 1.0))  # Ensure result is between 0 and 1


