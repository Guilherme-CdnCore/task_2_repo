from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import math
import re
from src.space_etl.ml_model import train_and_save_model, predict_success_probability


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUT_DIR = DATA_DIR / "out"
REJECTS_DIR = OUT_DIR / "rejects"


app = FastAPI(title="SpaceX ETL API", version="1.0")


def _load_csv(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name}.csv not found")
    return pd.read_csv(path)


@app.get("/launches")
def launches(limit: int | None = None) -> List[Dict[str, Any]]:
    df = _load_csv("launches")
    if "date_lisbon" in df.columns:
        df["date_lisbon"] = pd.to_datetime(df["date_lisbon"], errors="coerce")
    if limit is not None:
        df = df.sort_values("date_lisbon", ascending=False).head(limit)
    return df.fillna("").to_dict(orient="records")


@app.get("/success-rates/monthly")
def success_rates_monthly():
    df = _load_csv("launches")
    if "date_lisbon" not in df.columns:
        raise HTTPException(status_code=500, detail="date_lisbon column missing")
    df["date_lisbon"] = pd.to_datetime(df["date_lisbon"], errors="coerce")
    df = df.dropna(subset=["date_lisbon"])  # ensure datetimelike
    df["month"] = df["date_lisbon"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        attempts=("id", "count"),
        successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
    ).reset_index()
    monthly["rate"] = monthly["successes"] / monthly["attempts"].replace(0, pd.NA)
    return monthly.fillna(0).to_dict(orient="records")


@app.get("/rocket-families")
def rocket_families():
    launches = _load_csv("launches")
    rockets = _load_csv("rockets")
    rockets_small = rockets[["id", "name"]].rename(columns={"id": "rocket", "name": "rocket_name"})
    dfm = launches.merge(rockets_small, on="rocket", how="left")
    

    def map_family(name: str) -> str:
        n = (str(name) if pd.notna(name) else "").lower()
        if re.search(r"falcon\s*1", n):
            return "Falcon 1"
        if re.search(r"falcon\s*9", n):
            return "Falcon 9"
        if re.search(r"falcon\s*heavy", n):
            return "Falcon Heavy"
        if re.search(r"starship|super\s*heavy|bfr", n):
            return "Starship"
        return "Other"

    dfm["family"] = dfm["rocket_name"].apply(map_family)
    grp = dfm.groupby("family").agg(
        launches=("id", "count"),
        successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
    ).reset_index()
    grp["success_rate"] = (grp["successes"] / grp["launches"]).fillna(0.0)
    return grp.to_dict(orient="records")


@app.get("/launchpads")
def launchpads():
    launches = _load_csv("launches")
    pads = _load_csv("launchpads")
    launches = launches.dropna(subset=["launchpad"])  # ensure key
    pads_small = pads[["id", "name", "region"]].rename(columns={"id": "launchpad", "name": "pad_name", "region": "pad_region"})
    dfm = launches.merge(pads_small, on="launchpad", how="left")
    agg = dfm.groupby(["launchpad", "pad_name", "pad_region"]).agg(
        attempts=("id", "count"),
        successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
    ).reset_index()

   

    def wilson_interval(x: int, n: int, z: float = 1.96):
        if n == 0:
            return 0.0, 0.0, 0.0
        phat = x / n
        denom = 1 + (z ** 2) / n
        center = (phat + (z ** 2) / (2 * n)) / denom
        margin = (z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n)) / denom
        return phat, max(0.0, center - margin), min(1.0, center + margin)

    rows = []
    for _, r in agg.iterrows():
        p, lo, hi = wilson_interval(int(r["successes"]), int(r["attempts"]))
        rows.append({
            "launchpad": r["launchpad"],
            "name": r["pad_name"],
            "region": r["pad_region"],
            "attempts": int(r["attempts"]),
            "successes": int(r["successes"]),
            "success_rate": p,
            "wilson_low": lo,
            "wilson_high": hi,
        })
    return rows


@app.get("/anomalies")
def anomalies():
    df = _load_csv("launches")
    df["date_lisbon"] = pd.to_datetime(df["date_lisbon"], errors="coerce")
    df = df.dropna(subset=["date_lisbon"]).copy()
    df["month"] = df["date_lisbon"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        attempts=("id", "count"),
        successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
    ).reset_index()
    monthly["rate"] = monthly["successes"] / monthly["attempts"].replace(0, pd.NA)
    monthly = monthly.dropna(subset=["rate"])  # require attempts>0
    mu = float(monthly["rate"].mean()) if not monthly.empty else 0.0
    sigma = float(monthly["rate"].std(ddof=0)) if len(monthly) > 1 else 0.0
    thr_high = mu + 3 * sigma
    thr_low = mu - 3 * sigma
    out = monthly[(monthly["rate"] > thr_high) | (monthly["rate"] < thr_low)]
    return {
        "mean": mu,
        "std": sigma,
        "threshold_low": thr_low,
        "threshold_high": thr_high,
        "anomalies": out.to_dict(orient="records"),
    }


@app.post("/ml/train")
def ml_train():
    path = train_and_save_model()
    return {"model_path": path}


@app.get("/ml/predict")
def ml_predict(reuse_count: int = 0, details_len: int = 0, rocket_name: str = "Falcon 9", pad_region: str = "Florida"):
    proba = predict_success_probability(reuse_count, details_len, rocket_name, pad_region)
    return {"success_probability": proba}


@app.get("/quality-report")
def quality_report():
    path = OUT_DIR / "quality_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="quality_report.json not found")
    return JSONResponse(content=pd.read_json(path).to_dict())


@app.get("/download/clean/{name}")
def download_clean(name: str):
    path = CLEAN_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name}.csv not found")
    return FileResponse(path)


@app.get("/download/rejects/{name}")
def download_rejects(name: str):
    path = REJECTS_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"rejects {name}.csv not found")
    return FileResponse(path)


if __name__ == "__main__":
    # Run: python api.py
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


