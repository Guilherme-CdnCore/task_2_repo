import streamlit as st
import pandas as pd
import time
import random
from pathlib import Path
import sys
import subprocess
import streamlit.components.v1 as components
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="SpaceX ETL Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 SpaceX Launches Dashboard")

ASSIGNMENT_SIGNATURE = "SPX-ETL-2025-IdrA"

@st.cache_data
def load_clean_launches() -> pd.DataFrame:
    base = Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "launches.csv"
    df = pd.read_csv(base)
    
    if "date_lisbon" in df.columns:
        # Parse to UTC then drop timezone 
        ser = pd.to_datetime(df["date_lisbon"], errors="coerce", utc=True)
        df["date_lisbon"] = ser.dt.tz_convert(None)
    if "success" in df.columns:
        
        df["success"] = df["success"].map({True: True, False: False, "True": True, "False": False}).astype("float").map({1.0: True, 0.0: False})
    return df

@st.cache_data
def load_quality_report() -> dict:
    path = Path(__file__).resolve().parents[1] / "Backend" / "data" / "out" / "quality_report.json"
    if path.exists():
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def run_backend_etl() -> str:
    backend_main = Path(__file__).resolve().parents[1] / "Backend" / "main.py"
    if not backend_main.exists():
        return "Backend main.py not found."
    try:
        
        result = subprocess.run([sys.executable, str(backend_main)], capture_output=True, text=True, cwd=str(backend_main.parent))
        if result.returncode == 0:
            # clear caches so new data is loaded
            load_clean_launches.clear()
            load_quality_report.clear()
            return "ETL completed successfully. Data refreshed."
        else:
            return f"ETL failed (code {result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error running ETL: {e}"

tabs = st.tabs(["Overview", "Rocket Families", "Launchpads (Wilson CI)", "Anomalies (>3σ)", "Data & Quality", "Machine Learning", "What‑If Launch Simulator", "Mission Stories"])

with tabs[0]:
    st.subheader("Overview (offline from clean data)")

    df = load_clean_launches()
    qreport = load_quality_report()

    # Filters - 11/9
    st.sidebar.header("Filters")
    if st.sidebar.button("Run backend ETL now"):
        with st.spinner("Running ETL... this may take a moment"):
            msg = run_backend_etl()
        st.sidebar.success(msg) if msg.startswith("ETL completed") else st.sidebar.error(msg)
        # Reload after ETL
        df = load_clean_launches()
        qreport = load_quality_report()
    # Drop rows without a valid date before computing ranges
    df_nonnull = df.dropna(subset=["date_lisbon"]) if not df.empty else df
    min_date = df_nonnull["date_lisbon"].min() if not df_nonnull.empty else None
    max_date = df_nonnull["date_lisbon"].max() if not df_nonnull.empty else None
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date.date() if min_date is not None else None, max_date.date() if max_date is not None else None)
        if min_date is not None and max_date is not None else (),
    )

    if df.empty:
        st.info("No launch data found. Ensure Backend/data/clean/launches.csv exists.")
    else:
        dff = df_nonnull.copy()
        dff["date_only"] = dff["date_lisbon"].dt.date
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
            start, end = date_range[0], date_range[1]
            dff = dff[(dff["date_only"] >= start) & (dff["date_only"] <= end)]

        # Launches per quarter
        dff["quarter"] = dff["date_lisbon"].dt.to_period("Q").astype(str)
        launches_per_quarter = dff.groupby("quarter").size().rename("launches").reset_index()

        # Cumulative successes by date
        dff_sorted = dff.sort_values("date_lisbon").copy()
        daily_success = (
            dff_sorted.assign(is_success=dff_sorted["success"].fillna(False).astype(int))
            .groupby("date_only")["is_success"].sum()
            .reset_index(name="daily_success")
        )
        daily_success["cumulative_success"] = daily_success["daily_success"].cumsum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Launches per quarter**")
            st.bar_chart(data=launches_per_quarter.set_index("quarter"))
            st.caption("Source: Backend/data/clean/launches.csv | Formula: count of launches grouped by quarter (date_lisbon → Q)")
        with col2:
            st.markdown("**Cumulative successes over time**")
            st.line_chart(data=daily_success.set_index("date_only")[["cumulative_success"]])
            st.caption("Source: Backend/data/clean/launches.csv | Formula: cumulative sum of success==True ordered by date")

        # Summary and signature
        total = int(len(dff))
        succ = int(dff["success"].fillna(False).sum())
        st.metric("Total launches (filtered)", total)
        st.metric("Total successes (filtered)", succ)
        st.caption(f"Assignment signature: {qreport.get('assignment_signature', ASSIGNMENT_SIGNATURE)}")

with tabs[1]:
    st.subheader("Rocket Families (regex mapping)")

    @st.cache_data
    def load_clean_rockets() -> pd.DataFrame:
        base = Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "rockets.csv"
        return pd.read_csv(base)

    launches = load_clean_launches()
    rockets = load_clean_rockets()

    if launches.empty or rockets.empty:
        st.info("Missing data. Ensure rockets.csv and launches.csv exist under Backend/data/clean/")
    else:
        # Prepare
        launches = launches.dropna(subset=["rocket"])  # rocket holds id
        rockets_small = rockets[["id", "name"]].rename(columns={"id": "rocket", "name": "rocket_name"})
        dfm = launches.merge(rockets_small, on="rocket", how="left")

        # Family mapping via regex on rocket_name
        import re
        def map_family(name: str) -> str:
            n = (name or "").lower()
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
        grp = dfm.groupby("family")
        summary = grp.agg(
            launches=("id", "count"),
            successes=("success", lambda s: int(pd.Series(s).fillna(False).sum()))
        ).reset_index()
        summary["success_rate"] = (summary["successes"] / summary["launches"]).fillna(0.0)

        st.dataframe(summary.sort_values("launches", ascending=False))
        st.bar_chart(summary.set_index("family")["success_rate"])
        st.caption("Sources: rockets.csv, launches.csv | Success rate = successes / launches | Family via regex on rocket_name")

with tabs[2]:
    st.subheader("Launchpads: Success rates with Wilson 95% CI (manual)")

    @st.cache_data
    def load_clean_launchpads() -> pd.DataFrame:
        base = Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "launchpads.csv"
        return pd.read_csv(base)

    launches = load_clean_launches()
    pads = load_clean_launchpads()

    if launches.empty or pads.empty:
        st.info("Missing data. Ensure launchpads.csv and launches.csv exist under Backend/data/clean/")
    else:
        # Join launches to launchpad names
        launches = launches.dropna(subset=["launchpad"])  # id on launches
        pads_small = pads[["id", "name", "full_name", "region"]].rename(columns={"id": "launchpad", "name": "pad_name", "region": "pad_region"})
        dfm = launches.merge(pads_small, on="launchpad", how="left")

        # Aggregate successes/attempts per pad
        agg = dfm.groupby(["launchpad", "pad_name", "pad_region"]).agg(
            attempts=("id", "count"),
            successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
        ).reset_index()

        # Manual Wilson 95% CI
        # p̂ = x/n; z=1.96; lower, upper per Wilson formula
        import math
        z = 1.96
        def wilson_interval(x: int, n: int, z: float = 1.96):
            if n == 0:
                return (0.0, 0.0, 0.0)
            phat = x / n
            denom = 1 + (z**2)/n
            center = (phat + (z**2)/(2*n)) / denom
            margin = (z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
            return (phat, max(0.0, center - margin), min(1.0, center + margin))

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
        res = pd.DataFrame(rows)

        st.dataframe(res.sort_values(["success_rate", "attempts"], ascending=[False, False]))
        st.caption("Source: launches.csv + launchpads.csv | Success rate = successes/attempts | Wilson 95% CI computed manually with z=1.96")

        # Simple visualization: error bars via text (Streamlit basic). 
        st.bar_chart(res.set_index("name")["success_rate"])

        # Explain formula
        with st.expander("What is Wilson 95% CI?"):
            st.write("The Wilson interval adjusts for small samples, computed as center = (p̂ + z²/(2n)) / (1 + z²/n) and margin = z * √((p̂(1−p̂) + z²/(4n))/n) / (1 + z²/n). We display [center−margin, center+margin].")

with tabs[3]:
    st.subheader("Anomalies: Monthly success rate >3σ from mean (manual)")

    launches = load_clean_launches()
    if launches.empty:
        st.info("Missing launches.csv. Ensure Backend/data/clean/ exists.")
    else:
        df = launches.dropna(subset=["date_lisbon"]).copy()
        df["month"] = df["date_lisbon"].dt.to_period("M").astype(str)
        monthly = df.groupby("month").agg(
            attempts=("id", "count"),
            successes=("success", lambda s: int(pd.Series(s).fillna(False).sum())),
        ).reset_index()
        monthly["rate"] = monthly["successes"] / monthly["attempts"].replace(0, pd.NA)
        monthly = monthly.dropna(subset=["rate"])  # drop months with 0 attempts

        mu = float(monthly["rate"].mean()) if not monthly.empty else 0.0
        sigma = float(monthly["rate"].std(ddof=0)) if len(monthly) > 1 else 0.0
        thr_high = mu + 3*sigma
        thr_low = mu - 3*sigma
        monthly["z"] = (monthly["rate"] - mu) / (sigma if sigma else 1.0)
        anomalies_df = monthly[(monthly["rate"] > thr_high) | (monthly["rate"] < thr_low)]

        st.line_chart(monthly.set_index("month")["rate"])
        st.caption("Source: launches.csv | Monthly success rate = successes/attempts | Anomaly if outside mean ± 3σ (computed manually)")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Summary stats**")
            st.metric("Mean rate", f"{mu:.2%}")
            st.metric("Std dev (σ)", f"{sigma:.2%}")
            st.metric(">3σ thresholds", f"[{max(0.0, thr_low):.2%}, {min(1.0, thr_high):.2%}]")
        with colB:
            st.markdown("**Detected anomalies**")
            st.dataframe(anomalies_df[["month", "attempts", "successes", "rate", "z"]])

        with st.expander("Why 3σ?"):
            st.write("Assuming rates fluctuate around a mean, points beyond ±3 standard deviations are rare under normal variation, so we flag them as anomalies. This is a simple heuristic, not a proof of root cause.")

with tabs[4]:
    st.subheader("Data & Quality")
    qreport = load_quality_report()
    if not qreport:
        st.info("quality_report.json not found. Please run backend ETL to generate it.")
    else:
        st.json(qreport)
        st.caption("Source: Backend/data/out/quality_report.json")

    # Show rejects if available and allow downloads for clean/rejects
    data_root = Path(__file__).resolve().parents[1] / "Backend" / "data"
    clean_dir = data_root / "clean"
    rejects_dir = data_root / "out" / "rejects"

    st.markdown("**Clean data files**")
    if clean_dir.exists():
        for file in sorted(clean_dir.glob("*.csv")):
            with open(file, "rb") as f:
                st.download_button(label=f"Download {file.name}", data=f, file_name=file.name, mime="text/csv")
    else:
        st.caption("No clean files found.")

    st.markdown("**Rejects**")
    if rejects_dir.exists():
        for file in sorted(rejects_dir.glob("*.csv")):
            st.markdown(f"- {file.name}")
            try:
                df_prev = pd.read_csv(file).head(20)
                st.dataframe(df_prev)
                with open(file, "rb") as f:
                    st.download_button(label=f"Download {file.name}", data=f, file_name=file.name, mime="text/csv")
            except Exception as e:
                st.caption(f"Could not preview {file.name}: {e}")
    else:
        st.caption("No rejects directory found.")

with tabs[5]:
    st.subheader("Machine Learning: Predicting Launch Success (Offline)")

    # Load data
    launches = load_clean_launches()
    rockets = pd.read_csv(Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "rockets.csv") if (Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "rockets.csv").exists() else pd.DataFrame()
    pads = pd.read_csv(Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "launchpads.csv") if (Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "launchpads.csv").exists() else pd.DataFrame()

    if launches.empty or rockets.empty or pads.empty:
        st.info("Missing clean data. Ensure launches.csv, rockets.csv, and launchpads.csv exist under Backend/data/clean/.")
    else:
        # Prepare dataset (simple features; offline)
        rockets_small = rockets[["id", "name"]].rename(columns={"id": "rocket", "name": "rocket_name"})
        pads_small = pads[["id", "name", "region"]].rename(columns={"id": "launchpad", "name": "pad_name", "region": "pad_region"})
        df = launches.merge(rockets_small, on="rocket", how="left").merge(pads_small, on="launchpad", how="left")
        df = df[df["success"].isin([True, False])].copy()
        df["reuse_count"] = df.get("core_0_flight", 0).fillna(0)
        df["details_len"] = df.get("details", "").fillna("").astype(str).str.len()
        feat_cols = ["reuse_count", "details_len", "rocket_name", "pad_region"]
        X = pd.get_dummies(df[feat_cols].fillna({"rocket_name": "Unknown", "pad_region": "Unknown"}), drop_first=True)
        y = df["success"].astype(int)

        # Controls
        st.sidebar.subheader("ML Settings")
        do_cv = st.sidebar.checkbox("Use 5-fold CV for RandomForest tuning", value=True)
        rf_estimators = st.sidebar.slider("RF n_estimators", 50, 400, 150, step=50)
        test_size = 0.2

        @st.cache_data
        def train_models_cached(X_df: pd.DataFrame, y_series: pd.Series, n_estimators: int, use_cv: bool):
            from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

            X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=test_size, random_state=42, stratify=y_series)

            # Logistic Regression baseline
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            y_proba_lr = getattr(lr, "predict_proba", None)
            proba_lr = y_proba_lr(X_test)[:, 1] if y_proba_lr else None

            # RandomForest
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            if use_cv:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
                cv_info = {"cv_f1_mean": float(cv_scores.mean()), "cv_f1_std": float(cv_scores.std())}
            else:
                cv_info = {"cv_f1_mean": None, "cv_f1_std": None}
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            proba_rf = rf.predict_proba(X_test)[:, 1]

            def metrics(y_true, y_pred):
                return {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "cm": confusion_matrix(y_true, y_pred).tolist(),
                }

            # ROC/AUC
            fpr_lr, tpr_lr, auc_lr = None, None, None
            if proba_lr is not None:
                fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
                auc_lr = float(auc(fpr_lr, tpr_lr))
            fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
            auc_rf = float(auc(fpr_rf, tpr_rf))

            return {
                "lr": {"model": lr, "metrics": metrics(y_test, y_pred_lr), "roc": {"fpr": fpr_lr, "tpr": tpr_lr, "auc": auc_lr}},
                "rf": {"model": rf, "metrics": metrics(y_test, y_pred_rf), "roc": {"fpr": fpr_rf, "tpr": tpr_rf, "auc": auc_rf}, "cv": cv_info},
                "feature_names": list(X_df.columns),
            }

        results = train_models_cached(X, y, rf_estimators, do_cv)

        # Display metrics
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Logistic Regression (baseline)**")
            st.write({k: v for k, v in results["lr"]["metrics"].items() if k != "cm"})
        with colB:
            st.markdown("**RandomForest**")
            rf_info = results["rf"]["metrics"].copy()
            cv_info = results["rf"]["cv"]
            if cv_info["cv_f1_mean"] is not None:
                rf_info.update({"cv_f1_mean": cv_info["cv_f1_mean"], "cv_f1_std": cv_info["cv_f1_std"]})
            st.write({k: v for k, v in rf_info.items() if k != "cm"})

        # Confusion matrices
        st.markdown("**Confusion Matrices** (rows=true, cols=pred)")
        cm_cols = ["Pred 0", "Pred 1"]
        cm_index = ["True 0", "True 1"]
        cm_lr = pd.DataFrame(results["lr"]["metrics"]["cm"], index=cm_index, columns=cm_cols)
        cm_rf = pd.DataFrame(results["rf"]["metrics"]["cm"], index=cm_index, columns=cm_cols)
        cm1, cm2 = st.columns(2)
        with cm1:
            st.caption("Logistic Regression")
            st.dataframe(cm_lr)
        with cm2:
            st.caption("RandomForest")
            st.dataframe(cm_rf)

        # ROC curves
        st.markdown("**ROC Curves** (optional)")
        roc_df = pd.DataFrame()
        if results["lr"]["roc"]["fpr"] is not None:
            roc_df["LR_FPR"] = results["lr"]["roc"]["fpr"]
            roc_df["LR_TPR"] = results["lr"]["roc"]["tpr"]
            st.caption(f"LR AUC = {results['lr']['roc']['auc']:.3f}")
        roc_df["RF_FPR"] = results["rf"]["roc"]["fpr"]
        roc_df["RF_TPR"] = results["rf"]["roc"]["tpr"]
        st.caption(f"RF AUC = {results['rf']['roc']['auc']:.3f}")
        # Show RF ROC as line chart (FPR vs TPR)
        if not roc_df.empty:
            st.line_chart(roc_df[[c for c in roc_df.columns if "TPR" in c]])

        # Feature importances (RF)
        st.markdown("**RandomForest Feature Importance**")
        rf_model = results["rf"]["model"]
        importances = pd.Series(rf_model.feature_importances_, index=results["feature_names"]).sort_values(ascending=False).head(15)
        st.bar_chart(importances)

        # Logistic Regression coefficients
        st.markdown("**Logistic Regression Coefficients**")
        lr_model = results["lr"]["model"]
        try:
            coefs = pd.Series(lr_model.coef_[0], index=results["feature_names"]).sort_values(key=abs, ascending=False).head(15)
            st.bar_chart(coefs)
        except Exception:
            st.caption("Coefficients unavailable for this solver.")

        with st.expander("How to read these metrics"):
            st.markdown("- Accuracy: overall correct predictions / all predictions")
            st.markdown("- Precision: among predicted successes, how many were actually successes")
            st.markdown("- Recall: among actual successes, how many we correctly found")
            st.markdown("- F1: harmonic mean of Precision and Recall (good when classes are imbalanced)")

        st.caption("Sources: Backend/data/clean/*.csv | Split: 80/20 | Optional RF 5-fold CV | Metrics computed with scikit-learn")

        st.markdown("---")
        st.subheader("Additional insights")
        # Reuse histogram
        reuse_hist = df["reuse_count"].astype(int).value_counts().sort_index()
        st.markdown("**Booster reuse count distribution**")
        st.bar_chart(reuse_hist)

        # Payload mass distribution (if available via payloads + bridge)
        payloads_path = Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "payloads.csv"
        bridge_path = Path(__file__).resolve().parents[1] / "Backend" / "data" / "clean" / "launch_payload_bridge.csv"
        if payloads_path.exists() and bridge_path.exists():
            try:
                payloads = pd.read_csv(payloads_path)
                bridge = pd.read_csv(bridge_path)
                # Expect columns: payloads.id, payloads.mass_kg (if present), bridge.launch_id, bridge.payload_id
                if "mass_kg" in payloads.columns:
                    b = bridge.rename(columns={"launch_id": "id", "payload_id": "payload_id"})
                    merged = b.merge(payloads[["id", "mass_kg"]].rename(columns={"id": "payload_id"}), on="payload_id", how="left")
                    mass_per_launch = merged.groupby("id")["mass_kg"].sum(min_count=1).dropna()
                    st.markdown("**Total payload mass per launch (kg)**")
                    st.line_chart(mass_per_launch.sort_index())
                else:
                    st.caption("Payload mass_kg not available in payloads.csv; skipping mass chart.")
            except Exception as e:
                st.caption(f"Could not compute payload mass distribution: {e}")
        else:
            st.caption("payloads.csv or launch_payload_bridge.csv not found; skipping payload mass chart.")

        # Simple correlations among numeric features
        st.markdown("**Correlation matrix (numeric features)**")
        corr_df = pd.DataFrame({
            "reuse_count": df["reuse_count"].astype(float),
            "details_len": df["details_len"].astype(float),
            "success": df["success"].astype(int),
        })
        st.dataframe(corr_df.corr(numeric_only=True).round(3))
    st.subheader("Lets play with launch conditions and see the odds!")

    col1, col2, col3 = st.columns(3)

    with col1:
        weather = st.selectbox(
            "Weather",
            ["Clear", "Partly Cloudy", "Windy", "Rain", "Storm"],
            index=0,
            help="Simplified launch day conditions"
        )
    with col2:
        reuse_count = st.slider(
            "Core reuse count",
            min_value=0,
            max_value=15,
            value=1,
            help="How many times this booster has flown"
        )
    with col3:
        payload_mass = st.slider(
            "Payload mass (kg)",
            min_value=0,
            max_value=23_000,
            value=8_000,
            step=500,
            help="Approximate mass to orbit"
        )

    landing_option = st.select_slider(
        "Landing plan",
        options=["ASDS Drone Ship", "RTLS (Return To Launch Site)", "Expendable"],
        value="ASDS Drone Ship",
        help="Intended landing method"
    )

    
    base_prob = 0.78
    weather_adjustments = {
        "Clear": 0.07,
        "Partly Cloudy": 0.03,
        "Windy": -0.06,
        "Rain": -0.10,
        "Storm": -0.18,
    }
    landing_adjustments = {
        "ASDS Drone Ship": -0.02,
        "RTLS (Return To Launch Site)": 0.02,
        "Expendable": 0.04,
    }

    payload_penalty = max(0, (payload_mass - 15_000) / 20_000) * 0.15
    reuse_penalty = min(reuse_count * 0.01, 0.10)

    # Try ML API first (if online), else fallback heuristic
    probability = None
    try:
        resp = requests.get(
            "http://127.0.0.1:8000/ml/predict",
            params={
                "reuse_count": reuse_count,
                "details_len": 120 if weather == "Clear" else 60,
                "rocket_name": "Falcon 9",
                "pad_region": "Florida" if landing_option != "Expendable" else "California",
            },
            timeout=0.5,
        )
        if resp.ok:
            probability = float(resp.json().get("success_probability", 0.0))
    except Exception:
        probability = None

    if probability is None or probability == 0.0:
        probability = base_prob
        probability += weather_adjustments.get(weather, 0)
        probability += landing_adjustments.get(landing_option, 0)
        probability -= payload_penalty
        probability -= reuse_penalty
        probability = max(0.02, min(0.98, probability))

    left, right = st.columns([1, 1])
    with left:
        st.metric("Estimated success probability", f"{probability*100:.1f}%")
    with right:
        st.caption("Beware fellow spaceX enthusiast, this is just  a simplified, playful simulator—not real launch advice my friend!")

    launch = st.button("Ignite engines and launch! 🚀")

    if launch:
        progress = st.progress(0, text="T‑10s: Engine chill...")
        status_text = st.empty()
        phases = [
            (15, "T‑6s: Engine ignition sequence start"),
            (30, "T‑3s: Throttle up"),
            (60, "Liftoff! Clearing the tower"),
            (80, "Max‑Q"),
            (100, "MECO and stage separation"),
        ]
        for p, msg in phases:
            progress.progress(p, text=msg)
            status_text.write(msg)
            time.sleep(0.6)

        result_roll = random.random()
        success = result_roll < probability
        if success:
            st.success("Nominal ascent! Payload on target trajectory.")
            def load_lottieurl(url: str):
                try:
                    r = requests.get(url, timeout=2)
                    if r.status_code != 200:
                        return None
                    return r.json()
                except Exception:
                    return None
            lottie_rocket = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
            if lottie_rocket:
                st_lottie(lottie_rocket, height=300, key="rocket")
            else:
                components.html(
                    """
                    <div style='position:relative;height:220px;background:linear-gradient(#001022,#0a2342);overflow:hidden;border-radius:8px;'>
                      <div style='position:absolute;left:50%;bottom:0;transform:translateX(-50%);font-size:48px;animation:lift 2.5s ease-in forwards'>🚀</div>
                      <div style='position:absolute;left:50%;bottom:0;transform:translateX(-50%);width:6px;height:120px;background:linear-gradient(transparent,rgba(255,200,120,0.8));filter:blur(2px);animation:flame 0.5s infinite alternate;'></div>
                      <style>
                        @keyframes lift { from { bottom:0; } to { bottom:180px; } }
                        @keyframes flame { from { opacity:0.6; } to { opacity:1; } }
                      </style>
                    </div>
                    """,
                    height=230,
                )
        else:
            st.error("Anomaly detected. Flight terminated safely.")
            st.snow()

with tabs[6]:
    st.subheader("Mission Stories (from backend ETL)")

    def _stories_path() -> Path:
        this_file = Path(__file__).resolve()
        backend_dir = this_file.parents[1] / "Backend"
        return backend_dir / "data" / "out" / "mission_stories.txt"

    path = _stories_path()
    if path.exists():
        text = path.read_text(encoding="utf-8")
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]

        col_a, col_b = st.columns([1, 1])
        with col_a:
            show_random = st.button("Show a random story ✨")
        with col_b:
            st.caption(f"Loaded from {path.as_posix()}")

        if show_random and stories:
            st.markdown(stories[random.randrange(len(stories))])
        else:
            st.text_area("Stories", value=text, height=300)
            st.caption("Tip: Run the backend ETL to refresh stories.")
    else:
        st.info("No stories file found yet. Run the backend ETL to generate mission stories.")




