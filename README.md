# SpaceX ETL Dashboard with Machine Learning

A comprehensive dashboard for analyzing SpaceX launch data with machine learning capabilities for predicting launch success.

## Features

- **Offline Operation**: Runs entirely from local CSV files (`Backend/data/clean/`)
- **Multi-page Dashboard**: Overview, Rocket Families, Launchpads, Anomalies, Data Quality, Machine Learning
- **Machine Learning**: Logistic Regression + RandomForest with 80/20 split and 5-fold CV
- **Interactive Simulator**: What-if launch simulator with ML predictions
- **API Backend**: Optional FastAPI server for programmatic access
- **Mission Stories**: Playful narrative generation from launch data

## Quick Start (Windows)

### Option 1: GUI Launcher (I did this for me Mehran! ahah)
1. Double-click the desktop shortcut
2. Choose mode:
   - **Offline**: Streamlit dashboard only
   - **Online**: API + Streamlit dashboard  
   - **API only**: Backend server only

### Option 2: Command Line
```powershell
# Install dependencies
pip install -r requirements.txt

# Run with GUI launcher
python run.py

# Or run directly
python run.py offline    # Streamlit only
python run.py online     # API + Streamlit
python run.py api        # API only
```

### Option 3: Executable
1. Build: `pyinstaller --onefile --noconsole --icon app.ico run.py`
2. Run: `dist/run.exe`

## Setup Requirements

- **Python**: 3.11 or 3.12 (scikit-learn compatibility)
- **OS**: Windows 10/11
- **Dependencies**: See `requirements.txt`

### Virtual Environment (Recommended)
```powershell
# Create venv
py -3.12 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Requirements

Ensure these files exist in `Backend/data/clean/`:
- `launches.csv`
- `rockets.csv` 
- `launchpads.csv`
- `payloads.csv`
- `cores.csv`
- `launch_payload_bridge.csv`
- `launch_core_bridge.csv`

And in `Backend/data/out/`:
- `quality_report.json`

## Machine Learning

### Models
- **Logistic Regression**: Baseline linear classifier
- **RandomForest**: Tree-based ensemble with feature importance

### Evaluation
- **Split**: 80% train, 20% test
- **Cross-Validation**: 5-fold CV on training set (optional)
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
- **Features**: Reuse count, mission complexity, rocket type, launchpad region

### Training (Online Mode)
1. Start in "Online" mode
2. Train model: `Invoke-WebRequest -Uri http://127.0.0.1:8000/ml/train -Method POST`
3. Simulator will use ML predictions instead of heuristics

## API Endpoints (Online Mode)

- `GET /launches` - Launch data
- `GET /success-rates/monthly` - Monthly success rates
- `GET /rocket-families` - Success rates by rocket family
- `GET /launchpads` - Launchpad statistics with Wilson CI
- `GET /anomalies` - Anomaly detection results
- `GET /quality-report` - Data quality metrics
- `POST /ml/train` - Train ML model
- `GET /ml/predict` - Predict launch success
- `GET /download/clean/{name}` - Download clean data
- `GET /download/rejects/{name}` - Download rejected data

## Pages Overview

### Overview
- Launches per quarter (bar chart)
- Cumulative successes over time (line chart)
- Date range filtering

### Rocket Families  
- Success rates by family (regex mapping)
- Falcon 1, Falcon 9, Falcon Heavy, Starship classification

### Launchpads
- Success rates with Wilson 95% CI (manual implementation)
- Per-launchpad statistics

### Anomalies
- Monthly success rate >3σ detection (manual implementation)
- Statistical thresholds and outlier identification

### Data & Quality
- Quality report display
- Clean/reject data downloads
- Assignment signature

### Machine Learning
- Model performance metrics
- Confusion matrices
- ROC curves
- Feature importance (RandomForest)
- Coefficient analysis (Logistic Regression)
- Additional insights (reuse distribution, correlations)

### What-If Simulator
- Interactive launch simulation
- ML-powered success probability (Online mode)
- Animated rocket launch (Lottie)

### Mission Stories
- Playful narrative generation from launch data
- Random story selection

## Technical Details

### Architecture
- **Frontend**: Streamlit (offline-first)
- **Backend**: FastAPI (optional)
- **ML**: scikit-learn (Logistic Regression, RandomForest)
- **Data**: Pandas, local CSV files
- **Visualization**: Streamlit native charts, Lottie animations

### Key Implementations
- **Wilson CI**: Manual implementation (no stats libraries)
- **3σ Anomaly Detection**: Manual statistical calculation
- **Regex Family Mapping**: Custom rocket classification
- **Offline ML**: All training/prediction runs locally

## Troubleshooting

### Common Issues
1. **Port 8000 in use**: Stop existing API or restart
2. **Python 3.13**: Use Python 3.11/3.12 for scikit-learn compatibility
3. **Missing data**: Run backend ETL first to generate clean files
4. **Multiple browser tabs**: Use GUI launcher to prevent auto-opening

### Dependencies
If you encounter import errors:
```powershell
pip install --upgrade -r requirements.txt
```

## Documentation
- **README.md** - Setup and usage instructions
- **BIAS_ANALYSIS.md** - Comprehensive bias analysis and overfitting prevention
- **requirements.txt** - Python dependencies

## Assignment Signature
`SPX-ETL-2025-IdrA`

## File Structure
```
SpaceX_ETL/
├── Backend/
│   ├── data/clean/          # Clean CSV files
│   ├── data/out/            # Quality reports, models
│   ├── src/space_etl/       # ETL modules
│   ├── api.py               # FastAPI server
│   └── main.py              # ETL pipeline
├── Frontend/
│   └── app.py               # Streamlit dashboard
├── run.py                   # GUI launcher
├── requirements.txt         # Dependencies
└── README.md               # This file
```
