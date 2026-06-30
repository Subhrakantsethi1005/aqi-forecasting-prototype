# 🌍 AQI Forecasting Model: A Station-Aware ML Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ML-Powered](https://img.shields.io/badge/ML-GradientBoosting%20%26%20LSTM-orange.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

An end-to-end machine learning **prototype** for forecasting the **Air Quality Index (AQI)**. This project demonstrates a modular, station-aware forecasting pipeline, transitioning from an experimental Jupyter notebook to a packaged library with a live dashboard. It is a portfolio/learning prototype trained on synthetic-style data, not a validated production system (see [MODEL_CARD.md](MODEL_CARD.md)).

---

## 🚀 Key Features

*   **Station-Aware Architecture:** Tailors forecasts based on specific monitoring station dynamics, including geographical context and localized trends.
*   **Dual-Model Engine:** Supports both **LSTM (Long Short-Term Memory)** for time-series deep learning and **gradient-boosted trees** for robust tabular forecasting. The shipped/served artifact is a scikit-learn `GradientBoostingRegressor` (optional XGBoost/LightGBM backends are also wired in).
*   **Automated CPCB AQI Calculation:** Implements the official Central Pollution Control Board (CPCB) methodology for pollutant sub-indexing and AQI derivation.
*   **Interactive Dashboard:** A full-stack FastAPI application with an integrated HTML/JS dashboard for real-time station selection and visual forecasting.
*   **Data Engineering Pipeline:** Advanced feature engineering including lag features, rolling averages, and temporal encoding (hour/day/month).

---

## 🛠️ Project Structure

```text
├── app/                    # FastAPI application & interactive dashboard
├── src/aqi_forecasting/    # Core ML library (modular code)
│   ├── aqi.py              # Official CPCB AQI logic
│   ├── features.py         # Feature engineering (lags, rolling, time)
│   ├── tree_model.py       # GradientBoosting regressor + shared feature builder
│   ├── classify.py         # Binary "unhealthy air" classifier (AUC)
│   ├── walk_forward.py     # Expanding-window validation
│   ├── baselines.py        # Persistence & seasonal-naive baselines
│   ├── sources.py          # Dataset fetchers (Google Drive, OpenAQ v3)
│   ├── model.py            # LSTM architecture & artifact I/O
│   └── predict.py          # Prediction wrapper & serving logic
├── scripts/                # Thin CLI wrappers (train, predict, walk_forward, …)
├── notebooks/              # Original research notebook
├── models/                 # Serialized model artifacts
├── data/                   # Sample/generated datasets
└── tests/                  # Unit tests (28 passing)
```

---

## ⚡ Quick Start (Recruiter/Dev Demo)

### 1. Setup Environment
```bash
# Clone and enter repo
git clone https://github.com/Subhrakantsethi1005/aqi-forecasting-prototype.git
cd aqi-forecasting-prototype

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Demo Data
Don't have a dataset? Use the built-in generator to create a sample instantly:
```bash
python scripts/generate_sample_data.py
```

For **real** measurements (including genuine high-AQI/hazardous-air days, which the synthetic sample lacks), pull from the OpenAQ v3 API with a free API key:
```bash
export OPENAQ_API_KEY=your_key   # get one at https://docs.openaq.org/
python scripts/download_dataset.py --source openaq --iso IN \
  --date-from 2024-11-01 --date-to 2024-12-01 --output data/aqi_real.csv
```
The fetcher **auto-converts units** to what CPCB AQI expects (CO → mg/m³, gases → µg/m³, ppm/ppb → mass units) and prints a `[unit-check]` warning for any sensor whose unit it couldn't recognise. Still spot-check the output before training — see [MODEL_CARD.md](MODEL_CARD.md).

### 3. Launch the Dashboard
Experience the model in action through the web interface:
```bash
# Option A: FastAPI + HTML Dashboard (Recommended)
python scripts/run_dashboard.py

# Option B: Streamlit Dashboard
streamlit run dashboard/streamlit_app.py
```
Visit `http://localhost:8000` (for FastAPI) or the displayed URL (for Streamlit) in your browser.

---

## 🧠 Technical Deep Dive

### Data Processing
The pipeline handles raw pollutant readings (`PM2.5`, `PM10`, `NO2`, etc.) and weather variables. It automatically fills gaps and computes the 24-hour rolling averages required for official AQI standards.

### Feature Engineering
To capture temporal dynamics, the model uses:
*   **Lagged Features:** Past 1-24 hour readings to capture momentum.
*   **Rolling Stats:** Means and standard deviations to capture volatility.
*   **Cyclical Encoding:** Sine/Cosine transforms for time of day to ensure the model understands 23:00 is close to 00:00.

### Model Performance
The served model is a **GradientBoostingRegressor** forecasting AQI one hour ahead. On a chronological 80/20 test split it beats a persistence baseline, which is the honest test of forecasting skill:

| Model | MAE | R² |
|---|---|---|
| Persistence (AQI₍ₜ₊₁₎ = AQI₍ₜ₎) | 12.85 | 0.43 |
| **GradientBoosting (this model)** | **8.96** | **0.73** |

### Classification (AUC)
A companion **binary "unhealthy air" classifier** (`python scripts/train_classifier.py --data data/aqi_dataset.csv`) predicts whether next-hour AQI will exceed 100 (Moderate or worse). It uses **balanced class weighting** and a **decision threshold tuned on a validation tail** so it actually catches the rare dangerous hours. Effect on the holdout set (234 unhealthy hours):

| Setting | ROC-AUC | Recall | Precision | Unhealthy hours caught |
|---|---|---|---|---|
| Unweighted, threshold 0.5 | 0.92 | 0.11 | 0.43 | 26 / 234 |
| **Balanced + tuned threshold (0.32)** | 0.91 | **0.88** | 0.17 | **205 / 234** |

ROC-AUC barely moves (it's threshold-independent), but **recall jumps from 11% to 88%** — the model now flags ~9 in 10 dangerous hours. The cost is lower precision (more false alarms), the classic recall/precision trade-off; for a health alert, missing dangerous air is worse than a false alarm. This is also why a rare-event model must be judged on **recall/PR-AUC**, not ROC-AUC alone. Disable either fix with `--no-balance` / `--no-tune-threshold`.

### Regression
Under **walk-forward (expanding-window) validation** — which trains on the past and tests on the future across 5 folds instead of a single split — the picture is slightly more sober: mean R² ≈ **0.66 ± 0.07**, mean MAE ≈ **8.5**. Run it yourself with `python scripts/walk_forward.py --data data/aqi_dataset.csv`.

**Caveats (read these):**
*   The overall R² of 0.73 is inflated by *between-station* variance. Per-station R² is much lower — **ST1 ≈ 0.33, ST2 ≈ 0.28, ST3 ≈ 0.06** on the single split (≈ 0.22 / 0.18 / 0.10 averaged across walk-forward folds) — which better reflects true within-station forecasting skill.
*   The dataset is dominated by Good/Satisfactory air (max AQI ≈ 200). The model performs worst on the higher "Moderate" band (MAE ≈ 23) and has never seen Poor/Very Poor/Severe conditions, so it should **not** be trusted to forecast hazardous-air events.
*   The dataset is synthetic-style demo data, not validated real-world CPCB measurements.

---

## 🧪 Quality Assurance
We take code quality seriously. The project includes automated tests for the core AQI logic:
```bash
python -m pytest tests/
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created as a demonstration of end-to-end Machine Learning Engineering (research notebook → packaged library → API + dashboard).*
