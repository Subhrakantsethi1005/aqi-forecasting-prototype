# 🌍 AQI Forecasting Model: A Station-Aware ML Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ML-Powered](https://img.shields.io/badge/ML-LSTM%20%26%20RandomForest-orange.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

An end-to-end machine learning solution for forecasting the **Air Quality Index (AQI)**. This project demonstrates a production-ready pipeline, transitioning from an experimental Jupyter notebook to a modular, station-aware forecasting engine with a live dashboard.

---

## 🚀 Key Features

*   **Station-Aware Architecture:** Tailors forecasts based on specific monitoring station dynamics, including geographical context and localized trends.
*   **Dual-Model Engine:** Supports both **LSTM (Long Short-Term Memory)** for time-series deep learning and **Random Forest** (Tree-based) for robust tabular forecasting.
*   **Automated CPCB AQI Calculation:** Implements the official Central Pollution Control Board (CPCB) methodology for pollutant sub-indexing and AQI derivation.
*   **Interactive Dashboard:** A full-stack FastAPI application with an integrated HTML/JS dashboard for real-time station selection and visual forecasting.
*   **Data Engineering Pipeline:** Advanced feature engineering including lag features, rolling averages, and temporal encoding (hour/day/month).

---

## 🛠️ Project Structure

```text
├── app/                # FastAPI Application & Interactive Dashboard
├── src/                # Core ML Library (Modular Code)
│   └── aqi_forecasting/
│       ├── aqi.py      # Official CPCB AQI Logic
│       ├── model.py    # LSTM & Tree Model Architectures
│       └── predict.py  # Prediction Wrapper & Business Logic
├── notebooks/          # Research & Development (Original Experiments)
├── models/             # Serialized Model Artifacts & Scalers
├── scripts/            # Utility Scripts (Training, Data Gen, etc.)
└── tests/              # Unit Tests for Logic Validation
```

---

## ⚡ Quick Start (Recruiter/Dev Demo)

### 1. Setup Environment
```bash
# Clone and enter repo
git clone https://github.com/your-username/aqi-forecasting-model.git
cd aqi-forecasting-model

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Demo Data
Don't have a dataset? Use the built-in generator to create a sample instantly:
```bash
python scripts/generate_sample_data.py
```

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
*   **Random Forest:** Achieves an R² of ~0.73 on the validation set.
*   **MAE:** ~8.95 (Mean Absolute Error), providing high-confidence "Satisfactory" vs "Moderate" classification.

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
*Created as a demonstration of production-grade Machine Learning Engineering.*
