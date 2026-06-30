from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from aqi_forecasting.predict import load_predictor


MODEL_DIR = os.getenv("AQI_MODEL_DIR", "models/aqi_tree")
app = FastAPI(title="AQI Forecasting API", version="0.1.0")
predictor = None
model_load_error: str | None = None


class PredictionRequest(BaseModel):
    rows: list[dict[str, object]] = Field(..., description="Recent feature rows. The API uses the latest rows.")


def _demo_dataset() -> pd.DataFrame:
    path = Path("data/aqi_dataset.csv")
    if not path.exists():
        raise FileNotFoundError("data/aqi_dataset.csv was not found.")
    frame = pd.read_csv(path)
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce", dayfirst=True)
    frame = frame.dropna(subset=["Datetime"]).sort_values(["station_id", "Datetime"])
    return frame


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AQI Forecasting Dashboard</title>
        <style>
          :root {
            --ink: #13212e;
            --muted: #607086;
            --line: #d9e2ec;
            --panel: #ffffff;
            --soft: #f5f8fb;
            --brand: #0f766e;
            --brand-dark: #115e59;
            --accent: #f59e0b;
            --danger: #b91c1c;
            --shadow: 0 18px 45px rgba(18, 33, 46, 0.12);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
            background:
              linear-gradient(180deg, rgba(15,118,110,0.10), rgba(245,248,251,0) 360px),
              var(--soft);
          }
          .shell { max-width: 1180px; margin: 0 auto; padding: 28px 20px 42px; }
          .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 28px;
          }
          .brand { display: flex; align-items: center; gap: 12px; font-weight: 750; }
          .mark {
            width: 38px; height: 38px; border-radius: 8px;
            display: grid; place-items: center;
            color: white; background: var(--brand);
            box-shadow: 0 10px 25px rgba(15,118,110,0.22);
          }
          .pill {
            border: 1px solid rgba(15,118,110,0.25);
            color: var(--brand-dark);
            background: rgba(15,118,110,0.08);
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 13px;
          }
          .hero {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
            gap: 22px;
            align-items: stretch;
          }
          .hero-copy {
            padding: 34px 0 18px;
          }
          h1 {
            font-size: clamp(34px, 5vw, 56px);
            line-height: 1.02;
            margin: 0 0 16px;
            letter-spacing: 0;
          }
          .lead {
            max-width: 700px;
            font-size: 18px;
            line-height: 1.62;
            color: var(--muted);
            margin: 0;
          }
          .hero-panel, .panel, .metric {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
          }
          .hero-panel { padding: 22px; }
          .model-line {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 14px;
            margin-bottom: 16px;
          }
          .label { font-size: 12px; text-transform: uppercase; color: var(--muted); font-weight: 700; }
          .value { font-size: 15px; font-weight: 700; margin-top: 4px; }
          .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
          }
          .metric { box-shadow: none; padding: 14px; }
          .metric strong { display: block; font-size: 22px; margin-top: 4px; }
          .workspace {
            display: grid;
            grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1.1fr);
            gap: 22px;
            margin-top: 22px;
          }
          .panel { padding: 22px; box-shadow: none; }
          h2 { font-size: 20px; margin: 0 0 14px; }
          .muted { color: var(--muted); }
          .upload-box {
            border: 1px dashed #9fb1c1;
            background: #f8fafc;
            border-radius: 8px;
            padding: 18px;
            margin: 16px 0;
          }
          input[type="file"] { width: 100%; }
          .actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
          button, .link-button {
            appearance: none;
            border: 0;
            background: var(--brand);
            color: white;
            padding: 11px 15px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            min-height: 42px;
          }
          button.secondary, .link-button.secondary {
            color: var(--brand-dark);
            background: rgba(15,118,110,0.10);
          }
          button:disabled { opacity: 0.55; cursor: not-allowed; }
          .result {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 14px;
          }
          .result-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            background: #fbfdff;
          }
          .aqi-value { font-size: 42px; line-height: 1; font-weight: 800; margin-top: 8px; }
          .category {
            display: inline-flex;
            align-items: center;
            min-height: 34px;
            padding: 8px 11px;
            border-radius: 999px;
            font-weight: 800;
            margin-top: 10px;
          }
          .Good { color: #166534; background: #dcfce7; }
          .Satisfactory { color: #3f6212; background: #ecfccb; }
          .Moderate { color: #854d0e; background: #fef3c7; }
          .Poor { color: #9a3412; background: #ffedd5; }
          .VeryPoor, .Severe { color: #991b1b; background: #fee2e2; }
          .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
          table { width: 100%; border-collapse: collapse; font-size: 13px; background: white; }
          th, td { border-bottom: 1px solid #edf2f7; padding: 9px 10px; text-align: left; white-space: nowrap; }
          th { background: #f8fafc; font-size: 12px; color: var(--muted); }
          pre { background: #0f172a; color: #dbeafe; padding: 14px; border-radius: 8px; overflow: auto; min-height: 86px; }
          .status { margin-top: 12px; font-size: 14px; color: var(--muted); }
          .error { color: var(--danger); font-weight: 700; }
          @media (max-width: 860px) {
            .hero, .workspace, .result { grid-template-columns: 1fr; }
            .hero-copy { padding-top: 12px; }
          }
        </style>
      </head>
      <body>
        <main class="shell">
          <nav class="topbar">
            <div class="brand"><div class="mark">AQ</div><span>AQI Forecasting Model</span></div>
            <span class="pill" id="health">Checking model...</span>
          </nav>

          <section class="hero">
            <div class="hero-copy">
              <h1>Station-aware AQI forecasting</h1>
              <p class="lead">Choose a monitoring station and timestamp. The model uses that station's recent pollutant, weather, location, lag, and rolling-trend history to forecast the next hour's AQI.</p>
            </div>
            <aside class="hero-panel">
              <div class="model-line">
                <div>
                  <div class="label">Active Model</div>
                  <div class="value">future AQI tree forecaster</div>
                </div>
                <div>
                  <div class="label">Default Path</div>
                  <div class="value">models/aqi_tree</div>
                </div>
              </div>
              <div class="metrics">
                <div class="metric"><span class="label">MAE</span><strong id="metricMae">--</strong></div>
                <div class="metric"><span class="label">RMSE</span><strong id="metricRmse">--</strong></div>
                <div class="metric"><span class="label">R2</span><strong id="metricR2">--</strong></div>
              </div>
            </aside>
          </section>

          <section class="workspace">
            <div class="panel">
              <h2>Forecast From Station History</h2>
              <p class="muted">Pick a station and historical timestamp. The app uses rows up to that time and forecasts the next hour.</p>
              <div class="upload-box">
                <label class="label" for="stationSelect">Station</label>
                <select id="stationSelect"></select>
                <br><br>
                <label class="label" for="timeSelect">Timestamp</label>
                <input id="timeSelect" type="datetime-local">
                <div class="actions">
                  <button id="stationPredictBtn" type="button">Forecast Next Hour</button>
                </div>
                <div class="status" id="stationStatus">Loading station data...</div>
              </div>
              <h2>Forecast Result</h2>
              <div class="result">
                <div class="result-card">
                  <div class="label">Forecast AQI</div>
                  <div class="aqi-value" id="aqiValue">--</div>
                </div>
                <div class="result-card">
                  <div class="label">AQI Category</div>
                  <div class="category" id="category">Waiting</div>
                </div>
              </div>
              <pre id="result">Choose station and timestamp, then run forecast.</pre>
            </div>

            <div class="panel">
              <h2>Station Context</h2>
              <p class="muted">This preview shows recent rows used to create lag and rolling features for the selected station.</p>
              <div class="table-wrap" id="preview">No station selected yet.</div>
            </div>
          </section>

          <section class="workspace">
            <div class="panel">
              <h2>Advanced CSV Upload</h2>
              <p class="muted">Use this only if you want to test another CSV with the same schema. The station selector above is the main demo flow.</p>
              <div class="upload-box">
                <form id="form">
                  <input id="file" type="file" accept=".csv" required>
                  <div class="actions">
                    <button id="predictBtn" type="submit">Predict AQI</button>
                    <a class="link-button secondary" href="/sample_csv">Download sample CSV</a>
                  </div>
                </form>
                <div class="status" id="fileStatus">No file selected.</div>
              </div>
              <h2>Prediction</h2>
              <pre id="uploadResult">Upload a CSV and run prediction.</pre>
            </div>

            <div class="panel">
              <h2>Uploaded CSV Preview</h2>
              <p class="muted">The preview shows the first rows from the uploaded file. The model predicts from the latest row after rebuilding lag and rolling features.</p>
              <div class="table-wrap" id="uploadPreview">No upload preview yet.</div>
            </div>
          </section>
        </main>

        <script>
          const fileInput = document.getElementById("file");
          const fileStatus = document.getElementById("fileStatus");
          const preview = document.getElementById("uploadPreview");
          const result = document.getElementById("result");
          const uploadResult = document.getElementById("uploadResult");
          const aqiValue = document.getElementById("aqiValue");
          const category = document.getElementById("category");
          const predictBtn = document.getElementById("predictBtn");
          const stationSelect = document.getElementById("stationSelect");
          const timeSelect = document.getElementById("timeSelect");
          const stationStatus = document.getElementById("stationStatus");
          const stationPredictBtn = document.getElementById("stationPredictBtn");

          function parseCsvPreview(text) {
            const rows = text.trim().split(/\\r?\\n/).slice(0, 7).map(line => line.split(","));
            if (!rows.length) return "No rows found.";
            const header = rows[0];
            const body = rows.slice(1);
            return `<table><thead><tr>${header.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
          }

          function setCategory(label) {
            category.className = "category " + String(label).replace(/\\s+/g, "");
            category.textContent = label || "Unknown";
          }

          fetch("/health")
            .then(r => r.json())
            .then(data => {
              document.getElementById("health").textContent = data.model_loaded ? "Model loaded" : "Model not loaded";
            })
            .catch(() => {
              document.getElementById("health").textContent = "Server online";
            });

          function toDatetimeLocal(value) {
            const d = new Date(value);
            const pad = n => String(n).padStart(2, "0");
            return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
          }

          function renderRows(rows, target) {
            if (!rows || !rows.length) {
              target.innerHTML = "No rows available.";
              return;
            }
            const header = Object.keys(rows[0]);
            target.innerHTML = `<table><thead><tr>${header.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${header.map(c => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
          }

          async function loadDemo() {
            const response = await fetch("/demo_info");
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Could not load demo info");
            document.getElementById("metricMae").textContent = data.metrics?.mae?.toFixed?.(2) ?? "--";
            document.getElementById("metricRmse").textContent = data.metrics?.rmse?.toFixed?.(2) ?? "--";
            document.getElementById("metricR2").textContent = data.metrics?.r2?.toFixed?.(3) ?? "--";
            stationSelect.innerHTML = data.stations.map(s => `<option value="${s}">${s}</option>`).join("");
            timeSelect.min = toDatetimeLocal(data.min_datetime);
            timeSelect.max = toDatetimeLocal(data.max_datetime);
            timeSelect.value = toDatetimeLocal(data.max_datetime);
            stationStatus.textContent = `${data.row_count} rows loaded across ${data.stations.length} stations.`;
            await refreshStationPreview();
          }

          async function refreshStationPreview() {
            const qs = new URLSearchParams({ station_id: stationSelect.value, datetime: timeSelect.value });
            const response = await fetch(`/station_recent?${qs.toString()}`);
            const data = await response.json();
            renderRows(data.rows, document.getElementById("preview"));
          }

          stationSelect.addEventListener("change", refreshStationPreview);
          timeSelect.addEventListener("change", refreshStationPreview);

          stationPredictBtn.addEventListener("click", async () => {
            stationPredictBtn.disabled = true;
            result.textContent = "Forecasting next hour...";
            aqiValue.textContent = "--";
            setCategory("Waiting");
            try {
              const qs = new URLSearchParams({ station_id: stationSelect.value, datetime: timeSelect.value });
              const response = await fetch(`/forecast_station?${qs.toString()}`, { method: "POST" });
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || "Forecast failed");
              aqiValue.textContent = Number(data.predicted_aqi).toFixed(2);
              setCategory(data.category);
              result.textContent = JSON.stringify(data, null, 2);
              await refreshStationPreview();
            } catch (error) {
              result.innerHTML = `<span class="error">${error.message}</span>`;
              setCategory("Error");
            } finally {
              stationPredictBtn.disabled = false;
            }
          });

          loadDemo().catch(error => {
            stationStatus.innerHTML = `<span class="error">${error.message}</span>`;
          });

          fileInput.addEventListener("change", async () => {
            const file = fileInput.files[0];
            if (!file) return;
            fileStatus.textContent = `${file.name} selected`;
            const text = await file.text();
            preview.innerHTML = parseCsvPreview(text);
          });

          document.getElementById("form").addEventListener("submit", async (event) => {
            event.preventDefault();
            const file = fileInput.files[0];
            const body = new FormData();
            body.append("file", file);
            predictBtn.disabled = true;
              uploadResult.textContent = "Predicting...";
              try {
                const response = await fetch("/predict_csv", { method: "POST", body });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Prediction failed");
                if (data.forecast_horizon_hours !== undefined) {
                  data.description = `Forecast for approximately ${data.forecast_horizon_hours} hour(s) after the latest uploaded reading.`;
                }
                uploadResult.textContent = JSON.stringify(data, null, 2);
              } catch (error) {
                uploadResult.innerHTML = `<span class="error">${error.message}</span>`;
              } finally {
                predictBtn.disabled = false;
              }
          });
        </script>
      </body>
    </html>
    """


@app.get("/sample_csv")
def sample_csv() -> Response:
    sample_path = Path("data/aqi_dataset.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample_aqi_dataset.csv")
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="No sample CSV found.")

    frame = pd.read_csv(sample_path).tail(72)
    return Response(
        content=frame.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_recent_readings.csv"},
    )


@app.get("/demo_info")
def demo_info() -> dict:
    try:
        frame = _demo_dataset()
        metrics_path = Path(MODEL_DIR) / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("overall", {})
        return {
            "stations": sorted(frame["station_id"].dropna().astype(str).unique().tolist()),
            "row_count": int(len(frame)),
            "min_datetime": frame["Datetime"].min().isoformat(),
            "max_datetime": frame["Datetime"].max().isoformat(),
            "metrics": metrics,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/station_recent")
def station_recent(station_id: str, datetime: str | None = None) -> dict:
    try:
        frame = _history_for_station(station_id, datetime)
        display = frame.tail(8).copy()
        display["Datetime"] = display["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        return {"rows": display.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast_station")
def forecast_station(station_id: str, datetime: str | None = None) -> dict:
    _ensure_predictor_loaded()
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded. Train a model or set AQI_MODEL_DIR. Error: {model_load_error}")
    try:
        history = _history_for_station(station_id, datetime)
        result = predictor.predict_with_category(history)
        latest_time = history["Datetime"].max()
        horizon = int(result.get("forecast_horizon_hours", 1))
        forecast_time = latest_time + pd.to_timedelta(horizon, unit="h")
        latest = history.sort_values("Datetime").iloc[-1]
        result.update(
            {
                "station_id": station_id,
                "latest_input_time": latest_time.isoformat(),
                "forecast_time": forecast_time.isoformat(),
                "lat": float(latest["lat"]) if "lat" in latest else None,
                "lon": float(latest["lon"]) if "lon" in latest else None,
                "description": f"Forecast for {station_id} at {forecast_time.strftime('%Y-%m-%d %H:%M')}",
            }
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _history_for_station(station_id: str, datetime: str | None = None) -> pd.DataFrame:
    frame = _demo_dataset()
    station_frame = frame[frame["station_id"].astype(str) == str(station_id)].copy()
    if station_frame.empty:
        raise ValueError(f"Station '{station_id}' was not found.")
    if datetime:
        selected_time = pd.to_datetime(datetime, errors="coerce")
        if pd.isna(selected_time):
            raise ValueError("Invalid datetime selected.")
        station_frame = station_frame[station_frame["Datetime"] <= selected_time]
    if len(station_frame) < 25:
        raise ValueError("Need at least 25 historical rows for this station to build lag features.")
    return station_frame.tail(96)


@app.on_event("startup")
def load_model() -> None:
    global predictor, model_load_error
    try:
        predictor = load_predictor(MODEL_DIR)
        model_load_error = None
    except Exception as exc:
        predictor = None
        model_load_error = str(exc)


def _ensure_predictor_loaded() -> None:
    if predictor is None:
        load_model()


@app.get("/health")
def health() -> dict:
    _ensure_predictor_loaded()
    return {"status": "ok", "model_loaded": predictor is not None, "model_dir": MODEL_DIR, "model_load_error": model_load_error}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    _ensure_predictor_loaded()
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded. Train a model or set AQI_MODEL_DIR. Error: {model_load_error}")
    try:
        result = predictor.predict_with_category(pd.DataFrame(request.rows))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> dict:
    _ensure_predictor_loaded()
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded. Train a model or set AQI_MODEL_DIR. Error: {model_load_error}")
    try:
        contents = await file.read()
        frame = pd.read_csv(BytesIO(contents))
        result = predictor.predict_with_category(frame)
        result["filename"] = file.filename
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
