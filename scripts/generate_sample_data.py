import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(output_path: str = "data/sample_aqi_dataset.csv", rows: int = 100):
    """Generates a small synthetic dataset for testing and demo purposes."""
    np.random.seed(42)
    
    start_date = pd.to_datetime("2024-01-01")
    dates = [start_date + pd.to_timedelta(i, unit='h') for i in range(rows)]
    
    data = {
        "Datetime": [d.strftime("%Y-%m-%d %H:%M") for d in dates],
        "station_id": ["ST_DEMO"] * rows,
        "PM2.5": np.random.uniform(10, 100, rows),
        "PM10": np.random.uniform(20, 150, rows),
        "NO2": np.random.uniform(5, 50, rows),
        "SO2": np.random.uniform(2, 20, rows),
        "CO": np.random.uniform(0.1, 2.0, rows),
        "O3": np.random.uniform(10, 60, rows),
        "Temperature": np.random.uniform(15, 35, rows),
        "Humidity": np.random.uniform(30, 80, rows),
        "WindSpeed": np.random.uniform(0, 15, rows),
        "lat": [28.6139] * rows,
        "lon": [77.2090] * rows
    }
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sample data generated at: {output_path}")

if __name__ == "__main__":
    generate_sample_data()
