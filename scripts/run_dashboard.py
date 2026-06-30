from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("AQI_MODEL_DIR", "models/aqi_tree")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
