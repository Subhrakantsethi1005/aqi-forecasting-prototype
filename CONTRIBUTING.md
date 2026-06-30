# Contributing to AQI Forecasting Prototype

Thanks for your interest in contributing! This project is an end-to-end machine
learning pipeline for forecasting the Air Quality Index (AQI) with
station-specific predictions, built with production-grade ML engineering
practices. Contributions that improve accuracy, robustness, or usability are
welcome.

## Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/Subhrakantsethi1005/aqi-forecasting-prototype.git
   cd aqi-forecasting-prototype
   ```
2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. (Optional) Run the stack with Docker:
   ```bash
   docker-compose up
   ```

## How to Contribute

- **Bugs:** open an issue with steps to reproduce and your environment details.
- **Models:** improvements to the LSTM or Random Forest models, or new
  forecasting approaches, are welcome — please open an issue to discuss first.
- **Feature engineering:** new lag/rolling/temporal features that improve
  validation R² or MAE.
- **Docs:** improvements to the README, `MODEL_CARD.md`, or `IMPROVEMENTS.md`.

## Development Guidelines

- Keep the core library code in `src/aqi_forecasting/` modular and well-documented.
- Follow the official CPCB AQI methodology for any sub-indexing changes.
- Add or update unit tests in `tests/` for any new behaviour.
- Run the test suite before submitting:
  ```bash
  pytest
  ```
- Ensure the FastAPI app in `app/` still runs after changes.

## Pull Requests

1. Create a descriptive branch (e.g. `feature/temporal-attention`).
2. Make focused commits with clear messages.
3. Open a pull request summarising the change and reporting any impact on
   validation metrics (R², MAE).

## Maintainers

- Subhrakant Sethi
- Ayush Singh
