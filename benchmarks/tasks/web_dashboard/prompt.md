# Web Dashboard with Chart.js

Build a web dashboard page that displays metrics and charts. Backend API provides data, frontend renders charts.

## Backend (Python/FastAPI)

- `api.py`: FastAPI app serving:
  - `GET /api/metrics` — returns JSON: `{"cpu": [...], "memory": [...], "requests": [...], "labels": [...]}` where each list has 7 data points (simulate a week of data)
  - `GET /api/summary` — returns: `{"total_requests": 12345, "avg_cpu": 67.3, "avg_memory": 42.1, "uptime_hours": 720}`
  - CORS enabled
  - Serve static files from `./static`

- `data.py`: Data generation module:
  - `def generate_metrics() -> dict` — returns random-but-realistic metrics
  - `def compute_summary(metrics: dict) -> dict` — computes aggregate stats
  - Use `random` module with fixed seed for reproducibility: `random.seed(42)`

## Frontend

- `static/index.html`: Dashboard page with:
  - Header with title "System Dashboard"
  - Summary cards row: Total Requests, Avg CPU, Avg Memory, Uptime (4 cards)
  - Line chart area for CPU and Memory over time
  - Bar chart area for Requests
  - Clean modern design with CSS grid/flexbox

- `static/dashboard.js`: JavaScript that:
  - Fetches `/api/metrics` and `/api/summary` on page load
  - Populates summary cards with values
  - Creates Chart.js charts: line chart for CPU/Memory, bar chart for requests
  - Handle loading state and errors gracefully
  - Load Chart.js from CDN: `<script src="https://cdn.jsdelivr.net/npm/chart.js">`

- `static/styles.css`: Styling — dark theme preferred, responsive grid layout

## Output Files

- `api.py`
- `data.py`
- `static/index.html`
- `static/dashboard.js`
- `static/styles.css`
- `requirements.txt` (fastapi, uvicorn)
