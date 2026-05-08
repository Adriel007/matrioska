# CLI Data Pipeline

Create a multi-module Python CLI data pipeline that processes CSV files through stages: config → extract → transform → load.

## Architecture

The pipeline has 4 stages, each in its own file. They import from each other in a linear chain.

### Files

- `config.py`: Define a `Config` dataclass with fields:
  - `input_file: str` — path to input CSV
  - `output_file: str` — path to output CSV
  - `columns: list[str]` — columns to keep
  - `filters: dict[str, str]` — column→value filters (e.g. {"status": "active"})
  - `batch_size: int = 1000`

- `extract.py`: `def extract(config: Config) -> list[dict]` — reads CSV from `config.input_file`, returns list of dicts. Handle FileNotFoundError with clear error message. Use `csv.DictReader`.

- `transform.py`: `def transform(rows: list[dict], config: Config) -> list[dict]` — keeps only columns listed in `config.columns`, filters rows matching `config.filters` (exact string match). Returns filtered list.

- `load.py`: `def load(rows: list[dict], config: Config) -> int` — writes rows to CSV at `config.output_file` using `csv.DictWriter`. Returns number of rows written.

- `main.py` or `pipeline.py`: CLI entry point that:
  - Accepts `--config <file>` (JSON config file) or individual flags
  - Creates a `Config` from the JSON file
  - Runs: extract → transform → load
  - Prints: "Extracted N rows → Filtered to M rows → Wrote M rows to <output>"
  - Exits 0 on success, 1 on error

### Config JSON format
```json
{
  "input_file": "data.csv",
  "output_file": "output.csv",
  "columns": ["name", "email", "status"],
  "filters": {"status": "active"},
  "batch_size": 500
}
```

## Output Files

- `config.py`
- `extract.py`
- `transform.py`
- `load.py`
- `pipeline.py` (or `main.py`)
- `requirements.txt` (if any beyond stdlib)
