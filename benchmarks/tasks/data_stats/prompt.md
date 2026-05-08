# CSV Statistics Calculator

Create a Python script that reads a CSV file and computes descriptive statistics.

## Requirements

- Read a CSV file provided as command-line argument: `python stats.py data.csv`
- Print the following for each numeric column: count, mean, median, std, min, max, 25th percentile, 75th percentile
- Print value counts for each non-numeric column (top 10 values)
- Handle empty files gracefully (print "Empty file" and exit 0)
- Handle missing file gracefully (print "File not found: <path>" and exit 1)
- Use only stdlib (csv module)

## Output Format

```
Column: <name> (numeric)
  count: <n>
  mean: <x>
  median: <x>
  std: <x>
  min: <x>
  max: <x>
  p25: <x>
  p75: <x>

Column: <name> (categorical)
  <value>: <count>
  <value>: <count>
  ...
```

## Output Files

- `stats.py`: Main script
- `requirements.txt`: Dependencies (if any)
