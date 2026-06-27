# data/snapshots/

Committed, frozen copies of the live data sources used by the project.

These files make the repository a self-contained replication package: with the
snapshots present, every script and notebook runs in the default `snapshot`
data mode (see `Library/DataAccess.py`) with no network access required.

## How to generate

Run once, in an environment with network access:

```
python Scripts/export_snapshots.py
```

## Expected contents

| File | Source | Required for |
| --- | --- | --- |
| `prices_SPX.csv` | FinanceDataReader `^SPX` adjusted close | P-MLE, VaR, skew calibration |
| `prices_COIN.csv` | FinanceDataReader `COIN` adjusted close | P-MLE, VaR, collar study |
| `portfolio.csv` *(optional)* | `Scripts/load_portfolio.py` (run as `__main__`) | Notebooks can read this directly or call `build_portfolio()`. |

The P-MLE parameter estimates are not produced here — they ship as per-date
CSVs under `Study/Estimated Parameters PMLE/` and are produced directly by
`Scripts/run_pmle_kimyi2025.py`.

Once generated, commit the CSV files so collaborators and referees can
reproduce the results without re-pulling vendor data (which is revised over
time).
