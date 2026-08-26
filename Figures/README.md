# Figures/

Central output directory for all PDF plots produced by the replication
pipeline. `Scripts/reproduce_paper.py` writes every figure here, so the
repository root and code folders stay clean.

## Paper figures

| File | Paper reference | Produced by |
| --- | --- | --- |
| `Figure_1_density_simulated_returns.pdf` | Figure 1 — Density of 1-day simulated returns (LA vs BS) | `report_collar_asian.py` |
| `Figure_2_simulated_pl.pdf` | Figure 2 — Density of 1-day simulated P&L (LA vs BS) | `report_collar_asian.py` |
| `Figure_4_SPX_VOL_SKEW_{YYYYMMDD}.pdf` | Figure 4 — SPX volatility skew, market vs calibrated | `skew_calibration_main.py` |
| `Figure_5_COIN_VOL_SKEW_{YYYYMMDD}.pdf` | Figure 5 — COIN volatility skew, market vs calibrated | `skew_calibration_main.py` |
| `Figure_6_filtered_liquidity_process.pdf` | Figure 6 — Filtered liquidity process time series | `report_collar_asian.py` |

Figures 3 (VaR surface) and 7 (VaR term structure) are produced by
`run_var_kimyi2025.py`. If those specific PDFs are missing after a full
pipeline run, check the run_var script's output cells — they may still be
saving under a diagnostic name.

## Diagnostic / supporting plots

| File | Purpose |
| --- | --- |
| `daily_prices_returns_SPX.pdf`, `daily_prices_returns_COIN.pdf` | Sanity check: daily prices and returns for each underlying |
| `daily_var_comparison.pdf` | LA vs BS VaR by valuation date (Table 2 data visualised) |
| `pl_ladder.pdf` | Per-position P&L ladder under simulated spot shocks |
| `run_var_density_*.pdf`, `run_var_filtered_liquidity_process.pdf` | Diagnostic-only re-renders of Figures 1, 2, 6 from the VaR-focused script; not the paper's canonical versions |

## Regenerating

```bash
python Scripts/reproduce_paper.py
```

Set individual `CONFIG` flags in `reproduce_paper.py` to skip steps.
