# mkt-depth-n-resiliency

Code accompanying the research paper
**"Systematic Liquidity Risk Management: A Novel Perspective on Derivatives"**
(Yi & Kim), SSRN: https://dx.doi.org/10.2139/ssrn.5454874

The code calibrates and simulates the paper's liquidity-adjusted jump-diffusion
model, prices the worked-example derivative portfolio (a discrete-Asian collar
with vanilla hedges on COIN, with `^SPX` as the systematic liquidity proxy),
and produces the paper's tables and figures.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `Library/` | Core engine: option pricers (`OptionPricerBSM1973`, `OptionPricerHeston1993`, `OptionPricerKou2002`, `OptionPricerKimYi2025`), the risk engine (`RiskEngineKimYi2025`), skew calibrators, RNG abstraction (`Random`), Monte Carlo statistics, payoff/path-dependent machinery. |
| `Library/DataAccess.py` | Unified data layer — committed snapshots by default, live FinanceDataReader / PostgreSQL optionally. **New; see Data modes below.** |
| `Scripts/` | Entry points: `run_pmle_kimyi2025.py` (P-MLE calibration), `run_var_kimyi2025.py` (VaR), `skew_calibration_*.py` (volatility skew), `load_portfolio.py` (portfolio definition), `export_snapshots.py` (snapshot generation). |
| `Study/` | Analyses and outputs: `Collar Asian/` notebooks, cached P-MLE parameter CSVs (`Estimated Parameters PMLE/`), cached volatility calibrations (`Estimated Parameters QLSQ/`, `Vol Surface From Model/`). |
| `data/snapshots/` | Committed, frozen copies of live data sources (**new**; see Data modes). |
| `requirements.txt` | Python dependencies (**new**). |

---

## Installation

Python 3.10+ is recommended (the project's compiled artifacts are cpython-310).

### Recommended: conda (`environment.yml`)

PyMC officially recommends conda-forge — it builds PyTensor's C / BLAS toolchain
cleanly, which matters in particular on macOS. Four packages
(`FinanceDataReader`, `databento`, `ustreasurycurve`, `nelson-siegel-svensson`)
are not on conda-forge, so `environment.yml` installs them via pip *within* the
same conda environment (conda first, pip last).

```bash
conda env create -f environment.yml
conda activate liquidity
```

For an exact-reproducible environment later:

```bash
conda env export > environment-lock.yml
```

### Alternative: pip (`requirements.txt`)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements-lock.txt   # exact-reproducible lock file
```

Note: the pure-pip path builds PyMC/PyTensor from source and needs a working
C compiler; if it fails on macOS, prefer the conda route above.

Optional GPU acceleration: `ImportLibs.py` transparently swaps NumPy for CuPy
when a CUDA GPU is present. Install the wheel/package matching your CUDA
toolkit (`conda install -c conda-forge cupy`, or `pip install cupy-cuda12x`);
leave it uninstalled to run on CPU.

---

## Data availability

Full data source statement is in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
Short version:

- **Equity spot prices** (SPX, COIN): included as committed snapshots under
  `data/snapshots/`. Public data via FinanceDataReader.
- **Cboe implied volatility surface** (SPX, COIN option chains): **NOT
  included**. Proprietary Cboe LiveVol DataShop data whose licensing prohibits
  redistribution. Replicators who wish to regenerate the volatility skew
  calibration (Figures 4-5) must obtain this data independently; see
  `data/options/README.md` for instructions.
- **Q-calibrated parameters** (derived from Cboe implied vols): included as
  cached outputs under `Study/Estimated Parameters QLSQ/`. These allow the
  downstream derivative pricing, VaR simulation, and figure generation to be
  reproduced without needing the raw option chain data.

---

## Data modes

Historically the scripts read price history live from FinanceDataReader and
read/wrote estimated parameters to a local PostgreSQL instance. That made the
repository impossible to reproduce later: FinanceDataReader history is revised
by the vendor over time, and the database was not part of the repository.

The PostgreSQL dependency has been removed. P-MLE parameter estimates are read
from and written directly to per-date CSVs under
`Study/Estimated Parameters PMLE/`; portfolio definitions are constructed in
code by `Scripts/load_portfolio.build_portfolio()` (with an optional CSV
snapshot at `data/snapshots/portfolio.csv`). The only remaining live source is
the price history, controlled by `Library/DataAccess.py` and the
`MKTDEPTH_DATA_MODE` environment variable:

| Mode | Behaviour |
| --- | --- |
| `snapshot` *(default)* | Reads committed CSV snapshots from `data/snapshots/` and the per-date P-MLE parameter CSVs under `Study/Estimated Parameters PMLE/`. No network, no database. This is what makes the repository a self-contained replication package. |
| `live` | Restores the FinanceDataReader pull. Use this to refresh prices or extend the sample. |

```bash
export MKTDEPTH_DATA_MODE=snapshot   # default; committed inputs only
export MKTDEPTH_DATA_MODE=live       # refresh prices from FinanceDataReader
```

### Generating the price snapshots

The snapshots are not yet committed. To create them, run once in an
environment with network access:

```bash
python Scripts/export_snapshots.py
```

This writes `data/snapshots/prices_SPX.csv` and `data/snapshots/prices_COIN.csv`.
Commit those files so collaborators and referees can reproduce the results
without re-pulling vendor data.

---

## Reproducing the paper's outputs

**One-shot entry point:**

```bash
python Scripts/reproduce_paper.py
```

`reproduce_paper.py` is the orchestrator that runs every step needed to
regenerate every table and figure in the paper. Toggle any step in the
`CONFIG` dict at the top of the file to skip it (useful when re-running a
single figure without redoing multi-hour calibrations).

Under the hood it invokes four focused pipeline scripts, each of which
owns a distinct subset of the paper's outputs:

| Output | Generated by | Notes |
| --- | --- | --- |
| **Table 1** — P-MLE parameter estimates (systematic `^SPX` and idiosyncratic `COIN`) | `Scripts/run_pmle_kimyi2025.py` | The estimates already ship per-date under `Study/Estimated Parameters PMLE/`; re-running performs the MCMC calibration. Incremental — skips dates that already have a CSV. |
| **Figures 4–5** — SPX / COIN volatility skew | `Scripts/skew_calibration_main.py` (with `skew_calibration_heston1993.py`, `skew_calibration_kimyi2025.py`) | Requires raw Cboe option data via `LIQUIDITY_CBOE_DATA_DIR`; skips cleanly if not present. Pre-generated PDFs ship at `Scripts/SPX_VOL_SKEW_*.pdf` and `Scripts/COIN_VOL_SKEW_*.pdf`. |
| **Figures 1, 2, 6** + **Table 2 CSVs** — return / P&L density, filtered liquidity, VaR outputs | `Study/Collar Asian/report_collar_asian.py` | Authoritative source for these outputs. |
| **Figures 3, 7** — VaR surface and term structure | `Scripts/run_var_kimyi2025.py` | Authoritative source for these two figures. Any density / filtered-liquidity plots this script also emits are diagnostic (not the paper's canonical versions). |

All entry points should be run from the repository root so that the
`Library.` / `Scripts.` package imports resolve.

---

## Known reproducibility caveats

1. **Live data drift.** In `live` mode, FinanceDataReader history is revised by
   the vendor over time; only the committed snapshots give bit-for-bit
   reproducibility.
2. **MCMC sampler.** P-MLE uses the `nutpie` NUTS sampler. The seed is fixed,
   but cross-platform / cross-version determinism of the sampler backend is not
   guaranteed; expect small numerical differences across environments.
3. **NumPy vs. CuPy.** `ImportLibs.py` silently switches the array backend when
   a GPU is present, which changes the numerical path. For reference results,
   run on CPU.
4. **Dependency versions.** `requirements.txt` carries known-compatible floors,
   not the exact versions used for the paper. Commit a `requirements-lock.txt`
   from the original environment for exact reproduction.
5. **Data hygiene.** `Study/Estimated Parameters PMLE/^SPX/` currently also
   contains `COIN`-named files; the canonical per-underlying path is
   `Study/Estimated Parameters PMLE/{underlying}/estimated_params_pmle_{underlying}_{YYYYMMDD}.csv`.

---

## License

MIT — see `LICENSE`.

---

The codes were used to generate the numerical outputs in the research paper *Systematic Liquidity Risk Management: A Novel Perspective on Derivatives*, which can be found in https://dx.doi.org/10.2139/ssrn.5454874. The codes and the outputs generated from the codes are for illustration purpose only.
