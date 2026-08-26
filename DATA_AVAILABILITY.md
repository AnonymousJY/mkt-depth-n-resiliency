# Data Availability Statement

This replication package accompanies the paper **"Systematic Liquidity Risk
Management: A Novel Perspective on Derivatives"** by Yi and Kim.

The paper uses three categories of market data. This document lists each
category, its source, its licensing status, and whether it is included in
this replication package.

---

## 1. Equity spot price history — Included

**Underlying tickers:** `^SPX` (S&P 500 index), `COIN` (Coinbase Global, Inc.)

**Source:** [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)
(adjusted close series), which aggregates publicly available end-of-day
prices from Yahoo Finance and similar providers.

**Licensing:** Publicly available financial market data. No redistribution
restriction to the best of our knowledge.

**Included in this package:** Yes.
* `data/snapshots/prices_SPX.csv`
* `data/snapshots/prices_COIN.csv`

These committed CSV snapshots make the P-measure calibration
(`Scripts/run_pmle_kimyi2025.py`) and downstream VaR simulation fully
reproducible without any network access.

To refresh with the latest vendor data (which is revised over time), run
`python Scripts/export_snapshots.py` in an environment with network access.

---

## 2. Equity option chain and implied volatility surface — NOT included

**Underlying tickers:** SPX and COIN option chains as of the eleven valuation
dates (April 2–16, 2025).

**Source:** Cboe LiveVol
[DataShop](https://datashop.cboe.com/), end-of-day option quotes and
Cboe-calculated implied volatilities.

**Licensing:** Proprietary. Cboe's
[Policies Applicable to Historical Data Services](https://datashop.cboe.com/documents/DataShop_Policies_for_Historical_Data_Services.pdf)
prohibit redistribution of the underlying data by subscribers. Academic use
requires an accredited educational institution as subscriber, execution of the
Historical Market Data Subscription Agreement and Academic Subscriber
Addendum, and expressly excludes commercial or industry-funded research.

**Included in this package:** No. The raw option-level data cannot be
redistributed under Cboe's licensing terms.

**How to obtain the same data:**
1. Register at [https://datashop.cboe.com/](https://datashop.cboe.com/).
2. Purchase (or apply for academic discount for) the **End-of-Day (EOD)
   Options** dataset for the underlyings and dates listed below.
3. Filter to the eleven valuation dates: April 2, 3, 4, 7, 8, 9, 10, 11, 14,
   15, 16, 2025.
4. Underlyings required: `SPX` (S&P 500 index options) and `COIN`
   (Coinbase equity options).
5. Fields required: strike, expiration date, option type (call/put),
   bid, ask, last, volume, open interest, and Cboe-calculated implied
   volatility.

The calibration scripts (`Scripts/skew_calibrate_*.py`, notebooks
`skew_calibration_*.ipynb`) read the option data via
`Library/DataAccess.py`. To adapt them to your obtained dataset, place the
option chain data under `data/options/` and update the loader accordingly.
See `data/options/README.md` for details.

---

## 3. Q-measure calibrated parameters — Included (as derivatives, not raw data)

The Q-measure jump-diffusion parameters calibrated from the SPX and COIN
implied volatility surfaces are stored under `Study/Estimated Parameters QLSQ/`
as pickle and parquet files. These are model-derived summary parameters
computed by our calibration code from the underlying Cboe implied volatility
data. They are included in this package because they are our own derived
outputs, not the raw proprietary data.

A replicator who has independently obtained the Cboe data (per Section 2 above)
can regenerate these files by running the calibration scripts. Otherwise, the
cached files allow downstream reproduction of the paper's derivative pricing
and VaR results without needing the raw option data.

---

## 4. Portfolio composition (worked example) — Included

The collar / vanilla-hedge portfolio on COIN with SPX as systematic proxy is
defined in code at `Scripts/load_portfolio.py`. An optional CSV snapshot may
live at `data/snapshots/portfolio.csv`. Fully reproducible.

---

## Summary

| Category | Source | In package? | Notes |
| --- | --- | --- | --- |
| SPX / COIN spot prices | FinanceDataReader | Yes | Public data |
| SPX / COIN option chains + implied vols | Cboe LiveVol DataShop | **No** | Proprietary; obtain independently |
| Q-calibrated parameters (derived) | This paper's code | Yes | Cached outputs |
| Portfolio definition | This paper's code | Yes | In `Scripts/load_portfolio.py` |

For questions about accessing the Cboe option data, contact
`sales@livevol.com`.

For questions about this replication package or the paper's methodology,
contact the corresponding author.
