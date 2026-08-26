# data/options/

Placeholder directory for the SPX and COIN end-of-day option chain data used
by the volatility skew calibration scripts and the Q-measure derivative
pricing.

**This data is not included in the replication package.** It is proprietary
Cboe LiveVol DataShop data and cannot be redistributed under Cboe's licensing
terms. See the top-level [`DATA_AVAILABILITY.md`](../../DATA_AVAILABILITY.md)
for full details and instructions on how to obtain it.

## What replicators should place here

For each of the eleven valuation dates (April 2, 3, 4, 7, 8, 9, 10, 11, 14,
15, 16, 2025), the option chain snapshot for SPX and COIN. The exact file
format expected by `Library/DataAccess.py` is documented in that module.

## Alternative: use the cached Q-calibration outputs

If a replicator does not have access to the Cboe data, the cached
Q-calibrated parameters under `Study/Estimated Parameters QLSQ/` allow the
downstream derivative pricing, VaR simulation, and figure generation to be
reproduced without needing the raw option chain data.

Only the volatility skew calibration itself (Figures 4 and 5) requires the
raw option data.
