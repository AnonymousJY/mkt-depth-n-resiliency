#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts/config_skew.py — inputs for Scripts/skew_calibrate_systematic.py.

Mirrors the systematic-stage calibration cell in
Scripts/skew_calibration_kimyi2025.ipynb, but as a set of editable knobs
instead of hard-coded notebook constants, so the calibration can be re-run
from the command line and incrementally extended to new dates.

Edit these values to point at real data / a new sample window; the
calibration logic in the script itself should not need to change.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives in Scripts/, so the repo root is one level up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root folder holding one subfolder per underlying (e.g. "SPX/", "COIN/"),
# each containing daily implied-vol CSV snapshots (quote_date, expiration,
# strike, option_type, active_underlying_price_1545,
# implied_volatility_1545, trade_volume, underlying_symbol, ...).
# Same layout / location as DATA_PATH_STR in skew_calibration_kimyi2025.ipynb.
DATA_PATH_IMPLIED_VOL = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Data/Implied Volatility"
)

# Where calibration results are cached. Uses the same pickle the notebooks
# read/write, keyed by "{ticker}-{YYYYMMDD}" -> scipy OptimizeResult, so this
# script and the notebooks can share a cache incrementally.
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Study", "Estimated Parameters QLSQ")
OUTPUT_PKL_NAME = "kimyi2025_vol_calibration.pkl"

# Used instead of OUTPUT_PKL_NAME whenever USE_SYNTHETIC_DATA is True, so a
# synthetic smoke test can never read from or write into the real cache
# above (real cache keys could otherwise collide with synthetic test dates
# and be mistaken for "already calibrated").
OUTPUT_PKL_NAME_SYNTHETIC = "kimyi2025_vol_calibration_synthetic_smoketest.pkl"

# ---------------------------------------------------------------------------
# Underlying / sample window
# ---------------------------------------------------------------------------
SYSTEMATIC_TICKER = "^SPX"
DIVIDEND_YIELD_PCT = 1.25  # percentage points, matches DIVIDEND_YIELDS['^SPX']

VALUATION_DATE_BEG = "2025-03-18"
VALUATION_DATE_END = "2025-04-17"

# Target days-to-expiry by weekday (Mon=0 ... Fri=4), matching the weekly SPX
# options calendar used in the notebook.
EXPIRY_MAP = {0: 11, 1: 10, 2: 9, 3: 8, 4: 7}

# One-off calendar adjustments: (date_beg, date_end_exclusive, day_offset)
# applied to the EXPIRY_MAP lookup for dates in [date_beg, date_end). The
# notebook default accounts for the April 2025 holiday-shortened week.
EXPIRY_ADJUSTMENTS = [
    ("2025-04-07", "2025-04-14", -1),
]

# ---------------------------------------------------------------------------
# Option filters (applied before calibration, matching the notebook)
# ---------------------------------------------------------------------------
MIN_TRADE_VOLUME = 0     # keep rows with trade_volume > MIN_TRADE_VOLUME
MIN_EXPIRY_DAYS = 5      # keep rows with iEXPIRY > MIN_EXPIRY_DAYS
MONEYNESS_PUT_RANGE = (50.0, 100.0)   # puts:  50% <= moneyness <= 100%
MONEYNESS_CALL_RANGE = (100.0, 150.0)  # calls: 100% <= moneyness <= 150%

# ---------------------------------------------------------------------------
# Risk-free rate curve
# ---------------------------------------------------------------------------
# "ust_curve": pull the UST par curve via ustreasurycurve and NSS-fit it
#              per valuation date (requires network + nelson_siegel_svensson).
# "flat":      use FLAT_RATE for every date/expiry (no network; for dev/test
#              or when a curve fetch isn't available).
RATE_SOURCE = "ust_curve"
FLAT_RATE = 0.05

# ---------------------------------------------------------------------------
# Kim-Yi (2025) systematic calibration: sigma, pprob, lamb, eta1, eta2
# ---------------------------------------------------------------------------
INITIAL_VALUES = np.array([0.2, 0.2, 5.0, 22.0, 7.0])

BOUNDS = {
    "dSIGMA": (1e-4, None),
    "dPPROB": (0.0, 1.0),
    "dLAMB": (1e-4, None),
    "dETA1": (1.5, None),
    "dETA2": (0.5, None),
}

OPTIMIZER_METHOD = "SLSQP"
OPTIMIZER_TOL = 1e-6
OPTIMIZER_MAXITER = 10_000

# ---------------------------------------------------------------------------
# Development / testing
# ---------------------------------------------------------------------------
# If True, the script skips DATA_PATH_IMPLIED_VOL and RATE_SOURCE entirely
# and calibrates against synthetic option data generated from a known set of
# "true" parameters (via the real KimYiSkewCalibrationSystematic model), so
# the optimization path can be validated end-to-end without market data.
USE_SYNTHETIC_DATA = True
RANDOM_SEED = 20250409

# The Newton IV-inversion inside KimYiSkewCalibrationSystematic.model_vol is
# not cheap per point: a handful of dates/strikes is enough to smoke-test the
# optimizer end-to-end. Kept separate from VALUATION_DATE_BEG/END so a
# synthetic smoke test never has to touch the production window.
SYNTHETIC_VALUATION_DATE_BEG = "2019-01-02"
SYNTHETIC_VALUATION_DATE_END = "2019-01-04"
SYNTHETIC_N_MONEYNESS = 9
