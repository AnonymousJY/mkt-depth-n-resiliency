#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts/skew_calibrate_systematic.py — systematic-stage skew calibration
(Kim-Yi 2025 model), scripted from Scripts/skew_calibration_kimyi2025.ipynb.

For each valuation date in the configured window, fits the five common
Kim-Yi (2025) parameters (sigma, pprob, lamb, eta1, eta2) to the systematic
underlying's (config_skew.SYSTEMATIC_TICKER, e.g. "^SPX") implied-vol smile
via vega-weighted least squares (Library.SkewCalibrationKimYi2025.
KimYiSkewCalibrationSystematic + scipy.optimize.minimize).

Results are cached, incrementally, to a pickle keyed by "{ticker}-{YYYYMMDD}"
-> scipy OptimizeResult, at config_skew.OUTPUT_DIR / OUTPUT_PKL_NAME. Dates
already present in the cache are skipped, so reruns only calibrate new dates
-- mirrors the "skip what's already estimated" pattern used by
Scripts/run_pmle_kimyi2025.py, but for the QLSQ (option-implied) calibration
instead of the P-MLE (historical-return) calibration.

The idiosyncratic stage (KimYiSkewCalibrationIdiosyncratic, conditional on
these systematic parameters) is intentionally out of scope for this script;
see the "idiosyncratic" cells of skew_calibration_kimyi2025.ipynb.

All inputs live in Scripts/config_skew.py. With config_skew.USE_SYNTHETIC_DATA =
True (the default), the script calibrates against option data synthesized
from a known parameter set (via the real KimYiSkewCalibrationSystematic
model) so the optimization path can be validated without market data or a
network connection. Set USE_SYNTHETIC_DATA = False and point
DATA_PATH_IMPLIED_VOL at real snapshots to run for real.

Run from the repository root:

    python Scripts/skew_calibrate_systematic.py
"""

from __future__ import annotations

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Library.OptionPricerBSM1973 import BlackScholesMertonPut
from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic

# Module logger only -- no handler/level is configured here. Configuration
# (level, format, handlers) is the entry point's job; see configure_logging()
# and the __main__ guard below. Importing this module elsewhere (tests,
# notebooks, other scripts) must not silently install a root handler.
logger = logging.getLogger(__name__)

PARAM_NAMES = ["dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------
def target_expiry_days(valuation_date: pd.Timestamp) -> int:
    """Target days-to-expiry for a valuation date, per config_skew.EXPIRY_MAP,
    with config_skew.EXPIRY_ADJUSTMENTS applied."""
    target = cfg.EXPIRY_MAP[valuation_date.weekday()]
    for beg, end, offset in cfg.EXPIRY_ADJUSTMENTS:
        if pd.to_datetime(beg) <= valuation_date < pd.to_datetime(end):
            target += offset
    return target


# ---------------------------------------------------------------------------
# Real market data (CSV snapshots + UST curve)
# ---------------------------------------------------------------------------
def load_market_vol_data(ticker: str) -> pd.DataFrame:
    """Load and clean implied-vol snapshots for one ticker, matching the
    transformations in skew_calibration_kimyi2025.ipynb."""
    safe_name = ticker.split("^")[-1]
    path = os.path.join(cfg.DATA_PATH_IMPLIED_VOL, safe_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No implied-vol data folder at {path}. Point "
            f"config_skew.DATA_PATH_IMPLIED_VOL at your data, or set "
            f"USE_SYNTHETIC_DATA = True to run against synthetic data."
        )

    files = sorted(os.listdir(path))
    df = pd.concat((pd.read_csv(os.path.join(path, f)) for f in files), ignore_index=True)

    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["iEXPIRY"] = (df["expiration"] - df["quote_date"]).dt.days
    df["dEXPIRY"] = df["iEXPIRY"] / 365.0
    df["dUND_PRICE"] = df["active_underlying_price_1545"]
    df["dUND_STRIKE"] = df["strike"]
    df["dMONEYNESS"] = df["dUND_STRIKE"] / df["dUND_PRICE"] * 100.0
    df["dMKT_IMP_VOL"] = df["implied_volatility_1545"]
    df["bIS_CALL_OPTION"] = df["option_type"] == "C"

    mask = (
        (df["trade_volume"] > cfg.MIN_TRADE_VOLUME)
        & (df["iEXPIRY"] > cfg.MIN_EXPIRY_DAYS)
        & (df["dMKT_IMP_VOL"] > 0)
    )
    df = df.loc[mask]

    put_lo, put_hi = cfg.MONEYNESS_PUT_RANGE
    call_lo, call_hi = cfg.MONEYNESS_CALL_RANGE
    is_put_in_range = (~df["bIS_CALL_OPTION"]) & df["dMONEYNESS"].between(put_lo, put_hi)
    is_call_in_range = df["bIS_CALL_OPTION"] & df["dMONEYNESS"].between(call_lo, call_hi)
    df = df.loc[is_put_in_range | is_call_in_range].copy()

    df["dDIVIDEND_YIELD"] = cfg.DIVIDEND_YIELD_PCT / 100.0
    return df


def attach_risk_free_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Populate df['dRISK_FREE_RATE'] per row, by valuation date + expiry."""
    if cfg.RATE_SOURCE == "flat":
        df["dRISK_FREE_RATE"] = cfg.FLAT_RATE
        return df

    if cfg.RATE_SOURCE != "ust_curve":
        raise ValueError(f"Unknown RATE_SOURCE: {cfg.RATE_SOURCE!r}")

    import ustreasurycurve as ustcurve
    from nelson_siegel_svensson.calibrate import calibrate_nss_ols

    from Library.Utility import UST_TENOR_MAP

    dates = df["quote_date"].unique()
    rates_df = ustcurve.nominalRates(
        pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
    ).set_index("date")
    rates_df = (
        pd.DataFrame(index=pd.date_range(rates_df.index.min(), rates_df.index.max()))
        .join(rates_df)
        .ffill(axis=1)
        .bfill(axis=1)
        .ffill()
        .bfill()
    )
    tenors = np.array([UST_TENOR_MAP[c] for c in rates_df.columns])

    for quote_date in dates:
        curve_fit, _ = calibrate_nss_ols(tenors, rates_df.xs(quote_date).to_numpy() / 100.0)
        mask = df["quote_date"] == quote_date
        df.loc[mask, "dRISK_FREE_RATE"] = df.loc[mask, "dEXPIRY"].apply(curve_fit)

    return df


def attach_vega_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Populate df['dVEGA'] with per-row Black-Scholes-Merton put vega
    (used as the option_weights passed into KimYiSkewCalibrationSystematic)."""
    vegas = np.empty(len(df))
    for i, row in enumerate(df.itertuples()):
        vega = BlackScholesMertonPut(
            und_price=np.array(row.dUND_PRICE),
            und_strike=np.array(row.dUND_STRIKE),
            risk_free_rate=np.array(row.dRISK_FREE_RATE),
            dividend_yield=np.array(row.dDIVIDEND_YIELD),
            time_to_expiry=np.array(row.dEXPIRY),
        ).vega(np.array(row.dMKT_IMP_VOL))
        vegas[i] = vega.reshape(-1)[0] / 100.0
    df = df.copy()
    df["dVEGA"] = vegas
    return df


# ---------------------------------------------------------------------------
# Synthetic data (dev / test, no market data or network required)
# ---------------------------------------------------------------------------
TRUE_PARAMS_SYNTHETIC = {
    "dSIGMA": 0.18, "dPPROB": 0.25, "dLAMB": 6.0, "dETA1": 20.0, "dETA2": 8.0,
}


def _make_synthetic_market_vol_data(ticker: str) -> pd.DataFrame:
    """Synthesize an implied-vol smile per valuation date, generated from
    TRUE_PARAMS_SYNTHETIC via the real KimYiSkewCalibrationSystematic model
    (plus noise), so calibrate_date() can be validated end-to-end."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    dates = pd.bdate_range(cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END)

    S0, r, q = 100.0, 0.04, cfg.DIVIDEND_YIELD_PCT / 100.0
    moneyness = np.linspace(60.0, 140.0, cfg.SYNTHETIC_N_MONEYNESS)
    strikes = moneyness / 100.0 * S0
    is_call = moneyness >= 100.0

    rows = []
    for quote_date in dates:
        expiry_days = target_expiry_days(quote_date)
        T = expiry_days / 365.0

        fitter = KimYiSkewCalibrationSystematic(
            mkt_imp_vol=np.zeros_like(strikes),
            und_price=np.full_like(strikes, S0),
            und_strike=strikes,
            risk_free_rate=np.full_like(strikes, r),
            dividend_yield=np.full_like(strikes, q),
            time_to_expiry=np.full_like(strikes, T),
            is_call_option=is_call,
            option_weights=np.ones_like(strikes),
        )
        x_true = np.array([TRUE_PARAMS_SYNTHETIC[p] for p in PARAM_NAMES])
        true_vol = fitter.model_vol(x=x_true).reshape(-1)
        noisy_vol = true_vol + rng.normal(0, 0.002, size=true_vol.shape[0])

        # model_vol() stacks puts then calls (see SkewCalibrationKimYi2025.py)
        ordered_is_call = np.concatenate([is_call[~is_call], is_call[is_call]])
        ordered_strike = np.concatenate([strikes[~is_call], strikes[is_call]])

        for k, c, v in zip(ordered_strike, ordered_is_call, noisy_vol):
            rows.append({
                "quote_date": quote_date,
                "underlying_symbol": ticker,
                "iEXPIRY": expiry_days,
                "dEXPIRY": T,
                "dUND_PRICE": S0,
                "dUND_STRIKE": k,
                "dMONEYNESS": k / S0 * 100.0,
                "dMKT_IMP_VOL": max(v, 1e-4),
                "bIS_CALL_OPTION": bool(c),
                "dRISK_FREE_RATE": r,
                "dDIVIDEND_YIELD": q,
                "dVEGA": 1.0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_date(df_date: pd.DataFrame):
    """Fit the systematic Kim-Yi (2025) parameters to one date's smile."""
    weights = df_date["dVEGA"].to_numpy()
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Vega weights sum to zero; check the input smile.")

    fitter = KimYiSkewCalibrationSystematic(
        mkt_imp_vol=df_date["dMKT_IMP_VOL"].to_numpy(),
        und_price=df_date["dUND_PRICE"].to_numpy(),
        und_strike=df_date["dUND_STRIKE"].to_numpy(),
        risk_free_rate=df_date["dRISK_FREE_RATE"].to_numpy(),
        dividend_yield=df_date["dDIVIDEND_YIELD"].to_numpy(),
        time_to_expiry=df_date["dEXPIRY"].to_numpy(),
        is_call_option=df_date["bIS_CALL_OPTION"].to_numpy(),
        option_weights=weights / weight_sum,
    )

    return minimize(
        fitter.target,
        x0=cfg.INITIAL_VALUES,
        method=cfg.OPTIMIZER_METHOD,
        bounds=[cfg.BOUNDS[p] for p in PARAM_NAMES],
        tol=cfg.OPTIMIZER_TOL,
        options={"maxiter": cfg.OPTIMIZER_MAXITER},
    )


# ---------------------------------------------------------------------------
# Result cache (pickle, shared format with the notebooks)
# ---------------------------------------------------------------------------
def cache_path() -> str:
    name = cfg.OUTPUT_PKL_NAME_SYNTHETIC if cfg.USE_SYNTHETIC_DATA else cfg.OUTPUT_PKL_NAME
    return os.path.join(cfg.OUTPUT_DIR, name)


def load_cache() -> dict:
    path = cache_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> str:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = cache_path()
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """Entry-point logging setup. Only called from the __main__ guard, so
    importing this module never has the side effect of installing handlers
    on the root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    ticker = cfg.SYSTEMATIC_TICKER

    if cfg.USE_SYNTHETIC_DATA:
        logger.warning(
            "config_skew.USE_SYNTHETIC_DATA is True: calibrating against "
            "synthetic data, not market data."
        )
        mkt_vol_df = _make_synthetic_market_vol_data(ticker)
        valuation_dates = pd.bdate_range(
            cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END
        )
    else:
        mkt_vol_df = load_market_vol_data(ticker)
        mkt_vol_df = attach_risk_free_rate(mkt_vol_df)
        mkt_vol_df = attach_vega_weights(mkt_vol_df)
        valuation_dates = pd.bdate_range(cfg.VALUATION_DATE_BEG, cfg.VALUATION_DATE_END)
    available_dates = set(pd.to_datetime(mkt_vol_df["quote_date"].unique()))

    cache = load_cache()
    n_done = 0

    for valuation_date in valuation_dates:
        if valuation_date not in available_dates:
            continue

        date_str = valuation_date.strftime("%Y%m%d")
        key = f"{ticker}-{date_str}"
        if key in cache:
            continue

        target_dte = target_expiry_days(valuation_date)
        mask = (
            (mkt_vol_df["underlying_symbol"] == ticker)
            & (mkt_vol_df["quote_date"] == valuation_date)
            & (mkt_vol_df["iEXPIRY"] == target_dte)
        )
        df_date = mkt_vol_df.loc[mask].sort_values("dMONEYNESS")
        if df_date.empty:
            logger.info("%s: no rows at target DTE=%d, skipping", date_str, target_dte)
            continue

        result = calibrate_date(df_date)
        cache[key] = result
        save_cache(cache)
        n_done += 1

        params = dict(zip(PARAM_NAMES, result.x))
        param_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
        logger.info(
            "%s  converged=%s  n_obs=%d  %s",
            date_str, result.success, len(df_date), param_str,
        )

    logger.info("Calibrated %d new date(s); %d total in cache.", n_done, len(cache))
    logger.info("Cache: %s", cache_path())


if __name__ == "__main__":
    configure_logging()
    main()
