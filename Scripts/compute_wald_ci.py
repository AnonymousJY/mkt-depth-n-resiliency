"""
Scripts/compute_wald_ci.py -- Wald (frequentist) 95% and 99% confidence
intervals for the P-MLE parameter estimates.

Rationale
---------
The original v13 paper reports Bayesian 95% credible intervals from an
MCMC posterior (Table 1). The referee's complaint that the intervals are
"too wide" partly reflects a mis-match between the referee's frequentist
expectation and the Bayesian estimator. This script produces the standard
frequentist alternative — Wald confidence intervals derived from the
observed Fisher information at the MLE point estimate:

    SE       = sqrt(diag(-H^{-1}))
    Wald CI  = theta_hat +/- z_alpha * SE

with H the log-likelihood Hessian at theta_hat (computed numerically via
numdifftools), z_{95%} = 1.96, z_{99%} = 2.576.

The point estimates and the log-likelihood function are the same as those
used by the MCMC estimation in Scripts/run_pmle_kimyi2025.py; only the
uncertainty summary is different.

Approach
--------
For each cached (ticker, valuation_date) row in Study/Estimated Parameters
PMLE/:

  1. Load point estimates via Library.DataAccess.get_pmle_params.
  2. Reconstruct the return vector used for that estimation (same
     lookback + same price source as run_pmle_kimyi2025.py).
  3. Build a compiled log-likelihood evaluator over NATURAL parameter space
     using Library.RiskEngineKimYi2025.KimYiLogLike (bypassing pymc's
     log / tanh transforms so SEs are directly on the natural parameters
     the paper reports).
  4. Compute the numerical Hessian at the point estimate via
     numdifftools.Hessian.
  5. Invert -H to get the asymptotic covariance; extract per-parameter SE
     and construct Wald 95% and 99% CIs.
  6. Write results as a wide Parquet keyed by (ticker, date, param).

Practical safeguards
--------------------
  * If -H is not positive-definite (singular, or negative diagonal in
    -H^{-1}), that parameter's SE is set to NaN with a warning. This can
    happen when the likelihood surface is locally flat or the numerical
    step size is inappropriate.
  * The default finite-differencing step (1e-4) is small enough to
    resolve local curvature but large enough to avoid catastrophic
    cancellation. Adjust via --hessian-step if needed.

Requires: numdifftools. Install via
    conda install -c conda-forge numdifftools
or
    pip install numdifftools

Run from the repository root:

    # All cached (ticker, date) rows in the window
    python Scripts/compute_wald_ci.py \\
        --valuation-date-beg 2025-04-02 \\
        --valuation-date-end 2025-04-16

    # Only SPX
    python Scripts/compute_wald_ci.py --systematic-only \\
        --valuation-date-beg 2025-04-02 --valuation-date-end 2025-04-16

    # Custom output path
    python Scripts/compute_wald_ci.py \\
        --valuation-date-beg 2025-04-02 --valuation-date-end 2025-04-16 \\
        --out Study/wald_ci_kimyi2025.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from Scripts.load_portfolio import get_idiosyncratic_ids
from Library.DataAccess import get_price_panel, get_pmle_params, pmle_params_exists
from Library.RiskEngineKimYi2025 import KimYiLogLike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Standard normal two-sided quantiles. Using higher-precision constants
# than 1.96 / 2.576 so the reported CI widths match what a rigorous reader
# would compute themselves.
Z_95 = 1.959963984540054   # Two-sided 95% (i.e., 0.975 quantile of N(0,1))
Z_99 = 2.5758293035489004  # Two-sided 99% (i.e., 0.995 quantile of N(0,1))

SYSTEMATIC_PARAM_NAMES = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
IDIOSYNCRATIC_PARAM_NAMES = ["dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]

# Parameter bounds -- used only for reporting whether a computed CI extends
# beyond the parameter's admissible region. Not enforced during Hessian
# computation.
PARAM_BOUNDS = {
    "dALPHA":  (0.0, 1.0),
    "dSIGMA":  (0.0, np.inf),
    "dPPROB":  (0.0, 1.0),
    "dLAMB":   (0.0, np.inf),
    "dETA1":   (0.0, np.inf),
    "dETA2":   (0.0, np.inf),
    "dMUI":    (-np.inf, np.inf),
    "dKAPPAI": (0.0, np.inf),
    "dGAMMAI": (0.0, np.inf),
    "dBETAI":  (0.0, np.inf),
    "dRHOIX":  (-1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Compiled log-likelihood evaluators
# ---------------------------------------------------------------------------
def build_systematic_loglik(returns: np.ndarray, delta_t: float):
    """Compile a log-likelihood callable f(theta) -> scalar for the
    systematic (SPX-only) model.

    ``theta`` is a 6-vector in NATURAL parameter space:
        [alpha, sigma, pprob, lamb, eta1, eta2]

    Note: this bypasses pymc's log() and tanh() transforms so numdifftools
    computes the Hessian directly with respect to the natural parameters
    the paper reports.
    """
    # Deferred import so the module can be loaded even without pytensor
    # (e.g. for pure-python tests of the CLI).
    import pytensor
    import pytensor.tensor as pt

    theta = pt.dvector("theta")
    alpha  = theta[0]
    sigma  = theta[1]
    pprob  = theta[2]
    lamb   = theta[3]
    eta1   = theta[4]
    eta2   = theta[5]

    y_data = np.cumsum(returns).reshape((-1, 1))

    ll = KimYiLogLike(
        mui=np.array(0.0),
        kappai=np.array(0.0),
        gammai=np.array(1.0),
        betai=np.array(1.0),
        rhoix=np.array(0.0),
        alpha=alpha,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        dt=np.array(delta_t),
    ).logp(y=pt.as_tensor(y_data))

    total_ll = pt.sum(ll)
    return pytensor.function([theta], total_ll)


def build_idiosyncratic_loglik(
    idi_returns: np.ndarray,
    params_sys: dict,
    delta_t: float,
):
    """Compile a log-likelihood callable f(theta) -> scalar for the
    idiosyncratic model, conditional on the systematic parameters.

    ``theta`` is a 5-vector in NATURAL parameter space:
        [mui, kappai, gammai, betai, rhoix]

    ``params_sys`` must supply the six systematic parameters as
    {"dALPHA":..., "dSIGMA":..., "dPPROB":..., "dLAMB":..., "dETA1":..., "dETA2":...}.
    These are held fixed at their MLE values (i.e., we compute the
    idiosyncratic profile likelihood at the estimated systematic point).
    """
    import pytensor
    import pytensor.tensor as pt

    theta = pt.dvector("theta")
    mui    = theta[0]
    kappai = theta[1]
    gammai = theta[2]
    betai  = theta[3]
    rhoix  = theta[4]

    y_data = np.cumsum(idi_returns).reshape((-1, 1))

    ll = KimYiLogLike(
        mui=mui,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        alpha=np.array(params_sys["dALPHA"]),
        sigma=np.array(params_sys["dSIGMA"]),
        pprob=np.array(params_sys["dPPROB"]),
        lamb=np.array(params_sys["dLAMB"]),
        eta1=np.array(params_sys["dETA1"]),
        eta2=np.array(params_sys["dETA2"]),
        dt=np.array(delta_t),
    ).logp(y=pt.as_tensor(y_data))

    total_ll = pt.sum(ll)
    return pytensor.function([theta], total_ll)


# ---------------------------------------------------------------------------
# Hessian + Wald CI
# ---------------------------------------------------------------------------
def compute_numerical_hessian(f, x_star: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Compute the Hessian of scalar f at x_star numerically.

    Uses numdifftools.Hessian with a specified step size. If numdifftools
    is unavailable, raises ImportError with an install hint."""
    try:
        import numdifftools as nd
    except ImportError as e:
        raise ImportError(
            "numdifftools is required for Wald CI computation. Install via:\n"
            "  conda install -c conda-forge numdifftools\n"
            "  or  pip install numdifftools"
        ) from e

    # Wrap f in a plain Python callable so numdifftools can differentiate it
    # (pytensor-compiled functions typically accept a single array argument).
    def f_plain(x):
        return float(f(x))

    return nd.Hessian(f_plain, step=step)(x_star)


def wald_ci_from_hessian(
    theta_hat: np.ndarray,
    hessian: np.ndarray,
    z: float,
    param_names: List[str],
) -> tuple:
    """Given theta_hat, log-likelihood Hessian, and z-quantile, return
    (se, lower, upper) with SE = sqrt(diag(-H^{-1})) and CI = theta ± z * SE.

    If -H is not invertible, or if some diagonal of -H^{-1} is
    non-positive, the corresponding SE / CI entries are set to NaN with
    a warning."""
    n = len(theta_hat)
    se = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    try:
        cov = -np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        logger.warning(
            "Hessian is singular; cannot invert. All SEs set to NaN."
        )
        return se, lower, upper

    diag_cov = np.diag(cov)
    for i, name in enumerate(param_names):
        if diag_cov[i] <= 0:
            logger.warning(
                "Non-positive-definite Hessian at parameter %s (diag(cov)[%d] = %.4g). "
                "SE set to NaN.", name, i, diag_cov[i],
            )
            continue
        se[i] = np.sqrt(diag_cov[i])
        lower[i] = theta_hat[i] - z * se[i]
        upper[i] = theta_hat[i] + z * se[i]

    return se, lower, upper


# ---------------------------------------------------------------------------
# Per-date computation
# ---------------------------------------------------------------------------
def compute_systematic(
    ticker: str,
    valuation_date: str,
    returns: np.ndarray,
    delta_t: float,
    hessian_step: float,
) -> pd.DataFrame:
    """Compute Wald CIs for one systematic (ticker, date)."""
    point = get_pmle_params(valuation_date, ticker)
    theta_hat = np.array([float(point[name]) for name in SYSTEMATIC_PARAM_NAMES])

    logger.info("[%s %s] Building systematic log-likelihood ...", ticker, valuation_date)
    f_ll = build_systematic_loglik(returns, delta_t)

    logger.info("[%s %s] Computing numerical Hessian (6x6) ...", ticker, valuation_date)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress overflow chatter in tail regions
        hess = compute_numerical_hessian(f_ll, theta_hat, step=hessian_step)

    se, lo95, hi95 = wald_ci_from_hessian(theta_hat, hess, Z_95, SYSTEMATIC_PARAM_NAMES)
    _,  lo99, hi99 = wald_ci_from_hessian(theta_hat, hess, Z_99, SYSTEMATIC_PARAM_NAMES)

    return _rows_to_df(ticker, valuation_date, SYSTEMATIC_PARAM_NAMES,
                       theta_hat, se, lo95, hi95, lo99, hi99)


def compute_idiosyncratic(
    ticker: str,
    valuation_date: str,
    idi_returns: np.ndarray,
    systematic_ticker: str,
    delta_t: float,
    hessian_step: float,
) -> pd.DataFrame:
    """Compute Wald CIs for one idiosyncratic (ticker, date), conditional
    on the same-date systematic parameters."""
    point = get_pmle_params(valuation_date, ticker)
    theta_hat = np.array([float(point[name]) for name in IDIOSYNCRATIC_PARAM_NAMES])

    # Fetch the same-date systematic parameters (point estimates).
    sys_point = get_pmle_params(valuation_date, systematic_ticker)
    params_sys = {k: float(sys_point[k]) for k in SYSTEMATIC_PARAM_NAMES}

    logger.info("[%s %s] Building idiosyncratic log-likelihood conditional on %s ...",
                ticker, valuation_date, systematic_ticker)
    f_ll = build_idiosyncratic_loglik(idi_returns, params_sys, delta_t)

    logger.info("[%s %s] Computing numerical Hessian (5x5) ...", ticker, valuation_date)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hess = compute_numerical_hessian(f_ll, theta_hat, step=hessian_step)

    se, lo95, hi95 = wald_ci_from_hessian(theta_hat, hess, Z_95, IDIOSYNCRATIC_PARAM_NAMES)
    _,  lo99, hi99 = wald_ci_from_hessian(theta_hat, hess, Z_99, IDIOSYNCRATIC_PARAM_NAMES)

    return _rows_to_df(ticker, valuation_date, IDIOSYNCRATIC_PARAM_NAMES,
                       theta_hat, se, lo95, hi95, lo99, hi99)


def _rows_to_df(ticker, valuation_date, param_names,
                theta_hat, se, lo95, hi95, lo99, hi99) -> pd.DataFrame:
    """Assemble the per-parameter results into a tidy DataFrame with
    bound-violation flags."""
    rows = []
    for i, name in enumerate(param_names):
        lo, hi = PARAM_BOUNDS.get(name, (-np.inf, np.inf))
        bnd_violation_95 = (
            (not np.isnan(lo95[i]) and lo95[i] < lo)
            or (not np.isnan(hi95[i]) and hi95[i] > hi)
        )
        bnd_violation_99 = (
            (not np.isnan(lo99[i]) and lo99[i] < lo)
            or (not np.isnan(hi99[i]) and hi99[i] > hi)
        )
        rows.append({
            "sTICKER": ticker,
            "sVALUATION_DATE": valuation_date,
            "sPARAM": name,
            "dPOINT": theta_hat[i],
            "dSE": se[i],
            "dCI_LOWER_95": lo95[i],
            "dCI_UPPER_95": hi95[i],
            "dCI_LOWER_99": lo99[i],
            "dCI_UPPER_99": hi99[i],
            "bCI_95_EXCEEDS_BOUND": bnd_violation_95,
            "bCI_99_EXCEEDS_BOUND": bnd_violation_99,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Wald 95%% and 99%% CIs for P-MLE parameter "
                    "estimates using numerical Hessian at the MLE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--valuation-date-beg", required=True,
                   help='First valuation date, YYYYMMDD or ISO. E.g. 2025-04-02')
    p.add_argument("--valuation-date-end", required=True,
                   help='Last valuation date (inclusive).')
    p.add_argument("--systematic-ticker", default="^SPX",
                   help="Systematic ticker (must match the P-MLE cache).")
    p.add_argument("--idiosyncratic-tickers", nargs="+", default=None,
                   help="Which idiosyncratic tickers to process. "
                        "Default: every ticker returned by get_idiosyncratic_ids().")
    p.add_argument("--systematic-only", action="store_true",
                   help="Skip idiosyncratic stage.")
    p.add_argument("--idiosyncratic-only", action="store_true",
                   help="Skip systematic stage (idiosyncratic still needs "
                        "cached systematic params for its conditional Hessian).")
    p.add_argument("--lookback-days", type=int, default=252,
                   help="Return-history lookback window (business days). "
                        "Must match the run_pmle_kimyi2025.py setting used to "
                        "produce the cached point estimates.")
    p.add_argument("--base-days", type=int, default=252,
                   help="Base days in a year (for delta_t = 1 / base_days).")
    p.add_argument("--hessian-step", type=float, default=1e-4,
                   help="Finite-difference step for numerical Hessian.")
    p.add_argument("--out", default=None,
                   help="Output Parquet path. "
                        "Default: Study/Estimated Parameters PMLE/wald_ci_kimyi2025.parquet")
    return p


def _to_yyyymmdd(date_str: str) -> str:
    """Accept either YYYYMMDD or ISO; return YYYYMMDD."""
    return pd.Timestamp(date_str).strftime("%Y%m%d")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.systematic_only and args.idiosyncratic_only:
        raise SystemExit("Cannot pass both --systematic-only and --idiosyncratic-only.")

    valuation_beg = _to_yyyymmdd(args.valuation_date_beg)
    valuation_end = _to_yyyymmdd(args.valuation_date_end)
    valuation_window = pd.bdate_range(
        pd.to_datetime(valuation_beg, format="%Y%m%d"),
        pd.to_datetime(valuation_end, format="%Y%m%d"),
    )
    valuation_dates = [dt.strftime("%Y%m%d") for dt in valuation_window]

    delta_t = 1.0 / args.base_days
    systematic_id = args.systematic_ticker
    idiosyncratic_ids = args.idiosyncratic_tickers or get_idiosyncratic_ids()

    logger.info("Loading price panel for %s + %s ...", systematic_id, idiosyncratic_ids)
    price_ts = get_price_panel([systematic_id] + idiosyncratic_ids)
    return_ts = price_ts.pct_change().dropna()

    out_path = args.out or os.path.join(
        _REPO_ROOT, "Study", "Estimated Parameters PMLE", "wald_ci_kimyi2025.parquet"
    )

    all_rows: List[pd.DataFrame] = []

    # --- Systematic stage ---------------------------------------------------
    if not args.idiosyncratic_only:
        for dt in valuation_dates:
            if not pmle_params_exists(dt, systematic_id):
                logger.warning("No P-MLE cache for %s %s -- skipping.", systematic_id, dt)
                continue
            return_vec = (
                return_ts.loc[return_ts.index <= dt, systematic_id]
                .iloc[-args.lookback_days:]
                .to_numpy()
            )
            try:
                df = compute_systematic(
                    ticker=systematic_id, valuation_date=dt,
                    returns=return_vec, delta_t=delta_t,
                    hessian_step=args.hessian_step,
                )
                all_rows.append(df)
                _log_row_summary(df)
            except Exception as e:
                logger.exception("Failed for %s %s: %s", systematic_id, dt, e)

    # --- Idiosyncratic stage ------------------------------------------------
    if not args.systematic_only:
        for idi_id in idiosyncratic_ids:
            for dt in valuation_dates:
                if not pmle_params_exists(dt, idi_id):
                    logger.warning("No P-MLE cache for %s %s -- skipping.", idi_id, dt)
                    continue
                if not pmle_params_exists(dt, systematic_id):
                    logger.warning(
                        "Idiosyncratic %s %s needs cached systematic %s -- skipping.",
                        idi_id, dt, systematic_id,
                    )
                    continue
                return_vec = (
                    return_ts.loc[return_ts.index <= dt, idi_id]
                    .iloc[-args.lookback_days:]
                    .to_numpy()
                )
                try:
                    df = compute_idiosyncratic(
                        ticker=idi_id, valuation_date=dt,
                        idi_returns=return_vec,
                        systematic_ticker=systematic_id,
                        delta_t=delta_t,
                        hessian_step=args.hessian_step,
                    )
                    all_rows.append(df)
                    _log_row_summary(df)
                except Exception as e:
                    logger.exception("Failed for %s %s: %s", idi_id, dt, e)

    if not all_rows:
        logger.warning("No results produced.")
        return

    result = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), out_path)

    # Also write a companion CSV for easy human inspection.
    csv_path = out_path.replace(".parquet", ".csv")
    result.to_csv(csv_path, index=False, float_format="%.6g")
    logger.info("Also wrote CSV: %s", csv_path)


def _log_row_summary(df: pd.DataFrame) -> None:
    """One log line per parameter, showing point ± SE and 95% CI."""
    ticker = df["sTICKER"].iloc[0]
    date = df["sVALUATION_DATE"].iloc[0]
    parts = []
    for _, row in df.iterrows():
        se_str = f"{row['dSE']:.4g}" if not np.isnan(row["dSE"]) else "NA"
        parts.append(
            f"{row['sPARAM']}={row['dPOINT']:.4g}±{se_str}"
        )
    logger.info("  %s %s  %s", ticker, date, "  ".join(parts))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
