"""
report_collar_asian - end-to-end study for the paper's COIN collar / Asian
derivative portfolio.

Produces:
  - Figure 1: density_simulated_returns.pdf
  - Figure 2: density_simulated_returns.pdf (P&L comparison)
  - Figure 6: filtered_liquidity_process.pdf
  - VaR CSVs: final_output_{H}d.csv, final_output_{H}d_base.csv
  - Additional plots: daily_prices_returns_*.pdf, daily_var_comparison.pdf,
                       simulated_pl.pdf, pl_ladder.pdf

Data mode: uses committed snapshots via Library.DataAccess (no live network
access required).

Auto-converted from report_collar_asian2.ipynb, with FinanceDataReader
replaced by Library.DataAccess.get_price_panel for reproducibility.
"""

# Ensure package imports resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# --- logging ---
from Library.Logging import setup_logging
logger = setup_logging(__name__)
logger.info("Starting %s", __name__)

# --- centralised figures output directory ---
_FIGURES_DIR = _REPO_ROOT / "Figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% cell 1
logger.info("[cell %d/51] compute", 1)
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from Library.DataAccess import get_price_panel
import matplotlib.dates as mdates
import ustreasurycurve as ustcurve

from collections import namedtuple
from Library.Interpolation import pchip_interpolator2d
from Library.RiskEngineKimYi2025 import *
from Library.PayoffFactory import *
from Library.OptionPricerBSM1973 import *
from Library.ExoticEngine import *
from Library.Utility import UST_TENOR_MAP, year_frac, get_fixings_vec
from Library.StatisticsMC import StatisticsMCMean, StatisticsMCQuantile
from Library.PathDependent import PathDependentAsianDiscrete
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

az.style.use("arviz-darkgrid")

print(f"Running on PyMC v{pm.__version__}")
print(f"Running on NumPy v{np.__version__}")

# %% cell 2
logger.info("[cell %d/51] inputs", 2)
# inputs
valuation_beg_date = '20250331'
valuation_end_date = '20250417'

UND_TICKERS_DICT = {'systematic': '^SPX', 'idiosyncratic': ['COIN']}
SET_OF_TICKERS = [UND_TICKERS_DICT['systematic']] + [*UND_TICKERS_DICT['idiosyncratic']]

DIVIDEND_YIELDS = {'^SPX': 1.25, 'COIN': 0.}

origin_date = (pd.to_datetime(valuation_beg_date) - pd.Timedelta(730, "D")).strftime("%Y%m%d")
total_date_set = pd.bdate_range(origin_date, valuation_end_date)
valuation_date_array = pd.bdate_range(valuation_beg_date, pd.to_datetime(valuation_end_date))

# %% cell 3
logger.info("[cell %d/51] compute template_df", 3)
template_df = pd.DataFrame(index=total_date_set)
template_df.index.name = 'dtDATE'

# Price panel via Library.DataAccess (snapshot mode by default).
# The previous FinanceDataReader loop is replaced by a single call. Volume was
# fetched but never used downstream, so it is dropped in this migration.
_price_panel = get_price_panel(SET_OF_TICKERS)
_price_panel.index.name = 'dtDATE'
prices_df = template_df.join(_price_panel).ffill().bfill()

return_df = prices_df.sort_index().pct_change().fillna(0.)

# %% cell 4
logger.info("[cell %d/51] fig, ax = plt.subplots(nrows=2, figsize=(15, 7))", 4)
fig, ax = plt.subplots(nrows=2, figsize=(15, 7))

t1 = total_date_set.max() - pd.DateOffset(years=1)
t2 = total_date_set.max()

sns.lineplot(data=prices_df.loc[t1:t2, SET_OF_TICKERS[0]].reset_index(), x='dtDATE', y=SET_OF_TICKERS[0], color='green', ax=ax[0])
sns.lineplot(data=return_df.loc[t1:t2, SET_OF_TICKERS[0]].multiply(100.).reset_index(), x='dtDATE', y=SET_OF_TICKERS[0], color='blue', ax=ax[1])

ax[1].set_xlabel(None)
ax[0].set_ylabel('Prices ($)', color='green')
ax[1].set_ylabel('Returns (%)', color='blue')
ax[0].set_title(f'Daily Prices', fontsize=14, fontweight='bold')
ax[1].set_title(f'Daily Returns', fontsize=14, fontweight='bold')
ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

ax[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
ax[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))

fig.suptitle(f"S&P 500 ({SET_OF_TICKERS[0]})", fontsize=16, fontweight='bold')
fig.autofmt_xdate()

plt.savefig(str(_FIGURES_DIR / "daily_prices_returns_SPX.pdf"), dpi=300)
plt.show();

# %% cell 5
logger.info("[cell %d/51] fig, ax = plt.subplots(nrows=2, figsize=(15, 7))", 5)
fig, ax = plt.subplots(nrows=2, figsize=(15, 7))

t1 = total_date_set.max() - pd.DateOffset(years=1)
t2 = total_date_set.max()

sns.lineplot(data=prices_df.loc[t1:t2, SET_OF_TICKERS[-1]].reset_index(), x='dtDATE', y=SET_OF_TICKERS[-1], color='green', ax=ax[0])
sns.lineplot(data=return_df.loc[t1:t2, SET_OF_TICKERS[-1]].multiply(100.).reset_index(), x='dtDATE', y=SET_OF_TICKERS[-1], color='blue', ax=ax[1])

ax[1].set_xlabel(None)
ax[0].set_ylabel('Prices ($)', color='green')
ax[1].set_ylabel('Returns (%)', color='blue')
ax[0].set_title(f'Daily Prices', fontsize=14, fontweight='bold')
ax[1].set_title(f'Daily Returns', fontsize=14, fontweight='bold')

ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

ax[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))

fig.suptitle(f"Coinbase Global, Inc. ({SET_OF_TICKERS[-1]})", fontsize=16, fontweight='bold')
fig.autofmt_xdate()

plt.savefig(str(_FIGURES_DIR / f"daily_prices_returns_{SET_OF_TICKERS[-1]}.pdf"), dpi=300)
plt.show();

# %% cell 6
logger.info("[cell %d/51] https://home.treasury.gov/policy-issues/financing-the-govern", 6)
# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27
rates_data_df = ustcurve.nominalRates(valuation_date_array[0].strftime("%Y-%m-%d"), valuation_date_array[-1].strftime("%Y-%m-%d")).set_index('date')
rates_data_df = pd.DataFrame(index=pd.bdate_range(rates_data_df.index.min(), rates_data_df.index.max())).join(rates_data_df).ffill(axis=1).bfill(axis=1).ffill().bfill()
rates_data_df.head()

# %% cell 7
logger.info("[cell %d/51] compute _vol_surface_path", 7)
from Library.Serialization import load_vol_surface
_vol_surface_path = _REPO_ROOT / "Study" / "Vol Surface From Model" / "kimyi2025_vol_surface"
vol_surf_df = load_vol_surface(_vol_surface_path)
logger.info("Loaded vol surface from %s(.parquet|.pkl)", _vol_surface_path)

# %% cell 8
logger.info("[cell %d/51] define generate_und_dt_key()", 8)
def generate_und_dt_key(underlying_name: str, valuation_date: str) -> str:
    return f"{underlying_name}-{valuation_date}"

# %% cell 9
logger.info("[cell %d/51] compute valuation_date", 9)
valuation_date = '20250409'
lookback_period = 252
returns_df = prices_df.loc[prices_df.index <= pd.to_datetime(valuation_date)].pct_change()
returns_df = returns_df.iloc[-lookback_period:]

underlying_name = UND_TICKERS_DICT['systematic']

returns = returns_df[underlying_name].to_numpy()
base_days = 252
delta_t = np.array(1 / base_days)


def _load_params_from_cache(underlying_name, valuation_date):
    """Return a {param: ParamsResults(dMEAN, dCI_LOWER, dCI_UPPER)} dict
    from the cached PMLE CSV under ``Study/Estimated Parameters PMLE/``.
    Matches the shape returned by ``pmle_kimyirisk_*`` so downstream code
    is unchanged. Falls back to running MCMC if the cache is missing."""
    from Library.DataAccess import get_pmle_params
    from Library.RiskEngineKimYi2025 import ParamsResults
    series = get_pmle_params(valuation_date, underlying_name)
    out = {}
    for col in series.index:
        if (
            col.startswith("d")
            and not col.startswith("dt")
            and "_CI_" not in col
        ):
            out[col] = ParamsResults(
                dMEAN=float(series[col]),
                dCI_LOWER=float(series[f"{col}_CI_LOWER"]),
                dCI_UPPER=float(series[f"{col}_CI_UPPER"]),
            )
    return out


params_dict = {}
key = generate_und_dt_key(underlying_name, valuation_date)
try:
    params_dict[key] = _load_params_from_cache(underlying_name, valuation_date)
    logger.info("Loaded %s params from cached PMLE CSV", key)
except FileNotFoundError:
    logger.warning("No cache for %s; running MCMC (requires numba)", key)
    params_dict[key] = pmle_kimyirisk_systematic(sys_returns=returns, delta_t=delta_t)

sys_params = params_dict[key]

underlying_name = UND_TICKERS_DICT["idiosyncratic"][0]
returns = returns_df[underlying_name].to_numpy()
key = generate_und_dt_key(underlying_name, valuation_date)
try:
    params_dict[key] = _load_params_from_cache(underlying_name, valuation_date)
    logger.info("Loaded %s params from cached PMLE CSV", key)
except FileNotFoundError:
    logger.warning("No cache for %s; running MCMC (requires numba)", key)
    _sys_means = {k: v.dMEAN for k, v in sys_params.items()}
    params_dict[key] = pmle_kimyirisk_idiosyncratic(
        idi_returns=returns, params_sys=_sys_means, delta_t=delta_t
    )

params_dict

# %% cell 10
logger.info("[cell %d/51] define kimyi2025_pmle_helper()", 10)
def kimyi2025_pmle_helper(risk_horizon, base_days: int = 252) -> str:

    Delta_t = np.array(risk_horizon / base_days)
    days_to_look_back = base_days

    pmle_params_sys_df = []
    pmle_params_idi_df = []

    # P-MLE for Systematic Asset
    underlying_name = SET_OF_TICKERS[0]
    PATH = str(_REPO_ROOT / 'Study' / 'Estimated Parameters PMLE' / underlying_name)
    print(PATH)
    for valuation_date in valuation_date_array:
        filename= f'estimated_params_pmle_{underlying_name}_{valuation_date.strftime("%Y%m%d")}.csv'
        print(filename)
        if filename in os.listdir(PATH):
            pmle_params_sys_df.append(pd.read_csv(os.path.join(PATH, filename)))
        else:
            valuation_date_str = valuation_date.strftime("%Y-%m-%d")
            tmp_return_df = prices_df.loc[prices_df.index <= pd.to_datetime(valuation_date_str)].pct_change(periods=risk_horizon)
            return_df = tmp_return_df.iloc[-days_to_look_back:]
            sys_returns_df = return_df[underlying_name]
            tmp_df = pmle_kimyirisk_systematic(
                valuation_date=valuation_date_str,
                sys_returns_df=sys_returns_df,
                delta_t=Delta_t
            )
            pmle_params_sys_df.append(tmp_df)
            tmp_df.to_csv(os.path.join(PATH, f'estimated_params_pmle_{underlying_name}_{valuation_date.strftime("%Y%m%d")}.csv'), index=False)

    pmle_params_sys_df = pd.concat(pmle_params_sys_df)
    pmle_params_sys_df.dtVALUATION_DATE = pd.to_datetime(pmle_params_sys_df.dtVALUATION_DATE)

    # P-MLE for Idiosyncratic Asset
    for underlying_name in SET_OF_TICKERS[1:]:
        PATH = str(_REPO_ROOT / 'Study' / 'Estimated Parameters PMLE' / underlying_name)
        print(PATH)
        for valuation_date in valuation_date_array:
            filename= f'estimated_params_pmle_{underlying_name}_{valuation_date.strftime("%Y%m%d")}.csv'
            print(filename)
            if filename in os.listdir(PATH):
                pmle_params_idi_df.append(pd.read_csv(os.path.join(PATH, filename)))
            else:
                valuation_date_str = valuation_date.strftime("%Y-%m-%d")
                tmp_return_df = prices_df.loc[prices_df.index <= pd.to_datetime(valuation_date_str)].pct_change(periods=risk_horizon)
                return_df = tmp_return_df.iloc[-days_to_look_back:]
                idi_returns_df = return_df[underlying_name]
                tmp_df = pmle_kimyirisk_idiosyncratic(
                    valuation_date=valuation_date_str,
                    idi_returns_df=idi_returns_df,
                    params_sys_df=pmle_params_sys_df.set_index('dtVALUATION_DATE').xs(valuation_date),
                    delta_t=Delta_t
                )
                pmle_params_idi_df.append(tmp_df)
                tmp_df.to_csv(os.path.join(PATH, f'estimated_params_pmle_{underlying_name}_{valuation_date.strftime("%Y%m%d")}.csv'), index=False)
    # pmle_params_idi_df = pd.concat(pmle_params_idi_df)
    # pmle_params_idi_df.dtVALUATION_DATE = pd.to_datetime(pmle_params_idi_df.dtVALUATION_DATE)
    # pmle_params_df = pd.concat([pmle_params_sys_df, pmle_params_idi_df]).set_index(['dtVALUATION_DATE', 'sUNDERLYING_NAME'])
    return f"Completed successfully P-MLE exercise given {risk_horizon}-days risk horizon."

def get_pmle_estimated_params() -> pd.DataFrame:
    df = []
    for underlying_name in SET_OF_TICKERS:
        PATH = str(_REPO_ROOT / 'Study' / 'Estimated Parameters PMLE' / underlying_name)
        for file_name in os.listdir(PATH):
            print(file_name)
            file_name = os.path.join(PATH, file_name)
            tmp_df = pd.read_csv(file_name)
            df.append(tmp_df)
    df = pd.concat(df)
    df.dtVALUATION_DATE = pd.to_datetime(df.dtVALUATION_DATE)
    return df.set_index(['dtVALUATION_DATE', 'sUNDERLYING_NAME'])

# %% cell 11
logger.info("[cell %d/51] compute risk_horizons", 11)
risk_horizons = [1]

if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("fork", force=True)

    with ProcessPoolExecutor() as executor:

        results = executor.map(kimyi2025_pmle_helper, risk_horizons)

        for i, result in enumerate(results):
            print(f"{i}: {result}")

# %% cell 12
logger.info("[cell %d/51] Materialise the panel of cached PMLE parameter estimates fro", 12)
# Materialise the panel of cached PMLE parameter estimates from the
# per-date CSVs. This dataframe is what cells 12+ consume; the previous
# notebook produced it as a side effect of cell 10 but the .py conversion
# left the assignment out.
pmle_params_df = get_pmle_estimated_params()
# Defensively drop any duplicate (date, underlying) rows: the cache is
# occasionally re-generated, and duplicates make xs() return a 2-row
# DataFrame instead of a Series, which breaks downstream Monte Carlo
# broadcast (shape mismatch in Random.get_poisson).
pmle_params_df = pmle_params_df[~pmle_params_df.index.duplicated(keep="last")]
logger.info(
    "pmle_params_df loaded: %d unique (date, underlying) rows",
    len(pmle_params_df),
)
mask = pmle_params_df.index.get_level_values("sUNDERLYING_NAME")!=UND_TICKERS_DICT['systematic']
pmle_params_tmp_df = pmle_params_df[mask]
rhoix  = (pmle_params_tmp_df.dSIGMA * pmle_params_tmp_df.dBETAI)**2
rhoix += 2 * pmle_params_tmp_df.dSIGMA * pmle_params_tmp_df.dBETAI * pmle_params_tmp_df.dKAPPAI * pmle_params_tmp_df.dRHOIX
rhoix += pmle_params_tmp_df.dKAPPAI**2
rhoix **=(1/2)
rhoix
# rhoix.loc[rhoix.index.get_level_values('sUNDERLYING_NAME')!=UND_TICKERS_DICT['systematic']]

# %% cell 13
logger.info("[cell %d/51] define simulate_shock_returns()", 13)
def simulate_shock_returns(params: pd.Series, rng: RandomBase, risk_horizon: int, size: Tuple[int, int, int], base_days: int = 252) -> NDArray[np.float64]:

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI_CI_LOWER)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=np.array(risk_horizon / base_days)
    ).random(rng=rng, size=size)

    return ret[0]

def simulate_shock_returns_base(params: pd.Series, rng: RandomBase, risk_horizon: int, size: Tuple[int, int, int], base_days: int = 252) -> NDArray[np.float64]:

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(np.array(.0))],
        betai=[ParametersConstant(np.array(.0))],
        rhoix=[ParametersConstant(np.array(.0))],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=np.array(risk_horizon / base_days)
    ).random(rng=rng, size=size)

    return ret[0]

def est_liquidity_process(params: pd.Series, observed_data: NDArray[np.float64], risk_horizon: int, base_days: int = 252) -> NDArray[np.float64]:

    Psi = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=np.array(risk_horizon / base_days)
    ).est_liquidity_process(observed_data)

    return Psi

# %% cell 14
logger.info("[cell %d/51] compute n_paths", 14)
n_paths = 10_000
m_steps = 20
h = 1
seed_number = 20251231

def simulate_shock_returns_helper(valuation_date: str) -> dict:

    output = {}
    for underlying_name in SET_OF_TICKERS:
        output[underlying_name] = simulate_shock_returns(
            pmle_params_df.xs(pd.to_datetime(valuation_date, format="%Y%m%d")).xs(underlying_name), RandomMT19937(np.uint64(seed_number)), h, (1, n_paths, m_steps)
        )

    return {valuation_date: output}


def simulate_shock_returns_base_helper(valuation_date: str) -> dict:

    output = {}
    for underlying_name in SET_OF_TICKERS:
        output[underlying_name] = simulate_shock_returns_base(
            pmle_params_df.xs(pd.to_datetime(valuation_date, format="%Y%m%d")).xs(underlying_name), RandomMT19937(np.uint64(seed_number)), h, (1, n_paths, m_steps)
        )

    return {valuation_date: output}


def est_liquidity_process_helper(valuation_date: str) -> dict:
    tmp_return_df = prices_df.loc[prices_df.index <= pd.to_datetime(valuation_date)].pct_change(periods=h)
    tmp_return = tmp_return_df.iloc[-252:]

    output = {}
    for underlying_name in SET_OF_TICKERS:
        output[underlying_name] = est_liquidity_process(
            pmle_params_df.xs(pd.to_datetime(valuation_date, format="%Y%m%d")).xs(underlying_name), tmp_return[underlying_name].to_numpy()[1:].reshape((-1, 1)), h
        )

    return {valuation_date: output}

# %% cell 15
logger.info("[cell %d/51] multiprocessing.set_start_method('fork', force=True)", 15)
if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("fork", force=True)

    valuation_dates = [date.strftime("%Y%m%d") for date in valuation_date_array]

    with ProcessPoolExecutor() as executor:

        results1 = executor.map(simulate_shock_returns_helper, valuation_dates)
        results2 = executor.map(simulate_shock_returns_base_helper, valuation_dates)
        results3 = executor.map(est_liquidity_process_helper, valuation_dates)

        shock_returns_dict = {}
        shock_returns_base_dict = {}
        liquidity_process_dict = {}

        for result1 in results1:
            for date_key, inner_dict in result1.items():
                shock_returns_dict[date_key] = inner_dict

        for result2 in results2:
            for date_key, inner_dict in result2.items():
                shock_returns_base_dict[date_key] = inner_dict

        for result3 in results3:
            for date_key, inner_dict in result3.items():
                liquidity_process_dict[date_key] = inner_dict

# %% cell 16
logger.info("[cell %d/51] compute df1", 16)
df1 = pd.DataFrame(data=shock_returns_dict['20250409']['COIN'][:, 1] * 100., columns=['dSIMULATED_RETURNS'])
df2 = pd.DataFrame(data=shock_returns_base_dict['20250409']['COIN'][:, 1] * 100., columns=['dSIMULATED_RETURNS'])
df1['Return Assumption'] = 'LA'
df2['Return Assumption'] = 'BS'
simulated_returns_df = pd.concat([df1, df2])
simulated_returns_df = simulated_returns_df.set_index('Return Assumption')

# %% cell 17
logger.info("[cell %d/51] fig, ax = plt.subplots(figsize=(15, 7))", 17)
fig, ax = plt.subplots(figsize=(15, 7))

# Paper Figure 1: solid lines with small periodic markers.
# BS -> orange with star markers ('*'), LA -> blue with circle markers ('o').
# bw_adjust=5 matches the paper Notes: "smoothing parameter of five".
# common_norm=False: each hue category integrates to 1 on its own.
# (Default common_norm=True rescales each category by its share of the
# joint dataset -- with equal BS/LA sample counts that halves both peaks.)
sns.kdeplot(data=simulated_returns_df, x='dSIMULATED_RETURNS', hue='Return Assumption',
            fill=False, gridsize=500, bw_adjust=5., common_norm=False, ax=ax)
# Overlay markers on each KDE line so the plot matches the paper style.
# seaborn kdeplot draws lines in reverse hue order: [0] = BS (orange), [1] = LA (blue).
# Labels on the line objects are '_child0'/'_child1', so match by index instead.
_kde_lines = ax.get_lines()
_marker_specs = [('*', 5), ('o', 4)]  # (BS star, LA circle)
for _line, (_m, _size) in zip(_kde_lines, _marker_specs):
    _line.set_marker(_m)
    _line.set_markersize(_size)
    _line.set_markevery(8)
    _line.set_markeredgewidth(0)
# Diagnostic: log the empirical std of the plotted shocks AND the underlying
# COIN parameters. Direct calculation gives BS sigma = kappa*sqrt(1/252),
# so for COIN with kappa ~ 0.62 the expected BS sigma is ~3.9% and the LA
# sigma is ~6.0%, which produces KDE peaks ~0.079 / ~0.052 (matching the
# paper's Figure 1). If the logged std differs from this, the simulation
# is not producing the expected shocks and something upstream (parameters,
# end_dt, or the KimYiRiskEngine loop) needs to be inspected.
_coin_params = pmle_params_df.xs(pd.to_datetime('20250409', format='%Y%m%d')).xs('COIN')
logger.info(
    "Figure 1 diagnostic -- COIN kappa=%.4f gamma=%.4f beta=%.4f rho=%.4f sigma=%.4f",
    float(_coin_params.dKAPPAI), float(_coin_params.dGAMMAI),
    float(_coin_params.dBETAI), float(_coin_params.dRHOIX), float(_coin_params.dSIGMA),
)
logger.info(
    "Figure 1 diagnostic -- expected BS sigma = kappa/sqrt(252) = %.3f%%",
    float(_coin_params.dKAPPAI) / np.sqrt(252) * 100,
)
logger.info(
    "Figure 1 diagnostic -- observed BS sigma=%.3f%%  LA sigma=%.3f%%  (ratio LA/BS=%.2f)",
    float(df2['dSIMULATED_RETURNS'].std()),
    float(df1['dSIMULATED_RETURNS'].std()),
    float(df1['dSIMULATED_RETURNS'].std()) / max(float(df2['dSIMULATED_RETURNS'].std()), 1e-9),
)

# Match paper Figure 1 tick spacing: x every 2%, y every 0.02.
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
# Force y-axis to extend to 0.08 so the tick set matches the paper
# (0.00, 0.02, 0.04, 0.06, 0.08) regardless of the observed peak height.
ax.set_ylim(0, 0.09)
ax.tick_params(which='both', length=3, width=0.8, direction='out')
ax.tick_params(axis='x', which='both', labelrotation=45)

ax.set_title(f'Density Comparison of Liquidity Adjusted (LA) and Baseline (BS) {h}-Day Simulated Returns', fontweight='bold')
ax.set_xlabel(f"Simulated {h}-Day Price Shocks (%)")

plt.setp(ax.get_legend().get_title(), fontsize='14')
plt.xlim(-30, 30)   # matches paper Figure 1 caption: "-30% to +30%"
plt.savefig(str(_FIGURES_DIR / "Figure_1_density_simulated_returns.pdf"), dpi=300)
plt.show()

# %% cell 18
logger.info("[cell %d/51] liquidity process for last date", 18)
# liquidity process for last date
valuation_date_str = '20250409'

index = return_df.loc[return_df.index <= pd.to_datetime(valuation_date_str)].iloc[-252:].index

liquidity_process_df = pd.concat(
    [
        pd.Series(liquidity_process_dict[valuation_date_str][SET_OF_TICKERS[0]].squeeze(), index=index, name=SET_OF_TICKERS[0]),
        pd.Series(liquidity_process_dict[valuation_date_str][SET_OF_TICKERS[1]].squeeze(), index=index, name=SET_OF_TICKERS[1])
    ],
    axis=1
)

fig, ax = plt.subplots(figsize=(15, 7))

sns.lineplot(x=liquidity_process_df.index, y=liquidity_process_df[SET_OF_TICKERS[0]].squeeze(), ax=ax, label=rf'{SET_OF_TICKERS[0].split("^")[1]}', color='black', linestyle='-.', linewidth=2)
sns.lineplot(x=liquidity_process_df.index, y=liquidity_process_df[SET_OF_TICKERS[1]].squeeze(), ax=ax, label=rf'{SET_OF_TICKERS[1]}', color='blue', linestyle=':', linewidth=2)
sns.lineplot(x=liquidity_process_df.index, y=0, ax=ax, color='red', linestyle='--', linewidth=1.5, label='BS')

ax.set_xlabel(None)
ax.set_ylabel(None)
# ax.legend(title='Underlying Tickers', title_fontproperties={'weight': 'bold', 'size': 'large'})
ax.set_title(r'Filtered Liquidity Process, $\mathbf{{\hat{\Psi}_{{i, t}}}}$', fontweight='bold')
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=-1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b %d, %y'))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))

ax.tick_params(axis='x', which='both', labelrotation=45)

ax.annotate(
    text="BoJ increased rate by 0.25% \nunexpectedly on Jul 31, 2024.",
    xy=(pd.to_datetime('20240731'), liquidity_process_df.xs('20240731')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20240815'), liquidity_process_df.xs('20240731')[SET_OF_TICKERS[1]] - .05),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="There are rumors of the yen carry trade unwinding. \nSPX sells off by 3% on Aug 5, 2024.",
    xy=(pd.to_datetime('20240805'), liquidity_process_df.xs('20240805')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20240430'), liquidity_process_df.xs('20240805')[SET_OF_TICKERS[1]] - .1),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="SPX sells off 1.72% from \nweak job report on Sep 6, 2024.",
    xy=(pd.to_datetime('20240906'), liquidity_process_df.xs('20240906')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20240701'), liquidity_process_df.xs('20240906')[SET_OF_TICKERS[1]] + .05),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="Trump wins the 2024 presidential election \non Nov 5, 2024. SPX rallies by 1.23%.",
    xy=(pd.to_datetime('20241105'), liquidity_process_df.xs('20241105')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20241015'), liquidity_process_df.xs('20241105')[SET_OF_TICKERS[1]] - .15),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="Republicans projected to win US Senate majority \non Nov 6, 2024. SPX rallies by 2.53%.",
    xy=(pd.to_datetime('20241106'), liquidity_process_df.xs('20241106')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20240731'), liquidity_process_df.xs('20241106')[SET_OF_TICKERS[1]] + .1),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="The FED cuts rates by 0.25% but signals less cuts in 2025 \non Dec 18, 2024. SPX sells off by 2.95%.",
    xy=(pd.to_datetime('20241218'), liquidity_process_df.xs('20241218')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20241231'), liquidity_process_df.xs('20241218')[SET_OF_TICKERS[1]] + .1),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="SPX sells off by 2.7% from tariff tweets \nand talk of recession on Mar 10, 2025.",
    xy=(pd.to_datetime('20250310'), liquidity_process_df.xs('20250310')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20241215'), liquidity_process_df.xs('20250310')[SET_OF_TICKERS[1]]),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="Trump officially announces \ntariffs on Apr 2, 2025. \nSPX sells off by 4.84% next day.",
    xy=(pd.to_datetime('20250402'), liquidity_process_df.xs('20250402')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20250115'), liquidity_process_df.xs('20250402')[SET_OF_TICKERS[1]] - .15),
    arrowprops=dict(facecolor='grey')
)

ax.annotate(
    text="Trump announces 90-day tariff pause \non Apr 9, 2025. SPX rallies by 9.52%.",
    xy=(pd.to_datetime('20250409'), liquidity_process_df.xs('20250409')[SET_OF_TICKERS[1]]),
    xytext=(pd.to_datetime('20250115'), liquidity_process_df.xs('20250409')[SET_OF_TICKERS[1]] - .25),
    arrowprops=dict(facecolor='grey')
)

# plt.tight_layout()
plt.xlim(index[0], index[-1])
plt.savefig(str(_FIGURES_DIR / "Figure_6_filtered_liquidity_process.pdf"), dpi=300)
plt.show();

# %% cell 19
logger.info("[cell %d/51] compute portfolio_pos_names_list", 19)
portfolio_pos_names_list = [
    'sVALUATION_DATE',
    'sUNDERLYING',
    'sSTRATEGY',
    'sTYPOLOGY',
    'sPAYOFF',
    'sEXPIRY_DATE',
    'dSTRIKE_PRICE',
    'iQUANTITY',
    'sFIXING_DATE_BEG',
    'sFIXING_DATE_END',
    'iBASE_DAYS',
    'dSPOT_PRICE',
    'dVOL_SURFACE',
    'dDIVIDEND_YIELD',
    'oRISK_FREE_RATE_TERM'
]

PortfolioPos = namedtuple('PortfolioPos', portfolio_pos_names_list)

iBASE_DAYS = np.int64(252)

portfolio_pos_dict = {}

valuation_start_date = '20250331'

for valuation_date in valuation_date_array:

    valuation_date_str = valuation_date.strftime("%Y%m%d")

    portfolio_inner_dict = {}

    for underlying_name in SET_OF_TICKERS[1:]:

        spot_price = prices_df.xs(valuation_date_str)[underlying_name]
        id = f"{underlying_name}-{valuation_date_str}"
        vol_surface = pd.DataFrame(data=vol_surf_df[id]['dVOL'], index=vol_surf_df[id]['iEXPIRY'], columns=vol_surf_df[id]['dMONEYNESS'] * 100.)

        rates_data = rates_data_df.xs(valuation_date)
        tenors = np.array([UST_TENOR_MAP[tenor] for tenor in rates_data.index]).squeeze()
        rates = rates_data.to_numpy().squeeze() / 100.
        curve_fit, _ = calibrate_nss_ols(tenors, rates)

        # product detail
        fixing_date_beg_str = (valuation_date_array[0] + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fixing_date_end_str = valuation_date_array[-1]
        expiry_date_str = (valuation_date_array[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")

        dividend_yield = np.array(DIVIDEND_YIELDS[underlying_name] / 100.)

        portfolio_inner_dict[underlying_name] = [
            PortfolioPos(
                sVALUATION_DATE=valuation_date_str, sUNDERLYING=underlying_name, sSTRATEGY='collar', sTYPOLOGY='asian discrete', sPAYOFF='put',
                sEXPIRY_DATE=expiry_date_str, dSTRIKE_PRICE=np.round(prices_df.xs(valuation_start_date)[underlying_name], 4), iQUANTITY=np.int64(-100), sFIXING_DATE_BEG=fixing_date_beg_str, sFIXING_DATE_END=fixing_date_end_str, iBASE_DAYS=iBASE_DAYS,
                dSPOT_PRICE=spot_price, dVOL_SURFACE=vol_surface, dDIVIDEND_YIELD=dividend_yield, oRISK_FREE_RATE_TERM=curve_fit
            ),
            PortfolioPos(
                sVALUATION_DATE=valuation_date_str, sUNDERLYING=underlying_name, sSTRATEGY='collar', sTYPOLOGY='asian discrete', sPAYOFF='call',
                sEXPIRY_DATE=expiry_date_str, dSTRIKE_PRICE=np.round(prices_df.xs(valuation_start_date)[underlying_name] * 1.15, 4), iQUANTITY=np.int64(100), sFIXING_DATE_BEG=fixing_date_beg_str, sFIXING_DATE_END=fixing_date_end_str, iBASE_DAYS=iBASE_DAYS,
                dSPOT_PRICE=spot_price, dVOL_SURFACE=vol_surface, dDIVIDEND_YIELD=dividend_yield, oRISK_FREE_RATE_TERM=curve_fit
            ),
            PortfolioPos(
                sVALUATION_DATE=valuation_date_str, sUNDERLYING=underlying_name, sSTRATEGY='hedge', sTYPOLOGY='vanilla', sPAYOFF='put',
                sEXPIRY_DATE=expiry_date_str, dSTRIKE_PRICE=np.round(prices_df.xs(valuation_start_date)[underlying_name] * 1.05, 4), iQUANTITY=np.int64(85), sFIXING_DATE_BEG=None, sFIXING_DATE_END=None, iBASE_DAYS=iBASE_DAYS,
                dSPOT_PRICE=spot_price, dVOL_SURFACE=vol_surface, dDIVIDEND_YIELD=dividend_yield, oRISK_FREE_RATE_TERM=curve_fit
            ),
            PortfolioPos(
                sVALUATION_DATE=valuation_date_str, sUNDERLYING=underlying_name, sSTRATEGY='hedge', sTYPOLOGY='vanilla', sPAYOFF='call',
                sEXPIRY_DATE=expiry_date_str, dSTRIKE_PRICE=np.round(prices_df.xs(valuation_start_date)[underlying_name] * 1.2, 4), iQUANTITY=np.int64(-120), sFIXING_DATE_BEG=None, sFIXING_DATE_END=None, iBASE_DAYS=iBASE_DAYS,
                dSPOT_PRICE=spot_price, dVOL_SURFACE=vol_surface, dDIVIDEND_YIELD=dividend_yield, oRISK_FREE_RATE_TERM=curve_fit
            )
        ]

    portfolio_pos_dict[valuation_date_str] = portfolio_inner_dict

# %% cell 20
logger.info("[cell %d/51] compute portfolio_risk_names_list", 20)
portfolio_risk_names_list = [
    'sVALUATION_DATE',
    'sUNDERLYING',
    'iRISK_HORIZON_DAYS',
    'dSHOCKS'
]

PortfolioRisk = namedtuple("PortfolioRisk", portfolio_risk_names_list)

iRISK_HORIZON_DAYS = np.int64(1)

portfolio_risk_dict = {}

for valuation_date in valuation_date_array:

    valuation_date_str = valuation_date.strftime("%Y%m%d")

    portfolio_inner_dict = {}

    for underlying_name in SET_OF_TICKERS[1:]:

        portfolio_inner_dict[underlying_name] = PortfolioRisk(
            sVALUATION_DATE=valuation_date_str,
            sUNDERLYING=underlying_name,
            iRISK_HORIZON_DAYS=iRISK_HORIZON_DAYS,
            dSHOCKS=shock_returns_dict[valuation_date_str][underlying_name]
        )

    portfolio_risk_dict[valuation_date_str] = portfolio_inner_dict

# %% cell 21
logger.info("[cell %d/51] compute portfolio_risk_base_dict", 21)
portfolio_risk_base_dict = {}

for valuation_date in valuation_date_array:

    valuation_date_str = valuation_date.strftime("%Y%m%d")

    portfolio_inner_dict = {}

    for underlying_name in SET_OF_TICKERS[1:]:

        portfolio_inner_dict[underlying_name] = PortfolioRisk(
            sVALUATION_DATE=valuation_date_str,
            sUNDERLYING=underlying_name,
            iRISK_HORIZON_DAYS=iRISK_HORIZON_DAYS,
            dSHOCKS=shock_returns_base_dict[valuation_date_str][underlying_name]
        )

    portfolio_risk_base_dict[valuation_date_str] = portfolio_inner_dict

# %% cell 22
logger.info("[cell %d/51] compute portfolio_risk_up1pct_dict", 22)
portfolio_risk_up1pct_dict = {}

for valuation_date in valuation_date_array:

    valuation_date_str = valuation_date.strftime("%Y%m%d")

    portfolio_inner_dict = {}

    for underlying_name in SET_OF_TICKERS[1:]:

        portfolio_inner_dict[underlying_name] = PortfolioRisk(
            sVALUATION_DATE=valuation_date_str,
            sUNDERLYING=underlying_name,
            iRISK_HORIZON_DAYS=np.int64(0),
            dSHOCKS=np.array(.01)
        )

    portfolio_risk_up1pct_dict[valuation_date_str] = portfolio_inner_dict

# %% cell 23
logger.info("[cell %d/51] compute portfolio_risk_dn1pct_dict", 23)
portfolio_risk_dn1pct_dict = {}

for valuation_date in valuation_date_array:

    valuation_date_str = valuation_date.strftime("%Y%m%d")

    portfolio_inner_dict = {}

    for underlying_name in SET_OF_TICKERS[1:]:

        portfolio_inner_dict[underlying_name] = PortfolioRisk(
            sVALUATION_DATE=valuation_date_str,
            sUNDERLYING=underlying_name,
            iRISK_HORIZON_DAYS=np.int64(0),
            dSHOCKS=np.array(.01) * -1.
        )

    portfolio_risk_dn1pct_dict[valuation_date_str] = portfolio_inner_dict

# %% cell 24
logger.info("[cell %d/51] define get_pv()", 24)
def get_pv(portfolio1: namedtuple, portfolio2: namedtuple = None) -> dict:

    sTYPOLOGY = portfolio1.sTYPOLOGY

    if sTYPOLOGY.lower() == 'vanilla':

        return _get_pv_vanilla(portfolio1, portfolio2)

    else:

        return _get_pv_exotic(portfolio1, portfolio2)


def _get_pv_vanilla(portfolio1: namedtuple, portfolio2: namedtuple = None) -> dict:

    iEXPIRY = np.int64((pd.to_datetime(portfolio1.sEXPIRY_DATE) - pd.to_datetime(portfolio1.sVALUATION_DATE)).days)
    dEXPIRY = year_frac(portfolio1.iBASE_DAYS)(np.array(0.), iEXPIRY)

    if not portfolio2 is None:
        iRISK_HORIZON_DAYS = portfolio2.iRISK_HORIZON_DAYS
        dSPOT_SHIFT = portfolio2.dSHOCKS
        iSCENARIO_NUM = range(1)

        if dSPOT_SHIFT.size > 1:
            iRISK_HORIZON_DAYS_ADJ = np.int64(np.maximum(np.minimum(iEXPIRY, iRISK_HORIZON_DAYS), 1))
            dSPOT_SHIFT = dSPOT_SHIFT[:, :iRISK_HORIZON_DAYS_ADJ + 1].sum(axis=1)
            iSCENARIO_NUM = range(dSPOT_SHIFT.shape[0])

    else:
        iRISK_HORIZON_DAYS = np.int64(0)
        dSPOT_SHIFT = np.array(0.)
        iSCENARIO_NUM = range(1)

    sTYPOLOGY = portfolio1.sTYPOLOGY
    sSTRATEGY = portfolio1.sSTRATEGY
    sPAYOFF = portfolio1.sPAYOFF

    iQUANTITY = portfolio1.iQUANTITY

    dSPOT_PRICE = portfolio1.dSPOT_PRICE * (1. + dSPOT_SHIFT)
    dSTRIKE_PRICE = portfolio1.dSTRIKE_PRICE
    dVOL_SURFACE = portfolio1.dVOL_SURFACE

    x = dVOL_SURFACE.columns
    y = dVOL_SURFACE.index
    z = dVOL_SURFACE.to_numpy() * .01

    dMONEYNESS = np.array(dSTRIKE_PRICE / dSPOT_PRICE * 100.)
    dRISK_FREE_RATE = portfolio1.oRISK_FREE_RATE_TERM(dEXPIRY)
    dDIVIDEND_YIELD = portfolio1.dDIVIDEND_YIELD
    dIMP_VOLATILITY = np.array(pchip_interpolator2d(x=x, y=y, z=z.T, x1=dMONEYNESS, x2=np.array(iEXPIRY)).squeeze())

    if sPAYOFF.lower() == 'put':

        pricer_obj = BlackScholesMertonPut(
            und_price=dSPOT_PRICE,
            und_strike=dSTRIKE_PRICE,
            risk_free_rate=dRISK_FREE_RATE,
            dividend_yield=dDIVIDEND_YIELD,
            time_to_expiry=dEXPIRY
        )

    else:

        pricer_obj = BlackScholesMertonCall(
            und_price=dSPOT_PRICE,
            und_strike=dSTRIKE_PRICE,
            risk_free_rate=dRISK_FREE_RATE,
            dividend_yield=dDIVIDEND_YIELD,
            time_to_expiry=dEXPIRY
        )

    pv = pricer_obj.price(dIMP_VOLATILITY) * iQUANTITY

    output = {
        'sVALUATION_DATE': portfolio1.sVALUATION_DATE,
        'sUNDERLYING': portfolio1.sUNDERLYING,
        'sTYPOLOGY': sTYPOLOGY,
        'sSTRATEGY': sSTRATEGY,
        'sPAYOFF': sPAYOFF,
        'iRISK_HORIZON_DAYS': iRISK_HORIZON_DAYS,
        'sEXPIRY_DATE': portfolio1.sEXPIRY_DATE,
        'iQUANTITY': iQUANTITY,
        'dSPOT_PRICE': portfolio1.dSPOT_PRICE,
        'dSPOT_SHIFT': dSPOT_SHIFT,
        'dSTRIKE_PRICE': dSTRIKE_PRICE,
        'dMONEYNESS':  dMONEYNESS,
        'dIMP_VOLATILITY': np.array(dIMP_VOLATILITY).reshape((-1,)) * 100.,
        'iSCENARIO_NUM': iSCENARIO_NUM,
        'dPV': pv.reshape((-1,))
    }

    return output


def _get_pv_exotic(portfolio1: namedtuple, portfolio2: namedtuple = None) -> dict:

    iEXPIRY = np.int64((pd.to_datetime(portfolio1.sEXPIRY_DATE) - pd.to_datetime(portfolio1.sVALUATION_DATE)).days)
    dEXPIRY = year_frac(portfolio1.iBASE_DAYS)(np.array(0.), iEXPIRY)

    if not portfolio2 is None:
        iRISK_HORIZON_DAYS = portfolio2.iRISK_HORIZON_DAYS
        dSPOT_SHIFT = portfolio2.dSHOCKS
        iSCENARIO_NUM = range(1)

        if dSPOT_SHIFT.size > 1:
            iRISK_HORIZON_DAYS_ADJ = np.int64(np.maximum(np.minimum(iEXPIRY, iRISK_HORIZON_DAYS), 1))
            dSPOT_SHIFT = dSPOT_SHIFT[:, :iRISK_HORIZON_DAYS_ADJ + 1].sum(axis=1)
            iSCENARIO_NUM = range(dSPOT_SHIFT.shape[0])

    else:
        iRISK_HORIZON_DAYS = np.int64(0)
        dSPOT_SHIFT = np.array(0.)
        iSCENARIO_NUM = range(1)

    sTYPOLOGY = portfolio1.sTYPOLOGY
    sSTRATEGY = portfolio1.sSTRATEGY
    sPAYOFF = portfolio1.sPAYOFF

    iQUANTITY = portfolio1.iQUANTITY

    dSPOT_PRICE = portfolio1.dSPOT_PRICE * (1. + dSPOT_SHIFT)
    dSTRIKE_PRICE = portfolio1.dSTRIKE_PRICE
    dVOL_SURFACE = portfolio1.dVOL_SURFACE

    x = dVOL_SURFACE.columns
    y = dVOL_SURFACE.index
    z = dVOL_SURFACE.to_numpy() * .01

    dMONEYNESS = np.array(dSTRIKE_PRICE / dSPOT_PRICE * 100.)
    dRISK_FREE_RATE = portfolio1.oRISK_FREE_RATE_TERM(dEXPIRY)
    dDIVIDEND_YIELD = portfolio1.dDIVIDEND_YIELD
    dIMP_VOLATILITY = np.array(pchip_interpolator2d(x=x, y=y, z=z.T, x1=dMONEYNESS, x2=np.array(iEXPIRY)).squeeze())

    oSTATS_GATHERER = StatisticsMCMean()
    oRAND_GENERATOR = RandomMT19937(np.uint64(20251231))

    oPRODUCT = None

    if sTYPOLOGY.lower() == 'asian discrete':

        dFIXING_TIMES = pd.date_range(portfolio1.sFIXING_DATE_BEG, portfolio1.sFIXING_DATE_END)
        dFIXING_TIMES = dFIXING_TIMES[dFIXING_TIMES >= pd.to_datetime(portfolio1.sVALUATION_DATE)]
        dFIXING_TIMES = get_fixings_vec(dFIXING_TIMES.shape[0], dEXPIRY)

        oPAYOFF = payoff_mc_factory(sPAYOFF.lower())(dSTRIKE_PRICE)
        oPRODUCT = PathDependentAsianDiscrete(
            fixing_times=dFIXING_TIMES,
            delivery_time=dEXPIRY,
            the_payoff=oPAYOFF,
            quantity_amount=iQUANTITY
        )

    if portfolio2 is None:

        ExoticEngineBlackScholesMerton(
            the_product=oPRODUCT,
            risk_free_rate=ParametersConstant(dRISK_FREE_RATE),
            dividend_yield=[ParametersConstant(dDIVIDEND_YIELD)],
            imp_volatility=[ParametersConstant(dIMP_VOLATILITY)],
            rand_generator=oRAND_GENERATOR,
            spot_price=dSPOT_PRICE
        ).do_simulation(oSTATS_GATHERER)

        pv = oSTATS_GATHERER.get_result_so_far().reshape((-1,))

        oRAND_GENERATOR.reset()

    else:

        try:
            _ = iter(dSPOT_PRICE)
        except TypeError as e:
            # print(f"{dSPOT_PRICE} is not iterable... will cast to list")
            dSPOT_PRICE = [dSPOT_PRICE]

        try:
            _ = iter(dIMP_VOLATILITY)
        except TypeError as e:
            # print(f"{dIMP_VOLATILITY} is not iterable... will cast to list")
            dIMP_VOLATILITY = [dIMP_VOLATILITY]

        pv = []

        for spot_price, imp_volatility in zip(dSPOT_PRICE, dIMP_VOLATILITY):

            ExoticEngineBlackScholesMerton(
                the_product=oPRODUCT,
                risk_free_rate=ParametersConstant(dRISK_FREE_RATE),
                dividend_yield=[ParametersConstant(dDIVIDEND_YIELD)],
                imp_volatility=[ParametersConstant(imp_volatility)],
                rand_generator=oRAND_GENERATOR,
                spot_price=spot_price
            ).do_simulation(oSTATS_GATHERER)

            pv.append(oSTATS_GATHERER.get_result_so_far()[0, 0])

            oRAND_GENERATOR.reset()

    output = {
        'sVALUATION_DATE': portfolio1.sVALUATION_DATE,
        'sUNDERLYING': portfolio1.sUNDERLYING,
        'sTYPOLOGY': sTYPOLOGY,
        'sSTRATEGY': sSTRATEGY,
        'sPAYOFF': sPAYOFF,
        'iRISK_HORIZON_DAYS': iRISK_HORIZON_DAYS,
        'sEXPIRY_DATE': portfolio1.sEXPIRY_DATE,
        'iQUANTITY': iQUANTITY,
        'dSPOT_PRICE': portfolio1.dSPOT_PRICE,
        'dSPOT_SHIFT': dSPOT_SHIFT,
        'dSTRIKE_PRICE': dSTRIKE_PRICE,
        'dMONEYNESS':  dMONEYNESS,
        'dIMP_VOLATILITY': np.array(dIMP_VOLATILITY).reshape((-1,)) * 100.,
        'iSCENARIO_NUM': iSCENARIO_NUM,
        'dPV': pv
    }

    return output

# %% cell 25
logger.info("[cell %d/51] compute portfolio_list", 25)
portfolio_list = []
portfolio_up_list = []
portfolio_dn_list = []
portfolio_risk_list = []
portfolio_risk_base_list = []

for valuation_date in valuation_date_array:
    valuation_date_str = valuation_date.strftime("%Y%m%d")
    for underlying_name in SET_OF_TICKERS[1:]:
        for portfolio in portfolio_pos_dict[valuation_date_str][underlying_name]:
            portfolio_list.append(portfolio)
            portfolio_up_list.append(portfolio_risk_up1pct_dict[valuation_date_str][underlying_name])
            portfolio_dn_list.append(portfolio_risk_dn1pct_dict[valuation_date_str][underlying_name])
            portfolio_risk_list.append(portfolio_risk_dict[valuation_date_str][underlying_name])
            portfolio_risk_base_list.append(portfolio_risk_base_dict[valuation_date_str][underlying_name])

# %% cell 26
logger.info("[cell %d/51] compute portfolio_tuple", 26)
portfolio_tuple = portfolio_list[2]
portfolio_tuple._fields

# %% cell 27
logger.info("[cell %d/51] compute", 27)
from Library.PayoffFactory import payoff_mc_factory

# %% cell 28
logger.info("[cell %d/51] define get_vanilla_pv()", 28)
def get_vanilla_pv(
        payoff_type: str,
        imp_volatility: NDArray[np.float64],
        spot_price: NDArray[np.float64],
        strike_price: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    if payoff_type.lower() == 'put':

        pricer_obj = BlackScholesMertonPut(
            und_price=spot_price,
            und_strike=strike_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            time_to_expiry=expiry
        )

    else:

        pricer_obj = BlackScholesMertonCall(
            und_price=spot_price,
            und_strike=strike_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            time_to_expiry=expiry
        )

    return pricer_obj.price(imp_volatility)

# %% cell 29
logger.info("[cell %d/51] compute sVALUATION_DATE", 29)
sVALUATION_DATE = portfolio_tuple.sVALUATION_DATE
sUNDERLYING = portfolio_tuple.sUNDERLYING
sEXPIRY_DATE = portfolio_tuple.sEXPIRY_DATE
iBASE_DAYS = portfolio_tuple.iBASE_DAYS
sPAYOFF = portfolio_tuple.sPAYOFF.lower()
dSTRIKE_PRICE = portfolio_tuple.dSTRIKE_PRICE
dSPOT_PRICE = portfolio_tuple.dSPOT_PRICE
dDIVIDEND_YIELD = portfolio_tuple.dDIVIDEND_YIELD
iQUANTITY = portfolio_tuple.iQUANTITY

# compute expiry in year fraction
iEXPIRY = (pd.to_datetime(sEXPIRY_DATE) - pd.to_datetime(sVALUATION_DATE)).days
dEXPIRY = iEXPIRY / iBASE_DAYS

# compute risk-free rate
dRISK_FREE_RATE = portfolio_tuple.oRISK_FREE_RATE_TERM(dEXPIRY)

# get implied vol surface
dVOL_SURFACE = portfolio_tuple.dVOL_SURFACE

x = dVOL_SURFACE.columns
y = dVOL_SURFACE.index
z = dVOL_SURFACE.to_numpy() * .01

dFWD_PRICE = dSPOT_PRICE * np.exp((dRISK_FREE_RATE - dDIVIDEND_YIELD) * dEXPIRY)
dMONEYNESS = np.array(dSTRIKE_PRICE / dFWD_PRICE * 100.)
dIMP_VOLATILITY = np.array(pchip_interpolator2d(x=x, y=y, z=z.T, x1=dMONEYNESS, x2=np.array(iEXPIRY)).squeeze())

pv = get_vanilla_pv(payoff_type=sPAYOFF, imp_volatility=dIMP_VOLATILITY, spot_price=dSPOT_PRICE, strike_price=dSTRIKE_PRICE, risk_free_rate=dRISK_FREE_RATE, dividend_yield=dDIVIDEND_YIELD, expiry=dEXPIRY)
pv * iQUANTITY

# %% cell 30
logger.info("[cell %d/51] compute iRISK_HORIZON_DAYS", 30)
iRISK_HORIZON_DAYS = np.int64(10)

shock_matrix_daily = np.copy(shock_returns_dict[sVALUATION_DATE][sUNDERLYING][:, 1:iRISK_HORIZON_DAYS + 1])

shock_matrix_daily

# %% cell 31
logger.info("[cell %d/51] compute x", 31)
x = (1 + -0.067882) * (1 + -0.028083) - 1.
(1 + x) * (1 + 0.016855) - 1.

# %% cell 32
logger.info("[cell %d/51] compute iRISK_HORIZON_DAYS", 32)
iRISK_HORIZON_DAYS = np.int64(10)

shock_matrix_daily = np.copy(shock_returns_dict[sVALUATION_DATE][sUNDERLYING][:, 1:iRISK_HORIZON_DAYS + 1])

for i in range(1, shock_matrix_daily.shape[1]):
    shock_matrix_daily[:, i] = (1. + shock_matrix_daily[:, i-1]) * (1. + shock_matrix_daily[:, i]) - 1.

shock_matrix_daily

# %% cell 33
logger.info("[cell %d/51] Cells 30 and 32 overwrote iRISK_HORIZON_DAYS to 10 for a scr", 33)
# Cells 30 and 32 overwrote iRISK_HORIZON_DAYS to 10 for a scratch demo.
# The canonical Table 2 pipeline is a 1-day risk horizon; downstream cell 34
# reads back "final_output_1d.csv", so reset the horizon here.
iRISK_HORIZON_DAYS = np.int64(1)
if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor() as executor:

        results1 = executor.map(get_pv, portfolio_list)
        results2 = executor.map(get_pv, portfolio_list, portfolio_up_list)
        results3 = executor.map(get_pv, portfolio_list, portfolio_dn_list)
        results4 = executor.map(get_pv, portfolio_list, portfolio_risk_list)
        results5 = executor.map(get_pv, portfolio_list, portfolio_risk_base_list)

        pv_df = []
        pv_up_df = []
        pv_dn_df = []
        pv_risk_df = []
        pv_risk_base_df = []
        for result1, result2, result3, result4, result5 in zip(results1, results2, results3, results4, results5):
            pv_df.append(pd.DataFrame(result1))
            pv_up_df.append(pd.DataFrame(result2))
            pv_dn_df.append(pd.DataFrame(result3))
            pv_risk_df.append(pd.DataFrame(result4))
            pv_risk_base_df.append(pd.DataFrame(result5))

pv_df = pd.concat(pv_df)
pv_up_df = pd.concat(pv_up_df)
pv_dn_df = pd.concat(pv_dn_df)
pv_risk_df = pd.concat(pv_risk_df)
pv_risk_base_df = pd.concat(pv_risk_base_df)

pv_df = pv_df.rename(columns={'dPV': 'dPV_T0'})
pv_risk_df = pv_risk_df.rename(columns={'dPV': 'dPV_STRESSED'})
pv_risk_base_df = pv_risk_base_df.rename(columns={'dPV': 'dPV_STRESSED'})

index_names = ['sVALUATION_DATE', 'sUNDERLYING', 'sTYPOLOGY', 'sSTRATEGY', 'sPAYOFF', 'sEXPIRY_DATE', 'dSTRIKE_PRICE']
Delta_df = (pv_up_df.set_index(index_names).dPV - pv_dn_df.set_index(index_names).dPV) / (pv_up_df.dSPOT_SHIFT * pv_up_df.dSPOT_PRICE * pv_up_df.iQUANTITY.apply(np.abs)).to_numpy() * .5
Delta_df.name = 'dDELTA'

pl_df = (pv_risk_df.set_index(index_names).
         join(pv_df.set_index(index_names)['dPV_T0']).
         join(Delta_df))

pl_base_df = (pv_risk_base_df.set_index(index_names).
              join(pv_df.set_index(index_names)['dPV_T0']).
              join(Delta_df))

pl_df['dPL_DELTA'] = pl_df.dDELTA * pl_df.dSPOT_PRICE * pl_df.dSPOT_SHIFT * pl_df.iQUANTITY.apply(np.abs)
pl_df['dPL'] = (pl_df.dPV_STRESSED - pl_df.dPV_T0) - pl_df.dPL_DELTA
pl_df.to_csv(str(_REPO_ROOT / "Study" / "Collar Asian" / f"final_output_{iRISK_HORIZON_DAYS}d.csv"))

pl_base_df['dPL_DELTA'] = pl_base_df.dDELTA * pl_base_df.dSPOT_PRICE * pl_base_df.dSPOT_SHIFT * pl_base_df.iQUANTITY.apply(np.abs)
pl_base_df['dPL'] = (pl_base_df.dPV_STRESSED - pl_base_df.dPV_T0) - pl_base_df.dPL_DELTA
pl_base_df.to_csv(str(_REPO_ROOT / "Study" / "Collar Asian" / f"final_output_{iRISK_HORIZON_DAYS}d_base.csv"))

# %% cell 34
logger.info("[cell %d/51] compute pl_df1", 34)
pl_df1 = pd.read_csv(str(_REPO_ROOT / "Study" / "Collar Asian" / "final_output_1d.csv"))
pl_df2 = pd.read_csv(str(_REPO_ROOT / "Study" / "Collar Asian" / "final_output_1d_base.csv"))

pl_df1['bIS_BASE_SCENARIO'] = False
pl_df2['bIS_BASE_SCENARIO'] = True

pl_df = pd.concat([pl_df1, pl_df2])

index_names = ['sVALUATION_DATE', 'sUNDERLYING', 'sTYPOLOGY', 'sSTRATEGY', 'sPAYOFF', 'sEXPIRY_DATE', 'dSTRIKE_PRICE', 'bIS_BASE_SCENARIO']

pl_df.sVALUATION_DATE = pl_df.sVALUATION_DATE.astype(str)
pl_df = pl_df.set_index(index_names)

# %% cell 35
logger.info("[cell %d/51] parallel Monte Carlo simulation", 35)
if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor() as executor:

        results1 = executor.map(get_pv, portfolio_list)
        results2 = executor.map(get_pv, portfolio_list, portfolio_up_list)
        results3 = executor.map(get_pv, portfolio_list, portfolio_dn_list)

        pv_df = []
        pv_up_df = []
        pv_dn_df = []
        for result1, result2, result3 in zip(results1, results2, results3):
            pv_df.append(pd.DataFrame(result1))
            pv_up_df.append(pd.DataFrame(result2))
            pv_dn_df.append(pd.DataFrame(result3))

pv_df = pd.concat(pv_df)
pv_up_df = pd.concat(pv_up_df)
pv_dn_df = pd.concat(pv_dn_df)

pv_df = pv_df.rename(columns={'dPV': 'dPV_T0'})

# %% cell 36
logger.info("[cell %d/51] compute confidence_level1", 36)
confidence_level1 = 1
confidence_level99 = 99
stats_gatherer_risk1 = StatisticsMCQuantile(alpha=np.array(confidence_level1) * .01)
stats_gatherer_risk99 = StatisticsMCQuantile(alpha=np.array(confidence_level99) * .01)

var_df = []

for underlying_name in SET_OF_TICKERS[1:]:
    for valuation_date in valuation_date_array[:-1]:
        for is_base_scenario in [True, False]:
            valuation_date_str = valuation_date.strftime("%Y%m%d")
            tmp_df = pl_df.loc[
                (pl_df.index.get_level_values('sVALUATION_DATE')==valuation_date_str) &
                (pl_df.index.get_level_values('sUNDERLYING')==underlying_name) &
                (pl_df.index.get_level_values('bIS_BASE_SCENARIO')==is_base_scenario),
                ['iSCENARIO_NUM', 'dPL']
            ].groupby(['iSCENARIO_NUM']).sum()
            stats_gatherer_risk1.dump_result(tmp_df.to_numpy())
            stats_gatherer_risk99.dump_result(tmp_df.to_numpy())
            var_df.append((valuation_date_str, underlying_name, is_base_scenario, 'VaR', confidence_level1, -1 * stats_gatherer_risk1.get_result_so_far()[0, 0]))
            var_df.append((valuation_date_str, underlying_name, is_base_scenario, 'VaR', confidence_level99, -1 * stats_gatherer_risk99.get_result_so_far()[0, 0]))

var_df = pd.DataFrame(var_df, columns=['sVALUATION_DATE', 'sUNDERLYING', 'bIS_BASE_SCENARIO', 'sLOSS_DESC', 'iCONFIDENCE_LEVEL', 'dLOSS'])
var_df.loc[var_df.iCONFIDENCE_LEVEL==1]

# %% cell 37
logger.info("[cell %d/51] (var_df.loc[(var_df.iCONFIDENCE_LEVEL==1) & (var_df.bIS_BASE", 37)
(var_df.loc[(var_df.iCONFIDENCE_LEVEL==1) & (var_df.bIS_BASE_SCENARIO==False), ['sVALUATION_DATE', 'dLOSS']].set_index('sVALUATION_DATE') /
 var_df.loc[(var_df.iCONFIDENCE_LEVEL==1) & (var_df.bIS_BASE_SCENARIO==True), ['sVALUATION_DATE', 'dLOSS']].set_index('sVALUATION_DATE'))

# %% cell 38
logger.info("[cell %d/51] compute ci_var_la", 38)
ci_var_la = {}
ci_var_bs = {}

for t in valuation_date_array:
    t = t.strftime('%Y%m%d')
    pl_vec_la = pl_df.xs((t, False), level=('sVALUATION_DATE', 'bIS_BASE_SCENARIO')).pivot_table(index='iSCENARIO_NUM', values='dPL', aggfunc='sum').to_numpy()
    pl_vec_bs = pl_df.xs((t, True), level=('sVALUATION_DATE', 'bIS_BASE_SCENARIO')).pivot_table(index='iSCENARIO_NUM', values='dPL', aggfunc='sum').to_numpy()

    bootstrap_vec_la = []
    bootstrap_vec_bs = []
    rng_gen = RandomMT19937(seed=np.int64(20241231))

    for i in range(n_paths):
        sample = rng_gen.generator.choice(n_paths, size=n_paths, replace=True)

        stats_gatherer_risk1.dump_result(pl_vec_la[sample])
        bootstrap_vec_la.append(stats_gatherer_risk1.get_result_so_far()[0, 0])

        stats_gatherer_risk1.dump_result(pl_vec_bs[sample])
        bootstrap_vec_bs.append(stats_gatherer_risk1.get_result_so_far()[0, 0])

    ci_var_la[t] = (np.quantile(bootstrap_vec_la, 1-.975), np.quantile(bootstrap_vec_la, .975))
    ci_var_bs[t] = (np.quantile(bootstrap_vec_bs, 1-.975), np.quantile(bootstrap_vec_bs, .975))

# %% cell 39
logger.info("[cell %d/51] pd.DataFrame().from_dict(ci_var_la).T.round(3)", 39)
pd.DataFrame().from_dict(ci_var_la).T.round(3)

# %% cell 40
logger.info("[cell %d/51] pd.DataFrame().from_dict(ci_var_bs).T.round(3)", 40)
pd.DataFrame().from_dict(ci_var_bs).T.round(3)

# %% cell 41
logger.info("[cell %d/51] (pd.DataFrame().from_dict(ci_var_la).T / pd.DataFrame().from", 41)
(pd.DataFrame().from_dict(ci_var_la).T / pd.DataFrame().from_dict(ci_var_bs).T).round(3)

# %% cell 42
logger.info("[cell %d/51] fig, ax = plt.subplots(figsize=(15, 7))", 42)
fig, ax = plt.subplots(figsize=(15, 7))

underlying_name = SET_OF_TICKERS[-1]

tmp_df = var_df.copy()
tmp_df['dtVALUATION_DATE'] = pd.to_datetime(tmp_df.sVALUATION_DATE)
# VaR is reported as a positive loss magnitude by convention.
tmp_df['dLOSS'] = tmp_df['dLOSS'].abs()

tmp_df2 = tmp_df.copy()
tmp_df2 = tmp_df2.pivot_table(index=['sUNDERLYING', 'iCONFIDENCE_LEVEL', 'dtVALUATION_DATE'], columns='bIS_BASE_SCENARIO', values='dLOSS')
tmp_df2 = tmp_df2[False] / np.minimum(tmp_df2[True], 1.)
tmp_df2.name = 'dRATIO'
#
# tmp_df3 = prices_df.loc[valuation_date_array[:-1], underlying_name]
# tmp_df3.name = 'dPRICES'
# tmp_df3.index.name = 'dtVALUATION_DATE'

# ax2 = ax.twinx()

sns.lineplot(data=tmp_df.loc[(tmp_df.sUNDERLYING == underlying_name) & (tmp_df.bIS_BASE_SCENARIO == True) & (tmp_df.iCONFIDENCE_LEVEL==confidence_level1)],
             x='dtVALUATION_DATE', y='dLOSS', ax=ax, label="BS VaR", linestyle='--', linewidth=2)
sns.lineplot(data=tmp_df.loc[(tmp_df.sUNDERLYING == underlying_name) & (tmp_df.bIS_BASE_SCENARIO == False) & (tmp_df.iCONFIDENCE_LEVEL==confidence_level1)],
             x='dtVALUATION_DATE', y='dLOSS', ax=ax, label="LA VaR", linestyle=':', color='red', linewidth=2)

# sns.lineplot(data=tmp_df2.xs(1, level='iCONFIDENCE_LEVEL').reset_index(), x='dtVALUATION_DATE', y='dRATIO', linestyle='-.', linewidth=1, marker='P', color='grey', label='Ratio of Liquidity to Baseline VaR (RHS)', ax=ax2)

handles1, labels1 = ax.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()

# plt.legend(handles1 + handles2, labels1 + labels2, loc='best')


ax.set_xlabel(r'Underlying Price Shocks (%)')
ax.set_xlabel(None)
ax.set_ylabel(f"{tmp_df.sLOSS_DESC.iloc[0]} ($)")
ax.set_title(rf'1-Day 99 {tmp_df.sLOSS_DESC.iloc[0]} Comparison between Baseline (BS) and Liquidity Adjusted (LA)', fontweight='bold')

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlim(tmp_df.dtVALUATION_DATE.min(), tmp_df.dtVALUATION_DATE.max())
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
# ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
# ax2.set_ylabel('Ratio', rotation=270, labelpad=20)

fig.autofmt_xdate()

plt.savefig(str(_FIGURES_DIR / "daily_var_comparison.pdf"), dpi=300)
plt.show()

# %% cell 43
logger.info("[cell %d/51] compute valuation_date", 43)
valuation_date = '20250409'
mask = pl_df.index.get_level_values('sVALUATION_DATE')==valuation_date

# Paper Figure 2 plots portfolio-level 1-day P&L (sum across the four
# collar+asian legs per Monte Carlo path), not per-position P&L. Aggregating
# by scenario collapses 40,000 per-leg rows into 10,000 portfolio rows.
_la_slice = pl_df.loc[mask & (pl_df.index.get_level_values('bIS_BASE_SCENARIO')==False)].reset_index()
_bs_slice = pl_df.loc[mask & (pl_df.index.get_level_values('bIS_BASE_SCENARIO')==True)].reset_index()
df1 = _la_slice.groupby('iSCENARIO_NUM', as_index=False)['dPL'].sum()
df2 = _bs_slice.groupby('iSCENARIO_NUM', as_index=False)['dPL'].sum()
df1['Return Assumption'] = 'LA'
df2['Return Assumption'] = 'BS'
simulated_pl_df = pd.concat([df1, df2])
simulated_pl_df = simulated_pl_df.set_index('Return Assumption')

# %% cell 44
logger.info("[cell %d/51] compute valuation_date", 44)
valuation_date = '20250409'
mask = pl_df.index.get_level_values('sVALUATION_DATE')==valuation_date

fig, ax = plt.subplots(figsize=(15, 7))

# Paper Figure 2: solid lines with small periodic markers.
# BS -> orange with star markers ('*'), LA -> blue with circle markers ('o').
# bw_adjust=5 matches the paper Notes: "smoothing parameter of five".
# common_norm=False so each hue integrates to 1 independently
# (default common_norm=True halves the peak when BS/LA sample counts are equal).
sns.kdeplot(data=simulated_pl_df.reset_index(), x='dPL', hue='Return Assumption',
            fill=False, gridsize=500, bw_adjust=5., common_norm=False, ax=ax)
# Overlay markers on the two KDE lines. [0] = BS (orange star), [1] = LA (blue circle).
_pl_kde_lines = ax.get_lines()
_pl_marker_specs = [('*', 5), ('o', 4)]
for _line, (_m, _size) in zip(_pl_kde_lines, _pl_marker_specs):
    _line.set_marker(_m)
    _line.set_markersize(_size)
    _line.set_markevery(8)
    _line.set_markeredgewidth(0)

ax.set_title(rf'Density Comparison of Liquidity Adjusted (LA) and Baseline (BS) 1-Day Simulated P&L Vector', fontweight='bold')
ax.set_xlabel(r'Simulated 1-Day P&L ($)')
# Match paper Figure 2 tick spacing: x every $25, y every 0.002.
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.002))
# Force y-axis to extend to 0.018 so the paper's nine ticks (0.000..0.016) render
# regardless of the observed peak height.
ax.set_ylim(0, 0.018)
ax.tick_params(which='both', length=3, width=0.8, direction='out')
ax.tick_params(axis='x', labelrotation=45)
plt.setp(ax.get_legend().get_title(), fontsize='14')

plt.xlim(-200, 150)   # matches paper Figure 2 caption: "-$200 to +$150"
# Diagnostic: log the empirical std of the plotted P&L distributions.
logger.info(
    "Figure 2 diagnostic -- observed BS PL sigma=%.2f  LA PL sigma=%.2f  (ratio LA/BS=%.2f)",
    float(df2['dPL'].std()),
    float(df1['dPL'].std()),
    float(df1['dPL'].std()) / max(float(df2['dPL'].std()), 1e-9),
)
plt.savefig(str(_FIGURES_DIR / "Figure_2_simulated_pl.pdf"), dpi=300)
plt.show();

# %% cell 45
logger.info("[cell %d/51] fig, ax = plt.subplots(figsize=(15, 7))", 45)
fig, ax = plt.subplots(figsize=(15, 7))

underlying_name = SET_OF_TICKERS[-1]
valuation_date = valuation_date_array[0]

spot_price_t0 = prices_df.xs(valuation_date)[underlying_name]
strike_lower_asian = portfolio_pos_dict[valuation_date.strftime("%Y%m%d")]['COIN'][0].dSTRIKE_PRICE
strike_upper_asian = portfolio_pos_dict[valuation_date.strftime("%Y%m%d")]['COIN'][1].dSTRIKE_PRICE
strike_lower_hedge = portfolio_pos_dict[valuation_date.strftime("%Y%m%d")]['COIN'][2].dSTRIKE_PRICE
strike_upper_hedge = portfolio_pos_dict[valuation_date.strftime("%Y%m%d")]['COIN'][3].dSTRIKE_PRICE

dates = [pd.to_datetime('20250402'), pd.to_datetime('20250403'), pd.to_datetime('20250409'), valuation_date_array[-2]]
linestyles = ['-', ':', '-.', '--']

for valuation_date, linestyle in zip(dates, linestyles):
    tmp_df1 = pl_df.loc[
        (pl_df.index.get_level_values('sVALUATION_DATE')==valuation_date.strftime("%Y%m%d")) &
        (pl_df.index.get_level_values('sUNDERLYING')==underlying_name) &
        (pl_df.index.get_level_values('bIS_BASE_SCENARIO')==False),
        ['iSCENARIO_NUM', 'dSPOT_SHIFT', 'dPL', 'dPL_DELTA']
    ]
    tmp_df1.dSPOT_SHIFT = tmp_df1.dSPOT_SHIFT * 100.
    tmp_df1 = tmp_df1.groupby(['iSCENARIO_NUM', 'dSPOT_SHIFT']).sum()
    sns.lineplot(
        data=tmp_df1.reset_index(), x='dSPOT_SHIFT', y='dPL', ax=ax,
        label=rf"{valuation_date.strftime('%b %d')}, $S=\${prices_df.xs(valuation_date)[underlying_name]:.2f}$", linestyle=linestyle)

# tmp_df2 = pl_df.loc[
#     (pl_df.index.get_level_values('sVALUATION_DATE')==valuation_date.strftime("%Y%m%d")) & (pl_df.index.get_level_values('sUNDERLYING')==underlying_name) & (pl_df.index.get_level_values('bIS_BASE_SCENARIO')==True),
#     ['iSCENARIO_NUM', 'dSPOT_SHIFT', 'dPL']
# ]
# tmp_df2.dSPOT_SHIFT = tmp_df2.dSPOT_SHIFT * 100.
# tmp_df2 = tmp_df2.groupby(['iSCENARIO_NUM', 'dSPOT_SHIFT']).sum()
# sns.scatterplot(data=tmp_df2.reset_index(), x='dSPOT_SHIFT', y='dPL', ax=ax, label=r"$\gamma_i = \beta_i = \rho_{i,x} = 0$")

ax.set_xlabel(r'Simulated 1-Day Price Shocks (%)')
ax.set_ylabel('P&L ($)')
ax.set_title("P&L Ladder per Simulated Price Shock", fontweight='bold')

ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.tick_params(axis='x', labelrotation = 45)
plt.legend(title='Risk Date, Spot Price (S)', title_fontsize=14)
plt.xlim(-20., 20.)
plt.ylim(-700, 300.)

plt.savefig(str(_FIGURES_DIR / "pl_ladder.pdf"), dpi=300)
plt.show()

# %% cell 46
logger.info("[cell %d/51] VaR term structure across risk horizons (needs multi-horizon CSVs)", 46)
# Cells 46-51 assemble a VaR term structure plot from per-horizon output
# CSVs (final_output_{1,5,10,15,20}d_horizon.csv). Those files are only
# produced when the full pipeline of cells 33 is looped over multiple risk
# horizons — the equivalent plot is produced by Scripts/run_var_kimyi2025.py
# (Figure 7), so this stanza is redundant for the paper's outputs.
# We skip cleanly if the horizon files are absent.
file_names = [
    "final_output_1d_horizon.csv",
    "final_output_5d_horizon.csv",
    "final_output_10d_horizon.csv",
    "final_output_15d_horizon.csv",
    "final_output_20d_horizon.csv",
]

_missing = [
    fn for fn in file_names
    if not (_REPO_ROOT / "Study" / "Collar Asian" / fn).exists()
]
if _missing:
    logger.warning(
        "Skipping cells 46-51 (VaR term structure ladder). Missing horizon "
        "CSVs: %s. Figure 7 in the paper is produced by "
        "Scripts/run_var_kimyi2025.py; this stanza is redundant.",
        _missing,
    )
    import sys as _sys
    _sys.exit(0)

df = []
for file_name in file_names:
    df.append(pd.read_csv(str(_REPO_ROOT / "Study" / "Collar Asian" / file_name)))

df = pd.concat(df)
df.info()

# %% cell 47
logger.info("[cell %d/51] compute df2", 47)
df2 = df.pivot_table(index=['sVALUATION_DATE', 'iSCENARIO_NUM'], columns='iRISK_HORIZON_DAYS', values='dPL', aggfunc='sum').copy()
df2

# %% cell 48
logger.info("[cell %d/51] compute xi", 48)
xi = np.array(df2.columns)
yi = np.quantile(df2.xs(20250331).to_numpy().T, q=0.01, axis=1)
x = np.linspace(1, 20, 20, dtype=np.int64)
yi

# %% cell 49
logger.info("[cell %d/51] compute", 49)
from scipy.interpolate import pchip_interpolate

# %% cell 50
logger.info("[cell %d/51] compute y", 50)
y = pchip_interpolate(xi=xi, yi=yi, x=x)
y

# %% cell 51
logger.info("[cell %d/51] fig, ax = plt.subplots(figsize=(15, 7))", 51)
fig, ax = plt.subplots(figsize=(15, 7))

sns.lineplot(x=x, y=y * -1., ax=ax)

ax.set_xlabel(r'Risk Horizon in Days')
ax.set_ylabel('VaR ($)')
ax.set_title("P&L Ladder per Simulated Price Shock", fontweight='bold')

ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.show()
