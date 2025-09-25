#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from scipy.optimize import newton
# from Library.RootFinder import newton_raphson
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.OptionPricerKimYi2025 import kimyi_call, kimyi_put


class KimYiSkewCalibration(ABC):

    @abstractmethod
    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class KimYiSkewCalibrationSystematic(KimYiSkewCalibration):

    def __init__(
            self,
            mkt_imp_vol: NDArray[np.float64],
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64],
            is_call_option: NDArray[np.bool_],
            option_weights: NDArray[np.float64]
    ):
        self.mkt_imp_vol = mkt_imp_vol
        self.und_price = und_price
        self.und_strike = und_strike
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.time_to_expiry = time_to_expiry
        self.is_call_option = is_call_option
        self.option_weights = option_weights
        self.penalty = np.array(0.)

    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        mod_imp_vol = self.model_vol(x=x)

        return 0.5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        sigma, pprob, lamb, eta1, eta2 = x

        mod_imp_vol_put = _kimyi_imp_vol_put(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[~self.is_call_option],
            und_strike=self.und_strike[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option],
            time_to_expiry=self.time_to_expiry[~self.is_call_option]
        )

        mod_imp_vol_call = _kimyi_imp_vol_call(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[self.is_call_option],
            und_strike=self.und_strike[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option],
            time_to_expiry=self.time_to_expiry[self.is_call_option]
        )

        return np.vstack((mod_imp_vol_put, mod_imp_vol_call))

    @property
    def mkt_imp_vol(self) -> NDArray[np.float64]:
        return self._mkt_imp_vol

    @mkt_imp_vol.setter
    def mkt_imp_vol(self, value: NDArray[np.float64]):
        self._mkt_imp_vol = value.reshape((-1, 1))

    @property
    def und_price(self) -> NDArray[np.float64]:
        return self._und_price

    @und_price.setter
    def und_price(self, value: NDArray[np.float64]):
        self._und_price = value.reshape((-1, 1))

    @property
    def risk_free_rate(self) -> NDArray[np.float64]:
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: NDArray[np.float64]):
        self._risk_free_rate = value.reshape((-1, 1))

    @property
    def dividend_yield(self) -> NDArray[np.float64]:
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: NDArray[np.float64]):
        self._dividend_yield = value.reshape((-1, 1))

    @property
    def time_to_expiry(self) -> NDArray[np.float64]:
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, value: NDArray[np.float64]):
        self._time_to_expiry = value.reshape((-1, 1))

    @property
    def option_weights(self) -> NDArray[np.float64]:
        return self._option_weights

    @option_weights.setter
    def option_weights(self, value: NDArray[np.float64]):
        self._option_weights = value.reshape((-1, 1))


def _kimyi_imp_vol_call(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    prices = kimyi_call(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_call_obj = BlackScholesMertonCall(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_call_obj.price(volatility=x) - prices

    mod_iv_call = newton(
        func=obj_func,
        x0=np.array([1.5] * prices.shape[0]).reshape((-1, 1)),
        fprime=bsm_call_obj.vega,
        fprime2=bsm_call_obj.vomma
    )

    # mod_iv_call = newton_raphson(
    #     func=bsm_call_obj.price,
    #     func_deriv=bsm_call_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_call


def _kimyi_imp_vol_put(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    prices = kimyi_put(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_put_obj = BlackScholesMertonPut(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_put_obj.price(volatility=x) - prices

    mod_iv_put = newton(
        func=obj_func,
        x0=np.array([1.5] * prices.shape[0]).reshape((-1, 1)),
        fprime=bsm_put_obj.vega,
        fprime2=bsm_put_obj.vomma
    )

    # mod_iv_put = newton_raphson(
    #     func=bsm_put_obj.price,
    #     func_deriv=bsm_put_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_put
