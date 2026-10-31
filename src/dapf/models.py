"""Forecasting models: naive benchmark, LEAR-style per-hour Lasso, gradient boosting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed
from sklearn.linear_model import LassoLarsIC


def naive_forecast(P: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
    """Standard EPF naive benchmark (Lago et al. 2021): Mon/Sat/Sun use the same day last week, other days D-1."""
    lag = 7 if day.day_of_week in (0, 5, 6) else 1
    return P.loc[day - pd.Timedelta(days=lag)].to_numpy(float)


class AsinhScaler:
    """Median/MAD normalisation followed by the area hyperbolic sine (variance-stabilising, handles negatives)."""

    def fit(self, x):
        x = np.asarray(x, float)
        self.med = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - self.med), axis=0) * 1.4826
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transform(self, x):
        return np.arcsinh((np.asarray(x, float) - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


def _fit_hour(Z, z):
    return LassoLarsIC(criterion="aic", max_iter=500).fit(Z, z)


class LEAR:
    """LEAR-style model: 24 hourly Lasso regressions (LARS path, AIC-selected penalty) on asinh-scaled data,
    in the spirit of Lago, Marcjasz, De Schutter & Weron (2021), simplified (fixed lag set, single window)."""

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        X, Y = X[ok].to_numpy(float), Y[ok].to_numpy(float)
        self.sx = AsinhScaler().fit(X)
        # one scaler for all 24 target hours keeps the price scale consistent across hours
        self.sy = AsinhScaler().fit(Y.reshape(-1, 1))
        Z = self.sx.transform(X); Zy = self.sy.transform(Y.reshape(-1, 1)).reshape(Y.shape)
        if self.n_jobs == 1:
            self.models = [_fit_hour(Z, Zy[:, h]) for h in range(24)]
        else:
            self.models = Parallel(n_jobs=self.n_jobs)(delayed(_fit_hour)(Z, Zy[:, h]) for h in range(24))
        return self

    def predict(self, x_row: pd.Series | pd.DataFrame) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_row, float))
        z = self.sx.transform(x)
        zy = np.array([[m.predict(z)[i] for m in self.models] for i in range(len(z))])
        return self.sy.inverse(zy.reshape(-1, 1)).reshape(zy.shape)

    def n_selected(self) -> np.ndarray:
        return np.array([(np.abs(m.coef_) > 0).sum() for m in self.models])


GBM_FEATURES = ["hour", "dow", "month", "holiday", "load", "solar", "wind", "resload", "res_share", "resload_mean",
                "resload_max", "resload_min", "resload_d1", "resload_delta", "solar_sum", "wind_mean", "p_d1", "p_d2",
                "p_d7", "p_d1_mean", "p_d1_max", "p_d1_min", "p_d1_h19", "p_d1_h12", "p_7d_mean"]


def make_gbm(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, categorical_features=[0, 1, 2, 3],
                                         random_state=seed)
