import numpy as np
import pandas as pd

from dapf.evaluate import dm_test, mae, rmae, rmse, smape
from dapf.models import LEAR, AsinhScaler, naive_forecast


def test_metrics_basic():
    y = np.array([10.0, -10.0, 20.0]); f = np.array([12.0, -10.0, 17.0])
    assert np.isclose(mae(y, f), 5 / 3) and np.isclose(rmse(y, f), np.sqrt(13 / 3))
    assert np.isclose(rmae(y, f, y + 5), (5 / 3) / 5)
    assert smape(y, y) == 0.0


def test_dm_test_detects_better_forecast():
    rng = np.random.default_rng(1)
    y = rng.normal(80, 30, (300, 24))
    good = y + rng.normal(0, 5, y.shape); bad = y + rng.normal(0, 15, y.shape)
    _, p = dm_test(y, bad, good)
    assert p < 1e-6                                              # good significantly better than bad
    _, p_rev = dm_test(y, good, bad)
    assert p_rev > 0.99


def test_naive_rule():
    idx = pd.date_range("2026-01-01", periods=21, freq="D")
    P = pd.DataFrame(np.repeat(np.arange(21.0)[:, None], 24, axis=1), index=idx)
    assert naive_forecast(P, pd.Timestamp("2026-01-14"))[0] == 12.0   # Wednesday -> D-1
    assert naive_forecast(P, pd.Timestamp("2026-01-19"))[0] == 11.0   # Monday -> D-7
    assert naive_forecast(P, pd.Timestamp("2026-01-17"))[0] == 9.0    # Saturday -> D-7


def test_asinh_scaler_roundtrip_with_negatives():
    x = np.array([[-500.0], [-5.0], [0.0], [50.0], [900.0]])
    s = AsinhScaler().fit(x)
    assert np.allclose(s.inverse(s.transform(x)), x)


def test_lear_recovers_linear_signal():
    rng = np.random.default_rng(2)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"x{i}" for i in range(10)])
    Y = pd.DataFrame({h: 60 + 8 * X["x0"] - 5 * X["x3"] + rng.normal(0, 0.5, n) for h in range(24)})
    m = LEAR().fit(X.iloc[:300], Y.iloc[:300])
    pred = m.predict(X.iloc[300:])
    assert pred.shape == (100, 24)
    assert np.mean(np.abs(pred - Y.iloc[300:].to_numpy())) < 1.5
    assert all(abs(mod.coef_[0]) > 0 and abs(mod.coef_[3]) > 0 for mod in m.models)   # true drivers always kept
    assert m.n_selected().mean() < 8                             # sparse on average: most irrelevant inputs dropped
