import numpy as np

from dapf.probabilistic import (TAUS, conformal_symmetric, coverage, dm_pinball,
                                historical_simulation, pinball, prob_below, qra,
                                quantile_regression, winkler)


def _toy(n_days=80, seed=0):
    rng = np.random.default_rng(seed)
    f = 80 + 20 * np.sin(np.linspace(0, 2 * np.pi, 24))[None, :] + rng.normal(0, 5, (n_days, 24))
    y = f + rng.normal(0, 10, (n_days, 24))
    return y, f


def test_quantile_regression_recovers_known_quantile():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 400)
    y = 2 + 3 * x + rng.normal(0, 1, 400)
    X = np.column_stack([np.ones_like(x), x])
    b50 = quantile_regression(X, y, 0.5)
    assert abs(b50[1] - 3) < 0.1 and abs(b50[0] - 2) < 0.4
    b90 = quantile_regression(X, y, 0.9)
    assert 0.85 <= np.mean(y <= X @ b90) <= 0.95


def test_methods_are_ex_ante_and_ordered():
    y, f = _toy()
    t = 60
    for fn in (lambda Y: historical_simulation(Y, f, t, 56), lambda Y: conformal_symmetric(Y, f, t, 56),
               lambda Y: qra(Y, [f, f + 1.0], t, 28)):
        q = fn(y)
        assert q.shape == (24, len(TAUS))
        assert np.all(np.diff(q, axis=1) >= -1e-9)          # no quantile crossing
        y2 = y.copy(); y2[t:] += 1e4                        # changing day t or later must not change day t
        assert np.allclose(fn(y2), q)


def test_conformal_is_symmetric_and_median_is_point():
    y, f = _toy()
    q = conformal_symmetric(y, f, 70, 56)
    mid = int(np.argmin(np.abs(TAUS - 0.5)))
    assert np.allclose(q[:, mid], f[70])
    assert np.allclose(q[:, -1] - f[70], f[70] - q[:, 0])


def test_calibration_on_exchangeable_errors():
    y, f = _toy(n_days=400, seed=3)
    cov = []
    for t in range(56, 400):
        q = conformal_symmetric(y, f, t, 56)
        cov.append(coverage(y[t], q[:, 1], q[:, -2]))   # 10 % .. 90 % -> 80 % interval
    assert 0.76 < np.mean(cov) < 0.86


def test_scores_and_negative_probability():
    taus = TAUS
    q = np.tile(np.linspace(-10, 10, len(taus)), (2, 1))
    p = prob_below(q, 0.0)
    assert np.allclose(p, 0.5)
    assert np.allclose(prob_below(q + 100, 0.0), taus[0] / 2)
    assert np.allclose(prob_below(q - 100, 0.0), 1 - (1 - taus[-1]) / 2)
    y = np.array([0.0, 50.0])
    pb = pinball(y, q)
    assert pb.shape == (2, len(taus)) and np.all(pb >= 0)
    assert winkler(np.array([0.0]), np.array([-1.0]), np.array([1.0]), 0.1) == 2.0
    assert winkler(np.array([3.0]), np.array([-1.0]), np.array([1.0]), 0.1) == 2.0 + 20 * 2.0
    a, b = np.ones((30, 24, 3)), np.zeros((30, 24, 3)) + np.linspace(0, 0.5, 30)[:, None, None]
    stat, pval = dm_pinball(a, b)
    assert stat > 0 and pval < 0.01


def test_cqr_widens_undercovering_intervals_and_keeps_median():
    from dapf.probabilistic import conformalize
    rng = np.random.default_rng(5)
    k = 40
    y_hist = rng.normal(0, 10, (k, 24))
    narrow = np.tile(np.linspace(-3, 3, len(TAUS)), (k, 24, 1))   # far too narrow for sd 10
    q_t = np.tile(np.linspace(-3, 3, len(TAUS)), (24, 1))
    q_adj = conformalize(narrow, y_hist, q_t)
    mid = int(np.argmin(np.abs(TAUS - 0.5)))
    assert np.allclose(q_adj[:, mid], q_t[:, mid])
    assert np.all(q_adj[:, -1] > q_t[:, -1]) and np.all(q_adj[:, 0] < q_t[:, 0])
    # the 80 % interval after recalibration covers about 80 % of fresh draws
    y_new = rng.normal(0, 10, (2000, 24))
    i10, i90 = int(np.argmin(np.abs(TAUS - 0.1))), int(np.argmin(np.abs(TAUS - 0.9)))
    cov = np.mean((y_new >= q_adj[0, i10]) & (y_new <= q_adj[0, i90]))
    assert 0.74 < cov < 0.86
