import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from estimator import DistributionalEstimator, bayes_prevalence, rogan_gladen  # noqa: E402


def _synthetic(seed=0, V=1500):
    rng = np.random.default_rng(seed)
    ph = rng.beta(0.5, 8, V)
    pm = np.clip(ph * np.exp(rng.normal(0, 0.6, V)), 0, 0.95)

    def sample(p, n):
        return [set(np.where(rng.random(V) < p)[0].astype(str)) for _ in range(n)]
    return ph, pm, sample


def test_recovers_alpha_within_tolerance():
    ph, pm, sample = _synthetic()
    H = sample(ph, 2000)
    M = {"m1": sample(pm, 1000), "m2": sample(np.clip(pm * np.exp(np.random.default_rng(1).normal(0, 0.2, len(pm))), 0, 0.95), 1000)}
    est = DistributionalEstimator(selection="ratio", top_k=600, min_df=0.002).fit_reference(H, M)
    for a in [0.0, 0.05, 0.2]:
        n, k = 800, int(800 * a)
        C = sample(ph, n - k) + sample(pm, k)
        r = est.estimate(C, n_boot=50)
        assert abs(r["alpha"] - a) < 0.03, (a, r)
        assert r["ci_lo"] <= r["alpha"] <= r["ci_hi"]


def test_selection_modes_run():
    ph, pm, sample = _synthetic(2)
    H, M = sample(ph, 500), {"m": sample(pm, 500)}
    for sel in ["ratio", "excess", "all"]:
        est = DistributionalEstimator(selection=sel, top_k=200, min_df=0.01).fit_reference(H, M)
        assert 0 <= est.estimate(sample(ph, 200))["alpha"] <= 1


def test_save_load_roundtrip(tmp_path):
    ph, pm, sample = _synthetic(3)
    est = DistributionalEstimator(selection="all", min_df=0.01).fit_reference(sample(ph, 300), {"m": sample(pm, 300)})
    est.save(tmp_path / "ref.json")
    est2 = DistributionalEstimator.load(tmp_path / "ref.json")
    C = sample(pm, 100)
    assert abs(est.estimate(C)["alpha"] - est2.estimate(C)["alpha"]) < 1e-9


def test_rogan_gladen_and_bayes_agree():
    rg = rogan_gladen(0.12, 0.8, 0.97, 1000, 500, 500)
    by = bayes_prevalence(120, 1000, 400, 500, 485, 500)
    assert abs(rg["alpha"] - by["alpha"]) < 0.02
    assert by["ci_lo"] < by["alpha"] < by["ci_hi"]
