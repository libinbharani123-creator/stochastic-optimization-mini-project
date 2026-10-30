"""Demand representation for the (s,S) DP."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class DemandDist:
    support: np.ndarray            # integer demand values
    pmf: np.ndarray                # probability mass function, sums to 1
    label: str = "empirical"

    def mean(self) -> float:
        return float((self.support * self.pmf).sum())

    def std(self) -> float:
        m = self.mean()
        return float(np.sqrt(((self.support - m) ** 2 * self.pmf).sum()))

    def cv(self) -> float:
        m = self.mean()
        return self.std() / m if m > 0 else 0.0

    def quantile(self, q: float) -> int:
        cdf = np.cumsum(self.pmf)
        idx = int(np.searchsorted(cdf, q))
        return int(self.support[min(idx, len(self.support) - 1)])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.support, size=n, p=self.pmf)

    def summary(self) -> str:
        return (f"{self.label}: mean={self.mean():.1f}  "
                f"std={self.std():.1f}  cv={self.cv():.2f}  "
                f"support=[{self.support[0]},{self.support[-1]}]")


def from_empirical(values, max_value=None) -> DemandDist:
    v = np.asarray(values, dtype=int)
    v = v[v >= 0]
    if max_value is None:
        max_value = int(max(v.max(), 1))
    support = np.arange(0, max_value + 1)
    counts = np.bincount(v, minlength=max_value + 1)[:max_value + 1].astype(float)
    pmf = counts / counts.sum()
    return DemandDist(support, pmf, "empirical")


def from_negbin(mean: float, cv: float, quantile: float = 0.9999) -> DemandDist:
    from scipy.stats import nbinom
    n = 1.0 / cv ** 2
    p = n / (n + mean)
    hi = int(nbinom.ppf(quantile, n, p))
    support = np.arange(0, hi + 1)
    pmf = nbinom.pmf(support, n, p)
    return DemandDist(support, pmf / pmf.sum(), f"negbin(mean={mean:.0f},cv={cv:.2f})")


def from_poisson(mean: float, quantile: float = 0.9999) -> DemandDist:
    from scipy.stats import poisson
    hi = int(poisson.ppf(quantile, mean))
    support = np.arange(0, hi + 1)
    pmf = poisson.pmf(support, mean)
    return DemandDist(support, pmf / pmf.sum(), f"poisson(mean={mean:.0f})")


def from_normal(mean: float, std: float, quantile: float = 0.9999) -> DemandDist:
    from scipy.stats import norm
    hi = int(norm.ppf(quantile, mean, std))
    support = np.arange(0, max(hi, 1) + 1)
    pmf = norm.pdf(support, mean, std)
    pmf[0] += norm.cdf(0, mean, std)          # mass below 0 folded onto 0
    pmf = pmf / pmf.sum()
    return DemandDist(support, pmf, f"normal(mean={mean:.0f},std={std:.0f})")


def fit_from_data(path: str) -> DemandDist:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if "stockout_flag" in df:
        df = df[df["stockout_flag"] == 0]
    return from_empirical(df["units_sold"].to_numpy())
