#!/usr/bin/env python3
"""Generate synthetic daily sales with trend, weekday, season, promo."""
import numpy as np
import pandas as pd


def generate(n_days: int = 1095, base: float = 100.0, seed: int = 2024):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 1.0 + 0.0002 * t
    dow_f = np.array([1.00, 0.94, 0.91, 0.95, 1.06, 1.24, 1.14])[dates.dayofweek]
    annual = 1.0 + 0.08 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
    promo = (rng.random(n_days) < 0.08).astype(int)
    promo_lift = np.where(promo == 1, 1.55, 1.0)

    mu = base * trend * dow_f * annual * promo_lift

    cv = 0.22
    n_nb = 1.0 / cv ** 2
    p_nb = n_nb / (n_nb + mu)
    demand = rng.negative_binomial(n_nb, p_nb)

    hist_order = np.round(mu * 1.30)
    observed = np.minimum(demand, hist_order)
    stockout = (demand > hist_order).astype(int)

    return pd.DataFrame({
        "date": dates, "units_sold": observed, "promo": promo,
        "stockout_flag": stockout, "true_demand": demand,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("daily_sales.csv", index=False)
    print(f"wrote daily_sales.csv ({len(df)} rows)")
    print(f"  mean demand   : {df.true_demand.mean():.1f}")
    print(f"  stockout days : {df.stockout_flag.mean()*100:.1f}%")
