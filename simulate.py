"""Simulate a periodic-review inventory policy with lead time."""
from __future__ import annotations
import numpy as np
import pandas as pd

from demand import DemandDist


def simulate(policy, dd: DemandDist, n_periods: int,
             K: float, c: float, h: float, b: float,
             lead_time: int = 0,
             rng: np.random.Generator | None = None,
             init_ip: int | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    if init_ip is None:
        init_ip = 100
    on_hand = float(init_ip)
    backlog = 0.0
    pipeline = [0.0] * lead_time
    demand_seq = dd.sample(n_periods, rng)

    rows = []
    for t in range(n_periods):
        if lead_time > 0:
            on_hand += pipeline.pop(0)

        IP = on_hand + sum(pipeline) - backlog
        q = max(policy(IP), 0.0)

        if lead_time > 0:
            pipeline.append(q)
        else:
            on_hand += q

        D = float(demand_seq[t])
        net_before = on_hand - backlog
        served = min(D, max(net_before, 0.0))
        net_after = net_before - D
        if net_after >= 0:
            on_hand, backlog = net_after, 0.0
        else:
            on_hand, backlog = 0.0, -net_after

        cost = K * (q > 0) + c * q + h * on_hand + b * backlog
        rows.append({
            "t": t, "IP_before": IP, "q": q, "D": D, "served": served,
            "on_hand": on_hand, "backlog": backlog,
            "net": on_hand - backlog, "cost": cost,
        })

    return pd.DataFrame(rows)


def ss_policy(s: int, S: int):
    def policy(IP):
        return max(S - IP, 0) if IP <= s else 0
    return policy


def base_stock_policy(S: int):
    def policy(IP):
        return max(S - IP, 0)
    return policy


def summarise(df: pd.DataFrame, label: str = "") -> dict:
    warm = df.iloc[len(df) // 10:]  # drop warm-up
    return {
        "policy": label,
        "avg cost / period": warm["cost"].mean(),
        "order frequency %": 100 * (warm["q"] > 0).mean(),
        "avg on-hand": warm["on_hand"].mean(),
        "avg backlog": warm["backlog"].mean(),
        "stockout periods %": 100 * (warm["backlog"] > 0).mean(),
        "fill rate %": 100 * warm["served"].sum() / max(warm["D"].sum(), 1e-9),
        "avg order size": warm.loc[warm["q"] > 0, "q"].mean(),
    }
