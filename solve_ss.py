#!/usr/bin/env python3
"""
Fit demand -> solve (s,S) DP -> simulate -> compare policies.
"""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd

import demand as dem
import dp as dpmod
import simulate as sim
import plot as pl


def parse_args():
    ap = argparse.ArgumentParser(description="(s,S) inventory optimisation")
    ap.add_argument("--data", default=None,
                    help="CSV with columns [date, units_sold]; optional")
    ap.add_argument("--mean", type=float, default=100.0,
                    help="mean demand if --data omitted")
    ap.add_argument("--cv", type=float, default=0.20,
                    help="coefficient of variation if --data omitted")
    ap.add_argument("--K", type=float, default=150.0, help="fixed order cost")
    ap.add_argument("--c", type=float, default=50.0, help="per-unit cost")
    ap.add_argument("--h", type=float, default=3.0, help="holding cost / unit / period")
    ap.add_argument("--b", type=float, default=40.0, help="backorder cost / unit / period")
    ap.add_argument("--lead-time", type=int, default=1,
                    help="order lead time in periods (0 = instant)")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="DP discount factor")
    ap.add_argument("--sim-periods", type=int, default=20_000,
                    help="periods for policy evaluation")
    ap.add_argument("--sweep", action="store_true",
                    help="run (s,S) vs K sensitivity and save plot")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open the plot windows")
    return ap.parse_args()


def main():
    args = parse_args()

    # -- 1. demand ------------------------------------------------------
    if args.data:
        dd = dem.fit_from_data(args.data)
        source = args.data
    else:
        dd = dem.from_negbin(args.mean, args.cv)
        source = "synthetic negbin"
    print("=" * 68)
    print("MULTI-PERIOD (s,S) INVENTORY OPTIMISATION")
    print("=" * 68)
    print(f"demand    : {dd.summary()}")
    print(f"source    : {source}")
    print(f"economics : K={args.K}  c={args.c}  h={args.h}  b={args.b}  "
          f"L={args.lead_time}  gamma={args.gamma}")
    print(f"base-stock reference (K=0): S_bs = {dpmod.base_stock_level(dd, args.h, args.b)}")

    # -- 2. (s,S) DP ----------------------------------------------------
    pol = dpmod.solve_dp(dd, args.K, args.c, args.h, args.b,
                         gamma=args.gamma, verbose=True)
    print("-" * 68)
    print(f"(s, S) from DP : s = {pol.s}   S = {pol.S}   width = {pol.S - pol.s}")

    S_bs = dpmod.base_stock_level(dd, args.h, args.b)
    print(f"sanity check   : S >= S_bs ({pol.S} >= {S_bs}) = {pol.S >= S_bs}, "
          f"s <= S_bs ({pol.s} <= {S_bs}) = {pol.s <= S_bs}")

    # -- 3. simulation comparison --------------------------------------
    rng = np.random.default_rng(42)
    common = dict(n_periods=args.sim_periods, K=args.K, c=args.c,
                  h=args.h, b=args.b, lead_time=args.lead_time,
                  rng=rng, init_ip=pol.S)
    policies = {
        f"(s,S) DP (s={pol.s}, S={pol.S})": sim.ss_policy(pol.s, pol.S),
        f"base-stock S={S_bs}":            sim.base_stock_policy(S_bs),
        "myopic (s,S) heuristic":          sim.ss_policy(
            int(S_bs - np.sqrt(max(2 * args.K * dd.mean() / max(args.h, 1e-9), 1))),
            S_bs),
    }
    results = []
    sim_dfs = {}
    for name, fn in policies.items():
        df = sim.simulate(fn, dd, **common)
        results.append(sim.summarise(df, name))
        sim_dfs[name] = df

    print("-" * 68)
    summary = pd.DataFrame(results)
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(summary.to_string(index=False))

    # -- 4. plots -------------------------------------------------------
    pl.plot_value_and_policy(pol, pol.s, pol.S, outfile="value_policy.png")
    print("[saved] value_policy.png")

    key = next(iter(sim_dfs))
    pl.plot_simulation(sim_dfs[key], outfile="simulation.png")
    print("[saved] simulation.png")

    # -- 5. sensitivity on K -------------------------------------------
    if args.sweep:
        K_vals = [10, 50, 100, 200, 400, 800, 1500]
        s_vals, S_vals = [], []
        print("-" * 68)
        print("sensitivity: (s,S) vs fixed ordering cost K")
        print(f"{'K':>8} {'s':>8} {'S':>8} {'S - s':>8}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for K in K_vals:
                p = dpmod.solve_dp(dd, K, args.c, args.h, args.b,
                                   gamma=args.gamma, verbose=False,
                                   max_iter=2000)
                s_vals.append(p.s); S_vals.append(p.S)
                print(f"{K:>8} {p.s:>8} {p.S:>8} {p.S - p.s:>8}")
        pl.plot_sensitivity(K_vals, s_vals, S_vals, outfile="sS_vs_K.png")
        print("[saved] sS_vs_K.png")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
