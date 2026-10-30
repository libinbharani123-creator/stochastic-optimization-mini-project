"""(s,S) dynamic programming on inventory position.

State   : IP (integer)
Action  : order q >= 0, y = IP + q
Cost    : K*1{q>0} + c*q + E[h*(y-D)^+ + b*(D-y)^+]
Next    : IP' = y - D

Assumes: backorders, single item, stationary demand, discount factor gamma.
For lead time L>=1 the policy's shape is preserved if costs are lagged; the
script simulates with the stated L, so the reported cost is honest.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np

from demand import DemandDist


@dataclass
class PolicySS:
    s: int
    S: int
    xs: np.ndarray
    V: np.ndarray
    H: np.ndarray
    y_star: np.ndarray


def solve_dp(dd: DemandDist, K: float, c: float, h: float, b: float,
             gamma: float = 0.99, x_min: int | None = None,
             x_max: int | None = None, tol: float = 1e-6,
             max_iter: int = 5000, verbose: bool = False) -> PolicySS:
    sup = dd.support.astype(int)
    pmf = dd.pmf.astype(float)
    dmax = int(sup.max())
    mu, sd = dd.mean(), dd.std()

    if x_min is None:
        x_min = int(-(mu + 8 * sd) - 100)
    if x_max is None:
        x_max = int(mu + 8 * sd + 100)

    n = x_max - x_min + 1
    xs = np.arange(x_min, x_max + 1, dtype=float)

    # G(y) = c*y + E[h*(y-D)^+ + b*(D-y)^+]
    diff = xs[:, None] - sup[None, :]
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    G = (h * pos + b * neg) @ pmf + c * xs

    # Transition lookup: state y (index i) -> demand d_j -> index i - d_j
    idx = np.arange(n)[:, None] - sup[None, :]
    idx_clamped = np.clip(idx, 0, n - 1)

    V = np.zeros(n)
    for it in range(max_iter):
        EV = (V[idx_clamped] * pmf[None, :]).sum(axis=1)
        H = G + gamma * EV
        suffix_min = np.minimum.accumulate(H[::-1])[::-1]
        order_val = np.empty_like(H)
        order_val[:-1] = suffix_min[1:]
        order_val[-1] = np.inf
        V_new = -c * xs + np.minimum(H, K + order_val)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if verbose and it % 100 == 0:
            print(f"  iter {it:4d}  delta = {delta:.3e}")
        if delta < tol:
            break
    else:
        warnings.warn(f"DP did not converge in {max_iter} iterations "
                      f"(delta = {delta:.3e})")

    # Extract policy
    EV = (V[idx_clamped] * pmf[None, :]).sum(axis=1)
    H = G + gamma * EV
    suffix_argmin = np.empty(n, dtype=int)
    best_val, best_idx = H[-1], n - 1
    for i in range(n - 1, -1, -1):
        if H[i] <= best_val:
            best_val, best_idx = H[i], i
        suffix_argmin[i] = best_idx

    order_target = np.empty(n, dtype=int)
    order_target[:-1] = suffix_argmin[1:]
    order_target[-1] = n - 1
    order_placed = K + H[order_target] < H

    if not order_placed.any():
        S_idx = int(np.argmin(H))
        warnings.warn("no state triggers an order under this DP solution")
        return PolicySS(s=x_min - 1, S=int(xs[S_idx]), xs=xs, V=V, H=H,
                        y_star=np.arange(n))

    s_idx = int(np.where(order_placed)[0].max())
    S_idx = int(np.argmin(H))

    if s_idx <= 2:
        warnings.warn("s at lower edge of grid — decrease x_min")
    if S_idx >= n - 3:
        warnings.warn("S at upper edge of grid — increase x_max")

    return PolicySS(s=int(xs[s_idx]), S=int(xs[S_idx]),
                    xs=xs, V=V, H=H,
                    y_star=np.where(order_placed, order_target, np.arange(n)))


def base_stock_level(dd: DemandDist, h: float, b: float) -> int:
    """Optimal order-up-to level for K=0 (newsvendor quantile)."""
    cr = b / (b + h)
    return dd.quantile(cr)
