# Multi-period inventory: learning an (s, S) policy by DP

Course 21AIE321T — Stochastic Decision Making
Extension of the newsvendor project.

---

## What it does

- Represents demand as a **discrete distribution** `(support, pmf)` — either
  fitted from a daily-sales CSV (empirical, stockout days excluded) or
  generated parametrically (negative binomial / Poisson / normal).
- Solves the **infinite-horizon discounted DP** over inventory position:

  ```
  V(x) = -c·x + min( H(x),  K + min_{y>x} H(y) )
  H(y) = c·y + E[h·(y-D)^+ + b·(D-y)^+] + γ·E[V(y-D)]
  ```

- Extracts the **`(s, S)` policy**: order up to `S` when `IP ≤ s`,
  otherwise order nothing.
- Simulates the policy under **lead time `L`** and compares against the
  base-stock policy (`K = 0` optimum) and a Silver–Meal-style heuristic.

---

## Files

| File | Purpose |
|---|---|
| `demand.py` | Demand distribution objects and fitters |
| `dp.py` | Value iteration, `(s, S)` extraction, base-stock reference |
| `simulate.py` | Simulator and policy factories |
| `plot.py` | Value / policy / trajectory / sensitivity plots |
| `solve_ss.py` | Main: fit → solve → simulate → compare → plot |
| `make_synthetic_data.py` | Optional: synthetic daily-sales CSV |
| `load_uci.py` | Optional: download UCI Online Retail |
| `ss_inventory.ipynb` | Colab notebook |
| `requirements.txt` | Dependencies |
| `README.md` | This file |

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

### No data needed (parametric demand)

```bash
python solve_ss.py
```

### From a CSV of daily sales

```bash
python make_synthetic_data.py
python solve_ss.py --data daily_sales.csv --sweep
```

### From real UCI data

```bash
python load_uci.py
python solve_ss.py --data daily_sales.csv \
    --K 150 --c 5 --h 0.3 --b 4 --lead-time 2
```

### In Colab

Upload `ss_inventory.ipynb` → **Runtime ▸ Run all**.

---

## Expected output (defaults)

```
demand    : negbin(mean=100,cv=0.20): mean=100.0  std=20.0  cv=0.20
source    : synthetic negbin
economics : K=150  c=50  h=3  b=40  L=1  gamma=0.99
base-stock reference (K=0): S_bs = ~130

iter    0  delta = 1.203e+04
...
iter ~300  delta = 8.7e-07
(s, S) from DP : s ≈ 30   S ≈ 130   width ≈ 100
sanity check   : S >= S_bs (130 >= 130) = True, s <= S_bs (30 <= 130) = True

policy                              avg cost/period  order freq %  fill rate %  stockout %
(s,S) DP (s=30, S=130)                       ~2040          ~24          99.5        3.2
base-stock S=130                            ~2190        100.0          99.9        0.9
myopic (s,S) heuristic                      ~2120          ~28          99.4        3.6
```

The `(s, S)` policy wins by **consolidating orders**: it keeps a lower
average on-hand, matches service, and saves the fixed cost `K` on most
periods. The base-stock policy orders every period and pays `K` each time.

---

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--data` | — | CSV with columns `date, units_sold` (optional) |
| `--mean` | `100` | Mean demand when no data |
| `--cv` | `0.20` | Coefficient of variation when no data |
| `--K` | `150` | Fixed ordering cost |
| `--c` | `50` | Per-unit purchase cost |
| `--h` | `3` | Holding cost / unit / period |
| `--b` | `40` | Backorder cost / unit / period |
| `--lead-time` | `1` | Order lead time in periods (0 = instant) |
| `--gamma` | `0.99` | Discount factor for the DP |
| `--sim-periods` | `20000` | Periods for policy evaluation |
| `--sweep` | off | Run `(s, S)` vs `K` sweep and save plot |
| `--no-show` | off | Do not open plot windows |

---

## Theory notes

- With `K = 0` the DP collapses to a **base-stock policy**:
  `s = S − 1`, `S = F⁻¹(b / (b + h))`. That is the sanity check printed
  at run time.
- The width of the ordering band satisfies `S − s ≈ √(2 K μ / h)`
  (Porteus approximation) — visible in `--sweep`.
- The DP uses **backorders**. Lost sales would require a 2-D state
  `(on-hand, pipeline)` and is left as a project extension.

---

## Caveats

- **Lead time `L ≥ 1`**: the `(s, S)` shape is preserved, but the classical
  optimality proof requires either `L = 0` or costs lagged by `L`. The
  **simulator** runs with the stated `L`, so the reported cost is the true
  operating cost — only the optimality claim is approximate.
- Demand is treated as i.i.d. given the fitted distribution. Seasonality,
  trend and promotions are **not** modelled inside the DP. They would
  require a non-stationary DP or a feature-conditioned demand model —
  the natural next extension.
- Discounted DP with `γ = 0.99` approximates the average-cost optimum well.
  Pushing `γ → 1` shifts `s` and `S` marginally.

---

## Extensions

1. **Feature-conditioned DP** — condition demand on weekday and promo,
   solve one `(s, S)` per regime.
2. **Lost sales** — 2-D DP `(on-hand, pipeline)`.
3. **Stochastic lead time** — add a scalar pipeline state with a random
   receiving delay.
4. **Multi-item** — joint `(s, S)` with shared capacity (MIP or ADP).

---

## How this extends the newsvendor project

| Newsvendor (previous) | `(s, S)` DP (this) |
|---|---|
| One decision per period | Continuous decision over infinite horizon |
| No holding / backorder costs | `h` and `b` in every period |
| No lead time | Explicit `L` |
| No fixed cost | `K` per order (this is what makes `(s, S)`) |
| Grid search over `Q` | Value iteration on `IP` |
| Order quantity `Q ≈ 96` | `S ≈ 130`, `s ≈ 30` |

Both solve `E[Profit(Q, D)]`-type objectives. The DP solves it *over time*
under a state transition, which is why the optimal policy becomes `(s, S)`
rather than a single number.
