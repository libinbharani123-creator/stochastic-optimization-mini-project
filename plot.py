"""Plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt

from dp import PolicySS


def plot_value_and_policy(pol: PolicySS, s: int, S: int,
                          outfile: str = "value_policy.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(pol.xs, pol.V, lw=2)
    ax.set_xlabel("inventory position x")
    ax.set_ylabel("V(x)")
    ax.set_title("Value function")
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.step(pol.xs, pol.y_star, where="post", lw=2, label="order-up-to y*(x)")
    ax.plot(pol.xs, pol.xs, "k:", lw=1, label="y = x (no order)")
    ax.axvline(s, color="tab:red", ls="--", alpha=0.6, label=f"s = {s}")
    ax.axhline(S, color="tab:red", ls="--", alpha=0.6, label=f"S = {S}")
    ax.set_xlabel("inventory position x")
    ax.set_ylabel("post-order IP")
    ax.set_title("(s, S) policy from the DP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    return fig


def plot_simulation(sim_df, outfile: str = "simulation.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sim_df["t"], sim_df["on_hand"], lw=0.9, label="on-hand")
    ax.plot(sim_df["t"], sim_df["backlog"], lw=0.9, label="backlog")
    ax.plot(sim_df["t"], sim_df["q"], lw=0.6, alpha=0.6, label="order q")
    ax.set_xlabel("period t")
    ax.set_ylabel("units / rupees")
    ax.set_title("Simulated inventory trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    return fig


def plot_sensitivity(K_vals, s_vals, S_vals, outfile: str = "sS_vs_K.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(K_vals, s_vals, "o-", label="reorder point s")
    ax.plot(K_vals, S_vals, "s-", label="order-up-to level S")
    ax.fill_between(K_vals, s_vals, S_vals, alpha=0.15, label="S - s")
    ax.set_xlabel("fixed ordering cost K")
    ax.set_ylabel("units")
    ax.set_title("(s, S) as a function of the fixed order cost")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    return fig
