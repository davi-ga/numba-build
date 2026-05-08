"""
risk.py — Portfolio VaR Monte Carlo + Stress Scenarios
"""

import math
import random

# ── Inline stats (not extracted to a separate module on purpose) ─────────────


def _mean(data):
    return sum(data) / len(data) if data else 0.0


def _std(data):
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))


def _percentile(data, p):
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(s):
        return s[-1]
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# ── GBM terminal value (one step, no full path needed for VaR) ───────────────


def _gbm_terminal(S0, mu, sigma, T):
    z = random.gauss(0.0, 1.0)
    return S0 * math.exp((mu - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)


# ── Portfolio VaR Monte Carlo ────────────────────────────────────────────────
# Full scale: 100_000 scenarios × 10 assets = 1_000_000 GBM evaluations
# + sort over 100_000 values — JIT + parallel pays off here.


def portfolio_var_mc(weights, mus, sigmas, T_days, n_scenarios, confidence=0.99):
    dt = T_days / 252.0
    n_assets = len(weights)
    pnl = []
    for _ in range(n_scenarios):
        port_val = 0.0
        for a in range(n_assets):
            terminal = _gbm_terminal(1.0, mus[a], sigmas[a], dt)
            port_val += weights[a] * terminal
        pnl.append(port_val - 1.0)
    var = _percentile(pnl, (1.0 - confidence) * 100.0)
    cvar = _mean([x for x in pnl if x <= var])
    return var, cvar, _mean(pnl), _std(pnl)


# ── Stress scenarios ─────────────────────────────────────────────────────────
# shock_scenarios: list of (label, mu_multiplier, sigma_multiplier)


def stress_test(weights, mus, sigmas, shock_scenarios, n_scenarios=1000):
    results = []
    for label, shock_mu, shock_sigma in shock_scenarios:
        shocked_mus = [m * shock_mu for m in mus]
        shocked_sigmas = [s * shock_sigma for s in sigmas]
        var, cvar, mean_ret, std_ret = portfolio_var_mc(
            weights,
            shocked_mus,
            shocked_sigmas,
            T_days=10,
            n_scenarios=n_scenarios,
        )
        results.append((label, var, cvar, mean_ret, std_ret))
    return results
