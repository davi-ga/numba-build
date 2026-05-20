"""
pricing.py — Monte Carlo Options Surface + Implied Volatility Surface

These are the raw, un-optimised helpers. During the CI/CD pipeline,
forge reads this file, annotates the eligible functions with
@numba.njit, and writes the result to the output directory (e.g. optimized/).
The application imports from that output directory at runtime.
"""

import math
import random

# ── Black-Scholes helpers (used by implied-vol Newton-Raphson) ──────────────


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S, K, T, r, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_vega(S, K, T, r, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return 1e-10
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * _norm_pdf(d1)


# ── GBM path (single asset, pure Python) ────────────────────────────────────


def gbm_path(S0, mu, sigma, steps, dt):
    path = [0.0] * (steps + 1)
    path[0] = S0
    for i in range(1, steps + 1):
        z = random.gauss(0.0, 1.0)
        path[i] = path[i - 1] * math.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        )
    return path


# ── Monte Carlo call pricer ──────────────────────────────────────────────────


def mc_call_price(S, K, T, r, sigma, n_paths, steps):
    dt = T / steps
    total = 0.0
    for _ in range(n_paths):
        path = gbm_path(S, r, sigma, steps, dt)
        total += max(path[-1] - K, 0.0)
    return math.exp(-r * T) * (total / n_paths)


# ── Options surface: n_strikes × n_mats × n_paths ───────────────────────────


def mc_options_surface(S, strikes, maturities, r, sigma, n_paths, steps_per_year=52):
    surface = []
    for K in strikes:
        row = []
        for T in maturities:
            steps = max(1, int(steps_per_year * T))
            row.append(mc_call_price(S, K, T, r, sigma, n_paths, steps))
        surface.append(row)
    return surface


# ── Implied vol via Newton-Raphson ───────────────────────────────────────────


def implied_vol(S, K, T, r, market_price, max_iter=100, tol=1e-7):
    sigma = 0.20
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= 0.0:
            sigma = 1e-6
    return sigma


def implied_vol_surface(S, strikes, maturities, r, market_surface):
    iv_surface = []
    for i, K in enumerate(strikes):
        row = []
        for j, T in enumerate(maturities):
            row.append(implied_vol(S, K, T, r, market_surface[i][j]))
        iv_surface.append(row)
    return iv_surface
