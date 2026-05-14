"""
app.py — Example Flask web server for a pricing/risk service.

At runtime the application imports the numba-optimised versions of pricing
and risk from the `optimized/` directory, which is produced by numba-build
during the CI/CD pipeline and copied into the container image.

Endpoints
---------
GET  /health
POST /pricing/options-surface   — Monte Carlo options surface
POST /pricing/implied-vol       — Implied volatility surface
POST /risk/var                  — Portfolio VaR & CVaR (Monte Carlo)
POST /risk/stress               — Stress scenarios
"""

import os
import sys

# At runtime (inside the container) the optimised modules live in optimized/.
# For local development against the raw sources, fall back to utils/.
_BASE = os.path.dirname(__file__)
_OPTIMIZED = os.path.join(_BASE, "optimized")
_UTILS = os.path.join(_BASE, "utils")

sys.path.insert(0, _OPTIMIZED if os.path.isdir(_OPTIMIZED) else _UTILS)

from flask import Flask, jsonify, request

from pricing import implied_vol_surface, mc_options_surface
from risk import portfolio_var_mc, stress_test

app = Flask(__name__)


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# ── Pricing ───────────────────────────────────────────────────────────────────


@app.post("/pricing/options-surface")
def options_surface():
    body = request.get_json(force=True)
    S = float(body["S"])
    strikes = [float(k) for k in body["strikes"]]
    maturities = [float(t) for t in body["maturities"]]
    r = float(body.get("r", 0.05))
    sigma = float(body.get("sigma", 0.2))
    n_paths = int(body.get("n_paths", 500))
    steps_per_year = int(body.get("steps_per_year", 52))

    surface = mc_options_surface(S, strikes, maturities, r, sigma, n_paths, steps_per_year)
    return jsonify({"surface": [[float(v) for v in row] for row in surface]})


@app.post("/pricing/implied-vol")
def implied_vol():
    body = request.get_json(force=True)
    S = float(body["S"])
    strikes = [float(k) for k in body["strikes"]]
    maturities = [float(t) for t in body["maturities"]]
    r = float(body.get("r", 0.05))
    market_surface = body["market_surface"]

    iv = implied_vol_surface(S, strikes, maturities, r, market_surface)
    return jsonify({"implied_vol_surface": [[float(v) for v in row] for row in iv]})


# ── Risk ──────────────────────────────────────────────────────────────────────


@app.post("/risk/var")
def var():
    body = request.get_json(force=True)
    weights = [float(w) for w in body["weights"]]
    mus = [float(m) for m in body["mus"]]
    sigmas = [float(s) for s in body["sigmas"]]
    T_days = int(body.get("T_days", 21))
    n_scenarios = int(body.get("n_scenarios", 10000))
    confidence = float(body.get("confidence", 0.99))

    var_val, cvar_val, mean_ret, std_ret = portfolio_var_mc(
        weights, mus, sigmas, T_days, n_scenarios, confidence
    )
    return jsonify(
        {
            "var": var_val,
            "cvar": cvar_val,
            "mean_return": mean_ret,
            "std_return": std_ret,
        }
    )


@app.post("/risk/stress")
def stress():
    body = request.get_json(force=True)
    weights = [float(w) for w in body["weights"]]
    mus = [float(m) for m in body["mus"]]
    sigmas = [float(s) for s in body["sigmas"]]
    shock_scenarios = [
        (s["label"], float(s["mu_multiplier"]), float(s["sigma_multiplier"]))
        for s in body["scenarios"]
    ]
    n_scenarios = int(body.get("n_scenarios", 1000))

    results = stress_test(weights, mus, sigmas, shock_scenarios, n_scenarios)
    return jsonify({"results": [list(r) for r in results]})


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
