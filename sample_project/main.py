import random
import time

from pricing import mc_options_surface, implied_vol_surface
from risk import portfolio_var_mc, stress_test

random.seed(42)


S = 100.0        # spot price
r = 0.05         # risk-free rate
sigma = 0.20     # base volatility

strikes = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
maturities = [0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]  # years

# Scale these up once JIT is added:
N_PATHS = 2_000        # target: 200_000
N_SCENARIOS = 20_000   # target: 100_000
N_ASSETS = 20
T_DAYS = 21

weights = [1.0 / N_ASSETS] * N_ASSETS
mus    = [0.06 + 0.01 * i for i in range(N_ASSETS)]
sigmas = [0.10 + 0.02 * i for i in range(N_ASSETS)]


print("Building MC options surface...")
t0 = time.perf_counter()
mc_surface = mc_options_surface(
    S, strikes, maturities, r, sigma,
    n_paths=N_PATHS, steps_per_year=252,
)
t1 = time.perf_counter()
print(f"  done in {t1 - t0:.2f}s  "
      f"({len(strikes)}×{len(maturities)} grid, {N_PATHS} paths each)")


print("Computing implied vol surface (Newton-Raphson)...")
t0 = time.perf_counter()
iv_surface = implied_vol_surface(S, strikes, maturities, r, mc_surface)
t1 = time.perf_counter()
print(f"  done in {t1 - t0:.2f}s")

# print a readable slice
T_COLS = [1, 2, 4, 6]   # indices into maturities list
header = f"  {'Strike':>6} |" + "".join(f"  T={maturities[j]:.2f}" for j in T_COLS)
print("\n" + header)
print("  " + "-" * (len(header) - 2))
for i, K in enumerate(strikes):
    row = f"  {K:6.0f} |"
    for j in T_COLS:
        row += f"   {iv_surface[i][j]:.4f}"
    print(row)

# ── Portfolio VaR ────────────────────────────────────────────────────────────

print("\nRunning portfolio VaR Monte Carlo...")
t0 = time.perf_counter()
var99, cvar99, mean_ret, std_ret = portfolio_var_mc(
    weights, mus, sigmas,
    T_days=T_DAYS, n_scenarios=N_SCENARIOS, confidence=0.99,
)
t1 = time.perf_counter()
print(f"  done in {t1 - t0:.2f}s  "
      f"({N_ASSETS} assets, {N_SCENARIOS} scenarios, {T_DAYS}-day horizon)")
print(f"  99% VaR  : {var99:.4%}")
print(f"  99% CVaR : {cvar99:.4%}")
print(f"  Mean ret : {mean_ret:.4%}")
print(f"  Std dev  : {std_ret:.4%}")

# ── Stress Scenarios ─────────────────────────────────────────────────────────

print("\nRunning stress scenarios...")
shock_scenarios = [
    ("2008 Crisis",   -2.0,  3.0),
    ("Flash Crash",   -5.0,  5.0),
    ("Rate Shock",    -0.5,  1.5),
    ("Bull Run",       1.5,  0.8),
    ("Stagflation",   -1.2,  2.5),
    ("Black Swan",   -10.0,  8.0),
    ("Mild Sell-off", -0.8,  1.2),
    ("Volatility Crunch", 1.0, 0.3),
]
results = stress_test(weights, mus, sigmas, shock_scenarios, n_scenarios=N_SCENARIOS)

print(f"\n  {'Scenario':<14} {'VaR 99%':>9} {'CVaR 99%':>10} {'Mean':>8} {'Std':>8}")
print("  " + "-" * 56)
for label, var, cvar, mean_r, std_r in results:
    print(f"  {label:<14} {var:>9.4%} {cvar:>10.4%} {mean_r:>8.4%} {std_r:>8.4%}")

print("\nDone.")
