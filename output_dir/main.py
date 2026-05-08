import numba
import random
import time
from pricing_core import mc_options_surface, implied_vol_surface
from risk_engine import portfolio_var_mc, run_stress_scenarios

def run_main():
    random.seed(42)
    S = 100.0
    r = 0.05
    sigma = 0.2
    strikes = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0]
    maturities = [0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    N_PATHS = 2000
    N_SCENARIOS = 20000
    N_ASSETS = 20
    T_DAYS = 21
    weights = [1.0 / N_ASSETS for _ in range(N_ASSETS)]
    mus = [0.06 + 0.01 * i for i in range(N_ASSETS)]
    sigmas = [0.1 + 0.02 * i for i in range(N_ASSETS)]
    print('Building MC options surface...')
    t0 = time.perf_counter()
    mc_surface = mc_options_surface(S, strikes, maturities, r, sigma, N_PATHS, 252)
    t1 = time.perf_counter()
    print(f'  done in {t1 - t0:.2f}s')
    print('Computing implied vol surface...')
    t0 = time.perf_counter()
    iv_surface = implied_vol_surface(S, strikes, maturities, r, mc_surface)
    t1 = time.perf_counter()
    print(f'  done in {t1 - t0:.2f}s')
    print('Running portfolio VaR Monte Carlo...')
    t0 = time.perf_counter()
    var99, cvar99, mean_ret, std_ret = portfolio_var_mc(weights, mus, sigmas, T_DAYS, N_SCENARIOS, 0.99)
    t1 = time.perf_counter()
    print(f'  done in {t1 - t0:.2f}s')
    print(f'  99% VaR  : {var99:.4%}')
    print(f'  99% CVaR : {cvar99:.4%}')
    print('\nRunning stress scenarios...')
    shock_scenarios = [(1.0, -2.0, 3.0), (1.0, -5.0, 5.0), (1.0, -0.5, 1.5)]
    results = run_stress_scenarios(weights, mus, sigmas, shock_scenarios, N_SCENARIOS)
    for res in results:
        print(f'  Res: {res}')
if __name__ == '__main__':
    run_main()