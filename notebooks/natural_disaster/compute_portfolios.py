###### IMPORTS ############
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools as it
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import os
import sys
from pathlib import Path

project_root = os.path.join(Path.cwd(), '..', '..')
sys.path.insert(0, str(project_root))

from src.environments.natural_disaster import (
    need_based_policy,
    per_capita_need_policy,
    population_based_policy,
    income_based_policy,
    proximity_based_policy,
    randomized_weighted_hybrid_policy,
    mixed_random_policy_k_increments,
    generate_action_space,
    simulate_policy_dynamic_with_tpm
)
from src.p_mean import generalized_p_mean, get_optimum_vector, generate_p_grid
from src.portfolio import (
    Policy, Portfolio,
    budget_portfolio_with_suboptimalities, portfolio_with_line_search,
    compute_portfolio_worst_approx_ratio, portfolio_of_random_policies,
    portfolio_of_random_norms, portfolio_with_gpi, gpi
)

import multiprocessing as mp

# avoid thread oversubscription inside BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
############################

############################

### HELPERS
def get_optimum_policy(p):
    return get_optimum_vector(vectors=scores, p=p)

def get_performance(policy, p):
    return generalized_p_mean(x=policy, p=p)

def get_optimal_value(p):
    return optimal_values[p] if p in optimal_values else get_performance(get_optimum_policy(p), p)

# ---------- Parallel helpers ----------
def _init_worker(_scores, _optimal_values=None):
    """Initializer to set globals in each worker process."""
    global scores, optimal_values
    scores = _scores
    if _optimal_values is not None:
        optimal_values = _optimal_values

def _compute_opt_for_p(p):
    """Used to build optimal_values in parallel (uses global scores)."""
    return p, get_performance(get_optimum_policy(p), p)

def _rn_once(args):
    """One random-norm replicate. Returns approximation (float)."""
    K, seed, initial_p, p_grid = args
    portfolio_random_norm_sample = portfolio_of_random_norms(
        initial_p=initial_p, K=K, get_optimum_policy=get_optimum_policy, seed=seed
    )
    val = compute_portfolio_worst_approx_ratio(
        portfolio=portfolio_random_norm_sample,
        get_optimal_value=get_optimal_value,
        p_grid=p_grid,
        get_performance=get_performance
    )
    return float(np.round(val, 6))

def _rp_once(args):
    """One random-policy replicate. Returns approximation (float)."""
    K, seed, p_grid = args
    portfolio_random_policy = portfolio_of_random_policies(
        policies=[Policy(scores[i]) for i in range(len(scores))], K=K
    )
    val = compute_portfolio_worst_approx_ratio(
        portfolio=portfolio_random_policy,
        get_optimal_value=get_optimal_value,
        p_grid=p_grid,
        get_performance=get_performance
    )
    return float(np.round(val, 6))
# --------------------------------------

def heuristic(p_grid):
    heuristic_results = pd.DataFrame(
        columns=['K', 'portfolio_size', 'approximation', 'p_values']
    ).set_index('K')

    print('Started running heuristic...')
    for K in range(1, 11):
        h_portfolio = budget_portfolio_with_suboptimalities(
            initial_p=-100, K=K, get_optimum_policy=get_optimum_policy, get_performance=get_performance,
        )
        approx = compute_portfolio_worst_approx_ratio(
            portfolio=h_portfolio, get_performance=get_performance,
            get_optimal_value=get_optimal_value, p_grid=p_grid
        )
        heuristic_results.at[K, 'portfolio_size'] = len(h_portfolio)
        heuristic_results.at[K, 'approximation'] = approx
        heuristic_results.at[K, 'p_values'] = [round(policy.p, 3) for policy in h_portfolio]
    print('Finished running heuristic...')
    return heuristic_results


def line_search_portfolio(alpha_values, N, p_grid):
    ls_results = pd.DataFrame(
        columns=['K', 'alpha', 'oracle_calls', 'approximation', 'p_values']
    ).set_index('K')

    print('\nStarted running line search...')
    for alpha in alpha_values:
        ls_portfolio = portfolio_with_line_search(
            get_performance=get_performance, get_optimum_policy=get_optimum_policy, d=N, alpha=alpha,
        )
        K = len(ls_portfolio)
        if K not in ls_results.index:
            approx = compute_portfolio_worst_approx_ratio(
                portfolio=ls_portfolio, get_performance=get_performance,
                get_optimal_value=get_optimal_value, p_grid=p_grid
            )
            ls_results.at[K, 'alpha'] = alpha
            ls_results.at[K, 'oracle_calls'] = ls_portfolio.oracle_calls
            ls_results.at[K, 'approximation'] = approx
            ls_results.at[K, 'p_values'] = [round(policy.p, 3) for policy in ls_portfolio]
            if approx == 1.0:
                break
    print('Finished running line search...')
    return ls_results


def random_norm_portfolio(alpha_0, N, p_grid):
    initial_p = - np.log(N) / np.log(1/alpha_0)

    K_values = np.arange(1, 10)
    rn_results = pd.DataFrame(columns=['K', 'approximation']).set_index('K')

    print('\nStarted running random norm...')
    T = 10
    for K in K_values:
        vals = []
        for seed in range(T):
            port = portfolio_of_random_norms(
                initial_p=initial_p, K=K, get_optimum_policy=get_optimum_policy, seed=seed
            )
            v = compute_portfolio_worst_approx_ratio(
                portfolio=port, get_optimal_value=get_optimal_value,
                p_grid=p_grid, get_performance=get_performance
            )
            vals.append(v)
        rn_results.at[K, 'approximation'] = float(np.round(np.mean(vals), 4))
    print('Finished running random norm...')
    return rn_results


def random_policy_portfolio(p_grid):
    K_values = np.arange(1, 10)
    rp_results = pd.DataFrame(columns=['K', 'approximation']).set_index('K')

    print('\nStarted running random policy...')
    T = 10
    for K in K_values:
        vals = []
        for seed in range(T):  # seed kept for symmetry, portfolio_of_random_policies may not accept it
            port = portfolio_of_random_policies(
                policies=[Policy(scores[i]) for i in range(len(scores))], K=K
            )
            v = compute_portfolio_worst_approx_ratio(
                portfolio=port, get_optimal_value=get_optimal_value,
                p_grid=p_grid, get_performance=get_performance
            )
            vals.append(v)
        rp_results.loc[K, 'approximation'] = float(np.round(np.mean(vals), 4))
    print('Finished running random policy...')
    return rp_results

def _perf_for_vector(args):
    vec, p_grid = args
    return [get_performance(vec, p) for p in p_grid]


def gpi_portfolio(p_grid):
    print('\nStarted running GPI...')

    # GPI selection (sequential by design)
    _, chosen_vectors = gpi(vectors=scores, portfolio_size=10)
    Kmax = len(chosen_vectors)

    # Optimal values across p_grid
    try:
        opt_arr = np.array([optimal_values[p] for p in p_grid], dtype=float)
    except Exception:
        opt_arr = np.array([get_optimal_value(p) for p in p_grid], dtype=float)

    # Performance matrix (Kmax x |p_grid|)
    n_workers = int(os.environ.get("N_WORKERS", mp.cpu_count()))
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            rows = pool.map(_perf_for_vector, [(vec, p_grid) for vec in chosen_vectors],
                            chunksize=max(1, len(p_grid)//(n_workers*4) or 1))
        V = np.asarray(rows, dtype=float)
    else:
        V = np.asarray([_perf_for_vector((vec, p_grid)) for vec in chosen_vectors], dtype=float)

    # Best prefix over K and worst-case approximation over p
    best_prefix = np.maximum.accumulate(V, axis=0)
    approx = best_prefix / opt_arr
    worst_by_K = approx.min(axis=1)

    gpi_results = pd.DataFrame({'approximation': np.round(worst_by_K, 6)})
    gpi_results.index = np.arange(1, Kmax + 1)
    gpi_results.index.name = 'K'

    print('Finished running GPI...')
    return gpi_results



## Main
def main(alpha, gridsize, alpha_values, score_fname, alpha_0, suffix, outfile_loc):
    global scores, optimal_values

    # Load scores fast and as list of 1D arrays
    arr = pd.read_csv(score_fname).values  # (M, N)
    scores = [arr[i] for i in range(arr.shape[0])]
    N = arr.shape[1]

    p_grid = generate_p_grid(N=N, alpha=alpha, grid_size=gridsize)

    # Build optimal_values across p_grid (avoid inner Pool inside outer Pool workers)
    n_workers = int(os.environ.get("N_WORKERS", mp.cpu_count()))
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(scores,)) as pool:
            chunk = max(1, len(p_grid) // (n_workers * 4) or 1)
            tuples = pool.map(_compute_opt_for_p, p_grid, chunksize=chunk)
        optimal_values = dict(tuples)
    else:
        # No Pool (we're inside an outer worker); compute serially
        _init_worker(scores)  # ensure globals in this process
        optimal_values = {p: get_performance(get_optimum_policy(p), p) for p in p_grid}

    heuristic_results = heuristic(p_grid)
    heuristic_results.to_csv(os.path.join(outfile_loc, f'heuristic_portfolio_{suffix}.csv'))

    ls_results = line_search_portfolio(alpha_values, N, p_grid)
    ls_results.to_csv(os.path.join(outfile_loc, f'line_search_portfolio_{suffix}.csv'))

    rn_results = random_norm_portfolio(alpha_0, N, p_grid)
    rn_results.to_csv(os.path.join(outfile_loc, f'random_norm_portfolio_{suffix}.csv'))

    rp_results = random_policy_portfolio(p_grid)
    rp_results.to_csv(os.path.join(outfile_loc, f'random_policy_portfolio_{suffix}.csv'))

    gpi_results = gpi_portfolio(p_grid)
    gpi_results.to_csv(os.path.join(outfile_loc, f'gpi_portfolio_{suffix}.csv'))

    return heuristic_results, ls_results, rn_results, rp_results, gpi_results

def _run_one_scorefile(args):
    score_file, suffix, alpha, gridsize, alpha_values, alpha_0, outfile_loc, n_workers_inner = args
    # Disable inner multiprocessing entirely inside outer worker
    os.environ["N_WORKERS"] = str(int(n_workers_inner or 0))
    return main(alpha, gridsize, alpha_values, score_file, alpha_0, suffix, outfile_loc)



if __name__ == '__main__':
    # mp.freeze_support()   # uncomment on Windows if needed
    N_WORKERS_INNER = int(os.environ.get("N_WORKERS_INNER", 0))  # 0 means "no inner pool"

    ####### INPUT ########
    alpha = 0.95
    alpha_0 = 0.90
    gridsize = 500

    alpha_values = [0.05 * j for j in range(2, 20)] + [0.99]

    preferred_clusters = [{5}]
    per_unit_bonuses = [.05, .10, .2]
    no_bonus = True

    outfile_loc = os.path.join('..', '..', 'data', 'natural_disaster', 'portfolios', 'with_bias')
    os.makedirs(outfile_loc, exist_ok=True)
    ########################

    all_bonuses = list(it.product(preferred_clusters, per_unit_bonuses))

    score_files = [f'policy_rewards_bonus_{"-".join(map(str, sorted(pref)))}_{inc}.csv'
                   for pref, inc in all_bonuses]
    score_files = [os.path.join('..', '..', 'data', 'natural_disaster', i) for i in score_files]
    suffix_list = [f'bonus_{"-".join(map(str, sorted(pref)))}_{inc}' for pref, inc in all_bonuses]
    if no_bonus:
        score_files.append(os.path.join('..', '..', 'data', 'natural_disaster', 'policy_rewards.csv'))
        suffix_list.append('')

    tasks = [
        (score_file, suffix, alpha, gridsize, alpha_values, alpha_0, outfile_loc, 1)  # inner N_WORKERS=1
        for score_file, suffix in zip(score_files, suffix_list)
    ]

    # OUTER parallelism
    OUTER_WORKERS = int(os.environ.get("OUTER_WORKERS",
                                       min(mp.cpu_count(), max(1, len(tasks)))))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=OUTER_WORKERS) as pool:
        for _ in pool.imap_unordered(_run_one_scorefile, tasks, chunksize=1):
            pass


        
        
        
        
        
        
        
        
        
    