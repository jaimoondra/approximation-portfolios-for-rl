from taxi_portfolio_main import get_optimum
from portfolio import portfolio_with_line_search, generalized_p_mean
import numpy as np
log = False


def get_optimal_value(p):
    """
    Placeholder for a subroutine that returns the optimal value (i.e., score) for a given p
  
    """
    p_mean, vectors = get_optimum(p)
    return p_mean


def get_performance(policy, p):
    """
    Placeholder for v^{(l)}(policy,p).
    Returns a real-valued performance of 'policy' under parameter p.
    """
    opt_vec = policy

    # out = get_optimum(p, eval=True, load_p=load_p)
    out = generalized_p_mean(opt_vec, p)
    if log:
        print('Performance: ', out)
    return out


def generate_p_grid(N, alpha, grid_size=100):
    """
    Generates a grid of values of p between -\infty and 1.
    :param N: number of reward functions
    :param alpha: real number in (0, 1)
    :param grid_size: number of desired points in the grid
    :return: grid of values of p
    """
    grid = []
    p_mid = - np.log(N)/np.log(1/alpha)
    
    q_least = 0
    q_most = 1/p_mid

    print(p_mid)
    
    grid.append(-np.inf)
    for i in range(1, grid_size//2):
        q = q_least + i * (q_most - q_least) / (grid_size//2)
        grid.append(1/q)
    
    p_most = 1
    for i in range(1, grid_size//2):
        p = p_mid + i * (p_most - p_mid) / (grid_size//2)
        grid.append(p)
    
    return grid


def precompute_optimal_values(get_optimum, N, alpha, grid_size=100):
    """
    Precompute the optimal performance (max over all policies) for p
    on a grid from p_min = -log2(N) to p_max = 1, in increments of 'step'.

    :return:
       p_to_optval: dict mapping p -> float (optimal performance at that p)
       p_grid:      sorted list of p-values used
    """
    p_vals = generate_p_grid(N=N, alpha=alpha, grid_size=grid_size)
    print('grid: ', p_vals)
    p_to_optval = {}
    p_to_optvec = {}
    for p_val in p_vals:
        print('p val: ', p_val)
        p_mean, vectors = get_optimum(p_val)
        p_to_optval[p_val] = p_mean
        p_to_optvec[p_val] = vectors

    p_vals = sorted(p_vals)
    return p_to_optval, p_to_optvec, p_vals


def compute_portfolio_worst_approx_ratio(
    portfolio_policies,  # list of policies (or policy vectors)
    p_to_optval,         # dict p -> optimal performance
    p_grid,              # sorted list of p-values
    get_performance
):
    """
    Given:
      - portfolio_policies: a list of policies 
      - p_to_optval: precomputed optimal performance for each p in p_grid
      - p_grid: list of p-values spanning [initial_p, 1]
      - get_performance function

    We compute, for each p in p_grid:
       ratio_p = max_{pi in portfolio} [ v^{(l)}(pi,p)  / v^{(l)}(globally optimal policy,p)  ]

    Then the 'worst-case approximation ratio' = min_{p in p_grid} ratio_p.
    """
    # worst_ratio = float('inf')
    all_ratios = []
    matrix = []

    for p_val in p_grid:
        opt_val = p_to_optval[p_val]

        # If the optimal is near zero, you have to decide how to handle that (skip or set ratio=1 or =∞).
        if opt_val <= 1e-12:
            # For demonstration, we skip p_val where optimum ~ 0 
            # (or you might define ratio = 1 if your policy is also 0, or ∞ if your policy is > 0)
            continue

        # 1) Evaluate each policy in the portfolio
        best_portfolio_val = 0.0
        portfolio_vals = []
        ratios = []
        for policy in portfolio_policies:
            if log:
                print(f'Evaluating {policy}-policy on p_value of {p_val}')
            val = get_performance(policy, p_val)
            portfolio_vals.append(val)
            if val > best_portfolio_val:
                best_portfolio_val = val
            ratios.append(min(1, val/opt_val))
        matrix.append(ratios)
        # 2) Compute the ratio for p_val
        if log:
            print(f'\t\tFor p_val {p_val}, our portfolio gave values: {portfolio_vals}')
            print(f'\t\tBest portfolio val: {best_portfolio_val}. Optimal val: {opt_val}')
        ratio_p = best_portfolio_val / opt_val
        if log:
            print('\t\tRatio Metric: ', ratio_p)

        # 3) Track the minimum across all p
        # if ratio_p < worst_ratio:
        #     worst_ratio = ratio_p
        if ratio_p>1:
            all_ratios.append(1)
        else:
            all_ratios.append(ratio_p)
    worst_ratio = min(all_ratios)
    mean_ratio = sum(all_ratios)/len(all_ratios)
    return worst_ratio, mean_ratio, matrix



